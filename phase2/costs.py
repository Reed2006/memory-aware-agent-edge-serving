"""Observable, per-concurrency costs and bounded empirical external-wait estimates."""

import copy
import json
import math
from pathlib import Path


SCHEMA = 'phase2-cost-profile-v1'
# Candidate lease horizons: quantiles of the owner's remaining external wait.
# 0.9 was added on dev (2026-11-17): with only Q25-Q75 (~3.3 s on M4) a lease could
# not cover the wait tail, which a fixed 10 s TTL covered at a 99% history hit rate.
# 0.99 was added with the expected-idle value (2026-11-18): holding through the thin
# tail now costs little idle time, so an expensive history can be kept for nearly
# every return, as the 10 s TTL did (on M4 dev waits Q90 = 4.8 s, max 12.4 s).
LEASE_QUANTILES = (.25, .5, .75, .9, .99)


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _count(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')
    return value


def _curve(value, name: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or 'seconds_per_token' not in value:
        raise ValueError(f'{name} needs seconds_per_token')
    return dict(seconds_per_token=_number(value['seconds_per_token'], name),
                intercept_s=_number(value.get('intercept_s', 0), name))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class CostModel:
    """Linear incremental costs; missing calibration never invents saved work.

    ``prompt_n`` is the number actually evaluated by llama.cpp, not total prompt
    length. Concurrency buckets are deliberately not extrapolated across K.
    """

    def __init__(self, profile: dict | None = None):
        value = copy.deepcopy(profile) if profile is not None else {}
        if not isinstance(value, dict) or value.get('schema', SCHEMA) != SCHEMA:
            raise ValueError('Unsupported Phase 2 cost profile')
        self.alpha = _number(value.get('ewma_alpha', .2), 'ewma_alpha', positive=True)
        if self.alpha > 1:
            raise ValueError('ewma_alpha must be at most one')
        self.max_samples = _count(value.get('max_wait_samples', 512), 'max_wait_samples', positive=True)
        self.prefill_min_tokens = _count(value.get('prefill_min_tokens', 128), 'prefill_min_tokens', positive=True)
        self.online_prefill = value.get('online_prefill', True)
        if not isinstance(self.online_prefill, bool):
            raise ValueError('online_prefill must be a boolean')
        self.online_io = value.get('online_io', True)
        if not isinstance(self.online_io, bool):
            raise ValueError('online_io must be a boolean')
        self.fallback_wait_s = _number(value.get('fallback_wait_s', 1), 'fallback_wait_s', positive=True)
        raw_prefill = value.get('prefill', {})
        if not isinstance(raw_prefill, dict):
            raise ValueError('prefill must map concurrency to cost curves')
        self.prefill = {}
        for concurrency, curve in raw_prefill.items():
            try:
                key = int(concurrency)
            except (ValueError, TypeError) as error:
                raise ValueError('Invalid prefill concurrency') from error
            _count(key, 'prefill concurrency', positive=True)
            parsed = _curve(curve, 'prefill')
            if parsed is None:
                raise ValueError('prefill curve cannot be null')
            self.prefill[key] = parsed
        self.decode = _curve(value.get('decode'), 'decode')
        self.save = _curve(value.get('save'), 'save')
        self.restore = _curve(value.get('restore'), 'restore')
        waits = value.get('waits_s', [])
        if not isinstance(waits, list):
            raise ValueError('waits_s must be a list of completed observations')
        self.waits = [_number(wait, 'wait duration') for wait in waits][-self.max_samples:]
        self.extra = {key: item for key, item in value.items() if key not in {
            'schema', 'prefill', 'decode', 'save', 'restore', 'waits_s',
            'ewma_alpha', 'fallback_wait_s', 'max_wait_samples',
            'prefill_min_tokens', 'online_prefill', 'online_io',
        }}

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(
            self.extra, schema=SCHEMA, prefill={str(key): curve for key, curve in self.prefill.items()},
            decode=self.decode, save=self.save, restore=self.restore, waits_s=self.waits,
            ewma_alpha=self.alpha, fallback_wait_s=self.fallback_wait_s,
            max_wait_samples=self.max_samples,
            prefill_min_tokens=self.prefill_min_tokens, online_prefill=self.online_prefill,
            online_io=self.online_io,
        ))

    def update(self, timings: dict, concurrency: int = 1) -> None:
        concurrency = _count(concurrency, 'concurrency', positive=True)
        for count_key, time_key, kind in [('prompt_n', 'prompt_ms', 'prefill'),
                                          ('predicted_n', 'predicted_ms', 'decode')]:
            try:
                count = _number(timings.get(count_key), count_key, positive=True)
                seconds = _number(timings.get(time_key), time_key, positive=True) / 1000
            except ValueError:
                continue
            if kind == 'prefill' and (not self.online_prefill or count < self.prefill_min_tokens):
                # A short cached continuation's timing can include a whole batch's
                # prefill wait. Dividing it by a few tokens is not a rebuild rate.
                continue
            previous = self.prefill.get(concurrency) if kind == 'prefill' else self.decode
            intercept = previous['intercept_s'] if previous else 0
            rate = max(0, seconds - intercept) / count
            if previous:
                rate = (1 - self.alpha) * previous['seconds_per_token'] + self.alpha * rate
            curve = dict(seconds_per_token=rate, intercept_s=intercept)
            if kind == 'prefill':
                self.prefill[concurrency] = curve
            else:
                self.decode = curve

    def observe_io(self, kind: str, tokens, seconds) -> None:
        """EWMA-update a calibrated save/restore curve from a completed backend call.

        Probes time slot I/O on an idle server. Under load the call also waits for
        the batch in progress: on the M4 pilot, in-run save + restore took about
        1.8x the probe price, so checkpoints were quoted too cheap. An unprofiled
        curve stays unprofiled; no checkpoint saving is invented.
        """
        if kind not in ('save', 'restore'):
            raise ValueError('kind must be save or restore')
        tokens = _count(tokens, 'tokens')
        seconds = _number(seconds, 'seconds')
        previous = getattr(self, kind)
        if not self.online_io or previous is None or not tokens:
            return
        rate = max(0, seconds - previous['intercept_s']) / tokens
        setattr(self, kind, dict(previous, seconds_per_token=(1 - self.alpha) *
                                 previous['seconds_per_token'] + self.alpha * rate))

    def observe_wait(self, duration_s) -> None:
        self.waits.append(_number(duration_s, 'wait duration'))
        self.waits = self.waits[-self.max_samples:]

    def prefill_s(self, tokens: int, reused: int = 0, concurrency: int = 1) -> float:
        tokens = _count(tokens, 'tokens')
        reused = _count(reused, 'reused')
        concurrency = _count(concurrency, 'concurrency', positive=True)
        curve = self.prefill.get(concurrency)
        evaluated = max(0, tokens - reused)
        return self._evaluate(curve, evaluated) or 0.0

    def decode_s(self, tokens: int) -> float:
        return self._evaluate(self.decode, _count(tokens, 'tokens')) or 0.0

    def loss_s(self, cached_tokens: int, surviving_tokens: int, concurrency: int = 1) -> float:
        cached_tokens = _count(cached_tokens, 'cached_tokens')
        surviving_tokens = _count(surviving_tokens, 'surviving_tokens')
        curve = self.prefill.get(_count(concurrency, 'concurrency', positive=True))
        # The request's fixed prefill overhead is paid with or without eviction.
        return max(0, cached_tokens - surviving_tokens) * curve['seconds_per_token'] if curve else 0.0

    def return_probability(self, elapsed_s: float, horizon_s: float) -> float:
        elapsed = _number(elapsed_s, 'elapsed_s')
        horizon = _number(horizon_s, 'horizon_s')
        if not horizon or not any(wait > 0 for wait in self.waits):
            return 0.0
        survivors = [wait for wait in self.waits if wait > elapsed]
        if survivors:
            return sum(wait <= elapsed + horizon for wait in survivors) / len(survivors)
        # Extrapolate only beyond an observed tail; no future replay gap enters here.
        scale = max(self.fallback_wait_s, sum(self.waits) / len(self.waits))
        return -math.expm1(-horizon / scale)

    def expected_idle_s(self, elapsed_s: float, horizon_s: float) -> float:
        """E[min(remaining wait, horizon)]: how long a slot held for its owner sits idle.

        The owner usually returns before the lease ends, and the slot is busy again
        from that moment, so a lease of h seconds does not cost h seconds of capacity.
        """
        elapsed = _number(elapsed_s, 'elapsed_s')
        horizon = _number(horizon_s, 'horizon_s')
        if not horizon or not any(wait > 0 for wait in self.waits):
            return 0.0
        residual = [wait - elapsed for wait in self.waits if wait > elapsed]
        if residual:
            return sum(min(wait, horizon) for wait in residual) / len(residual)
        scale = max(self.fallback_wait_s, sum(self.waits) / len(self.waits))
        return -scale * math.expm1(-horizon / scale)  # E[min(Exp(scale), h)]

    def best_lease(self, elapsed_s, loss_s, queue_weight=1, max_s=10,
                   quantiles=LEASE_QUANTILES) -> tuple[float, float]:
        """Best lease horizon and its net value in seconds of waiting saved.

        Both sides count delay to the owner and to the ``queue_weight`` ready
        requests that no other free slot can serve. Giving the slot away costs the
        owner ``loss_s`` when it returns, and recomputing (or the save/restore pause)
        occupies a slot for about as long, so the waiting requests lose it too:
        (1 + w) x loss, saved only if the owner returns within h (otherwise the lease
        expires and the loss is paid anyway). Holding costs the waiting requests
        only the time the slot actually sits idle, E[min(remaining wait, h)].

        Revised on dev (2026-11-18). The first version charged w x h and counted
        the loss for the owner alone, which overstated holding and understated
        eviction: under contention it evicted owners whose recomputation then
        delayed everyone, and a fixed 10 s TTL beat it.
        """
        elapsed = _number(elapsed_s, 'elapsed_s')
        loss = _number(loss_s, 'loss_s')
        weight = _number(queue_weight, 'queue_weight')
        maximum = _number(max_s, 'max_s')
        probabilities = [_number(q, 'quantile') for q in quantiles]
        if any(q > 1 for q in probabilities):
            raise ValueError('quantiles must be between zero and one')
        if not loss or not maximum or not any(wait > 0 for wait in self.waits):
            return 0.0, 0.0
        residual = [wait - elapsed for wait in self.waits if wait > elapsed]
        if residual:
            candidates = [min(maximum, _quantile(residual, q)) for q in probabilities]
        else:
            scale = max(self.fallback_wait_s, sum(self.waits) / len(self.waits))
            candidates = [min(maximum, -scale * math.log1p(-q)) if q < 1 else maximum
                          for q in probabilities]
        best = (0.0, 0.0)
        for horizon in sorted(set(candidates)):
            value = (self.return_probability(elapsed, horizon) * (1 + weight) * loss
                     - weight * self.expected_idle_s(elapsed, horizon))
            if value > best[1]:
                best = horizon, value
        return best

    @staticmethod
    def _evaluate(curve, tokens: int) -> float | None:
        if curve is None:
            return None
        return curve['intercept_s'] + curve['seconds_per_token'] * tokens if tokens else 0.0

    def save_s(self, tokens: int) -> float | None:
        return self._evaluate(self.save, _count(tokens, 'tokens'))

    def restore_s(self, tokens: int) -> float | None:
        return self._evaluate(self.restore, _count(tokens, 'tokens'))

    def checkpoint_worthwhile(self, tokens, loss_s, other_ready=0, p_use=.8, margin_s=.05,
                             *, saved=False) -> bool:
        loss = _number(loss_s, 'loss_s')
        pressure = 1 + _count(other_ready, 'other_ready')
        probability = _number(p_use, 'p_use')
        margin = _number(margin_s, 'margin_s')
        if probability > 1:
            raise ValueError('p_use must be at most one')
        if not isinstance(saved, bool):
            raise ValueError('saved must be a boolean')
        tokens = _count(tokens, 'tokens')
        save, restore = (0.0 if saved else self.save_s(tokens)), self.restore_s(tokens)
        if save is None or restore is None:
            return False
        # A synchronous pause is counted once for the candidate and once per other
        # ready request. The same conservative pressure proxy is used for restore.
        return probability * (loss - pressure * restore) > pressure * save + margin
