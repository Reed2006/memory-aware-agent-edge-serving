"""Nonpreemptive dispatch with explicit slot ownership and bounded cache leases.

The scheduler sees only current requests and observed events. Replay annotations
are copied to audit logs; they are never used to rank requests or predict returns.
An ambiguous backend result stops the entire scheduler, including subsequent runs
on that backend, until the operator establishes a clean server state.
"""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any

import httpx

from phase1.core import Jsonl, common_prefix
from phase2.checkpoints import KV_BYTES_PER_TOKEN, STATE_HEADER_BYTES, CheckpointUncertain
from phase2.costs import CostModel
from phase2.state import PolicyConfig, SlotState, TaskPhase, TaskState


def token_digest(tokens: list[int]) -> str:
    """Short digest of generated tokens; greedy outputs should match the isolated run."""
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()[:16]


class BackendStateUncertain(RuntimeError):
    """Backend work may still be running; restarting this client is not recovery."""


class SchedulerClosed(RuntimeError):
    pass


class WorkloadIntegrityError(RuntimeError):
    pass


class _DispatchDeferred(Exception):
    """Retry before a model call; any prior slot I/O has already completed."""


AUDIT_FIELDS = {
    'schema_version', 'label', 'run_id', 'policy', 'param_hash', 'workload_hash',
    'cost_hash', 'model_hash', 'platform_id', 'seed', 'repeat', 'trial', 'trial_order',
    'P', 'ctx_size', 'ctx_per_slot', 'kv_unified', 'cache_ram_mib', 'kv_precision',
    'k', 'j', 'gap_mode', 'arrival', 'steps_limit', 'job', 'task_id', 'source_task_id',
    'runtime_task_id', 'step', 'reusable_tokens', 'new_tokens', 'expected_output',
    'prefix_verified',
    'workload_split', 'synthetic_performance',
}


@dataclass
class ReadyRequest:
    task_id: str
    payload: dict
    audit: dict
    sequence: int
    arrival_s: float
    enqueue_s: float | None
    future: asyncio.Future
    peak_concurrency: int = 0


class Scheduler:
    def __init__(self, client, config: PolicyConfig, request_log: Jsonl,
                 decision_log: Jsonl, costs: CostModel | None = None, checkpoints=None,
                 clock=time.perf_counter):
        if config.checkpoint and checkpoints is None:
            raise ValueError('checkpoint policy requires a bounded CheckpointStore')
        self.client, self.config = client, config
        self.request_log, self.decision_log = request_log, decision_log
        self.costs = costs if costs is not None else CostModel()
        self.checkpoints, self.clock = checkpoints, clock
        self.slots = [SlotState(i) for i in range(config.slots)]
        self.tasks: dict[str, TaskState] = {}
        self._ready: list[ReadyRequest] = []
        self._waiting: list[ReadyRequest] = []
        self._running: set[asyncio.Task] = set()
        self._model_requests: dict[int, ReadyRequest] = {}
        self._outstanding: set[str] = set()
        self._deferred_cleanup: set[str] = set()
        self._sequence = 0
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task | None = None
        self._closing = False
        self._io_count = 0
        self._fatal: BackendStateUncertain | None = None
        self._decision_context: dict = {}
        self._task_audits: dict[str, dict] = {}

    @property
    def _io_active(self) -> bool:
        return self._io_count > 0

    async def __aenter__(self):
        if self._dispatcher is not None:
            raise RuntimeError('scheduler is single-use')
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._closing = True
        self._reject_pending(SchedulerClosed('scheduler context closed'))
        self._wake.set()
        cancelled = False
        if self._dispatcher is not None:
            while True:
                try:
                    await asyncio.shield(self._dispatcher)
                    break
                except asyncio.CancelledError:
                    # Cancellation of the context owner must not cancel the task
                    # draining operations that still own backend slots.
                    cancelled = True
        for task in self.tasks.values():
            task.finish_requested = True
            self._release_finished(task)
        if self.checkpoints is not None:
            self.checkpoints.close()
        if cancelled:
            raise asyncio.CancelledError
        if self._fatal is not None and exc_type is None:
            raise self._fatal

    async def submit(self, task_id: str, payload: dict, metadata: dict | None = None) -> dict:
        if self._fatal:
            raise self._fatal
        if self._dispatcher is None or self._closing:
            raise SchedulerClosed('enter the scheduler context before submitting')
        if not isinstance(task_id, str) or not task_id:
            raise ValueError('runtime task ID must be a nonempty string')
        if task_id in self._outstanding:
            raise RuntimeError('one task cannot have multiple outstanding model steps')
        now = self.clock()
        task = self.tasks.setdefault(task_id, TaskState(task_id, now))
        if task.finish_requested:
            raise RuntimeError('cannot submit a finished task')
        self._validate_payload(payload)
        if task.tool_started_s is not None:
            self._end_wait(task, now)
        task.phase = TaskPhase.READY
        self._outstanding.add(task_id)
        future = asyncio.get_running_loop().create_future()
        audit = {k: v for k, v in (metadata or {}).items() if k in AUDIT_FIELDS}
        self._task_audits[task_id] = audit
        ready = ReadyRequest(task_id, dict(payload, prompt=list(payload['prompt'])), audit,
                             self._sequence, now, None, future)
        self._sequence += 1
        if task.admitted_s is None:
            self._waiting.append(ready)
        else:
            ready.enqueue_s = now
            self._ready.append(ready)
        self._admit()
        self._wake.set()
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancel the waiter, not the model operation holding a backend permit.
            future.cancel()
            task.finish_requested, task.finish_status = True, 'cancelled'
            if not task.inflight:
                self._remove_cancelled()
                self._release_finished(task)
                self._admit()
            self._wake.set()
            raise

    async def tool_start(self, task_id: str) -> None:
        task = self.tasks[task_id]
        if task.inflight or task_id in self._outstanding or task.finish_requested:
            raise RuntimeError('external wait must follow a completed model step')
        if task.tool_started_s is None:
            task.tool_started_s = self.clock()
        task.phase = TaskPhase.EXTERNAL_WAIT
        self._wake.set()

    async def tool_end(self, task_id: str) -> None:
        task = self.tasks[task_id]
        self._end_wait(task, self.clock())
        if not task.finish_requested:
            task.phase = TaskPhase.READY
        self._wake.set()

    def _end_wait(self, task: TaskState, now: float) -> None:
        if task.tool_started_s is not None:
            duration = max(0.0, now - task.tool_started_s)
            self.costs.observe_wait(duration)
            self._decision('observed_wait', audit_context=self._task_audits.get(task.task_id, {}),
                           runtime_task_id=task.task_id, status='ok', observed_wait_s=duration)
            task.tool_started_s = None

    async def finish(self, task_id: str, status: str = 'ok') -> None:
        task = self.tasks.get(task_id)
        if task is None:
            return
        task.finish_requested, task.finish_status = True, status
        if not task.inflight:
            self._remove_task_requests(task_id)
            self._release_finished(task)
            self._admit()
        self._wake.set()

    def task_metrics(self, task_id: str) -> dict:
        task = self.tasks[task_id]
        admitted = task.admitted_s
        return dict(arrival_s=task.arrival_s, admitted_s=admitted,
                    first_dispatch_s=task.first_dispatch_s,
                    admission_wait_s=(admitted - task.arrival_s) if admitted is not None else None,
                    first_dispatch_wait_s=(task.first_dispatch_s - task.arrival_s)
                    if task.first_dispatch_s is not None else None,
                    cumulative_service_s=task.cumulative_service_s,
                    completed_steps=task.completed_steps, phase=task.phase.value)

    def _validate_payload(self, payload: dict) -> None:
        prompt, output = payload.get('prompt'), payload.get('n_predict')
        if not isinstance(prompt, list) or not prompt or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in prompt):
            raise ValueError('prompt must be a nonempty list of token IDs')
        if isinstance(output, bool) or not isinstance(output, int) or output < 1:
            raise ValueError('n_predict must be a positive integer')
        if len(prompt) + output > self.config.ctx_per_slot:
            raise ValueError('current prompt plus output limit exceeds per-slot capacity')
        if 'id_slot' in payload:
            raise ValueError('slot selection belongs to the scheduler')

    def _admit(self) -> None:
        active = sum(t.admitted_s is not None and (not t.finish_requested or t.inflight)
                     for t in self.tasks.values())
        while self._waiting and (not self.config.jobs_active or active < self.config.jobs_active):
            ready = self._waiting.pop(0)
            if ready.future.cancelled():
                self._outstanding.discard(ready.task_id)
                continue
            task = self.tasks[ready.task_id]
            task.admitted_s = self.clock()
            ready.enqueue_s = task.admitted_s
            self._ready.append(ready)
            active += 1

    def _rank(self, ready: ReadyRequest, now: float) -> tuple:
        wait = now - ready.enqueue_s
        if wait >= self.config.aging_s:
            return (0, ready.enqueue_s, ready.sequence)
        task = self.tasks[ready.task_id]
        if self.config.queue_rule == 'continuation':
            return (1, 0 if task.completed_steps else 1, ready.enqueue_s, ready.sequence)
        if self.config.queue_rule == 'attained':
            return (1, task.cumulative_service_s, ready.enqueue_s, ready.sequence)
        return (1, ready.enqueue_s, ready.sequence)

    def _decision(self, action: str, ready: ReadyRequest | None = None,
                  audit_context: dict | None = None, **values) -> None:
        context = audit_context if audit_context is not None else (
            ready.audit if ready is not None else self._decision_context)
        self.decision_log.write(dict(context, schema_version=2, policy=self.config.name,
            param_hash=self.config.param_hash, time_s=self.clock(), action=action, **values))

    def _loss(self, slot: SlotState, ready: ReadyRequest) -> float:
        return self.costs.loss_s(len(slot.tokens), common_prefix(slot.tokens, ready.payload['prompt']),
                                 self.config.concurrency)

    def _other_ready(self, ready: ReadyRequest) -> int:
        # The candidate moves from _ready into _running between quote and execute;
        # neither representation should count it as an additional affected task.
        own = self.tasks.get(ready.task_id)
        return sum(item is not ready for item in self._ready) + max(
            0, len(self._running) - int(own is not None and own.inflight))

    def _queue_weight(self, ready: ReadyRequest, slot: SlotState | None = None) -> int:
        free = sum(not s.busy and not s.uncertain for s in self.slots)
        if slot is not None and slot.busy and not slot.uncertain:
            free += 1  # the candidate currently reserves this otherwise idle slot
        count = len(self._ready) + int(not any(item is ready for item in self._ready))
        return max(1, count - max(0, free - 1))

    def _eviction_cost(self, slot: SlotState, ready: ReadyRequest,
                       *, allow_save=True) -> tuple[float, float, str]:
        """Cost of giving the slot away: recompute later, or save now and restore later.

        This is the keep / save / evict comparison. A lease (keep) must beat the
        cheaper of the two ways to give the slot away; otherwise a cheap checkpoint
        would be ignored and a lease would over-protect the slot.
        """
        owner = self.tasks.get(slot.owner)
        if owner is None or owner.finish_requested:
            return 0.0, 0.0, 'discard'
        recompute = self._loss(slot, ready)
        if not self.config.checkpoint or self.checkpoints is None:
            return recompute, recompute, 'recompute'
        tokens = len(slot.tokens)
        snapshot = self.checkpoints.get(slot.owner, slot.version)
        saved = snapshot is not None
        if not saved and (not allow_save or owner.inflight):
            return recompute, recompute, 'recompute'
        save = 0.0 if saved else self.costs.save_s(tokens)
        restore = self.costs.restore_s(tokens)
        reserve = tokens * KV_BYTES_PER_TOKEN + STATE_HEADER_BYTES
        if save is None or restore is None or \
                (not saved and self.checkpoints.bytes_used + reserve > self.checkpoints.budget_bytes):
            return recompute, recompute, 'recompute'
        if not self.costs.checkpoint_worthwhile(tokens, recompute,
                other_ready=self._other_ready(ready), p_use=self.config.checkpoint_use_probability,
                margin_s=self.config.checkpoint_margin_s, saved=saved):
            return recompute, recompute, 'recompute'
        checkpoint = save + restore
        return (checkpoint, recompute, 'checkpoint') if checkpoint < recompute else \
            (recompute, recompute, 'recompute')

    def _protect(self, slot: SlotState, ready: ReadyRequest, now: float, queue_weight: float,
                 *, reconsider=False, allow_save=True) -> bool:
        if slot.lease_until_s is not None:
            return slot.lease_until_s > now
        task = self.tasks.get(slot.owner)
        if task is None or task.finish_requested or task.tool_started_s is None:
            return False
        if slot.lease_version == slot.version and not reconsider:
            return False  # a version gets at most one lease, including after expiry
        effective, recompute, alternative = self._eviction_cost(slot, ready, allow_save=allow_save)
        elapsed = max(0.0, now - task.tool_started_s)
        if self.config.cache_policy == 'ttl':
            duration = max(0.0, task.tool_started_s + self.config.ttl_s - now)
            value = None
        elif self.config.cache_policy == 'cost':
            duration, value = self.costs.best_lease(
                elapsed_s=elapsed, loss_s=effective,
                queue_weight=queue_weight, max_s=self.config.lease_max_s)
        else:
            return False
        slot.lease_version, slot.lease_alternative = slot.version, alternative
        # A re-pricing is a second decision on the same version: the summary
        # counts its lease (the slot is really held) but not a second no_lease.
        again = {'reconsidered': True} if reconsider else {}
        if duration > 0:
            slot.lease_until_s = now + duration
            self._decision('lease', ready, slot=slot.slot_id, owner=slot.owner,
                           context_version=slot.version, duration_s=duration, value_s=value,
                           predicted_loss_s=recompute, effective_loss_s=effective,
                           eviction_alternative=alternative, queue_weight=queue_weight,
                           elapsed_wait_s=elapsed,
                           expected_idle_s=self.costs.expected_idle_s(elapsed, duration),
                           expiry_s=slot.lease_until_s, **again)
            return True
        self._decision('no_lease', ready, slot=slot.slot_id, owner=slot.owner,
                       context_version=slot.version, value_s=value, predicted_loss_s=recompute,
                       effective_loss_s=effective, eviction_alternative=alternative,
                       queue_weight=queue_weight, **again)
        return False

    def _select_slot(self, ready: ReadyRequest, now: float) -> SlotState | None:
        free = [s for s in self.slots if not s.busy and not s.uncertain]
        if not free:
            return None
        if not self.config.affinity:
            return free[0]  # bookkeeping permit; native placement stays in llama-server
        own = [s for s in free if s.owner == ready.task_id]
        if own:
            return max(own, key=lambda s: common_prefix(s.tokens, ready.payload['prompt']))
        empty = [s for s in free if s.owner is None]
        if empty:
            return max(empty, key=lambda s: (common_prefix(s.tokens, ready.payload['prompt']), -s.slot_id))
        overdue = now - ready.enqueue_s >= self.config.aging_s
        queue_weight = self._queue_weight(ready)
        candidates = [s for s in free if overdue or not self._protect(s, ready, now, queue_weight)]
        if not candidates:
            return None
        if self.config.cache_policy == 'cost':
            return min(candidates, key=lambda s: (self._loss(s, ready), s.last_used_s, s.slot_id))
        return min(candidates, key=lambda s: (s.last_used_s, s.slot_id))

    def _pick(self) -> tuple[ReadyRequest, SlotState] | None:
        if self._io_active:
            return None
        now = self.clock()
        ordered = sorted(self._ready, key=lambda r: self._rank(r, now))
        # FIFO may wait for a lease. Aging affects its head but does not silently
        # turn the cache-only ablation into a different request ordering policy.
        candidates = ordered[:1] if self.config.queue_rule == 'fifo' else ordered
        started = time.perf_counter()
        for position, ready in enumerate(candidates):
            slot = self._select_slot(ready, now)
            if slot is not None:
                return self._take(ready, slot, now, started, position, 'rank')
        if self.config.queue_rule == 'fifo' and self.config.affinity:
            # A lease reserves a slot for its owner. When the FIFO head is blocked
            # by that lease, the owner's returning step may use exactly the reserved
            # slot: the head could not have used it, so no capacity is taken from it.
            # Without this, the TTL/cache-only baselines would wait on themselves.
            for position, ready in enumerate(ordered[1:], 1):
                reserved = [s for s in self.slots if s.owner == ready.task_id and not s.busy
                            and not s.uncertain and s.lease_until_s is not None
                            and s.lease_until_s > now]
                if reserved:
                    return self._take(ready, reserved[0], now, started, position, 'lease_owner')
        return None

    def _take(self, ready: ReadyRequest, slot: SlotState, now: float, started: float,
              position: int, reason: str) -> tuple[ReadyRequest, SlotState]:
        self._ready.remove(ready)
        self._decision('dispatch', ready, slot=slot.slot_id if self.config.affinity else None,
                       runtime_task_id=ready.task_id, sequence=ready.sequence,
                       bypassed=position, reason=reason,
                       aged=now - ready.enqueue_s >= self.config.aging_s,
                       ready_wait_s=now - ready.enqueue_s,
                       continuation=self.tasks[ready.task_id].completed_steps > 0,
                       decision_cpu_s=time.perf_counter() - started)
        return ready, slot

    def _next_deadline(self) -> float | None:
        now = self.clock()
        deadlines = [s.lease_until_s for s in self.slots
                     if s.lease_until_s is not None and s.lease_until_s > now]
        deadlines += [r.enqueue_s + self.config.aging_s for r in self._ready
                      if r.enqueue_s + self.config.aging_s > now]
        return max(.0001, min(deadlines) - now) if deadlines else None

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                self._wake.clear()
                self._remove_cancelled()
                self._admit()
                while not self._closing and not self._fatal and len(self._running) < self.config.concurrency:
                    choice = self._pick()
                    if choice is None:
                        break
                    ready, slot = choice
                    slot.busy = True
                    task = self.tasks[ready.task_id]
                    task.inflight, task.phase = True, TaskPhase.RUNNING
                    if task.first_dispatch_s is None:
                        task.first_dispatch_s = self.clock()
                    operation = asyncio.create_task(self._execute(ready, slot))
                    self._running.add(operation)
                if self._closing and not self._running:
                    return
                if self._fatal and not self._running:
                    return
                deadline = self._next_deadline() if self._ready and not self._io_active else None
                if deadline is None:
                    await self._wake.wait()
                else:
                    try:
                        await asyncio.wait_for(self._wake.wait(), deadline)
                    except TimeoutError:
                        pass
        except Exception:
            self._fail_closed('scheduler internal failure')
            if self._running:
                await asyncio.gather(*list(self._running), return_exceptions=True)

    async def _prepare_slot(self, ready: ReadyRequest, slot: SlotState) -> None:
        if not self.config.affinity:
            return
        if slot.owner is not None and slot.owner != ready.task_id:
            _, loss, alternative = self._eviction_cost(slot, ready)
            # Every lease policy prices a context version once. joint-checkpoint
            # re-prices only a version that was priced against a checkpoint which
            # does not happen (the gate, budget or owner state changed, or the save
            # was rejected); otherwise it would differ from joint in more than
            # checkpointing.
            on_checkpoint = (self.config.checkpoint and slot.lease_version == slot.version
                             and slot.lease_alternative == 'checkpoint')
            if on_checkpoint and alternative != 'checkpoint':
                self._reconsider_eviction(ready, slot)
            if (alternative == 'checkpoint' and
                    self.checkpoints.get(slot.owner, slot.version) is None):
                self._io_count += 1
                try:
                    snapshot = await self.checkpoints.save(slot.owner, slot.version, slot.slot_id, len(slot.tokens))
                    self._decision('save_decision' if snapshot else 'save_skipped', ready,
                                   slot=slot.slot_id, owner=slot.owner, predicted_loss_s=loss)
                    if snapshot is not None:
                        self._observe_io('save', snapshot.tokens)
                finally:
                    self._io_count -= 1
                    self._wake.set()
                if snapshot is None and on_checkpoint:
                    self._reconsider_eviction(ready, slot)
            self._decision('evict', ready, slot=slot.slot_id, owner=slot.owner,
                           context_version=slot.version, predicted_loss_s=loss)
        if self.checkpoints is not None:
            task = self.tasks[ready.task_id]
            snapshot = self.checkpoints.get(ready.task_id, task.version)
            if snapshot is not None:
                restore = self.costs.restore_s(snapshot.tokens)
                reused = common_prefix(slot.tokens, ready.payload['prompt'])
                direct = self.costs.prefill_s(len(ready.payload['prompt']), reused, self.config.concurrency)
                after = self.costs.prefill_s(len(ready.payload['prompt']), snapshot.tokens, self.config.concurrency)
                interference = max(0, len(self._running) - 1) * (restore or 0.0)
                if restore is not None and restore + after + interference < direct:
                    self._io_count += 1
                    try:
                        await self.checkpoints.restore(snapshot, slot.slot_id)
                        self._decision('restore_decision', ready, slot=slot.slot_id,
                                       direct_s=direct, restore_s=restore, remaining_prefill_s=after)
                        self._observe_io('restore', snapshot.tokens)
                    finally:
                        self._io_count -= 1
                        self._wake.set()
                else:
                    self._decision('recompute', ready, slot=slot.slot_id, direct_s=direct,
                                   reason='restore_not_cheaper_or_unprofiled')

    def _observe_io(self, kind: str, tokens: int) -> None:
        # The store times the backend call itself, so waiting for its I/O lock
        # (another task's save) is not charged to this snapshot's size.
        # Learning is best effort: a bad timing must never fail a completed save.
        seconds = getattr(self.checkpoints, 'last_roundtrip_s', None)
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) \
                and math.isfinite(seconds) and seconds >= 0:
            self.costs.observe_io(kind, tokens, seconds)

    def _reconsider_eviction(self, ready: ReadyRequest, slot: SlotState) -> None:
        """A checkpoint priced but not taken is not permission to overwrite its owner.

        Called only for a version that was priced against a checkpoint that will
        not happen now; it is priced again against recomputation. Deferring
        releases the candidate's permit before waiting, so the protected owner's
        returning request can use its slot even with K=1 and strict FIFO. A
        version whose actual lease expired never gets a renewed lease here.
        """
        now = self.clock()
        if now - ready.enqueue_s >= self.config.aging_s:
            return
        owner = self.tasks.get(slot.owner)
        if owner is None or owner.finish_requested:
            return
        if slot.lease_until_s is not None:
            if slot.lease_until_s > now:
                raise _DispatchDeferred('existing_lease')
            return
        if (owner.tool_started_s is None and self._loss(slot, ready) > 0
                and any(r.task_id == slot.owner for r in self._ready)):
            # The owner came back after the checkpoint pricing (e.g. during the
            # rejected save). Its remaining wait is zero; hand its resident slot
            # back within the candidate's aging bound.
            remaining = ready.enqueue_s + self.config.aging_s - now
            duration = min(self.config.lease_max_s, remaining)
            if duration > 0:
                slot.lease_version, slot.lease_until_s = slot.version, now + duration
                slot.lease_alternative = 'recompute'
                self._decision('lease', ready, slot=slot.slot_id, owner=slot.owner,
                    context_version=slot.version, duration_s=duration,
                    expiry_s=slot.lease_until_s, reason='checkpoint_fallback_owner_ready',
                    reconsidered=True)
                raise _DispatchDeferred('checkpoint_fallback_owner_ready')
        if self._protect(slot, ready, now, self._queue_weight(ready, slot),
                         reconsider=True, allow_save=False):
            raise _DispatchDeferred('checkpoint_alternative_reconsidered')

    async def _execute(self, ready: ReadyRequest, slot: SlotState) -> None:
        task = self.tasks[ready.task_id]
        started = self.clock()
        self._decision_context = ready.audit
        row: dict[str, Any] = dict(ready.audit, schema_version=2, policy=self.config.name,
            param_hash=self.config.param_hash, runtime_task_id=ready.task_id,
            sequence=ready.sequence, enqueue_s=ready.enqueue_s, dispatch_s=started,
            queue_s=started - ready.enqueue_s, input_tokens=len(ready.payload['prompt']),
            policy_wait_s=started - ready.enqueue_s)
        deferred = False
        try:
            await self._prepare_slot(ready, slot)
            body = dict(ready.payload, return_tokens=True)
            if self.config.affinity:
                body['id_slot'] = slot.slot_id
            model_started = self.clock()
            self._model_requests[ready.sequence] = ready
            for active in self._model_requests.values():
                active.peak_concurrency = max(active.peak_concurrency, len(self._model_requests))
            try:
                response = await self.client.complete(body)
            finally:
                self._model_requests.pop(ready.sequence, None)
            model_done = self.clock()
            if self.config.affinity and response.get('id_slot') != slot.slot_id:
                raise BackendStateUncertain('backend did not honor explicit slot assignment')
            tokens = response.get('tokens')
            if (not isinstance(tokens, list) or len(tokens) != body['n_predict'] or
                    any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens) or
                    response.get('truncated')):
                raise WorkloadIntegrityError('output length mismatch, invalid tokens, or truncation')
            timings = response.get('timings', {})
            self.costs.update(timings, max(1, ready.peak_concurrency))
            service_s = sum(max(0.0, float(timings.get(key, 0) or 0))
                            for key in ('prompt_ms', 'predicted_ms')) / 1000
            task.cumulative_service_s += service_s or (model_done - model_started)
            task.version += 1
            task.completed_steps += 1
            task.phase = TaskPhase.EXTERNAL_WAIT
            if not task.finish_requested and task.tool_started_s is None:
                # The external wait starts when the step ends, not when the caller
                # gets around to tool_start(): the dispatcher runs first and would
                # otherwise see an unprotected owner and evict it without a lease.
                task.tool_started_s = model_done
            if self.config.affinity:
                slot.owner, slot.version = ready.task_id, task.version
                # The last sampled token has not yet been evaluated by llama.cpp.
                slot.tokens = (body['prompt'] + tokens)[:-1]
                slot.lease_until_s = slot.lease_version = None
            if self.checkpoints is not None:
                self.checkpoints.discard(ready.task_id)
            row.update(status='ok', timings=timings, slot=response.get('id_slot'),
                       output_sha=token_digest(tokens),
                       tokens_cached=response.get('tokens_cached'),
                       tokens_evaluated=response.get('tokens_evaluated'), truncated=False,
                       model_start_s=model_started, model_done_s=model_done,
                       model_roundtrip_s=model_done - model_started,
                       observed_peak_concurrency=ready.peak_concurrency,
                       cache_action_s=model_started - started)
            if not ready.future.done():
                ready.future.set_result(response)
        except _DispatchDeferred as error:
            deferred = True
            task.phase = TaskPhase.READY
            if not task.completed_steps:
                task.first_dispatch_s = None  # this attempt never reached the model
            if not task.finish_requested and not ready.future.cancelled():
                self._ready.append(ready)
            self._decision('dispatch_deferred', ready, slot=slot.slot_id, reason=str(error),
                           sequence=ready.sequence)
        except (httpx.RequestError, CheckpointUncertain, BackendStateUncertain) as error:
            self._fail_closed(f'ambiguous backend operation ({type(error).__name__}); confirm or restart backend')
            row.update(status='uncertain', error_type=type(error).__name__)
            if not ready.future.done():
                ready.future.set_exception(self._fatal)
        except Exception as error:
            row.update(status='error', error_type=type(error).__name__)
            # A completed error response ends the operation. Forget local prefix
            # assumptions; a future request can safely re-prefill its full prompt.
            slot.owner, slot.tokens = None, []
            slot.lease_until_s = slot.lease_version = None
            if not ready.future.done():
                ready.future.set_exception(error)
        finally:
            row['done_s'] = self.clock()
            row['roundtrip_s'] = row['done_s'] - started
            if not deferred:
                self.request_log.write(row)
            slot.busy = False
            if not deferred:
                slot.last_used_s = self.clock()
            task.inflight = False
            if not deferred or task.finish_requested or ready.future.cancelled():
                self._outstanding.discard(ready.task_id)
            self._running.discard(asyncio.current_task())
            if task.finish_requested:
                self._release_finished(task)
            for task_id in list(self._deferred_cleanup):
                self._release_finished(self.tasks[task_id])
            self._admit()
            self._wake.set()

    def _release_finished(self, task: TaskState) -> None:
        if task.inflight:
            return
        task.phase = TaskPhase.FINISHED if task.finish_status == 'ok' else TaskPhase.FAILED
        for slot in self.slots:
            if slot.owner == task.task_id and not slot.busy and not slot.uncertain:
                # Release the reservation, retaining knowledge of the still-cached
                # prefix. Releasing ownership does not require erasing the KV pool.
                slot.owner, slot.version = None, 0
                slot.lease_until_s = slot.lease_version = None
        if self._io_active or any(s.busy and s.owner == task.task_id for s in self.slots):
            self._deferred_cleanup.add(task.task_id)
        else:
            if self.checkpoints is not None:
                self.checkpoints.discard(task.task_id)
            self._deferred_cleanup.discard(task.task_id)

    def _remove_task_requests(self, task_id: str) -> None:
        for queue in (self._ready, self._waiting):
            for ready in list(queue):
                if ready.task_id == task_id:
                    queue.remove(ready)
                    if not ready.future.done():
                        ready.future.set_exception(SchedulerClosed('task finished before dispatch'))
                    self._outstanding.discard(task_id)

    def _remove_cancelled(self) -> None:
        for queue in (self._ready, self._waiting):
            for ready in list(queue):
                if ready.future.cancelled():
                    queue.remove(ready)
                    self._outstanding.discard(ready.task_id)
                    self.request_log.write(dict(ready.audit, schema_version=2,
                        policy=self.config.name, param_hash=self.config.param_hash,
                        runtime_task_id=ready.task_id, sequence=ready.sequence,
                        enqueue_s=ready.enqueue_s, done_s=self.clock(),
                        status='cancelled_before_dispatch'))

    def _reject_pending(self, error: Exception) -> None:
        for queue in (self._ready, self._waiting):
            while queue:
                ready = queue.pop(0)
                if not ready.future.done():
                    ready.future.set_exception(error)
                self._outstanding.discard(ready.task_id)
                self.request_log.write(dict(ready.audit, schema_version=2,
                    policy=self.config.name, param_hash=self.config.param_hash,
                    runtime_task_id=ready.task_id, sequence=ready.sequence,
                    enqueue_s=ready.enqueue_s, done_s=self.clock(),
                    status='aborted_before_dispatch', error_type=type(error).__name__))

    def _fail_closed(self, reason: str) -> None:
        self._fatal = self._fatal or BackendStateUncertain(reason)
        for slot in self.slots:
            slot.uncertain = True
        self._reject_pending(self._fatal)
        self._decision('quarantine', reason=reason)
        self._wake.set()
