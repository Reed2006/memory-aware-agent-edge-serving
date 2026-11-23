"""Small real-backend mechanism probes and dev-only cost-profile calibration.

Mock transports validate sequencing only; their timing is never a model performance result.
Uncertain snapshots remain untouched; only definitely completed files are cleaned up.
"""

import argparse
import asyncio
import json
import math
import statistics
import stat
import time
import uuid
from pathlib import Path

import httpx

from phase1.bench import payload
from phase1.core import Jsonl, ModelClient
from phase2.bench import (_server_metadata, _validate_backend_process, backend_lease,
                          canonical_hash, machine_state, require_ac_power)


SUITES = ('affinity', 'incremental-recompute', 'interference', 'restore', 'ram-cache')


class ProbeStateUncertain(RuntimeError):
    """Do not reset this backend or touch its snapshots until it has been inspected."""


def _cleanup_completed(directory, filename):
    # Names are minted by this invocation, and this is called only after both
    # the slot operation and any competing stream have definitely completed.
    if directory is None or Path(filename).name != filename:
        return
    for name in [filename, filename + '.json']:
        path = Path(directory) / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            path.unlink()


def _measure(response, prompt, elapsed):
    timings = response.get('timings', {})
    return dict(prompt_tokens=len(prompt), reused_tokens=timings.get('cache_n', 0),
        evaluated_tokens=timings.get('prompt_n'), prompt_ms=timings.get('prompt_ms'),
        predicted_ms=timings.get('predicted_ms'),
        generated_tokens=len(response.get('tokens', [])), slot=response.get('id_slot'),
        roundtrip_s=elapsed)


async def _complete(client, prompt, output, slot=None):
    options = {'id_slot': slot} if slot is not None else {}
    started = time.perf_counter()
    response = await client.complete(payload(prompt, output, **options))
    if len(response.get('tokens', [])) != output or response.get('truncated', False):
        raise ValueError('probe output length mismatch or truncation')
    if slot is not None and response.get('id_slot') != slot:
        raise ProbeStateUncertain('backend did not honor explicit slot assignment')
    return response, time.perf_counter() - started


async def _slot(client, slot, action, filename=None):
    started = time.perf_counter()
    try:
        result = await client.post(f'/slots/{slot}?action={action}',
                                   {'filename': filename} if filename else {})
    except Exception as error:
        if action in {'save', 'restore'}:
            raise ProbeStateUncertain(f'{action} did not complete reliably; inspect or restart backend') from error
        raise
    return result, time.perf_counter() - started


async def _stream_decode(client, prompt, output, slot, started_event):
    """Record decode token arrival times while slot I/O runs in the other slot."""
    token_times = []
    begin = time.perf_counter()
    async with client.http.stream('POST', '/completion',
            json=payload(prompt, output, id_slot=slot, stream=True)) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data:'):
                continue
            content = line[5:].strip()
            if not content or content == '[DONE]':
                continue
            item = json.loads(content)
            if item.get('content') or item.get('tokens'):
                token_times.append(time.perf_counter())
                started_event.set()
    return dict(stream_roundtrip_s=time.perf_counter() - begin,
        token_arrival_s=token_times,
        stream_max_gap_s=max((b - a for a, b in zip(token_times, token_times[1:])), default=0),
        stream_events=len(token_times))


async def run_probes(client, suites, results_dir: Path, **kwargs) -> list[dict]:
    info = dict(kwargs.get('server_metadata') or {})
    if info.get('pid') is None:
        info['_test_scope'] = str(Path(results_dir).resolve())
    else:
        _validate_backend_process(info, str(client.http.base_url).rstrip('/'))
    kwargs['server_metadata'] = info
    with backend_lease(info, str(client.http.base_url).rstrip('/')):
        return await _run_probes(client, suites, results_dir, **kwargs)


async def _run_probes(client, suites, results_dir: Path, *, shared_prefix_tokens=4891,
                     suffix_lengths=(512, 1024, 2048, 4096), output_tokens=32,
                     trials=2, slot_save_path=None, server_metadata=None,
                     interference_output_tokens=128, stream=True, keep_snapshots=False) -> list[dict]:
    """Exercise a running P>=2 backend; caller controls startup and resource isolation."""
    all_requested = suites == 'all' or suites == ['all']
    suites = list(SUITES) if all_requested else list(suites)
    if set(suites) - set(SUITES):
        raise ValueError('unknown probe suite')
    if shared_prefix_tokens < 1 or output_tokens < 1 or trials < 1 or any(n < 1 for n in suffix_lengths):
        raise ValueError('probe lengths and trial count must be positive')
    if {'restore', 'interference'} & set(suites) and slot_save_path is None:
        raise ValueError('save/restore probes require the actual backend slot-save directory')
    info = server_metadata or {}
    if info and info.get('parallel', info.get('P', 2)) < 2:
        raise ValueError('mechanism probes require at least two server slots')
    ram_available = bool(info.get('cache_ram_mib') and info.get('cache_idle_slots'))
    if 'ram-cache' in suites and not ram_available and not all_requested:
        raise ValueError('ram-cache probe requires --cache-ram and --cache-idle-slots')
    if slot_save_path is not None:
        directory = Path(slot_save_path).resolve()
        if not directory.is_dir():
            raise ValueError('backend slot-save directory does not exist')
        recorded = info.get('slot_save_path')
        if recorded and Path(recorded).resolve() != directory:
            raise ValueError('probe directory differs from backend slot-save directory')
    log = Jsonl(Path(results_dir) / 'raw/probes.jsonl')
    rows = []
    shared = [11] * shared_prefix_tokens
    invocation = uuid.uuid4().hex

    def emit(row):
        row = dict(schema_version=2, probe_id=invocation, recorded_unix=time.time(),
                   synthetic_performance=isinstance(client.http._transport, httpx.MockTransport),
                   model_hash=info.get('model_hash'), kv_precision=info.get('kv_precision', 'f16'),
                   platform_id=info.get('platform_id'), **row)
        log.write(row)
        rows.append(row)

    for suite in suites:
        if suite == 'affinity':
            for trial in range(trials):
                for condition in ['auto', 'explicit']:
                    await client.erase(2)
                    prompts = [shared + [21 + slot] * suffix_lengths[0] for slot in range(2)]
                    generated = [[], []]
                    for step in range(2):
                        for agent in range(2):
                            prompt = prompts[agent] + generated[agent] + ([31] if step else [])
                            base = dict(suite=suite, condition=condition, trial=trial,
                                        agent=agent, step=step, concurrency=1)
                            try:
                                response, elapsed = await _complete(client, prompt, output_tokens,
                                    agent if condition == 'explicit' else None)
                                prompts[agent], generated[agent] = prompt, response['tokens']
                                emit(dict(base, status='ok', **_measure(response, prompt, elapsed)))
                            except Exception as error:
                                emit(dict(base, status='error', error_type=type(error).__name__))
                                if isinstance(error, httpx.TransportError):
                                    raise

        elif suite == 'incremental-recompute':
            for trial in range(trials):
                for suffix in suffix_lengths:
                    for concurrency in [1, 2]:
                        for condition in ['retain', 'overwrite', 'erase']:
                            await client.erase(2)
                            continuations = []
                            for slot in range(concurrency):
                                prompt = shared + [21 + slot] * suffix
                                warm, _ = await _complete(client, prompt, output_tokens, slot)
                                continuation = prompt + warm['tokens'] + [31]
                                continuations.append(continuation)
                                if condition == 'overwrite':
                                    await _complete(client, shared + [41 + slot] * suffix,
                                                    output_tokens, slot)
                                elif condition == 'erase':
                                    await _slot(client, slot, 'erase')
                            async def measure(slot, prompt):
                                base = dict(suite=suite, condition=condition, trial=trial,
                                            suffix_tokens=suffix, concurrency=concurrency, agent=slot)
                                try:
                                    response, elapsed = await _complete(client, prompt, output_tokens, slot)
                                    emit(dict(base, status='ok', **_measure(response, prompt, elapsed)))
                                except Exception as error:
                                    emit(dict(base, status='error', error_type=type(error).__name__))
                                    if isinstance(error, httpx.TransportError):
                                        raise
                            await asyncio.gather(*(measure(slot, prompt)
                                for slot, prompt in enumerate(continuations)))

        elif suite == 'restore':
            for trial in range(trials):
                for suffix in suffix_lengths:
                    for target_slot in [0, 1]:
                        base = dict(suite=suite, condition='same-slot' if target_slot == 0 else 'cross-slot',
                                    trial=trial, suffix_tokens=suffix, concurrency=1)
                        filename = f'probe-{invocation}-restore-{trial}-{suffix}-{target_slot}.bin'
                        try:
                            await client.erase(2)
                            prompt = shared + [21] * suffix
                            warm, _ = await _complete(client, prompt, output_tokens, 0)
                            save, save_s = await _slot(client, 0, 'save', filename)
                            saved_count = len(prompt) + len(warm['tokens']) - 1
                            if save.get('n_saved') != saved_count:
                                raise ProbeStateUncertain('snapshot token count disagrees with resident prefix')
                            continuation = prompt + warm['tokens'] + [31]
                            reference, _ = await _complete(client, continuation, 64, 0)
                            await _complete(client, [41] * len(continuation), output_tokens, target_slot)
                            restored_state, restore_s = await _slot(client, target_slot, 'restore', filename)
                            if restored_state.get('n_restored') != saved_count:
                                raise ProbeStateUncertain('restored token count disagrees with saved state')
                            restored, elapsed = await _complete(client, continuation, 64, target_slot)
                            identical = reference['tokens'] == restored['tokens']
                            reused = restored.get('timings', {}).get('cache_n', 0) >= saved_count
                            verified = identical and reused
                            emit(dict(base, status='ok' if verified else 'mismatch',
                                identical_tokens=identical, restore_verified=verified,
                                save_s=save_s, restore_s=restore_s,
                                snapshot_tokens=saved_count,
                                snapshot_bytes=save.get('n_written'), filename=filename,
                                **_measure(restored, continuation, elapsed)))
                            if verified and not keep_snapshots:
                                _cleanup_completed(slot_save_path, filename)
                        except Exception as error:
                            emit(dict(base, status='error', filename=filename,
                                      error_type=type(error).__name__))
                            if isinstance(error, (httpx.TransportError, ProbeStateUncertain)):
                                raise

        elif suite == 'interference':
            for trial in range(trials):
                for suffix in suffix_lengths:
                    for action in ['none', 'save', 'restore']:
                        base = dict(suite=suite, condition=action, trial=trial,
                                    suffix_tokens=suffix, concurrency=2)
                        filename = f'probe-{invocation}-interference-{trial}-{suffix}-{action}.bin'
                        stream_task = None
                        started_waiter = None
                        try:
                            await client.erase(2)
                            prompt = shared + [21] * suffix
                            warm, _ = await _complete(client, prompt, output_tokens, 0)
                            save, _ = await _slot(client, 0, 'save', filename)
                            started = asyncio.Event()
                            other_prompt = shared + [22] * suffix
                            if stream:
                                stream_task = asyncio.create_task(_stream_decode(client, other_prompt,
                                    interference_output_tokens, 1, started))
                            else:
                                async def roundtrip():
                                    started.set()
                                    response, elapsed = await _complete(client, other_prompt,
                                                                       interference_output_tokens, 1)
                                    return dict(stream_roundtrip_s=elapsed,
                                        **{f'other_{key}': value for key, value in
                                           _measure(response, other_prompt, elapsed).items()})
                                stream_task = asyncio.create_task(roundtrip())
                            # Wait for the first decoded event, then attempt I/O while decoding.
                            started_waiter = asyncio.create_task(started.wait())
                            done, _ = await asyncio.wait([stream_task, started_waiter],
                                                        return_when=asyncio.FIRST_COMPLETED)
                            if stream_task in done:
                                await stream_task
                                overlapped = False
                            else:
                                overlapped = True
                            operation_s = 0
                            operation = {}
                            operation_started = time.perf_counter()
                            if action != 'none':
                                operation, operation_s = await _slot(client, 0, action, filename)
                            operation_done = time.perf_counter()
                            decoded = await stream_task
                            decoded.pop('token_arrival_s', None)
                            emit(dict(base, status='ok', overlap_attempted=overlapped,
                                operation_started_s=operation_started, operation_done_s=operation_done,
                                prompt_tokens=len(prompt) + len(warm['tokens']) - 1,
                                snapshot_bytes=operation.get('n_written', save.get('n_written')),
                                **({f'{action}_s': operation_s} if action != 'none' else {}), **decoded))
                            if not keep_snapshots:
                                _cleanup_completed(slot_save_path, filename)
                        except Exception as error:
                            # Drain stream before another probe can erase either slot.
                            if stream_task is not None:
                                await asyncio.gather(stream_task, return_exceptions=True)
                            emit(dict(base, status='error', filename=filename,
                                      error_type=type(error).__name__))
                            if isinstance(error, (httpx.TransportError, ProbeStateUncertain)):
                                raise
                        finally:
                            if started_waiter is not None and not started_waiter.done():
                                started_waiter.cancel()
                                await asyncio.gather(started_waiter, return_exceptions=True)

        elif suite == 'ram-cache':
            if not ram_available:
                emit(dict(suite=suite, status='skipped_precondition',
                          reason='requires separately launched RAM-cache backend'))
                continue
            for trial in range(trials):
                for suffix in suffix_lengths:
                    await client.erase(2)
                    prompt = shared + [21] * suffix
                    warm, _ = await _complete(client, prompt, output_tokens, 0)
                    # Both idle slots are replaced, forcing reuse to come from host cache.
                    for slot in range(2):
                        await _complete(client, [41 + slot] * len(prompt), output_tokens, slot)
                    continuation = prompt + warm['tokens'] + [31]
                    response, elapsed = await _complete(client, continuation, output_tokens)
                    emit(dict(suite=suite, condition='idle-cache-eviction', trial=trial,
                        suffix_tokens=suffix, concurrency=1, status='ok',
                        host_cache_enabled=True, **_measure(response, continuation, elapsed)))
    return rows


def calibrate_profile(records, wait_records=()) -> dict:
    """Fit conservative per-token medians from completed dev/probe measurements only."""
    records, wait_records = list(records), list(wait_records)
    identities = {}
    for row in records + wait_records:
        if row.get('synthetic_performance'):
            raise ValueError('synthetic mock timings cannot calibrate a performance profile')
        for key in ['platform_id', 'model_hash', 'kv_precision']:
            if row.get(key) is not None:
                identities.setdefault(key, set()).add(row[key])
    for key, values in identities.items():
        if len(values) != 1:
            raise ValueError(f'cannot combine different {key} identities in one cost profile')
    metadata = {key: next(iter(values)) for key, values in identities.items()}
    metadata.update(example_only=not all(key in metadata for key in ['platform_id', 'model_hash', 'kv_precision']),
                    probe_records_hash=canonical_hash(records))
    prefill, decode, saves, restores, waits = {}, [], [], [], []
    for row in records:
        if row.get('status') != 'ok':
            continue
        total = row.get('prompt_tokens', 0)
        evaluated = row.get('evaluated_tokens')
        if evaluated is None:
            evaluated = total - row.get('reused_tokens', 0)
        milliseconds = row.get('prompt_ms')
        # A retained continuation evaluates ~2 tokens; its fixed per-request cost
        # would dominate a per-token ratio. Lost history is what the policy prices.
        if row.get('suite') == 'incremental-recompute' and row.get('condition') != 'retain' \
                and evaluated > 0 and milliseconds is not None:
            value = milliseconds / 1000 / evaluated
            if math.isfinite(value) and value > 0:
                prefill.setdefault(str(row.get('concurrency', 1)), []).append(value)
        tokens = row.get('generated_tokens', 0)
        if tokens > 1 and row.get('predicted_ms') is not None:
            value = row['predicted_ms'] / 1000 / (tokens - 1)
            if math.isfinite(value) and value > 0:
                decode.append(value)
        snapshot_tokens = row.get('snapshot_tokens', total)
        if snapshot_tokens > 0:
            for key, target in [('save_s', saves), ('restore_s', restores)]:
                value = row.get(key)
                if value is not None and math.isfinite(value) and value >= 0:
                    target.append(value / snapshot_tokens)
    for row in wait_records:
        if row.get('workload_split') in {'test', 'eval', 'evaluation'}:
            raise ValueError('eval waits cannot calibrate the frozen dev profile')
        if row.get('action') != 'observed_wait' or row.get('status') != 'ok':
            continue
        wait = row.get('observed_wait_s')
        if wait is not None and math.isfinite(wait) and wait >= 0:
            waits.append(wait)
    def curve(values):
        return dict(seconds_per_token=statistics.median(values), intercept_s=0) if values else None
    metadata.update(prefill_buckets=sorted(prefill, key=int), wait_records_hash=canonical_hash(wait_records))
    return dict(schema='phase2-cost-profile-v1', metadata=metadata,
        prefill={key: curve(values) for key, values in prefill.items()},
        decode=dict(seconds_per_token=statistics.median(decode) if decode else 0),
        save=curve(saves), restore=curve(restores), waits_s=waits,
        ewma_alpha=.2, fallback_wait_s=1, max_wait_samples=512,
        prefill_min_tokens=128, online_prefill=True, online_io=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', choices=['run', 'calibrate'], default='run')
    p.add_argument('--suite', default='all', help='all or comma-separated suite names')
    p.add_argument('--results-dir', type=Path, default=Path('runs/phase2-probes'))
    p.add_argument('--base-url', default='http://127.0.0.1:18080')
    p.add_argument('--server-record', type=Path, default=Path('.server.json'))
    p.add_argument('--platform-id', help='Hardware-specific identity for measured profiles')
    p.add_argument('--shared-prefix-tokens', type=int, default=4891)
    p.add_argument('--suffix-lengths', default='512,1024,2048,4096')
    p.add_argument('--output-tokens', type=int, default=32)
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--interference-output-tokens', type=int, default=128)
    p.add_argument('--roundtrip-only', action='store_true')
    p.add_argument('--keep-snapshots', action='store_true')
    p.add_argument('--input', type=Path, action='append',
                   help='probe JSONL for calibration; repeat to combine suites run separately')
    p.add_argument('--waits-input', type=Path, help='Explicit dev policy_decisions JSONL; observed_wait events only')
    p.add_argument('--output', type=Path, help='calibrated profile JSON path')
    p.add_argument('--allow-battery', action='store_true',
                   help='Permit a battery-powered mechanism check; never calibrate from it')
    return p


def main():
    args = parser().parse_args()
    if args.action == 'calibrate':
        if args.input is None or args.output is None:
            raise SystemExit('calibrate requires --input and --output')
        records = [json.loads(line) for path in args.input
                   for line in path.read_text().splitlines() if line.strip()]
        waits = ([json.loads(line) for line in args.waits_input.read_text().splitlines() if line.strip()]
                 if args.waits_input else [])
        profile = calibrate_profile(records, waits)
        profile['metadata']['sources'] = [str(path.resolve()) for path in args.input] + (
            [str(args.waits_input.resolve())] if args.waits_input else [])
        if not profile['prefill']:
            raise SystemExit('no successful incremental prefill probes to calibrate')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(profile, indent=2) + '\n')
        print('Wrote cost profile', args.output)
        return
    info = _server_metadata(args.server_record)
    info['platform_id'] = args.platform_id or info.get('platform_id')
    if not info['platform_id']:
        raise SystemExit('measured probes require --platform-id matching the later benchmark')
    require_ac_power(args.allow_battery)
    _validate_backend_process(info, args.base_url)
    meta = Jsonl(args.results_dir / 'raw/probe_meta.jsonl')
    meta.write(dict(event='start', suite=args.suite, server=info, machine=machine_state(),
                    allow_battery=args.allow_battery))
    async def run():
        async with ModelClient(args.base_url) as client:
            return await run_probes(client, args.suite.split(','), args.results_dir,
                shared_prefix_tokens=args.shared_prefix_tokens,
                suffix_lengths=[int(n) for n in args.suffix_lengths.split(',')],
                output_tokens=args.output_tokens, trials=args.trials,
                slot_save_path=info.get('slot_save_path'), server_metadata=info,
                interference_output_tokens=args.interference_output_tokens,
                stream=not args.roundtrip_only, keep_snapshots=args.keep_snapshots)
    rows = asyncio.run(run())
    meta.write(dict(event='completed', machine=machine_state(), records=len(rows)))
    print(json.dumps(dict(records=len(rows), failures=sum(r['status'] != 'ok' for r in rows))))


if __name__ == '__main__':
    main()
