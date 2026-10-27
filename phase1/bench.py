"""Microbenchmarks and tau-derived append-only multi-step serving replay.

Macro defaults are the Phase 1 main condition. Sweep axes for macro:
  --k        request concurrency of the external FIFO (in-flight model calls)
  --jobs-active  J, task-level admission: at most J agent tasks hold state at once
  --gap      tool/idle delay between steps: recorded | zero | fixed:S | exp:MEAN | lognormal:MEDIAN:SIGMA
  --arrival  batch (all tasks within 0.3 s) | poisson:JOBS_PER_MIN
Server-side slots P and KV budget come from the running llama-server (phase1.server)
and are recorded with every row. A J=1, K=1 run supplies the isolated service time
used by phase1.summarize for slowdown/fairness.
"""

import argparse
import asyncio
import itertools
import json
import math
import random
import subprocess
import time
from pathlib import Path

import psutil

from phase1.core import Fifo, Jsonl, ModelClient, add_results_arg, append_step, common_prefix, server_info

BATCH_SPREAD_S = 0.3


def payload(prompt, output=64, **kwargs):
    return dict(prompt=prompt, n_predict=output, ignore_eos=True, stop=[],
                temperature=0, seed=291, return_tokens=True, cache_prompt=True,
                **kwargs)


def gap_delay(spec: str, recorded: float, key: str) -> float:
    """Deterministic per-(repeat, job, step) delay so every configuration sees the same gaps."""
    kind, *params = spec.split(':')
    rng = random.Random(f'gap-{spec}-{key}')
    if kind == 'recorded':
        return recorded
    if kind == 'zero':
        return 0.0
    if kind == 'fixed':
        return float(params[0])
    if kind == 'exp':
        return rng.expovariate(1 / float(params[0]))
    if kind == 'lognormal':
        median, sigma = map(float, params)
        return rng.lognormvariate(math.log(median), sigma)
    raise ValueError(f'Unknown gap spec {spec!r}')


def arrival_times(spec: str, count: int, seed: int) -> list[float]:
    kind, *params = spec.split(':')
    rng = random.Random(f'arrival-{spec}-{seed}')
    if kind == 'batch':
        return [rng.uniform(0, BATCH_SPREAD_S) for _ in range(count)]
    if kind == 'poisson':
        rate_per_s = float(params[0]) / 60
        times, now = [], 0.0
        for _ in range(count):
            times.append(now)
            now += rng.expovariate(rate_per_s)
        return times
    raise ValueError(f'Unknown arrival spec {spec!r}')


def job_sources(tasks: list, count: int, seed: int) -> list:
    """Balanced reuse of source tasks: each appears floor/ceil(count/len) times, order seeded."""
    order = []
    rng = random.Random(f'jobs-{seed}')
    while len(order) < count:
        block = list(range(len(tasks)))
        rng.shuffle(block)
        order += block
    return [tasks[i] for i in order[:count]]


class Sampler:
    def __init__(self, label, results):
        self.label = label
        self.stop = asyncio.Event()
        self.pid = server_info().get('pid')
        self.log = Jsonl(results / 'raw/resources.jsonl')

    async def run(self):
        process = psutil.Process(self.pid) if self.pid else None
        while not self.stop.is_set():
            memory, swap = psutil.virtual_memory(), psutil.swap_memory()
            self.log.write(dict(label=self.label, time_s=time.perf_counter(),
                server_rss=process.memory_info().rss if process else None,
                runner_rss=psutil.Process().memory_info().rss,
                system_available=memory.available, swap_used=swap.used,
                swap_in=swap.sin, swap_out=swap.sout))
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=0.25)
            except TimeoutError:
                pass


async def measured(label, operation, results):
    sampler = Sampler(label, results)
    task = asyncio.create_task(sampler.run())
    try:
        return await operation()
    finally:
        sampler.stop.set()
        await task


async def server_config(client) -> dict:
    props = await client.props()
    info = server_info()
    slots = props.get('total_slots') or info.get('parallel', 2)
    return dict(P=slots,
                ctx_per_slot=props.get('default_generation_settings', {}).get('n_ctx'),
                ctx_size=info.get('ctx_size'), kv_unified=info.get('kv_unified'),
                cache_ram_mib=info.get('cache_ram_mib'), build=props.get('build_info'))


def power_state() -> str | None:
    try:
        return subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # not macOS


def git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def micro(client, results, lengths, outputs, ks, repeats, slots):
    request_log = Jsonl(results / 'raw/micro_requests.jsonl')
    sentence = 'A factual report discusses systems, memory, queues and tools. '
    base = await client.tokens(sentence * max(1600, max(lengths) // 8))
    if len(base) < max(lengths):
        raise ValueError('Synthetic text is shorter than the largest requested input')
    cases = list(itertools.product(lengths, outputs, ks, range(repeats)))
    random.Random(291).shuffle(cases)
    for index, (length, output, k, repeat) in enumerate(cases):
        await client.erase(slots)
        label = f'micro-L{length}-O{output}-K{k}-R{repeat}'

        async def run():
            async with Fifo(k, client.complete, request_log) as fifo:
                # k distinct requests are admitted together to exercise concurrency.
                return await asyncio.gather(*[
                    fifo.submit({**payload([base[0] + j] + base[1:length], output), 'cache_prompt': False},
                                dict(label=label, length=length, output=output, k=k,
                                     repeat=repeat, request=j)) for j in range(k)
                ], return_exceptions=True)
        await measured(label, run, results)
        print('micro', index + 1, '/', len(cases), label, flush=True)


async def cache_bench(client, results, lengths, repeats, slots):
    log = Jsonl(results / 'raw/cache.jsonl')
    base = await client.tokens('An agent retains historical evidence about an order and its items. ' *
                               max(900, max(lengths) // 8))
    delta = await client.tokens('\nTool result: order is eligible. Continue.\n')
    cases = list(itertools.product(lengths, range(repeats),
                                  ['retain', 'wait', 'overwrite', 'erase', 'save_restore']))
    random.Random(2526).shuffle(cases)
    for length, repeat, mode in cases:
        await client.erase(slots)
        label = f'cache-L{length}-R{repeat}-{mode}'

        async def run():
            first = await client.complete(payload(base[:length], 32, id_slot=0))
            prompt = append_step(base[:length], first['tokens'], delta)
            row = dict(label=label, length=length, repeat=repeat, mode=mode)
            start = time.perf_counter()
            if mode == 'wait':
                await asyncio.sleep(0.5)
            elif mode == 'overwrite':
                await client.complete(payload([base[0] + 10] + list(reversed(base[1:length])), 32, id_slot=0))
            elif mode == 'erase':
                await client.post('/slots/0?action=erase', {})
            elif mode == 'save_restore':
                action_start = time.perf_counter()
                row['save'] = await client.post('/slots/0?action=save', {'filename': 'probe.bin'})
                row['save_roundtrip_s'] = time.perf_counter() - action_start
                action_start = time.perf_counter()
                row['erase'] = await client.post('/slots/0?action=erase', {})
                row['erase_roundtrip_s'] = time.perf_counter() - action_start
                await client.complete(payload(list(reversed(base[:length])), 32, id_slot=0))
                await client.post('/slots/0?action=erase', {})
                action_start = time.perf_counter()
                row['restore'] = await client.post('/slots/0?action=restore', {'filename': 'probe.bin'})
                row['restore_roundtrip_s'] = time.perf_counter() - action_start
            row['intervening_s'] = time.perf_counter() - start
            start = time.perf_counter()
            response = await client.complete(payload(prompt, 32, id_slot=0))
            row.update(roundtrip_s=time.perf_counter() - start, timings=response['timings'],
                       input_tokens=len(prompt), predicted_n=response['timings']['predicted_n'])
            log.write(row)
        try:
            await measured(label, run, results)
        except Exception as error:
            log.write(dict(label=label, length=length, repeat=repeat, mode=mode,
                           status='error', error_type=type(error).__name__))
        print(label, flush=True)
    slot_dir = Path(server_info().get('slot_save_path', results / 'cache'))
    (slot_dir / 'probe.bin').unlink(missing_ok=True)


def make_workload(live_dir: Path, source_count: int):
    workloads = []
    selection = json.loads((live_dir / 'selection.json').read_text())
    # Use a prefix of the pre-recorded random order, independent of semantic success.
    selected_ids = selection['task_ids'][:source_count] if source_count else selection['task_ids']
    for path in sorted(live_dir.glob('task_*.json')):
        record = json.loads(path.read_text())
        if record['task_id'] not in selected_ids:
            continue
        calls = record['calls']
        if not calls:
            continue
        steps = []
        for index, call in enumerate(calls):
            output = int(call['response'].get('timings', {}).get('predicted_n') or
                         call['response']['usage']['completion_tokens'])
            if index == 0:
                delta = call['prompt_tokens']
                source_growth = len(delta)
            else:
                source_growth = len(call['prompt_tokens']) - len(calls[index-1]['prompt_tokens'])
                # Append-only replay retains the *newly generated* tokens. New
                # observation length is calibrated to the captured input growth.
                previous_output = steps[-1]['output']
                count = max(0, source_growth - previous_output)
                delta = call['prompt_tokens'][-count:] if count else []
            steps.append(dict(delta=delta, output=output,
                              gap_s=call['between_calls_s'] if index else 0,
                              source_prompt_n=len(call['prompt_tokens']),
                              source_growth=source_growth))
        workloads.append(dict(task_id=record['task_id'], steps=steps,
                              live_status=record['status'], env_reward=record.get('env_reward')))
    if not workloads:
        raise RuntimeError(f'No recorded workloads in {live_dir}')
    return dict(format='tau-derived-append-only-v1', tasks=workloads,
                caveat='Recorded step counts/output counts and input growth; prompt rewriting is transformed to append-only. Not semantic execution.',
                source_dir=str(live_dir), source_split=selection.get('split'),
                source_seed=selection.get('seed'), selected_ids=selected_ids,
                selection_rule=f'First {source_count or "all"} IDs in the frozen random order; includes semantic failures',
                available_live_tasks=len(selection['task_ids']))


def quarantine_partial(results: Path, label: str) -> None:
    """Move records of an interrupted run (no macro_runs entry) to raw/interrupted/ so a
    rerun of the same label is not double-counted; the interrupted attempt stays on disk."""
    for name in ['macro_requests.jsonl', 'macro_jobs.jsonl', 'resources.jsonl']:
        path = results / 'raw' / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines(keepends=True)
        partial = [s for s in lines if s.strip() and json.loads(s).get('label') == label]
        if not partial:
            continue
        target = results / 'raw/interrupted' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a') as stream:
            stream.writelines(partial)
        path.write_text(''.join(s for s in lines if s not in partial))
        print(f'quarantined {len(partial)} partial rows of {label} from {name}', flush=True)


async def macro(client, results, args, server):
    if args.workload:
        workload = json.loads(Path(args.workload).read_text())
    else:
        workload = make_workload(Path(args.live_dir), args.source_count)
    (results / 'workload.json').write_text(json.dumps(workload))
    tasks = workload['tasks']
    if args.steps_limit:
        # Smoke tests only: truncating changes the workload, and the label records it.
        tasks = [dict(t, steps=t['steps'][:args.steps_limit]) for t in tasks]
    num_jobs = args.num_jobs or len(tasks)
    request_log = Jsonl(results / 'raw/macro_requests.jsonl')
    job_log = Jsonl(results / 'raw/macro_jobs.jsonl')
    run_log = Jsonl(results / 'raw/macro_runs.jsonl')
    P, C = server['P'], server['ctx_size']
    cases = list(itertools.product(args.k, args.jobs_active, args.gap, args.arrival,
                                   range(args.repeat_start, args.repeat_start + args.repeats)))
    random.Random(1025).shuffle(cases)
    done = set()
    if args.skip_done and (results / 'raw/macro_runs.jsonl').exists():
        done = {json.loads(s)['label'] for s in (results / 'raw/macro_runs.jsonl').read_text().splitlines() if s}
    for k, j, gap_spec, arrival_spec, repeat in cases:
        suffix = f'-S{args.steps_limit}' if args.steps_limit else ''
        mode = ('u' if server['kv_unified'] else '') + (f'-RAM{server["cache_ram_mib"]}' if server['cache_ram_mib'] else '')
        label = f'macro-P{P}{mode}-C{C}-K{k}-J{j or "all"}-{gap_spec}-{arrival_spec}-R{repeat}{suffix}'
        if label in done:
            print('skip completed', label, flush=True)
            continue
        quarantine_partial(results, label)
        await client.erase(P)
        config = dict(label=label, P=P, ctx_size=C, ctx_per_slot=server['ctx_per_slot'],
                      kv_unified=server['kv_unified'], cache_ram_mib=server['cache_ram_mib'],
                      k=k, j=j, gap_mode=gap_spec, arrival=arrival_spec, repeat=repeat,
                      steps_limit=args.steps_limit)
        # Same repeat => same arrivals, job order and gaps across all configurations.
        sources = job_sources(tasks, num_jobs, 291 + repeat)
        arrivals = arrival_times(arrival_spec, num_jobs, 291 + repeat)
        admission = asyncio.Semaphore(j) if j else None
        start = time.perf_counter()

        async def run():
            async with Fifo(k, client.complete, request_log) as fifo:
                async def job(job_index, source, arrival):
                    await asyncio.sleep(max(0, start + arrival - time.perf_counter()))
                    arrived = time.perf_counter()
                    if admission:
                        await admission.acquire()
                    admitted = time.perf_counter()
                    prompt, generated = [], []
                    total_output, prefix_verified = 0, True
                    status, error, done_steps, gap_total = 'ok', None, 0, 0.0
                    try:
                        for step_index, step in enumerate(source['steps']):
                            if step_index:
                                delay = gap_delay(gap_spec, step['gap_s'], f'{repeat}-{job_index}-{step_index}')
                                gap_total += delay
                                await asyncio.sleep(delay)
                            previous = prompt + generated
                            prompt = append_step(prompt, generated, step['delta'])
                            prefix_verified &= common_prefix(previous, prompt) == len(previous)
                            # The last generated token is usually not yet in KV, hence -1.
                            reusable = max(0, len(previous) - 1)
                            response = await fifo.submit(payload(prompt, step['output']), dict(
                                config, job=job_index, task_id=source['task_id'], step=step_index,
                                expected_output=step['output'], prefix_verified=prefix_verified,
                                reusable_tokens=reusable, new_tokens=len(prompt) - reusable))
                            generated = response['tokens']
                            if len(generated) != step['output'] or response.get('truncated'):
                                raise ValueError('output length mismatch or truncation')
                            total_output += len(generated)
                            done_steps += 1
                    except Exception as exc:
                        status = type(exc).__name__
                        error = str(exc)[:300]
                    finally:
                        if admission:
                            admission.release()
                    row = dict(config, job=job_index, task_id=source['task_id'],
                        arrival_s=arrived, admitted_s=admitted, done_s=time.perf_counter(),
                        admission_wait_s=admitted - arrived, gap_total_s=gap_total,
                        status=status, error=error, steps=done_steps,
                        expected_steps=len(source['steps']), output_tokens=total_output,
                        expected_output_tokens=sum(s['output'] for s in source['steps']),
                        max_input_tokens=sum(len(s['delta']) + s['output'] for s in source['steps']) - source['steps'][-1]['output'],
                        prefix_verified=prefix_verified)
                    row['latency_s'] = row['done_s'] - arrived
                    row['service_s'] = row['done_s'] - admitted
                    job_log.write(row)
                    return row
                return await asyncio.gather(*[job(i, source, arrival)
                    for i, (source, arrival) in enumerate(zip(sources, arrivals))])
        rows = await measured(label, run, results)
        elapsed = max(r['done_s'] for r in rows) - min(r['arrival_s'] for r in rows)
        completed = sum(r['status'] == 'ok' for r in rows)
        summary = dict(config, tasks=len(rows), completed=completed, failed=len(rows) - completed,
                       makespan_s=elapsed, jobs_per_minute=completed / elapsed * 60)
        run_log.write(summary)
        print(json.dumps(summary), flush=True)


def int_list(text):
    return [int(x) for x in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('suite', choices=['micro', 'cache', 'macro'])
    add_results_arg(parser)
    parser.add_argument('--base-url', default=None, help='Defaults to the running phase1.server port')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--repeat-start', type=int, default=0,
                        help='Index of the first macro repeat (seeds arrivals, job order and gaps)')
    parser.add_argument('--timeout', type=float, default=300, help='Per-request HTTP deadline (s)')
    # micro / cache
    parser.add_argument('--lengths', type=int_list, default=None, help='e.g. 512,2048,4096,8192')
    parser.add_argument('--outputs', type=int_list, default=[64, 128])
    parser.add_argument('--k', type=int_list, default=[1, 2], help='Request concurrency list')
    # macro
    parser.add_argument('--workload', help='Frozen workload JSON; no tau/DeepSeek needed')
    parser.add_argument('--live-dir', default='runs/final/raw/live_test')
    parser.add_argument('--source-count', type=int, default=0, help='0 = all captured tasks')
    parser.add_argument('--num-jobs', type=int, default=0, help='Jobs per run; default one per source task')
    parser.add_argument('--jobs-active', type=lambda s: [int(x) for x in s.split(',')], default=[0],
                        help='Task admission limits J; 0 = unlimited')
    parser.add_argument('--gap', type=lambda s: s.split(','), default=['lognormal:2:1'])
    parser.add_argument('--arrival', type=lambda s: s.split(','), default=['poisson:1'])
    parser.add_argument('--steps-limit', type=int, default=0, help='Smoke tests only')
    parser.add_argument('--skip-done', action='store_true', help='Skip macro runs whose label is already logged')
    args = parser.parse_args()
    for spec in args.gap:
        gap_delay(spec, 0.0, 'validate')
    for spec in args.arrival:
        arrival_times(spec, 1, 0)
    results = Path(args.results_dir)
    (results / 'raw').mkdir(parents=True, exist_ok=True)
    asyncio.run(run_suite(args, results))


async def run_suite(args, results):
    base = args.base_url or f'http://127.0.0.1:{server_info().get("port", 18080)}'
    async with ModelClient(base, timeout=args.timeout) as client:
        server = await server_config(client)
        Jsonl(results / 'raw/runs_meta.jsonl').write(dict(
            suite=args.suite, started_unix=time.time(), git_commit=git_commit(),
            args=vars(args), server=server, server_command=server_info().get('command'),
            power=power_state()))
        # Warm up the compute path; individual suites reset their cache state.
        await client.complete(payload('Warm up the model.', 16))
        if args.suite == 'micro':
            await micro(client, results, args.lengths or [512, 2048, 4096], args.outputs,
                        args.k, args.repeats, server['P'])
        elif args.suite == 'cache':
            await cache_bench(client, results, args.lengths or [2048, 4096], args.repeats, server['P'])
        else:
            await macro(client, results, args, server)


if __name__ == '__main__':
    main()
