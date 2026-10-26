"""Microbenchmarks and tau-derived append-only multi-step serving replay."""

import argparse
import asyncio
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import psutil

from phase1.core import Fifo, Jsonl, ModelClient, append_step, common_prefix

BASE = 'http://127.0.0.1:18080'


def payload(prompt, output=64, **kwargs):
    return dict(prompt=prompt, n_predict=output, ignore_eos=True, stop=[],
                temperature=0, seed=291, return_tokens=True, cache_prompt=True,
                **kwargs)


class Sampler:
    def __init__(self, label):
        self.label = label
        self.stop = asyncio.Event()
        self.pid = json.loads(Path('results/server.json').read_text())['pid']
        self.log = Jsonl(Path('results/raw/resources.jsonl'))

    async def run(self):
        process = psutil.Process(self.pid)
        while not self.stop.is_set():
            memory, swap = psutil.virtual_memory(), psutil.swap_memory()
            self.log.write(dict(label=self.label, time_s=time.perf_counter(),
                server_rss=process.memory_info().rss,
                runner_rss=psutil.Process().memory_info().rss,
                system_available=memory.available, swap_used=swap.used,
                swap_in=swap.sin, swap_out=swap.sout))
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=0.25)
            except TimeoutError:
                pass


async def measured(label, operation):
    sampler = Sampler(label)
    task = asyncio.create_task(sampler.run())
    try:
        return await operation()
    finally:
        sampler.stop.set()
        await task


async def micro(client):
    request_log = Jsonl(Path('results/raw/micro_requests.jsonl'))
    base = await client.tokens('A factual report discusses systems, memory, queues and tools. ' * 1600)
    cases = list(itertools.product([512, 2048, 4096], [64, 128], [1, 2], range(3)))
    random.Random(291).shuffle(cases)
    for index, (length, output, k, repeat) in enumerate(cases):
        await client.erase()
        label = f'micro-L{length}-O{output}-K{k}-R{repeat}'

        async def run():
            async with Fifo(k, client.complete, request_log) as fifo:
                # k distinct requests are admitted together to exercise concurrency.
                return await asyncio.gather(*[
                    fifo.submit({**payload([base[0] + j] + base[1:length], output), 'cache_prompt': False},
                                dict(label=label, length=length, output=output, k=k,
                                     repeat=repeat, request=j)) for j in range(k)
                ], return_exceptions=True)
        await measured(label, run)
        print('micro', index + 1, '/', len(cases), label, flush=True)


async def cache_bench(client):
    log = Jsonl(Path('results/raw/cache.jsonl'))
    base = await client.tokens('An agent retains historical evidence about an order and its items. ' * 900)
    delta = await client.tokens('\nTool result: order is eligible. Continue.\n')
    cases = list(itertools.product([2048, 4096], range(3),
                                  ['retain', 'wait', 'overwrite', 'erase', 'save_restore']))
    random.Random(2526).shuffle(cases)
    for length, repeat, mode in cases:
        await client.erase()
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
            await measured(label, run)
        except Exception as error:
            log.write(dict(label=label, length=length, repeat=repeat, mode=mode,
                           status='error', error_type=type(error).__name__))
        print(label, flush=True)
    Path('results/cache/probe.bin').unlink(missing_ok=True)


async def make_workload(client):
    workloads = []
    selection = json.loads(Path('results/raw/live_test/selection.json').read_text())
    # The local pilot uses the first four IDs in the pre-recorded random order,
    # independent of semantic success. All eight live runs remain available.
    selected_ids = selection['task_ids'][:4]
    for path in sorted(Path('results/raw/live_test').glob('task_*.json')):
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
        raise RuntimeError('No recorded evaluation workloads')
    result = dict(format='tau-derived-append-only-v1', tasks=workloads,
                  caveat='Recorded step counts/output counts and input growth; prompt rewriting is transformed to append-only. Not semantic execution.',
                  source_split='test', source_seed=291, selected_ids=selected_ids,
                  selection_rule='First 4 IDs in the frozen random order; includes semantic failures',
                  available_live_tasks=len(selection['task_ids']))
    Path('results/workload.json').write_text(json.dumps(result))
    return result


async def macro(client, selected_k=None, gaps=None, workload_path=None, repeats=3):
    workload = (json.loads(Path(workload_path).read_text()) if workload_path
                else await make_workload(client))
    if workload_path:
        Path('results/workload.json').write_text(json.dumps(workload))
    request_log = Jsonl(Path('results/raw/macro_requests.jsonl'))
    job_log = Jsonl(Path('results/raw/macro_jobs.jsonl'))
    run_log = Jsonl(Path('results/raw/macro_runs.jsonl'))
    cases = list(itertools.product(selected_k or [1, 2], gaps or ['recorded', 'zero'], range(repeats)))
    random.Random(1025).shuffle(cases)
    for k, gap_mode, repeat in cases:
        await client.erase()
        label = f'macro-K{k}-{gap_mode}-R{repeat}'
        # Repeated seeds alter arrival staggering while each configuration sees
        # the same schedule for that repeat. This is a finite batch, not open loop.
        rng = random.Random(291 + repeat)
        arrivals = [rng.uniform(0, 0.3) for _ in workload['tasks']]
        start = time.perf_counter()

        async def run():
            async with Fifo(k, client.complete, request_log) as fifo:
                async def job(source, arrival):
                    await asyncio.sleep(max(0, start + arrival - time.perf_counter()))
                    arrived = time.perf_counter()
                    prompt, generated = [], []
                    total_output, prefix_verified = 0, True
                    status, done_steps = 'ok', 0
                    try:
                        for step_index, step in enumerate(source['steps']):
                            if step_index and gap_mode == 'recorded':
                                await asyncio.sleep(step['gap_s'])
                            previous = prompt + generated
                            prompt = append_step(prompt, generated, step['delta'])
                            prefix_verified &= common_prefix(previous, prompt) == len(previous)
                            response = await fifo.submit(payload(prompt, step['output']),
                                dict(label=label, k=k, repeat=repeat, gap_mode=gap_mode,
                                     task_id=source['task_id'], step=step_index,
                                     expected_output=step['output'], prefix_verified=prefix_verified))
                            generated = response['tokens']
                            if len(generated) != step['output'] or response.get('truncated'):
                                raise ValueError('output length mismatch or truncation')
                            total_output += len(generated)
                            done_steps += 1
                    except Exception as error:
                        status = type(error).__name__
                    row = dict(label=label, k=k, repeat=repeat, gap_mode=gap_mode,
                        task_id=source['task_id'], arrival_s=arrived, done_s=time.perf_counter(),
                        status=status, steps=done_steps, expected_steps=len(source['steps']),
                        output_tokens=total_output, prefix_verified=prefix_verified)
                    row['latency_s'] = row['done_s'] - arrived
                    job_log.write(row)
                    return row
                return await asyncio.gather(*[job(source, arrival)
                    for source, arrival in zip(workload['tasks'], arrivals)])
        rows = await measured(label, run)
        elapsed = max(r['done_s'] for r in rows) - min(r['arrival_s'] for r in rows)
        completed = sum(r['status'] == 'ok' for r in rows)
        summary = dict(label=label, k=k, repeat=repeat, gap_mode=gap_mode,
                       tasks=len(rows), completed=completed, makespan_s=elapsed,
                       jobs_per_minute=completed / elapsed * 60)
        run_log.write(summary)
        print(json.dumps(summary), flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['micro', 'cache', 'macro'])
    parser.add_argument('--workload', help='Use frozen workload without importing or calling tau/DeepSeek')
    parser.add_argument('--gap-mode', choices=['recorded', 'zero', 'both'], default='recorded')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    async with ModelClient(BASE) as client:
        # Warm up the compute path; individual suites reset their cache state.
        await client.complete(payload('Warm up the model.', 16))
        if args.suite == 'micro':
            await micro(client)
        elif args.suite == 'cache':
            await cache_bench(client)
        else:
            await macro(client, gaps=None if args.gap_mode == 'both' else [args.gap_mode],
                        workload_path=args.workload, repeats=args.repeats)


if __name__ == '__main__':
    asyncio.run(main())
