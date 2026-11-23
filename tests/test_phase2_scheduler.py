import asyncio
import json

import httpx
import pytest

from phase1.core import Jsonl, common_prefix
from phase2.scheduler import BackendStateUncertain, Scheduler
from phase2.state import PolicyConfig


class Backend:
    """A slot-bearing fake, not a source of performance measurements."""

    def __init__(self, slots=2, delay=.002):
        self.slots = [[] for _ in range(slots)]
        self.delay = delay
        self.started = []
        self.active = self.peak = 0
        self.block = None
        self.entered = asyncio.Event()
        self.error = None

    async def complete(self, payload):
        slot = payload.get('id_slot', 0)
        self.started.append((payload['prompt'][-1], slot))
        self.active += 1
        self.peak = max(self.active, self.peak)
        self.entered.set()
        try:
            if self.block:
                await self.block.wait()
            await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            old = self.slots[slot]
            cached = common_prefix(old, payload['prompt']) if payload.get('cache_prompt', True) else 0
            output = [700 + i for i in range(payload['n_predict'])]
            self.slots[slot] = (payload['prompt'] + output)[:-1]
            return dict(id_slot=slot, tokens=output, timings=dict(
                cache_n=cached, prompt_n=len(payload['prompt']) - cached,
                prompt_ms=float(len(payload['prompt']) - cached),
                predicted_n=len(output), predicted_ms=float(len(output))))
        finally:
            self.active -= 1


def request(marker, prompt=None):
    return dict(prompt=prompt or [1, marker], n_predict=2, cache_prompt=True)


def make_scheduler(tmp_path, backend, name='affinity', **kwargs):
    return Scheduler(backend, PolicyConfig.for_policy(name, **kwargs),
                     Jsonl(tmp_path / 'requests.jsonl'), Jsonl(tmp_path / 'decisions.jsonl'))


@pytest.mark.asyncio
async def test_affinity_uses_both_slots_and_reuses_exact_history(tmp_path):
    backend = Backend()
    async with make_scheduler(tmp_path, backend, concurrency=1) as scheduler:
        a = await scheduler.submit('a', request(10))
        await scheduler.tool_start('a')
        await scheduler.submit('b', request(20))
        await scheduler.tool_end('a')
        response = await scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30]))
        assert response['timings']['cache_n'] == 3
        assert backend.started == [(10, 0), (20, 1), (30, 0)]
        await scheduler.finish('a')
        await scheduler.finish('b')
    assert all(slot.owner is None for slot in scheduler.slots)
    rows = [json.loads(line) for line in (tmp_path / 'requests.jsonl').read_text().splitlines()]
    assert len(rows) == 3 and all(row['status'] == 'ok' for row in rows)


@pytest.mark.asyncio
async def test_native_fifo_leaves_slot_placement_to_backend(tmp_path):
    backend = Backend()
    async with make_scheduler(tmp_path, backend, 'fifo', concurrency=1) as scheduler:
        await scheduler.submit('a', request(10))
        await scheduler.submit('b', request(20))
    assert backend.started == [(10, 0), (20, 0)]


@pytest.mark.asyncio
async def test_dispatch_bound_and_finished_reservations_release(tmp_path):
    backend = Backend(delay=.005)
    async with make_scheduler(tmp_path, backend) as scheduler:
        await asyncio.gather(*(scheduler.submit(str(i), request(i)) for i in range(8)))
        for i in range(8):
            await scheduler.finish(str(i))
    assert backend.peak == 2 and backend.active == 0
    assert all(not slot.busy and slot.owner is None for slot in scheduler.slots)


@pytest.mark.asyncio
async def test_task_cap_does_not_block_admitted_continuation(tmp_path):
    backend = Backend()
    async with make_scheduler(tmp_path, backend, jobs_active=1) as scheduler:
        a = await scheduler.submit('a', request(10))
        await scheduler.tool_start('a')
        waiting = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.sleep(.005)
        assert not waiting.done()
        await scheduler.tool_end('a')
        await scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30]))
        await scheduler.finish('a')
        await waiting
        assert scheduler.task_metrics('b')['admission_wait_s'] > 0
        await scheduler.finish('b')
    assert [event[0] for event in backend.started] == [10, 30, 20]


async def queued_order(tmp_path, policy, aging_s=10, age_new=False):
    backend = Backend()
    async with make_scheduler(tmp_path, backend, policy, slots=1, concurrency=1, aging_s=aging_s) as scheduler:
        first = await scheduler.submit('continuing', request(10))
        backend.block = asyncio.Event()
        backend.entered.clear()
        busy = asyncio.create_task(scheduler.submit('blocker', request(40)))
        await backend.entered.wait()
        new = asyncio.create_task(scheduler.submit('new', request(20)))
        await asyncio.sleep(.02 if age_new else .001)
        continuation = asyncio.create_task(scheduler.submit(
            'continuing', request(30, [1, 10] + first['tokens'] + [30])))
        await asyncio.sleep(0)
        backend.block.set()
        await asyncio.gather(busy, new, continuation)
    return [event[0] for event in backend.started]


@pytest.mark.asyncio
async def test_continuation_priority_and_aging_have_observable_effect(tmp_path):
    assert await queued_order(tmp_path, 'queue') == [10, 40, 30, 20]
    assert await queued_order(tmp_path, 'affinity') == [10, 40, 20, 30]
    assert await queued_order(tmp_path, 'queue', .01, age_new=True) == [10, 40, 20, 30]


@pytest.mark.asyncio
async def test_ttl_expires_without_new_events_or_indefinite_renewal(tmp_path):
    backend = Backend()
    async with make_scheduler(tmp_path, backend, 'ttl', slots=1, concurrency=1,
                              ttl_s=.02, aging_s=1) as scheduler:
        await scheduler.submit('a', request(10))
        await scheduler.tool_start('a')
        await asyncio.wait_for(scheduler.submit('b', request(20)), .5)
    decisions = [json.loads(line) for line in (tmp_path / 'decisions.jsonl').read_text().splitlines()]
    assert any(row['action'] == 'lease' for row in decisions)
    assert backend.started[-1][0] == 20


@pytest.mark.asyncio
async def test_cancelled_caller_never_releases_an_inflight_backend_permit(tmp_path):
    backend = Backend()
    backend.block = asyncio.Event()
    async with make_scheduler(tmp_path, backend, concurrency=1) as scheduler:
        first = asyncio.create_task(scheduler.submit('a', request(10)))
        await backend.entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await scheduler.finish('a', status='cancelled')
        second = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.sleep(.005)
        assert len(backend.started) == 1
        backend.block.set()
        await second
    assert backend.peak == 1


@pytest.mark.asyncio
async def test_transport_timeout_fails_closed_and_never_reuses_slot(tmp_path):
    backend = Backend()
    backend.error = httpx.ReadTimeout('ambiguous')
    scheduler = make_scheduler(tmp_path, backend, concurrency=1)
    with pytest.raises(BackendStateUncertain):
        async with scheduler:
            results = await asyncio.gather(scheduler.submit('a', request(10)),
                                           scheduler.submit('b', request(20)), return_exceptions=True)
            assert all(isinstance(result, BackendStateUncertain) for result in results)
    assert len(backend.started) == 1 and all(slot.uncertain for slot in scheduler.slots)
    rows = [json.loads(line) for line in (tmp_path / 'requests.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and {row['status'] for row in rows} == {'uncertain', 'aborted_before_dispatch'}


@pytest.mark.asyncio
async def test_definitive_server_error_recovers_without_logging_response_body(tmp_path):
    backend = Backend()
    backend.error = httpx.HTTPStatusError('secret body', request=httpx.Request('POST', 'http://test'),
                                        response=httpx.Response(400, text='do not log this body'))
    async with make_scheduler(tmp_path, backend, concurrency=1) as scheduler:
        with pytest.raises(httpx.HTTPStatusError):
            await scheduler.submit('a', request(10))
        await scheduler.finish('a', status='failed')
        backend.error = None
        await scheduler.submit('b', request(20))
    text = (tmp_path / 'requests.jsonl').read_text()
    assert 'do not log' not in text and 'secret body' not in text


@pytest.mark.asyncio
async def test_validation_prevents_truncation_and_parallel_steps_of_one_task(tmp_path):
    backend = Backend()
    backend.block = asyncio.Event()
    async with make_scheduler(tmp_path, backend, ctx_per_slot=10) as scheduler:
        with pytest.raises(ValueError):
            await scheduler.submit('too-big', request(1, [1] * 10))
        with pytest.raises(ValueError):
            await scheduler.submit('bad-prompt', {'prompt': 'text', 'n_predict': 1})
        first = asyncio.create_task(scheduler.submit('a', request(10)))
        await backend.entered.wait()
        with pytest.raises(RuntimeError):
            await scheduler.submit('a', request(20))
        backend.block.set()
        await first


@pytest.mark.asyncio
async def test_audit_future_fields_cannot_change_dispatch_decision(tmp_path):
    async def replay(future):
        backend = Backend()
        async with make_scheduler(tmp_path, backend, 'queue', concurrency=1) as scheduler:
            await asyncio.gather(*(scheduler.submit(str(i), request(i),
                                   {'expected_steps': future - i, 'future_gap': future}) for i in range(4)))
        return backend.started
    assert await replay(1) == await replay(1000)


class SnapshotBackend(Backend):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.restore_count = 0
        self.fail_save = False

    async def post(self, route, payload):
        slot = int(route.split('/')[2].split('?')[0])
        path = self.directory / payload['filename']
        if route.endswith('action=save'):
            if self.fail_save:
                raise httpx.ReadTimeout('save may still be running')
            path.write_text(json.dumps(self.slots[slot]))
            return dict(n_saved=len(self.slots[slot]), n_written=path.stat().st_size)
        if route.endswith('action=restore'):
            self.restore_count += 1
            self.slots[slot] = json.loads(path.read_text())
            return dict(n_restored=len(self.slots[slot]), n_read=path.stat().st_size)
        raise AssertionError(route)


def profiled_costs(restore=.0001):
    from phase2.costs import CostModel
    return CostModel(dict(prefill={'1': dict(seconds_per_token=.05)},
        save=dict(seconds_per_token=0, intercept_s=.0001),
        restore=dict(seconds_per_token=0, intercept_s=restore)))


@pytest.mark.asyncio
async def test_checkpoint_eviction_restore_and_cleanup_end_to_end(tmp_path):
    from phase2.checkpoints import CheckpointStore
    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = CheckpointStore(backend, directory, 10 * 2**20)
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
                                     checkpoint_margin_s=0, aging_s=0)
    async with Scheduler(backend, config, Jsonl(tmp_path / 'requests.jsonl'),
                         Jsonl(tmp_path / 'decisions.jsonl'), costs=profiled_costs(),
                         checkpoints=store) as scheduler:
        a = await scheduler.submit('a', request(10))
        await scheduler.submit('b', request(20))
        assert store.get('a', 1) is not None
        await scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30]))
        await scheduler.finish('a')
        await scheduler.finish('b')
    assert backend.restore_count == 1
    assert store.stats['saved'] == 2 and store.stats['restored'] == 1
    assert store.bytes_used == 0 and not list(directory.iterdir())


@pytest.mark.asyncio
async def test_existing_snapshot_is_not_restored_when_direct_prefix_is_cheaper(tmp_path):
    from phase2.checkpoints import CheckpointStore
    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = CheckpointStore(backend, directory, 10 * 2**20)
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1, aging_s=0)
    async with Scheduler(backend, config, Jsonl(tmp_path / 'requests.jsonl'),
                         Jsonl(tmp_path / 'decisions.jsonl'), costs=profiled_costs(restore=5),
                         checkpoints=store) as scheduler:
        a = await scheduler.submit('a', request(10))
        await store.save('a', 1, 0, len(backend.slots[0]))
        await scheduler.submit('b', request(20))
        await scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30]))
    assert backend.restore_count == 0
    assert store.stats['unused_snapshots'] == 1


@pytest.mark.asyncio
async def test_checkpoint_timeout_quarantines_backend_and_preserves_unknown_files(tmp_path):
    from phase2.checkpoints import CheckpointStore
    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = CheckpointStore(backend, directory, 10 * 2**20)
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
                                     checkpoint_margin_s=0, aging_s=0)
    with pytest.raises(BackendStateUncertain):
        async with Scheduler(backend, config, Jsonl(tmp_path / 'requests.jsonl'),
                             Jsonl(tmp_path / 'decisions.jsonl'), costs=profiled_costs(),
                             checkpoints=store) as scheduler:
            await scheduler.submit('a', request(10))
            backend.fail_save = True
            with pytest.raises(BackendStateUncertain):
                await scheduler.submit('b', request(20))
    assert backend.started == [(10, 0)]
    assert store.bytes_used > 0 and list(directory.iterdir())


@pytest.mark.asyncio
async def test_queue_cancellation_and_context_exit_have_no_hanging_waiters(tmp_path):
    from phase2.scheduler import SchedulerClosed
    backend = Backend()
    scheduler = make_scheduler(tmp_path, backend, jobs_active=1)
    with pytest.raises(SchedulerClosed):
        await scheduler.submit('a', request(10))
    async with scheduler:
        await scheduler.submit('a', request(10))
        waiting = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await scheduler.finish('a')
        await scheduler.submit('c', request(30))
        await scheduler.finish('c')
        with pytest.raises(RuntimeError):
            await scheduler.submit('c', request(40))
    with pytest.raises(RuntimeError):
        await scheduler.__aenter__()
    assert [event[0] for event in backend.started] == [10, 30]


@pytest.mark.asyncio
async def test_rejects_backend_truncation_and_wrong_slot_assignment(tmp_path):
    backend = Backend()
    original = backend.complete

    async def truncated(payload):
        response = await original(payload)
        return dict(response, truncated=True)

    backend.complete = truncated
    from phase2.scheduler import WorkloadIntegrityError
    async with make_scheduler(tmp_path, backend) as scheduler:
        with pytest.raises(WorkloadIntegrityError):
            await scheduler.submit('a', request(10))
    async def wrong_slot(payload):
        response = await original(payload)
        return dict(response, id_slot=99)
    backend.complete = wrong_slot
    with pytest.raises(BackendStateUncertain):
        async with make_scheduler(tmp_path, backend) as scheduler:
            with pytest.raises(BackendStateUncertain):
                await scheduler.submit('a', request(10))


@pytest.mark.asyncio
async def test_cancelled_context_exit_drains_backend_before_returning(tmp_path):
    backend = Backend()
    backend.block = asyncio.Event()
    scheduler = make_scheduler(tmp_path, backend, concurrency=1)
    await scheduler.__aenter__()
    first = asyncio.create_task(scheduler.submit('a', request(10)))
    await backend.entered.wait()
    exiting = asyncio.create_task(scheduler.__aexit__(None, None, None))
    await asyncio.sleep(0)
    exiting.cancel()
    await asyncio.sleep(.005)
    waited = not exiting.done()
    backend.block.set()
    await first
    with pytest.raises(asyncio.CancelledError):
        await exiting
    assert waited and backend.active == 0
    assert scheduler._dispatcher.done()


@pytest.mark.asyncio
async def test_cost_learning_uses_observed_overlap_instead_of_configured_cap(tmp_path):
    from phase2.costs import CostModel

    class RecordedCosts(CostModel):
        def __init__(self):
            super().__init__()
            self.buckets = []

        def update(self, timings, concurrency=1):
            self.buckets.append(concurrency)
            super().update(timings, concurrency)

    backend, costs = Backend(delay=.01), RecordedCosts()
    async with Scheduler(backend, PolicyConfig.for_policy('affinity'),
                         Jsonl(tmp_path / 'requests.jsonl'), Jsonl(tmp_path / 'decisions.jsonl'),
                         costs=costs) as scheduler:
        await scheduler.submit('a', request(10))
        await scheduler.submit('b', request(20))
        await asyncio.gather(scheduler.submit('a', request(30)), scheduler.submit('b', request(40)))
    assert costs.buckets[:2] == [1, 1]
    assert costs.buckets[2:] == [2, 2]


@pytest.mark.asyncio
async def test_new_dispatch_waits_until_all_queued_checkpoint_io_has_drained(tmp_path):
    from phase2.checkpoints import CheckpointStore
    from phase2.costs import CostModel

    class ControlledBackend(SnapshotBackend):
        def __init__(self, directory):
            super().__init__(directory)
            self.slots.append([])
            self.saves_started = asyncio.Queue()

        async def post(self, route, payload):
            if route.endswith('action=save'):
                release = asyncio.Event()
                self.saves_started.put_nowait(release)
                await release.wait()
            return await super().post(route, payload)

    backend = ControlledBackend(tmp_path / 'cache')
    store = CheckpointStore(backend, tmp_path / 'cache', 10 * 2**20)
    costs = CostModel(dict(prefill={'3': dict(seconds_per_token=.05)},
        save=dict(seconds_per_token=0, intercept_s=.0001),
        restore=dict(seconds_per_token=0, intercept_s=.0001)))
    config = PolicyConfig.for_policy('joint-checkpoint', slots=3, concurrency=3,
                                     aging_s=0, checkpoint_margin_s=0)
    async with Scheduler(backend, config, Jsonl(tmp_path / 'requests.jsonl'),
                         Jsonl(tmp_path / 'decisions.jsonl'), costs=costs, checkpoints=store) as scheduler:
        for name, marker in [('a', 10), ('b', 20), ('c', 30)]:
            await scheduler.submit(name, request(marker))
        d, e, f = [asyncio.create_task(scheduler.submit(name, request(marker)))
                   for name, marker in [('d', 40), ('e', 50), ('f', 60)]]
        release_d = await asyncio.wait_for(backend.saves_started.get(), .5)
        release_d.set()
        release_e = await asyncio.wait_for(backend.saves_started.get(), .5)
        await d
        await scheduler.finish('d')
        g = asyncio.create_task(scheduler.submit('g', request(90)))
        await asyncio.sleep(.005)
        premature_dispatch = any(marker == 90 for marker, _ in backend.started)
        release_e.set()
        release_f = await asyncio.wait_for(backend.saves_started.get(), .5)
        release_f.set()
        await asyncio.gather(e, f, g)
    assert not premature_dispatch


@pytest.mark.asyncio
async def test_fifo_lease_owner_uses_its_reserved_slot_instead_of_waiting_on_itself(tmp_path):
    backend = Backend()
    async with make_scheduler(tmp_path, backend, 'ttl', slots=1, concurrency=1,
                              ttl_s=5, aging_s=60) as scheduler:
        a = await scheduler.submit('a', request(10))
        await scheduler.tool_start('a')
        blocked = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.sleep(.01)
        assert not blocked.done()  # TTL lease protects a's slot from the FIFO head
        await scheduler.tool_end('a')
        response = await asyncio.wait_for(
            scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30])), .5)
        assert response['timings']['cache_n'] == 3
        await scheduler.finish('a')
        await asyncio.wait_for(blocked, .5)
        await scheduler.finish('b')
    assert [event[0] for event in backend.started] == [10, 30, 20]
    decisions = [json.loads(line) for line in (tmp_path / 'decisions.jsonl').read_text().splitlines()]
    assert any(row['action'] == 'dispatch' and row.get('reason') == 'lease_owner' for row in decisions)


def test_cost_lease_must_beat_a_cheaper_checkpoint(tmp_path):
    from phase2.costs import CostModel
    from phase2.scheduler import ReadyRequest
    from phase2.state import TaskState

    class Store:
        bytes_used, budget_bytes = 0, 2**40

        def get(self, task_id, version):
            return None

    costs = CostModel(dict(prefill={'2': dict(seconds_per_token=.01)}, waits_s=[2.] * 20,
                           save=dict(seconds_per_token=0, intercept_s=.05),
                           restore=dict(seconds_per_token=0, intercept_s=.05)))

    def protected(policy, store):
        scheduler = Scheduler(None, PolicyConfig.for_policy(policy), Jsonl(tmp_path / 'r.jsonl'),
                              Jsonl(tmp_path / f'{policy}.jsonl'), costs=costs, checkpoints=store)
        scheduler.tasks['a'] = TaskState('a', 0., tool_started_s=0.)
        slot = scheduler.slots[0]
        slot.owner, slot.version, slot.tokens = 'a', 1, [1] * 1000 + [2] * 2000
        ready = ReadyRequest('b', dict(prompt=[1] * 1000 + [3], n_predict=1), {}, 0, 0., 0., None)
        return scheduler._protect(slot, ready, .5, 1)

    assert protected('joint', None)  # recompute 20 s > expected 1.5 s wait: keep
    assert not protected('joint-checkpoint', Store())  # save+restore 0.1 s: give the slot away
    rows = [json.loads(line) for line in (tmp_path / 'joint-checkpoint.jsonl').read_text().splitlines()]
    assert rows[-1]['action'] == 'no_lease' and rows[-1]['eviction_alternative'] == 'checkpoint'
    Store.bytes_used = Store.budget_bytes  # no checkpoint room: back to recompute pricing
    assert protected('joint-checkpoint', Store())


@pytest.mark.asyncio
async def test_slot_is_protected_from_the_moment_its_step_ends(tmp_path):
    """Regression (dev, 2026-11-17): a queued request was dispatched before the
    caller called tool_start(), so the owner looked idle and lost its slot unpriced."""
    backend = Backend()
    async with make_scheduler(tmp_path, backend, 'ttl', slots=1, concurrency=1,
                              ttl_s=5, aging_s=60) as scheduler:
        backend.block = asyncio.Event()
        first = asyncio.create_task(scheduler.submit('a', request(10)))
        await backend.entered.wait()
        queued = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.sleep(.005)
        backend.block.set()
        a = await first  # the dispatcher has already reconsidered the free slot
        assert not queued.done() and backend.started == [(10, 0)]
        await scheduler.tool_start('a')
        await scheduler.tool_end('a')
        response = await asyncio.wait_for(
            scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30])), .5)
        assert response['timings']['cache_n'] == 3
        await scheduler.finish('a')
        await asyncio.wait_for(queued, .5)
        await scheduler.finish('b')
    decisions = [json.loads(line) for line in (tmp_path / 'decisions.jsonl').read_text().splitlines()]
    assert [row['action'] for row in decisions if row['action'] in ('lease', 'evict')][0] == 'lease'


@pytest.mark.parametrize('settings,waiting', [
    ({}, 7), ({'checkpoint_use_probability': 0}, 0),
    ({'checkpoint_margin_s': 100}, 0),
])
def test_checkpoint_quote_cannot_bypass_the_actual_save_gate(tmp_path, settings, waiting):
    from phase2.costs import CostModel
    from phase2.scheduler import ReadyRequest
    from phase2.state import TaskState

    class Store:
        bytes_used, budget_bytes = 0, 2**40
        def get(self, task_id, version):
            return None

    costs = CostModel(dict(prefill={'2': dict(seconds_per_token=.01)}, waits_s=[2.] * 20,
                           save=dict(seconds_per_token=0, intercept_s=1),
                           restore=dict(seconds_per_token=0, intercept_s=.1)))
    scheduler = Scheduler(None, PolicyConfig.for_policy('joint-checkpoint', **settings),
                          Jsonl(tmp_path / 'requests'), Jsonl(tmp_path / 'decisions'),
                          costs=costs, checkpoints=Store())
    scheduler.tasks['a'] = TaskState('a', 0., version=1, tool_started_s=0.)
    scheduler._ready = [None] * waiting
    slot = scheduler.slots[0]
    slot.owner, slot.version, slot.tokens = 'a', 1, [1] * 1000 + [2] * 400
    ready = ReadyRequest('b', request(3, [1] * 1000 + [3]), {}, 0, 0., 0., None)
    assert scheduler._eviction_cost(slot, ready) == (4, 4, 'recompute')
    assert scheduler._protect(slot, ready, .1, max(1, waiting))


def test_existing_snapshot_is_quoted_without_reserving_another_copy(tmp_path):
    from types import SimpleNamespace
    from phase2.costs import CostModel
    from phase2.scheduler import ReadyRequest
    from phase2.state import TaskState

    class Store:
        bytes_used = budget_bytes = 2**40
        def get(self, task_id, version):
            return SimpleNamespace(tokens=3000) if task_id == 'a' and version == 1 else None

    costs = CostModel(dict(prefill={'2': dict(seconds_per_token=.01)}, waits_s=[2.] * 20,
                           restore=dict(seconds_per_token=0, intercept_s=.05)))
    scheduler = Scheduler(None, PolicyConfig.for_policy('joint-checkpoint'),
                          Jsonl(tmp_path / 'requests'), Jsonl(tmp_path / 'decisions'),
                          costs=costs, checkpoints=Store())
    owner = scheduler.tasks['a'] = TaskState('a', 0., version=1, tool_started_s=0.)
    slot = scheduler.slots[0]
    slot.owner, slot.version, slot.tokens = 'a', 1, [1] * 1000 + [2] * 2000
    ready = ReadyRequest('b', request(3, [1] * 1000 + [3]), {}, 0, 0., 0., None)
    assert scheduler._eviction_cost(slot, ready) == (.05, 20, 'checkpoint')
    owner.finish_requested = True
    assert scheduler._eviction_cost(slot, ready)[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['budget', 'save_rejected', 'owner_returns_during_save'])
async def test_checkpoint_fallback_requeues_before_overwriting_a_history_worth_keeping(tmp_path, failure):
    from phase2.checkpoints import CheckpointStore
    from phase2.costs import CostModel

    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)

    class Store(CheckpointStore):
        async def save(self, *args):
            if failure == 'owner_returns_during_save':
                save_started.set()
                await release_save.wait()
            return None

    class ChangedBudgetScheduler(Scheduler):
        changed = False
        async def _prepare_slot(self, ready, slot):
            if failure == 'budget' and slot.owner == 'a' and ready.task_id == 'b' and not self.changed:
                self.changed = True
                self.checkpoints.budget_bytes = 0
            return await super()._prepare_slot(ready, slot)

    save_started, release_save = asyncio.Event(), asyncio.Event()
    store = Store(backend, directory, 100 * 2**20)
    costs = CostModel(dict(prefill={'1': dict(seconds_per_token=.05)}, waits_s=[.05] * 20,
                           save=dict(seconds_per_token=0, intercept_s=.0001),
                           restore=dict(seconds_per_token=0, intercept_s=.0001)))
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
        queue_rule='fifo', checkpoint_margin_s=0, aging_s=60)
    async with ChangedBudgetScheduler(backend, config, Jsonl(tmp_path / 'requests.jsonl'),
            Jsonl(tmp_path / 'decisions.jsonl'), costs=costs, checkpoints=store) as scheduler:
        original = [1] + [2] * 300
        a = await scheduler.submit('a', request(2, original))
        b = asyncio.create_task(scheduler.submit('b', request(20)))
        if failure == 'owner_returns_during_save':
            await asyncio.wait_for(save_started.wait(), .5)
        else:
            await asyncio.sleep(.005)
            assert not b.done() and backend.started == [(2, 0)]
        await scheduler.tool_end('a')
        continuation = asyncio.create_task(scheduler.submit('a', request(30, original + a['tokens'] + [30])))
        if failure == 'owner_returns_during_save':
            await asyncio.sleep(0)
            release_save.set()
        response = await asyncio.wait_for(continuation, .5)
        assert response['timings']['cache_n'] == len(original) + len(a['tokens']) - 1
        await scheduler.finish('a')
        await asyncio.wait_for(b, .5)
        await scheduler.finish('b')
    assert backend.started == [(2, 0), (30, 0), (20, 0)]
    rows = [json.loads(line) for line in (tmp_path / 'requests.jsonl').read_text().splitlines()]
    assert len(rows) == 3 and all(row['status'] == 'ok' for row in rows)
    decisions = [json.loads(line) for line in (tmp_path / 'decisions.jsonl').read_text().splitlines()]
    assert any(row['action'] == 'dispatch_deferred' for row in decisions)
    # A deferred attempt never reached the model, so it is not b's first dispatch;
    # dispatch rows carry the request sequence the summary uses to drop it.
    deferred_at = max(row['time_s'] for row in decisions if row['action'] == 'dispatch_deferred')
    assert scheduler.tasks['b'].first_dispatch_s > deferred_at
    assert all('sequence' in row for row in decisions if row['action'] in ('dispatch', 'dispatch_deferred'))


@pytest.mark.asyncio
async def test_rejected_save_cannot_renew_an_expired_fallback_lease_forever(tmp_path):
    from phase2.checkpoints import CheckpointStore

    class Store(CheckpointStore):
        calls = 0
        async def save(self, *args):
            self.calls += 1
            return None

    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = Store(backend, directory, 10 * 2**20)
    costs = profiled_costs()
    costs.waits = [.01] * 20
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
                                     aging_s=60, checkpoint_margin_s=0)
    async with Scheduler(backend, config, Jsonl(tmp_path / 'requests'),
                         Jsonl(tmp_path / 'decisions'), costs=costs, checkpoints=store) as scheduler:
        await scheduler.submit('a', request(10))
        await asyncio.wait_for(scheduler.submit('b', request(20)), .5)
        assert not scheduler._running and not scheduler._outstanding
    rows = [json.loads(line) for line in (tmp_path / 'decisions').read_text().splitlines()]
    assert store.calls == 2
    assert sum(row['action'] == 'lease' for row in rows) == 1
    assert backend.started == [(10, 0), (20, 0)]


@pytest.mark.asyncio
async def test_cancelled_deferred_request_releases_its_permit_without_running_model(tmp_path):
    from phase2.checkpoints import CheckpointStore

    class Store(CheckpointStore):
        async def save(self, *args):
            return None

    class ObservedScheduler(Scheduler):
        def _decision(self, action, *args, **kwargs):
            super()._decision(action, *args, **kwargs)
            if action == 'dispatch_deferred':
                deferred.set()

    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = Store(backend, directory, 10 * 2**20)
    costs = profiled_costs()
    costs.waits = [.1] * 20
    deferred = asyncio.Event()
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
                                     aging_s=60, checkpoint_margin_s=0)
    async with ObservedScheduler(backend, config, Jsonl(tmp_path / 'requests'),
            Jsonl(tmp_path / 'decisions'), costs=costs, checkpoints=store) as scheduler:
        await scheduler.submit('a', request(10))
        b = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.wait_for(deferred.wait(), .5)
        assert not scheduler.slots[0].busy and not scheduler.tasks['b'].inflight
        assert not scheduler._running and not b.done()
        b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await b
        assert 'b' not in scheduler._outstanding
        await scheduler.finish('a')
        await asyncio.wait_for(scheduler.submit('c', request(30)), .5)
    assert backend.started == [(10, 0), (30, 0)]


@pytest.mark.asyncio
async def test_aging_breaks_retention_when_save_rejection_arrives_after_the_deadline(tmp_path):
    from phase2.checkpoints import CheckpointStore

    class Store(CheckpointStore):
        async def save(self, *args):
            saving.set()
            await release.wait()
            return None

    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = Store(backend, directory, 10 * 2**20)
    costs = profiled_costs()
    costs.waits = [.1] * 20
    clock, saving, release = [0.], asyncio.Event(), asyncio.Event()
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
                                     aging_s=1, checkpoint_margin_s=0)
    async with Scheduler(backend, config, Jsonl(tmp_path / 'requests'),
            Jsonl(tmp_path / 'decisions'), costs=costs, checkpoints=store,
            clock=lambda: clock[0]) as scheduler:
        await scheduler.submit('a', request(10))
        b = asyncio.create_task(scheduler.submit('b', request(20)))
        await asyncio.wait_for(saving.wait(), .5)
        clock[0] = 2.
        release.set()
        await asyncio.wait_for(b, .5)
    assert backend.started == [(10, 0), (20, 0)]
    rows = [json.loads(line) for line in (tmp_path / 'decisions').read_text().splitlines()]
    assert not any(row['action'] == 'dispatch_deferred' for row in rows)


@pytest.mark.parametrize('state', ['expired', 'active', 'unprofiled'])
def test_ready_owner_handoff_respects_existing_lease_and_unknown_cost(tmp_path, state):
    from phase2.costs import CostModel
    from phase2.scheduler import ReadyRequest, _DispatchDeferred
    from phase2.state import TaskState

    class Store:
        bytes_used, budget_bytes = 0, 2**40
        def get(self, *args):
            return None

    scheduler = Scheduler(None, PolicyConfig.for_policy('joint-checkpoint'),
        Jsonl(tmp_path / 'requests'), Jsonl(tmp_path / 'decisions'),
        costs=CostModel() if state == 'unprofiled' else profiled_costs(),
        checkpoints=Store(), clock=lambda: 0.)
    scheduler.tasks['a'] = TaskState('a', 0., version=1)
    slot = scheduler.slots[0]
    slot.owner, slot.version, slot.tokens = 'a', 1, [1, 10, 700]
    ready = ReadyRequest('b', request(20), {}, 0, 0., 0., None)
    scheduler._ready = [ReadyRequest('a', request(30), {}, 1, 0., 0., None)]
    expiry = {'expired': -1., 'active': 1., 'unprofiled': None}[state]
    slot.lease_until_s = expiry
    if state == 'active':
        with pytest.raises(_DispatchDeferred):
            scheduler._reconsider_eviction(ready, slot)
    else:
        scheduler._reconsider_eviction(ready, slot)
    assert slot.lease_until_s == expiry


def _priced_slot(tmp_path, name, *, owner_back, priced_before):
    """One free slot owned by 'a' (1500 private tokens), a candidate 'b', and a
    checkpoint budget that is already full, so no checkpoint can happen."""
    from phase2.costs import CostModel
    from phase2.scheduler import ReadyRequest
    from phase2.state import TaskState

    class FullStore:
        bytes_used = budget_bytes = 2**40
        def get(self, task_id, version):
            return None

    costs = CostModel(dict(prefill={'2': dict(seconds_per_token=.005)}, waits_s=[2.] * 20,
                           save=dict(seconds_per_token=1e-4), restore=dict(seconds_per_token=1e-4)))
    scheduler = Scheduler(None, PolicyConfig.for_policy(name, aging_s=80), Jsonl(tmp_path / f'{name}-r'),
                          Jsonl(tmp_path / f'{name}-d'), costs=costs,
                          checkpoints=FullStore() if name == 'joint-checkpoint' else None,
                          clock=lambda: 10.)
    scheduler.tasks['a'] = TaskState('a', 0., version=1, completed_steps=1,
                                     tool_started_s=None if owner_back else 9.)
    scheduler.tasks['b'] = TaskState('b', 0., completed_steps=1)
    for other in scheduler.slots[1:]:
        other.busy = True
    slot = scheduler.slots[0]
    slot.owner, slot.version, slot.tokens = 'a', 1, [1] * 4891 + [2] * 1500
    if priced_before:
        slot.lease_version, slot.lease_alternative = 1, 'recompute'  # an earlier no_lease
    b = ReadyRequest('b', dict(prompt=[1] * 4891 + [3]), {}, 0, 5., 5., None)
    scheduler._ready = [b] + ([ReadyRequest('a', dict(prompt=slot.tokens + [4]), {}, 1, 9., 9., None)]
                              if owner_back else [])
    return scheduler, slot, b


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_back,priced_before', [(True, False), (True, True), (False, True)])
async def test_checkpoint_policy_without_a_checkpoint_evicts_like_joint(tmp_path, owner_back, priced_before):
    # With no checkpoint possible, joint-checkpoint must make joint's decision:
    # no extra protection of a returned owner and no second pricing of a version.
    outcomes = {}
    for name in ('joint', 'joint-checkpoint'):
        scheduler, slot, b = _priced_slot(tmp_path, name, owner_back=owner_back, priced_before=priced_before)
        assert scheduler._select_slot(b, 10.) is slot
        scheduler._ready.remove(b)
        slot.busy, scheduler.tasks['b'].inflight = True, True
        await scheduler._prepare_slot(b, slot)  # would raise _DispatchDeferred if re-priced
        rows = [json.loads(line) for line in (tmp_path / f'{name}-d').read_text().splitlines()]
        outcomes[name] = [row['action'] for row in rows]
    assert outcomes['joint'] == outcomes['joint-checkpoint']
    assert outcomes['joint'][-1] == 'evict'


@pytest.mark.asyncio
async def test_completed_checkpoint_io_updates_its_cost_curve(tmp_path):
    from phase2.checkpoints import CheckpointStore
    class TimedStore(CheckpointStore):  # a loaded server: every call takes .5 s
        async def save(self, *args):
            snapshot = await super().save(*args)
            self.last_roundtrip_s = .5 if snapshot else None
            return snapshot

        async def restore(self, *args):
            response = await super().restore(*args)
            self.last_roundtrip_s = .5
            return response

    directory = tmp_path / 'cache'
    backend = SnapshotBackend(directory)
    store = TimedStore(backend, directory, 10 * 2**20)
    costs = profiled_costs()
    config = PolicyConfig.for_policy('joint-checkpoint', slots=1, concurrency=1,
                                     checkpoint_margin_s=0, aging_s=0)
    async with Scheduler(backend, config, Jsonl(tmp_path / 'requests.jsonl'),
                         Jsonl(tmp_path / 'decisions.jsonl'), costs=costs, checkpoints=store) as scheduler:
        a = await scheduler.submit('a', request(10))
        await scheduler.submit('b', request(20))
        await scheduler.submit('a', request(30, [1, 10] + a['tokens'] + [30]))
        await scheduler.finish('a')
        await scheduler.finish('b')
    assert store.stats['saved'] >= 1 and store.stats['restored'] == 1
    # The probe priced I/O at its intercept alone; observed calls raise the rate.
    assert costs.save['seconds_per_token'] > 0 and costs.restore['seconds_per_token'] > 0
    assert costs.save['intercept_s'] == .0001
