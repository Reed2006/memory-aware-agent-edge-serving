import asyncio
from dataclasses import replace

import httpx
import pytest

from phase2.checkpoints import CheckpointStore, CheckpointUncertain


class Backend:
    def __init__(self, directory, size=1024):
        self.directory = directory
        self.size = size
        self.calls = []
        self.error = None
        self.started = asyncio.Event()
        self.release = None
        self.active = self.peak = 0

    async def post(self, route, payload):
        self.calls.append((route, payload))
        self.active += 1
        self.peak = max(self.active, self.peak)
        self.started.set()
        try:
            if self.release:
                await self.release.wait()
            if self.error:
                raise self.error
            if 'save' in route:
                (self.directory / payload['filename']).write_bytes(b'x' * self.size)
                return {'n_saved': 10, 'n_written': self.size}
            return {'n_restored': 10, 'n_read': self.size}
        finally:
            self.active -= 1


def store(tmp_path, **kwargs):
    backend = Backend(tmp_path)
    return backend, CheckpointStore(backend, tmp_path, kwargs.pop('budget_bytes', 10_000_000), **kwargs)


@pytest.mark.asyncio
async def test_save_restore_discard_and_unused_accounting(tmp_path):
    backend, checkpoints = store(tmp_path)
    first = await checkpoints.save('../../task', 1, 0, 10)
    assert first and first.task_id == '../../task' and first.version == 1
    assert '/' not in first.filename and first.bytes == checkpoints.bytes_used == 1024
    assert checkpoints.get('../../task', 1) is first
    assert len(checkpoints.snapshots) == 1
    assert list(tmp_path.glob('*.json'))
    response = await checkpoints.restore(first, 1)
    assert response['n_restored'] == 10 and first.used
    assert backend.calls[-1][0] == '/slots/1?action=restore'
    checkpoints.discard('../../task')
    assert checkpoints.bytes_used == 0 and not list(tmp_path.iterdir())
    assert checkpoints.stats['unused_bytes'] == 0
    second = await checkpoints.save('unused', 2, 0, 10)
    checkpoints.close()
    assert second and checkpoints.stats['unused_bytes'] == second.bytes
    assert checkpoints.stats['unused_snapshots'] == 1


@pytest.mark.asyncio
async def test_quota_is_reserved_before_backend_io_and_versions_expire(tmp_path):
    backend, checkpoints = store(tmp_path, budget_bytes=1)
    assert await checkpoints.save('a', 1, 0, 10) is None
    assert not backend.calls and checkpoints.bytes_used == 0
    backend, checkpoints = store(tmp_path)
    old = await checkpoints.save('a', 1, 0, 10)
    assert checkpoints.get('a', 2) is None
    assert checkpoints.get('a', 1) is None
    assert not (tmp_path / old.filename).exists()
    new = await checkpoints.save('a', 2, 1, 10)
    assert checkpoints.get('a', 1) is None
    assert checkpoints.get('a', 2) is new
    assert await checkpoints.save('a', 2, 1, 10) is new
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_actual_bytes_over_quota_are_removed_after_definitive_response(tmp_path):
    backend, checkpoints = store(tmp_path, budget_bytes=3_000_000)
    backend.size = 3_000_001
    assert await checkpoints.save('a', 1, 0, 10) is None
    assert checkpoints.bytes_used == 0 and not list(tmp_path.iterdir())
    assert checkpoints.stats['quota_skips'] == 1


@pytest.mark.asyncio
async def test_save_token_count_must_match_the_owned_context(tmp_path):
    backend, checkpoints = store(tmp_path)
    with pytest.raises(CheckpointUncertain):
        await checkpoints.save('a', 1, 0, 11)
    assert not checkpoints.snapshots and checkpoints.bytes_used > 0
    checkpoints.close()
    assert list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_runtime_sidecars_are_counted_once_and_deleted_as_one_snapshot(tmp_path):
    backend, checkpoints = store(tmp_path)
    original = backend.post

    async def with_sidecars(route, payload):
        response = await original(route, payload)
        if 'save' in route:
            (tmp_path / (payload['filename'] + '.checkpoints')).write_bytes(b'c' * 64)
            (tmp_path / (payload['filename'] + '.draft')).write_bytes(b'd' * 32)
            response['n_written'] = 1024 + 64 + 32
        return response

    backend.post = with_sidecars
    snapshot = await checkpoints.save('a', 1, 0, 10)
    assert snapshot.bytes == checkpoints.bytes_used == 1120
    assert checkpoints.stats['runtime_sidecar_bytes'] == 96
    checkpoints.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_runtime_sidecar_symlink_is_preserved_and_quarantines_store(tmp_path):
    backend, checkpoints = store(tmp_path)
    target = tmp_path / 'foreign'
    target.write_bytes(b'keep')
    original = backend.post

    async def with_symlink(route, payload):
        response = await original(route, payload)
        (tmp_path / (payload['filename'] + '.draft')).symlink_to(target)
        return response

    backend.post = with_symlink
    with pytest.raises(CheckpointUncertain):
        await checkpoints.save('a', 1, 0, 10)
    before = sorted(tmp_path.iterdir())
    checkpoints.close()
    assert sorted(tmp_path.iterdir()) == before
    assert target.read_bytes() == b'keep'


def test_directory_with_symlink_ancestor_is_rejected_before_writing(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        CheckpointStore(Backend(real), alias / 'nested', 100)
    assert not (real / 'nested').exists()


@pytest.mark.asyncio
async def test_cancellation_before_io_does_not_quarantine_or_reserve(tmp_path):
    backend, checkpoints = store(tmp_path)
    backend.release = asyncio.Event()
    running = asyncio.create_task(checkpoints.save('a', 1, 0, 10))
    await backend.started.wait()
    waiting = asyncio.create_task(checkpoints.save('b', 1, 1, 10))
    await asyncio.sleep(.001)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert checkpoints.stats['uncertain'] == 0 and len(backend.calls) == 1
    backend.release.set()
    await running
    checkpoints.close()


@pytest.mark.asyncio
async def test_close_during_save_waits_for_definitive_result_before_deleting(tmp_path):
    backend, checkpoints = store(tmp_path)
    backend.release = asyncio.Event()
    running = asyncio.create_task(checkpoints.save('a', 1, 0, 10))
    await backend.started.wait()
    checkpoints.close()
    assert list(tmp_path.iterdir()) and checkpoints.bytes_used > 0
    backend.release.set()
    assert await running is None
    assert not list(tmp_path.iterdir()) and checkpoints.bytes_used == 0


@pytest.mark.asyncio
async def test_store_only_deletes_owned_files_and_rejects_forged_snapshot(tmp_path):
    foreign = tmp_path / 'existing.bin'
    foreign.write_bytes(b'keep')
    backend, checkpoints = store(tmp_path)
    snapshot = await checkpoints.save('a', 1, 0, 10)
    with pytest.raises(ValueError):
        await checkpoints.restore(replace(snapshot, filename='../existing.bin'), 1)
    checkpoints.close()
    assert foreign.read_bytes() == b'keep'


def test_directory_symlink_and_negative_budget_are_rejected(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        CheckpointStore(Backend(real), alias, 100)
    with pytest.raises(ValueError):
        CheckpointStore(Backend(real), real, -1)


@pytest.mark.asyncio
async def test_replaced_snapshot_symlink_never_reaches_backend(tmp_path):
    backend, checkpoints = store(tmp_path)
    snapshot = await checkpoints.save('a', 1, 0, 10)
    target = tmp_path / 'foreign'
    target.write_text('keep')
    path = tmp_path / snapshot.filename
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError):
        await checkpoints.restore(snapshot, 1)
    assert len(backend.calls) == 1
    checkpoints.close()
    assert target.read_text() == 'keep'


@pytest.mark.asyncio
async def test_global_io_is_serialized_across_stores(tmp_path):
    backend = Backend(tmp_path)
    backend.release = asyncio.Event()
    first = CheckpointStore(backend, tmp_path, 10_000_000)
    second = CheckpointStore(backend, tmp_path, 10_000_000)
    pending = asyncio.create_task(first.save('a', 1, 0, 10))
    await backend.started.wait()
    other = asyncio.create_task(second.save('b', 1, 1, 10))
    await asyncio.sleep(.01)
    assert len(backend.calls) == 1
    backend.release.set()
    await asyncio.gather(pending, other)
    assert backend.peak == 1
    first.close()
    second.close()


@pytest.mark.asyncio
async def test_definitive_save_http_failure_releases_quota(tmp_path):
    backend, checkpoints = store(tmp_path)
    request = httpx.Request('POST', 'http://test/slots/0')
    backend.error = httpx.HTTPStatusError('bad', request=request, response=httpx.Response(400, request=request))
    assert await checkpoints.save('a', 1, 0, 10) is None
    assert checkpoints.bytes_used == 0 and not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_unknown_save_keeps_reservation_and_files_even_on_close(tmp_path, cancel):
    backend, checkpoints = store(tmp_path)
    if cancel:
        backend.release = asyncio.Event()
        pending = asyncio.create_task(checkpoints.save('a', 1, 0, 10))
        await backend.started.wait()
        pending.cancel()
        operation = pending
    else:
        backend.error = httpx.ReadTimeout('timed out')
        operation = checkpoints.save('a', 1, 0, 10)
    with pytest.raises(CheckpointUncertain) as captured:
        await operation
    assert captured.value.slot_id == 0 and captured.value.action == 'save'
    assert checkpoints.bytes_used > 0 and checkpoints.stats['uncertain'] == 1
    assert not checkpoints.snapshots
    before = sorted(tmp_path.iterdir())
    assert before
    checkpoints.discard('a')
    checkpoints.close()
    assert sorted(tmp_path.iterdir()) == before
    assert checkpoints.bytes_used > 0
    with pytest.raises(CheckpointUncertain):
        await checkpoints.save('b', 1, 1, 10)


@pytest.mark.asyncio
async def test_restore_http_failure_is_uncertain_and_file_is_preserved(tmp_path):
    backend, checkpoints = store(tmp_path)
    snapshot = await checkpoints.save('a', 1, 0, 10)
    request = httpx.Request('POST', 'http://test/slots/1')
    backend.error = httpx.HTTPStatusError('bad', request=request, response=httpx.Response(500, request=request))
    with pytest.raises(CheckpointUncertain) as captured:
        await checkpoints.restore(snapshot, 1)
    assert captured.value.action == 'restore' and captured.value.slot_id == 1
    checkpoints.close()
    assert (tmp_path / snapshot.filename).is_file()
    assert checkpoints.bytes_used == snapshot.bytes


@pytest.mark.asyncio
async def test_restore_token_count_mismatch_is_not_treated_as_success(tmp_path):
    backend, checkpoints = store(tmp_path)
    snapshot = await checkpoints.save('a', 1, 0, 10)

    async def bad_restore(route, payload):
        return {'n_restored': 0}

    backend.post = bad_restore
    with pytest.raises(CheckpointUncertain):
        await checkpoints.restore(snapshot, 1)
    assert not snapshot.used and checkpoints.stats['restored'] == 0
    checkpoints.close()
    assert (tmp_path / snapshot.filename).exists()
