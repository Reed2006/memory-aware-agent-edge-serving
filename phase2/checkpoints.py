"""Quota-bounded slot snapshots; uncertain backend actions are never cleaned up."""

import asyncio
import json
import os
import stat
import time
import uuid
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from phase1.core import Jsonl


KV_BYTES_PER_TOKEN = 144 * 1024
STATE_HEADER_BYTES = 1024 * 1024


class CheckpointUncertain(RuntimeError):
    """The backend may still be accessing this slot or file; quarantine the slot."""

    def __init__(self, action: str, slot_id: int, filename: str, cause: BaseException | None = None):
        self.action, self.slot_id, self.filename = action, slot_id, filename
        self.cause = cause
        super().__init__(f'Checkpoint {action} outcome is uncertain for slot {slot_id}; quarantine required')


@dataclass
class Snapshot:
    task_id: str
    version: int
    filename: str
    tokens: int
    bytes: int
    used: bool = False


class _IoState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.uncertain: dict[int, CheckpointUncertain] = {}


class CheckpointStore:
    """Manage only names minted by this store in the server's real save directory.

    Quota includes valid state files and reservations for in-flight/uncertain files.
    Small JSON sidecars are metadata; API-reported state bytes are the quota unit.
    ``close`` is synchronous and should normally run after scheduler work is drained.
    """

    _io_states = weakref.WeakKeyDictionary()

    def __init__(self, backend, directory: Path, budget_bytes: int, log: Jsonl | None = None):
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError('budget_bytes must be a nonnegative integer')
        directory = Path(directory).expanduser().absolute()
        if any(part.is_symlink() for part in (directory, *directory.parents)):
            raise ValueError('Checkpoint directory and its ancestors cannot be symlinks')
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory.resolve(strict=True)
        self.backend, self.budget_bytes, self.log = backend, budget_bytes, log
        self._identity = self._directory_identity()
        self._dir_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._prefix = f'phase2-{uuid.uuid4().hex}-'
        self._snapshots: dict[str, Snapshot] = {}
        self._reservations: dict[str, int] = {}
        self._owned: set[str] = set()
        self._runtime_files: dict[str, set[str]] = {}
        self._active: set[str] = set()
        self._uncertain: set[str] = set()
        self._quarantined: CheckpointUncertain | None = None
        self._closed = False
        # Backend call time of the latest successful save/restore, for online costs.
        self.last_roundtrip_s: float | None = None
        self.stats = dict(saved=0, restored=0, quota_skips=0, errors=0, uncertain=0,
                          bytes_written=0, bytes_read=0, unused_bytes=0, unused_snapshots=0,
                          runtime_sidecar_bytes=0)

    @property
    def bytes_used(self) -> int:
        return sum(self._reservations.values())

    @property
    def snapshots(self) -> dict[str, Snapshot]:
        return dict(self._snapshots)

    def _directory_identity(self) -> tuple[int, int]:
        info = os.stat(self.directory, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('Checkpoint directory must remain a real directory')
        return info.st_dev, info.st_ino

    def _assert_directory(self) -> None:
        if self._directory_identity() != self._identity:
            raise ValueError('Checkpoint directory was replaced')

    def _io_state(self) -> _IoState:
        loop = asyncio.get_running_loop()
        state = self._io_states.get(loop)
        if state is None:
            state = self._io_states[loop] = _IoState()
        return state

    def _ensure_usable(self, state: _IoState) -> None:
        uncertain = self._quarantined or state.uncertain.get(id(self.backend))
        if uncertain:
            raise uncertain
        if self._closed:
            raise ValueError('Checkpoint store is closed')
        self._assert_directory()

    @staticmethod
    def _validate_request(task_id, version, slot_id, tokens=None) -> None:
        if not isinstance(task_id, str):
            raise ValueError('task_id must be a string')
        for name, value in [('version', version), ('slot_id', slot_id), ('tokens', tokens)]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer')

    def _create(self, name: str, data: bytes) -> None:
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self._dir_fd)
        self._owned.add(name)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)

    def _sidecar(self, filename: str, value: dict) -> None:
        target = filename + '.json'
        data = json.dumps(value, ensure_ascii=False).encode()
        if target not in self._owned:
            self._create(target, data)
            return
        temporary = target + '.' + uuid.uuid4().hex + '.tmp'
        self._create(temporary, data)
        os.replace(temporary, target, src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
        self._owned.discard(temporary)

    def _remove_files(self, filename: str) -> None:
        self._assert_directory()
        for name in list(self._owned):
            if name == filename or name.startswith(filename + '.'):
                try:
                    os.unlink(name, dir_fd=self._dir_fd)
                except FileNotFoundError:
                    pass
                self._owned.discard(name)
        self._reservations.pop(filename, None)
        self._runtime_files.pop(filename, None)

    def _discover_state_files(self, filename: str) -> tuple[int, int]:
        """Adopt runtime suffix files only after a definitive save response.

        The basename is an unguessable store-minted name. JSON sidecars belong to
        this manager; other basename suffixes belong to the runtime save group.
        Nothing is scanned or deleted after a timeout/cancellation.
        """
        self._assert_directory()
        runtime = self._runtime_files.setdefault(filename, set())
        for name in os.listdir(self._dir_fd):
            if (name.startswith(filename + '.') and name != filename + '.json'
                    and not name.startswith(filename + '.json.')):
                info = os.stat(name, dir_fd=self._dir_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('Runtime snapshot sidecar must be a regular file')
                self._owned.add(name)
                runtime.add(name)
        sizes = []
        for name in [filename, *runtime]:
            info = os.stat(name, dir_fd=self._dir_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Saved snapshot must be a regular file')
            sizes.append(info.st_size)
        return sum(sizes), sum(sizes[1:])

    def _validate_snapshot(self, snapshot: Snapshot) -> None:
        self._assert_directory()
        if self._snapshots.get(snapshot.task_id) is not snapshot or snapshot.filename not in self._owned:
            raise ValueError('Snapshot is not owned by this store')
        info = os.stat(snapshot.filename, dir_fd=self._dir_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Snapshot must remain a regular file')
        for name in self._runtime_files.get(snapshot.filename, ()):
            if not stat.S_ISREG(os.stat(name, dir_fd=self._dir_fd, follow_symlinks=False).st_mode):
                raise ValueError('Runtime snapshot sidecar must remain a regular file')

    def _record(self, action: str, **fields) -> None:
        if self.log:
            self.log.write(dict(time_s=time.perf_counter(), action=action, **fields))

    def _mark_uncertain(self, state, action, slot_id, filename, error) -> CheckpointUncertain:
        uncertain = CheckpointUncertain(action, slot_id, filename, error)
        self._quarantined = state.uncertain[id(self.backend)] = uncertain
        self._uncertain.add(filename)
        self.stats['uncertain'] += 1
        self._record(action, slot=slot_id, filename=filename, status='uncertain',
                     error_type=type(error).__name__, bytes_reserved=self._reservations.get(filename, 0))
        return uncertain

    async def save(self, task_id: str, version: int, slot_id: int, tokens: int) -> Snapshot | None:
        self._validate_request(task_id, version, slot_id, tokens)
        self.last_roundtrip_s = None
        state = self._io_state()
        self._ensure_usable(state)
        async with state.lock:
            self._ensure_usable(state)
            current = self.get(task_id, version)
            if current:
                return current
            previous = self._snapshots.get(task_id)
            if previous and previous.version > version:
                raise ValueError('Cannot save an older task context version')
            self.discard(task_id)
            reserved = tokens * KV_BYTES_PER_TOKEN + STATE_HEADER_BYTES
            if self.bytes_used + reserved > self.budget_bytes:
                self.stats['quota_skips'] += 1
                self._record('save', task_id=task_id, version=version, slot=slot_id,
                             status='quota_skip', bytes_reserved=reserved)
                return None
            filename = self._prefix + uuid.uuid4().hex + '.bin'
            self._reservations[filename] = reserved
            started = time.perf_counter()
            try:
                self._create(filename, b'')
                self._sidecar(filename, dict(task_id=task_id, version=version, tokens=tokens, status='pending'))
            except BaseException:
                self._remove_files(filename)
                raise
            self._active.add(filename)
            try:
                try:
                    response = await self.backend.post(f'/slots/{slot_id}?action=save', {'filename': filename})
                except httpx.HTTPStatusError as error:
                    self.stats['errors'] += 1
                    self._record('save', task_id=task_id, version=version, slot=slot_id,
                                 status='http_error', http_status=error.response.status_code)
                    try:
                        self._discover_state_files(filename)
                        self._remove_files(filename)
                    except (OSError, ValueError) as filesystem_error:
                        raise self._mark_uncertain(state, 'save', slot_id, filename, filesystem_error) from filesystem_error
                    return None
                except (httpx.RequestError, asyncio.CancelledError) as error:
                    raise self._mark_uncertain(state, 'save', slot_id, filename, error) from error
                except Exception as error:
                    raise self._mark_uncertain(state, 'save', slot_id, filename, error) from error
                try:
                    self._assert_directory()
                    state_bytes, sidecar_bytes = self._discover_state_files(filename)
                    reported = response.get('n_written', state_bytes)
                    saved_tokens = response.get('n_saved', tokens)
                    self._validate_request(task_id, version, slot_id, saved_tokens)
                    if saved_tokens != tokens:
                        raise ValueError('Saved token count differs from the owned context')
                    if isinstance(reported, bool) or not isinstance(reported, int) or reported < 0:
                        raise ValueError('Invalid state byte count from backend')
                    actual = max(reported, state_bytes)
                except Exception as error:
                    raise self._mark_uncertain(state, 'save', slot_id, filename, error) from error
                self.stats['bytes_written'] += actual
                self.stats['runtime_sidecar_bytes'] += sidecar_bytes
                self._reservations[filename] = actual
                if self.bytes_used > self.budget_bytes or self._closed:
                    if self.bytes_used > self.budget_bytes:
                        self.stats['quota_skips'] += 1
                    self._remove_files(filename)
                    return None
                snapshot = Snapshot(task_id, version, filename, saved_tokens, actual)
                try:
                    self._sidecar(filename, dict(asdict(snapshot), status='saved'))
                except OSError:
                    self._remove_files(filename)
                    raise
                self._snapshots[task_id] = snapshot
                self.stats['saved'] += 1
                self.last_roundtrip_s = time.perf_counter() - started
                self._record('save', **asdict(snapshot), slot=slot_id, status='ok',
                             roundtrip_s=self.last_roundtrip_s)
                return snapshot
            finally:
                self._active.discard(filename)
                self._maybe_close_descriptor()

    def get(self, task_id: str, version: int) -> Snapshot | None:
        self._validate_request(task_id, version, 0)
        snapshot = self._snapshots.get(task_id)
        if snapshot and version > snapshot.version:
            self.discard(task_id)
            return None
        return snapshot if snapshot and snapshot.version == version and snapshot.filename not in self._uncertain else None

    async def restore(self, snapshot: Snapshot, slot_id: int) -> dict:
        self._validate_request(snapshot.task_id, snapshot.version, slot_id)
        self.last_roundtrip_s = None
        state = self._io_state()
        self._ensure_usable(state)
        async with state.lock:
            self._ensure_usable(state)
            self._validate_snapshot(snapshot)
            self._active.add(snapshot.filename)
            started = time.perf_counter()
            try:
                try:
                    response = await self.backend.post(f'/slots/{slot_id}?action=restore', {'filename': snapshot.filename})
                    restored_tokens = response.get('n_restored')
                    if isinstance(restored_tokens, bool) or restored_tokens != snapshot.tokens:
                        raise ValueError('Restored token count differs from the saved context')
                except (Exception, asyncio.CancelledError) as error:
                    raise self._mark_uncertain(state, 'restore', slot_id, snapshot.filename, error) from error
                snapshot.used = True
                self.stats['restored'] += 1
                self.stats['bytes_read'] += snapshot.bytes
                self.last_roundtrip_s = time.perf_counter() - started
                self._record('restore', **asdict(snapshot), slot=slot_id, status='ok',
                             roundtrip_s=self.last_roundtrip_s)
                return response
            finally:
                self._active.discard(snapshot.filename)
                if self._closed and snapshot.filename not in self._uncertain:
                    self._discard_snapshot(snapshot)
                self._maybe_close_descriptor()

    def _discard_snapshot(self, snapshot: Snapshot) -> None:
        self._remove_files(snapshot.filename)
        self._snapshots.pop(snapshot.task_id, None)
        if not snapshot.used:
            self.stats['unused_bytes'] += snapshot.bytes
            self.stats['unused_snapshots'] += 1

    def discard(self, task_id: str) -> None:
        snapshot = self._snapshots.get(task_id)
        if snapshot and snapshot.filename not in self._uncertain and snapshot.filename not in self._active:
            self._discard_snapshot(snapshot)

    def _maybe_close_descriptor(self) -> None:
        if self._closed and not self._active and self._dir_fd is not None:
            os.close(self._dir_fd)
            self._dir_fd = None

    def close(self) -> None:
        if self._closed:
            return
        for task_id in list(self._snapshots):
            self.discard(task_id)
        self._closed = True
        self._maybe_close_descriptor()

    def __del__(self):
        descriptor = getattr(self, '_dir_fd', None)
        if descriptor is not None:
            os.close(descriptor)
