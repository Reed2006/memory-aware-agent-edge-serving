"""Validated policy configuration and independent task / residency state."""

from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
import hashlib
import json
import math
from typing import Any


COST_LEASE_MAX_S = 60.0


class TaskPhase(StrEnum):
    READY = 'ready'
    RUNNING = 'running'
    EXTERNAL_WAIT = 'external_wait'
    FINISHED = 'finished'
    FAILED = 'failed'


@dataclass(frozen=True)
class PolicyConfig:
    name: str = 'joint'
    concurrency: int = 2
    slots: int = 2
    ctx_per_slot: int = 12288
    jobs_active: int = 0
    queue_rule: str = 'continuation'
    cache_policy: str = 'cost'
    affinity: bool = True
    aging_s: float = 30.0
    ttl_s: float = 2.0
    lease_max_s: float = 10.0
    checkpoint: bool = False
    checkpoint_budget_bytes: int = 4 * 2**30
    checkpoint_margin_s: float = .05
    checkpoint_use_probability: float = .8

    def __post_init__(self) -> None:
        for key in ('affinity', 'checkpoint'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'{key} must be a boolean')
        for key in ('concurrency', 'slots', 'ctx_per_slot'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.concurrency > self.slots:
            raise ValueError('concurrency must not exceed backend slots')
        for key in ('jobs_active', 'checkpoint_budget_bytes'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        for key in ('aging_s', 'ttl_s', 'lease_max_s', 'checkpoint_margin_s'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if (isinstance(self.checkpoint_use_probability, bool) or
                not isinstance(self.checkpoint_use_probability, (int, float)) or
                not math.isfinite(self.checkpoint_use_probability) or
                not 0 <= self.checkpoint_use_probability <= 1):
            raise ValueError('checkpoint_use_probability must be in [0, 1]')
        if self.queue_rule not in {'fifo', 'continuation', 'attained'}:
            raise ValueError('unknown queue rule')
        if self.cache_policy not in {'lru', 'ttl', 'cost'}:
            raise ValueError('unknown cache policy')
        if not self.affinity and (self.cache_policy != 'lru' or self.checkpoint):
            raise ValueError('retention and checkpoints require explicit slot ownership')

    @classmethod
    def for_policy(cls, name: str, **overrides: Any) -> 'PolicyConfig':
        presets = {
            'fifo': dict(affinity=False, queue_rule='fifo', cache_policy='lru'),
            'fixed': dict(affinity=False, queue_rule='fifo', cache_policy='lru'),
            'ram-cache': dict(affinity=False, queue_rule='fifo', cache_policy='lru'),
            'affinity': dict(queue_rule='fifo', cache_policy='lru'),
            'affinity-fixed': dict(queue_rule='fifo', cache_policy='lru'),
            'ttl': dict(queue_rule='fifo', cache_policy='ttl'),
            'queue': dict(queue_rule='continuation', cache_policy='lru'),
            'attained': dict(queue_rule='attained', cache_policy='lru'),
            # Cost leases may run past 10 s (dev, 2026-11-18): their value now charges
            # only expected idle time, so the cap is a safety bound, not the policy.
            # Dev waits (lognormal, median 2 s) rarely pass 10 s, but the long-wait
            # sensitivity condition (fixed 15 s) does, and a 10 s cap would make the
            # lease expire just before every return there.
            'cache': dict(queue_rule='fifo', cache_policy='cost', lease_max_s=COST_LEASE_MAX_S),
            'joint': dict(queue_rule='continuation', cache_policy='cost', lease_max_s=COST_LEASE_MAX_S),
            'joint-checkpoint': dict(queue_rule='continuation', cache_policy='cost', checkpoint=True,
                                     lease_max_s=COST_LEASE_MAX_S),
        }
        if name not in presets:
            raise ValueError(f'unknown policy: {name}')
        return cls(name=name, **(presets[name] | overrides))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PolicyConfig':
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown policy keys: {sorted(unknown)}')
        return cls.for_policy(**data) if 'name' in data else cls(**data)

    @property
    def param_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class TaskState:
    task_id: str
    arrival_s: float
    phase: TaskPhase = TaskPhase.READY
    version: int = 0
    completed_steps: int = 0
    cumulative_service_s: float = 0.0
    admitted_s: float | None = None
    first_dispatch_s: float | None = None
    tool_started_s: float | None = None
    finish_requested: bool = False
    finish_status: str = 'ok'
    inflight: bool = False


@dataclass
class SlotState:
    slot_id: int
    owner: str | None = None
    version: int = 0
    tokens: list[int] = field(default_factory=list)
    busy: bool = False
    uncertain: bool = False
    last_used_s: float = 0.0
    lease_until_s: float | None = None
    lease_version: int | None = None
    # The give-away alternative ('recompute' or 'checkpoint') that version
    # lease_version was priced against; meaningful only for that version.
    lease_alternative: str | None = None

    def clear(self) -> None:
        if self.busy or self.uncertain:
            raise RuntimeError('cannot clear a busy or uncertain slot')
        self.owner = None
        self.version = 0
        self.tokens = []
        self.lease_until_s = None
        self.lease_version = None
        self.lease_alternative = None
