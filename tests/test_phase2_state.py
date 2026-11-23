import math

import pytest

from phase2.state import PolicyConfig, SlotState, TaskPhase, TaskState


def test_policy_presets_and_hashes_preserve_ablation_controls():
    base = PolicyConfig.for_policy('affinity')
    queue = PolicyConfig.for_policy('queue')
    cache = PolicyConfig.for_policy('cache')
    joint = PolicyConfig.for_policy('joint')
    assert base.queue_rule == cache.queue_rule == 'fifo'
    assert base.cache_policy == queue.cache_policy == 'lru'
    assert cache.cache_policy == joint.cache_policy == 'cost'
    assert queue.queue_rule == joint.queue_rule == 'continuation'
    assert all(p.affinity and p.jobs_active == 0 for p in (base, queue, cache, joint))
    assert not PolicyConfig.for_policy('fifo').affinity
    assert PolicyConfig.for_policy('joint-checkpoint').checkpoint
    assert base.param_hash == PolicyConfig.from_dict(base.to_dict()).param_hash
    assert base.param_hash != PolicyConfig.for_policy('affinity', aging_s=15).param_hash


@pytest.mark.parametrize('kwargs', [
    {'concurrency': 0}, {'slots': 0}, {'ctx_per_slot': 0},
    {'concurrency': 3, 'slots': 2}, {'jobs_active': -1},
    {'aging_s': -1}, {'aging_s': math.nan}, {'ttl_s': math.inf},
    {'checkpoint_budget_bytes': -1}, {'queue_rule': 'oracle'},
    {'cache_policy': 'oracle'}, {'affinity': False, 'cache_policy': 'cost'},
    {'checkpoint': True, 'affinity': False},
    {'affinity': 'false'}, {'checkpoint': 'true'}, {'aging_s': True},
    {'checkpoint_use_probability': True},
])
def test_invalid_policy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PolicyConfig.for_policy('joint', **kwargs)


def test_unknown_policy_and_config_keys_are_rejected():
    with pytest.raises(ValueError):
        PolicyConfig.for_policy('unknown')
    with pytest.raises(ValueError):
        PolicyConfig.from_dict({'name': 'fifo', 'future_steps': 4})


def test_slot_reservation_and_context_versions_are_explicit():
    task = TaskState('a', arrival_s=1)
    slot = SlotState(0)
    assert task.phase == TaskPhase.READY and task.version == 0
    assert slot.owner is None and not slot.busy and not slot.uncertain
    slot.owner, slot.version, slot.tokens = 'a', 1, [1, 2]
    slot.lease_until_s, slot.lease_version = 5, 1
    slot.clear()
    assert slot.owner is None and slot.tokens == [] and slot.lease_until_s is None
    slot.uncertain = True
    with pytest.raises(RuntimeError):
        slot.clear()


def test_only_cost_lease_policies_get_the_longer_lease_bound():
    from phase2.state import COST_LEASE_MAX_S
    for name in ('cache', 'joint', 'joint-checkpoint'):
        assert PolicyConfig.for_policy(name).lease_max_s == COST_LEASE_MAX_S
    # Other policies keep their configs (and param hashes) so their dev runs stay valid.
    for name in ('fifo', 'fixed', 'affinity', 'affinity-fixed', 'ttl', 'queue', 'attained', 'ram-cache'):
        assert PolicyConfig.for_policy(name).lease_max_s == 10.
