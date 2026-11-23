import json
import math

import pytest

from phase2.costs import CostModel


def profile(**updates):
    value = dict(
        schema='phase2-cost-profile-v1',
        prefill={'1': {'seconds_per_token': .01, 'intercept_s': .2},
                 '2': {'seconds_per_token': .02, 'intercept_s': .3}},
        decode={'seconds_per_token': .04},
        save={'seconds_per_token': .0001, 'intercept_s': .1},
        restore={'seconds_per_token': .00005, 'intercept_s': .05},
        waits_s=[1, 2, 3, 4], ewma_alpha=.5,
    )
    value.update(updates)
    return value


def test_unknown_costs_do_not_create_optimistic_leases_or_checkpoints():
    costs = CostModel()
    assert costs.prefill_s(1000) == costs.loss_s(1000, 0) == 0
    assert costs.decode_s(10) == 0
    assert costs.save_s(1000) is None and costs.restore_s(1000) is None
    assert costs.best_lease(0, 10) == (0, 0)
    assert not costs.checkpoint_worthwhile(1000, 10)


def test_prefill_is_incremental_and_keeps_concurrency_profiles_separate():
    costs = CostModel(profile())
    assert costs.prefill_s(1000, reused=900) == pytest.approx(1.2)
    assert costs.prefill_s(1000, reused=900, concurrency=2) == pytest.approx(2.3)
    assert costs.prefill_s(1000, reused=1000) == 0
    assert costs.prefill_s(1000, concurrency=3) == 0
    assert costs.loss_s(1000, 900) == pytest.approx(1)
    assert costs.loss_s(1000, 1200) == 0
    assert costs.decode_s(10) == pytest.approx(.4)


def test_online_updates_only_use_evaluated_tokens_and_observed_timings():
    costs = CostModel(profile(prefill_min_tokens=1))
    costs.update({'prompt_n': 100, 'cache_n': 900, 'prompt_ms': 3000,
                  'predicted_n': 10, 'predicted_ms': 600}, concurrency=1)
    assert costs.loss_s(100, 0) == pytest.approx(1.9)
    assert costs.prefill_s(100, concurrency=2) == pytest.approx(2.3)
    assert costs.decode_s(10) == pytest.approx(.5)
    previous = costs.to_dict()
    costs.update({'prompt_n': 0, 'prompt_ms': 9999, 'predicted_n': 0})
    costs.update({'prompt_n': 100, 'prompt_ms': float('nan')})
    assert costs.to_dict() == previous


def test_small_prefill_samples_do_not_inflate_loss_but_decode_still_updates():
    costs = CostModel(profile())
    costs.update({'prompt_n': 19, 'prompt_ms': 4182.732,
                  'predicted_n': 10, 'predicted_ms': 600}, concurrency=2)
    assert costs.loss_s(1000, 900, concurrency=2) == pytest.approx(2)
    assert costs.decode_s(10) == pytest.approx(.5)
    costs.update({'prompt_n': 127, 'prompt_ms': 10000}, concurrency=2)
    assert costs.loss_s(1000, 900, concurrency=2) == pytest.approx(2)
    costs.update({'prompt_n': 128, 'prompt_ms': 5420}, concurrency=2)
    assert costs.loss_s(1000, 900, concurrency=2) == pytest.approx(3)


def test_prefill_learning_can_be_frozen_without_disabling_decode_or_waits(tmp_path):
    costs = CostModel(profile(online_prefill=False, prefill_min_tokens=256))
    before = costs.to_dict()['prefill']
    costs.update({'prompt_n': 10000, 'prompt_ms': 900000,
                  'predicted_n': 10, 'predicted_ms': 600}, concurrency=1)
    costs.observe_wait(8)
    assert costs.to_dict()['prefill'] == before
    assert costs.decode_s(10) == pytest.approx(.5)
    assert costs.to_dict()['waits_s'][-1] == 8
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(costs.to_dict()))
    assert CostModel.from_file(path).to_dict() == costs.to_dict()
    assert CostModel().to_dict()['prefill_min_tokens'] == 128
    assert CostModel().to_dict()['online_prefill'] is True


@pytest.mark.parametrize('bad', [
    {'prefill_min_tokens': 0}, {'prefill_min_tokens': -1},
    {'prefill_min_tokens': 1.5}, {'prefill_min_tokens': True},
    {'online_prefill': 1}, {'online_prefill': 'false'},
])
def test_prefill_learning_profile_settings_are_validated(bad):
    with pytest.raises(ValueError):
        CostModel(profile(**bad))


def test_existing_checkpoint_ignores_sunk_save_cost_but_still_checks_net_benefit():
    costs = CostModel(profile(save=None))
    assert not costs.checkpoint_worthwhile(1000, 1)
    assert costs.checkpoint_worthwhile(1000, 1, saved=True)
    assert not costs.checkpoint_worthwhile(1000, 1, saved=True, p_use=0)
    assert not costs.checkpoint_worthwhile(1000, 1, saved=True, other_ready=20)
    with pytest.raises(ValueError):
        costs.checkpoint_worthwhile(1000, 1, saved=1)


def test_conditional_wait_probability_and_tail_fallback_are_finite():
    costs = CostModel(profile())
    assert costs.return_probability(2, 1) == pytest.approx(.5)
    assert costs.return_probability(2, 2) == 1
    probability = costs.return_probability(20, 1)
    assert 0 < probability < 1
    lease, value = costs.best_lease(20, 100, max_s=2)
    assert 0 < lease <= 2 and math.isfinite(value)
    assert CostModel().return_probability(0, 10) == 0


def test_lease_has_positive_net_value_and_respects_queue_pressure_and_cap():
    costs = CostModel(profile())
    lease, value = costs.best_lease(0, 10, queue_weight=1, max_s=2)
    assert 0 < lease <= 2 and value > 0
    assert costs.best_lease(0, 1, queue_weight=100) == (0, 0)
    assert costs.best_lease(0, 10, max_s=0) == (0, 0)
    assert costs.best_lease(0, 0) == (0, 0)


def test_checkpoint_counts_save_and_restore_pause_once_for_each_affected_request():
    costs = CostModel(profile())
    assert costs.checkpoint_worthwhile(1000, 1, other_ready=0, p_use=1, margin_s=0)
    assert not costs.checkpoint_worthwhile(1000, 1, other_ready=9, p_use=1, margin_s=0)
    assert not costs.checkpoint_worthwhile(1000, 1, p_use=0)


def test_profile_roundtrip_preserves_metadata_and_bounds_wait_history(tmp_path):
    costs = CostModel(profile(metadata={'platform_id': 'mock'}, max_wait_samples=3))
    costs.observe_wait(5)
    costs.observe_wait(6)
    assert costs.to_dict()['waits_s'] == [4, 5, 6]
    path = tmp_path / 'costs.json'
    path.write_text(json.dumps(costs.to_dict()))
    restored = CostModel.from_file(path)
    assert restored.to_dict() == costs.to_dict()
    assert restored.to_dict()['metadata'] == {'platform_id': 'mock'}


@pytest.mark.parametrize('bad', [
    {'schema': 'unknown'}, {'ewma_alpha': 0}, {'ewma_alpha': 2},
    {'prefill': {'1': {'seconds_per_token': -1}}},
    {'save': {'seconds_per_token': float('nan')}},
    {'waits_s': [-1]}, {'max_wait_samples': 0},
])
def test_invalid_profiles_fail_explicitly(bad):
    with pytest.raises(ValueError):
        CostModel(bad)


@pytest.mark.parametrize('operation', [
    lambda c: c.prefill_s(-1), lambda c: c.prefill_s(1, concurrency=0),
    lambda c: c.observe_wait(-1), lambda c: c.decode_s(-1),
    lambda c: c.best_lease(0, 1, quantiles=(2,)),
    lambda c: c.checkpoint_worthwhile(1, 1, p_use=2),
])
def test_invalid_runtime_inputs_fail_explicitly(operation):
    with pytest.raises(ValueError):
        operation(CostModel())


def test_lease_can_cover_the_wait_tail_when_the_history_is_expensive():
    # Most waits are short, a tenth are long: only a horizon near the tail keeps
    # the slot for those returns, and an expensive history makes that worth it.
    waits = [1.] * 70 + [2.] * 10 + [3.] * 8 + [9.] * 12
    costs = CostModel(profile(waits_s=waits))
    lease, value = costs.best_lease(0, 200, queue_weight=1, max_s=10)
    assert lease == pytest.approx(9.) and value > 0
    short, _ = costs.best_lease(0, 200, queue_weight=1, max_s=10, quantiles=(.25, .5, .75))
    assert short < 9.
    cheap, _ = costs.best_lease(0, 1, queue_weight=1, max_s=10)
    assert cheap < 9.  # a cheap history is not worth holding the slot through the tail


def test_holding_costs_expected_idle_time_not_the_whole_lease():
    costs = CostModel(profile(waits_s=[1.] * 80 + [30.] * 20))
    # 80% return after 1 s: a 30 s lease leaves the slot idle 6.8 s on average.
    assert costs.expected_idle_s(0, 30) == pytest.approx(.8 * 1 + .2 * 30)
    assert costs.expected_idle_s(0, 1) == pytest.approx(1)
    assert costs.expected_idle_s(10, 30) == pytest.approx(20)  # only the long waits remain
    assert CostModel().expected_idle_s(0, 30) == 0
    fallback = CostModel(profile(waits_s=[1., 2.]))  # beyond the observed tail
    assert 0 < fallback.expected_idle_s(5, 3) < 3
    # An expensive history is now held through the tail; charging the full 30 s
    # (the first version) made the 1 s horizon win and lost the slow returns.
    lease, value = costs.best_lease(0, 40, queue_weight=1, max_s=60)
    assert lease == pytest.approx(30) and value == pytest.approx(2 * 40 - 6.8)


def test_eviction_also_delays_the_requests_waiting_for_a_slot():
    costs = CostModel(profile(waits_s=[2.] * 20))
    # Recomputing 3 s later blocks a slot for 3 s, which the waiting requests feel
    # as much as 2 s of idle hold: keep, however long the queue.
    for weight in (1, 5, 50):
        lease, value = costs.best_lease(0, 3, queue_weight=weight, max_s=60)
        assert lease == pytest.approx(2) and value == pytest.approx((1 + weight) * 3 - weight * 2)
    # A history cheaper to recompute than the expected idle time is given away.
    for weight in (1, 5, 50):
        assert costs.best_lease(0, 1, queue_weight=weight, max_s=60) == (0, 0)


def test_observed_checkpoint_io_recalibrates_only_profiled_curves():
    costs = CostModel(profile(save={'seconds_per_token': .0001, 'intercept_s': .1}))
    costs.observe_io('save', 1000, 1.1)  # 1 ms/token observed under load vs .1 ms probed
    assert costs.save['seconds_per_token'] == pytest.approx(.5 * .0001 + .5 * .001)
    assert costs.save['intercept_s'] == .1
    before = costs.to_dict()
    costs.observe_io('restore', 0, 5)  # nothing to learn from an empty state
    assert costs.to_dict() == before
    assert costs.to_dict()['online_io'] is True
    unprofiled = CostModel()
    unprofiled.observe_io('save', 1000, 1)
    assert unprofiled.save is None and not unprofiled.checkpoint_worthwhile(1000, 10)
    frozen = CostModel(profile(online_io=False))
    frozen.observe_io('restore', 1000, 9)
    assert frozen.restore == CostModel(profile()).restore
    for bad in [lambda c: c.observe_io('copy', 1, 1), lambda c: c.observe_io('save', -1, 1),
                lambda c: c.observe_io('save', 1, float('nan'))]:
        with pytest.raises(ValueError):
            bad(costs)
    with pytest.raises(ValueError):
        CostModel(profile(online_io='yes'))
