"""Tests for src/metrics/.

New file. Does not touch test_data.py, test_shapes.py or test_ablation.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.metrics import accuracy, calibration, collision

# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_ade_and_fde_are_zero_for_a_perfect_prediction():
    truth = np.stack([np.arange(12, dtype=np.float64), np.zeros(12)], axis=1)
    pred = np.stack([truth, truth, truth])  # K = 3 draws, all exact
    assert accuracy.ade(pred, truth) == pytest.approx(0.0)
    assert accuracy.fde(pred, truth) == pytest.approx(0.0)
    assert accuracy.min_ade(pred, truth) == pytest.approx(0.0)
    assert accuracy.min_fde(pred, truth) == pytest.approx(0.0)


def test_ade_averages_the_draws_min_ade_takes_the_best():
    truth = np.zeros((2, 2), dtype=np.float64)
    pred = np.zeros((2, 2, 2), dtype=np.float64)
    pred[0] = [[3.0, 0.0], [3.0, 0.0]]  # draw 0 is 3 m off at every step
    pred[1] = [[0.0, 0.0], [0.0, 0.0]]  # draw 1 is exact
    assert accuracy.ade(pred, truth) == pytest.approx(1.5)
    assert accuracy.min_ade(pred, truth) == pytest.approx(0.0)


def test_fde_uses_only_the_final_step():
    truth = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[[5.0, 0.0], [10.0, 0.0]]])  # off at step 1, exact at step 2
    assert accuracy.fde(pred, truth) == pytest.approx(0.0)
    assert accuracy.ade(pred, truth) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# collision
# ---------------------------------------------------------------------------


def test_close_agents_collide():
    paths = np.zeros((2, 3, 2), dtype=np.float64)
    paths[1, :, 0] = 0.1  # 0.1 m away the whole time
    assert collision.has_collision(paths, radius=0.2) is True


def test_far_agents_do_not_collide():
    paths = np.zeros((2, 3, 2), dtype=np.float64)
    paths[1, :, 0] = 5.0
    assert collision.has_collision(paths, radius=0.2) is False


def test_a_single_agent_never_collides():
    paths = np.zeros((1, 3, 2), dtype=np.float64)
    assert collision.has_collision(paths, radius=0.2) is False


def test_collision_ignores_the_self_pair():
    # Three agents at the exact same point. Without the self-pair guard,
    # every agent would "collide" with itself even alone.
    paths = np.zeros((3, 4, 2), dtype=np.float64)
    assert collision.has_collision(paths, radius=0.2) is True  # they do collide with each other
    single = paths[:1]
    assert collision.has_collision(single, radius=0.2) is False


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_coverage_and_pit_hand_case():
    # Step 0: 5 draws at x = 0..4, truth sits exactly at the centroid (x=2).
    # Step 1: the same 5 draws shifted to x = 10..14, truth sits far outside.
    pred = np.zeros((5, 2, 2), dtype=np.float64)
    pred[:, 0, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    pred[:, 1, 0] = [10.0, 11.0, 12.0, 13.0, 14.0]
    truth = np.array([[2.0, 0.0], [20.0, 0.0]])

    assert calibration.pit(pred, truth) == pytest.approx([0.2, 1.0])
    assert calibration.coverage(pred, truth, level=0.6) == pytest.approx(0.5)


def test_coverage_stays_between_zero_and_one():
    generator = np.random.default_rng(0)
    pred = generator.normal(size=(20, 12, 2))
    truth = generator.normal(size=(12, 2))
    value = calibration.coverage(pred, truth)
    assert 0.0 <= value <= 1.0


def test_pit_values_stay_in_zero_one():
    generator = np.random.default_rng(1)
    pred = generator.normal(size=(20, 12, 2))
    truth = generator.normal(size=(12, 2))
    values = calibration.pit(pred, truth)
    assert (values >= 0.0).all() and (values <= 1.0).all()


def test_calibration_module_does_not_expose_ece():
    assert not hasattr(calibration, "ece")
    assert not hasattr(calibration, "expected_calibration_error")
    assert "ECE" in calibration.__doc__


def test_calibration_module_states_the_determinism_limit():
    assert "deterministic" in calibration.__doc__.lower()


def test_coverage_is_unbiased_for_a_calibrated_ensemble_at_k20():
    # TASK 1 acceptance: a perfectly calibrated K = 20 ensemble must score
    # near its nominal level, not 0.833 (the bias of the raw-quantile
    # estimator this replaces). Build K + 1 exchangeable points per step: K
    # "draws" and one independent "truth", both from the same distribution.
    generator = np.random.default_rng(0)
    n_draws = config.N_SAMPLES
    n_steps = 4000
    pred = generator.normal(size=(n_draws, n_steps, 2))
    truth = generator.normal(size=(n_steps, 2))

    value = calibration.coverage(pred, truth, level=0.90)
    assert value == pytest.approx(0.90, abs=0.02)


def test_pit_histogram_is_flat_for_a_uniform_rank_vector():
    # TASK 2 acceptance: K = 20 draws give K + 1 = 21 possible ranks. Feed
    # every one of the 21 exact PIT values an equal number of times and every
    # bin must come back with the same count -- no sawtooth.
    n_draws = config.N_SAMPLES
    repeats = 50
    exact_values = np.arange(n_draws + 1, dtype=np.float64) / n_draws
    pit_values = np.tile(exact_values, repeats)

    counts = calibration.pit_histogram(pit_values)
    assert len(counts) == n_draws + 1
    assert (counts == repeats).all()


def test_pit_histogram_bin_count_matches_n_samples_plus_one():
    assert calibration.N_PIT_BINS == config.N_SAMPLES + 1


# ---------------------------------------------------------------------------
# calibration.collision_share
# ---------------------------------------------------------------------------


def test_collision_share_is_the_fraction_of_draws_that_collide():
    # 4 draws, 2 agents each. Draws 0 and 2 put the agents 0.1 m apart
    # (collide at radius 0.2); draws 1 and 3 keep them 5 m apart.
    paths = np.zeros((4, 2, 3, 2), dtype=np.float64)
    paths[:, 1, :, 0] = 5.0
    paths[0, 1, :, 0] = 0.1
    paths[2, 1, :, 0] = 0.1

    assert calibration.collision_share(paths, radius=0.2) == pytest.approx(0.5)


def test_collision_share_is_zero_or_one_at_the_extremes():
    no_collision = np.full((3, 2, 2, 2), 0.0)
    no_collision[:, 1, :, 0] = 5.0
    assert calibration.collision_share(no_collision, radius=0.2) == pytest.approx(0.0)

    all_collision = np.zeros((3, 2, 2, 2), dtype=np.float64)
    assert calibration.collision_share(all_collision, radius=0.2) == pytest.approx(1.0)


def test_collision_bool_threshold_matches_config():
    # The bool the predictions table carries is collision_share thresholded
    # at config.COLLISION_SHARE_THRESHOLD, not "any of K draws".
    share = 0.5
    assert (share >= config.COLLISION_SHARE_THRESHOLD) is True
    assert (0.49 >= config.COLLISION_SHARE_THRESHOLD) is False
