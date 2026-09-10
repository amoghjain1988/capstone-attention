"""Tests for src/hypotheses/choose_test.py.

Each test builds a sample with a known shape and checks the rule that the
sample must trigger. See docs/FINISH_PLAN.md section 3.3.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.hypotheses.choose_test import MIN_NONZERO, choose, pseudomedian

# The exact key set that CONTRACT.md section 5.10 and docs/FINISH_PLAN.md
# section 3.3 demand of shape_stats.
EXPECTED_SHAPE_KEYS = {
    "n",
    "n_zero",
    "share_positive",
    "mean",
    "median",
    "pseudomedian",
    "skew",
    "skew_ci_low",
    "skew_ci_high",
    "kurtosis",
    "shapiro_w",
    "shapiro_p",
    "symmetric",
}


def _normal(n: int = 200, loc: float = 0.05, seed: int = 11) -> np.ndarray:
    """Return a symmetric sample: a normal shifted to a positive location."""
    return np.random.default_rng(seed).normal(loc=loc, scale=0.2, size=n)


def _right_skew(n: int = 300, seed: int = 12) -> np.ndarray:
    """Return a right-skewed sample: an exponential at a positive location."""
    return np.random.default_rng(seed).exponential(scale=0.3, size=n) + 0.1


def _left_skew(n: int = 300, seed: int = 13) -> np.ndarray:
    """Return a heavy left skew: the mirror of the exponential sample."""
    return -np.random.default_rng(seed).exponential(scale=0.5, size=n) - 0.1


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_symmetric_sample_gives_the_wilcoxon_test():
    """A normal D is symmetric, so the pseudomedian is the honest target."""
    result = choose(_normal())

    assert result["test"] == "wilcoxon_signed_rank"
    assert result["shape_stats"]["symmetric"] is True
    stats = result["shape_stats"]
    assert stats["skew_ci_low"] <= 0.0 <= stats["skew_ci_high"]


def test_a_right_skewed_sample_gives_the_sign_test():
    """An exponential D is skewed, so only the median survives."""
    result = choose(_right_skew())

    assert result["test"] == "sign_test"
    assert result["shape_stats"]["symmetric"] is False
    stats = result["shape_stats"]
    assert stats["skew"] > config.SYMMETRY_SKEW_TOL
    assert stats["skew_ci_low"] > 0.0


def test_a_left_skewed_sample_gives_the_sign_test():
    """A heavy left skew fails the same rule, in the other direction."""
    result = choose(_left_skew())

    assert result["test"] == "sign_test"
    assert result["shape_stats"]["symmetric"] is False
    stats = result["shape_stats"]
    assert stats["skew"] < -config.SYMMETRY_SKEW_TOL
    assert stats["skew_ci_high"] < 0.0


def test_a_small_skew_inside_the_tolerance_gives_the_wilcoxon_test():
    """The tolerance is the second gate.

    A large sample gives a tight interval that excludes 0 at a skewness far too
    small to move the pseudomedian away from the median. A mild lognormal holds
    that shape.
    """
    d = np.random.default_rng(14).lognormal(mean=0.0, sigma=0.1, size=1000) - 1.0
    result = choose(d)
    stats = result["shape_stats"]

    assert stats["skew_ci_low"] > 0.0  # the interval excludes 0
    assert abs(stats["skew"]) < config.SYMMETRY_SKEW_TOL  # the tolerance saves it
    assert result["test"] == "wilcoxon_signed_rank"
    assert "below the tolerance" in result["why"]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_result_carries_the_four_top_level_keys():
    result = choose(_normal())
    assert set(result) == {"test", "why", "sided", "shape_stats"}
    assert result["sided"] == config.SIDED


def test_the_shape_stats_keys_match_the_specification():
    result = choose(_normal())
    assert set(result["shape_stats"]) == EXPECTED_SHAPE_KEYS


def test_n_zero_counts_the_zero_differences():
    """A zero difference is real data. It counts, and it never sets the shape."""
    d = np.concatenate([_normal(n=50), np.zeros(7)])
    stats = choose(d)["shape_stats"]

    assert stats["n"] == 57
    assert stats["n_zero"] == 7
    # share_positive is the share among the NON-ZERO rows, per section 3.4.
    assert stats["share_positive"] == pytest.approx((d > 0.0).sum() / 50)


def test_the_location_values_run_on_the_whole_sample():
    d = np.array([-1.0, 0.0, 0.0, 1.0, 3.0])
    stats = choose(d)["shape_stats"]

    assert stats["mean"] == pytest.approx(d.mean())
    assert stats["median"] == pytest.approx(np.median(d))
    assert stats["pseudomedian"] == pytest.approx(pseudomedian(d))


def test_the_pseudomedian_is_the_median_of_the_walsh_averages():
    d = np.array([-1.0, 0.0, 2.0, 5.0])
    walsh = [
        (d[i] + d[j]) / 2.0 for i in range(d.size) for j in range(d.size) if i <= j
    ]
    assert pseudomedian(d) == pytest.approx(float(np.median(walsh)))


def test_why_is_one_present_tense_sentence():
    for sample in (_normal(), _right_skew(), _left_skew()):
        why = choose(sample)["why"]
        assert why.endswith(".")
        # One sentence only: no full stop stands in the middle of the string.
        assert ". " not in why
        assert "ing " not in why
        assert len(why.split()) <= 25


def test_why_names_the_rule_that_fired():
    symmetric = choose(_normal())["why"]
    skewed = choose(_right_skew())["why"]

    assert "contains 0" in symmetric
    assert "Wilcoxon signed-rank" in symmetric
    assert "excludes 0" in skewed
    assert "sign test" in skewed


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


def test_too_few_non_zero_rows_raise():
    """Two non-zero rows carry no shape, so the function refuses the sample."""
    d = np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    assert MIN_NONZERO == 3
    with pytest.raises(ValueError, match="non-zero"):
        choose(d)


def test_an_all_zero_sample_raises():
    with pytest.raises(ValueError):
        choose(np.zeros(20))


def test_the_result_is_deterministic_across_two_calls():
    """The seed is frozen, so the bootstrap gives the same interval twice."""
    d = _right_skew()
    first = choose(d)
    second = choose(d)

    assert first["test"] == second["test"]
    assert first["why"] == second["why"]
    assert first["shape_stats"] == second["shape_stats"]


def test_a_constant_non_zero_sample_reads_as_symmetric():
    """A point mass is symmetric about its own point. The skewness is 0.0."""
    stats = choose(np.full(10, 0.25))["shape_stats"]

    assert stats["skew"] == 0.0
    assert stats["symmetric"] is True
    assert np.isnan(stats["shapiro_w"])
