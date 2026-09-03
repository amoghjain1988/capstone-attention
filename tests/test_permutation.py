"""Tests for src/stats/permutation.py.

New file. It touches no other test file. Every permutation count stays small,
so the whole file runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import config
from src.stats import permutation

# ---------------------------------------------------------------------------
# A slow reference, written as a plain Python loop
# ---------------------------------------------------------------------------


def _slow_sign_flip_p(
    d: np.ndarray,
    clusters: np.ndarray,
    n_perm: int,
    seed: int,
    alternative: str = "greater",
) -> tuple[float, float]:
    """Return (stat_obs, p) of the signed-rank sign-flip test, row by row.

    The reference draws the same sign matrix as the module, then it loops in
    Python over every permutation and every row. It is slow, so the caller
    keeps the case tiny.
    """
    groups = list(np.unique(clusters))
    signs = permutation._sign_matrix(n_perm, len(groups), seed)
    ranks = scipy_stats.rankdata(np.abs(d))

    observed = 0.0
    for rank, value in zip(ranks, d):
        observed += rank * np.sign(value)

    count = 0
    for row in signs:
        total = 0.0
        for position in range(len(d)):
            group = groups.index(clusters[position])
            total += ranks[position] * np.sign(d[position]) * row[group]
        if alternative == "greater" and total >= observed:
            count += 1
        elif alternative == "less" and total <= observed:
            count += 1
        elif alternative == "two-sided" and abs(total) >= abs(observed):
            count += 1
    return float(observed), (1.0 + count) / (1.0 + n_perm)


def test_signed_rank_matches_the_slow_python_loop():
    d = np.array([0.4, -0.1, 0.9, 0.2, -0.7, 0.3])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    fast = permutation.sign_flip_test(d, clusters, n_perm=64, seed=0)
    slow_stat, slow_p = _slow_sign_flip_p(d, clusters, n_perm=64, seed=0)
    assert fast["stat_obs"] == pytest.approx(slow_stat)
    assert fast["p"] == pytest.approx(slow_p)


def test_the_slow_loop_also_matches_the_two_sided_rule():
    d = np.array([-0.5, 0.2, 0.8, -0.1])
    clusters = np.array(["a", "a", "b", "b"])
    fast = permutation.sign_flip_test(
        d, clusters, alternative="two-sided", n_perm=32, seed=1
    )
    _, slow_p = _slow_sign_flip_p(
        d, clusters, n_perm=32, seed=1, alternative="two-sided"
    )
    assert fast["p"] == pytest.approx(slow_p)


# ---------------------------------------------------------------------------
# The Pratt rule and the four statistics
# ---------------------------------------------------------------------------


def test_a_zero_difference_keeps_its_rank_and_adds_zero():
    # |d| = [0, 1, 2] -> ranks [1, 2, 3]. The zero holds rank 1 and adds 0,
    # so the statistic is 0 + 2 - 3 = -1.
    d = np.array([0.0, 1.0, -2.0])
    clusters = np.array(["a", "b", "c"])
    result = permutation.sign_flip_test(d, clusters, n_perm=32, seed=0)
    assert result["stat_obs"] == pytest.approx(-1.0)


def test_the_zero_still_occupies_a_rank_position():
    # Pratt: |d| = [0, 1, 2] -> ranks [1, 2, 3] and the statistic is
    # 0 + 2 + 3 = 5. A rule that drops the zero before the rank step would
    # rank [1, 2] and would report 3, so the two rules differ here.
    d = np.array([0.0, 1.0, 2.0])
    clusters = np.array(["a", "b", "c"])
    result = permutation.sign_flip_test(d, clusters, n_perm=32, seed=0)
    assert result["stat_obs"] == pytest.approx(5.0)
    assert scipy_stats.rankdata(np.abs(d)).tolist() == [1.0, 2.0, 3.0]


def test_the_sign_statistic_counts_the_positive_minus_the_negative():
    d = np.array([1.0, -1.0, 2.0, 0.0])
    clusters = np.array(["a", "b", "c", "d"])
    result = permutation.sign_flip_test(d, clusters, n_perm=32, seed=0, stat="sign")
    assert result["stat_obs"] == pytest.approx(1.0)


def test_the_mean_and_the_median_statistics_report_the_sample_value():
    d = np.array([1.0, 2.0, 4.0, 8.0])
    clusters = np.array(["a", "b", "c", "d"])
    mean = permutation.sign_flip_test(d, clusters, n_perm=32, seed=0, stat="mean")
    median = permutation.sign_flip_test(d, clusters, n_perm=32, seed=0, stat="median")
    assert mean["stat_obs"] == pytest.approx(float(np.mean(d)))
    assert median["stat_obs"] == pytest.approx(float(np.median(d)))


def _paired_sample(seed: int, shift: float):
    """Return paired differences with a cluster effect, plus the labels."""
    rng = np.random.default_rng(seed)
    n_groups, per_group = 20, 6
    labels = np.repeat(np.arange(n_groups), per_group)
    effect = rng.normal(scale=0.3, size=n_groups)[labels]
    return shift + effect + rng.normal(scale=0.3, size=labels.size), labels


def test_every_statistic_runs_and_agrees_on_a_clear_shift():
    d, labels = _paired_sample(0, shift=0.5)
    for name in permutation.STATS:
        result = permutation.sign_flip_test(d, labels, n_perm=400, seed=0, stat=name)
        assert result["stat"] == name
        assert 0.0 < result["p"] < 0.01
        assert result["n_clusters"] == 20
        assert result["n_perm"] == 400


def test_a_clear_positive_shift_gives_a_small_p():
    d, labels = _paired_sample(1, shift=0.6)
    assert permutation.sign_flip_p(d, labels, n_perm=400, seed=1) < 0.01


def test_a_null_gives_a_large_p_for_most_seeds():
    large = 0
    for seed in range(12):
        d, labels = _paired_sample(seed, shift=0.0)
        if permutation.sign_flip_p(d, labels, n_perm=300, seed=seed) > config.ALPHA:
            large += 1
    assert large >= 10


def test_sign_flip_p_returns_the_p_of_sign_flip_test():
    d, labels = _paired_sample(2, shift=0.2)
    assert permutation.sign_flip_p(d, labels, n_perm=200, seed=2) == pytest.approx(
        permutation.sign_flip_test(d, labels, n_perm=200, seed=2)["p"]
    )


def test_the_whole_cluster_flips_together():
    # One cluster only. Every row flips as one block, so the permutation
    # distribution holds two values and no p value can fall near 0.
    d, _ = _paired_sample(3, shift=0.6)
    one_cluster = np.zeros(d.size, dtype=int)
    per_row = np.arange(d.size)
    assert permutation.sign_flip_p(d, one_cluster, n_perm=400, seed=3) > 0.3
    assert permutation.sign_flip_p(d, per_row, n_perm=400, seed=3) < 0.01


def test_the_sign_flip_p_matches_the_exact_wilcoxon_test():
    """With one row per cluster the test IS the Wilcoxon signed-rank test.

    A sign flip of a singleton cluster flips one row, so the permutation
    distribution is the exact null distribution of the signed-rank statistic.
    The Monte Carlo p value must therefore track
    scipy.stats.wilcoxon(alternative="greater", zero_method="pratt") on data
    that holds no tie. The tolerance of 0.02 leaves room for the Monte Carlo
    error of 4,000 draws.
    """
    generator = np.random.default_rng(99)
    for trial in range(8):
        d = generator.normal(loc=0.25, size=25)
        singletons = np.arange(d.size)
        mine = permutation.sign_flip_p(d, singletons, n_perm=4000, seed=trial)
        exact = float(
            scipy_stats.wilcoxon(
                d, alternative="greater", zero_method="pratt"
            ).pvalue
        )
        assert abs(mine - exact) < 0.02, (trial, mine, exact)


def test_the_less_alternative_mirrors_the_greater_alternative():
    d, labels = _paired_sample(4, shift=-0.6)
    assert permutation.sign_flip_p(d, labels, alternative="less", n_perm=400, seed=4) < 0.01
    assert (
        permutation.sign_flip_p(d, labels, alternative="greater", n_perm=400, seed=4)
        > 0.9
    )


def test_the_two_sided_alternative_sees_both_directions():
    d, labels = _paired_sample(5, shift=-0.6)
    two_sided = permutation.sign_flip_p(
        d, labels, alternative="two-sided", n_perm=400, seed=5
    )
    assert two_sided < 0.01
    assert permutation.sign_flip_p(
        d, labels, alternative="two_sided", n_perm=400, seed=5
    ) == pytest.approx(two_sided)


def test_the_p_value_never_falls_to_zero():
    d = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 1, 2])
    result = permutation.sign_flip_test(d, labels, n_perm=50, seed=0)
    assert result["p"] >= 1.0 / 51.0


def test_sign_flip_test_returns_every_contract_key():
    d, labels = _paired_sample(6, shift=0.1)
    result = permutation.sign_flip_test(d, labels, n_perm=100, seed=6)
    assert set(result) == {
        "stat",
        "stat_obs",
        "p",
        "n_perm",
        "n_clusters",
        "alternative",
    }


def test_sign_flip_test_rejects_a_bad_statistic():
    with pytest.raises(ValueError):
        permutation.sign_flip_test(
            np.array([1.0]), np.array([0]), n_perm=10, stat="trimmed_mean"
        )


def test_sign_flip_test_rejects_a_bad_alternative():
    with pytest.raises(ValueError):
        permutation.sign_flip_test(
            np.array([1.0]), np.array([0]), n_perm=10, alternative="bigger"
        )


def test_sign_flip_test_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        permutation.sign_flip_test(np.array([1.0, 2.0]), np.array([0]), n_perm=10)


# ---------------------------------------------------------------------------
# attention_shuffle_p
# ---------------------------------------------------------------------------


def _shuffle_tables(seed: int, planted: bool):
    """Return the edges table and the single table of a toy study.

    When `planted` is true, the shift of an edge rises with its attention
    weight. When it is false, the shift is noise and attention says nothing.
    """
    rng = np.random.default_rng(seed)
    edge_rows, single_rows = [], []
    for scene in ("zara1", "eth"):
        for step in range(15):
            window_id = f"{scene}_{20 * step:06d}"
            n_edges = int(rng.integers(2, 5))
            attn = rng.random(n_edges)
            for position in range(n_edges):
                weight = float(attn[position])
                if planted:
                    shift = weight + float(rng.normal(scale=0.05))
                else:
                    shift = float(rng.normal(scale=0.5))
                edge_rows.append(
                    {"window_id": window_id, "src": position + 1, "attn": weight}
                )
                single_rows.append(
                    {
                        "window_id": window_id,
                        "edge_src": position + 1,
                        "shift": shift,
                    }
                )
    return pd.DataFrame(edge_rows), pd.DataFrame(single_rows)


def test_attention_shuffle_detects_a_planted_association():
    edges, single = _shuffle_tables(0, planted=True)
    result = permutation.attention_shuffle_p(edges, single, n_shuffle=300, seed=0)
    assert result["stat_obs"] > 0.0
    assert result["p"] < config.ALPHA
    assert result["n_windows"] == 30
    assert result["n_shuffle"] == 300


def test_attention_shuffle_gives_a_large_p_when_attention_is_noise():
    edges, single = _shuffle_tables(1, planted=False)
    result = permutation.attention_shuffle_p(edges, single, n_shuffle=300, seed=1)
    assert result["p"] > config.ALPHA


def test_attention_shuffle_holds_a_large_p_across_several_null_seeds():
    large = 0
    for seed in range(6):
        edges, single = _shuffle_tables(10 + seed, planted=False)
        result = permutation.attention_shuffle_p(
            edges, single, n_shuffle=200, seed=seed
        )
        large += int(result["p"] > config.ALPHA)
    assert large >= 4


def test_attention_shuffle_returns_every_contract_key():
    edges, single = _shuffle_tables(2, planted=True)
    result = permutation.attention_shuffle_p(edges, single, n_shuffle=100, seed=2)
    assert set(result) == {"stat_obs", "p", "n_shuffle", "n_windows"}


def test_attention_shuffle_reads_the_gap_of_a_hand_built_window():
    edges = pd.DataFrame(
        {
            "window_id": ["zara1_000000"] * 3,
            "src": [1, 2, 3],
            "attn": [0.1, 0.9, 0.5],
        }
    )
    single = pd.DataFrame(
        {
            "window_id": ["zara1_000000"] * 3,
            "edge_src": [1, 2, 3],
            "shift": [0.2, 0.7, 0.4],
        }
    )
    result = permutation.attention_shuffle_p(edges, single, n_shuffle=20, seed=0)
    # The top attention edge is src 2 with a shift of 0.7. The bottom one is
    # src 1 with a shift of 0.2. D is 0.5 and one window gives the median.
    assert result["stat_obs"] == pytest.approx(0.5)
    assert result["n_windows"] == 1


def test_attention_shuffle_drops_a_window_with_one_edge():
    edges, single = _shuffle_tables(3, planted=True)
    extra_edge = pd.DataFrame(
        {"window_id": ["zara1_000900"], "src": [1], "attn": [0.5]}
    )
    extra_single = pd.DataFrame(
        {"window_id": ["zara1_000900"], "edge_src": [1], "shift": [0.5]}
    )
    result = permutation.attention_shuffle_p(
        pd.concat([edges, extra_edge], ignore_index=True),
        pd.concat([single, extra_single], ignore_index=True),
        n_shuffle=100,
        seed=3,
    )
    assert result["n_windows"] == 30


def test_attention_shuffle_rejects_a_duplicate_single_row():
    edges, single = _shuffle_tables(4, planted=True)
    doubled = pd.concat([single, single.head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        permutation.attention_shuffle_p(edges, doubled, n_shuffle=10, seed=4)


def test_attention_shuffle_needs_the_shift_column():
    edges, single = _shuffle_tables(5, planted=True)
    with pytest.raises(KeyError):
        permutation.attention_shuffle_p(
            edges, single.drop(columns=["shift"]), n_shuffle=10, seed=5
        )


def test_attention_shuffle_is_deterministic_for_one_seed():
    edges, single = _shuffle_tables(6, planted=True)
    first = permutation.attention_shuffle_p(edges, single, n_shuffle=100, seed=9)
    second = permutation.attention_shuffle_p(edges, single, n_shuffle=100, seed=9)
    assert first == second
