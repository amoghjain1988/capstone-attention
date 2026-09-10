"""Tests for src/stats/clustered.py.

New file. It touches no other test file. The bootstrap sizes stay small, so
the whole file runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.stats import clustered

# ---------------------------------------------------------------------------
# The hand-built table
#
# scene zara1, config.WINDOW_LEN is 20, so t0 // 20 is the time block.
#   zara1_000000 ego 1 -> block 0
#   zara1_000000 ego 2 -> block 0   ego 1 and ego 2 share block 0
#   zara1_000020 ego 2 -> block 1   ego 2 spans block 0 and block 1
#   zara1_000060 ego 5 -> block 3   ego 5 stands alone
#   eth_000000   ego 1 -> block 0   another scene, another component
#
# The first three rows form one component. The fourth row is a second one.
# The fifth row is a third one, because the scene enters every node key.
# ---------------------------------------------------------------------------


def _hand_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [
                "zara1_000000",
                "zara1_000000",
                "zara1_000020",
                "zara1_000060",
                "eth_000000",
            ],
            "ego_id": [1, 2, 2, 5, 1],
        }
    )


def test_the_hand_table_holds_three_components():
    labels = clustered.cluster_labels(_hand_table(), "component")
    assert clustered.n_clusters(labels) == 3


def test_the_shared_block_and_the_spanned_block_join_one_component():
    labels = clustered.cluster_labels(_hand_table(), "component")
    # rows 0, 1, 2 chain through block 0 and ego 2.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_a_component_never_crosses_a_scene():
    labels = clustered.cluster_labels(_hand_table(), "component")
    assert labels[4].startswith("eth_")
    assert labels[4] not in set(labels[:4])


def test_the_component_label_carries_the_scene_and_an_index():
    labels = clustered.cluster_labels(_hand_table(), "component")
    assert list(labels) == ["zara1_c0", "zara1_c0", "zara1_c0", "zara1_c1", "eth_c0"]


def test_the_pedestrian_unit_labels_one_scene_and_one_ego():
    labels = clustered.cluster_labels(_hand_table(), "pedestrian")
    assert list(labels) == ["zara1_p1", "zara1_p2", "zara1_p2", "zara1_p5", "eth_p1"]
    assert clustered.n_clusters(labels) == 4


def test_the_time_block_unit_divides_t0_by_the_window_length():
    labels = clustered.cluster_labels(_hand_table(), "time_block")
    assert config.WINDOW_LEN == 20
    assert list(labels) == ["zara1_t0", "zara1_t0", "zara1_t1", "zara1_t3", "eth_t0"]
    assert clustered.n_clusters(labels) == 4


def test_the_scene_unit_returns_the_scene_name():
    labels = clustered.cluster_labels(_hand_table(), "scene")
    assert list(labels) == ["zara1", "zara1", "zara1", "zara1", "eth"]
    assert clustered.n_clusters(labels) == 2


def test_the_component_unit_is_the_coarsest_of_the_three_crossed_units():
    table = _hand_table()
    components = clustered.n_clusters(clustered.cluster_labels(table, "component"))
    pedestrians = clustered.n_clusters(clustered.cluster_labels(table, "pedestrian"))
    blocks = clustered.n_clusters(clustered.cluster_labels(table, "time_block"))
    assert components <= pedestrians
    assert components <= blocks


def test_component_clusters_returns_the_component_labels():
    table = _hand_table()
    assert list(clustered.component_clusters(table)) == list(
        clustered.cluster_labels(table, "component")
    )


def test_component_clusters_accepts_the_h3_table_with_a_scene_column():
    table = _hand_table()
    table["scene"] = [wid.rpartition("_")[0] for wid in table["window_id"]]
    labels = clustered.component_clusters(table)
    assert clustered.n_clusters(labels) == 3


def test_the_default_unit_is_the_frozen_cluster_unit():
    table = _hand_table()
    assert config.CLUSTER_UNIT == "component"
    assert list(clustered.cluster_labels(table)) == list(
        clustered.cluster_labels(table, "component")
    )


def test_cluster_labels_rejects_an_unknown_unit():
    with pytest.raises(ValueError):
        clustered.cluster_labels(_hand_table(), "window")


def test_cluster_labels_needs_an_ego_id_for_the_component_unit():
    table = _hand_table().drop(columns=["ego_id"])
    with pytest.raises(KeyError):
        clustered.cluster_labels(table, "component")


def test_cluster_labels_needs_a_window_id():
    table = pd.DataFrame({"ego_id": [1, 2]})
    with pytest.raises(KeyError):
        clustered.cluster_labels(table, "component")


def test_cluster_labels_of_an_empty_table_is_empty():
    empty = pd.DataFrame({"window_id": [], "ego_id": []})
    assert clustered.cluster_labels(empty, "component").size == 0
    assert clustered.n_clusters(np.array([])) == 0


# ---------------------------------------------------------------------------
# cluster_bootstrap_ci
# ---------------------------------------------------------------------------


def _clustered_sample(seed: int, location: float = 2.0):
    """Return values with a cluster effect, plus the cluster labels."""
    rng = np.random.default_rng(seed)
    n_groups, per_group = 40, 8
    labels = np.repeat(np.arange(n_groups), per_group)
    effect = rng.normal(scale=0.5, size=n_groups)[labels]
    values = location + effect + rng.normal(scale=0.5, size=labels.size)
    return values, labels


def test_cluster_bootstrap_ci_covers_the_true_median():
    values, labels = _clustered_sample(0)
    low, high = clustered.cluster_bootstrap_ci(values, labels, n_boot=400, seed=0)
    assert low < 2.0 < high


def test_cluster_bootstrap_ci_covers_the_truth_on_most_simulated_sets():
    covered = 0
    for seed in range(12):
        values, labels = _clustered_sample(seed)
        low, high = clustered.cluster_bootstrap_ci(
            values, labels, n_boot=300, seed=seed
        )
        covered += int(low < 2.0 < high)
    assert covered >= 10


def test_cluster_bootstrap_ci_returns_an_ordered_interval_for_the_mean():
    values, labels = _clustered_sample(1)
    low, high = clustered.cluster_bootstrap_ci(
        values, labels, stat=np.mean, n_boot=300, seed=1
    )
    assert low < high
    assert low < float(np.mean(values)) < high


def test_a_narrower_level_gives_a_narrower_interval():
    values, labels = _clustered_sample(2)
    wide = clustered.cluster_bootstrap_ci(values, labels, n_boot=400, seed=2)
    narrow = clustered.cluster_bootstrap_ci(
        values, labels, n_boot=400, level=0.50, seed=2
    )
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_cluster_bootstrap_ci_is_deterministic_for_one_seed():
    values, labels = _clustered_sample(3)
    first = clustered.cluster_bootstrap_ci(values, labels, n_boot=200, seed=7)
    second = clustered.cluster_bootstrap_ci(values, labels, n_boot=200, seed=7)
    assert first == second


def test_cluster_bootstrap_ci_of_one_constant_cluster_is_a_point():
    values = np.array([3.0, 3.0, 3.0, 3.0])
    labels = np.array(["a", "a", "b", "b"])
    low, high = clustered.cluster_bootstrap_ci(values, labels, n_boot=100, seed=0)
    assert low == pytest.approx(3.0)
    assert high == pytest.approx(3.0)


def test_cluster_bootstrap_ci_pools_whole_clusters():
    """One draw pools every row of a drawn cluster, never single rows.

    Cluster a holds one row and cluster b holds three. A draw of two clusters
    therefore pools 2, 4 or 6 rows, and the mean of the pool is 0, 0.75 or
    1.0, each with probability 1/4, 1/2 and 1/4. A bootstrap that drew ROWS
    would reach many other pool sizes and many other means.
    """
    values = np.array([0.0, 1.0, 1.0, 1.0])
    labels = np.array(["a", "b", "b", "b"])

    pools: list[np.ndarray] = []

    def watch(pool: np.ndarray) -> float:
        """Record one pooled sample and return its mean."""
        pools.append(np.sort(np.asarray(pool, dtype=np.float64)))
        return float(np.mean(pool))

    clustered.cluster_bootstrap_ci(values, labels, stat=watch, n_boot=2000, seed=1)

    sizes = {int(pool.size) for pool in pools}
    assert sizes == {2, 4, 6}
    assert {tuple(pool) for pool in pools} == {
        (0.0, 0.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    }
    share = np.mean([pool.size == 4 for pool in pools])
    assert 0.45 < float(share) < 0.55


def test_cluster_bootstrap_ci_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        clustered.cluster_bootstrap_ci(np.array([1.0, 2.0]), np.array(["a"]))


def test_cluster_bootstrap_ci_rejects_a_level_outside_the_unit_interval():
    values, labels = _clustered_sample(0)
    with pytest.raises(ValueError):
        clustered.cluster_bootstrap_ci(values, labels, n_boot=10, level=1.5)


def test_cluster_bootstrap_ci_of_an_empty_sample_is_not_a_number():
    low, high = clustered.cluster_bootstrap_ci(np.array([]), np.array([]))
    assert np.isnan(low)
    assert np.isnan(high)


# ---------------------------------------------------------------------------
# hodges_lehmann
# ---------------------------------------------------------------------------


def test_hodges_lehmann_on_a_hand_case():
    # d = [1, 2, 4]. The Walsh averages over i <= j are
    # 1.0, 1.5, 2.5, 2.0, 3.0, 4.0. Sorted: 1.0 1.5 2.0 2.5 3.0 4.0.
    # The median of the six is (2.0 + 2.5) / 2 = 2.25.
    assert clustered.hodges_lehmann([1.0, 2.0, 4.0]) == pytest.approx(2.25)


def test_hodges_lehmann_of_one_value_is_that_value():
    assert clustered.hodges_lehmann([5.0]) == pytest.approx(5.0)


def test_hodges_lehmann_of_a_symmetric_sample_is_zero():
    assert clustered.hodges_lehmann([-2.0, -1.0, 1.0, 2.0]) == pytest.approx(0.0)


def test_hodges_lehmann_shifts_with_the_data():
    d = np.array([1.0, 2.0, 4.0])
    assert clustered.hodges_lehmann(d + 10.0) == pytest.approx(
        clustered.hodges_lehmann(d) + 10.0
    )


def test_hodges_lehmann_of_an_empty_sample_is_not_a_number():
    assert np.isnan(clustered.hodges_lehmann([]))


# ---------------------------------------------------------------------------
# wild_cluster_bootstrap
# ---------------------------------------------------------------------------


def _regression_sample(seed: int, slope: float):
    """Return y, X and the cluster labels, with cluster-correlated errors."""
    rng = np.random.default_rng(seed)
    n_groups, per_group = 20, 8
    labels = np.repeat(np.arange(n_groups), per_group)
    n_rows = labels.size
    error = rng.normal(scale=1.0, size=n_groups)[labels] + rng.normal(
        scale=1.0, size=n_rows
    )
    x = rng.normal(scale=1.0, size=n_groups)[labels] + rng.normal(
        scale=1.0, size=n_rows
    )
    design = np.column_stack([np.ones(n_rows), x])
    return slope * x + error, design, labels


def test_wild_cluster_bootstrap_holds_the_false_positive_rate_under_the_null():
    # 30 simulated data sets with a cluster effect in both y and x, and no
    # true slope. A test that ignores the clusters rejects far too often.
    rejections = 0
    for seed in range(30):
        y, design, labels = _regression_sample(seed, slope=0.0)
        result = clustered.wild_cluster_bootstrap(
            y, design, labels, n_boot=299, seed=seed
        )
        rejections += int(result["joint_p"] < config.ALPHA)
    assert rejections / 30.0 <= 0.20


def test_wild_cluster_bootstrap_rejects_a_strong_effect():
    y, design, labels = _regression_sample(0, slope=2.0)
    result = clustered.wild_cluster_bootstrap(y, design, labels, n_boot=299, seed=0)
    assert result["joint_p"] < config.ALPHA
    assert result["beta"][1] == pytest.approx(2.0, abs=0.5)


def test_wild_cluster_bootstrap_returns_p_values_inside_the_unit_interval():
    y, design, labels = _regression_sample(4, slope=0.5)
    result = clustered.wild_cluster_bootstrap(y, design, labels, n_boot=200, seed=4)
    assert 0.0 <= result["joint_p"] <= 1.0
    assert result["p_values"].shape == (design.shape[1],)
    assert np.all(result["p_values"] >= 0.0)
    assert np.all(result["p_values"] <= 1.0)


def test_wild_cluster_bootstrap_returns_every_contract_key():
    y, design, labels = _regression_sample(5, slope=0.0)
    result = clustered.wild_cluster_bootstrap(y, design, labels, n_boot=100, seed=5)
    assert set(result) == {
        "beta",
        "se_cluster",
        "wald_obs",
        "joint_p",
        "p_values",
        "n_clusters",
        "n_boot",
    }
    assert result["n_clusters"] == 20
    assert result["n_boot"] == 100
    assert result["se_cluster"].shape == (design.shape[1],)
    assert np.all(result["se_cluster"] > 0.0)


def test_the_default_restriction_skips_the_intercept():
    # A large intercept and no slope. The default null covers the slope only,
    # so the joint p value stays large.
    y, design, labels = _regression_sample(6, slope=0.0)
    result = clustered.wild_cluster_bootstrap(
        y + 50.0, design, labels, n_boot=200, seed=6
    )
    assert result["beta"][0] == pytest.approx(50.0, abs=1.0)
    assert result["joint_p"] > config.ALPHA


def test_a_single_row_restriction_matches_the_squared_t_statistic():
    y, design, labels = _regression_sample(7, slope=1.0)
    rule = np.array([[0.0, 1.0]])
    result = clustered.wild_cluster_bootstrap(
        y, design, labels, n_boot=100, seed=7, restriction=rule
    )
    expected = (result["beta"][1] / result["se_cluster"][1]) ** 2
    assert result["wald_obs"] == pytest.approx(expected, rel=1e-8)


def test_the_joint_p_equals_the_slope_p_when_the_model_holds_one_slope():
    y, design, labels = _regression_sample(8, slope=0.8)
    result = clustered.wild_cluster_bootstrap(y, design, labels, n_boot=200, seed=8)
    assert result["joint_p"] == pytest.approx(result["p_values"][1])


def test_wild_cluster_bootstrap_is_deterministic_for_one_seed():
    y, design, labels = _regression_sample(9, slope=0.3)
    first = clustered.wild_cluster_bootstrap(y, design, labels, n_boot=100, seed=3)
    second = clustered.wild_cluster_bootstrap(y, design, labels, n_boot=100, seed=3)
    assert first["joint_p"] == second["joint_p"]
    assert np.allclose(first["p_values"], second["p_values"])


def test_wild_cluster_bootstrap_rejects_one_cluster():
    y, design, _ = _regression_sample(10, slope=0.0)
    with pytest.raises(ValueError):
        clustered.wild_cluster_bootstrap(
            y, design, np.zeros(y.size, dtype=int), n_boot=10
        )


def test_wild_cluster_bootstrap_rejects_a_length_mismatch():
    y, design, labels = _regression_sample(11, slope=0.0)
    with pytest.raises(ValueError):
        clustered.wild_cluster_bootstrap(y[:-1], design, labels, n_boot=10)


def test_wild_cluster_bootstrap_rejects_a_restriction_of_the_wrong_width():
    y, design, labels = _regression_sample(12, slope=0.0)
    with pytest.raises(ValueError):
        clustered.wild_cluster_bootstrap(
            y, design, labels, n_boot=10, restriction=np.eye(3)
        )


def test_wild_cluster_bootstrap_accepts_the_component_labels_of_a_table():
    # The end to end path of H3: build the labels, then fit.
    rng = np.random.default_rng(0)
    windows = [f"zara1_{20 * k:06d}" for k in range(30)]
    table = pd.DataFrame({"window_id": windows, "ego_id": rng.integers(1, 6, size=30)})
    labels = clustered.component_clusters(table)
    x = rng.normal(size=30)
    design = np.column_stack([np.ones(30), x])
    result = clustered.wild_cluster_bootstrap(
        rng.normal(size=30), design, labels, n_boot=100, seed=0
    )
    assert result["n_clusters"] == clustered.n_clusters(labels)
    assert 0.0 <= result["joint_p"] <= 1.0
