"""Tests for src/stats/effects.py and src/ablation/arm_overlap.py.

New file. Does not touch test_data.py, test_shapes.py or test_ablation.py.
Both modules are small, hand-checkable statistical tools, so they share one
test file, matching the worked examples of the agent brief.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ablation import arm_overlap
from src.stats import effects

# ---------------------------------------------------------------------------
# effects.cliffs_delta
# ---------------------------------------------------------------------------


def test_cliffs_delta_of_a_sample_against_itself_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert effects.cliffs_delta(a, a) == pytest.approx(0.0)


def test_cliffs_delta_is_minus_one_when_every_pair_favours_b():
    assert effects.cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)


def test_cliffs_delta_is_one_when_every_pair_favours_a():
    assert effects.cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# effects.rank_biserial
# ---------------------------------------------------------------------------


def test_rank_biserial_is_one_when_every_difference_is_positive():
    d = np.array([1.0, 2.0, 3.0])
    assert effects.rank_biserial(d) == pytest.approx(1.0)


def test_rank_biserial_is_zero_when_symmetric_around_zero():
    d = np.array([-3.0, -1.0, 1.0, 3.0])
    assert effects.rank_biserial(d) == pytest.approx(0.0)


def test_rank_biserial_follows_the_pratt_rule_on_a_zero_difference():
    # |d| = [0, 1, 2] -> ranks [1, 2, 3]. The zero keeps its rank but
    # contributes to neither the positive nor the negative sum.
    d = np.array([0.0, 1.0, -2.0])
    assert effects.rank_biserial(d) == pytest.approx((2 - 3) / 5)


# ---------------------------------------------------------------------------
# effects.spearman and effects.partial_corr
# ---------------------------------------------------------------------------


def test_spearman_is_perfect_on_a_monotone_pair():
    a = np.arange(10)
    b = a * 2 + 1
    rho, p_value = effects.spearman(a, b)
    assert rho == pytest.approx(1.0)
    assert p_value < 0.01


def test_partial_corr_removes_a_shared_covariate():
    generator = np.random.default_rng(0)
    z = generator.normal(size=500)
    x = z + generator.normal(scale=0.01, size=500)
    y = z + generator.normal(scale=0.01, size=500)  # x, y correlate only through z

    raw_r, _ = effects.spearman(x, y)
    partial_r, _ = effects.partial_corr(x, y, z)
    assert raw_r > 0.9
    assert abs(partial_r) < 0.1


# ---------------------------------------------------------------------------
# arm_overlap.jaccard
# ---------------------------------------------------------------------------


def test_jaccard_is_one_on_identical_top_k():
    assert arm_overlap.jaccard([1, 2, 3], [1, 2, 3], k=2) == pytest.approx(1.0)


def test_jaccard_is_zero_on_disjoint_top_k():
    assert arm_overlap.jaccard([1], [2], k=1) == pytest.approx(0.0)


def test_jaccard_is_one_when_both_are_empty():
    assert arm_overlap.jaccard([], [], k=1) == pytest.approx(1.0)


def test_jaccard_truncates_k_larger_than_the_list():
    assert arm_overlap.jaccard([1, 2], [1, 2], k=50) == pytest.approx(1.0)


def test_jaccard_partial_overlap():
    # top 2 of a: {1, 2}. top 2 of b: {2, 3}. intersection 1, union 3.
    assert arm_overlap.jaccard([1, 2, 4], [2, 3, 4], k=2) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# arm_overlap.attention_vs_distance and summary_by_scene
# ---------------------------------------------------------------------------


def _toy_edges() -> pd.DataFrame:
    """Two windows, one where attention and distance agree on the top edge,
    one where they disagree."""
    rows = [
        # eth_000000: attention and distance both rank src=2 first -> overlap
        {"window_id": "eth_000000", "src": 2, "dst": 1, "module": "encoder",
         "time_agg": "mean", "rank_attn": 1, "rank_dist": 1},
        {"window_id": "eth_000000", "src": 3, "dst": 1, "module": "encoder",
         "time_agg": "mean", "rank_attn": 2, "rank_dist": 2},
        # univ_000010: attention ranks src=5 first, distance ranks src=6 first -> no overlap
        {"window_id": "univ_000010", "src": 5, "dst": 4, "module": "encoder",
         "time_agg": "mean", "rank_attn": 1, "rank_dist": 2},
        {"window_id": "univ_000010", "src": 6, "dst": 4, "module": "encoder",
         "time_agg": "mean", "rank_attn": 2, "rank_dist": 1},
    ]
    return pd.DataFrame(rows)


def test_attention_vs_distance_matches_the_hand_worked_windows():
    out = arm_overlap.attention_vs_distance(_toy_edges(), k=1)
    by_window = out.set_index("window_id")["jaccard_k"]
    assert by_window["eth_000000"] == pytest.approx(1.0)
    assert by_window["univ_000010"] == pytest.approx(0.0)
    assert list(out["scene"]) == ["eth", "univ"]


def test_summary_by_scene_averages_within_a_scene(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "TABLE_DIR", tmp_path)
    overlap = arm_overlap.attention_vs_distance(_toy_edges(), k=1)
    table = arm_overlap.summary_by_scene(overlap)
    assert set(table["scene"]) == {"eth", "univ"}
    assert (tmp_path / "arm_overlap.csv").exists()


# ---------------------------------------------------------------------------
# arm_overlap._order -- a missing or null rank must raise, never guess
# ---------------------------------------------------------------------------


def test_order_raises_key_error_when_the_rank_column_is_absent():
    block = pd.DataFrame({"src": [1, 2], "rank_attn": [1, 2]})
    with pytest.raises(KeyError):
        arm_overlap._order(block, "rank_dist")


def test_order_raises_value_error_on_a_null_rank():
    block = pd.DataFrame({"src": [1, 2, 3], "rank_dist": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError):
        arm_overlap._order(block, "rank_dist")


def test_order_still_works_on_a_complete_rank_column():
    block = pd.DataFrame({"src": [3, 1, 2], "rank_dist": [3, 1, 2]})
    assert arm_overlap._order(block, "rank_dist") == [1, 2, 3]


def test_attention_vs_distance_refuses_a_null_rank_dist_instead_of_fabricating_a_score():
    # Before this fix, sort_values placed the null last and, being a stable
    # sort, left the rest in row order -- the window then scored a
    # fabricated 1.0 or 0.0 on that row order alone, in the direction that
    # kills hypothesis 2. The fixed code must refuse, not guess.
    rows = [
        {"window_id": "eth_000000", "src": 2, "dst": 1, "module": "encoder",
         "time_agg": "mean", "rank_attn": 1, "rank_dist": np.nan},
        {"window_id": "eth_000000", "src": 3, "dst": 1, "module": "encoder",
         "time_agg": "mean", "rank_attn": 2, "rank_dist": 2.0},
    ]
    with pytest.raises(ValueError):
        arm_overlap.attention_vs_distance(pd.DataFrame(rows), k=1)
