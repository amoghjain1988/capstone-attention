"""Tests for src/ablation/arms.py.

See GRACE_AGENT.md section 8, and Amogh's review on fix/unblock-ablation:
order_edges seeded the random arm from `seed` alone, so every window with
the same edge count got the identical positional order. random was not
actually random across windows. order_edges now also takes window_id and
seeds from seed AND window_id together, so the last three tests below prove
the fix directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ablation.arms import order_edges, weight_matched_set
from src.ablation.mask import build_mask


@pytest.fixture
def edges() -> pd.DataFrame:
    """Four edges with distinct attention, so ties never interfere here."""
    return pd.DataFrame({
        "src": [3, 1, 2, 4],
        "attn": [0.4, 0.1, 0.3, 0.2],
        "rank_dist": [2, 4, 1, 3],
    })


def test_morf_reverses_lerf(edges):
    morf = order_edges(edges, "morf", seed=0, window_id="w0")
    lerf = order_edges(edges, "lerf", seed=0, window_id="w0")
    assert morf == list(reversed(lerf))


def test_random_is_deterministic_for_the_same_seed_and_window(edges):
    first = order_edges(edges, "random", seed=0, window_id="w0")
    second = order_edges(edges, "random", seed=0, window_id="w0")
    assert first == second


def test_random_differs_across_two_windows_with_the_same_edges(edges):
    """The regression test for the bug itself. Before this fix, order_edges
    took no window_id and seeded from `seed` alone, so this would have been
    the same order twice. Two different windows must now get two different
    orders, even with identical edges and the identical seed."""
    in_window_a = order_edges(edges, "random", seed=0, window_id="window_a")
    in_window_b = order_edges(edges, "random", seed=0, window_id="window_b")
    assert in_window_a != in_window_b


def test_random_stays_a_permutation_of_the_same_src_ids(edges):
    """The fix must reshuffle, never drop or invent an id."""
    order = order_edges(edges, "random", seed=0, window_id="window_a")
    assert sorted(order) == sorted(edges["src"].astype(int).tolist())


def test_ties_break_by_the_lower_src_id():
    tied = pd.DataFrame({
        "src": [5, 2, 8],
        "attn": [0.5, 0.5, 0.5],  # every edge ties on attention
        "rank_dist": [1, 2, 3],
    })
    assert order_edges(tied, "morf", seed=0, window_id="w0") == [2, 5, 8]
    assert order_edges(tied, "lerf", seed=0, window_id="w0") == [2, 5, 8]


def test_single_raises(edges):
    with pytest.raises(ValueError, match="single"):
        order_edges(edges, "single", seed=0, window_id="w0")


def test_weight_matched_raises_in_order_edges(edges):
    with pytest.raises(ValueError, match="weight_matched"):
        order_edges(edges, "weight_matched", seed=0, window_id="w0")


def test_weight_matched_set_reaches_the_target_with_the_fewest_lowest_edges(edges):
    # weakest to strongest: src1(0.1), src4(0.2), src2(0.3), src3(0.4)
    chosen = weight_matched_set(edges, target_mass=0.25)
    assert chosen == [1, 4]  # one edge (0.1) is not enough; two (0.3) is


def test_weight_matched_set_unreachable_target_returns_every_edge(edges):
    with pytest.warns(UserWarning, match="exceeds"):
        chosen = weight_matched_set(edges, target_mass=100.0)
    assert set(chosen) == {1, 2, 3, 4}


def test_build_mask_sets_false_only_on_the_ego_row():
    mask = build_mask(n_agents=5, remove=[1, 3], ego_index=2)
    assert not mask[2, 1]
    assert not mask[2, 3]
    non_ego_row = 0
    assert mask[non_ego_row].all()
