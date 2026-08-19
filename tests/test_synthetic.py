"""The gate. This test must pass before any real-data number counts.

See GRACE.md and CONTRACT.md section 8. If the pipeline cannot recover a
graph we planted ourselves, no number from the real ETH/UCY data means
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.data import eligibility, synthetic, windows
from src.faithfulness.index import brute_force_single
from src.faithfulness.validation import PlantedPredictor, auroc_against_truth

N_AGENTS = 8
N_FRAMES = 200
SEED = 0
N_REAL = 2


def _gate_scores() -> tuple[np.ndarray, np.ndarray]:
    """Run brute_force_single on every eligible window of the planted scene.

    Return (shifts, is_real), one entry per edge, pooled across every window.
    """
    traj, true_edges = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=SEED, n_real=N_REAL
    )
    frame = windows.make_windows(traj, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE)
    frame = eligibility.mark_eligible(frame, SEED)
    eligible = frame.loc[frame["eligible"]]
    assert len(eligible) > 0, "the planted scene built no eligible window"

    influence = synthetic.influence_matrix(n_agents=N_AGENTS, seed=SEED, n_real=N_REAL)
    model = PlantedPredictor(influence)
    draws = np.arange(config.N_SAMPLES, dtype=np.int64) + SEED

    all_shifts: list[float] = []
    all_is_real: list[bool] = []

    for _, window_row in eligible.iterrows():
        ego_id = int(window_row["ego_id"])
        window_edges = true_edges.loc[
            (true_edges["window_id"] == window_row["window_id"])
            & (true_edges["dst"] == ego_id)
        ]
        if window_edges.empty:
            continue

        single = brute_force_single(window_row, model, window_edges[["src"]], draws)
        merged = single.merge(
            window_edges[["src", "is_real"]], on="src", how="left", validate="one_to_one"
        )
        assert not merged["is_real"].isna().any(), "every edge must carry a known truth"

        all_shifts.extend(merged["shift"].tolist())
        all_is_real.extend(merged["is_real"].tolist())

    return np.array(all_shifts, dtype=np.float64), np.array(all_is_real, dtype=bool)


# ---------------------------------------------------------------------------
# The named test. See CONTRACT.md section 8.
# ---------------------------------------------------------------------------


def test_the_gate_auroc_clears_point_nine():
    """If this fails, stop. No number from the real data means anything."""
    shifts, is_real = _gate_scores()
    auroc = auroc_against_truth(shifts, is_real)
    print(
        f"\ngate AUROC: {auroc:.4f}  "
        f"(edges={len(shifts)}, real={int(is_real.sum())}, "
        f"fake={int((~is_real).sum())})"
    )
    assert auroc > 0.9


# ---------------------------------------------------------------------------
# Supporting tests for the pieces the gate depends on.
# ---------------------------------------------------------------------------


def test_a_removed_fake_edge_gives_exactly_zero_shift():
    """PlantedPredictor never reacts to an edge outside the influence list."""
    influence = synthetic.influence_matrix(n_agents=N_AGENTS, seed=SEED, n_real=N_REAL)
    model = PlantedPredictor(influence)
    off_diagonal = ~np.eye(N_AGENTS, dtype=bool)
    fake_dst, fake_src = np.argwhere(~influence & off_diagonal)[0]

    generator = np.random.default_rng(3)
    hist = generator.normal(size=(N_AGENTS, config.N_HIST, 2))
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)

    base_pred = model.predict(hist, edge_mask=None, draws=draws)
    mask = np.ones((N_AGENTS, N_AGENTS), dtype=bool)
    mask[fake_dst, fake_src] = False
    masked_pred = model.predict(hist, edge_mask=mask, draws=draws)

    assert np.array_equal(base_pred[:, fake_dst], masked_pred[:, fake_dst])


def test_a_removed_real_edge_moves_a_planted_predictor():
    """PlantedPredictor reacts when a real influencer is removed."""
    influence = synthetic.influence_matrix(n_agents=N_AGENTS, seed=SEED, n_real=N_REAL)
    model = PlantedPredictor(influence)
    real_dst, real_src = np.argwhere(influence)[0]

    generator = np.random.default_rng(3)
    hist = generator.normal(size=(N_AGENTS, config.N_HIST, 2)) * 3.0
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)

    base_pred = model.predict(hist, edge_mask=None, draws=draws)
    mask = np.ones((N_AGENTS, N_AGENTS), dtype=bool)
    mask[real_dst, real_src] = False
    masked_pred = model.predict(hist, edge_mask=mask, draws=draws)

    assert not np.array_equal(base_pred[:, real_dst], masked_pred[:, real_dst])


def test_auroc_against_truth_is_perfect_when_scores_match_truth():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    is_real = np.array([True, True, False, False])
    assert auroc_against_truth(scores, is_real) == pytest.approx(1.0)


def test_auroc_against_truth_is_zero_when_scores_are_backwards():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    is_real = np.array([True, True, False, False])
    assert auroc_against_truth(scores, is_real) == pytest.approx(0.0)


def test_auroc_against_truth_handles_ties():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    is_real = np.array([True, False, True, False])
    assert auroc_against_truth(scores, is_real) == pytest.approx(0.5)


def test_auroc_against_truth_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="must match shapes"):
        auroc_against_truth(np.zeros(3), np.zeros(4, dtype=bool))
