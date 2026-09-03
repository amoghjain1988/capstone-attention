"""The gate. This test must pass before any real-data number counts.

See GRACE.md and CONTRACT.md section 8. If the pipeline cannot recover a
graph we planted ourselves, no number from the real ETH/UCY data means
anything.

Amogh's review of feat/ablation-faithfulness found that this file's gate
scored the ABLATION SHIFT against the planted truth, not the attention
weight. PlantedPredictor never reacts to a fake edge, so every fake edge
gave a shift of exactly 0 and every real edge gave a shift above 0. That
separates perfectly by construction, whatever src/attention/aggregate.py and
src/attention/rank.py do, or even if the gate never called them at all. The
old AUROC of 1.0000 proved the ablation MECHANISM obeys the planted graph.
It never once exercised the attention-extraction-and-ranking pipeline that
the real study depends on, and it still passed when the ranking direction
was reversed.

The gate below now scores PlantedPredictor's ATTENTION weight, collapsed and
ranked through the same two functions the real pipeline calls on
AgentFormer's own attention. See
src/faithfulness/validation.py::PlantedPredictor.attention for why that
signal is built to be strong but imperfect, on purpose: a signal that
separated real from fake edges with certainty would again prove nothing.

The old shift-based check still has a job. It stays below, renamed
test_the_planted_predictor_obeys_the_graph, as a fixture check that
PlantedPredictor's own forecast mechanism is sound. That is a real thing to
verify. It is just not, by itself, the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.attention import aggregate, rank
from src.data import eligibility, synthetic, windows, schema
from src.faithfulness.index import brute_force_single
from src.faithfulness.validation import PlantedPredictor, auroc_against_truth

N_AGENTS = 8
N_FRAMES = 200
SEED = 0
N_REAL = 2

# Amogh's second review, Check 3: one seed is not evidence that the gate
# reliably works, only that it works once. These are 5 independently planted
# scenes -- a different influence graph and a different simulated walk each
# time -- so the two tests near the bottom of this file can report every
# AUROC across several seeds, not just SEED.
GATE_SEEDS = (0, 1, 2, 3, 4)


def _eligible_windows(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (eligible windows, true_edges) of one planted scene.

    `seed` picks which planted scene: it feeds both the influence graph
    (`synthetic.make_scene`) and the window eligibility draw
    (`eligibility.mark_eligible`), so two different seeds give two genuinely
    different scenes, not the same scene read twice.
    """
    traj, true_edges = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=seed, n_real=N_REAL
    )
    frame = windows.make_windows(traj, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE)
    frame = eligibility.mark_eligible(frame, seed)
    eligible = frame.loc[frame["eligible"]]
    assert len(eligible) > 0, "the planted scene built no eligible window"
    return eligible, true_edges


# ---------------------------------------------------------------------------
# The named test. See CONTRACT.md section 8.
# ---------------------------------------------------------------------------


def _attention_scores(
    seed: int = SEED, rank_fn=rank.rank_edges
) -> tuple[np.ndarray, np.ndarray]:
    """Run PlantedPredictor's attention through collapse() and rank_fn on
    every eligible window of one planted scene.

    Return (attn, is_real), one entry per edge, pooled across every window.
    `seed` picks the planted scene, see _eligible_windows. `rank_fn` defaults
    to the real rank_edges. Passing a broken stand-in, the way
    test_a_reversed_ranking_fails_the_gate does, proves this function's
    result actually depends on ranking the ids correctly.
    """
    eligible, true_edges = _eligible_windows(seed)
    influence = synthetic.influence_matrix(n_agents=N_AGENTS, seed=seed, n_real=N_REAL)
    model = PlantedPredictor(influence)

    all_attn: list[float] = []
    all_is_real: list[bool] = []

    for _, window_row in eligible.iterrows():
        ego_id = int(window_row["ego_id"])
        window_edges = true_edges.loc[
            (true_edges["window_id"] == window_row["window_id"])
            & (true_edges["dst"] == ego_id)
        ]
        if window_edges.empty:
            continue

        hist = schema.unpack_hist(window_row, config.N_HIST)
        agent_order = [int(a) for a in window_row["agent_order"]]
        ego_index = schema.ego_index(window_row)

        raw = model.attention(hist)[config.PRIMARY_MODULE]
        graph = aggregate.collapse(raw, n_agents=N_AGENTS, time_agg=config.PRIMARY_TIME_AGG)
        ranked = rank_fn(graph, ego_index=ego_index, agent_order=agent_order)

        merged = ranked.merge(
            window_edges[["src", "is_real"]], on="src", how="left", validate="one_to_one"
        )
        assert not merged["is_real"].isna().any(), "every edge must carry a known truth"

        all_attn.extend(merged["attn"].tolist())
        all_is_real.extend(merged["is_real"].tolist())

    return np.array(all_attn, dtype=np.float64), np.array(all_is_real, dtype=bool)


def _reversed_rank_edges(
    attn: np.ndarray, ego_index: int, agent_order: list[int]
) -> pd.DataFrame:
    """A deliberately broken rank_edges. Ranks correctly, then flips the
    sign of every score, so the strongest edge now reads as the weakest.

    This is the negative control for the gate itself: if the real
    rank_edges ever shipped with its ranking direction silently reversed, or
    if collapse() ever silently swapped which axis is the ego's own row,
    this is the shape of bug that a reversed ranking here stands in for."""
    ranked = rank.rank_edges(attn, ego_index, agent_order)
    ranked = ranked.copy()
    ranked["attn"] = -ranked["attn"]
    return ranked


def test_the_gate_auroc_clears_point_nine():
    """If this fails, stop. No number from the real data means anything."""
    attn, is_real = _attention_scores()
    auroc = auroc_against_truth(attn, is_real)
    print(
        f"\ngate AUROC: {auroc:.4f}  "
        f"(edges={len(attn)}, real={int(is_real.sum())}, "
        f"fake={int((~is_real).sum())})"
    )
    assert auroc > 0.9


def test_a_reversed_ranking_fails_the_gate():
    """The negative control this gate needs. See PlantedPredictor.attention:
    its signal is calibrated to be strong but not certain, precisely so that
    reversing the ranking direction breaks the result instead of leaving it
    unchanged. The old shift-based gate could not run this check at all,
    because a shift of exactly 0 for every fake edge separates perfectly no
    matter which way the comparison points."""
    attn, is_real = _attention_scores(rank_fn=_reversed_rank_edges)
    auroc = auroc_against_truth(attn, is_real)
    print(f"\nreversed-ranking AUROC: {auroc:.4f}")
    assert auroc < 0.9


def test_the_gate_clears_point_nine_across_several_seeds():
    """One seed proves the gate can pass. It does not prove the gate
    reliably passes. This test plants 5 independent scenes -- a different
    influence graph and a different simulated walk each time, see
    GATE_SEEDS -- and requires every one of them to clear AUROC 0.9."""
    results = {}
    for seed in GATE_SEEDS:
        attn, is_real = _attention_scores(seed=seed)
        results[seed] = auroc_against_truth(attn, is_real)
    for seed, auroc in results.items():
        print(f"\ngate AUROC (seed={seed}): {auroc:.4f}")
    failing = {seed: auroc for seed, auroc in results.items() if auroc <= 0.9}
    assert not failing, f"these seeds did not clear 0.9: {failing}"


def test_a_reversed_ranking_fails_the_gate_across_several_seeds():
    """The negative control, repeated across the same 5 seeds: a reversed
    ranking must fail the gate every time, not just once."""
    results = {}
    for seed in GATE_SEEDS:
        attn, is_real = _attention_scores(seed=seed, rank_fn=_reversed_rank_edges)
        results[seed] = auroc_against_truth(attn, is_real)
    for seed, auroc in results.items():
        print(f"\nreversed-ranking AUROC (seed={seed}): {auroc:.4f}")
    failing = {seed: auroc for seed, auroc in results.items() if auroc >= 0.9}
    assert not failing, f"these seeds did not fail the gate: {failing}"


# ---------------------------------------------------------------------------
# Fixture check: PlantedPredictor's own forecast mechanism. See the module
# docstring. This is not the gate. It is a real thing to verify on its own:
# that PlantedPredictor's shift genuinely comes only from self.influence.
# ---------------------------------------------------------------------------


def test_the_planted_predictor_obeys_the_graph():
    """PlantedPredictor's SHIFT separates real from fake edges perfectly,
    and it must: predict() masks every edge outside self.influence to zero
    force, so a fake edge can never move the forecast. That is a fact about
    PlantedPredictor's own mechanism, proven directly here. It is not a
    claim about attention, and it is not the gate: see
    test_the_gate_auroc_clears_point_nine for that."""
    eligible, true_edges = _eligible_windows()
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

    shifts = np.array(all_shifts, dtype=np.float64)
    is_real = np.array(all_is_real, dtype=bool)
    assert auroc_against_truth(shifts, is_real) == pytest.approx(1.0)
    assert (shifts[~is_real] == 0.0).all(), "a fake edge must never move the forecast"


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
