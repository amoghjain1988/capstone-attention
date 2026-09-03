"""Tests for src/ablation/driver.py and faithfulness_from_curves.

See docs/FINISH_PLAN.md sections 3.10 and 3.11.

Every test runs against MockPredictor on a CPU. MockPredictor honours the edge
mask and it is deterministic, so the audit of consistency_check has a known
right answer: at n_removed 1 an arm removes one edge, and the single arm
already measured that same edge, so the two shifts must be the same float.

MockPredictor.predict does not accept a mask_policy argument, so
src/ablation/mask.py raises NotImplementedError for weight_zero. The driver
must catch that, drop the policy, and write no weight_zero row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.ablation import driver
from src.ablation.arms import ARMS, order_edges
from src.attention import aggregate, edges, rank
from src.data import eligibility, schema, synthetic, windows
from src.faithfulness.index import (
    brute_force_single,
    faithfulness_from_curves,
    faithfulness_index,
)
from src.models.mock import MockPredictor

N_AGENTS = 5
N_FRAMES = 40
N_WINDOWS = 3
N_EDGES = N_AGENTS - 1  # E, the edges that point into the ego
SCENE = "synthetic"


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


class CountingModel:
    """Wrap MockPredictor and count every forward pass that succeeds.

    A call that raises does not count, because it runs no forward pass. The
    weight_zero attempt against MockPredictor is one such call.
    """

    def __init__(self) -> None:
        self.inner = MockPredictor()
        self.calls = 0

    def predict(self, hist, edge_mask=None, draws=None, **extra):
        """Return (K, N, 12, 2) and count the pass."""
        out = self.inner.predict(hist, edge_mask=edge_mask, draws=draws, **extra)
        self.calls += 1
        return out

    def attention(self, hist):
        """Pass the attention call through."""
        return self.inner.attention(hist)


def cache_rows(model, window_row: pd.Series) -> list[dict]:
    """Return the cache records of one window.

    The loop mirrors scripts/extract_attention.py::edges_of_window, so the
    test feeds build_table the shape that the real cache holds.
    """
    n_agents = int(window_row["n_agents"])
    hist = schema.unpack_hist(window_row, config.N_HIST)
    ego_index = schema.ego_index(window_row)
    order = [int(a) for a in window_row["agent_order"]]

    records: list[dict] = []
    for module, raw in model.attention(hist).items():
        entropy = aggregate.agent_row_entropy(raw, n_agents)
        for time_agg in config.TIME_AGGS:
            graph = aggregate.collapse(raw, n_agents, time_agg)
            ranked = rank.rank_edges(graph, ego_index, order)
            for record in ranked.to_dict("records"):
                records.append(
                    {
                        "window_id": str(window_row["window_id"]),
                        "src": int(record["src"]),
                        "dst": int(record["dst"]),
                        "module": module,
                        "time_agg": time_agg,
                        "attn": float(record["attn"]),
                        "row_entropy": float(entropy[int(record["src_index"])]),
                        "rank_attn": int(record["rank_attn"]),
                    }
                )
    return records


@pytest.fixture(scope="module")
def scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (the first N_WINDOWS eligible windows, the primary edges)."""
    trajectories, _ = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=3
    )
    frame = windows.make_windows(
        trajectories, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE
    )
    frame["overlap_fraction"] = windows.overlap_fraction(frame, config.WINDOW_LEN)
    frame = eligibility.mark_eligible(frame, config.SEED)
    eligible = frame.loc[frame["eligible"]].head(N_WINDOWS).reset_index(drop=True)

    model = MockPredictor()
    records: list[dict] = []
    for _, row in eligible.iterrows():
        records.extend(cache_rows(model, row))
    table = edges.build_table(pd.DataFrame(records), eligible, trajectories)
    return eligible, edges.primary(table)


@pytest.fixture(scope="module")
def curves(scene) -> pd.DataFrame:
    """Return every curve row of the three windows."""
    eligible, edges_primary = scene
    model = MockPredictor()
    frames = [
        driver.window_curves(
            row, model, edges.edges_of_window(edges_primary, str(row["window_id"]))
        )
        for _, row in eligible.iterrows()
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# window_curves
# ---------------------------------------------------------------------------


def test_window_curves_runs_every_arm_of_the_primary_policy(curves):
    primary = curves.loc[
        curves["mask_policy"].astype(str) == config.PRIMARY_MASK_POLICY
    ]
    assert set(primary["arm"].astype(str)) == set(ARMS)


def test_window_curves_covers_every_window(curves):
    assert curves["window_id"].nunique() == N_WINDOWS


def test_window_curves_writes_no_weight_zero_row_for_the_mock(curves):
    """MockPredictor.predict takes no mask_policy, so the driver drops it.

    A weight_zero row here would be a silent copy of the primary policy under
    a second name. See src/ablation/mask.py::masked_predict.
    """
    assert set(curves["mask_policy"].astype(str)) == {config.PRIMARY_MASK_POLICY}


def test_window_curves_records_the_skipped_policy_once(scene):
    eligible, edges_primary = scene
    driver.reset_policy_skips()
    row = eligible.iloc[0]
    edges_window = edges.edges_of_window(edges_primary, str(row["window_id"]))

    with pytest.warns(UserWarning, match="weight_zero"):
        driver.window_curves(row, MockPredictor(), edges_window)
    assert driver.policy_skips() == (("MockPredictor", "weight_zero"),)


def test_window_curves_gives_every_arm_the_right_step_count(curves):
    counts = (
        curves.groupby([curves["window_id"].astype(str), curves["arm"].astype(str)])
        .size()
        .unstack()
    )
    for arm in ("morf", "lerf", "nearest", "random", "single"):
        assert (counts[arm] == N_EDGES).all()
    # weight_matched matches mass, not count, so it holds at most E steps.
    assert (counts["weight_matched"] <= N_EDGES).all()


def test_window_curves_names_the_removed_edge_only_on_the_single_arm(curves):
    single = curves.loc[curves["arm"].astype(str) == "single"]
    other = curves.loc[curves["arm"].astype(str) != "single"]
    assert (single["edge_src"].astype(int) >= 0).all()
    assert (other["edge_src"].astype(int) == -1).all()


def test_window_curves_rejects_a_window_with_no_edge(scene):
    eligible, edges_primary = scene
    empty = edges_primary.head(0)
    with pytest.raises(ValueError, match="no edge"):
        driver.window_curves(eligible.iloc[0], MockPredictor(), empty)


def test_window_curves_costs_about_seven_e_forward_passes(scene):
    """One baseline serves the window. Every arm step is one masked pass."""
    eligible, edges_primary = scene
    row = eligible.iloc[0]
    model = CountingModel()
    driver.window_curves(
        row, model, edges.edges_of_window(edges_primary, str(row["window_id"]))
    )
    # Five ordered arms of E steps each, plus one baseline, is the floor.
    # weight_matched adds at most E more. The mock refuses the second policy.
    assert 1 + 5 * N_EDGES <= model.calls <= driver.forward_passes_per_window(
        N_EDGES, n_policies=1
    )


def test_forward_passes_per_window_is_seven_e_plus_one_for_two_policies():
    assert driver.forward_passes_per_window(4, n_policies=2) == 7 * 4 + 1
    assert driver.forward_passes_per_window(1, n_policies=2) == 8


def test_scene_forward_passes_sums_over_the_windows(scene):
    eligible, edges_primary = scene
    total = driver.scene_forward_passes(eligible, edges_primary, n_policies=2)
    assert total == N_WINDOWS * driver.forward_passes_per_window(N_EDGES, 2)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def test_step_one_of_every_ordered_arm_equals_the_single_row_exactly(curves, scene):
    _, edges_primary = scene
    for window_id, block in curves.groupby(curves["window_id"].astype(str)):
        edges_window = edges.edges_of_window(edges_primary, window_id)
        single = block.loc[block["arm"].astype(str) == "single"]
        shift_of = {
            int(row.edge_src): float(row.shift) for row in single.itertuples()
        }
        for arm in driver.ORDERED_ARMS:
            src = order_edges(edges_window, arm, config.SEED)[0]
            step = block.loc[
                (block["arm"].astype(str) == arm) & (block["n_removed"] == 1)
            ]
            assert len(step) == 1
            assert float(step["shift"].iloc[0]) == shift_of[src]


def test_consistency_check_finds_nothing_with_the_edges(curves, scene):
    _, edges_primary = scene
    assert len(driver.consistency_check(curves, edges_primary)) == 0


def test_consistency_check_finds_nothing_without_the_edges(curves):
    """Without the edge table the audit names the edge from removed_mass."""
    assert len(driver.consistency_check(curves)) == 0


def test_consistency_check_catches_a_planted_mismatch(curves, scene):
    _, edges_primary = scene
    broken = curves.copy()
    target = broken.index[
        (broken["arm"].astype(str) == "morf") & (broken["n_removed"] == 1)
    ][0]
    broken.loc[target, "shift"] = float(broken.loc[target, "shift"]) + 1.0

    found = driver.consistency_check(broken, edges_primary)
    assert len(found) == 1
    assert found["arm"].iloc[0] == "morf"
    assert found["difference"].iloc[0] == pytest.approx(1.0)


def test_consistency_check_returns_an_empty_frame_for_no_curve():
    found = driver.consistency_check(driver.empty_curves())
    assert len(found) == 0
    assert list(found.columns) == list(driver.MISMATCH_COLUMNS)


# ---------------------------------------------------------------------------
# run_scene
# ---------------------------------------------------------------------------


def test_run_scene_writes_a_checkpoint_and_adds_nothing_on_a_second_call(
    scene, tmp_path
):
    eligible, edges_primary = scene
    target = tmp_path / f"curves_{SCENE}.parquet"

    model = CountingModel()
    first = driver.run_scene(SCENE, eligible, edges_primary, model, target)
    assert target.exists()
    assert len(first) > 0
    assert first["window_id"].nunique() == N_WINDOWS
    first_calls = model.calls
    assert first_calls > 0

    again = driver.run_scene(SCENE, eligible, edges_primary, model, target)
    assert len(again) == len(first)
    assert model.calls == first_calls  # no window runs twice


def test_run_scene_matches_window_curves(scene, tmp_path, curves):
    eligible, edges_primary = scene
    target = tmp_path / f"curves_{SCENE}.parquet"
    result = driver.run_scene(SCENE, eligible, edges_primary, MockPredictor(), target)
    assert len(result) == len(curves)
    assert result["shift"].sum() == pytest.approx(curves["shift"].sum())


def test_run_scene_honours_the_limit(scene, tmp_path):
    eligible, edges_primary = scene
    target = tmp_path / f"curves_{SCENE}.parquet"
    result = driver.run_scene(
        SCENE, eligible, edges_primary, MockPredictor(), target, limit=2
    )
    assert result["window_id"].nunique() == 2


def test_run_scene_force_runs_every_window_again(scene, tmp_path):
    eligible, edges_primary = scene
    target = tmp_path / f"curves_{SCENE}.parquet"
    model = CountingModel()
    first = driver.run_scene(SCENE, eligible, edges_primary, model, target)
    before = model.calls
    again = driver.run_scene(
        SCENE, eligible, edges_primary, model, target, force=True
    )
    assert model.calls == 2 * before
    assert len(again) == len(first)


def test_schema_write_accepts_the_result(scene, tmp_path):
    eligible, edges_primary = scene
    target = tmp_path / f"curves_{SCENE}.parquet"
    result = driver.run_scene(SCENE, eligible, edges_primary, MockPredictor(), target)

    path = schema.write(result, "perturbation_curves", tmp_path)
    back = schema.read("perturbation_curves", tmp_path)
    assert len(back) == len(result)
    assert path.exists()
    schema.validate(back, "perturbation_curves", partial=True)


# ---------------------------------------------------------------------------
# faithfulness_from_curves
# ---------------------------------------------------------------------------


def test_faithfulness_from_curves_matches_faithfulness_index(curves, scene):
    _, edges_primary = scene
    table = faithfulness_from_curves(curves, edges_primary)
    assert len(table) == N_WINDOWS
    assert list(table.columns) == list(schema.FAITHFULNESS)

    single = curves.loc[curves["arm"].astype(str) == "single"]
    top = edges_primary.loc[edges_primary["rank_attn"].astype(int) == 1]
    morf_of = dict(zip(top["window_id"].astype(str), top["src"].astype(int)))

    for row in table.itertuples():
        block = single.loc[single["window_id"].astype(str) == row.window_id]
        sweep = pd.DataFrame(
            {
                "src": block["edge_src"].astype(int).to_numpy(),
                "shift": block["shift"].astype(float).to_numpy(),
            }
        )
        assert row.n_edges == N_EDGES
        assert row.floor == pytest.approx(float(sweep["shift"].mean()))
        assert row.ceiling == pytest.approx(float(sweep["shift"].max()))
        assert row.fi == pytest.approx(
            faithfulness_index(sweep, morf_of[row.window_id])
        )


def test_the_single_arm_reproduces_brute_force_single(curves, scene):
    """The single rows and brute_force_single must be the same numbers."""
    eligible, edges_primary = scene
    model = MockPredictor()
    draws = np.arange(config.N_SAMPLES, dtype=np.int64) + config.SEED

    row = eligible.iloc[0]
    window_id = str(row["window_id"])
    edges_window = edges.edges_of_window(edges_primary, window_id)
    sweep = brute_force_single(row, model, edges_window, draws)

    block = curves.loc[
        (curves["window_id"].astype(str) == window_id)
        & (curves["arm"].astype(str) == "single")
    ]
    from_curves = dict(
        zip(block["edge_src"].astype(int), block["shift"].astype(float))
    )
    for record in sweep.itertuples():
        assert from_curves[int(record.src)] == float(record.shift)


def test_faithfulness_from_curves_returns_an_empty_table_for_no_curve(scene):
    _, edges_primary = scene
    table = faithfulness_from_curves(driver.empty_curves(), edges_primary)
    assert len(table) == 0
    assert list(table.columns) == list(schema.FAITHFULNESS)


def test_faithfulness_from_curves_passes_the_schema(curves, scene, tmp_path):
    _, edges_primary = scene
    table = faithfulness_from_curves(curves, edges_primary)
    schema.write(table, "faithfulness", tmp_path)
