"""Tests for src/attention/edges.py.

See docs/FINISH_PLAN.md section 3.9.

The fixture builds a synthetic scene, cuts it into windows, marks the ego, and
then rebuilds the cache rows that scripts/extract_attention.py writes. The
rebuild mirrors extract_attention.edges_of_window exactly, so the test feeds
build_table the same shape of input that the real cache holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.attention import aggregate, edges, rank
from src.data import eligibility, schema, synthetic, windows
from src.models.mock import MockPredictor

N_AGENTS = 5
N_FRAMES = 40
N_WINDOWS = 3
N_EDGES = N_AGENTS - 1  # E, the edges that point into the ego


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


def build_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trajectories, the first N_WINDOWS eligible windows)."""
    trajectories, _ = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=3
    )
    frame = windows.make_windows(
        trajectories, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE
    )
    frame["overlap_fraction"] = windows.overlap_fraction(frame, config.WINDOW_LEN)
    frame = eligibility.mark_eligible(frame, config.SEED)
    eligible = frame.loc[frame["eligible"]].head(N_WINDOWS).reset_index(drop=True)
    return trajectories, eligible


def cache_rows(model, window_row: pd.Series) -> list[dict]:
    """Return the cache records of one window.

    The loop is a copy of scripts/extract_attention.py::edges_of_window. A
    test must not depend on a script module, so the loop lives here instead.
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


def build_cache(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the cache of every window of the frame."""
    model = MockPredictor()
    records: list[dict] = []
    for _, row in frame.iterrows():
        records.extend(cache_rows(model, row))
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (trajectories, windows, cache)."""
    trajectories, frame = build_windows()
    return trajectories, frame, build_cache(frame)


@pytest.fixture(scope="module")
def table(fixture) -> pd.DataFrame:
    """Return the attention_edges table of the fixture."""
    trajectories, frame, cache = fixture
    return edges.build_table(cache, frame, trajectories)


# ---------------------------------------------------------------------------
# build_table
# ---------------------------------------------------------------------------


def test_the_fixture_holds_three_windows_of_five_agents(fixture):
    _, frame, _ = fixture
    assert len(frame) == N_WINDOWS
    assert set(frame["n_agents"]) == {N_AGENTS}


def test_build_table_returns_the_schema_columns(table):
    assert list(table.columns) == list(schema.ATTENTION_EDGES)


def test_build_table_holds_no_null(table):
    assert int(table.isna().sum().sum()) == 0


def test_build_table_holds_one_row_per_window_module_collapse_and_edge(table):
    expected = N_WINDOWS * len(config.MODULES) * len(config.TIME_AGGS) * N_EDGES
    assert len(table) == expected


def test_build_table_passes_the_schema(table):
    schema.validate(table, "attention_edges")


def test_build_table_keeps_the_rank_columns_in_range(table):
    assert set(table["rank_attn"]) == set(range(1, N_EDGES + 1))
    assert set(table["rank_dist"]) == set(range(1, N_EDGES + 1))


def test_build_table_gives_the_mock_the_same_attention_and_distance_order(table):
    """MockPredictor weights an edge by distance, so the two ranks agree.

    The test states the known answer of the fixture. If this ever fails, the
    join lined the distance of one pair up against the attention of another.
    """
    primary = edges.primary(table)
    assert (primary["rank_attn"].astype(int) == primary["rank_dist"].astype(int)).all()


def test_build_table_drops_a_window_that_is_not_eligible(fixture):
    trajectories, frame, cache = fixture
    fewer = frame.copy()
    fewer.loc[fewer.index[0], "eligible"] = False
    table = edges.build_table(cache, fewer, trajectories)
    kept = set(table["window_id"].astype(str))
    assert str(frame.loc[frame.index[0], "window_id"]) not in kept
    assert len(kept) == N_WINDOWS - 1


def test_build_table_drops_a_cache_row_of_an_unknown_window(fixture):
    """The cache on disk covers stride 1. The frozen table is a subset."""
    trajectories, frame, cache = fixture
    stranger = cache.head(1).copy()
    stranger["window_id"] = "synthetic_999999"
    bigger = pd.concat([cache, stranger], ignore_index=True)
    table = edges.build_table(bigger, frame, trajectories)
    assert "synthetic_999999" not in set(table["window_id"].astype(str))


def test_build_table_raises_when_the_join_leaves_a_null(fixture):
    trajectories, frame, cache = fixture
    orphan = cache.head(1).copy()
    orphan["src"] = 9999  # no such agent, so no distance and no kinematics
    broken = pd.concat([cache, orphan], ignore_index=True)
    with pytest.raises(ValueError, match="null"):
        edges.build_table(broken, frame, trajectories)


def test_build_table_raises_when_the_cache_lacks_a_column(fixture):
    trajectories, frame, cache = fixture
    with pytest.raises(KeyError, match="row_entropy"):
        edges.build_table(cache.drop(columns=["row_entropy"]), frame, trajectories)


def test_build_table_returns_an_empty_table_when_no_window_is_eligible(fixture):
    trajectories, frame, cache = fixture
    none_eligible = frame.assign(eligible=False)
    table = edges.build_table(cache, none_eligible, trajectories)
    assert len(table) == 0
    assert list(table.columns) == list(schema.ATTENTION_EDGES)


# ---------------------------------------------------------------------------
# primary, edges_of_window, edges_by_window
# ---------------------------------------------------------------------------


def test_primary_slices_one_module_and_one_collapse(table):
    primary = edges.primary(table)
    assert set(primary["module"].astype(str)) == {config.PRIMARY_MODULE}
    assert set(primary["time_agg"].astype(str)) == {config.PRIMARY_TIME_AGG}
    assert len(primary) == N_WINDOWS * N_EDGES


def test_primary_takes_another_module(table):
    primary = edges.primary(table, module="cross", time_agg="last")
    assert set(primary["module"].astype(str)) == {"cross"}
    assert set(primary["time_agg"].astype(str)) == {"last"}


def test_primary_rejects_an_unknown_module(table):
    with pytest.raises(ValueError, match="module"):
        edges.primary(table, module="mlp")


def test_primary_rejects_an_unknown_collapse(table):
    with pytest.raises(ValueError, match="time_agg"):
        edges.primary(table, time_agg="median")


def test_edges_of_window_returns_one_window_strongest_first(table, fixture):
    _, frame, _ = fixture
    window_id = str(frame.loc[frame.index[0], "window_id"])
    primary = edges.primary(table)
    block = edges.edges_of_window(primary, window_id)

    assert len(block) == N_EDGES
    assert set(block["window_id"].astype(str)) == {window_id}
    assert list(block["rank_attn"].astype(int)) == list(range(1, N_EDGES + 1))


def test_edges_of_window_carries_what_order_edges_needs(table, fixture):
    """order_edges reads window_id for random and rank_dist for nearest."""
    _, frame, _ = fixture
    window_id = str(frame.loc[frame.index[0], "window_id"])
    block = edges.edges_of_window(edges.primary(table), window_id)
    for name in ("window_id", "src", "attn", "rank_dist"):
        assert name in block.columns


def test_edges_by_window_covers_every_window(table, fixture):
    _, frame, _ = fixture
    grouped = edges.edges_by_window(edges.primary(table))
    assert set(grouped) == set(frame["window_id"].astype(str))
    assert all(len(block) == N_EDGES for block in grouped.values())


def test_edges_by_window_matches_edges_of_window(table, fixture):
    _, frame, _ = fixture
    primary = edges.primary(table)
    window_id = str(frame.loc[frame.index[1], "window_id"])
    grouped = edges.edges_by_window(primary)[window_id]
    single = edges.edges_of_window(primary, window_id)
    assert np.array_equal(
        grouped["src"].to_numpy(), single["src"].to_numpy()
    )
