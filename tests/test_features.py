"""Tests for src/features/.

New file. Does not touch test_data.py, test_shapes.py or test_ablation.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data import schema
from src.features import context, kinematics, proximity

WINDOWS_READY = (config.PROCESSED_DIR / "windows.parquet").exists()
needs_windows = pytest.mark.skipif(not WINDOWS_READY, reason="run scripts/run_all.py")


def _hand_window(agent_order, ego_id, positions, velocities=None):
    """Build a small window row by hand.

    positions maps agent_id to its (x, y) at the LAST observed frame.
    velocities maps agent_id to a constant (vx, vy), default (0, 0). hist is
    built backward from the last frame at that velocity, so the first
    difference of the last two frames reproduces the given velocity exactly.
    """
    velocities = velocities or {}
    n_agents = len(agent_order)
    hist = np.zeros((n_agents, config.N_HIST, 2), dtype=np.float64)
    for slot, agent in enumerate(agent_order):
        vx, vy = velocities.get(agent, (0.0, 0.0))
        x0, y0 = positions[agent]
        for t in range(config.N_HIST):
            back = (config.N_HIST - 1 - t) * config.DT
            hist[slot, t, 0] = x0 - vx * back
            hist[slot, t, 1] = y0 - vy * back
    fut = np.zeros((n_agents, config.N_FUT, 2), dtype=np.float64)

    return pd.Series(
        {
            "window_id": "hand_000000",
            "scene": "eth",
            "t0": 0,
            "ego_id": int(ego_id),
            "agent_order": [int(a) for a in agent_order],
            "n_agents": n_agents,
            "hist": hist.reshape(-1).tolist(),
            "fut": fut.reshape(-1).tolist(),
        }
    )


@pytest.fixture(scope="module")
def real_windows() -> pd.DataFrame:
    return schema.read("windows", config.PROCESSED_DIR)


@pytest.fixture(scope="module")
def real_trajectories() -> pd.DataFrame:
    return schema.read("trajectories", config.PROCESSED_DIR)


# ---------------------------------------------------------------------------
# proximity
# ---------------------------------------------------------------------------


def test_two_agents_three_metres_apart():
    row = _hand_window([1, 2], ego_id=1, positions={1: (0.0, 0.0), 2: (3.0, 0.0)})
    out = proximity.pair_distance(row, trajectories=None)
    assert list(out.columns) == [
        "window_id",
        "src",
        "dst",
        "dist_at_last_frame",
        "rank_dist",
    ]
    assert len(out) == 1
    assert int(out.iloc[0]["src"]) == 2
    assert int(out.iloc[0]["dst"]) == 1
    assert out.iloc[0]["dist_at_last_frame"] == pytest.approx(3.0)
    assert int(out.iloc[0]["rank_dist"]) == 1


def test_rank_dist_ties_break_by_the_lower_src_id():
    row = _hand_window(
        [1, 2, 3],
        ego_id=1,
        positions={1: (0.0, 0.0), 2: (5.0, 0.0), 3: (0.0, 5.0)},
    )
    out = proximity.pair_distance(row, trajectories=None)
    assert list(out["src"]) == [2, 3]
    assert list(out["rank_dist"]) == [1, 2]


@needs_windows
def test_proximity_runs_on_a_real_univ_window(real_windows, real_trajectories):
    univ = real_windows.loc[(real_windows["scene"] == "univ") & real_windows["eligible"]]
    row = univ.loc[univ["n_agents"].idxmax()]
    out = proximity.pair_distance(row, real_trajectories)
    assert len(out) == int(row["n_agents"]) - 1
    assert list(out["rank_dist"]) == list(range(1, len(out) + 1))
    assert out["dist_at_last_frame"].is_monotonic_increasing


@needs_windows
def test_proximity_runs_on_a_real_eth_window(real_windows, real_trajectories):
    eth = real_windows.loc[(real_windows["scene"] == "eth") & real_windows["eligible"]]
    row = eth.iloc[0]
    out = proximity.pair_distance(row, real_trajectories)
    assert len(out) == int(row["n_agents"]) - 1
    assert set(out["dst"]) == {int(row["ego_id"])}


@needs_windows
def test_pair_distance_all_matches_the_eligible_edge_count(real_windows, real_trajectories):
    out = proximity.pair_distance_all(real_windows, real_trajectories)
    expected = int((real_windows.loc[real_windows["eligible"], "n_agents"] - 1).sum())
    assert len(out) == expected


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------


def test_two_agents_approaching_at_one_metre_per_second():
    row = _hand_window(
        [1, 2],
        ego_id=1,
        positions={1: (0.0, 0.0), 2: (4.0, 0.0)},
        velocities={2: (-1.0, 0.0)},
    )
    out = kinematics.pair_kinematics(row, trajectories=None)
    assert out.iloc[0]["closing_speed"] == pytest.approx(1.0)
    assert out.iloc[0]["inv_ttc"] == pytest.approx(0.25)


def test_two_agents_moving_apart():
    row = _hand_window(
        [1, 2],
        ego_id=1,
        positions={1: (0.0, 0.0), 2: (4.0, 0.0)},
        velocities={2: (1.0, 0.0)},
    )
    out = kinematics.pair_kinematics(row, trajectories=None)
    assert out.iloc[0]["closing_speed"] < 0.0
    assert out.iloc[0]["inv_ttc"] == pytest.approx(0.0)


def test_kinematics_never_divides_by_zero_when_coincident():
    row = _hand_window([1, 2], ego_id=1, positions={1: (2.0, 2.0), 2: (2.0, 2.0)})
    out = kinematics.pair_kinematics(row, trajectories=None)
    assert out.iloc[0]["closing_speed"] == 0.0
    assert out.iloc[0]["inv_ttc"] == 0.0


def test_inv_ttc_is_never_negative_or_nan():
    row = _hand_window(
        [1, 2],
        ego_id=1,
        positions={1: (0.0, 0.0), 2: (4.0, 0.0)},
        velocities={2: (1.0, 0.0)},
    )
    out = kinematics.pair_kinematics(row, trajectories=None)
    assert not out["inv_ttc"].isna().any()
    assert (out["inv_ttc"] >= 0.0).all()


@needs_windows
def test_kinematics_runs_on_a_real_univ_and_eth_window(real_windows, real_trajectories):
    for scene in ("univ", "eth"):
        block = real_windows.loc[(real_windows["scene"] == scene) & real_windows["eligible"]]
        row = block.iloc[0]
        out = kinematics.pair_kinematics(row, real_trajectories)
        assert len(out) == int(row["n_agents"]) - 1
        assert not out["inv_ttc"].isna().any()
        assert (out["inv_ttc"] >= 0.0).all()


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


def test_context_counts_only_agents_inside_the_radius():
    row = _hand_window(
        [1, 2, 3],
        ego_id=1,
        positions={1: (0.0, 0.0), 2: (1.0, 0.0), 3: (100.0, 0.0)},
    )
    out = context.window_context(row, trajectories=None)
    assert out["window_id"] == "hand_000000"
    assert out["density"] == 1.0
    assert out["n_agents"] == 3


@needs_windows
def test_context_all_covers_every_eligible_window(real_windows, real_trajectories):
    out = context.window_context_all(real_windows, real_trajectories)
    assert len(out) == int(real_windows["eligible"].sum())
    assert (out["density"] >= 0.0).all()
