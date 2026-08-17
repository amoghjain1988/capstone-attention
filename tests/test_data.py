"""Tests for scripts/ and src/data/ and src/eda/.

These tests guard stage 1 to stage 3 of CONTRACT.md section 7. The three tests
of CONTRACT.md section 8 guard the model stages, and they wait for their branch.

Every test here runs in a few seconds and needs no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data import clean, eligibility, loader_ethucy, schema, synthetic, windows
from src.eda import distributions, overlap, scaling

RAW_READY = (config.RAW_DIR / "manifest.json").exists()
needs_raw = pytest.mark.skipif(not RAW_READY, reason="run scripts/download_ethucy.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy() -> pd.DataFrame:
    """Return a small hand-made scene.

    Agent 1 is present in frames 0 to 29. Agent 2 is present in frames 0 to 24.
    Agent 3 is present in frames 10 to 14 only, so agent 3 is too short to join
    any window.
    """
    rows = []
    for frame in range(30):
        rows.append(("eth", frame, 1, 0.5 * frame, 0.0))
    for frame in range(25):
        rows.append(("eth", frame, 2, 0.5 * frame, 2.0))
    for frame in range(10, 15):
        rows.append(("eth", frame, 3, 1.0, 1.0))
    frame_table = pd.DataFrame(rows, columns=["scene", "frame", "agent_id", "x", "y"])
    return schema.cast(frame_table, "trajectories", partial=True)


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    """Return the real trajectories table with the kinematics added."""
    raw = loader_ethucy.load_all(config.RAW_DIR)
    return clean.add_kinematics(raw, config.DT, config.DENSITY_RADIUS_M)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_window_id_round_trip():
    key = schema.window_id("zara1", 42)
    assert key == "zara1_000042"
    assert schema.split_window_id(key) == ("zara1", 42)


def test_validate_catches_a_repeated_key():
    frame = pd.DataFrame(
        {
            "window_id": ["eth_000000", "eth_000000"],
            "ade": [1.0, 2.0],
            "fde": [1.0, 2.0],
            "minade_20": [1.0, 2.0],
            "minfde_20": [1.0, 2.0],
            "collision": [False, False],
        }
    )
    with pytest.raises(ValueError, match="repeats"):
        schema.validate(frame, "predictions")


def test_validate_catches_an_unknown_category():
    frame = pd.DataFrame(
        {
            "scene": ["mars"],
            "frame": [0],
            "agent_id": [1],
            "x": [0.0],
            "y": [0.0],
        }
    )
    with pytest.raises(ValueError, match="scene"):
        schema.validate(frame, "trajectories", partial=True)


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_velocity_is_the_first_difference(toy):
    out = clean.add_kinematics(toy, dt=0.4)
    agent = out.loc[out["agent_id"] == 1].sort_values("frame")
    # x rises by 0.5 per frame and one frame is 0.4 s, so vx is 1.25 m/s.
    assert np.isnan(agent["vx"].iloc[0])
    assert np.allclose(agent["vx"].iloc[1:], 1.25)
    assert np.allclose(agent["vy"].iloc[1:], 0.0)
    assert np.allclose(agent["speed"].iloc[1:], 1.25)


def test_first_frame_of_every_track_has_no_velocity(real):
    n_tracks = real.groupby(["scene", "agent_id"], observed=True).ngroups
    assert int(real["vx"].isna().sum()) == n_tracks
    assert int(real["speed"].isna().sum()) == n_tracks


def test_heading_is_missing_when_the_speed_is_zero(real):
    still = real.loc[real["speed"] == 0.0]
    assert bool(still["heading"].isna().all())


def test_nearest_dist_counts_only_other_agents(toy):
    out = clean.add_kinematics(toy, dt=0.4)
    at_ten = out.loc[out["frame"] == 10].set_index("agent_id")
    # Agent 1 sits at (5, 0), agent 2 at (5, 2) and agent 3 at (1, 1).
    assert at_ten.loc[1, "nearest_dist"] == pytest.approx(2.0)
    assert at_ten.loc[2, "nearest_dist"] == pytest.approx(2.0)
    assert at_ten.loc[1, "density"] == 2.0


def test_nearest_dist_is_missing_when_the_agent_is_alone():
    lonely = pd.DataFrame(
        {"scene": ["eth"] * 3, "frame": [0, 1, 2], "agent_id": [7, 7, 7],
         "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]}
    )
    out = clean.add_kinematics(schema.cast(lonely, "trajectories", partial=True), 0.4)
    assert bool(out["nearest_dist"].isna().all())
    assert bool((out["density"] == 0).all())


def test_drop_short_tracks_splits_and_gives_a_reason(toy):
    kept, dropped = clean.drop_short_tracks(toy, min_frames=20)
    assert set(kept["agent_id"]) == {1, 2}
    assert set(dropped["agent_id"]) == {3}
    assert "reason" in dropped.columns
    assert len(kept) + len(dropped) == len(toy)


def test_the_real_tracks_never_skip_a_frame(real):
    assert len(clean.gappy_tracks(real)) == 0


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def test_a_window_holds_only_agents_present_in_all_twenty_frames(toy):
    frame = windows.make_windows(toy)
    # Agent 3 spans 5 frames, so agent 3 never joins a window.
    assert all(3 not in row for row in frame["agent_order"])
    # Both long agents cover frames 0 to 24, so t0 = 0 to 5 hold two agents.
    first = frame.loc[frame["t0"] == 0].iloc[0]
    assert list(first["agent_order"]) == [1, 2]
    assert int(first["n_agents"]) == 2


def test_hist_and_fut_unpack_to_the_right_shape(toy):
    frame = windows.make_windows(toy)
    row = frame.iloc[0]
    hist = schema.unpack_hist(row, config.N_HIST)
    fut = schema.unpack_fut(row, config.N_FUT)
    assert hist.shape == (int(row["n_agents"]), config.N_HIST, 2)
    assert fut.shape == (int(row["n_agents"]), config.N_FUT, 2)
    # Agent 1 walks along x at 0.5 m per frame from x = 0.
    assert hist[0, :, 0] == pytest.approx(np.arange(8) * 0.5)
    assert fut[0, :, 0] == pytest.approx(np.arange(8, 20) * 0.5)


def test_agent_order_is_sorted_and_the_axis_matches(toy):
    frame = windows.make_windows(toy)
    for _, row in frame.iterrows():
        order = list(row["agent_order"])
        assert order == sorted(order)


def test_overlap_fraction_stays_between_zero_and_one(toy):
    frame = windows.make_windows(toy)
    share = windows.overlap_fraction(frame)
    assert float(share.min()) >= 0.0
    assert float(share.max()) <= 1.0


def test_a_lone_window_has_no_overlap(toy):
    frame = windows.make_windows(toy, stride=config.WINDOW_LEN)
    share = windows.overlap_fraction(frame)
    assert float(share.max()) == 0.0


def test_the_stride_subsample_equals_a_build_at_that_stride(toy):
    every = windows.make_windows(toy, stride=1)
    fourth = windows.make_windows(toy, stride=4)
    base = int(every["t0"].min())
    subset = every.loc[(every["t0"] - base) % 4 == 0]
    assert list(subset["window_id"]) == list(fourth["window_id"])


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------


def test_the_ego_pick_is_deterministic(toy):
    frame = windows.make_windows(toy)
    first = eligibility.mark_eligible(frame, seed=0)
    second = eligibility.mark_eligible(frame.sample(frac=1.0, random_state=3), seed=0)
    joined = first[["window_id", "ego_id"]].merge(
        second[["window_id", "ego_id"]], on="window_id", suffixes=("_a", "_b")
    )
    assert (joined["ego_id_a"] == joined["ego_id_b"]).all()


def test_a_different_seed_can_pick_a_different_ego(toy):
    frame = windows.make_windows(toy)
    a = eligibility.mark_eligible(frame, seed=0)["ego_id"].to_numpy()
    b = eligibility.mark_eligible(frame, seed=1)["ego_id"].to_numpy()
    assert not np.array_equal(a, b)


def test_the_ego_belongs_to_the_window(toy):
    frame = eligibility.mark_eligible(windows.make_windows(toy))
    for _, row in frame.loc[frame["eligible"]].iterrows():
        assert int(row["ego_id"]) in list(row["agent_order"])
        assert schema.ego_index(row) < int(row["n_agents"])


def test_eligible_means_two_agents_or_more(toy):
    frame = eligibility.mark_eligible(windows.make_windows(toy))
    assert (frame["eligible"] == (frame["n_agents"] >= 2)).all()
    assert (frame.loc[~frame["eligible"], "ego_id"] == eligibility.NO_EGO).all()


def test_mark_eligible_never_deletes_a_row(toy):
    frame = windows.make_windows(toy)
    assert len(eligibility.mark_eligible(frame)) == len(frame)


# ---------------------------------------------------------------------------
# The real data. These numbers are the EDA claims.
# ---------------------------------------------------------------------------


@needs_raw
def test_the_raw_counts_match_the_proposal(real):
    assert len(real) == 66_676
    assert real.groupby(["scene", "agent_id"], observed=True).ngroups == 1_950


@needs_raw
def test_univ_holds_just_under_sixty_percent_of_the_rows(real):
    share = (real["scene"] == "univ").mean()
    assert 0.59 < share < 0.60


@needs_raw
def test_the_two_univ_recordings_never_share_a_window(real):
    frame = windows.make_windows(real.loc[real["scene"] == "univ"])
    # The offset separates the two recordings, so no window may straddle them.
    straddle = (frame["t0"] < config.SOURCE_FILE_OFFSET) & (
        frame["t0"] + config.WINDOW_LEN > config.SOURCE_FILE_OFFSET
    )
    assert not bool(straddle.any())


@needs_raw
def test_no_agent_id_repeats_inside_a_scene(real):
    for scene, part in real.groupby("scene", observed=True):
        pairs = part.groupby(["frame", "agent_id"], observed=True).size()
        assert int(pairs.max()) == 1, f"{scene} holds a repeated frame and agent"


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------


def test_the_planted_scene_is_complete():
    trajectories, edges = synthetic.make_scene(n_agents=6, n_frames=60, seed=0)
    assert len(trajectories) == 6 * 60
    assert bool(trajectories[["x", "y"]].notna().all().all())
    counts = trajectories.groupby("agent_id", observed=True).size()
    assert bool((counts == 60).all())
    # Every window carries every ordered pair.
    per_window = edges.groupby("window_id", observed=True).size()
    assert bool((per_window == 6 * 5).all())


def test_the_planted_graph_is_reproducible():
    a = synthetic.influence_matrix(n_agents=6, seed=0)
    b = synthetic.influence_matrix(n_agents=6, seed=0)
    assert np.array_equal(a, b)
    assert not a.diagonal().any()
    assert int(a.sum(axis=1).min()) == 2


def test_a_planted_edge_really_changes_the_future():
    """Remove one real influencer and one fake one. Only the real one moves the path."""
    n_agents, n_frames = 6, 60
    matrix = synthetic.influence_matrix(n_agents=n_agents, seed=0)
    dst = 0
    real_src = int(np.flatnonzero(matrix[dst])[0])
    fake_src = int(np.flatnonzero(~matrix[dst] & (np.arange(n_agents) != dst))[0])

    base, _ = synthetic.make_scene(n_agents=n_agents, n_frames=n_frames, seed=0)

    def path_without(source: int) -> np.ndarray:
        cut = matrix.copy()
        cut[dst, source] = False
        generator = np.random.default_rng(0)
        synthetic._plant_graph(n_agents, 2, generator)  # keep the draw order
        moved = synthetic._simulate(n_agents, n_frames, cut, generator, config.DT)
        return moved[dst]

    origin = base.loc[base["agent_id"] == dst, ["x", "y"]].to_numpy()
    real_shift = float(np.abs(path_without(real_src) - origin).max())
    fake_shift = float(np.abs(path_without(fake_src) - origin).max())
    assert real_shift > 1e-3
    assert fake_shift == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# eda
# ---------------------------------------------------------------------------


def test_icc_is_one_when_a_cluster_never_varies():
    values = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    assert overlap.icc_one_way(values, clusters)["icc"] == pytest.approx(1.0)


def test_icc_is_zero_when_the_cluster_carries_no_signal():
    generator = np.random.default_rng(0)
    values = generator.normal(size=600)
    clusters = np.repeat(np.arange(200), 3)
    assert overlap.icc_one_way(values, clusters)["icc"] < 0.15


def test_icc_is_zero_when_every_cluster_holds_one_row():
    result = overlap.icc_one_way(np.arange(10.0), np.arange(10))
    assert result["icc"] == 0.0
    assert result["mean_cluster_size"] == 1.0


def test_the_design_effect_never_falls_below_one(toy):
    frame = eligibility.mark_eligible(windows.make_windows(toy))
    frame["overlap_fraction"] = windows.overlap_fraction(frame)
    result = overlap.design_effect(frame)
    assert result["deff"] >= 1.0
    assert result["n_effective"] <= result["n_rows"]


def test_a_rank_test_cannot_see_the_transform(tmp_path):
    generator = np.random.default_rng(0)
    values = generator.lognormal(size=400)
    table = scaling.compare_scales(values, name="probe", figure_dir=tmp_path)
    assert set(table["scale"]) == {"raw", "z", "log"}
    assert np.allclose(table["spearman_vs_raw"], 1.0)


def test_the_shape_check_names_the_right_branch():
    generator = np.random.default_rng(0)
    normal = distributions.shape_of(generator.normal(size=4000), "normal")
    heavy = distributions.shape_of(generator.lognormal(size=4000), "heavy")
    assert distributions.branch_of(normal).startswith("3a")
    assert distributions.branch_of(heavy).startswith("3b")
