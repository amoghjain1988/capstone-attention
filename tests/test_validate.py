"""Tests for src/hypotheses/validate.py.

See docs/FINISH_PLAN.md sections 3.7 and 5.

THE FIXTURE. The planted scene of tests/test_h1_h2.py serves this file too.
That fixture builds once per session and it gives a non-zero difference for
H1 and for H2, so every row of the sensitivity table reads real numbers. This
file imports the builder rather than a second copy of it, so the two files
never drift apart.

THE FULL EDGE TABLE. tests/test_h1_h2.py returns the PRIMARY slice of the
attention edges, because H1 and H2 read that slice alone. The module check and
the time_agg check of validate.py read the other slices, so this file rebuilds
the whole table from the same planted scene and the same model. The rebuild
runs no forward pass of the ablation. It only reads the attention.

SPEED. A module fixture holds config.N_BOOT and config.N_PERM at 300 for the
whole file, and every call passes the same small configuration down. The file
then runs in a few seconds on one CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.attention import edges as attention_edges
from src.data import schema, synthetic
from src.faithfulness.validation import PlantedPredictor
from src.hypotheses import h1_morf_vs_lerf as h1
from src.hypotheses import validate

try:  # pytest puts tests/ on the import path, so both files share one module.
    from test_h1_h2 import (
        N_AGENTS,
        N_FRAMES,
        N_REAL,
        SCENE_SEED,
        SMALL,
        cache_rows,
        fixture,
        small_config,
    )
except ImportError:  # pragma: no cover - a plain python run needs the package
    from tests.test_h1_h2 import (
        N_AGENTS,
        N_FRAMES,
        N_REAL,
        SCENE_SEED,
        SMALL,
        cache_rows,
        fixture,
        small_config,
    )

# Every check that docs/FINISH_PLAN.md section 3.7 demands of the table.
REQUIRED_CHECKS = (
    "pooled",
    "loso",
    "heterogeneity",
    "n_removed",
    "module",
    "time_agg",
    "mask_policy",
    "noise_floor",
)

# The three checks that may hold a NaN p value. None of them runs a test.
NO_P_CHECKS = ("heterogeneity", "noise_floor")


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def small_resamples():
    """Hold config.N_BOOT and config.N_PERM at 300 for the whole file.

    `src/hypotheses/choose_test.py` reads config.N_BOOT at call time, so a
    small configuration object alone does not shorten that bootstrap. This
    patch does.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "N_BOOT", SMALL)
    patch.setattr(config, "N_PERM", SMALL)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def scene():
    """Return (windows, primary edges, curves) of the planted scene."""
    return fixture()


@pytest.fixture(scope="module")
def cfg():
    """Return the small configuration that every call passes down."""
    return small_config()


@pytest.fixture(scope="module")
def all_edges(scene):
    """Return the attention edges of every module and every time collapse."""
    picked, _, _ = scene
    trajectories, _ = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=SCENE_SEED, n_real=N_REAL
    )
    influence = synthetic.influence_matrix(
        n_agents=N_AGENTS, seed=SCENE_SEED, n_real=N_REAL
    )
    model = PlantedPredictor(influence)

    records: list[dict] = []
    for _, row in picked.iterrows():
        records.extend(cache_rows(model, row))
    return attention_edges.build_table(pd.DataFrame(records), picked, trajectories)


@pytest.fixture(scope="module")
def table(scene, all_edges, cfg):
    """Return the sensitivity table of the planted scene."""
    _, _, curves = scene
    return validate.run(curves, all_edges, cfg)


# ---------------------------------------------------------------------------
# The shape of the table
# ---------------------------------------------------------------------------


def test_the_column_set_is_exact(table):
    """The table holds the columns of section 3.7 and no other column."""
    assert list(table.columns) == list(validate.COLUMNS)


def test_every_check_appears_at_least_once(table):
    present = set(table["check"])
    for name in REQUIRED_CHECKS:
        assert name in present, name


def test_every_variant_of_every_check_appears(table):
    """The module, the time collapse and the sweep each name their variants."""
    def variants(name: str) -> set[str]:
        return set(table.loc[table["check"] == name, "variant"])

    assert variants("pooled") == {"all"}
    assert variants("module") == set(validate.MODULE_VARIANTS)
    assert variants("time_agg") == set(validate.TIME_AGG_VARIANTS)
    assert variants("n_removed") == {str(k) for k in validate.N_REMOVED_VARIANTS}
    assert variants("mask_policy") == {validate.SECOND_MASK_POLICY}
    assert variants("noise_floor") == {"morf_vs_random"}
    assert variants("heterogeneity") == set(validate.HYPOTHESES)


def test_both_hypotheses_carry_a_pooled_row(table):
    pooled = table.loc[table["check"] == "pooled"]
    assert set(pooled["hypothesis"]) == set(validate.HYPOTHESES)


# ---------------------------------------------------------------------------
# The p values
# ---------------------------------------------------------------------------


def test_every_p_value_lies_in_the_unit_interval_or_names_its_reason(table):
    """A NaN p value belongs only to a row that runs no test."""
    for row in table.itertuples():
        if np.isnan(row.p_raw):
            empty_mask_row = row.check == "mask_policy" and int(row.n_windows) == 0
            assert row.check in NO_P_CHECKS or empty_mask_row, (
                row.check,
                row.variant,
                row.hypothesis,
            )
            continue
        assert 0.0 <= row.p_raw <= 1.0, (row.check, row.variant)


def test_the_heterogeneity_row_carries_i_squared_and_the_range(table):
    rows = table.loc[table["check"] == "heterogeneity"]
    assert len(rows) == 2
    for row in rows.itertuples():
        assert row.effect_name == validate.HETEROGENEITY_EFFECT
        assert 0.0 <= row.effect <= 100.0
        assert row.location_m >= 0.0
        assert np.isnan(row.ci_low) and np.isnan(row.ci_high)
        assert np.isnan(row.p_raw)
        assert pd.isna(row.n_windows)


def test_the_noise_floor_row_names_a_share_and_a_median(table):
    row = table.loc[table["check"] == "noise_floor"].iloc[0]
    assert row["hypothesis"] == "h1"
    assert row["effect_name"] == validate.NOISE_FLOOR_EFFECT
    assert 0.0 <= float(row["effect"]) <= 1.0
    assert float(row["location_m"]) >= 0.0
    assert np.isnan(float(row["ci_low"]))
    assert np.isnan(float(row["p_raw"]))


def test_the_mask_policy_rows_stay_empty_without_weight_zero_curves(table, scene):
    """PlantedPredictor writes no weight_zero row, so both rows read no data."""
    _, _, curves = scene
    assert not validate.single_lookup(curves, validate.SECOND_MASK_POLICY)
    rows = table.loc[table["check"] == "mask_policy"]
    assert len(rows) == 2
    assert (rows["n_windows"] == 0).all()
    assert rows["p_raw"].isna().all()
    assert rows["effect"].isna().all()


# ---------------------------------------------------------------------------
# The pooled row against the primary test
# ---------------------------------------------------------------------------


def test_the_pooled_h1_row_repeats_the_primary_h1_test(table, scene, cfg):
    """The pooled row is the same test on the same rows, so it must agree."""
    _, _, curves = scene
    primary = h1.run(curves, cfg)["primary"]
    row = table.loc[(table["check"] == "pooled") & (table["hypothesis"] == "h1")].iloc[0]

    assert float(row["p_raw"]) == pytest.approx(float(primary["p_raw"]))
    assert float(row["effect"]) == pytest.approx(float(primary["effect"]))
    assert row["effect_name"] == primary["effect_name"]
    assert int(row["n_windows"]) == int(primary["n_windows"])
    assert int(row["n_clusters"]) == int(primary["n_clusters"])


# ---------------------------------------------------------------------------
# Leave one scene out
# ---------------------------------------------------------------------------


def test_the_loso_rows_cover_every_scene_of_the_curves(table, scene):
    _, _, curves = scene
    scenes = {
        schema.split_window_id(str(window_id))[0]
        for window_id in curves["window_id"].astype(str)
    }
    for hypothesis in validate.HYPOTHESES:
        rows = table.loc[
            (table["check"] == "loso") & (table["hypothesis"] == hypothesis)
        ]
        assert set(rows["variant"]) == scenes


def test_the_loso_cluster_count_comes_from_the_paired_test(table):
    """The count is the component count, so it never exceeds the window count."""
    rows = table.loc[table["check"] == "loso"]
    assert len(rows) > 0
    for row in rows.itertuples():
        assert 0 < int(row.n_clusters) <= int(row.n_windows)


# ---------------------------------------------------------------------------
# The n_removed sweep
# ---------------------------------------------------------------------------


def test_the_sweep_reads_no_more_windows_as_the_step_grows(table):
    """A window with E edges holds E steps, so a deep step reads fewer windows."""
    rows = table.loc[table["check"] == "n_removed"].copy()
    rows["step"] = rows["variant"].astype(int)
    rows = rows.sort_values("step")
    counts = [int(value) for value in rows["n_windows"]]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 0


def test_every_sweep_row_names_h1(table):
    rows = table.loc[table["check"] == "n_removed"]
    assert set(rows["hypothesis"]) == {"h1"}


# ---------------------------------------------------------------------------
# variant_d
# ---------------------------------------------------------------------------


def test_the_primary_variant_reproduces_the_pooled_h1_differences(
    scene, all_edges, cfg
):
    """The single sweep and the arms must agree at n_removed 1, bit for bit.

    At step 1 the MoRF arm removes the top attention edge and the LeRF arm
    removes the bottom one. The single arm already measured both edges under
    the same draws and the same mask. The two routes to D must therefore give
    the same float.
    """
    _, _, curves = scene
    pooled = h1.paired_shift(
        curves, "morf", "lerf", int(cfg.PRIMARY_N_REMOVED), str(cfg.PRIMARY_MASK_POLICY)
    )
    variant, skipped = validate.variant_d(
        curves,
        all_edges,
        "h1",
        module=str(cfg.PRIMARY_MODULE),
        time_agg=str(cfg.PRIMARY_TIME_AGG),
        seed=int(cfg.SEED),
    )

    assert skipped == 0
    assert len(variant) == len(pooled)
    merged = pooled.merge(variant, on="window_id", suffixes=("_arm", "_single"))
    assert len(merged) == len(pooled)
    assert (merged["d_arm"].to_numpy() == merged["d_single"].to_numpy()).all()
    assert (merged["shift_a_arm"].to_numpy() == merged["shift_a_single"].to_numpy()).all()


def test_variant_d_reads_the_nearest_edge_for_h2(scene, all_edges, cfg):
    """H2 compares the top attention edge against the edge with rank_dist 1."""
    _, _, curves = scene
    pooled = h1.paired_shift(
        curves,
        "morf",
        "nearest",
        int(cfg.PRIMARY_N_REMOVED),
        str(cfg.PRIMARY_MASK_POLICY),
    )
    variant, skipped = validate.variant_d(
        curves,
        all_edges,
        "h2",
        module=str(cfg.PRIMARY_MODULE),
        time_agg=str(cfg.PRIMARY_TIME_AGG),
        seed=int(cfg.SEED),
    )
    assert skipped == 0
    merged = pooled.merge(variant, on="window_id", suffixes=("_arm", "_single"))
    assert (merged["d_arm"].to_numpy() == merged["d_single"].to_numpy()).all()


def test_variant_d_counts_a_window_without_a_single_row(scene, all_edges, cfg):
    """A hole in the sweep drops the window, and the count reports it."""
    _, _, curves = scene
    victim = sorted(set(curves["window_id"].astype(str)))[0]
    broken = curves.loc[
        ~(
            (curves["window_id"].astype(str) == victim)
            & (curves["arm"].astype(str) == "single")
        )
    ]
    table, skipped = validate.variant_d(broken, all_edges, "h1", seed=int(cfg.SEED))
    assert skipped == 1
    assert victim not in set(table["window_id"])


def test_variant_d_refuses_an_unknown_hypothesis(scene, all_edges):
    _, _, curves = scene
    with pytest.raises(ValueError):
        validate.variant_d(curves, all_edges, "h3")


def test_variant_d_returns_an_empty_table_for_an_absent_policy(scene, all_edges):
    """weight_zero holds no row here, so the function skips every window."""
    _, _, curves = scene
    table, skipped = validate.variant_d(
        curves, all_edges, "h1", mask_policy=validate.SECOND_MASK_POLICY
    )
    assert len(table) == 0
    assert skipped > 0
    assert list(table.columns) == list(h1.PAIRED_COLUMNS)


# ---------------------------------------------------------------------------
# single_lookup
# ---------------------------------------------------------------------------


def _hand_built_single_rows() -> pd.DataFrame:
    """Return one window with two single edges, two policies and two draws.

    The lowest draw_id is 0. The map must therefore hold the shift of draw 0
    and never the shift of draw 1, and it must hold one policy at a time.
    """
    rows = [
        ("single", 1, 7, 0, "logit_neg_inf", 0.40),
        ("single", 2, 9, 0, "logit_neg_inf", 0.10),
        ("single", 1, 7, 1, "logit_neg_inf", 0.99),
        ("single", 2, 9, 1, "logit_neg_inf", 0.98),
        ("single", 1, 7, 0, "weight_zero", 0.20),
        ("morf", 1, -1, 0, "logit_neg_inf", 0.40),
    ]
    return pd.DataFrame(
        [
            {
                "window_id": "synthetic_000000",
                "ego_id": 3,
                "arm": arm,
                "n_removed": n_removed,
                "removed_mass": 0.5,
                "edge_src": edge_src,
                "draw_id": draw_id,
                "mask_policy": policy,
                "run_id": "test",
                "shift": shift_value,
            }
            for arm, n_removed, edge_src, draw_id, policy, shift_value in rows
        ]
    )


def test_single_lookup_maps_one_edge_to_one_shift():
    lookup = validate.single_lookup(_hand_built_single_rows(), "logit_neg_inf")
    assert lookup == {
        ("synthetic_000000", 7): 0.40,
        ("synthetic_000000", 9): 0.10,
    }


def test_single_lookup_reads_the_named_policy_only():
    lookup = validate.single_lookup(_hand_built_single_rows(), "weight_zero")
    assert lookup == {("synthetic_000000", 7): 0.20}


def test_single_lookup_returns_an_empty_map_for_an_absent_policy():
    frame = _hand_built_single_rows()
    frame = frame.loc[frame["mask_policy"] != "weight_zero"]
    assert validate.single_lookup(frame, "weight_zero") == {}


def test_single_lookup_requires_the_curve_columns():
    with pytest.raises(KeyError):
        validate.single_lookup(pd.DataFrame({"window_id": ["synthetic_000000"]}))


# ---------------------------------------------------------------------------
# The verbose flag
# ---------------------------------------------------------------------------


def test_the_verbose_flag_prints_one_line_per_row(scene, all_edges, cfg, capsys):
    _, _, curves = scene
    verbose_table = validate.run(curves, all_edges, cfg, verbose=True)
    printed = capsys.readouterr().out.splitlines()
    named = [line for line in printed if line.startswith("  ")]
    assert len(named) == len(verbose_table)
