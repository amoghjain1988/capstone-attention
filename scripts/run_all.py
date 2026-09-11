"""Run every stage that is ready, in the order of CONTRACT.md section 7.

See CONTRACT.md section 5.1 for the signatures and docs/FINISH_PLAN.md
section 4 for the order of the work.

    1. scripts/download_ethucy.py            the manifest check
    2. src/data/                             trajectories and windows
    3. src/eda/                              the exploratory tables
    4. src/features/ and src/attention/      the attention_edges table
       (this one stage also does the work of stage 6, see below)
    5. src/hypotheses/design.py              the frozen design
    7. src/ablation/                         the perturbation_curves table
    8. src/faithfulness/ and src/metrics/    the faithfulness table, the
                                             predictions table
    9. src/hypotheses/ and src/stats/        the verdicts table
   10. notebooks/capstone.ipynb              the notebook

TWO WINDOWS TABLES. config.WINDOW_STRIDE is 5 and config.MIN_EGO_EDGES is 2.
Those two numbers are frozen, and every stage after the exploratory analysis
reads the table that carries them. The exploratory sweep of
src/eda/overlap.py needs the stride 1 table instead, because it subsamples a
stride from that table. Stage 2 therefore builds both tables. The frozen table
goes to data/processed/windows.parquet, which is the table of CONTRACT.md
section 3. The stride 1 table goes to data/interim/windows_stride1.parquet,
which is a cache and not one of the seven tables.

STAGE 6. scripts/extract_attention.py already cached the attention of every
window on the GPU. Stage 4 reads that cache and joins it to the pair features,
so stage 4 and stage 6 are one stage here. The combined stage carries the
smaller number, so `--through 4` already builds the attention_edges table.

Every stage is idempotent. Run the script twice and nothing changes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.data import clean, eligibility, loader_ethucy, schema, windows  # noqa: E402
from src.eda import density, distributions, missing, overlap, scaling, summary  # noqa: E402

# The number of the last stage. --through takes any number up to this one.
LAST_STAGE = 10

# The columns of outputs/tables/design.csv.
DESIGN_COLUMNS = (
    "design",
    "stride",
    "unit",
    "n_windows",
    "n_components",
    "n_pedestrians",
    "n_time_blocks",
    "icc",
    "mean_cluster_size",
    "deff",
    "n_effective",
    "alpha",
    "sided",
    "primary_n_removed",
    "minimum_effect",
    "power",
)

# The columns of outputs/tables/fi_attrition.csv.
FI_ATTRITION_COLUMNS = (
    "scene",
    "n_windows",
    "n_two_edge",
    "n_nan",
    "median_fi",
    "share_fi_above_zero",
)

# The columns of outputs/tables/hypotheses_extra.csv. The verdicts table holds
# the columns of CONTRACT.md section 3 and nothing else, so every other number
# that a reader needs goes here: the shape report of src/hypotheses/
# choose_test.py, the location in metres with its interval, the share of
# agreement of H2, and the two H3 numbers that no verdict column carries.
HYPOTHESES_EXTRA_COLUMNS = (
    "hypothesis",
    "test_used",
    "why",
    "n_windows",
    "n_zero",
    "n_clusters",
    "effect_name",
    "effect",
    "location_m",
    "location_ci_low",
    "location_ci_high",
    "share_agree",
    "share_positive",
    "mean",
    "median",
    "pseudomedian",
    "skew",
    "skew_ci_low",
    "skew_ci_high",
    "kurtosis",
    "shapiro_w",
    "shapiro_p",
    "symmetric",
    "p_wald",
    "n_dropped_two_edge",
)

# The columns of the verdicts table that stage 9 prints. The table carries the
# `why` sentence too, and that sentence is too long for a terminal column, so
# the print drops it and the csv keeps it.
VERDICT_PRINT_COLUMNS = (
    "hypothesis",
    "role",
    "effect_name",
    "effect",
    "p_raw",
    "p_adj",
    "n_clusters",
    "verdict",
)

# The one sentence of the H3 primary row and of the H3 supporting row.
H3_PRIMARY_WHY = (
    "The headline fit drops the two-edge windows, and the wild cluster "
    "bootstrap by component tests the 4 context terms, with the scene fixed "
    "effects kept in the model."
)
H3_SUPPORTING_WHY = (
    "This fit keeps the two-edge windows, and the wild cluster bootstrap by "
    "component tests the 4 context terms, with the scene fixed effects kept "
    "in the model."
)

# The test name of both H3 rows.
H3_TEST_USED = (
    "ols with scene fixed effects, joint Wald test of the 4 context terms, "
    "wild cluster bootstrap by component"
)


def banner(text: str) -> None:
    """Print the title of one stage."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def short(path: Path) -> str:
    """Return the path under the repository root, or the whole path."""
    try:
        return str(Path(path).relative_to(config.ROOT))
    except ValueError:
        return str(path)


def save_table(frame: pd.DataFrame, name: str) -> Path:
    """Write one exploratory table to outputs/tables."""
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLE_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    print(f"  wrote {short(path)}  ({len(frame)} rows)")
    return path


# ---------------------------------------------------------------------------
# Stage 2: the two windows tables
# ---------------------------------------------------------------------------


def stride1_path() -> Path:
    """Return the path of the stride 1 windows cache.

    The function reads config.INTERIM_DIR at call time, not at import time, so
    a test that points config at a temporary folder still gets the right path.
    """
    return config.INTERIM_DIR / "windows_stride1.parquet"


def build_windows(trajectories: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Return the windows table at one stride, with the ego and eligible."""
    frame = windows.make_windows(trajectories, config.N_HIST, config.N_FUT, int(stride))
    frame["overlap_fraction"] = windows.overlap_fraction(frame, config.WINDOW_LEN)
    return eligibility.mark_eligible(frame, config.SEED)


def write_windows_cache(frame: pd.DataFrame, path: Path) -> Path:
    """Cast, check and write the stride 1 windows table to one exact path.

    schema.write names the file after the table, so it cannot write a second
    windows table under a second name. This function repeats the three steps
    of schema.write and writes the path it receives.
    """
    out = schema.cast(frame, "windows")
    schema.validate(out, "windows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def read_windows_cache(path: Path) -> pd.DataFrame | None:
    """Return the cached stride 1 windows table, or None when it is stale.

    The cache is stale when it holds an eligible window with fewer agents than
    config.MIN_EGO_EDGES allows. That happens when the edge floor changed after
    the cache was written. A stale cache is rebuilt, never read.
    """
    path = Path(path)
    if not path.exists():
        return None

    frame = schema.cast(pd.read_parquet(path), "windows", partial=True)
    good = frame.loc[frame["eligible"]]
    if len(good) and int(good["n_agents"].min()) < config.MIN_EGO_EDGES + 1:
        print(f"  the cache {short(path)} holds the old edge floor, so it is rebuilt")
        return None
    return frame


def stage_data(drops: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the trajectories table and both windows tables.

    Returns (trajectories, stride 1 windows, frozen windows). The frozen table
    carries config.WINDOW_STRIDE and config.MIN_EGO_EDGES and goes to
    data/processed/windows.parquet. Every stage after the exploratory analysis
    reads that table. The stride 1 table stays in memory for src/eda/overlap.py
    and goes to data/interim/windows_stride1.parquet, so a later run skips the
    build.
    """
    banner("STAGE 2  src/data")

    start = time.time()
    raw = loader_ethucy.load_all(config.RAW_DIR)
    print(f"  loaded {len(raw):,} rows in {time.time() - start:.1f} s")

    gappy = clean.gappy_tracks(raw)
    print(f"  tracks that skip a frame: {len(gappy)}")
    for _, row in gappy.iterrows():
        drops.append(
            pd.DataFrame(
                [
                    schema.drop_row(
                        stage="clean.gappy_tracks",
                        scene=str(row["scene"]),
                        key=f"agent_id={int(row['agent_id'])}",
                        reason=f"the track skips {int(row['n_gaps'])} frames",
                        n_rows=int(row["n_frames"]),
                    )
                ]
            )
        )

    # Add the kinematics BEFORE any drop. The density and the nearest distance
    # must see every pedestrian that was really there.
    trajectories = clean.add_kinematics(raw, config.DT, config.DENSITY_RADIUS_M)

    if config.DROP_SHORT_TRACKS:
        kept, dropped = clean.drop_short_tracks(trajectories, config.MIN_TRACK_FRAMES)
        drops.append(clean.short_track_log(dropped, config.MIN_TRACK_FRAMES))
        print(f"  dropped {len(dropped):,} rows of short tracks")
        trajectories = kept
    else:
        _, would_drop = clean.drop_short_tracks(trajectories, config.MIN_TRACK_FRAMES)
        print(
            f"  kept {len(would_drop):,} rows of short tracks as neighbours "
            f"(config.DROP_SHORT_TRACKS is False)"
        )

    trajectories = trajectories.drop(columns=["source"], errors="ignore")
    schema.write(trajectories, "trajectories", config.PROCESSED_DIR)
    print(f"  trajectories: {len(trajectories):,} rows")

    # The stride 1 table. src/eda/overlap.py subsamples a stride from it, so
    # the exploratory sweep needs every window start.
    start = time.time()
    stride1 = read_windows_cache(stride1_path())
    if stride1 is None:
        stride1 = build_windows(trajectories, 1)
        write_windows_cache(stride1, stride1_path())
        print(
            f"  windows at stride 1: {len(stride1):,} built, "
            f"{int(stride1['eligible'].sum()):,} eligible  "
            f"({time.time() - start:.1f} s), cached in {short(stride1_path())}"
        )
    else:
        print(
            f"  windows at stride 1: {len(stride1):,} read from "
            f"{short(stride1_path())}, {int(stride1['eligible'].sum()):,} eligible"
        )

    # The frozen table. Every stage from stage 4 onward reads this one.
    start = time.time()
    frozen = build_windows(trajectories, config.WINDOW_STRIDE)
    drops.append(eligibility.eligibility_log(frozen))
    schema.write(frozen, "windows", config.PROCESSED_DIR)
    print(
        f"  windows at the frozen stride {config.WINDOW_STRIDE}: {len(frozen):,} built, "
        f"{int(frozen['eligible'].sum()):,} eligible  ({time.time() - start:.1f} s)"
    )
    print(
        f"  the frozen table holds the edge floor config.MIN_EGO_EDGES "
        f"{config.MIN_EGO_EDGES}, so every eligible window carries "
        f"{config.MIN_EGO_EDGES} edges or more"
    )
    return trajectories, stride1, frozen


# ---------------------------------------------------------------------------
# Stage 3: the exploratory analysis
# ---------------------------------------------------------------------------


def stage_eda(trajectories: pd.DataFrame, stride1: pd.DataFrame) -> dict:
    """Run every exploratory module. Write one table and one figure each.

    The stage runs on the STRIDE 1 table. src/eda/overlap.py subsamples a
    stride from the table it receives, so it needs every window start. A run
    on the frozen table would report the design effect of stride 5, 10, 20 and
    so on, and src/hypotheses/design.py could not read the row it needs.
    """
    banner("STAGE 3  src/eda  (on the stride 1 windows table)")

    counts = summary.summary(trajectories, stride1)
    save_table(counts, "eda_summary")

    features = density.ego_neighbourhood(stride1, trajectories)
    neighbours = density.density_report(stride1, trajectories)
    save_table(neighbours, "eda_density")

    shapes = distributions.distributions(trajectories)
    save_table(shapes, "eda_distributions")

    gaps = missing.missing_report(trajectories)
    save_table(gaps, "eda_missing")
    save_table(missing.frame_occupancy(trajectories), "eda_frame_occupancy")

    scales = scaling.compare_scales(
        features["ego_nearest_dist"].to_numpy(), name="ego_nearest_dist"
    )
    save_table(scales, "eda_scaling")

    sweep = overlap.overlap_report(stride1, features=features)
    save_table(sweep, "eda_overlap")

    effect = overlap.design_effect(stride1, features=features, unit="pedestrian")
    print("\n  design effect at stride 1, cluster unit pedestrian:")
    for key, value in effect.items():
        print(f"    {key:20s} {value}")
    return effect


# ---------------------------------------------------------------------------
# Stage 4 and stage 6: the attention_edges table and the context table
# ---------------------------------------------------------------------------


def read_attention_cache() -> pd.DataFrame:
    """Return every cached attention row of every scene, in one frame.

    scripts/extract_attention.py writes one file per scene. The function reads
    every data/interim/attention_*.parquet file, so a scene that the cache
    does not hold simply adds no row.
    """
    paths = sorted(config.INTERIM_DIR.glob("attention_*.parquet"))
    if not paths:
        raise SystemExit(
            f"no attention cache under {short(config.INTERIM_DIR)}. "
            "Run scripts/extract_attention.py on a GPU machine first, or copy "
            "the cache folder in, then run this stage again."
        )
    frames = [pd.read_parquet(path) for path in paths]
    names = ", ".join(path.name for path in paths)
    print(f"  read {len(paths)} cache files: {names}")
    return pd.concat(frames, ignore_index=True)


def stage_edges(frozen: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Build the attention_edges table and the context table.

    The stage reads the GPU cache of scripts/extract_attention.py, keeps the
    rows of the eligible windows of the FROZEN table, and joins them to the
    pair features with src/attention/edges.py::build_table. It writes

        data/processed/attention_edges.parquet   the contract table
        data/processed/context.parquet           the H3 covariates
        outputs/tables/arm_overlap.csv           attention against distance

    The stage raises SystemExit when the cache lacks an eligible window,
    because a silent hole would drop that window from every later test.
    """
    banner("STAGE 4 and 6  src/features and src/attention  (attention_edges)")

    from src.ablation import arm_overlap
    from src.attention import edges as edges_module

    cache = read_attention_cache()
    eligible = frozen.loc[frozen["eligible"]]
    wanted = set(eligible["window_id"].astype(str))
    held = set(cache["window_id"].astype(str))
    print(
        f"  cache: {len(cache):,} edge rows over {len(held):,} windows. "
        f"the frozen table asks for {len(wanted):,} eligible windows"
    )

    absent = sorted(wanted - held)
    if absent:
        shown = absent[:20]
        tail = "" if len(absent) == len(shown) else f" and {len(absent) - len(shown)} more"
        raise SystemExit(
            f"the attention cache lacks {len(absent)} eligible windows: "
            f"{shown}{tail}. Run scripts/extract_attention.py on a GPU machine "
            "for those scenes, then run this stage again."
        )

    start = time.time()
    table = edges_module.build_table(cache, frozen, trajectories)
    path = schema.write(table, "attention_edges", config.PROCESSED_DIR)
    print(
        f"  wrote {short(path)}  ({len(table):,} edge rows over "
        f"{table['window_id'].nunique():,} windows, {time.time() - start:.1f} s)"
    )

    primary = edges_module.primary(table)
    print(
        f"  the primary slice (module {config.PRIMARY_MODULE}, time_agg "
        f"{config.PRIMARY_TIME_AGG}) holds {len(primary):,} edge rows over "
        f"{primary['window_id'].nunique():,} windows"
    )

    overlap_table = arm_overlap.attention_vs_distance(table)
    by_scene = arm_overlap.summary_by_scene(overlap_table)
    print(f"  wrote {short(config.TABLE_DIR / 'arm_overlap.csv')}")
    print("  mean Jaccard of attention against distance, top "
          f"{config.PRIMARY_N_REMOVED}, per scene:")
    for row in by_scene.itertuples():
        print(
            f"    {str(row.scene):8s} mean {row.mean:.3f}  median {row.median:.3f}  "
            f"{int(row.n_windows):,} windows"
        )

    # The H3 covariates. context.parquet is not one of the seven tables of
    # CONTRACT.md section 3, so it carries no schema entry and pandas writes it.
    from src.hypotheses import h3_context

    context = h3_context.build_context(frozen, trajectories, primary)
    context_path = config.PROCESSED_DIR / "context.parquet"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(context_path, index=False)
    print(f"  wrote {short(context_path)}  ({len(context):,} window rows)")

    return table


# ---------------------------------------------------------------------------
# Stage 5: the frozen design
# ---------------------------------------------------------------------------


def stage_design(frozen: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Write outputs/tables/design.csv and print both rows.

    The table holds two rows. The FROZEN row is the design that config.py and
    src/hypotheses/design.py state: the stride, the minimum effect, the power,
    alpha, the sided rule, the primary removal count, and the design effect
    that design.py reads from outputs/tables/eda_overlap.csv for the frozen
    stride and its headline cluster unit. The REALISED row measures the frozen
    table itself: the eligible window count, the component count, the
    pedestrian count, the time block count, and the design effect of
    src/eda/overlap.py at the same headline unit.

    The two rows must agree closely. A large gap means that the exploratory
    sweep and the frozen build no longer describe the same sample.
    """
    banner("STAGE 5  src/hypotheses/design.py  (the frozen design)")

    from src.hypotheses import design as design_module
    from src.stats import clustered

    unit = design_module.HEADLINE_UNIT
    deff_frozen, n_windows_frozen = design_module.read_design_effect(
        config.WINDOW_STRIDE, unit
    )
    stated = design_module.frozen_design(deff_frozen, n_windows_frozen)

    if float(stated["minimum_effect"]) != float(config.MINIMUM_EFFECT):
        print(
            f"  WARNING config.MINIMUM_EFFECT is {config.MINIMUM_EFFECT} and "
            f"design.py states {stated['minimum_effect']}. The two must agree."
        )

    eligible = frozen.loc[frozen["eligible"]]
    features = density.ego_neighbourhood(frozen, trajectories)
    realised = overlap.design_effect(frozen, features=features, unit=unit)

    counts = {
        name: clustered.n_clusters(clustered.cluster_labels(eligible, name))
        for name in ("component", "pedestrian", "time_block")
    }

    rows = [
        {
            "design": "frozen",
            "stride": config.WINDOW_STRIDE,
            "unit": unit,
            "n_windows": n_windows_frozen,
            "n_components": float("nan"),
            "n_pedestrians": float("nan"),
            "n_time_blocks": float("nan"),
            "icc": float("nan"),
            "mean_cluster_size": float("nan"),
            "deff": deff_frozen,
            "n_effective": stated["n_effective"],
            "alpha": stated["alpha"],
            "sided": stated["sided"],
            "primary_n_removed": stated["primary_n_removed"],
            "minimum_effect": stated["minimum_effect"],
            "power": stated["power"],
        },
        {
            "design": "realised",
            "stride": config.WINDOW_STRIDE,
            "unit": unit,
            "n_windows": int(len(eligible)),
            "n_components": counts["component"],
            "n_pedestrians": counts["pedestrian"],
            "n_time_blocks": counts["time_block"],
            "icc": realised["icc"],
            "mean_cluster_size": realised["mean_cluster_size"],
            "deff": realised["deff"],
            "n_effective": realised["n_effective"],
            "alpha": float("nan"),
            "sided": "",
            "primary_n_removed": float("nan"),
            "minimum_effect": float("nan"),
            "power": float("nan"),
        },
    ]

    table = pd.DataFrame(rows, columns=list(DESIGN_COLUMNS))
    save_table(table, "design")

    for row in rows:
        print(f"\n  the {row['design']} design")
        for name in DESIGN_COLUMNS[1:]:
            print(f"    {name:20s} {row[name]}")
    print(
        f"\n  the cluster unit of every test is {config.CLUSTER_UNIT}, so the "
        f"honest count beside a p value is {counts['component']} components, "
        f"not {len(eligible)} windows"
    )
    return table


# ---------------------------------------------------------------------------
# Stage 7: the ablation
# ---------------------------------------------------------------------------


def stage_ablation(
    frozen: pd.DataFrame, edges_primary: pd.DataFrame
) -> pd.DataFrame | None:
    """Read data/processed/perturbation_curves.parquet, or say how to build it.

    This stage NEVER runs the model. The ablation needs a masked forward pass
    per edge, so it needs the GPU, and scripts/run_ablation.py owns it. When
    the table is absent, the stage prints the command and the forward pass
    estimate, then returns None. Every later stage then skips its own work.

    When the table is present, the stage audits it with
    src/ablation/driver.py::consistency_check and
    src/hypotheses/data_checks.py::check, writes outputs/tables/attrition.csv,
    and returns the curves.

    THE WEIGHT_MATCHED ARM. data_checks.check asks every arm for a row at
    config.PRIMARY_N_REMOVED. The weight_matched arm matches MASS, not count,
    so src/ablation/curves.py writes the SET SIZE into n_removed, and a window
    at the edge floor needs two edges to reach the mass of the one morf edge.
    Such a window therefore carries no weight_matched row at n_removed 1, and
    check raises. That is a disagreement between two modules, not a hole in
    the data, so this stage catches the error, prints it, and goes on. Neither
    module belongs to this script. See docs/FINISH_PLAN.md section 3.5 for the
    rule that pairs the two arms by mass in h1_morf_vs_lerf.py.
    """
    banner("STAGE 7  src/ablation  (the perturbation_curves table)")

    from src.ablation import driver

    eligible = frozen.loc[frozen["eligible"]]
    path = schema.path_of("perturbation_curves", config.PROCESSED_DIR)

    if not path.exists():
        passes = driver.scene_forward_passes(eligible, edges_primary)
        print(f"  {short(path)} is missing. The ablation needs a GPU, so run:")
        print("\n    python scripts/run_ablation.py\n")
        print(
            f"  the run costs about {passes:,} forward passes over "
            f"{len(eligible):,} eligible windows, both mask policies included"
        )
        return None

    curves = schema.read("perturbation_curves", config.PROCESSED_DIR)
    print(
        f"  read {short(path)}  ({len(curves):,} rows over "
        f"{curves['window_id'].nunique():,} windows)"
    )

    mismatch = driver.consistency_check(curves, edges_primary)
    print(f"  consistency check: {len(mismatch)} mismatched rows")

    from src.hypotheses import data_checks

    try:
        attrition = data_checks.check(curves)
    except ValueError as error:
        print(f"  data_checks.check refuses the table: {error}")
        print(
            "  the stage writes no attrition table and goes on. The primary "
            "arms morf, lerf, nearest, random and single all carry a row at "
            f"n_removed {config.PRIMARY_N_REMOVED}. Only weight_matched does "
            "not, because it matches mass and not count. Take the fix to the "
            "owner of src/hypotheses/data_checks.py."
        )
    else:
        save_table(attrition, "attrition")
    return curves


# ---------------------------------------------------------------------------
# Stage 8: the faithfulness table and the predictions table
# ---------------------------------------------------------------------------


def fi_attrition(faithfulness: pd.DataFrame) -> pd.DataFrame:
    """Return one row per scene of the faithfulness table.

    Columns: scene, n_windows, n_two_edge, n_nan, median_fi,
    share_fi_above_zero. n_two_edge counts the windows at the frozen edge
    floor config.MIN_EGO_EDGES. Such a window has only two reachable index
    values, +1 and -1, because the floor is the mean of two shifts and the
    ceiling is their max. src/hypotheses/h3_context.py drops those windows
    from its headline fit for the same reason.
    """
    rows: list[dict] = []
    if len(faithfulness) == 0:
        return pd.DataFrame(columns=list(FI_ATTRITION_COLUMNS))

    scenes = faithfulness["window_id"].astype(str).map(
        lambda wid: schema.split_window_id(wid)[0]
    )
    for scene, block in faithfulness.groupby(scenes, sort=True):
        values = block["fi"].astype(float)
        good = values.dropna()
        rows.append(
            {
                "scene": str(scene),
                "n_windows": int(len(block)),
                "n_two_edge": int((block["n_edges"].astype(int) == config.MIN_EGO_EDGES).sum()),
                "n_nan": int(values.isna().sum()),
                "median_fi": float(good.median()) if len(good) else float("nan"),
                "share_fi_above_zero": float((good > 0.0).mean()) if len(good) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=list(FI_ATTRITION_COLUMNS))


def stage_faithfulness(
    curves: pd.DataFrame, edges_primary: pd.DataFrame
) -> pd.DataFrame:
    """Write data/processed/faithfulness.parquet and outputs/tables/fi_attrition.csv.

    The single arm of the ablation already removed every edge on its own, so
    src/faithfulness/index.py::faithfulness_from_curves needs no forward pass.
    """
    banner("STAGE 8  src/faithfulness  (the faithfulness table)")

    from src.faithfulness.index import faithfulness_from_curves

    table = faithfulness_from_curves(curves, edges_primary)
    path = schema.write(table, "faithfulness", config.PROCESSED_DIR)
    print(f"  wrote {short(path)}  ({len(table):,} window rows)")

    attrition = fi_attrition(table)
    save_table(attrition, "fi_attrition")
    for row in attrition.itertuples():
        print(
            f"    {str(row.scene):8s} {int(row.n_windows):5,} windows  "
            f"{int(row.n_two_edge):4,} at the edge floor  {int(row.n_nan):4,} NaN  "
            f"median fi {row.median_fi:.3f}  share above 0 {row.share_fi_above_zero:.3f}"
        )
    return table


def stage_predictions() -> None:
    """Build data/processed/predictions.parquet and outputs/tables/calibration.csv.

    This needs the GPU cache under data/interim (scripts/extract_attention.py's
    output), not a GPU itself. Print a clear note and return when that cache
    is not on disk yet. The stage does not fail the whole run.

    CachedPredictor reads the windows table from config.PROCESSED_DIR and keeps
    the eligible rows alone, so after stage 2 it already sees the FROZEN table
    and no stride 1 window reaches this stage. The window count printed below
    is the check: it must match the eligible count of the frozen table.
    """
    banner("STAGE 8  src/metrics (predictions and calibration)")
    if not any(config.INTERIM_DIR.glob("attention_*.parquet")):
        print(
            f"  no cache under {short(config.INTERIM_DIR)}. "
            "Run scripts/extract_attention.py on a GPU machine first, or copy "
            "the cache folder in, then re-run this stage."
        )
        return

    from src.metrics import calibration
    from src.models.cached import CachedPredictor

    cache = CachedPredictor()
    print(f"  {cache}")
    print(
        f"  the cache reads the frozen windows table and keeps "
        f"{len(cache.windows):,} eligible windows"
    )

    predictions = calibration.build_predictions(cache)
    print(f"  wrote data/processed/predictions.parquet  ({len(predictions):,} rows)")

    calib = calibration.calibration_table(cache)
    print(f"  wrote outputs/tables/calibration.csv  ({len(calib)} scene rows)")
    print(
        "  NOTE the determinism limit: AgentFormer is deterministic and the K "
        "draws are K fixed DLow modes, not K samples from a posterior. "
        "coverage and the PIT histogram describe how the K fixed modes sit "
        "around the truth, not a probabilistic calibration claim."
    )


# ---------------------------------------------------------------------------
# Stage 9: the three hypotheses, the verdicts table and the six figures
# ---------------------------------------------------------------------------


def h3_design(
    faithfulness: pd.DataFrame, context: pd.DataFrame, fit_name: str
) -> tuple:
    """Return (y, X, clusters) of one H3 fit, as the wild bootstrap wants them.

    `src/hypotheses/h3_context.py` fits the model with statsmodels and reports
    an ASYMPTOTIC cluster-robust Wald test. That test needs many clusters. The
    study holds about 39 components, and 39 is not many, so the honest p value
    of H3 comes from the wild cluster bootstrap of
    `src/stats/clustered.py::wild_cluster_bootstrap`.

    That function takes a plain design matrix, so this function rebuilds the
    same design that the named fit used. The steps mirror `h3_context.run`
    exactly: join the index onto the covariates, drop the null index, add the
    scene, and, for the headline fit, drop the windows at the frozen edge
    floor. The four covariates then pass through `h3_context._zscore`, so the
    scaling is the scaling of the fit and not a second rule. Column 0 of X is
    the intercept, the next four columns are the covariates, and the last
    columns are the scene dummies with one scene left out, which is the same
    parameterisation that `C(scene)` gives.
    """
    from src.hypotheses import h3_context
    from src.stats import clustered

    columns = list(h3_context.CONTEXT_COLUMNS)
    merged = faithfulness.merge(
        context.loc[:, ["window_id", *columns]],
        on="window_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["fi"].notna()].copy()
    merged["scene"] = (
        merged["window_id"].astype(str).map(lambda wid: schema.split_window_id(wid)[0])
    )
    if str(fit_name) == "without_two_edge":
        merged = merged.loc[merged["n_edges"] != h3_context.TWO_EDGE_TRAP]
    merged = merged.reset_index(drop=True)

    scored = h3_context._zscore(merged, columns)
    dummies = pd.get_dummies(
        scored["scene"].astype(str), prefix="scene", drop_first=True
    ).astype("float64")

    design = pd.concat(
        [
            pd.DataFrame({"intercept": 1.0}, index=scored.index),
            scored.loc[:, columns].astype("float64"),
            dummies,
        ],
        axis=1,
    )
    y = scored["fi"].to_numpy(dtype="float64")
    clusters = clustered.component_clusters(scored)
    return y, design.to_numpy(dtype="float64"), clusters


def h3_bootstrap(
    faithfulness: pd.DataFrame, context: pd.DataFrame, fit_name: str
) -> dict:
    """Return the wild cluster bootstrap of one H3 fit, under the context null.

    The null holds the 4 context columns of the design of `h3_design`, which
    are the columns 1 to 4. The scene dummies stay in the model and out of the
    null. See THE JOINT TEST in `src/hypotheses/h3_context.py` for the reason.
    """
    from src.hypotheses import h3_context
    from src.stats import clustered

    y, design, clusters = h3_design(faithfulness, context, fit_name)
    n_context = len(h3_context.CONTEXT_COLUMNS)
    restriction = np.eye(design.shape[1])[1 : 1 + n_context]
    boot = clustered.wild_cluster_bootstrap(
        y,
        design,
        clusters,
        n_boot=int(config.N_BOOT),
        seed=int(config.SEED),
        restriction=restriction,
    )
    print(
        f"  the wild cluster bootstrap of the {fit_name} fit runs "
        f"{boot['n_boot']:,} draws over {boot['n_clusters']} components on "
        f"{len(y):,} windows"
    )
    return boot


def h3_rows(h3: dict, faithfulness: pd.DataFrame, context: pd.DataFrame) -> list[dict]:
    """Return the H3 primary row and the H3 supporting row.

    The primary row reports the HEADLINE fit of `src/hypotheses/h3_context.py`,
    the fit that drops the windows at the frozen edge floor. The effect is the
    partial R squared of the context block over a scene-only model. The
    interval stays null, because a partial R squared has no cluster bootstrap
    interval in this project.

    The supporting row `h3_with_two_edge` reports the other fit, so a reader
    sees what the two-edge windows do to the answer.

    In both rows `p_raw` is the wild cluster bootstrap p value of the context
    null, and `p_wald` keeps the asymptotic Wald p value of the same null
    beside it. Both rows use the bootstrap, because the components are too few
    for the asymptotic test.
    """
    headline = str(h3["headline"])
    other = "with_two_edge" if headline == "without_two_edge" else "without_two_edge"
    fit = h3[headline]
    second = h3[other]

    boot = h3_bootstrap(faithfulness, context, headline)
    boot_second = h3_bootstrap(faithfulness, context, other)

    primary = {
        "hypothesis": "h3",
        "role": "primary",
        "test_used": H3_TEST_USED,
        "why": H3_PRIMARY_WHY,
        "sided": "two",
        "effect": float(fit["partial_r2"]),
        "effect_name": "partial_r2",
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": float(boot["joint_p"]),
        "n_clusters": int(fit["n_clusters"]),
        "p_wald": float(fit["joint_p"]),
        "n_windows": int(fit["n"]),
        "n_dropped_two_edge": int(h3["n_dropped_two_edge"]),
    }
    supporting = {
        "hypothesis": "h3_with_two_edge",
        "role": "supporting",
        "test_used": H3_TEST_USED,
        "why": H3_SUPPORTING_WHY,
        "sided": "two",
        "effect": float(second["partial_r2"]),
        "effect_name": "partial_r2",
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": float(boot_second["joint_p"]),
        "n_clusters": int(second["n_clusters"]),
        "p_wald": float(second["joint_p"]),
        "n_windows": int(second["n"]),
        "n_dropped_two_edge": 0,
    }
    return [primary, supporting]


def extra_row(result: dict, choice: dict | None, share_agree: float) -> dict:
    """Return one row of outputs/tables/hypotheses_extra.csv for H1 or H2.

    `choice` is the output of `src/hypotheses/choose_test.py::choose`. It is
    None when the sample held too few non-zero differences to read a shape. The
    shape columns then stay null and the row still names the test.
    """
    shape = dict(choice["shape_stats"]) if choice else {}
    row = {name: float("nan") for name in HYPOTHESES_EXTRA_COLUMNS}
    row.update(
        {
            "hypothesis": str(result["hypothesis"]),
            "test_used": str(result["test_used"]),
            "why": str(result["why"]),
            "n_windows": int(result["n_windows"]),
            "n_zero": int(result["n_zero"]),
            "n_clusters": int(result["n_clusters"]),
            "effect_name": str(result["effect_name"]),
            "effect": float(result["effect"]),
            "location_m": float(result["location_m"]),
            "location_ci_low": float(result["location_ci_low"]),
            "location_ci_high": float(result["location_ci_high"]),
            "share_agree": float(share_agree),
        }
    )
    for name in (
        "share_positive",
        "mean",
        "median",
        "pseudomedian",
        "skew",
        "skew_ci_low",
        "skew_ci_high",
        "kurtosis",
        "shapiro_w",
        "shapiro_p",
        "symmetric",
    ):
        if name in shape:
            row[name] = shape[name]
    return row


def hypotheses_extra(h1: dict, h2: dict, h3_primary: dict | None) -> pd.DataFrame:
    """Return the extra table, one row per hypothesis.

    The verdicts table of CONTRACT.md section 3 holds a fixed column list, so
    the shape report, the location in metres and the two H3 numbers have no
    home there. They go here instead, so that no number of the study lives only
    in a print.
    """
    rows = [
        extra_row(h1["primary"], h1["choice"], float("nan")),
        extra_row(h2["primary"], h2["choice"], float(h2["share_agree"])),
    ]
    if h3_primary is not None:
        row = {name: float("nan") for name in HYPOTHESES_EXTRA_COLUMNS}
        row.update(
            {
                "hypothesis": "h3",
                "test_used": str(h3_primary["test_used"]),
                "why": str(h3_primary["why"]),
                "n_windows": int(h3_primary["n_windows"]),
                "n_clusters": int(h3_primary["n_clusters"]),
                "effect_name": str(h3_primary["effect_name"]),
                "effect": float(h3_primary["effect"]),
                "p_wald": float(h3_primary["p_wald"]),
                "n_dropped_two_edge": int(h3_primary["n_dropped_two_edge"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=list(HYPOTHESES_EXTRA_COLUMNS))


def h3_tables(h3: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the coefficient table and the collinearity table of both fits.

    Both frames carry a `fit` column, so one file holds both fits and a reader
    never guesses which fit a row belongs to.
    """
    coefficients = []
    inflation = []
    for name in ("without_two_edge", "with_two_edge"):
        fit = h3[name]
        block = pd.DataFrame(fit["coefficients"]).copy()
        block.insert(0, "fit", name)
        block["headline"] = name == str(h3["headline"])
        block["joint_p"] = float(fit["joint_p"])
        block["partial_r2"] = float(fit["partial_r2"])
        block["n"] = int(fit["n"])
        block["n_clusters"] = int(fit["n_clusters"])
        coefficients.append(block)

        vif = pd.DataFrame(fit["vif"]).copy()
        vif.insert(0, "fit", name)
        inflation.append(vif)
    return (
        pd.concat(coefficients, ignore_index=True),
        pd.concat(inflation, ignore_index=True),
    )


def run_validate(curves: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame | None:
    """Return the sensitivity table, or None when the module is absent.

    `src/hypotheses/validate.py` of docs/FINISH_PLAN.md section 3.7 is written
    in parallel with this script. The import therefore stays inside the stage.
    A missing module costs the sensitivity table and the forest plot. It costs
    no verdict, because no p value of that table enters a family.
    """
    try:
        from src.hypotheses import validate
    except ImportError as error:
        print(
            f"  src/hypotheses/validate.py is not importable ({error}), so the "
            "stage writes no validate.csv and draws no forest plot. Every "
            "verdict below stands, because the sensitivity table is "
            "descriptive and no p value of it enters a family."
        )
        return None
    return validate.run(curves, edges)


def stage_hypotheses(
    curves: pd.DataFrame | None,
    edges: pd.DataFrame,
    faithfulness: pd.DataFrame | None,
    context: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Run the three hypotheses. Write the verdicts table and the six figures.

    The stage writes

        data/processed/verdicts.parquet     the contract table, from report.build
        outputs/tables/verdicts.csv         the same table as a csv
        outputs/tables/hypotheses_extra.csv the shape report and the location
        outputs/tables/h3_coefficients.csv  both H3 fits
        outputs/tables/h3_vif.csv           the collinearity check of both fits
        outputs/tables/validate.csv         the sensitivity table
        outputs/figures/hyp_*.png           the six figures

    THE H3 P VALUE. `src/hypotheses/h3_context.py` reports an asymptotic
    cluster-robust Wald test. The study holds too few components for that
    asymptotic result, so this stage runs the wild cluster bootstrap on the
    design of each fit and puts its joint p value into the H3 rows. The null
    holds the 4 context terms, and the scene fixed effects stay in the model.
    The asymptotic p value stays beside it in the extra table, under `p_wald`.

    The stage returns None when the curves are absent, because every test of
    this stage reads the perturbation_curves table.
    """
    banner("STAGE 9  src/hypotheses and src/stats  (the verdicts table)")

    if curves is None or faithfulness is None or context is None:
        print(
            "  the perturbation_curves table is missing, so no hypothesis "
            "runs. Run scripts/run_ablation.py on the GPU machine first, then "
            "run this script again."
        )
        return None

    from src.hypotheses import figures, h1_morf_vs_lerf, h2_beyond_proximity, h3_context
    from src.stats import report

    start = time.time()
    h1 = h1_morf_vs_lerf.run(curves)
    print(
        f"  H1 morf against lerf: p_raw {h1['primary']['p_raw']:.5f}, "
        f"{h1['primary']['effect_name']} {h1['primary']['effect']:.4f}, "
        f"{h1['primary']['n_windows']:,} windows, "
        f"{h1['primary']['n_clusters']} components  "
        f"({time.time() - start:.1f} s)"
    )

    start = time.time()
    h2 = h2_beyond_proximity.run(curves, edges)
    print(
        f"  H2 morf against nearest: p_raw {h2['primary']['p_raw']:.5f}, "
        f"{h2['primary']['effect_name']} {h2['primary']['effect']:.4f}, "
        f"{h2['primary']['n_windows']:,} windows, "
        f"{h2['primary']['n_clusters']} components, share_agree "
        f"{h2['share_agree']:.4f}  ({time.time() - start:.1f} s)"
    )

    # H3 needs a design matrix of full rank and a covariate with variance. A
    # sample that gives neither is a hole in the data, not a verdict, so the
    # stage says so and goes on with two primary rows.
    h3 = None
    rows_h3: list[dict] = []
    try:
        h3 = h3_context.run(faithfulness, context, config)
        rows_h3 = h3_rows(h3, faithfulness, context)
    except (ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"  H3 does not fit on this sample: {error}")
        print(
            "  WARNING the verdicts table then holds TWO primary rows, so Holm "
            "corrects over two tests and not three. State that in the report."
        )
    else:
        print(
            f"  H3 context on the fi: p_raw {rows_h3[0]['p_raw']:.5f} from the "
            f"wild cluster bootstrap, p_wald {rows_h3[0]['p_wald']:.5f}, "
            f"partial_r2 {rows_h3[0]['effect']:.4f}, "
            f"{rows_h3[0]['n_windows']:,} windows, "
            f"{rows_h3[0]['n_clusters']} components"
        )

    results = [
        h1["primary"],
        h2["primary"],
        *rows_h3[:1],
        *h1["supporting"],
        *h2["supporting"],
        *rows_h3[1:],
    ]
    verdicts = report.build(results)

    print("\n  the verdicts")
    print(
        verdicts.loc[:, list(VERDICT_PRINT_COLUMNS)]
        .to_string(index=False)
        .replace("\n", "\n    ")
    )

    save_table(verdicts, "verdicts")
    extra = hypotheses_extra(h1, h2, rows_h3[0] if rows_h3 else None)
    save_table(extra, "hypotheses_extra")
    if h3 is not None:
        coefficients, inflation = h3_tables(h3)
        save_table(coefficients, "h3_coefficients")
        save_table(inflation, "h3_vif")

    validate_table = run_validate(curves, edges)
    if validate_table is not None:
        save_table(validate_table, "validate")

    print("\n  the figures")
    print(f"  wrote {short(figures.curves_figure(curves))}")
    print(f"  wrote {short(figures.paired_figure(h1['d'], h2['d']))}")
    print(f"  wrote {short(figures.fi_figure(faithfulness))}")
    if h3 is not None:
        # The report headlines the wild cluster bootstrap p, not the
        # asymptotic Wald p of h3_context.py. rows_h3[0]["p_raw"] holds it.
        # The figure states the p that the report states.
        p_bootstrap = float(rows_h3[0]["p_raw"]) if rows_h3 else None
        print(f"  wrote {short(figures.h3_figure(h3, p_bootstrap=p_bootstrap))}")
    if validate_table is not None:
        print(f"  wrote {short(figures.loso_figure(validate_table))}")
    print(f"  wrote {short(figures.summary_figure(verdicts, extra))}")
    return verdicts


# ---------------------------------------------------------------------------
# Stage 10: the notebook
# ---------------------------------------------------------------------------


def load_make_notebook():
    """Return scripts/make_notebook.py as a module.

    The scripts folder is not a package, so importlib loads the file from its
    path. The same pattern loads this script inside tests/test_run_all.py.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "make_notebook.py"
    spec = importlib.util.spec_from_file_location("make_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_notebook() -> Path:
    """Rewrite notebooks/capstone.ipynb and return the path.

    The notebook reads the tables and the figures of the earlier stages. It
    computes nothing, so this stage only rebuilds the file. Every code cell
    goes to disk with no output, so the committed file holds no result.
    """
    banner("STAGE 10  notebooks/capstone.ipynb")

    make_notebook = load_make_notebook()
    path = make_notebook.build()
    n_cells = len(make_notebook.cells())
    print(f"  wrote {short(path)}  ({n_cells} cells, no stored output)")
    print(
        "  the notebook reads outputs/tables and outputs/figures. A cell whose "
        "file is absent prints one note and moves on."
    )
    return path


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def read_context() -> pd.DataFrame | None:
    """Return the H3 covariate table, or None when stage 4 has not run."""
    path = config.PROCESSED_DIR / "context.parquet"
    if not path.exists():
        print(f"  {short(path)} is missing, so H3 has no covariates.")
        return None
    return pd.read_parquet(path)


def main(eda_only: bool = False, through: int = LAST_STAGE) -> None:
    """Run the pipeline. The last stage that runs is `through`.

    Set `eda_only` to True to stop after stage 3. That flag is the same as
    `through=3` and it stays for the callers that already use it.
    """
    through = 3 if eda_only else int(through)
    if through < 1:
        raise SystemExit(f"--through must be at least 1, it is {through}")

    config.make_dirs()
    print(f"run_id config hash: {config.config_hash()}")
    print(f"stages 1 to {min(through, LAST_STAGE)}")

    banner("STAGE 1  scripts/download_ethucy.py")
    manifest = config.RAW_DIR / "manifest.json"
    if not manifest.exists():
        raise SystemExit("Run scripts/download_ethucy.py first. The manifest is missing.")
    print(f"  manifest present: {short(manifest)}")
    if through < 2:
        return

    drops: list[pd.DataFrame] = []
    trajectories, stride1, frozen = stage_data(drops)

    if through >= 3:
        stage_eda(trajectories, stride1)

    log = pd.concat([d for d in drops if len(d)], ignore_index=True) if drops else schema.empty_drops()
    if not len(log):
        log = schema.empty_drops()
    save_table(log, "drops")

    if through < 4:
        return

    # Stage 4 also does the work of stage 6. See the module docstring.
    from src.attention import edges as edges_module

    table = stage_edges(frozen, trajectories)
    edges_primary = edges_module.primary(table)

    if through >= 5:
        stage_design(frozen, trajectories)

    if through < 7:
        return

    curves = stage_ablation(frozen, edges_primary)

    if through < 8:
        return

    faithfulness = None
    if curves is None:
        banner("STAGE 8  src/faithfulness  (the faithfulness table)")
        print(
            "  the perturbation_curves table is missing, so the faithfulness "
            "table waits. Run scripts/run_ablation.py first."
        )
    else:
        faithfulness = stage_faithfulness(curves, edges_primary)

    stage_predictions()

    if through >= 9:
        context = read_context() if curves is not None else None
        stage_hypotheses(curves, table, faithfulness, context)

    if through >= 10:
        stage_notebook()

    banner("NEXT ACTION")
    if curves is None:
        print(
            "  Run scripts/run_ablation.py on the GPU machine. It writes\n"
            "  data/processed/perturbation_curves.parquet. Then run this\n"
            "  script again to get the verdicts table and the notebook."
        )
    else:
        print(
            "  Every stage ran. Read outputs/tables/verdicts.csv for the answer,\n"
            "  then open notebooks/capstone.ipynb and run it from the top."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ready stages.")
    parser.add_argument("--eda-only", action="store_true", help="stop after stage 3")
    parser.add_argument(
        "--through",
        type=int,
        default=LAST_STAGE,
        help=f"stop after this stage, 1 to {LAST_STAGE}",
    )
    main(**vars(parser.parse_args()))
