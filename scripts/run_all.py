"""Run every stage that is ready, in the order of CONTRACT.md section 7.

See CONTRACT.md section 5.1 for the signatures.

Stages 1 to 3 are ready. They need no GPU and no AgentFormer.

    1. scripts/download_ethucy.py
    2. src/data/, up to eligibility.py
    3. src/eda/, including overlap.py, which gives the design effect

Stages 4 and later wait for their branch. This script prints a clear note when
it reaches a stage that is not ready.

Every stage is idempotent. Run the script twice and nothing changes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.data import clean, eligibility, loader_ethucy, schema, windows  # noqa: E402
from src.eda import density, distributions, missing, overlap, scaling, summary  # noqa: E402


def banner(text: str) -> None:
    """Print a stage heading."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def save_table(frame: pd.DataFrame, name: str) -> Path:
    """Write one exploratory table to outputs/tables."""
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLE_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(config.ROOT)}  ({len(frame)} rows)")
    return path


def stage_data(drops: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the trajectories table and the windows table."""
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

    start = time.time()
    frame = windows.make_windows(trajectories, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE)
    frame["overlap_fraction"] = windows.overlap_fraction(frame, config.WINDOW_LEN)
    frame = eligibility.mark_eligible(frame, config.SEED)
    drops.append(eligibility.eligibility_log(frame))
    schema.write(frame, "windows", config.PROCESSED_DIR)
    print(
        f"  windows: {len(frame):,} built, {int(frame['eligible'].sum()):,} eligible"
        f"  ({time.time() - start:.1f} s)"
    )
    return trajectories, frame


def stage_eda(trajectories: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Run every exploratory module. Write one table and one figure each."""
    banner("STAGE 3  src/eda")

    counts = summary.summary(trajectories, frame)
    save_table(counts, "eda_summary")

    features = density.ego_neighbourhood(frame, trajectories)
    neighbours = density.density_report(frame, trajectories)
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

    sweep = overlap.overlap_report(frame, features=features)
    save_table(sweep, "eda_overlap")

    effect = overlap.design_effect(frame, features=features, unit="pedestrian")
    print("\n  design effect at the frozen stride:")
    for key, value in effect.items():
        print(f"    {key:20s} {value}")
    return effect


def stage_not_ready(number: int, name: str, owner: str) -> None:
    """Print a note for a stage that has no code yet."""
    print(f"\nSTAGE {number}  {name}: not ready. Owner {owner}. See CONTRACT.md section 7.")


def stage_predictions() -> None:
    """Build data/processed/predictions.parquet and outputs/tables/calibration.csv.

    This needs the GPU cache under data/interim (scripts/extract_attention.py's
    output), not a GPU itself. Print a clear note and return when that cache
    is not on disk yet, instead of failing the whole run.
    """
    banner("STAGE 8  src/metrics (predictions and calibration)")
    if not any(config.INTERIM_DIR.glob("attention_*.parquet")):
        print(
            f"  no cache under {config.INTERIM_DIR.relative_to(config.ROOT)}. "
            "Run scripts/extract_attention.py on a GPU machine first, or copy "
            "the cache folder in, then re-run this stage."
        )
        return

    from src.metrics import calibration
    from src.models.cached import CachedPredictor

    cache = CachedPredictor()
    print(f"  {cache}")

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


def main(eda_only: bool = False) -> None:
    """Run the pipeline."""
    config.make_dirs()
    print(f"run_id config hash: {config.config_hash()}")

    banner("STAGE 1  scripts/download_ethucy.py")
    manifest = config.RAW_DIR / "manifest.json"
    if not manifest.exists():
        raise SystemExit("Run scripts/download_ethucy.py first. The manifest is missing.")
    print(f"  manifest present: {manifest.relative_to(config.ROOT)}")

    drops: list[pd.DataFrame] = []
    trajectories, frame = stage_data(drops)
    effect = stage_eda(trajectories, frame)

    log = pd.concat([d for d in drops if len(d)], ignore_index=True) if drops else schema.empty_drops()
    if not len(log):
        log = schema.empty_drops()
    save_table(log, "drops")

    if eda_only:
        return

    for number, name, owner in (
        (4, "src/features", "P2"),
        (5, "config freeze in src/hypotheses/design.py", "P4"),
        (6, "scripts/extract_attention.py and src/attention", "Amogh"),
        (7, "src/ablation", "Amogh"),
    ):
        stage_not_ready(number, name, owner)

    stage_predictions()
    stage_not_ready(8, "src/faithfulness (the rest of stage 8)", "P3")

    for number, name, owner in (
        (9, "src/hypotheses and src/stats", "P4"),
        (10, "notebooks/capstone.ipynb", "all"),
    ):
        stage_not_ready(number, name, owner)

    banner("NEXT ACTION")
    print(
        "  Put deff and n_effective into src/hypotheses/design.py, choose\n"
        "  MINIMUM_EFFECT and POWER, fill them into config.py, then hash and\n"
        f"  commit. The current design effect is {effect['deff']:.3f} and the\n"
        f"  effective sample is {effect['n_effective']:.0f} windows."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ready stages.")
    parser.add_argument("--eda-only", action="store_true", help="stop after stage 3")
    main(**vars(parser.parse_args()))
