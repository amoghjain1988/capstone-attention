"""Run the ablation and write the perturbation_curves table.

See CONTRACT.md section 5.1 and docs/FINISH_PLAN.md section 3.10.

The script reads the frozen windows table and the attention_edges table, runs
every arm of every eligible window, and writes

    data/interim/curves_<scene>.parquet     the per-scene checkpoint
    data/processed/perturbation_curves.parquet
    outputs/tables/ablation_sanity.csv      zero_check and noise_floor
    outputs/tables/ablation_mismatch.csv    only when the audit finds a row

The script is idempotent. It records which windows it finished and skips them
on a second run. Pass --scene to do one scene, --limit to do a few windows,
and --force to ignore the checkpoint.

Pass --model mock for a CPU smoke run against MockPredictor. MockPredictor
does not honour a second mask policy, so that run writes weight_zero rows for
no window. The default --model agentformer needs a CUDA device.
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
from src.ablation import driver  # noqa: E402
from src.attention import edges  # noqa: E402
from src.data import schema  # noqa: E402
from src.faithfulness import sanity  # noqa: E402

SANITY_COLUMNS = (
    "scene",
    "zero_check",
    "noise_floor",
    "n_windows",
    "n_forward_passes_estimate",
)


def short(path: Path) -> str:
    """Return the path under the repository root, or the whole path."""
    try:
        return str(Path(path).relative_to(config.ROOT))
    except ValueError:
        return str(path)


def build_model(name: str, scene: str):
    """Return the predictor of one scene.

    The AgentFormer import stays inside the function, the same way
    scripts/extract_attention.py does it, so a CPU run with --model mock never
    touches torch and never loads a checkpoint.
    """
    if name == "mock":
        from src.models.mock import MockPredictor

        return MockPredictor()
    if name == "agentformer":
        from src.models.agentformer import AgentFormerPredictor

        return AgentFormerPredictor(scene, device="cuda", n_samples=config.N_SAMPLES)
    raise SystemExit(f"{name!r} is not a model. Choose mock or agentformer.")


def scene_block(
    windows: pd.DataFrame, scene: str, limit: int | None
) -> pd.DataFrame:
    """Return the eligible windows of one scene, in t0 order."""
    block = windows.loc[
        (windows["scene"].astype(str) == str(scene)) & windows["eligible"]
    ]
    block = block.sort_values("t0", kind="mergesort")
    if limit:
        block = block.head(int(limit))
    return block


def sanity_row(model, block: pd.DataFrame, edges_primary: pd.DataFrame) -> dict:
    """Return one row of outputs/tables/ablation_sanity.csv.

    zero_check and noise_floor run on the FIRST window of the scene. Both run
    against the raw model, never against the baseline wrapper of
    src/ablation/driver.py, so zero_check stays a true test.
    """
    first = block.iloc[0]
    hist = schema.unpack_hist(first, config.N_HIST)
    draws = np.arange(config.N_SAMPLES, dtype=np.int64) + config.SEED
    return {
        "scene": str(first["scene"]),
        "zero_check": float(sanity.zero_check(model, hist, draws)),
        "noise_floor": float(sanity.noise_floor(model, hist, 3, config.SEED)),
        "n_windows": int(len(block)),
        "n_forward_passes_estimate": driver.scene_forward_passes(block, edges_primary),
    }


def finalise(edges_primary: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every checkpoint, audit it, and write the table.

    The function reads every data/interim/curves_*.parquet file, not only the
    scene that just finished, so data/processed/perturbation_curves.parquet
    always holds everything that is on disk.
    """
    parts = []
    for path in sorted(config.INTERIM_DIR.glob("curves_*.parquet")):
        part = driver.read_checkpoint(path)
        if len(part):
            parts.append(part)
    if not parts:
        print("  no checkpoint on disk, so no table is written")
        return driver.empty_curves()

    curves = pd.concat(parts, ignore_index=True)
    mismatch = driver.consistency_check(curves, edges_primary)
    print(f"  consistency check: {len(mismatch)} mismatched rows")

    path = schema.write(curves, "perturbation_curves", config.PROCESSED_DIR)
    print(f"  wrote {short(path)}  ({len(curves):,} rows)")

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mismatch_path = config.TABLE_DIR / "ablation_mismatch.csv"
    if len(mismatch):
        mismatch.to_csv(mismatch_path, index=False)
        print(f"  wrote {short(mismatch_path)}")
    elif mismatch_path.exists():
        mismatch_path.unlink()  # remove the stale file of an earlier run
    return curves


def main(
    scene: str | None = None,
    limit: int | None = None,
    force: bool = False,
    model: str = "agentformer",
) -> None:
    """Run every scene, or the one that was named."""
    config.make_dirs()
    print(f"run_id config hash: {config.config_hash()}")

    for table in ("windows", "attention_edges"):
        if not schema.path_of(table, config.PROCESSED_DIR).exists():
            raise SystemExit(
                f"{schema.path_of(table, config.PROCESSED_DIR)} is missing. "
                "Run scripts/run_all.py stage 6 first. That stage joins the "
                "cache under data/interim to the pair features with "
                "src/attention/edges.py::build_table."
            )

    windows = schema.read("windows", config.PROCESSED_DIR)
    attention = schema.read("attention_edges", config.PROCESSED_DIR)
    edges_primary = edges.primary(attention)
    print(
        f"windows: {len(windows):,} rows, "
        f"{int(windows['eligible'].sum()):,} eligible. "
        f"primary edges: {len(edges_primary):,} rows"
    )

    scenes = [scene] if scene else list(config.SCENES)
    sanity_rows: list[dict] = []

    for name in scenes:
        if name not in config.SCENES:
            raise SystemExit(f"{name!r} is not a scene. Choose from {config.SCENES}.")

        block = scene_block(windows, name, limit)
        if block.empty:
            print(f"\n{name}: no eligible window")
            continue

        predictor = build_model(model, name)
        print(f"\n{name}: {predictor}")

        row = sanity_row(predictor, block, edges_primary)
        sanity_rows.append(row)
        print(
            f"  zero_check {row['zero_check']:.3e}  "
            f"noise_floor {row['noise_floor']:.3e}  "
            f"{row['n_forward_passes_estimate']:,} forward passes estimated"
        )

        start = time.time()
        driver.run_scene(
            name,
            windows,
            edges_primary,
            predictor,
            driver.scene_checkpoint(name),
            limit=limit,
            force=force,
            seed=config.SEED,
            verbose=True,
        )
        print(f"  {name} took {(time.time() - start) / 60:.1f} min")
        finalise(edges_primary)

    if sanity_rows:
        config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        sanity_path = config.TABLE_DIR / "ablation_sanity.csv"
        pd.DataFrame(sanity_rows, columns=list(SANITY_COLUMNS)).to_csv(
            sanity_path, index=False
        )
        print(f"\nwrote {short(sanity_path)}")

    for skipped_model, policy in driver.policy_skips():
        print(f"NOTE {skipped_model} skipped the mask policy {policy!r}")

    print("\nThe ablation is done. src/hypotheses/ can now run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ablation.")
    parser.add_argument("--scene", default=None, help="one scene, or every scene")
    parser.add_argument("--limit", type=int, default=None, help="only the first N windows")
    parser.add_argument("--force", action="store_true", help="ignore the checkpoint")
    parser.add_argument(
        "--model",
        default="agentformer",
        choices=("agentformer", "mock"),
        help="agentformer needs a GPU. mock is the CPU smoke run",
    )
    main(**vars(parser.parse_args()))
