"""Coverage and the PIT histogram, per scene.

See CONTRACT.md section 5.8 for the signatures.

This module does NOT implement expected calibration error. ECE bins a
classifier's confidence against a binary correct/incorrect outcome. We
forecast a continuous path, not a class label, so there is no confidence
score and no correct/incorrect outcome to bin. Coverage and the probability
integral transform (PIT) are the calibration tools that apply to a set of K
draws around a continuous truth, and CONTRACT.md section 5.8 names them for
that reason.

Both functions treat one time step of one ego as a small ensemble of K
points around a centroid. The centroid is the mean of the K draws at that
step. coverage() and pit() both start from the same two numbers: how far
each draw sits from the centroid, and how far the truth sits from the
centroid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema
from src.metrics import accuracy, collision

N_PIT_BINS = 10


def _step_stats(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (draw_dist, truth_dist), each shape (T,) or (K, T).

    draw_dist[k, t] is the distance of draw k from the step-t centroid.
    truth_dist[t] is the distance of the truth from the same centroid.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"pred must be (K, T, 2), it is {pred.shape}")
    if truth.shape != pred.shape[1:]:
        raise ValueError(f"truth must be {pred.shape[1:]}, it is {truth.shape}")

    centroid = pred.mean(axis=0)  # (T, 2)
    draw_dist = np.linalg.norm(pred - centroid[None, :, :], axis=-1)  # (K, T)
    truth_dist = np.linalg.norm(truth - centroid, axis=-1)  # (T,)
    return draw_dist, truth_dist


def coverage(pred: np.ndarray, truth: np.ndarray, level: float = 0.90) -> float:
    """Return the share of the 12 steps where the truth falls inside the
    `level` region of the K draws.

    The region at one step is the disc centred on that step's centroid, with
    a radius equal to the `level` quantile of the K draws' distance from the
    centroid.
    """
    draw_dist, truth_dist = _step_stats(pred, truth)
    radius = np.quantile(draw_dist, level, axis=0)  # (T,)
    return float((truth_dist <= radius).mean())


def pit(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return one probability integral transform value per step, shape (T,).

    The value at step t is the share of the K draws whose distance from the
    centroid is no larger than the truth's distance from the centroid. A
    well calibrated ensemble gives PIT values that are uniform on [0, 1]
    across many windows.
    """
    draw_dist, truth_dist = _step_stats(pred, truth)
    n_draws = draw_dist.shape[0]
    rank = (draw_dist <= truth_dist[None, :]).sum(axis=0)  # (T,)
    return rank.astype(np.float64) / n_draws


# ---------------------------------------------------------------------------
# The cache-backed tables
# ---------------------------------------------------------------------------


def build_predictions(cache) -> pd.DataFrame:
    """Build and write data/processed/predictions.parquet from a CachedPredictor.

    Columns: window_id, ade, fde, minade_20, minfde_20, collision. Reads
    every window the cache actually holds a prediction for.
    """
    rows: list[dict] = []
    for _, window_row in cache.windows.iterrows():
        window_id = str(window_row["window_id"])
        if window_id not in cache.predictions:
            continue

        hist = schema.unpack_hist(window_row, cache.n_hist)
        truth_all = schema.unpack_fut(window_row, cache.n_fut)
        ego_index = schema.ego_index(window_row)

        pred_all = cache.predict(hist)  # (K, N, 12, 2)
        pred_ego = pred_all[:, ego_index, :, :]
        truth_ego = truth_all[ego_index]

        rows.append(
            {
                "window_id": window_id,
                "ade": accuracy.ade(pred_ego, truth_ego),
                "fde": accuracy.fde(pred_ego, truth_ego),
                "minade_20": accuracy.min_ade(pred_ego, truth_ego),
                "minfde_20": accuracy.min_fde(pred_ego, truth_ego),
                "collision": any(
                    collision.has_collision(pred_all[k], config.COLLISION_RADIUS_M)
                    for k in range(pred_all.shape[0])
                ),
            }
        )

    table = pd.DataFrame(
        rows,
        columns=["window_id", "ade", "fde", "minade_20", "minfde_20", "collision"],
    )
    schema.write(table, "predictions", config.PROCESSED_DIR)
    return table


def calibration_table(cache, level: float = 0.90) -> pd.DataFrame:
    """Build the per-scene coverage and PIT histogram, and write it to
    outputs/tables/calibration.csv.

    Columns: scene, level, n_windows, coverage, pit_bin_0 .. pit_bin_9. Each
    pit_bin column is the count of (window, step) pairs whose PIT value fell
    in that tenth of [0, 1], pooled over every window of the scene.
    """
    per_scene_coverage: dict[str, list[float]] = {}
    per_scene_pit: dict[str, list[np.ndarray]] = {}

    for _, window_row in cache.windows.iterrows():
        window_id = str(window_row["window_id"])
        if window_id not in cache.predictions:
            continue

        hist = schema.unpack_hist(window_row, cache.n_hist)
        truth_all = schema.unpack_fut(window_row, cache.n_fut)
        ego_index = schema.ego_index(window_row)

        pred_ego = cache.predict(hist)[:, ego_index, :, :]
        truth_ego = truth_all[ego_index]

        scene = str(window_row["scene"])
        per_scene_coverage.setdefault(scene, []).append(coverage(pred_ego, truth_ego, level))
        per_scene_pit.setdefault(scene, []).append(pit(pred_ego, truth_ego))

    rows = []
    for scene in sorted(per_scene_coverage):
        pit_values = np.concatenate(per_scene_pit[scene])
        counts, _ = np.histogram(pit_values, bins=N_PIT_BINS, range=(0.0, 1.0))
        row = {
            "scene": scene,
            "level": level,
            "n_windows": len(per_scene_coverage[scene]),
            "coverage": float(np.mean(per_scene_coverage[scene])),
        }
        row.update({f"pit_bin_{i}": int(count) for i, count in enumerate(counts)})
        rows.append(row)

    table = pd.DataFrame(rows)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLE_DIR / "calibration.csv", index=False)
    return table
