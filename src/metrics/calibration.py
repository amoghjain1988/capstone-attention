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

DETERMINISM LIMIT. AgentFormer is deterministic. The K draws are K fixed
DLow modes, not K samples from a posterior. Every number in this module
therefore describes how the K FIXED modes sit around the truth, not a
probabilistic calibration claim in the usual sense: there is no experiment
in which the model draws a different set of K paths. Read `coverage` and
`pit` as a diagnostic on mode spread, and state this limit beside every
number this module reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema
from src.metrics import accuracy, collision

# K draws give K + 1 possible ranks (a truth can sit anywhere from closest
# to farthest among the draws, inclusive). A bin count below K + 1 packs a
# different count of ranks into different bins, so even a perfectly uniform
# rank vector shows a sawtooth. Use K + 1 bins so every rank owns exactly
# one bin.
N_PIT_BINS = config.N_SAMPLES + 1


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

    Uses the rank among K + 1 method, not the raw quantile of the K draws.
    The K draws and the truth are K + 1 exchangeable points under a
    calibrated ensemble, so this function ranks the truth's distance from
    the centroid among the K draws' distances. The result is an integer
    rank in 0 .. K. The step counts as covered when that rank falls in the
    smallest `level` share of the K + 1 slots. The raw quantile of the K
    draws alone, with no K + 1 correction, is biased at small K: a
    perfectly calibrated K = 20 ensemble scores 0.833 against a nominal
    level of 0.90 under that estimator, not 0.90.
    """
    draw_dist, truth_dist = _step_stats(pred, truth)
    n_draws = draw_dist.shape[0]
    rank = (draw_dist <= truth_dist[None, :]).sum(axis=0)  # (T,), 0 .. K
    threshold = level * (n_draws + 1)
    return float((rank < threshold).mean())


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


def collision_share(paths: np.ndarray, radius: float = config.COLLISION_RADIUS_M) -> float:
    """Return the share of the K draws that hold a collision.

    `paths` is (K, N, 12, 2), K stored futures for the same window. A single
    draw only says whether that ONE sampled future collides.
    `any(has_collision(draw) for draw in paths)` only rises with K and
    describes the draw count, not the model. This function reports the
    fraction of the K draws that collide instead, so the number stays
    meaningful as K changes.
    """
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"paths must be (K, N, T, 2), it is {array.shape}")
    hits = [collision.has_collision(array[k], radius) for k in range(array.shape[0])]
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# The cache-backed tables
# ---------------------------------------------------------------------------


def build_predictions(cache) -> pd.DataFrame:
    """Build and write data/processed/predictions.parquet from a CachedPredictor.

    Columns: window_id, ade, fde, minade_20, minfde_20, collision,
    collision_share. Reads every window the cache actually holds a
    prediction for.

    `collision` is a bool, per CONTRACT.md section 3: True when
    `collision_share` (the fraction of the K draws that hold a collision)
    reaches config.COLLISION_SHARE_THRESHOLD. `collision_share` is the float
    this table actually trusts; the bool exists to satisfy the contract
    column and loses information the float keeps. This split was raised with
    the team before it shipped; confirm it still holds before the bool alone
    appears in a report. See the determinism limit in the module docstring:
    K here is K fixed modes, not K samples, so neither column is a
    probability.
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

        share = collision_share(pred_all, config.COLLISION_RADIUS_M)
        rows.append(
            {
                "window_id": window_id,
                "ade": accuracy.ade(pred_ego, truth_ego),
                "fde": accuracy.fde(pred_ego, truth_ego),
                "minade_20": accuracy.min_ade(pred_ego, truth_ego),
                "minfde_20": accuracy.min_fde(pred_ego, truth_ego),
                "collision": share >= config.COLLISION_SHARE_THRESHOLD,
                "collision_share": share,
            }
        )

    table = pd.DataFrame(
        rows,
        columns=[
            "window_id", "ade", "fde", "minade_20", "minfde_20",
            "collision", "collision_share",
        ],
    )
    schema.write(table, "predictions", config.PROCESSED_DIR)
    return table


def pit_histogram(pit_values: np.ndarray, n_bins: int = N_PIT_BINS) -> np.ndarray:
    """Return the count of pit_values in each of n_bins equal-width bins on [0, 1].

    pit() returns one of K + 1 exact values (0/K, 1/K, .. K/K), so n_bins
    must be K + 1 for every rank to land in its own bin. A smaller bin count
    packs a different number of ranks into different bins, which shows up as
    a sawtooth even when the ranks are perfectly uniform.
    """
    counts, _ = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))
    return counts


def calibration_table(cache, level: float = 0.90) -> pd.DataFrame:
    """Build the per-scene coverage and PIT histogram, and write it to
    outputs/tables/calibration.csv.

    Columns: scene, level, n_windows, coverage, pit_bin_0 .. pit_bin_K. Each
    pit_bin column is the count of (window, step) pairs whose PIT value fell
    in that bin, pooled over every window of the scene. See the determinism
    limit in the module docstring: the K draws are K fixed modes, so
    `coverage` and the PIT histogram describe mode spread around the truth,
    not a probabilistic calibration claim.
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
        counts = pit_histogram(pit_values)
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
