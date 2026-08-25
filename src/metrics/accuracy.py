"""ADE, FDE, and the best of the K draws.

See CONTRACT.md section 5.8 for the signatures.
"""

from __future__ import annotations

import numpy as np


def _step_distance(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the euclidean distance at every step, for every draw. (K, T)."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"pred must be (K, T, 2), it is {pred.shape}")
    if truth.shape != pred.shape[1:]:
        raise ValueError(f"truth must be {pred.shape[1:]}, it is {truth.shape}")
    return np.linalg.norm(pred - truth[None, :, :], axis=-1)


def ade(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return the mean displacement over the 12 steps, then over the K draws."""
    return float(_step_distance(pred, truth).mean(axis=1).mean())


def fde(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return the mean displacement at the final step, over the K draws."""
    return float(_step_distance(pred, truth)[:, -1].mean())


def min_ade(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return the ADE of the single best draw."""
    return float(_step_distance(pred, truth).mean(axis=1).min())


def min_fde(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return the FDE of the single best draw."""
    return float(_step_distance(pred, truth)[:, -1].min())
