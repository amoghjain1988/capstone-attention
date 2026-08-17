"""How far the EGO forecast moved.

See CONTRACT.md section 5.7 for the signatures.

The reference is the UNMASKED prediction, never the ground truth. We measure a
change in the model, not an error against the world. A window where the model
is wrong twice still gives a shift of 0 if the mask changed nothing.
"""

from __future__ import annotations

import numpy as np


def shift(path_masked: np.ndarray, path_unmasked: np.ndarray) -> float:
    """Return the mean displacement of the ego path, in metres.

    Both inputs are (K, 12, 2) for the EGO ONLY, over the SAME draws. The
    function takes the euclidean distance at every one of the 12 steps, then
    the mean over the steps, then the mean over the K draws.
    """
    masked = np.asarray(path_masked, dtype=np.float64)
    unmasked = np.asarray(path_unmasked, dtype=np.float64)
    if masked.shape != unmasked.shape:
        raise ValueError(
            f"the two paths must match: {masked.shape} against {unmasked.shape}"
        )
    if masked.ndim != 3 or masked.shape[-1] != 2:
        raise ValueError(f"a path must be (K, T, 2), it is {masked.shape}")

    step_distance = np.linalg.norm(masked - unmasked, axis=-1)  # (K, T)
    return float(step_distance.mean(axis=1).mean())


def ego_path(prediction: np.ndarray, ego_index: int) -> np.ndarray:
    """Return the ego rows of a (K, N, 12, 2) prediction, shape (K, 12, 2)."""
    array = np.asarray(prediction, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"a prediction must be (K, N, T, 2), it is {array.shape}")
    return array[:, int(ego_index), :, :]
