"""Does any predicted pair touch.

See CONTRACT.md section 5.8 for the signatures.
"""

from __future__ import annotations

import numpy as np

import config


def has_collision(paths: np.ndarray, radius: float = config.COLLISION_RADIUS_M) -> bool:
    """Return True when any pair of distinct agents comes closer than radius.

    paths is (N, 12, 2). The check runs at every one of the 12 steps. The
    self pair, agent against itself, is always ignored.
    """
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"paths must be (N, T, 2), it is {array.shape}")

    n_agents = array.shape[0]
    if n_agents < 2:
        return False

    delta = array[:, None, :, :] - array[None, :, :, :]  # (N, N, T, 2)
    distance = np.sqrt((delta * delta).sum(axis=-1))  # (N, N, T)
    off_diagonal = ~np.eye(n_agents, dtype=bool)[:, :, None]
    return bool(((distance < radius) & off_diagonal).any())
