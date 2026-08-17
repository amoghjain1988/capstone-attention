"""Constant velocity. The scale for the shift and a pipeline smoke test.

See CONTRACT.md section 5.5 for the signatures.

Constant velocity is a strong baseline on ETH and UCY. It ignores every
neighbour, so it can never react to a mask. Its shift is therefore exactly 0
for every arm, and that is the point. It gives the floor of the whole study.
"""

from __future__ import annotations

import numpy as np

import config
from src.models import base


class ConstantVelocity:
    """Walk on at the last observed velocity. K is 1 and the model is exact."""

    def __init__(
        self,
        dt: float = config.DT,
        n_hist: int = config.N_HIST,
        n_fut: int = config.N_FUT,
    ) -> None:
        self.dt = float(dt)
        self.n_hist = int(n_hist)
        self.n_fut = int(n_fut)

    def predict(
        self,
        hist: np.ndarray,
        edge_mask: np.ndarray | None = None,
        draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (1, N, 12, 2) float64 in metres.

        The arguments edge_mask and draws are accepted and then ignored. The
        model reads one agent at a time, so no edge exists to remove.
        """
        hist = base.check_history(hist, self.n_hist)
        n_agents = int(hist.shape[0])
        base.check_edge_mask(edge_mask, n_agents)

        position = hist[:, -1, :]
        velocity = (hist[:, -1, :] - hist[:, -2, :]) / self.dt
        steps = np.arange(1, self.n_fut + 1, dtype=np.float64) * self.dt
        path = position[:, None, :] + velocity[:, None, :] * steps[None, :, None]
        return base.check_prediction(path[None, ...], n_agents, self.n_fut)
