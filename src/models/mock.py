"""A fake predictor so the pipeline runs on day one.

See CONTRACT.md section 5.5 for the signatures.

MockPredictor satisfies MaskablePredictor. It really honours edge_mask and it
really returns an attention map, so the whole ablation loop can be built and
tested against it with no GPU and no AgentFormer.

The rule is simple and it is on purpose. An agent walks on at its last observed
velocity, and every KEPT neighbour pushes it away. Remove an edge and the push
disappears, so the path moves. A closer neighbour carries more attention and
also pushes harder, so a faithful ranking is the distance ranking. That gives
every test a known right answer.
"""

from __future__ import annotations

import numpy as np

import config
from src.models import base

PUSH = 1.2  # metres per second squared at 1 metre
MIN_RANGE_M = 0.3
TEMPERATURE = 1.0


class MockPredictor:
    """A social force toy that obeys the section 4.1 protocol."""

    def __init__(
        self,
        seed: int = config.SEED,
        n_samples: int = config.N_SAMPLES,
        dt: float = config.DT,
        n_hist: int = config.N_HIST,
        n_fut: int = config.N_FUT,
        noise_m: float = 0.0,
    ) -> None:
        """Build the predictor.

        Set noise_m above 0 to make the draws differ. The default of 0 keeps
        the predictor deterministic, which matches the released AgentFormer.
        """
        self.seed = int(seed)
        self.n_samples = int(n_samples)
        self.dt = float(dt)
        self.n_hist = int(n_hist)
        self.n_fut = int(n_fut)
        self.noise_m = float(noise_m)

    # -- helpers ----------------------------------------------------------

    def _last_state(self, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the last observed position and velocity of every agent."""
        position = hist[:, -1, :].copy()
        velocity = (hist[:, -1, :] - hist[:, -2, :]) / self.dt
        return position, velocity

    def _keep_matrix(self, edge_mask: np.ndarray | None, n_agents: int) -> np.ndarray:
        """Return the (N, N) float matrix that switches every edge on or off."""
        if edge_mask is None:
            keep = np.ones((n_agents, n_agents), dtype=np.float64)
        else:
            keep = np.asarray(edge_mask, dtype=bool).astype(np.float64)
        np.fill_diagonal(keep, 0.0)
        return keep

    # -- the protocol -----------------------------------------------------

    def predict(
        self,
        hist: np.ndarray,
        edge_mask: np.ndarray | None = None,
        draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (K, N, 12, 2) float64 in metres."""
        hist = base.check_history(hist, self.n_hist)
        n_agents = int(hist.shape[0])
        base.check_edge_mask(edge_mask, n_agents)
        keep = self._keep_matrix(edge_mask, n_agents)

        if draws is None:
            seeds = np.arange(self.n_samples, dtype=np.int64) + self.seed
        else:
            seeds = np.asarray(draws, dtype=np.int64)

        out = np.empty((seeds.size, n_agents, self.n_fut, 2), dtype=np.float64)
        for index, seed in enumerate(seeds):
            out[index] = self._one_draw(hist, keep, int(seed))
        return base.check_prediction(out, n_agents, self.n_fut)

    def _one_draw(self, hist: np.ndarray, keep: np.ndarray, seed: int) -> np.ndarray:
        """Roll one path forward. The push comes only from the kept edges."""
        generator = np.random.default_rng(seed)
        position, velocity = self._last_state(hist)
        path = np.empty((hist.shape[0], self.n_fut, 2), dtype=np.float64)

        for step in range(self.n_fut):
            delta = position[:, None, :] - position[None, :, :]
            distance = np.linalg.norm(delta, axis=-1)
            distance = np.clip(distance, MIN_RANGE_M, None)
            force = (delta / (distance**2)[:, :, None]) * keep[:, :, None]
            velocity = velocity + PUSH * force.sum(axis=1) * self.dt
            position = position + velocity * self.dt
            if self.noise_m > 0.0:
                position = position + generator.normal(0.0, self.noise_m, position.shape)
            path[:, step, :] = position
        return path

    def attention(self, hist: np.ndarray) -> dict[str, np.ndarray]:
        """Return module -> (N*8, N*8). Every row sums to 1.

        The weight falls with the distance at the last observed frame, so the
        attention ranking equals the distance ranking. A test can therefore
        check that the pipeline recovers the ranking it was given.
        """
        hist = base.check_history(hist, self.n_hist)
        n_agents, n_time = int(hist.shape[0]), self.n_hist

        delta = hist[:, -1, None, :] - hist[None, :, -1, :]
        distance = np.linalg.norm(delta, axis=-1)
        distance = np.clip(distance, MIN_RANGE_M, None)

        logit = -distance / TEMPERATURE
        np.fill_diagonal(logit, -np.inf)  # an agent does not attend to itself
        weight = np.exp(logit - logit.max(axis=1, keepdims=True))
        weight = weight / weight.sum(axis=1, keepdims=True)

        # Spread each agent pair evenly over its own 8 by 8 time block, so the
        # token map is agent-major and every row still sums to 1.
        token = np.repeat(np.repeat(weight, n_time, axis=0), n_time, axis=1) / n_time
        return {name: token.copy() for name in config.MODULES}
