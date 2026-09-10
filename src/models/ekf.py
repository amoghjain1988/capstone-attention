"""Extended Kalman filter. A pipeline smoke test.

See CONTRACT.md section 5.5 for the signatures.

The state is x, y, vx, vy. The transition is constant velocity and the
measurement is the position. Both are linear, so the extended filter reduces
to the plain linear Kalman filter here. The name stays as CONTRACT.md section
4.2 gives it, and this note says plainly why no Jacobian appears below.

The filter reads one agent at a time. No neighbour enters the state, so the
model can never react to a mask and its shift is exactly 0 for every arm. That
is the same floor that src/models/cv.py gives, with a covariance attached.

The filter starts from the first two observed frames, runs over the 8 observed
frames, and then rolls the state forward for 12 frames with no measurement.
"""

from __future__ import annotations

import numpy as np

import config
from src.models import base


def transition_matrix(dt: float) -> np.ndarray:
    """Return the (4, 4) constant velocity transition for one frame."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def process_covariance(q: float, dt: float) -> np.ndarray:
    """Return the (4, 4) discrete white noise acceleration covariance.

    One unknown acceleration acts over one frame. Its variance is q. The
    acceleration adds dt squared over 2 to the position and dt to the speed,
    so the covariance of the pair is q times the outer product of those two
    numbers. The x axis and the y axis carry the same, independent noise.
    """
    position_gain = 0.5 * dt * dt
    speed_gain = dt
    gain = np.array([position_gain, speed_gain], dtype=np.float64)
    block = q * np.outer(gain, gain)  # (2, 2) over one axis: position, speed

    out = np.zeros((4, 4), dtype=np.float64)
    for axis in (0, 1):  # 0 is the x axis, 1 is the y axis
        position, speed = axis, axis + 2
        out[position, position] = block[0, 0]
        out[position, speed] = block[0, 1]
        out[speed, position] = block[1, 0]
        out[speed, speed] = block[1, 1]
    return out


class ExtendedKalman:
    """A constant velocity Kalman filter that obeys the section 4.1 protocol."""

    def __init__(
        self,
        dt: float = config.DT,
        q: float = config.EKF_PROCESS_NOISE,
        r: float = config.EKF_MEASUREMENT_NOISE,
        n_hist: int = config.N_HIST,
        n_fut: int = config.N_FUT,
    ) -> None:
        """Build the filter. K is 1 and the filter is exact.

        `q` is the variance of the unknown acceleration, in metres per second
        squared, squared. `r` is the variance of one observed coordinate, in
        metres squared. Both come from config.py.
        """
        self.dt = float(dt)
        self.q = float(q)
        self.r = float(r)
        self.n_hist = int(n_hist)
        self.n_fut = int(n_fut)
        if self.n_hist < 2:
            raise ValueError(f"n_hist must be at least 2, it is {self.n_hist}")

        self.transition = transition_matrix(self.dt)
        self.process_noise = process_covariance(self.q, self.dt)
        self.observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self.measurement_noise = self.r * np.eye(2, dtype=np.float64)

    # -- helpers ----------------------------------------------------------

    def _initial_state(self, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the state and the covariance at the FIRST observed frame.

        The position comes from frame 0. The speed comes from the first two
        frames. A first difference of two measurements carries twice the
        measurement variance, divided by the square of the frame length, so
        the initial covariance needs no new constant.
        """
        n_agents = int(hist.shape[0])
        velocity = (hist[:, 1, :] - hist[:, 0, :]) / self.dt

        state = np.empty((n_agents, 4), dtype=np.float64)
        state[:, 0:2] = hist[:, 0, :]
        state[:, 2:4] = velocity

        speed_variance = 2.0 * self.r / (self.dt * self.dt)
        covariance = np.diag(
            np.array([self.r, self.r, speed_variance, speed_variance])
        )
        return state, np.repeat(covariance[None, :, :], n_agents, axis=0)

    def _step(
        self, state: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one predict step and one update step for every agent."""
        transition = self.transition
        observation = self.observation

        # Predict.
        state = state @ transition.T
        covariance = transition @ covariance @ transition.T + self.process_noise

        # Update.
        innovation = measurement - state @ observation.T  # (N, 2)
        innovation_cov = (
            observation @ covariance @ observation.T + self.measurement_noise
        )  # (N, 2, 2)
        gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)  # (N, 4, 2)

        state = state + np.einsum("nij,nj->ni", gain, innovation)
        eye = np.eye(4, dtype=np.float64)
        covariance = (eye - gain @ observation) @ covariance
        return state, covariance

    def _filter(self, hist: np.ndarray) -> np.ndarray:
        """Return the (N, 4) state at the LAST observed frame."""
        state, covariance = self._initial_state(hist)
        for frame in range(1, self.n_hist):
            state, covariance = self._step(state, covariance, hist[:, frame, :])
        return state

    # -- the protocol -----------------------------------------------------

    def predict(
        self,
        hist: np.ndarray,
        edge_mask: np.ndarray | None = None,
        draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (1, N, 12, 2) float64 in metres.

        The arguments edge_mask and draws are accepted and then ignored, the
        same way src/models/cv.py ignores them. The filter reads one agent at
        a time, so no edge exists to remove, and the filter is deterministic,
        so every draw is the same path.
        """
        hist = base.check_history(hist, self.n_hist)
        n_agents = int(hist.shape[0])
        base.check_edge_mask(edge_mask, n_agents)

        state = self._filter(hist)
        path = np.empty((n_agents, self.n_fut, 2), dtype=np.float64)
        for step in range(self.n_fut):
            state = state @ self.transition.T
            path[:, step, :] = state[:, 0:2]
        return base.check_prediction(path[None, ...], n_agents, self.n_fut)
