"""The gate. AUROC against a graph we planted ourselves.

See GRACE.md and GRACE_AGENT.md Task 5. If tests/test_synthetic.py cannot
clear an AUROC of 0.9 here, no number from the real data means anything.
"""

from __future__ import annotations

import numpy as np

import config
from src.data import synthetic
from src.models import base


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Return the 1-indexed rank of every value. Tied values share the mean rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc_against_truth(scores: np.ndarray, is_real: np.ndarray) -> float:
    """Return the AUROC of `scores` against the planted truth `is_real`.

    `scores` is the attention weight or the measured shift, one value per
    edge. `is_real` is True where the edge is a planted influencer. Ties in
    `scores` share the average rank. This uses the Mann-Whitney U relation to
    AUROC. It is plain numpy: no new dependency.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_real = np.asarray(is_real, dtype=bool)
    if scores.shape != is_real.shape:
        raise ValueError(
            f"scores and is_real must match shapes: {scores.shape} vs {is_real.shape}"
        )

    n_pos = int(is_real.sum())
    n_neg = int((~is_real).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc_against_truth needs at least one real and one fake edge")

    ranks = _average_rank(scores)
    sum_ranks_pos = float(ranks[is_real].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class PlantedPredictor:
    """A predictor whose physics obeys ONE planted influence matrix.

    Reproduces src/data/synthetic.py's exact force law (the same REPULSION,
    RELAX and MIN_RANGE_M constants), so a masked edge changes the forecast
    only when that edge is really on the influence list. This is the
    "predictor that obeys the planted graph" that GRACE_AGENT.md Task 5 calls
    for. It lives here and not in src/data/synthetic.py, because that module
    is frozen and owned by Amogh.

    Deterministic: draws changes nothing, matching the released AgentFormer.
    """

    # Calibrated for the attention() method below, not for _simulate. See
    # that method's docstring for what these two numbers are doing and why
    # neither one is allowed to make attention a perfect oracle.
    ATTENTION_TEMPERATURE = 1.0
    INFLUENCE_BOOST = 64.0

    def __init__(
        self,
        influence: np.ndarray,
        dt: float = config.DT,
        n_fut: int = config.N_FUT,
    ) -> None:
        self.influence = np.asarray(influence, dtype=bool)
        self.dt = float(dt)
        self.n_fut = int(n_fut)

    def predict(
        self,
        hist: np.ndarray,
        edge_mask: np.ndarray | None = None,
        draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (K, N, n_fut, 2). Every draw is identical, on purpose."""
        n_hist = int(np.asarray(hist).shape[1])
        hist = base.check_history(hist, n_hist)
        n_agents = int(hist.shape[0])
        base.check_edge_mask(edge_mask, n_agents)

        if edge_mask is None:
            keep = np.ones((n_agents, n_agents), dtype=bool)
        else:
            keep = np.asarray(edge_mask, dtype=bool)
        active = keep & self.influence
        np.fill_diagonal(active, False)

        path = self._simulate(hist, active)

        n_draws = 1 if draws is None else int(np.asarray(draws).size)
        out = np.repeat(path[None, :, :, :], n_draws, axis=0)
        return base.check_prediction(out, n_agents, self.n_fut)

    def _simulate(self, hist: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Roll the planted physics forward n_fut steps. Deterministic.

        `desired` is the velocity read off the last two observed frames, held
        fixed for the whole forecast. It stands in for the goal-directed term
        of src/data/synthetic.py's generator, which is not recoverable from
        `hist` alone. `desired` is identical for a masked and an unmasked
        call, so it cancels out of the shift: only `active` decides whether a
        removed edge moves the ego.
        """
        n_agents = hist.shape[0]
        position = hist[:, -1, :].copy()
        desired = (hist[:, -1, :] - hist[:, -2, :]) / self.dt
        velocity = desired.copy()

        path = np.empty((n_agents, self.n_fut, 2), dtype=np.float64)
        for step in range(self.n_fut):
            delta = position[:, None, :] - position[None, :, :]
            distance = np.linalg.norm(delta, axis=-1)
            distance = np.clip(distance, synthetic.MIN_RANGE_M, None)
            push = delta / (distance**3)[:, :, None]
            push = np.where(active[:, :, None], push, 0.0)
            force = synthetic.REPULSION * push.sum(axis=1)

            velocity = velocity + (synthetic.RELAX * (desired - velocity) + force) * self.dt
            position = position + velocity * self.dt
            path[:, step, :] = position
        return path

    def attention(self, hist: np.ndarray) -> dict[str, np.ndarray]:
        """Return module -> (N*n_hist, N*n_hist). Every row sums to 1.

        PlantedPredictor has no attention of its own: `predict` only ever
        consults `self.influence`, gated hard, inside `_simulate`. This
        method exists only because tests/test_synthetic.py needs a genuine
        attention-shaped signal to carry through
        src/attention/aggregate.py `collapse()` and
        src/attention/rank.py `rank_edges()`, the same two functions the real
        pipeline calls on AgentFormer's own attention. Without a real
        `attention()` here, that pipeline could never be exercised by the
        gate at all.

        The weight starts from the same geometric rule `src/models/mock.py`
        uses for `MockPredictor`: it falls with distance at the last
        observed frame. On its own this rule is USELESS here, because
        `src/data/synthetic.py` plants real influence uniformly at random,
        independent of geometry: a purely geometric score against this scene
        scores close to a coin flip. A real edge therefore adds a fixed
        bonus, `INFLUENCE_BOOST`, to that neighbour's logit before the
        softmax.

        `INFLUENCE_BOOST` is calibrated so real influence usually beats
        ordinary distance noise, but not always: at short range a nearby
        fake neighbour can still outrank a distant real one. This
        imperfection is deliberate, not a defect to remove. A signal that
        separated real from fake edges with certainty would prove nothing
        about the gate, for the same reason the shift-based version this
        method replaces proved nothing: either one would pass however
        `collapse` and `rank_edges` wired the ids together, even wired
        backwards. `tests/test_synthetic.py::test_a_reversed_ranking_fails_the_gate`
        is the test that checks this method's signal is not that trivial.
        """
        n_hist = int(np.asarray(hist).shape[1])
        hist = base.check_history(hist, n_hist)
        n_agents = int(hist.shape[0])

        delta = hist[:, -1, None, :] - hist[None, :, -1, :]
        distance = np.linalg.norm(delta, axis=-1)
        distance = np.clip(distance, synthetic.MIN_RANGE_M, None)

        logit = -distance / self.ATTENTION_TEMPERATURE
        logit = logit + np.where(self.influence, self.INFLUENCE_BOOST, 0.0)
        np.fill_diagonal(logit, -np.inf)
        weight = np.exp(logit - logit.max(axis=1, keepdims=True))
        weight = weight / weight.sum(axis=1, keepdims=True)

        # Spread each agent pair evenly over its own n_hist by n_hist time
        # block, so the token map is agent-major and every row still sums to
        # 1, matching MockPredictor.attention.
        token = np.repeat(np.repeat(weight, n_hist, axis=0), n_hist, axis=1) / n_hist
        return {name: token.copy() for name in config.MODULES}
