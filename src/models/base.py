"""The three protocols. This is the swap seam.

See CONTRACT.md section 4.1 for the signatures.

Nobody imports AgentFormer outside src/models/agentformer.py. Every other
module takes a Predictor and does not care which one it got.

TOKEN ORDER. CONTRACT.md section 4.1 says that the token order is agent-major,
that is index = agent_index * T + time_index. AgentFormer is natively
time-major, that is index = time_index * N + agent_index. See the note in
src/models/agentformer.py. AgentFormerPredictor.attention() transposes the map
before it returns, so every caller sees the agent-major order that the contract
promises. Use to_agent_major() if you ever hold a raw AgentFormer map.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Predictor(Protocol):
    """Forecast 12 future steps for every agent of a window."""

    def predict(
        self,
        hist: np.ndarray,  # (N, 8, 2) float64, metres
        edge_mask: np.ndarray | None = None,  # (N, N) bool. True keeps the edge
        draws: np.ndarray | None = None,  # (K,) int64 seeds. None draws fresh
    ) -> np.ndarray:  # (K, N, 12, 2) float64, metres
        ...


@runtime_checkable
class AttentionPredictor(Predictor, Protocol):
    """A predictor that can also show its attention."""

    def attention(
        self,
        hist: np.ndarray,  # (N, 8, 2)
    ) -> dict[str, np.ndarray]:  # module name -> (N*8, N*8) float64
        """Rows sum to 1. Token order is agent-major:
        index = agent_index * 8 + time_index."""
        ...


@runtime_checkable
class MaskablePredictor(AttentionPredictor, Protocol):
    """Honours edge_mask. cached.py does NOT satisfy this protocol."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def agent_major_permutation(n_agents: int, n_time: int) -> np.ndarray:
    """Return the index that turns a time-major axis into an agent-major axis.

    A time-major token sits at time_index * n_agents + agent_index.
    An agent-major token sits at agent_index * n_time + time_index.
    The returned array p satisfies out[k] = raw[p[k]] for the agent-major k.
    """
    return np.arange(n_time * n_agents).reshape(n_time, n_agents).T.reshape(-1)


def to_agent_major(raw: np.ndarray, n_agents: int, n_time: int) -> np.ndarray:
    """Reorder both axes of a square token map from time-major to agent-major."""
    order = agent_major_permutation(n_agents, n_time)
    return np.asarray(raw)[np.ix_(order, order)]


def check_prediction(pred: np.ndarray, n_agents: int, n_fut: int) -> np.ndarray:
    """Raise when a predictor breaks the shape promise of section 4.1."""
    array = np.asarray(pred, dtype=np.float64)
    if array.ndim != 4 or array.shape[1:] != (n_agents, n_fut, 2):
        raise ValueError(
            f"predict must return (K, {n_agents}, {n_fut}, 2), it returned "
            f"{array.shape}"
        )
    return array


def check_history(hist: np.ndarray, n_hist: int) -> np.ndarray:
    """Raise when the history is not (N, n_hist, 2)."""
    array = np.asarray(hist, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != n_hist or array.shape[2] != 2:
        raise ValueError(f"hist must be (N, {n_hist}, 2), it is {array.shape}")
    return array


def check_edge_mask(edge_mask: np.ndarray | None, n_agents: int) -> np.ndarray | None:
    """Raise when the edge mask is not (N, N) boolean.

    Entry (dst, src) is True when the edge from src into dst stays. The row
    index is the agent that attends. The column index is the agent it attends
    to. That matches attn[dst, src] and matches the src and dst columns of the
    attention_edges table.
    """
    if edge_mask is None:
        return None
    array = np.asarray(edge_mask)
    if array.shape != (n_agents, n_agents):
        raise ValueError(f"edge_mask must be ({n_agents}, {n_agents}), it is {array.shape}")
    return array.astype(bool)
