"""Every predictor returns (K, N, 12, 2).

See CONTRACT.md section 8. This is the first of the three named tests.

The AgentFormer cases need a GPU and the checkpoints, so they skip themselves
when either is absent. Every other case runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.models import base
from src.models.cv import ConstantVelocity
from src.models.mock import MockPredictor

GPU_READY = (config.CHECKPOINT_DIR / "manifest.json").exists()
needs_gpu = pytest.mark.skipif(not GPU_READY, reason="run scripts/setup_agentformer.py")


@pytest.fixture
def hist() -> np.ndarray:
    """Return a small window of 4 agents that walk in different directions."""
    generator = np.random.default_rng(0)
    start = generator.uniform(-5.0, 5.0, size=(4, 2))
    velocity = generator.uniform(-1.0, 1.0, size=(4, 2))
    steps = np.arange(config.N_HIST, dtype=np.float64) * config.DT
    return start[:, None, :] + velocity[:, None, :] * steps[None, :, None]


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def test_mock_returns_the_contract_shape(hist):
    pred = MockPredictor().predict(hist)
    assert pred.shape == (config.N_SAMPLES, 4, config.N_FUT, 2)
    assert pred.dtype == np.float64
    assert np.isfinite(pred).all()


def test_constant_velocity_returns_one_draw(hist):
    pred = ConstantVelocity().predict(hist)
    assert pred.shape == (1, 4, config.N_FUT, 2)
    assert np.isfinite(pred).all()


def test_every_cpu_model_satisfies_the_protocol():
    assert isinstance(MockPredictor(), base.Predictor)
    assert isinstance(ConstantVelocity(), base.Predictor)
    assert isinstance(MockPredictor(), base.MaskablePredictor)


def test_constant_velocity_is_exact():
    """A straight walk must come back as the same straight walk."""
    steps = np.arange(config.N_HIST, dtype=np.float64) * config.DT
    hist = np.stack([np.stack([steps, np.zeros_like(steps)], axis=1)], axis=0)
    pred = ConstantVelocity().predict(hist)
    future = (np.arange(1, config.N_FUT + 1) * config.DT) + steps[-1]
    assert pred[0, 0, :, 0] == pytest.approx(future)
    assert pred[0, 0, :, 1] == pytest.approx(np.zeros(config.N_FUT))


def test_the_shape_check_catches_a_wrong_shape():
    with pytest.raises(ValueError, match="predict must return"):
        base.check_prediction(np.zeros((5, 3, 2)), 3, 12)


# ---------------------------------------------------------------------------
# The token order
# ---------------------------------------------------------------------------


def test_agent_major_permutation_inverts_the_time_major_layout():
    """AgentFormer is time-major. The contract promises agent-major."""
    n_agents, n_time = 3, 4
    # Build a time-major token index, that is time_index * N + agent_index.
    time_major = np.arange(n_time * n_agents)
    order = base.agent_major_permutation(n_agents, n_time)
    agent_major = time_major[order]
    # Agent 0 must now own the first n_time slots, and so on.
    for agent in range(n_agents):
        block = agent_major[agent * n_time : (agent + 1) * n_time]
        assert list(block) == [t * n_agents + agent for t in range(n_time)]


def test_to_agent_major_moves_both_axes():
    n_agents, n_time = 2, 3
    size = n_agents * n_time
    raw = np.arange(size * size).reshape(size, size)
    out = base.to_agent_major(raw, n_agents, n_time)
    order = base.agent_major_permutation(n_agents, n_time)
    assert out[0, 0] == raw[order[0], order[0]]
    assert out[1, 2] == raw[order[1], order[2]]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def test_mock_attention_rows_sum_to_one(hist):
    maps = MockPredictor().attention(hist)
    assert set(maps) == set(config.MODULES)
    for name, array in maps.items():
        assert array.shape == (4 * config.N_HIST, 4 * config.N_HIST), name
        assert array.sum(axis=1) == pytest.approx(np.ones(4 * config.N_HIST))


def test_mock_attention_ranks_the_closest_neighbour_first():
    """The mock is built so that attention order equals distance order."""
    hist = np.zeros((3, config.N_HIST, 2))
    hist[0, :, 0] = 0.0  # the ego
    hist[1, :, 0] = 1.0  # near
    hist[2, :, 0] = 8.0  # far
    maps = MockPredictor().attention(hist)
    from src.attention import aggregate

    graph = aggregate.collapse(maps["encoder"], 3, "mean")
    assert graph[0, 1] > graph[0, 2]


# ---------------------------------------------------------------------------
# AgentFormer
# ---------------------------------------------------------------------------


@needs_gpu
def test_agentformer_returns_the_contract_shape(hist):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from src.models.agentformer import AgentFormerPredictor

    model = AgentFormerPredictor("eth", device="cuda")
    pred = model.predict(hist)
    assert pred.shape == (config.N_SAMPLES, 4, config.N_FUT, 2)
    assert np.isfinite(pred).all()


@needs_gpu
def test_agentformer_attention_rows_sum_to_one(hist):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from src.models.agentformer import AgentFormerPredictor

    maps = AgentFormerPredictor("eth", device="cuda").attention(hist)
    assert maps["encoder"].shape == (4 * config.N_HIST, 4 * config.N_HIST)
    assert maps["decoder"].shape == (4 * config.N_FUT, 4 * config.N_FUT)
    assert maps["cross"].shape == (4 * config.N_FUT, 4 * config.N_HIST)
    for name, array in maps.items():
        assert array.sum(axis=1) == pytest.approx(np.ones(array.shape[0]), abs=1e-5), name
