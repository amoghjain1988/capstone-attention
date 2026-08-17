"""Fixed draws and an all-True mask give a shift of exactly 0.

See CONTRACT.md section 8. This is the second of the three named tests.

It is the most important test in the repository. If an unmasked re-run does not
reproduce itself, then every shift we report is sampling noise wearing the
costume of an effect.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.ablation.shift import ego_path, shift
from src.models.cv import ConstantVelocity
from src.models.mock import MockPredictor

GPU_READY = (config.CHECKPOINT_DIR / "manifest.json").exists()
needs_gpu = pytest.mark.skipif(not GPU_READY, reason="run scripts/setup_agentformer.py")


@pytest.fixture
def hist() -> np.ndarray:
    """Return 4 agents that walk toward each other, so the edges matter."""
    generator = np.random.default_rng(7)
    start = generator.uniform(-4.0, 4.0, size=(4, 2))
    velocity = -start / 6.0
    steps = np.arange(config.N_HIST, dtype=np.float64) * config.DT
    return start[:, None, :] + velocity[:, None, :] * steps[None, :, None]


# ---------------------------------------------------------------------------
# The named test
# ---------------------------------------------------------------------------


def test_an_all_true_mask_gives_exactly_zero_shift(hist):
    model = MockPredictor()
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)
    keep_all = np.ones((4, 4), dtype=bool)

    base_pred = model.predict(hist, edge_mask=None, draws=draws)
    kept = model.predict(hist, edge_mask=keep_all, draws=draws)
    assert shift(ego_path(kept, 0), ego_path(base_pred, 0)) == 0.0


def test_the_same_draws_reproduce_the_same_path(hist):
    model = MockPredictor(noise_m=0.05)
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)
    first = model.predict(hist, draws=draws)
    second = model.predict(hist, draws=draws)
    assert np.array_equal(first, second)


def test_different_draws_move_a_noisy_model(hist):
    model = MockPredictor(noise_m=0.05)
    a = model.predict(hist, draws=np.arange(config.N_SAMPLES, dtype=np.int64))
    b = model.predict(hist, draws=np.arange(100, 100 + config.N_SAMPLES, dtype=np.int64))
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# The shift itself
# ---------------------------------------------------------------------------


def test_shift_measures_metres():
    """Move every step of every draw by 3 metres and read 3 back."""
    unmasked = np.zeros((5, config.N_FUT, 2))
    masked = unmasked.copy()
    masked[:, :, 0] = 3.0
    assert shift(masked, unmasked) == pytest.approx(3.0)


def test_shift_rejects_a_mismatched_shape():
    with pytest.raises(ValueError, match="must match"):
        shift(np.zeros((5, 12, 2)), np.zeros((4, 12, 2)))


def test_removing_an_edge_moves_the_ego(hist):
    """A mock neighbour really pushes, so removing it must change the path."""
    model = MockPredictor()
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)
    base_pred = model.predict(hist, draws=draws)

    mask = np.ones((4, 4), dtype=bool)
    mask[0, 1] = False  # remove the edge from agent 1 into the ego
    masked = model.predict(hist, edge_mask=mask, draws=draws)
    assert shift(ego_path(masked, 0), ego_path(base_pred, 0)) > 0.0


def test_a_mask_on_another_agent_leaves_the_ego_alone(hist):
    """Only the edges into the ego may change the ego at the first step."""
    model = MockPredictor()
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)
    base_pred = model.predict(hist, draws=draws)

    mask = np.ones((4, 4), dtype=bool)
    mask[2, 3] = False  # an edge between two other agents
    masked = model.predict(hist, edge_mask=mask, draws=draws)
    first_step = np.abs(masked[:, 0, 0, :] - base_pred[:, 0, 0, :]).max()
    assert first_step == pytest.approx(0.0, abs=1e-12)


def test_constant_velocity_can_never_react_to_a_mask(hist):
    """It reads one agent at a time, so its shift is the floor of the study."""
    model = ConstantVelocity()
    base_pred = model.predict(hist)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 1] = False
    masked = model.predict(hist, edge_mask=mask)
    assert shift(ego_path(masked, 0), ego_path(base_pred, 0)) == 0.0


# ---------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------


@needs_gpu
def test_agentformer_all_true_mask_gives_exactly_zero_shift(hist):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from src.models.agentformer import AgentFormerPredictor

    model = AgentFormerPredictor("eth", device="cuda")
    base_pred = model.predict(hist)
    kept = model.predict(hist, edge_mask=np.ones((4, 4), dtype=bool))
    assert shift(ego_path(kept, 0), ego_path(base_pred, 0)) == 0.0


@needs_gpu
def test_agentformer_is_deterministic(hist):
    """The released DLow model draws no epsilon, so the noise floor is 0.

    This contradicts CONTRACT.md section 4.2. See docs/EDA_FINDINGS.md F14.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from src.models.agentformer import AgentFormerPredictor

    model = AgentFormerPredictor("eth", device="cuda")
    assert np.array_equal(model.predict(hist), model.predict(hist))


@needs_gpu
def test_agentformer_reacts_to_a_removed_edge(hist):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from src.models.agentformer import AgentFormerPredictor

    model = AgentFormerPredictor("eth", device="cuda")
    base_pred = model.predict(hist)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 1] = False
    masked = model.predict(hist, edge_mask=mask)
    assert shift(ego_path(masked, 0), ego_path(base_pred, 0)) > 0.0


@needs_gpu
def test_agentformer_rejects_an_unsupported_mask_policy(hist):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from src.models.agentformer import AgentFormerPredictor

    model = AgentFormerPredictor("eth", device="cuda")
    with pytest.raises(NotImplementedError, match="logit_neg_inf"):
        model.predict(
            hist, edge_mask=np.ones((4, 4), dtype=bool), mask_policy="weight_zero"
        )
