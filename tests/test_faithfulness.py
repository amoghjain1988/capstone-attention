"""Tests for src/faithfulness/index.py and src/faithfulness/sanity.py.

See GRACE_AGENT.md section 8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.attention import aggregate, rank
from src.faithfulness.index import brute_force_single, faithfulness_index
from src.faithfulness.sanity import noise_floor, zero_check
from src.models.mock import MockPredictor


@pytest.fixture
def hist() -> np.ndarray:
    """4 agents walking toward each other, so distance -- and so attention
    and the true shift -- meaningfully differ between them."""
    generator = np.random.default_rng(11)
    start = generator.uniform(-3.0, 3.0, size=(4, 2))
    velocity = -start / 6.0
    steps = np.arange(config.N_HIST, dtype=np.float64) * config.DT
    return start[:, None, :] + velocity[:, None, :] * steps[None, :, None]


@pytest.fixture
def window_row(hist) -> pd.Series:
    return pd.Series(
        {
            "window_id": "test_000000",
            "ego_id": 0,
            "agent_order": [0, 1, 2, 3],
            "n_agents": 4,
            "hist": hist.reshape(-1).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# sanity.py
# ---------------------------------------------------------------------------


def test_zero_check_returns_exactly_zero(hist):
    model = MockPredictor()
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)
    assert zero_check(model, hist, draws) == 0.0


def test_noise_floor_is_positive_on_a_noisy_model(hist):
    model = MockPredictor(noise_m=0.05)
    assert noise_floor(model, hist, n_repeat=5, seed=0) > 0.0


def test_noise_floor_is_zero_on_a_deterministic_model(hist):
    model = MockPredictor()
    assert noise_floor(model, hist, n_repeat=5, seed=0) == 0.0


# ---------------------------------------------------------------------------
# index.py
# ---------------------------------------------------------------------------


def test_faithfulness_index_is_one_when_morf_picked_the_max_shift_edge():
    single = pd.DataFrame({"src": [1, 2, 3], "shift": [0.2, 0.5, 1.0]})
    assert faithfulness_index(single, morf_edge=3) == pytest.approx(1.0)


def test_faithfulness_index_is_nan_when_e_equals_one():
    single = pd.DataFrame({"src": [1], "shift": [0.5]})
    assert np.isnan(faithfulness_index(single, morf_edge=1))


def test_mock_predictor_gives_a_faithfulness_index_close_to_one(window_row, hist):
    """MockPredictor's attention rank equals its distance rank, so the edge
    attention picks as strongest should also be close to the truly best
    single edge to remove -- the mock is built so the answer is known."""
    model = MockPredictor()
    draws = np.arange(config.N_SAMPLES, dtype=np.int64)
    edges = pd.DataFrame({"src": [1, 2, 3]})

    single = brute_force_single(window_row, model, edges, draws)

    raw_map = model.attention(hist)["encoder"]
    graph = aggregate.collapse(raw_map, n_agents=4, time_agg="mean")
    ranked = rank.rank_edges(graph, ego_index=0, agent_order=[0, 1, 2, 3])
    morf_edge = int(ranked.sort_values("rank_attn").iloc[0]["src"])

    fi = faithfulness_index(single, morf_edge)
    assert fi == pytest.approx(1.0, abs=0.15)
