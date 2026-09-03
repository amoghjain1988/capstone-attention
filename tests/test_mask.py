"""Tests for src/ablation/mask.py, masked_predict.

Amogh's review of feat/ablation-faithfulness found that masked_predict
validated mask_policy, then called model.predict without it. weight_zero
therefore silently ran logit_neg_inf, and a robustness check that compares
the two policies read two identical answers as strong evidence.

This file proves mask_policy now reaches the model, and proves a model that
cannot honour a policy raises instead of returning a silent duplicate.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.ablation.mask import masked_predict
from src.models.mock import MockPredictor


class _TwoPolicyStub:
    """A model built only for this test file. It really tells the two
    policies apart, which no model on this machine can do today: the
    released AgentFormer implements only logit_neg_inf, and MockPredictor
    accepts no mask_policy argument at all. This stub isolates masked_predict
    OWN plumbing from that gap."""

    def predict(
        self,
        hist,
        edge_mask=None,
        draws=None,
        mask_policy: str = config.PRIMARY_MASK_POLICY,
    ):
        offset = 0.0 if mask_policy == "logit_neg_inf" else 5.0
        n_agents = hist.shape[0]
        return np.full((1, n_agents, config.N_FUT, 2), offset, dtype=np.float64)


class _DropsPolicyStub:
    """A model that reproduces the exact bug this file guards against. It
    accepts mask_policy as a parameter, then ignores its value."""

    def predict(
        self,
        hist,
        edge_mask=None,
        draws=None,
        mask_policy: str = config.PRIMARY_MASK_POLICY,
    ):
        n_agents = hist.shape[0]
        return np.zeros((1, n_agents, config.N_FUT, 2), dtype=np.float64)


@pytest.fixture
def hist() -> np.ndarray:
    return np.zeros((3, config.N_HIST, 2), dtype=np.float64)


@pytest.fixture
def draws() -> np.ndarray:
    return np.arange(1, dtype=np.int64)


@pytest.fixture
def mask() -> np.ndarray:
    return np.ones((3, 3), dtype=bool)


def test_the_two_policies_give_different_predictions_on_a_model_that_honours_both(
    hist, draws, mask
):
    """The regression test for the bug itself. If masked_predict ever drops
    mask_policy again, both calls reach the stub with its default value, and
    this assertion fails."""
    model = _TwoPolicyStub()
    neg_inf = masked_predict(model, hist, mask, draws, "logit_neg_inf")
    zero = masked_predict(model, hist, mask, draws, "weight_zero")
    assert not np.array_equal(neg_inf, zero)


def test_mock_predictor_still_runs_under_the_primary_policy(hist, draws, mask):
    """MockPredictor has one behaviour and no mask_policy parameter. The
    primary policy must still run against it, unchanged from before this
    fix, so every one of the 108 tests that already pass keeps passing."""
    model = MockPredictor()
    result = masked_predict(model, hist, mask, draws, config.PRIMARY_MASK_POLICY)
    assert result.shape == (1, 3, config.N_FUT, 2)


def test_a_non_primary_policy_raises_on_mock_predictor(hist, draws, mask):
    """weight_zero must never silently become logit_neg_inf. MockPredictor
    has no mask_policy parameter, so it must refuse the request outright."""
    model = MockPredictor()
    with pytest.raises(NotImplementedError, match="weight_zero"):
        masked_predict(model, hist, mask, draws, "weight_zero")


def test_a_model_that_drops_mask_policy_is_not_caught_by_this_guard(hist, draws, mask):
    """State a real limit in plain language, rather than hide it.

    A model that ACCEPTS mask_policy as a parameter and then ignores its
    value cannot be told apart from a model that honours it, from outside.
    masked_predict can only stop a model from ignoring the argument when the
    model refuses to accept it at all, or when the model's own predict
    method carries its own check, as AgentFormerPredictor.predict does. This
    test names that limit, so nobody assumes this guard catches every case.
    """
    model = _DropsPolicyStub()
    neg_inf = masked_predict(model, hist, mask, draws, "logit_neg_inf")
    zero = masked_predict(model, hist, mask, draws, "weight_zero")
    assert np.array_equal(neg_inf, zero)  # the silent duplicate, still possible here
