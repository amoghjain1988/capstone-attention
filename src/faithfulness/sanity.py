"""Proof the measurement is not measuring noise.

See CONTRACT.md section 5.8 and GRACE_AGENT.md Task 6.

Neither function takes an ego. Both compare FULL (K, N, T, 2) predictions,
not the ego-only paths that src/ablation/shift.py compares -- there is no
single ego to filter to when the question is "does an unmasked re-run
reproduce itself" or "how much does sampling alone move every agent".
"""

from __future__ import annotations

import numpy as np

import config


def _full_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean euclidean distance between two (K, N, T, 2) arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"the two predictions must match: {a.shape} vs {b.shape}")
    step_distance = np.linalg.norm(a - b, axis=-1)  # (K, N, T)
    return float(step_distance.mean())


def zero_check(model, hist: np.ndarray, draws: np.ndarray) -> float:
    """Fixed draws, mask nothing. Must return exactly 0.0.

    Runs the SAME unmasked call twice with the SAME `draws` and measures the
    distance between the two predictions. If this is not exactly 0.0, an
    unmasked re-run does not reproduce itself, and every shift reported
    elsewhere is sampling noise wearing the costume of an effect.
    """
    first = model.predict(hist, edge_mask=None, draws=draws)
    second = model.predict(hist, edge_mask=None, draws=draws)
    return _full_distance(first, second)


def noise_floor(model, hist: np.ndarray, n_repeat: int, seed: int) -> float:
    """Free draws, mask nothing. The shift that sampling alone makes.

    Runs `n_repeat` unmasked calls, each with its OWN, independently drawn
    set of draws (unlike zero_check, which pins the draws). Returns the mean
    distance of every repeat against the first, in the same units as
    src/ablation/shift.py's shift().

    On a deterministic model -- the released AgentFormer, or
    `MockPredictor()` with noise_m=0 -- this returns exactly 0.0, because the
    model ignores which draws it was given. Keep the function anyway: test it
    against `MockPredictor(noise_m=0.05)`, which IS stochastic on purpose, so
    the function is proven correct for the day a sampling model is swapped
    in. See docs/EDA_FINDINGS.md fact F14.
    """
    if int(n_repeat) < 2:
        return 0.0

    generator = np.random.default_rng(int(seed))
    predictions = []
    for _ in range(int(n_repeat)):
        draws = generator.integers(0, 2**31 - 1, size=config.N_SAMPLES, dtype=np.int64)
        predictions.append(model.predict(hist, edge_mask=None, draws=draws))

    reference = predictions[0]
    distances = [_full_distance(pred, reference) for pred in predictions[1:]]
    return float(np.mean(distances))
