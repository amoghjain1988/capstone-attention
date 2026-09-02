"""Tests for src/stats/correction.py.

The worked example below is checked by hand in the module docstring style:
see GRACE_AGENT.md Task 4. p-values h1=0.01, h2=0.04, h3=0.03.

Holm, sorted ascending (h1, h3, h2), multiplier (m - rank + 1):
  h1: rank 1, (3-1+1)*0.01 = 3*0.01 = 0.03
  h3: rank 2, (3-2+1)*0.03 = 2*0.03 = 0.06, running max 0.06
  h2: rank 3, (3-3+1)*0.04 = 1*0.04 = 0.04, running max stays 0.06
  -> h1=0.03, h3=0.06, h2=0.06

Benjamini-Hochberg, same sorted order, multiplier m / rank, running MINIMUM
from the largest rank down:
  h2: rank 3, 0.04*3/3 = 0.04, running min 0.04
  h3: rank 2, 0.03*3/2 = 0.045, running min stays 0.04
  h1: rank 1, 0.01*3/1 = 0.03, running min drops to 0.03
  -> h1=0.03, h3=0.04, h2=0.04
"""

from __future__ import annotations

import pytest

from src.stats.correction import benjamini_hochberg, holm


@pytest.fixture
def pvals() -> dict[str, float]:
    return {"h1": 0.01, "h2": 0.04, "h3": 0.03}


def test_holm_matches_the_worked_example_by_hand(pvals):
    adjusted = holm(pvals)
    assert adjusted["h1"] == pytest.approx(0.03)
    assert adjusted["h3"] == pytest.approx(0.06)
    assert adjusted["h2"] == pytest.approx(0.06)


def test_benjamini_hochberg_matches_the_worked_example_by_hand(pvals):
    adjusted = benjamini_hochberg(pvals)
    assert adjusted["h1"] == pytest.approx(0.03)
    assert adjusted["h3"] == pytest.approx(0.04)
    assert adjusted["h2"] == pytest.approx(0.04)


def test_holm_caps_at_one():
    """(3 - 1 + 1) * 0.6 = 1.2 for the smallest p-value here. Holm must
    report 1.0, never a number above it: it is meant to read as a p-value."""
    adjusted = holm({"a": 0.6, "b": 0.9})
    assert adjusted["a"] == pytest.approx(1.0)
    assert adjusted["b"] == pytest.approx(1.0)


def test_holm_never_decreases_going_down_the_sorted_order():
    """The running maximum is the whole point of Holm: a noisier, larger
    p-value earlier in the sort can never make a later one look smaller."""
    raw = {"a": 0.05, "b": 0.001, "c": 0.02}
    adjusted = holm(raw)
    order = sorted(raw, key=lambda key: raw[key])  # smallest p-value first
    values = [adjusted[key] for key in order]
    assert values == sorted(values)


def test_a_single_hypothesis_is_unchanged_by_either_function():
    """With m=1, both multipliers reduce to 1, so the correction is a
    no-op. This is the simplest case either function can face."""
    assert holm({"h1": 0.03})["h1"] == pytest.approx(0.03)
    assert benjamini_hochberg({"h1": 0.03})["h1"] == pytest.approx(0.03)


def test_both_functions_return_every_key_they_were_given(pvals):
    assert set(holm(pvals)) == set(pvals)
    assert set(benjamini_hochberg(pvals)) == set(pvals)


def test_benjamini_hochberg_is_at_least_as_lenient_as_holm(pvals):
    """A textbook fact about the two corrections, not a coincidence of this
    fixture: BH controls a weaker error rate than Holm's family-wise
    guarantee, so it never asks for a stricter (larger) adjusted p-value."""
    holm_adjusted = holm(pvals)
    bh_adjusted = benjamini_hochberg(pvals)
    for key in pvals:
        assert bh_adjusted[key] <= holm_adjusted[key] + 1e-12


# ---------------------------------------------------------------------------
# Amogh's second review, Check 4: a tie case and an all-null case, on top of
# the worked example above.
# ---------------------------------------------------------------------------


def test_holm_gives_a_tied_pair_the_same_adjusted_value():
    """"a" and "b" start with the identical raw p-value. sorted() breaks the
    tie by insertion order, so one of them lands at rank 1 and the other at
    rank 2 -- but the running MAXIMUM must still pull the rank-1 value up to
    match rank-2's, so a tie in the input stays a tie in the output,
    whichever key sorted() happened to place first.

    By hand, m=3, sorted ascending (a, b, c), all reading raw=0.02 for the
    tied pair: rank1 (3-1+1)*0.02=0.06, running max 0.06; rank2
    (3-2+1)*0.02=0.04, running max stays 0.06; rank3 (3-3+1)*0.10=0.10,
    running max 0.10.
    """
    adjusted = holm({"a": 0.02, "b": 0.02, "c": 0.10})
    assert adjusted["a"] == pytest.approx(adjusted["b"])
    assert adjusted["a"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.10)


def test_benjamini_hochberg_gives_a_tied_pair_the_same_adjusted_value():
    """Same tie, the BH rule. By hand, m=3, from the largest rank down:
    rank3 (c) 0.10*3/3=0.10, running min 0.10; rank2 (one of the tied pair)
    0.02*3/2=0.03, running min drops to 0.03; rank1 (the other tied value)
    0.02*3/1=0.06, running min stays 0.03. Both tied keys land at 0.03."""
    adjusted = benjamini_hochberg({"a": 0.02, "b": 0.02, "c": 0.10})
    assert adjusted["a"] == pytest.approx(adjusted["b"])
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.10)


def test_holm_leaves_an_all_null_family_at_the_ceiling():
    """Every hypothesis in this family already reads a raw p-value of 1.0 --
    the family where nothing is significant anywhere. Correction must never
    invent significance: every adjusted value stays pinned at the 1.0
    ceiling, for every key."""
    adjusted = holm({"a": 1.0, "b": 1.0, "c": 1.0})
    assert adjusted == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_benjamini_hochberg_leaves_an_all_null_family_at_the_ceiling():
    adjusted = benjamini_hochberg({"a": 1.0, "b": 1.0, "c": 1.0})
    assert adjusted == {"a": 1.0, "b": 1.0, "c": 1.0}
