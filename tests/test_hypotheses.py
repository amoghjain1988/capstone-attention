"""Tests for the hypothesis layer.

This file grows as each hypothesis file lands. For now it locks the frozen
design. Add the choose_test and data_checks tests when those files exist.
"""

from __future__ import annotations

import pytest

from src.hypotheses.design import frozen_design


def test_frozen_design_returns_all_six_keys():
    """The design dictionary carries every key the report needs."""
    design = frozen_design(deff=3.18, n_rows=571)
    expected = {
        "alpha",
        "sided",
        "primary_n_removed",
        "minimum_effect",
        "power",
        "n_effective",
    }
    assert set(design) == expected


def test_frozen_design_shrinks_the_sample():
    """The effective sample is below the row count when the design effect is above one."""
    design = frozen_design(deff=3.18, n_rows=571)
    assert design["n_effective"] < 571
    assert design["n_effective"] == pytest.approx(571 / 3.18)


def test_frozen_design_rejects_a_design_effect_below_one():
    """A design effect below one is impossible. The function must refuse it."""
    with pytest.raises(ValueError):
        frozen_design(deff=0.9, n_rows=571)
