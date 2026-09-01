"""Tests for src/stats/report.py.

The Holm numbers below reuse the exact worked example from
tests/test_correction.py: p_raw h1=0.01, h2=0.04, h3=0.03 adjusts to
h1=0.03, h3=0.06, h2=0.06. See that file for the hand computation.
"""

from __future__ import annotations

import pytest

from src.stats.report import build

RAW_FIELDS = {
    "test_used": "wilcoxon signed-rank, one-sided",
    "why": "D is symmetric",
    "sided": "one",
    "effect": 0.3,
    "effect_name": "rank_biserial",
    "ci_low": 0.1,
    "ci_high": 0.5,
    "n_clusters": 39,
}


def _row(hypothesis: str, role: str, p_raw: float) -> dict:
    return {"hypothesis": hypothesis, "role": role, "p_raw": p_raw, **RAW_FIELDS}


def test_build_applies_holm_across_the_primary_family():
    results = [
        _row("h1", "primary", 0.01),
        _row("h2", "primary", 0.04),
        _row("h3", "primary", 0.03),
    ]
    out = build(results).set_index("hypothesis")
    assert out.loc["h1", "p_adj"] == pytest.approx(0.03)
    assert out.loc["h3", "p_adj"] == pytest.approx(0.06)
    assert out.loc["h2", "p_adj"] == pytest.approx(0.06)


def test_build_applies_benjamini_hochberg_across_the_supporting_family():
    """Same three p-values, but as a supporting family. BH must give a
    DIFFERENT answer than Holm did above, since it is a different
    correction: h1 must land at 0.03 under BH too (rank 1 of 3, m/i = 3),
    but h2 and h3 must land at 0.04, not Holm's 0.06."""
    results = [
        _row("s1", "supporting", 0.01),
        _row("s2", "supporting", 0.04),
        _row("s3", "supporting", 0.03),
    ]
    out = build(results).set_index("hypothesis")
    assert out.loc["s1", "p_adj"] == pytest.approx(0.03)
    assert out.loc["s3", "p_adj"] == pytest.approx(0.04)
    assert out.loc["s2", "p_adj"] == pytest.approx(0.04)


def test_holm_and_benjamini_hochberg_never_mix_in_one_call():
    """A primary hypothesis and a supporting one, each alone in its own
    family. With m=1 either correction is a no-op, so p_adj must equal
    p_raw for both, proving the two roles were corrected separately and
    not pooled into one family of two."""
    results = [_row("h1", "primary", 0.02), _row("s1", "supporting", 0.02)]
    out = build(results).set_index("hypothesis")
    assert out.loc["h1", "p_adj"] == pytest.approx(0.02)
    assert out.loc["s1", "p_adj"] == pytest.approx(0.02)


def test_verdict_rejects_at_or_below_alpha():
    """config.ALPHA is 0.05. A p_adj of exactly 0.05 must reject: this
    project pins the boundary at alpha itself, not strictly below it."""
    results = [_row("h1", "primary", 0.05)]
    out = build(results).set_index("hypothesis")
    assert out.loc["h1", "verdict"] == "reject"


def test_verdict_does_not_reject_above_alpha():
    results = [_row("h1", "primary", 0.06)]
    out = build(results).set_index("hypothesis")
    assert out.loc["h1", "verdict"] == "do not reject"


def test_a_supplied_p_adj_or_verdict_is_overwritten_not_trusted():
    """build() is the only place that sees the whole family, so any p_adj
    or verdict a caller supplies could not have accounted for it. build()
    must ignore both and compute its own."""
    row = _row("h1", "primary", 0.01)
    row["p_adj"] = 0.999
    row["verdict"] = "do not reject"
    out = build([row]).set_index("hypothesis")
    assert out.loc["h1", "p_adj"] == pytest.approx(0.01)  # m=1, no-op correction
    assert out.loc["h1", "verdict"] == "reject"


def test_build_rejects_an_empty_result_list():
    with pytest.raises(ValueError, match="at least one result"):
        build([])


def test_build_rejects_a_result_missing_a_required_field():
    incomplete = {"hypothesis": "h1", "role": "primary", "p_raw": 0.01}
    with pytest.raises(KeyError):
        build([incomplete])


def test_build_rejects_a_role_that_is_neither_primary_nor_supporting():
    with pytest.raises(ValueError, match="primary or supporting"):
        build([_row("h1", "exploratory", 0.01)])


def test_build_rejects_a_repeated_hypothesis():
    """schema.py's own duplicate-key check, not a check this file adds."""
    with pytest.raises(ValueError, match="repeats"):
        build([_row("h1", "primary", 0.01), _row("h1", "primary", 0.02)])


def test_build_returns_every_verdicts_column_in_the_contract_order():
    out = build([_row("h1", "primary", 0.01)])
    assert list(out.columns) == [
        "hypothesis",
        "role",
        "test_used",
        "why",
        "sided",
        "effect",
        "effect_name",
        "ci_low",
        "ci_high",
        "p_raw",
        "p_adj",
        "n_clusters",
        "verdict",
    ]
