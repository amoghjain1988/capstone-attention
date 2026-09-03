"""Tests for the hypothesis layer.

This file grows as each hypothesis file lands. For now it locks the frozen
design. Add the choose_test and data_checks tests when those files exist.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import config
from src.hypotheses import data_checks, h3_context
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


# ---------------------------------------------------------------------------
# data_checks.check
# ---------------------------------------------------------------------------


def _curve_rows(window_id: str, arms: dict[str, float]) -> list[dict]:
    """Build primary-slice perturbation_curves rows for one window.

    `arms` maps arm name to its shift. Every row uses config.PRIMARY_N_REMOVED
    and config.PRIMARY_MASK_POLICY, so it lands in the primary slice check().
    """
    return [
        {
            "window_id": window_id,
            "arm": arm,
            "n_removed": config.PRIMARY_N_REMOVED,
            "mask_policy": config.PRIMARY_MASK_POLICY,
            "shift": shift_value,
        }
        for arm, shift_value in arms.items()
    ]


def _complete_arms(**overrides: float) -> dict[str, float]:
    base = {arm: 0.1 for arm in data_checks.REQUIRED_ARMS}
    base.update(overrides)
    return base


def test_check_returns_the_attrition_table_when_every_window_is_complete():
    rows = _curve_rows("eth_000000", _complete_arms())
    rows += _curve_rows("univ_000010", _complete_arms())
    table = data_checks.check(pd.DataFrame(rows))

    assert set(table["scene"]) == {"eth", "univ"}
    assert (table["n_incomplete"] == 0).all()
    assert (table["n_complete"] == table["n_windows"]).all()


def test_check_keeps_a_zero_difference_and_does_not_raise():
    # A shift of exactly 0.0 is real data, not a gap.
    rows = _curve_rows("eth_000000", _complete_arms(morf=0.0, lerf=0.0))
    table = data_checks.check(pd.DataFrame(rows))
    assert int(table["n_incomplete"].iloc[0]) == 0


def test_check_raises_when_a_window_is_missing_an_arm():
    arms = _complete_arms()
    del arms["nearest"]
    rows = _curve_rows("eth_000000", arms)
    with pytest.raises(ValueError, match="eth_000000"):
        data_checks.check(pd.DataFrame(rows))


def test_check_raises_when_a_required_arm_holds_a_null_shift():
    rows = _curve_rows("eth_000000", _complete_arms(morf=np.nan))
    with pytest.raises(ValueError):
        data_checks.check(pd.DataFrame(rows))


def test_check_ignores_non_primary_rows_when_the_primary_slice_is_complete():
    rows = _curve_rows("eth_000000", _complete_arms())
    # A non-primary mask_policy row is incomplete on its own, but check()
    # only looks at the primary slice, so it must not raise.
    rows.append(
        {
            "window_id": "eth_000000",
            "arm": "morf",
            "n_removed": config.PRIMARY_N_REMOVED,
            "mask_policy": "weight_zero",
            "shift": np.nan,
        }
    )
    table = data_checks.check(pd.DataFrame(rows))
    assert int(table["n_incomplete"].iloc[0]) == 0


def test_check_requires_a_window_id_and_arm_column():
    with pytest.raises(KeyError):
        data_checks.check(pd.DataFrame({"arm": ["morf"]}))
    with pytest.raises(KeyError):
        data_checks.check(pd.DataFrame({"window_id": ["eth_000000"]}))


# ---------------------------------------------------------------------------
# h3_context.run
# ---------------------------------------------------------------------------


def _install_fake_component_clusters(monkeypatch, cluster_fn=None):
    """Inject a stand-in src.stats.clustered module for h3_context's lazy
    import. This is a TEST-ONLY stub, never a second copy of Prem's real
    connected-component algorithm -- src/stats/clustered.py does not exist
    yet, so h3_context.run() cannot import the real thing.
    """
    fake = types.ModuleType("src.stats.clustered")
    fake.component_clusters = cluster_fn or (lambda table: table["ego_id"].to_numpy())
    monkeypatch.setitem(sys.modules, "src.stats.clustered", fake)


def _synthetic_h3_data(n_per_scene: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small, hand-built fi/context pair with real signal and a
    deliberate share of TWO_EDGE_TRAP windows, across 2 scenes."""
    rng = np.random.default_rng(seed)
    rows_fi: list[dict] = []
    rows_context: list[dict] = []
    counter = 0

    for scene in ("eth", "univ"):
        for _ in range(n_per_scene):
            window_id = f"{scene}_{counter:06d}"
            counter += 1
            n_edges = int(rng.integers(2, 8))
            density = float(rng.uniform(0.0, 10.0))
            n_agents = int(rng.integers(2, 15))
            closing_speed = float(rng.normal())
            inv_ttc = float(abs(rng.normal()))
            ego_id = int(rng.integers(0, 100))

            if n_edges == h3_context.TWO_EDGE_TRAP:
                fi_value = 1.0 if rng.uniform() < 0.5 else -1.0
            else:
                signal = 0.4 * density - 0.2 * closing_speed
                fi_value = float(np.clip(signal + rng.normal(scale=0.3), -1.0, 1.0))

            rows_fi.append(
                {
                    "window_id": window_id,
                    "ego_id": ego_id,
                    "n_edges": n_edges,
                    "floor": 0.1,
                    "ceiling": 0.5,
                    "fi": fi_value,
                }
            )
            rows_context.append(
                {
                    "window_id": window_id,
                    "density": density,
                    "n_agents": n_agents,
                    "closing_speed": closing_speed,
                    "inv_ttc": inv_ttc,
                }
            )

    return pd.DataFrame(rows_fi), pd.DataFrame(rows_context)


def test_h3_run_splits_the_two_edge_trap(monkeypatch):
    _install_fake_component_clusters(monkeypatch)
    fi, context = _synthetic_h3_data()

    result = h3_context.run(fi, context, cfg=config)

    assert set(result) == {
        "with_two_edge", "without_two_edge", "headline", "n_dropped_two_edge",
    }
    assert result["headline"] == "without_two_edge"
    expected_dropped = int((fi["n_edges"] == h3_context.TWO_EDGE_TRAP).sum())
    assert result["n_dropped_two_edge"] == expected_dropped
    assert expected_dropped > 0  # the synthetic data must actually hit the trap
    assert (
        result["with_two_edge"]["n"] - result["without_two_edge"]["n"]
        == expected_dropped
    )


def test_h3_fit_reports_a_coefficient_table_and_valid_ranges(monkeypatch):
    _install_fake_component_clusters(monkeypatch)
    fi, context = _synthetic_h3_data()

    result = h3_context.run(fi, context, cfg=config)

    for key in ("with_two_edge", "without_two_edge"):
        fit = result[key]
        assert 0.0 <= fit["joint_p"] <= 1.0
        assert fit["partial_r2"] == pytest.approx(fit["partial_r2"])  # not NaN
        terms = set(fit["coefficients"]["term"])
        assert terms >= {"Intercept", "density", "n_agents", "closing_speed", "inv_ttc"}
        assert (fit["vif"]["vif"] > 0.0).all()
        assert fit["n_clusters"] <= fit["n"]


def test_h3_run_drops_rows_with_an_undefined_fi(monkeypatch):
    _install_fake_component_clusters(monkeypatch)
    fi, context = _synthetic_h3_data()
    fi = fi.copy()
    fi.loc[fi.index[0], "fi"] = np.nan

    result = h3_context.run(fi, context, cfg=config)
    assert result["with_two_edge"]["n"] == len(fi) - 1


def test_h3_run_requires_the_context_columns(monkeypatch):
    _install_fake_component_clusters(monkeypatch)
    fi, context = _synthetic_h3_data()
    with pytest.raises(KeyError):
        h3_context.run(fi, context.drop(columns=["density"]), cfg=config)


def test_h3_run_requires_the_fi_columns(monkeypatch):
    _install_fake_component_clusters(monkeypatch)
    fi, context = _synthetic_h3_data()
    with pytest.raises(KeyError):
        h3_context.run(fi.drop(columns=["n_edges"]), context, cfg=config)


def test_h3_run_rejects_a_non_component_cluster_unit(monkeypatch):
    _install_fake_component_clusters(monkeypatch)
    fi, context = _synthetic_h3_data()

    class FakeCfg:
        CLUSTER_UNIT = "pedestrian"

    with pytest.raises(ValueError):
        h3_context.run(fi, context, cfg=FakeCfg())
