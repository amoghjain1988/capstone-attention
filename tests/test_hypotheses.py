"""Tests for the hypothesis layer.

This file locks the frozen design, the completeness check of the ablation
curves, and the H3 regression.

THE REAL CLUSTER LABELS. An earlier version of this file installed a stand-in
module for `src.stats.clustered`, because that file did not exist yet.
`src/stats/clustered.py` exists now, so `h3_context.run` imports the real
`component_clusters` and every H3 test below reads the frozen cluster unit of
config.CLUSTER_UNIT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.hypotheses import data_checks, h3_context
from src.hypotheses.design import frozen_design
from src.stats.clustered import component_clusters, n_clusters


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


def _curve_rows(
    window_id: str,
    arms: dict[str, float],
    mass_matched_step: int | None = 2,
    mass_matched_shift: float = 0.1,
) -> list[dict]:
    """Build primary-slice perturbation_curves rows for one window.

    `arms` maps arm name to its shift. Every such row uses
    config.PRIMARY_N_REMOVED and config.PRIMARY_MASK_POLICY, so it lands in
    the primary step that check() reads.

    The mass matched arm follows its own rule, so it gets its own row.
    `mass_matched_step` is the n_removed of that row, which is the SIZE of the
    mass matched set and not the removal step. The default of 2 is what a
    two-edge window really writes. Pass None to leave the arm out.
    """
    rows = [
        {
            "window_id": window_id,
            "arm": arm,
            "n_removed": config.PRIMARY_N_REMOVED,
            "mask_policy": config.PRIMARY_MASK_POLICY,
            "shift": shift_value,
        }
        for arm, shift_value in arms.items()
    ]
    if mass_matched_step is not None:
        rows.append(
            {
                "window_id": window_id,
                "arm": data_checks.MASS_MATCHED_ARM,
                "n_removed": int(mass_matched_step),
                "mask_policy": config.PRIMARY_MASK_POLICY,
                "shift": mass_matched_shift,
            }
        )
    return rows


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


def test_check_accepts_a_weight_matched_row_at_another_step():
    """The mass matched arm writes the SET SIZE into n_removed.

    A two-edge window needs both edges to match the mass that MoRF removes at
    step 1, so its only weight_matched row sits at n_removed 2. The check must
    accept that window. See the module docstring of data_checks.py.
    """
    rows = _curve_rows("eth_000000", _complete_arms(), mass_matched_step=2)
    rows += _curve_rows("univ_000010", _complete_arms(), mass_matched_step=5)
    table = data_checks.check(pd.DataFrame(rows))
    assert (table["n_incomplete"] == 0).all()


def test_check_raises_when_the_mass_matched_arm_is_absent():
    rows = _curve_rows("eth_000000", _complete_arms(), mass_matched_step=None)
    with pytest.raises(ValueError, match="eth_000000"):
        data_checks.check(pd.DataFrame(rows))


def test_check_raises_when_the_mass_matched_shift_is_null():
    rows = _curve_rows("eth_000000", _complete_arms(), mass_matched_shift=np.nan)
    with pytest.raises(ValueError, match="eth_000000"):
        data_checks.check(pd.DataFrame(rows))


def test_check_reads_a_window_that_lost_its_whole_primary_step():
    """A window with rows under the policy but none at step 1 still fails."""
    rows = _curve_rows("eth_000000", _complete_arms())
    rows = [row for row in rows if row["n_removed"] != config.PRIMARY_N_REMOVED]
    with pytest.raises(ValueError, match="eth_000000"):
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


def _synthetic_h3_data(n_per_scene: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small, hand-built fi/context pair with real signal and a
    deliberate share of TWO_EDGE_TRAP windows, across 2 scenes.

    THE WINDOW KEY. `t0` steps by config.WINDOW_STRIDE inside each scene, and
    each scene starts again at 0, the way src/data/windows.py builds the real
    table. The real `component_clusters` reads `t0` out of the key and blocks
    it by config.WINDOW_LEN, so a realistic spread of `t0` gives a realistic
    component count. A key that packed every window of a scene into two time
    blocks collapsed the sample into 3 clusters and left the joint Wald test
    rank deficient.
    """
    rng = np.random.default_rng(seed)
    rows_fi: list[dict] = []
    rows_context: list[dict] = []

    for scene in ("eth", "univ"):
        for step in range(n_per_scene):
            window_id = f"{scene}_{step * config.WINDOW_STRIDE:06d}"
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


def test_h3_run_splits_the_two_edge_trap():
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


def test_h3_fit_reports_a_coefficient_table_and_valid_ranges():
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


def test_h3_joint_p_does_not_move_when_one_scene_shifts():
    """The null holds the 4 context terms only. A constant shift of every fi
    in one scene moves only the fixed effect of that scene, so the joint p and
    the partial R squared stay the same. A null that also held the scene terms
    would move with the shift."""
    fi, context = _synthetic_h3_data()
    base = h3_context.run(fi, context, cfg=config)

    shifted = fi.copy()
    in_univ = shifted["window_id"].str.startswith("univ_")
    shifted.loc[in_univ, "fi"] = shifted.loc[in_univ, "fi"] + 5.0
    moved = h3_context.run(shifted, context, cfg=config)

    for key in ("with_two_edge", "without_two_edge"):
        assert moved[key]["joint_p"] == pytest.approx(base[key]["joint_p"], rel=1e-6)
        assert moved[key]["partial_r2"] == pytest.approx(base[key]["partial_r2"], rel=1e-6)


def test_h3_reads_the_real_component_clusters():
    """The fit clusters on config.CLUSTER_UNIT, not on the ego id.

    `h3_context.run` imports `src.stats.clustered.component_clusters` inside
    the call. This test names the same function and checks that the fit
    reports the same count, so no stand-in can slip back in unnoticed.
    """
    fi, context = _synthetic_h3_data()
    result = h3_context.run(fi, context, cfg=config)

    labels = component_clusters(fi.loc[:, ["window_id", "ego_id"]])
    assert result["with_two_edge"]["n_clusters"] == n_clusters(labels)
    assert n_clusters(labels) > 1


def test_h3_run_drops_rows_with_an_undefined_fi():
    fi, context = _synthetic_h3_data()
    fi = fi.copy()
    fi.loc[fi.index[0], "fi"] = np.nan

    result = h3_context.run(fi, context, cfg=config)
    assert result["with_two_edge"]["n"] == len(fi) - 1


def test_h3_run_requires_the_context_columns():
    fi, context = _synthetic_h3_data()
    with pytest.raises(KeyError):
        h3_context.run(fi, context.drop(columns=["density"]), cfg=config)


def test_h3_run_requires_the_fi_columns():
    fi, context = _synthetic_h3_data()
    with pytest.raises(KeyError):
        h3_context.run(fi.drop(columns=["n_edges"]), context, cfg=config)


def test_h3_run_rejects_a_non_component_cluster_unit():
    fi, context = _synthetic_h3_data()

    class FakeCfg:
        CLUSTER_UNIT = "pedestrian"

    with pytest.raises(ValueError):
        h3_context.run(fi, context, cfg=FakeCfg())
