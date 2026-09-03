"""Tests for src/hypotheses/figures.py.

Every input here is built by hand. A figure test cannot check that a picture
is correct, so each test checks the three facts that a caller depends on: the
function returns a path, the path holds a png inside the configured figure
directory, and the file is large enough to hold real marks. Each figure
also runs against the rows of one scene alone, because
src/hypotheses/validate.py calls the same functions per scene.

A few tests go further and read one number OUT of a drawn panel. A number
that a figure states must agree with the same number in outputs/tables, and a
test is the only place that holds the two together.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import config
from src.data import schema
from src.hypotheses import figures

# A png of a real figure is far above this size. An empty axes frame is far
# below it, so the check catches a figure that draws nothing.
MIN_BYTES = 5_000

SCENES = ("eth", "hotel", "univ", "zara1", "zara2")
ARMS = ("morf", "lerf", "nearest", "random")
STEPS = (1, 2, 3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def figure_dir(tmp_path, monkeypatch):
    """Point config.FIGURE_DIR at a directory of this test run."""
    target = tmp_path / "figures"
    monkeypatch.setattr(config, "FIGURE_DIR", target)
    return target


def _windows(scenes=SCENES, per_scene: int = 2) -> list[tuple[str, int]]:
    """Return (window_id, ego_id) pairs, two per scene by default."""
    out = []
    for scene_index, scene in enumerate(scenes):
        for index in range(per_scene):
            t0 = 20 * (index + 1)
            out.append((schema.window_id(scene, t0), 100 * scene_index + index + 1))
    return out


def _curves(scenes=SCENES, per_scene: int = 2) -> pd.DataFrame:
    """Return a small perturbation_curves table.

    The table holds six windows or more, every arm of the figure, three
    steps, the mass matched arm, the single arm and a second mask policy.
    morf carries the largest shift, so the drawn order is the expected one.
    """
    generator = np.random.default_rng(0)
    strength = {"morf": 0.40, "lerf": 0.10, "nearest": 0.28, "random": 0.18}
    rows = []
    for window, ego in _windows(scenes, per_scene):
        for arm in ARMS:
            for step in STEPS:
                rows.append(
                    {
                        "window_id": window,
                        "ego_id": ego,
                        "arm": arm,
                        "n_removed": step,
                        "removed_mass": 0.22 * step + 0.02 * generator.normal(),
                        "edge_src": -1,
                        "draw_id": 0,
                        "mask_policy": "logit_neg_inf",
                        "run_id": "test",
                        "shift": strength[arm] * step + 0.03 * generator.normal(),
                    }
                )
        for step in STEPS:
            rows.append(
                {
                    "window_id": window,
                    "ego_id": ego,
                    "arm": "weight_matched",
                    "n_removed": step,
                    "removed_mass": 0.22 * step + 0.03 * generator.normal(),
                    "edge_src": -1,
                    "draw_id": 0,
                    "mask_policy": "logit_neg_inf",
                    "run_id": "test",
                    "shift": 0.14 * step + 0.03 * generator.normal(),
                }
            )
        for edge in (7, 8, 9):
            for policy in ("logit_neg_inf", "weight_zero"):
                rows.append(
                    {
                        "window_id": window,
                        "ego_id": ego,
                        "arm": "single",
                        "n_removed": 1,
                        "removed_mass": 0.2,
                        "edge_src": edge,
                        "draw_id": 0,
                        "mask_policy": policy,
                        "run_id": "test",
                        "shift": 0.2 + 0.02 * edge,
                    }
                )
    return pd.DataFrame(rows)


def _d_table(seed: int, shift: float, n_rows: int = 40) -> pd.DataFrame:
    """Return one D table with window_id, ego_id, shift_a, shift_b and d."""
    generator = np.random.default_rng(seed)
    scenes = np.array(SCENES)[generator.integers(0, len(SCENES), size=n_rows)]
    windows = [
        schema.window_id(scene, 20 * (index + 1))
        for index, scene in enumerate(scenes)
    ]
    shift_a = np.abs(generator.normal(0.5, 0.2, size=n_rows))
    d = generator.normal(shift, 0.08, size=n_rows)
    # Five ties, because the Pratt rule keeps a d of exactly 0.
    d[:5] = 0.0
    return pd.DataFrame(
        {
            "window_id": windows,
            "ego_id": generator.integers(1, 12, size=n_rows),
            "shift_a": shift_a,
            "shift_b": shift_a - d,
            "d": d,
        }
    )


def _faithfulness(scenes=SCENES, per_scene: int = 8) -> pd.DataFrame:
    """Return a faithfulness table with two-edge windows and null values."""
    generator = np.random.default_rng(1)
    rows = []
    for scene in scenes:
        for index in range(per_scene):
            n_edges = 2 if index % 4 == 0 else int(generator.integers(3, 9))
            if n_edges == config.MIN_EGO_EDGES:
                fi = 1.0 if index % 8 == 0 else -1.0
            else:
                fi = float(generator.uniform(-0.4, 1.0))
            floor = float(generator.uniform(0.05, 0.2))
            ceiling = floor + 0.3
            if index == per_scene - 1:
                fi = np.nan
                ceiling = floor
            rows.append(
                {
                    "window_id": schema.window_id(scene, 20 * (index + 1)),
                    "ego_id": index + 1,
                    "n_edges": n_edges,
                    "floor": floor,
                    "ceiling": ceiling,
                    "fi": fi,
                }
            )
    return pd.DataFrame(rows)


def _h3_result(scenes=SCENES) -> dict:
    """Return a fit result with the shape of src/hypotheses/h3_context.run."""
    terms = ["Intercept"]
    terms += [f"C(scene)[T.{scene}]" for scene in scenes[1:]]
    terms += ["density", "n_agents", "closing_speed", "inv_ttc"]
    generator = np.random.default_rng(2)
    coefficients = pd.DataFrame(
        {
            "term": terms,
            "b": generator.normal(0.0, 0.15, size=len(terms)),
            "se": generator.uniform(0.03, 0.12, size=len(terms)),
            "p": generator.uniform(0.001, 0.9, size=len(terms)),
        }
    )
    fit = {
        "n": 412,
        "n_clusters": 39,
        "coefficients": coefficients,
        "joint_statistic": 3.41,
        "joint_p": 0.0132,
        "partial_r2": 0.081,
        "vif": pd.DataFrame({"term": ["density"], "vif": [1.4]}),
    }
    return {
        "with_two_edge": fit,
        "without_two_edge": fit,
        "headline": "without_two_edge",
        "n_dropped_two_edge": 51,
    }


def _validate_table(scenes=SCENES, pooled: bool = True) -> pd.DataFrame:
    """Return a validate table with the loso rows of H1 and H2."""
    generator = np.random.default_rng(3)
    rows = []
    for hypothesis, centre in (("h1", 0.09), ("h2", 0.04)):
        for scene in scenes:
            location = centre + 0.02 * generator.normal()
            half = float(generator.uniform(0.02, 0.05))
            rows.append(
                {
                    "check": "loso",
                    "variant": scene,
                    "hypothesis": hypothesis,
                    "n_windows": int(generator.integers(30, 140)),
                    "n_clusters": int(generator.integers(4, 12)),
                    "effect_name": "rank_biserial",
                    "effect": float(generator.uniform(0.2, 0.7)),
                    "location_m": location,
                    "ci_low": location - half,
                    "ci_high": location + half,
                    "p_raw": float(generator.uniform(0.0001, 0.2)),
                }
            )
        if pooled:
            rows.append(
                {
                    "check": "pooled",
                    "variant": "all",
                    "hypothesis": hypothesis,
                    "n_windows": 480,
                    "n_clusters": 39,
                    "effect_name": "rank_biserial",
                    "effect": 0.44,
                    "location_m": centre,
                    "ci_low": centre - 0.02,
                    "ci_high": centre + 0.02,
                    "p_raw": 0.001,
                }
            )
        # A row of another check must never reach the forest.
        rows.append(
            {
                "check": "heterogeneity",
                "variant": hypothesis,
                "hypothesis": hypothesis,
                "n_windows": 480,
                "n_clusters": 39,
                "effect_name": "i_squared",
                "effect": 0.31,
                "location_m": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_raw": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _verdicts(with_h3: bool = True) -> pd.DataFrame:
    """Return a verdicts table with the primary rows and one supporting row.

    The shape follows `src/stats/report.py`: one row per test, a role, an
    effect with its name, the cluster bootstrap interval, the adjusted p and
    the verdict. The H3 row carries no interval, because a partial R squared
    has none.
    """
    rows = [
        {
            "hypothesis": "h1",
            "role": "primary",
            "effect_name": "share_positive",
            "effect": 0.852,
            "ci_low": 0.809,
            "ci_high": 0.907,
            "p_raw": 0.0001,
            "p_adj": 0.0003,
            "n_clusters": 72,
            "verdict": "reject",
        },
        {
            "hypothesis": "h2",
            "role": "primary",
            "effect_name": "share_positive",
            "effect": 0.804,
            "ci_low": 0.756,
            "ci_high": 0.846,
            "p_raw": 0.0001,
            "p_adj": 0.0003,
            "n_clusters": 72,
            "verdict": "reject",
        },
        {
            "hypothesis": "h1_morf_vs_random",
            "role": "supporting",
            "effect_name": "rank_biserial",
            "effect": 0.611,
            "ci_low": 0.505,
            "ci_high": 0.744,
            "p_raw": 0.0001,
            "p_adj": 0.0001,
            "n_clusters": 72,
            "verdict": "reject",
        },
    ]
    if with_h3:
        rows.insert(
            2,
            {
                "hypothesis": "h3",
                "role": "primary",
                "effect_name": "partial_r2",
                "effect": 0.0167,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_raw": 0.6033,
                "p_adj": 0.6033,
                "n_clusters": 62,
                "verdict": "do not reject",
            },
        )
    return pd.DataFrame(rows)


def _extra(with_h3: bool = True) -> pd.DataFrame:
    """Return the extra table, with the window count of every primary row."""
    rows = [
        {"hypothesis": "h1", "n_windows": 481.0},
        {"hypothesis": "h2", "n_windows": 481.0},
    ]
    if with_h3:
        rows.append({"hypothesis": "h3", "n_windows": 421.0})
    return pd.DataFrame(rows)


def _check(path, figure_dir, name: str) -> None:
    """Check that one figure landed in the configured directory."""
    assert path == figure_dir / f"{name}.png"
    assert path.exists()
    assert path.stat().st_size > MIN_BYTES


# ---------------------------------------------------------------------------
# curves_figure
# ---------------------------------------------------------------------------


def test_curves_figure_writes_one_png(figure_dir):
    """The curve figure writes one png of every arm over five scenes."""
    path = figures.curves_figure(_curves())
    _check(path, figure_dir, "hyp_curves")


def test_curves_figure_takes_one_scene(figure_dir):
    """The curve figure runs on the rows of one scene alone."""
    curves = _curves()
    one = curves.loc[curves["window_id"].str.startswith("zara1_")]
    assert len(one) > 0
    path = figures.curves_figure(one)
    _check(path, figure_dir, "hyp_curves")


def test_curves_figure_honours_n_max(figure_dir):
    """A smaller n_max still writes a png."""
    path = figures.curves_figure(_curves(), n_max=2)
    _check(path, figure_dir, "hyp_curves")


def test_primary_rows_keeps_the_primary_policy():
    """The helper drops the other mask policy and averages over the draws."""
    frame = pd.DataFrame(
        {
            "window_id": ["zara1_000020"] * 3,
            "ego_id": [1, 1, 1],
            "arm": ["morf", "morf", "morf"],
            "n_removed": [1, 1, 1],
            "removed_mass": [0.2, 0.2, 0.2],
            "draw_id": [0, 1, 0],
            "mask_policy": [
                config.PRIMARY_MASK_POLICY,
                config.PRIMARY_MASK_POLICY,
                "weight_zero",
            ],
            "shift": [0.4, 0.6, 99.0],
        }
    )
    rows = figures._primary_rows(frame)
    assert len(rows) == 1
    assert rows["shift"].iloc[0] == pytest.approx(0.5)
    assert rows["scene"].iloc[0] == "zara1"


# ---------------------------------------------------------------------------
# paired_figure
# ---------------------------------------------------------------------------


def test_paired_figure_writes_one_png(figure_dir):
    """The paired figure writes one png of both D tables."""
    path = figures.paired_figure(_d_table(10, 0.08), _d_table(11, 0.03))
    _check(path, figure_dir, "hyp_paired")


def test_paired_bins_hold_an_edge_at_zero():
    """No histogram bar of the paired figure straddles 0.

    A bar that runs from below 0 to above 0 mixes the windows that move with
    the windows that do not, and the reader then reads mass on the wrong side
    of the zero line.
    """
    values = np.array([-0.4, -0.1, 0.0, 0.2, 0.9], dtype=np.float64)
    edges = figures._paired_bins(values, (-0.5, 1.0))
    assert np.isclose(np.abs(edges).min(), 0.0)
    assert edges[0] <= -0.5 and edges[-1] >= 1.0


def test_paired_panel_states_the_share_of_the_sign_test():
    """The share on the panel counts the ties out, as share_positive does.

    src/hypotheses/h1_morf_vs_lerf.py::share_positive divides by the NON-ZERO
    differences, and outputs/tables/verdicts.csv reports that number. A share
    over every window would put a second, smaller number under one name.
    """
    d = np.array([0.3, 0.2, 0.1, 0.4, 0.5, 0.6, -0.1, -0.2, 0.0, 0.0])
    table = pd.DataFrame({"d": d})
    figure, axis = plt.subplots()
    try:
        figures._paired_panel(
            axis, table, "h2", "nearest", "#2a78d6", (-0.5, 1.0)
        )
        # The house style puts the title on the left, so the centre title is
        # empty. Read both, and the test does not depend on that setting.
        title = axis.get_title(loc="left") + axis.get_title()
    finally:
        plt.close(figure)

    # 6 of the 8 windows that are not ties are positive.
    assert "0.750" in title
    assert "0.600" not in title
    assert "ties at 0 = 2" in title


def test_paired_figure_takes_one_scene(figure_dir):
    """The paired figure runs on the rows of one scene alone."""
    d_h1 = _d_table(10, 0.08)
    d_h2 = _d_table(11, 0.03)
    one_h1 = d_h1.loc[d_h1["window_id"].str.startswith("univ_")]
    one_h2 = d_h2.loc[d_h2["window_id"].str.startswith("univ_")]
    path = figures.paired_figure(one_h1, one_h2)
    _check(path, figure_dir, "hyp_paired")


# ---------------------------------------------------------------------------
# fi_figure
# ---------------------------------------------------------------------------


def test_fi_figure_writes_one_png(figure_dir):
    """The faithfulness figure writes one png of five scenes."""
    path = figures.fi_figure(_faithfulness())
    _check(path, figure_dir, "hyp_faithfulness")


def test_fi_figure_takes_one_scene(figure_dir):
    """The faithfulness figure runs on the rows of one scene alone."""
    table = _faithfulness(scenes=("hotel",), per_scene=10)
    path = figures.fi_figure(table)
    _check(path, figure_dir, "hyp_faithfulness")


# ---------------------------------------------------------------------------
# h3_figure
# ---------------------------------------------------------------------------


def test_h3_figure_writes_one_png(figure_dir):
    """The coefficient figure writes one png of the headline fit."""
    path = figures.h3_figure(_h3_result())
    _check(path, figure_dir, "hyp_h3_coefficients")


def test_h3_figure_takes_one_scene(figure_dir):
    """A fit of one scene holds no scene term, and the figure still draws."""
    result = _h3_result(scenes=("zara1",))
    terms = list(result["without_two_edge"]["coefficients"]["term"])
    assert not any(term.startswith("C(scene)") for term in terms)
    path = figures.h3_figure(result)
    _check(path, figure_dir, "hyp_h3_coefficients")


def test_h3_figure_drops_the_intercept(figure_dir):
    """The intercept never reaches the drawn terms."""
    result = _h3_result()
    path = figures.h3_figure(result)
    assert path.exists()
    # The headline names the fit that the figure draws.
    assert result["headline"] == "without_two_edge"


def test_h3_figure_takes_a_bootstrap_p(figure_dir):
    """The figure accepts the p of the wild cluster bootstrap.

    scripts/run_all.py computes that p and the report headlines it. The
    asymptotic Wald p of the result dict is not the p of the report, so the
    caller passes the bootstrap p and the title states it.
    """
    path = figures.h3_figure(_h3_result(), p_bootstrap=0.6033)
    _check(path, figure_dir, "hyp_h3_coefficients")


def test_p_text_states_a_bound_for_a_tiny_p():
    """A p below 0.0001 prints as a bound, never as 0.0000."""
    assert figures._p_text(0.6033) == "0.6033"
    assert figures._p_text(4.02e-56) == "< 0.0001"
    assert figures._p_text(float("nan")) == "not available"
    assert figures._p_text(None) == "not available"


def test_reference_scene_names_the_dropped_level():
    """The scene block compares against the level that carries no term."""
    terms = ["density"] + [f"C(scene)[T.{scene}]" for scene in config.SCENES[1:]]
    assert figures._reference_scene(terms) == config.SCENES[0]


# ---------------------------------------------------------------------------
# loso_figure
# ---------------------------------------------------------------------------


def test_loso_figure_writes_one_png(figure_dir):
    """The forest writes one png of five scenes and both hypotheses."""
    path = figures.loso_figure(_validate_table())
    _check(path, figure_dir, "hyp_loso")


def test_loso_figure_without_a_pooled_row(figure_dir):
    """Without a pooled row the forest draws no pooled line."""
    path = figures.loso_figure(_validate_table(pooled=False))
    _check(path, figure_dir, "hyp_loso")


def test_loso_figure_takes_one_scene(figure_dir):
    """The forest runs on one held-out scene alone."""
    table = _validate_table(scenes=("eth",), pooled=False)
    path = figures.loso_figure(table)
    _check(path, figure_dir, "hyp_loso")


def test_loso_figure_draws_an_interval_that_misses_the_point(figure_dir):
    """A percentile interval above the point estimate still draws.

    src/stats/clustered.py::cluster_bootstrap_ci returns a PERCENTILE
    interval. Such an interval does not have to hold the point that it
    describes, and the median and the Hodges-Lehmann location of a small
    cluster count often land outside it. The forest must therefore draw the
    two bounds as they stand. An error bar around the marker would need a
    negative half width and would raise.
    """
    table = _validate_table(scenes=("eth", "hotel"), pooled=False)
    loso = table["check"] == "loso"
    table.loc[loso, "location_m"] = 0.02
    table.loc[loso, "ci_low"] = 0.035
    table.loc[loso, "ci_high"] = 0.090

    path = figures.loso_figure(table)
    _check(path, figure_dir, "hyp_loso")


def test_loso_figure_draws_a_row_with_no_interval(figure_dir):
    """A row with a null interval still draws, and the label says so.

    outputs/tables/validate.csv holds such a row: the H2 fit that leaves out
    the scene eth returns no interval. The marker alone would look like a
    point estimate of perfect precision, so the tick label states the hole.
    """
    table = _validate_table(scenes=("eth", "hotel"), pooled=True)
    hole = (table["check"] == "loso") & (table["variant"] == "eth")
    table.loc[hole, ["ci_low", "ci_high"]] = np.nan

    path = figures.loso_figure(table)
    _check(path, figure_dir, "hyp_loso")


# ---------------------------------------------------------------------------
# summary_figure
# ---------------------------------------------------------------------------


def test_summary_figure_writes_one_png(figure_dir):
    """The summary writes one png of the three primary rows."""
    path = figures.summary_figure(_verdicts(), _extra())
    _check(path, figure_dir, "hyp_summary")


def test_summary_figure_without_the_h3_row(figure_dir):
    """Two primary rows still draw.

    scripts/run_all.py writes two primary rows when H3 does not fit on the
    sample, so the summary must not need the third row.
    """
    path = figures.summary_figure(_verdicts(with_h3=False), _extra(with_h3=False))
    _check(path, figure_dir, "hyp_summary")


def test_summary_figure_drops_the_supporting_rows(figure_dir):
    """A supporting row never reaches the panel."""
    table = _verdicts()
    assert "supporting" in set(table["role"])
    path = figures.summary_figure(table, _extra())
    assert path.exists()


def test_summary_figure_without_a_window_count(figure_dir):
    """An extra table with no count still draws, without the count."""
    path = figures.summary_figure(_verdicts(), pd.DataFrame({"other": [1]}))
    _check(path, figure_dir, "hyp_summary")
