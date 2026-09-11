"""Tests for the stages of scripts/run_all.py.

See docs/FINISH_PLAN.md section 4 and CONTRACT.md section 5.1.

The script is not a package, so the tests load it from its path with
importlib. Every test runs on a CPU against the synthetic scene of
src/data/synthetic.py and against MockPredictor, the same fixture pattern that
tests/test_driver.py uses. No test loads AgentFormer and no test runs the
ablation.

The stages under test are the stages that own a decision:

    build_context     the maximum reduction over the edges of one window
    stage_edges       the refusal when the attention cache lacks a window
    stage_faithfulness the faithfulness table and its attrition file
    stage_hypotheses  the verdicts table, the extra table and the five figures
    stage_notebook    the notebook, through scripts/make_notebook.py

Stage 9 runs on the planted scene of tests/test_h1_h2.py, because that scene
holds a known answer and it needs no GPU. Stage 10 runs on a temporary output
folder, so the check proves that every cell survives a missing file.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from src.ablation.arms import order_edges
from src.attention import aggregate, edges, rank
from src.data import eligibility, schema, synthetic, windows
from src.faithfulness.index import faithfulness_from_curves
from src.faithfulness.validation import PlantedPredictor
from src.hypotheses import h3_context
from src.models.mock import MockPredictor

N_AGENTS = 5
N_FRAMES = 40
N_WINDOWS = 3
SCENE = "synthetic"

# The count of resamples inside a test of stage 9. tests/test_h1_h2.py uses the
# same number for the same reason: the full file then runs in seconds on one
# CPU. The number changes no interface and no verdict of the real run.
SMALL = 300

# The five figures that stage 9 draws. See docs/FINISH_PLAN.md section 3.8.
STAGE_9_FIGURES = (
    "hyp_curves",
    "hyp_paired",
    "hyp_faithfulness",
    "hyp_h3_coefficients",
)


# ---------------------------------------------------------------------------
# The script under test
# ---------------------------------------------------------------------------


def load_run_all():
    """Return scripts/run_all.py as a module.

    The script lives outside every package, so importlib loads it from its
    path. The module inserts the repository root into sys.path when it runs,
    so its own imports resolve.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_all = load_run_all()


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


def cache_rows(model, window_row: pd.Series) -> list[dict]:
    """Return the cache records of one window.

    The loop mirrors scripts/extract_attention.py::edges_of_window, so the
    test feeds the stage the shape that the real cache holds.
    """
    n_agents = int(window_row["n_agents"])
    hist = schema.unpack_hist(window_row, config.N_HIST)
    ego_index = schema.ego_index(window_row)
    order = [int(a) for a in window_row["agent_order"]]

    records: list[dict] = []
    for module, raw in model.attention(hist).items():
        entropy = aggregate.agent_row_entropy(raw, n_agents)
        for time_agg in config.TIME_AGGS:
            graph = aggregate.collapse(raw, n_agents, time_agg)
            ranked = rank.rank_edges(graph, ego_index, order)
            for record in ranked.to_dict("records"):
                records.append(
                    {
                        "window_id": str(window_row["window_id"]),
                        "src": int(record["src"]),
                        "dst": int(record["dst"]),
                        "module": module,
                        "time_agg": time_agg,
                        "attn": float(record["attn"]),
                        "row_entropy": float(entropy[int(record["src_index"])]),
                        "rank_attn": int(record["rank_attn"]),
                    }
                )
    return records


@pytest.fixture(scope="module")
def scene() -> SimpleNamespace:
    """Return the synthetic trajectories, the windows, the cache and the edges."""
    trajectories, _ = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=3
    )
    frame = windows.make_windows(
        trajectories, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE
    )
    frame["overlap_fraction"] = windows.overlap_fraction(frame, config.WINDOW_LEN)
    frame = eligibility.mark_eligible(frame, config.SEED)
    eligible = frame.loc[frame["eligible"]].head(N_WINDOWS).reset_index(drop=True)

    model = MockPredictor()
    records: list[dict] = []
    for _, row in eligible.iterrows():
        records.extend(cache_rows(model, row))
    cache = pd.DataFrame(records)
    table = edges.build_table(cache, eligible, trajectories)

    return SimpleNamespace(
        trajectories=trajectories,
        windows=eligible,
        cache=cache,
        edges=table,
        primary=edges.primary(table),
    )


def point_config_at(monkeypatch, tmp_path: Path) -> None:
    """Point every output folder of config at a temporary folder."""
    for name, child in (
        ("INTERIM_DIR", "interim"),
        ("PROCESSED_DIR", "processed"),
        ("TABLE_DIR", "tables"),
        ("FIGURE_DIR", "figures"),
    ):
        target = tmp_path / child
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, target)


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


def test_build_context_gives_one_row_per_eligible_window(scene):
    table = h3_context.build_context(scene.windows, scene.trajectories, scene.primary)
    assert len(table) == N_WINDOWS
    assert set(table["window_id"]) == set(scene.windows["window_id"].astype(str))


def test_build_context_carries_every_column_that_run_needs(scene):
    table = h3_context.build_context(scene.windows, scene.trajectories, scene.primary)
    assert list(table.columns) == [
        "window_id",
        "density",
        "n_agents",
        "closing_speed",
        "inv_ttc",
    ]
    assert set(h3_context.CONTEXT_COLUMNS) <= set(table.columns)


def test_build_context_holds_no_null(scene):
    table = h3_context.build_context(scene.windows, scene.trajectories, scene.primary)
    assert not table.isna().any().any()


def test_build_context_takes_the_maximum_over_the_edges(scene):
    table = h3_context.build_context(scene.windows, scene.trajectories, scene.primary)
    wanted = (
        scene.primary.groupby(scene.primary["window_id"].astype(str))[
            ["closing_speed", "inv_ttc"]
        ]
        .max()
        .reset_index()
    )
    merged = table.merge(wanted, on="window_id", suffixes=("", "_wanted"))
    assert len(merged) == N_WINDOWS
    assert merged["closing_speed"].to_list() == pytest.approx(
        merged["closing_speed_wanted"].to_list()
    )
    assert merged["inv_ttc"].to_list() == pytest.approx(
        merged["inv_ttc_wanted"].to_list()
    )


def test_build_context_raises_when_a_window_carries_no_edge(scene):
    """A window with no edge is a pipeline bug, not a natural gap."""
    first = str(scene.windows["window_id"].iloc[0])
    short_edges = scene.primary.loc[
        scene.primary["window_id"].astype(str) != first
    ]
    with pytest.raises(ValueError, match="no primary edge"):
        h3_context.build_context(scene.windows, scene.trajectories, short_edges)


# ---------------------------------------------------------------------------
# stage_edges
# ---------------------------------------------------------------------------


def test_stage_edges_raises_when_the_cache_lacks_a_window(scene, tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)

    first = str(scene.windows["window_id"].iloc[0])
    missing = sorted(set(scene.windows["window_id"].astype(str)) - {first})
    part = scene.cache.loc[scene.cache["window_id"].astype(str) == first]
    part.to_parquet(config.INTERIM_DIR / f"attention_{SCENE}.parquet", index=False)

    with pytest.raises(SystemExit) as error:
        run_all.stage_edges(scene.windows, scene.trajectories)

    message = str(error.value)
    assert "extract_attention" in message
    for window_id in missing:
        assert window_id in message


def test_stage_edges_raises_without_any_cache(scene, tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="no attention cache"):
        run_all.stage_edges(scene.windows, scene.trajectories)


def test_stage_edges_writes_every_table(scene, tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    scene.cache.to_parquet(
        config.INTERIM_DIR / f"attention_{SCENE}.parquet", index=False
    )

    table = run_all.stage_edges(scene.windows, scene.trajectories)

    assert len(table) == len(scene.edges)
    assert schema.path_of("attention_edges", config.PROCESSED_DIR).exists()
    assert (config.PROCESSED_DIR / "context.parquet").exists()
    assert (config.TABLE_DIR / "arm_overlap.csv").exists()

    context = pd.read_parquet(config.PROCESSED_DIR / "context.parquet")
    assert len(context) == N_WINDOWS

    back = schema.read("attention_edges", config.PROCESSED_DIR)
    schema.validate(back, "attention_edges", partial=True)


# ---------------------------------------------------------------------------
# stage_ablation
# ---------------------------------------------------------------------------


def complete_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (curves, edges_primary) for one window that carries every arm.

    The window holds two edges. Every shift agrees with the single row of the
    edge that the arm removes first, so driver.consistency_check finds
    nothing. The removal order of the random arm comes from
    src/ablation/arms.py::order_edges, the same function the arms themselves
    use, so the test never guesses it.
    """
    window_id = "synthetic_000000"
    edges_primary = pd.DataFrame(
        [
            {
                "window_id": window_id,
                "src": 10,
                "dst": 1,
                "attn": 0.6,
                "rank_attn": 1,
                "rank_dist": 2,
            },
            {
                "window_id": window_id,
                "src": 11,
                "dst": 1,
                "attn": 0.4,
                "rank_attn": 2,
                "rank_dist": 1,
            },
        ]
    )
    shift_of = {10: 0.5, 11: 0.1}
    mass_of = {10: 0.6, 11: 0.4}

    rows: list[dict] = []

    def add(arm: str, n_removed: int, mass: float, edge_src: int, shift: float) -> None:
        rows.append(
            {
                "window_id": window_id,
                "ego_id": 1,
                "arm": arm,
                "n_removed": n_removed,
                "removed_mass": mass,
                "edge_src": edge_src,
                "draw_id": config.SEED,
                "mask_policy": config.PRIMARY_MASK_POLICY,
                "run_id": "test",
                "shift": shift,
            }
        )

    for position, src in enumerate(sorted(shift_of), start=1):
        add("single", position, mass_of[src], src, shift_of[src])

    for arm in ("morf", "lerf", "nearest", "random"):
        first = int(order_edges(edges_primary, arm, config.SEED)[0])
        add(arm, 1, mass_of[first], -1, shift_of[first])

    # weight_matched reaches the mass of the one morf edge with the one edge
    # whose mass already matches it, so this window holds a row at step 1.
    add("weight_matched", 1, mass_of[11], -1, shift_of[11])

    return pd.DataFrame(rows), edges_primary


def test_stage_ablation_prints_the_command_when_the_table_is_missing(
    scene, tmp_path, monkeypatch, capsys
):
    point_config_at(monkeypatch, tmp_path)
    result = run_all.stage_ablation(scene.windows, scene.primary)
    assert result is None

    printed = capsys.readouterr().out
    assert "python scripts/run_ablation.py" in printed
    assert "forward passes" in printed


def test_stage_ablation_audits_a_complete_table(tmp_path, monkeypatch, capsys):
    point_config_at(monkeypatch, tmp_path)
    curves, edges_primary = complete_curves()
    schema.write(curves, "perturbation_curves", config.PROCESSED_DIR)

    frozen = pd.DataFrame(
        {"window_id": ["synthetic_000000"], "n_agents": [3], "eligible": [True]}
    )
    result = run_all.stage_ablation(frozen, edges_primary)

    assert len(result) == len(curves)
    assert "consistency check: 0 mismatched rows" in capsys.readouterr().out
    assert (config.TABLE_DIR / "attrition.csv").exists()


def test_stage_ablation_goes_on_when_data_checks_refuses(
    tmp_path, monkeypatch, capsys
):
    """weight_matched matches mass, so a window can hold no row at step 1."""
    point_config_at(monkeypatch, tmp_path)
    curves, edges_primary = complete_curves()
    curves = curves.loc[curves["arm"] != "weight_matched"]
    schema.write(curves, "perturbation_curves", config.PROCESSED_DIR)

    frozen = pd.DataFrame(
        {"window_id": ["synthetic_000000"], "n_agents": [3], "eligible": [True]}
    )
    result = run_all.stage_ablation(frozen, edges_primary)

    assert len(result) == len(curves)
    assert "data_checks.check refuses the table" in capsys.readouterr().out
    assert not (config.TABLE_DIR / "attrition.csv").exists()


# ---------------------------------------------------------------------------
# stage_faithfulness
# ---------------------------------------------------------------------------


def small_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (curves, edges_primary) for two hand-built windows.

    The first window holds three edges and the second holds two, which is the
    frozen floor config.MIN_EGO_EDGES. Only the single arm matters here,
    because faithfulness_from_curves reads that arm alone.
    """
    single = [
        ("synthetic_000000", 1, 10, 0.5),
        ("synthetic_000000", 1, 11, 0.1),
        ("synthetic_000000", 1, 12, 0.0),
        ("synthetic_000005", 2, 20, 0.3),
        ("synthetic_000005", 2, 21, 0.1),
    ]
    curves = pd.DataFrame(
        [
            {
                "window_id": window_id,
                "ego_id": ego_id,
                "arm": "single",
                "n_removed": position,
                "removed_mass": 0.1 * (position + 1),
                "edge_src": edge_src,
                "draw_id": 0,
                "mask_policy": config.PRIMARY_MASK_POLICY,
                "run_id": "test",
                "shift": value,
            }
            for position, (window_id, ego_id, edge_src, value) in enumerate(single)
        ]
    )

    ranks = [
        ("synthetic_000000", 10, 1),
        ("synthetic_000000", 11, 2),
        ("synthetic_000000", 12, 3),
        ("synthetic_000005", 20, 1),
        ("synthetic_000005", 21, 2),
    ]
    edges_primary = pd.DataFrame(
        [
            {"window_id": window_id, "src": src, "dst": 1, "rank_attn": rank_attn}
            for window_id, src, rank_attn in ranks
        ]
    )
    return curves, edges_primary


def test_stage_faithfulness_writes_the_table_and_the_attrition(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    curves, edges_primary = small_curves()

    table = run_all.stage_faithfulness(curves, edges_primary)

    assert len(table) == 2
    assert list(table.columns) == list(schema.FAITHFULNESS)
    assert schema.path_of("faithfulness", config.PROCESSED_DIR).exists()

    back = schema.read("faithfulness", config.PROCESSED_DIR)
    assert len(back) == 2

    path = config.TABLE_DIR / "fi_attrition.csv"
    assert path.exists()
    attrition = pd.read_csv(path)
    assert list(attrition.columns) == list(run_all.FI_ATTRITION_COLUMNS)
    assert len(attrition) == 1

    row = attrition.iloc[0]
    assert row["scene"] == SCENE
    assert int(row["n_windows"]) == 2
    assert int(row["n_two_edge"]) == 1  # the window at config.MIN_EGO_EDGES
    assert int(row["n_nan"]) == 0
    assert float(row["share_fi_above_zero"]) == pytest.approx(1.0)


def test_fi_attrition_counts_a_null_index():
    """A window whose ceiling equals its floor gives fi of NaN, not a number."""
    table = pd.DataFrame(
        {
            "window_id": ["synthetic_000000", "synthetic_000005"],
            "ego_id": [1, 2],
            "n_edges": [3, 2],
            "floor": [0.2, 0.1],
            "ceiling": [0.5, 0.1],
            "fi": [1.0, float("nan")],
        }
    )
    attrition = run_all.fi_attrition(table)
    assert int(attrition["n_nan"].iloc[0]) == 1
    assert int(attrition["n_windows"].iloc[0]) == 2
    assert float(attrition["median_fi"].iloc[0]) == pytest.approx(1.0)


def test_fi_attrition_of_an_empty_table_is_empty():
    attrition = run_all.fi_attrition(pd.DataFrame(columns=list(schema.FAITHFULNESS)))
    assert len(attrition) == 0
    assert list(attrition.columns) == list(run_all.FI_ATTRITION_COLUMNS)


# ---------------------------------------------------------------------------
# The stride 1 cache of stage 2
# ---------------------------------------------------------------------------


def test_the_windows_cache_makes_a_round_trip(scene, tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    path = run_all.stride1_path()
    assert run_all.read_windows_cache(path) is None  # nothing on disk yet

    run_all.write_windows_cache(scene.windows, path)
    back = run_all.read_windows_cache(path)
    assert back is not None
    assert len(back) == len(scene.windows)
    assert set(back["window_id"]) == set(scene.windows["window_id"].astype(str))


def test_the_windows_cache_is_rebuilt_when_the_edge_floor_changed(
    scene, tmp_path, monkeypatch
):
    """A cache that holds the old edge floor is stale, so it is not read."""
    point_config_at(monkeypatch, tmp_path)
    path = run_all.stride1_path()
    run_all.write_windows_cache(scene.windows, path)

    monkeypatch.setattr(config, "MIN_EGO_EDGES", int(scene.windows["n_agents"].max()))
    assert run_all.read_windows_cache(path) is None


# ---------------------------------------------------------------------------
# stage_hypotheses
# ---------------------------------------------------------------------------


def load_planted():
    """Return tests/test_h1_h2.py as a module, for its planted fixture.

    The planted scene of that file already holds a known answer: the model
    moves only on a real edge, so MoRF beats LeRF and MoRF beats Nearest.
    importlib loads the file from its path, so this test needs no assumption
    about sys.path.
    """
    path = Path(__file__).resolve().parent / "test_h1_h2.py"
    spec = importlib.util.spec_from_file_location("planted_scene_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def planted() -> SimpleNamespace:
    """Return the four frames that stage_hypotheses reads.

    The frames are the curves of the planted scene, the whole attention_edges
    table, the faithfulness table of
    src/faithfulness/index.py::faithfulness_from_curves, and the covariate
    table of src/hypotheses/h3_context.py::build_context.

    TWO COLUMNS CARRY A RIPPLE. The planted scene holds 8 agents in every
    window, so `n_agents` is constant, and the planted model always picks the
    strongest edge, so `fi` is exactly 1.0 on every window. A constant column
    has no z-score and a constant response has no residual, so H3 cannot fit on
    those two columns as they stand. The fixture therefore adds one small
    deterministic ripple to each of the two columns. The ripple gives the H3
    design matrix full rank and a defined partial R squared. It changes no
    interface, and every other number of the fixture stays as the pipeline
    built it. The five real scenes need no such ripple, because their agent
    count and their index both vary.
    """
    module = load_planted()
    picked, edges_primary, curves = module.fixture()

    trajectories, _ = synthetic.make_scene(
        n_agents=module.N_AGENTS,
        n_frames=module.N_FRAMES,
        seed=module.SCENE_SEED,
        n_real=module.N_REAL,
    )
    influence = synthetic.influence_matrix(
        n_agents=module.N_AGENTS, seed=module.SCENE_SEED, n_real=module.N_REAL
    )
    model = PlantedPredictor(influence)
    records: list[dict] = []
    for _, row in picked.iterrows():
        records.extend(module.cache_rows(model, row))
    table = edges.build_table(pd.DataFrame(records), picked, trajectories)

    faithfulness = (
        faithfulness_from_curves(curves, edges_primary)
        .sort_values("window_id")
        .reset_index(drop=True)
    )
    context = (
        h3_context.build_context(picked, trajectories, edges_primary)
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    position = np.arange(len(context))
    context["n_agents"] = context["n_agents"].astype("float64") + (position % 3)
    faithfulness["fi"] = 1.0 - 0.05 * (np.arange(len(faithfulness)) % 4)

    return SimpleNamespace(
        curves=curves,
        edges=table,
        primary=edges_primary,
        faithfulness=faithfulness,
        context=context,
    )


def test_stage_hypotheses_returns_none_without_the_curves(tmp_path, monkeypatch, capsys):
    point_config_at(monkeypatch, tmp_path)
    assert run_all.stage_hypotheses(None, None, None, None) is None
    assert "run_ablation.py" in capsys.readouterr().out


def test_stage_hypotheses_writes_the_verdicts_and_every_figure(
    planted, tmp_path, monkeypatch
):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "N_BOOT", SMALL)
    monkeypatch.setattr(config, "N_PERM", SMALL)

    verdicts = run_all.stage_hypotheses(
        planted.curves, planted.edges, planted.faithfulness, planted.context
    )

    assert verdicts is not None
    assert list(verdicts.columns) == list(schema.VERDICTS)
    assert schema.path_of("verdicts", config.PROCESSED_DIR).exists()

    path = config.TABLE_DIR / "verdicts.csv"
    assert path.exists()
    table = pd.read_csv(path)
    assert {"h1", "h2", "h3"} <= set(table["hypothesis"].astype(str))
    assert set(table.loc[table["hypothesis"].isin(["h1", "h2", "h3"]), "role"]) == {
        "primary"
    }
    assert table["p_adj"].notna().all()

    for name in STAGE_9_FIGURES:
        assert (config.FIGURE_DIR / f"{name}.png").exists(), name

    # src/hypotheses/validate.py is written in parallel with this script. The
    # stage skips the sensitivity table and the forest plot when the import
    # fails, so the test asks for the plot only when the table is on disk.
    if (config.TABLE_DIR / "validate.csv").exists():
        assert (config.FIGURE_DIR / "hyp_loso.png").exists()


def test_stage_hypotheses_writes_the_extra_tables(planted, tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "N_BOOT", SMALL)
    monkeypatch.setattr(config, "N_PERM", SMALL)

    run_all.stage_hypotheses(
        planted.curves, planted.edges, planted.faithfulness, planted.context
    )

    extra = pd.read_csv(config.TABLE_DIR / "hypotheses_extra.csv")
    assert list(extra.columns) == list(run_all.HYPOTHESES_EXTRA_COLUMNS)
    assert list(extra["hypothesis"]) == ["h1", "h2", "h3"]
    assert extra.loc[extra["hypothesis"] == "h2", "share_agree"].notna().all()
    assert extra.loc[extra["hypothesis"] == "h3", "p_wald"].notna().all()
    assert extra.loc[extra["hypothesis"] == "h1", "location_m"].notna().all()

    coefficients = pd.read_csv(config.TABLE_DIR / "h3_coefficients.csv")
    assert set(coefficients["fit"]) == {"with_two_edge", "without_two_edge"}
    assert {"term", "b", "se", "p"} <= set(coefficients.columns)

    inflation = pd.read_csv(config.TABLE_DIR / "h3_vif.csv")
    assert set(inflation["fit"]) == {"with_two_edge", "without_two_edge"}
    assert set(h3_context.CONTEXT_COLUMNS) <= set(inflation["term"])


def test_the_h3_p_value_comes_from_the_wild_cluster_bootstrap(
    planted, tmp_path, monkeypatch
):
    """The primary p value is the bootstrap p, and p_wald keeps the Wald p."""
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "N_BOOT", SMALL)
    monkeypatch.setattr(config, "N_PERM", SMALL)

    h3 = h3_context.run(planted.faithfulness, planted.context, config)
    rows = run_all.h3_rows(h3, planted.faithfulness, planted.context)

    primary, supporting = rows
    assert primary["hypothesis"] == "h3"
    assert primary["role"] == "primary"
    assert primary["sided"] == "two"
    assert primary["effect_name"] == "partial_r2"
    assert np.isnan(primary["ci_low"]) and np.isnan(primary["ci_high"])
    assert 0.0 <= primary["p_raw"] <= 1.0
    assert primary["p_wald"] == pytest.approx(h3[h3["headline"]]["joint_p"])
    assert primary["n_clusters"] == h3[h3["headline"]]["n_clusters"]
    assert "wild cluster bootstrap" in primary["why"]

    assert supporting["hypothesis"] == "h3_with_two_edge"
    assert supporting["role"] == "supporting"
    assert 0.0 <= supporting["p_raw"] <= 1.0
    assert supporting["p_wald"] == pytest.approx(h3["with_two_edge"]["joint_p"])
    assert "wild cluster bootstrap" in supporting["why"]


def test_the_h3_bootstrap_tests_the_four_context_terms_only(planted, monkeypatch):
    """The null of the bootstrap holds the columns 1 to 4 of the design, the
    context terms. The intercept and the scene dummies stay out of it."""
    from src.stats import clustered

    monkeypatch.setattr(config, "N_BOOT", SMALL)
    seen: dict = {}
    real = clustered.wild_cluster_bootstrap

    def spy(y, design, clusters, **kwargs):
        seen["restriction"] = kwargs["restriction"]
        seen["width"] = design.shape[1]
        return real(y, design, clusters, **kwargs)

    monkeypatch.setattr(clustered, "wild_cluster_bootstrap", spy)
    run_all.h3_bootstrap(planted.faithfulness, planted.context, "without_two_edge")

    n_context = len(h3_context.CONTEXT_COLUMNS)
    expected = np.eye(seen["width"])[1 : 1 + n_context]
    assert np.array_equal(seen["restriction"], expected)


def test_the_h3_design_matches_the_headline_fit(planted):
    """The rebuilt design holds the intercept, the four covariates and the rows."""
    y, design, clusters = run_all.h3_design(
        planted.faithfulness, planted.context, "without_two_edge"
    )
    n_rows = int((planted.faithfulness["fi"].notna()).sum())
    assert y.size == n_rows
    assert design.shape[0] == n_rows
    # One scene, so the dummies drop out and only the intercept and the four
    # z-scored covariates are left.
    assert design.shape[1] == 1 + len(h3_context.CONTEXT_COLUMNS)
    assert np.allclose(design[:, 0], 1.0)
    assert np.allclose(design[:, 1:].mean(axis=0), 0.0, atol=1e-9)
    assert clusters.size == n_rows


# ---------------------------------------------------------------------------
# stage_notebook and scripts/make_notebook.py
# ---------------------------------------------------------------------------


def test_make_notebook_builds_a_notebook_that_runs(tmp_path, monkeypatch):
    """The file is valid JSON, it holds no output, and every cell runs."""
    point_config_at(monkeypatch, tmp_path)
    make_notebook = run_all.load_make_notebook()

    path = make_notebook.build(tmp_path / "capstone.ipynb")
    assert path.exists()

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    assert len(document["cells"]) >= 20
    assert document["cells"][0]["cell_type"] == "markdown"
    assert document["cells"][1]["cell_type"] == "code"

    code = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    for cell in code:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None

    # Every output folder of config is empty here, so the run proves that a
    # missing file costs one printed note and never an exception.
    assert make_notebook.check(path, use_kernel=False) == len(code)


def test_make_notebook_names_the_research_question(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    make_notebook = run_all.load_make_notebook()
    title = "".join(make_notebook.cells()[0]["source"])
    assert "research question" in title
    assert "H1" in title and "H2" in title and "H3" in title


def test_stage_notebook_writes_the_notebook(tmp_path, monkeypatch, capsys):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "ROOT", tmp_path)

    path = run_all.stage_notebook()

    assert path == tmp_path / "notebooks" / "capstone.ipynb"
    assert path.exists()
    assert "cells" in capsys.readouterr().out
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["cells"]) >= 20
