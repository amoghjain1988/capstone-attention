"""Tests for src/hypotheses/h1_morf_vs_lerf.py and h2_beyond_proximity.py.

See docs/FINISH_PLAN.md sections 3.4, 3.5, 3.6 and 5.

THE FIXTURE. The scene comes from src/data/synthetic.py, so the answer is
known before any test runs. The model is PlantedPredictor of
src/faithfulness/validation.py. Two facts make it the right model here.

    Its shift is exactly 0 for a fake edge and above 0 for a real edge, so
    D = shift(MoRF) - shift(LeRF) holds a strong positive signal.

    Its attention rank order differs from the distance rank order, because the
    planted influence is independent of the geometry. H2 therefore holds
    non-zero differences and does not collapse to a table of ties.

ONE WINDOW PER TIME BLOCK. config.CLUSTER_UNIT is the connected component of
the bipartite graph of egos against time blocks. At the frozen stride of 5,
four windows share every time block, so a single planted scene collapses into
ONE component. A sign flip of one cluster has two outcomes, so no p value
below 0.5 exists there, and the test could not read the planted signal at all.
The fixture therefore keeps one window per time block. The real study reads 5
scenes and about 480 windows, and the pilot of config.py reports 39
components. See config.CLUSTER_UNIT for the measured false positive rate of
each candidate unit.

THE EGO SEED. src/data/eligibility.py picks the ego of every window from a
seed. Seed 2 gives 7 components over these 10 windows and config.SEED gives 5.
Both reject, and 7 leaves a wider margin, so the fixture pins seed 2. The seed
changes which agent is the ego. It changes no other number of the study.

SPEED. Every test passes a SimpleNamespace copy of config with N_BOOT and
N_PERM at 300. The full file then runs in well under 60 seconds on one CPU.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pandas as pd
import pytest

import config
from src.ablation import driver
from src.attention import aggregate, edges as attention_edges, rank
from src.data import eligibility, schema, synthetic, windows
from src.faithfulness.validation import PlantedPredictor
from src.hypotheses import h1_morf_vs_lerf as h1
from src.hypotheses import h2_beyond_proximity as h2
from src.stats import report

N_AGENTS = 8
N_FRAMES = 200
N_REAL = 2
SCENE_SEED = 0
EGO_SEED = 2

# The count of resamples inside a test. See the module docstring.
SMALL = 300

# Every key that docs/FINISH_PLAN.md section 3.4 demands of one result dict.
REQUIRED_KEYS = (
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
    "n_clusters",
    "location_m",
    "location_ci_low",
    "location_ci_high",
    "n_windows",
)


def small_config():
    """Return a copy of config with a small resample count."""
    cfg = types.SimpleNamespace(**{
        name: value for name, value in vars(config).items()
        if not name.startswith("__")
    })
    cfg.N_BOOT = SMALL
    cfg.N_PERM = SMALL
    return cfg


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


def cache_rows(model, window_row: pd.Series) -> list[dict]:
    """Return the cache records of one window, one row per edge and variant.

    The loop mirrors scripts/extract_attention.py, so the test feeds
    src/attention/edges.py::build_table the shape that the real cache holds.
    """
    n_agents = int(window_row["n_agents"])
    hist = schema.unpack_hist(window_row, config.N_HIST)
    ego_index = schema.ego_index(window_row)
    order = [int(agent) for agent in window_row["agent_order"]]

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


def _build_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (the picked windows, the primary edges, the curves).

    The function runs once per module. See the module docstring for the choice
    of one window per time block and of the ego seed.
    """
    trajectories, _ = synthetic.make_scene(
        n_agents=N_AGENTS, n_frames=N_FRAMES, seed=SCENE_SEED, n_real=N_REAL
    )
    frame = windows.make_windows(
        trajectories, config.N_HIST, config.N_FUT, config.WINDOW_STRIDE
    )
    frame["overlap_fraction"] = windows.overlap_fraction(frame, config.WINDOW_LEN)
    frame = eligibility.mark_eligible(frame, EGO_SEED)

    eligible = frame.loc[frame["eligible"]]
    picked = eligible.loc[
        eligible["t0"].astype(int) % config.WINDOW_LEN == 0
    ].reset_index(drop=True)

    influence = synthetic.influence_matrix(
        n_agents=N_AGENTS, seed=SCENE_SEED, n_real=N_REAL
    )
    model = PlantedPredictor(influence)

    records: list[dict] = []
    for _, row in picked.iterrows():
        records.extend(cache_rows(model, row))
    table = attention_edges.build_table(pd.DataFrame(records), picked, trajectories)
    edges_primary = attention_edges.primary(table)

    by_window = attention_edges.edges_by_window(edges_primary)
    frames = []
    with warnings.catch_warnings():
        # PlantedPredictor.predict takes no mask_policy, so the driver drops
        # weight_zero and warns once. That skip is expected here.
        warnings.simplefilter("ignore", UserWarning)
        for _, row in picked.iterrows():
            frames.append(
                driver.window_curves(row, model, by_window[str(row["window_id"])])
            )
    return picked, edges_primary, pd.concat(frames, ignore_index=True)


_FIXTURE = None


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the cached fixture. The scene builds once for the whole file."""
    global _FIXTURE
    if _FIXTURE is None:
        _FIXTURE = _build_fixture()
    return _FIXTURE


@pytest.fixture(scope="module")
def scene():
    """Return (windows, primary edges, curves) of the planted scene."""
    return fixture()


@pytest.fixture(scope="module")
def cfg():
    """Return the small configuration that every test passes down."""
    return small_config()


@pytest.fixture(scope="module")
def h1_result(scene, cfg):
    """Return the H1 result of the planted scene."""
    _, _, curves = scene
    return h1.run(curves, cfg)


@pytest.fixture(scope="module")
def h2_result(scene, cfg):
    """Return the H2 result of the planted scene."""
    _, edges_primary, curves = scene
    return h2.run(curves, edges_primary, cfg)


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------


def test_the_fixture_holds_more_than_one_component(scene, h1_result):
    """A single component gives no p value below 0.5. See the module docstring."""
    picked, _, _ = scene
    assert len(picked) >= 5
    assert h1_result["primary"]["n_clusters"] >= 3


# ---------------------------------------------------------------------------
# paired_shift
# ---------------------------------------------------------------------------


def test_paired_shift_returns_one_row_per_window(scene):
    _, _, curves = scene
    table = h1.paired_shift(curves, "morf", "lerf")
    assert list(table.columns) == list(h1.PAIRED_COLUMNS)
    assert table["window_id"].nunique() == len(table)
    assert len(table) == curves["window_id"].nunique()


def test_paired_shift_subtracts_the_second_arm(scene):
    _, _, curves = scene
    table = h1.paired_shift(curves, "morf", "lerf")
    assert np.allclose(table["d"], table["shift_a"] - table["shift_b"])


def test_paired_shift_raises_when_a_window_lacks_the_second_arm(scene):
    _, _, curves = scene
    victim = str(curves["window_id"].astype(str).iloc[0])
    broken = curves.loc[
        ~(
            (curves["window_id"].astype(str) == victim)
            & (curves["arm"].astype(str) == "lerf")
            & (curves["n_removed"] == config.PRIMARY_N_REMOVED)
        )
    ]
    with pytest.raises(ValueError, match=victim):
        h1.paired_shift(broken, "morf", "lerf")


def _hand_built_curves() -> pd.DataFrame:
    """Return one window with one morf row and three weight-matched rows.

    The morf row removes a mass of 0.50. The three matched sets reach 0.30,
    0.55 and 0.80. The rule takes the smallest mass at or above 0.50, so the
    partner is the set of mass 0.55 and the shift is 0.20.
    """
    rows = [
        ("morf", 1, 0.50, 0.90),
        ("weight_matched", 2, 0.30, 0.10),
        ("weight_matched", 3, 0.55, 0.20),
        ("weight_matched", 4, 0.80, 0.30),
    ]
    return pd.DataFrame(
        [
            {
                "window_id": "synthetic_000000",
                "ego_id": 3,
                "arm": arm,
                "n_removed": n_removed,
                "removed_mass": mass,
                "edge_src": -1,
                "draw_id": 0,
                "mask_policy": config.PRIMARY_MASK_POLICY,
                "run_id": "test",
                "shift": shift,
            }
            for arm, n_removed, mass, shift in rows
        ]
    )


def test_the_weight_matched_partner_is_the_smallest_set_that_reaches_the_mass():
    table = h1.paired_shift(_hand_built_curves(), "morf", "weight_matched")
    assert len(table) == 1
    assert table["shift_a"].iloc[0] == pytest.approx(0.90)
    assert table["shift_b"].iloc[0] == pytest.approx(0.20)
    assert table["d"].iloc[0] == pytest.approx(0.70)


def test_the_weight_matched_partner_accepts_an_exact_mass_match():
    frame = _hand_built_curves()
    frame.loc[frame["removed_mass"] == 0.55, "removed_mass"] = 0.50
    table = h1.paired_shift(frame, "morf", "weight_matched")
    assert table["shift_b"].iloc[0] == pytest.approx(0.20)


def test_the_weight_matched_partner_is_a_real_row_on_the_planted_scene(scene):
    _, _, curves = scene
    table = h1.paired_shift(curves, "morf", "weight_matched")
    matched = curves.loc[curves["arm"].astype(str) == "weight_matched"]
    known = set(zip(matched["window_id"].astype(str), matched["shift"].astype(float)))
    for row in table.itertuples():
        assert (row.window_id, float(row.shift_b)) in known


# ---------------------------------------------------------------------------
# test_paired
# ---------------------------------------------------------------------------


def test_the_primary_dict_holds_every_required_key(h1_result):
    primary = h1_result["primary"]
    for key in REQUIRED_KEYS:
        assert key in primary, key
    assert primary["hypothesis"] == "h1"
    assert primary["role"] == "primary"
    assert primary["sided"] == config.SIDED


def test_the_primary_p_value_lies_in_the_unit_interval(h1_result):
    assert 0.0 <= h1_result["primary"]["p_raw"] <= 1.0


def test_the_primary_cluster_count_is_an_int_at_or_below_the_window_count(h1_result):
    primary = h1_result["primary"]
    assert isinstance(primary["n_clusters"], int)
    assert 0 < primary["n_clusters"] <= primary["n_windows"]


def test_the_primary_effect_lies_inside_its_own_range(h1_result):
    primary = h1_result["primary"]
    if primary["effect_name"] == "rank_biserial":
        assert -1.0 <= primary["effect"] <= 1.0
    else:
        assert primary["effect_name"] == "share_positive"
        assert 0.0 <= primary["effect"] <= 1.0


def test_h1_rejects_on_the_planted_scene(h1_result):
    """The planted model moves only on a real edge, so MoRF must beat LeRF."""
    primary = h1_result["primary"]
    print(
        f"\nH1 p_raw={primary['p_raw']:.5f}  "
        f"{primary['effect_name']}={primary['effect']:.4f}  "
        f"location_m={primary['location_m']:.5f}  "
        f"n_clusters={primary['n_clusters']}  n_windows={primary['n_windows']}"
    )
    assert primary["p_raw"] < 0.05
    assert primary["effect"] > 0.0
    assert primary["location_m"] > 0.0


def test_h1_returns_two_supporting_dicts(h1_result):
    names = [row["hypothesis"] for row in h1_result["supporting"]]
    assert names == ["h1_morf_vs_random", "h1_morf_vs_weight_matched"]
    for row in h1_result["supporting"]:
        assert row["role"] == "supporting"
        assert 0.0 <= row["p_raw"] <= 1.0
        for key in REQUIRED_KEYS:
            assert key in row, key


def test_h1_returns_the_difference_table_and_the_choice(h1_result):
    assert list(h1_result["d"].columns) == list(h1.PAIRED_COLUMNS)
    assert h1_result["choice"]["test"] in (
        "wilcoxon_signed_rank",
        "sign_test",
    )


def test_the_degenerate_branch_runs_no_test(scene, cfg):
    """Fewer than 3 non-zero differences leave no shape to read."""
    _, _, curves = scene
    table = h1.paired_shift(curves, "morf", "lerf")
    table["shift_b"] = table["shift_a"]
    table["d"] = 0.0

    result = h1.test_paired(table, "h1", "primary", cfg)
    assert result["test_used"] == "none"
    assert result["p_raw"] == 1.0
    assert result["effect"] == 0.0
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])
    assert result["location_m"] == 0.0
    assert "non-zero" in result["why"]
    assert result["n_zero"] == len(table)


def test_the_degenerate_branch_accepts_an_empty_table(cfg):
    empty = pd.DataFrame({name: [] for name in h1.PAIRED_COLUMNS})
    result = h1.test_paired(empty, "h1_morf_vs_random", "supporting", cfg)
    assert result["test_used"] == "none"
    assert result["p_raw"] == 1.0
    assert result["n_clusters"] == 0
    # An empty table names no location, so the branch reports NaN and never a
    # measured 0.0. src/hypotheses/validate.py copies location_m into its
    # table and src/stats/loso.py::heterogeneity drops a NaN, so a fabricated
    # 0.0 would enter the forest plot and the spread as a real estimate.
    assert np.isnan(result["location_m"])


def test_the_degenerate_branch_reports_the_median_that_the_rows_show(cfg):
    """Two non-zero rows give no shape, but they do give a median."""
    table = pd.DataFrame(
        {
            "window_id": ["zara1_000000", "zara1_000020"],
            "ego_id": [1, 1],
            "shift_a": [0.60, 0.70],
            "shift_b": [0.10, 0.10],
            "d": [0.50, 0.60],
        }
    )
    result = h1.test_paired(table, "h1", "primary", cfg)
    assert result["test_used"] == "none"
    assert result["location_m"] == pytest.approx(0.55)


def test_share_positive_counts_only_the_non_zero_rows():
    assert h1.share_positive(np.array([1.0, -1.0, 0.0, 0.0])) == pytest.approx(0.5)
    assert h1.share_positive(np.array([2.0, 3.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(h1.share_positive(np.zeros(4)))


def test_the_alternative_follows_the_sided_setting():
    one = types.SimpleNamespace(SIDED="one")
    two = types.SimpleNamespace(SIDED="two")
    assert h1.alternative_of(one) == "greater"
    assert h1.alternative_of(two) == "two-sided"


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------


def test_h2_returns_the_primary_and_four_supporting_dicts(h2_result):
    assert h2_result["primary"]["hypothesis"] == "h2"
    assert h2_result["primary"]["role"] == "primary"
    names = [row["hypothesis"] for row in h2_result["supporting"]]
    assert names == [
        "h2_disagree_only",
        "h2_spearman_attn_dist",
        "h2_partial_attn_shift_given_dist",
        "h2_attention_shuffle",
    ]
    for row in [h2_result["primary"], *h2_result["supporting"]]:
        assert row["sided"] in ("one", "two")
        for key in REQUIRED_KEYS:
            assert key in row, (row["hypothesis"], key)
    for row in h2_result["supporting"]:
        assert row["role"] == "supporting"


def test_every_h2_p_value_lies_in_the_unit_interval(h2_result):
    every = [h2_result["primary"], *h2_result["supporting"]]
    for row in every:
        assert 0.0 <= row["p_raw"] <= 1.0, row["hypothesis"]
        assert np.isfinite(row["p_raw"]), row["hypothesis"]


def test_the_h2_share_of_agreement_lies_in_the_unit_interval(h2_result):
    share = h2_result["share_agree"]
    print(f"\nH2 share_agree={share:.4f}  p_raw={h2_result['primary']['p_raw']:.5f}")
    assert 0.0 <= share <= 1.0


def test_h2_returns_the_difference_table_and_the_choice(h2_result):
    assert list(h2_result["d"].columns) == list(h1.PAIRED_COLUMNS)
    assert h2_result["choice"] is None or "test" in h2_result["choice"]


def test_the_shuffle_row_names_its_own_statistic(h2_result):
    row = h2_result["supporting"][3]
    assert row["effect_name"] == h2.SHUFFLE_EFFECT_NAME
    assert np.isnan(row["ci_low"])
    assert np.isnan(row["ci_high"])


def test_the_disagree_subset_is_no_larger_than_the_whole(h2_result):
    whole = h2_result["primary"]["n_windows"]
    part = h2_result["supporting"][0]["n_windows"]
    assert 0 <= part <= whole


def test_the_single_rows_hold_one_row_per_edge(scene, cfg):
    _, _, curves = scene
    single = h2.single_rows(curves, cfg)
    assert not single.duplicated(subset=["window_id", "edge_src"]).any()
    assert (single["edge_src"] >= 0).all()


def test_the_top_edge_agreement_reads_every_window(scene):
    _, edges_primary, curves = scene
    window_ids = sorted(set(curves["window_id"].astype(str)))
    agreement = h2.top_edge_agreement(edges_primary, window_ids, config.SEED)
    assert len(agreement) == len(window_ids)
    assert agreement.dtype == bool


# ---------------------------------------------------------------------------
# report.build accepts the whole family
# ---------------------------------------------------------------------------


def test_report_build_accepts_every_row(h1_result, h2_result, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    results = [
        h1_result["primary"],
        h2_result["primary"],
        *h1_result["supporting"],
        *h2_result["supporting"],
    ]
    table = report.build(results)

    assert len(table) == len(results)
    assert list(table.columns) == list(schema.VERDICTS)
    assert table["p_adj"].notna().all()
    assert set(table["verdict"]) <= {"reject", "do not reject"}
    assert (tmp_path / "verdicts.parquet").exists()

    back = schema.read("verdicts", tmp_path)
    assert len(back) == len(results)
    print("\n" + table[["hypothesis", "test_used", "effect", "p_raw", "p_adj",
                        "verdict"]].to_string(index=False))
