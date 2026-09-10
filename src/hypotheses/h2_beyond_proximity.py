"""H2: attention beats proximity, not just chance.

See CONTRACT.md section 5.10 and docs/FINISH_PLAN.md sections 3.4 and 3.6 for
the signatures. Every number comes from config.py.

THE CLAIM. D_i = shift(MoRF) - shift(Nearest) at config.PRIMARY_N_REMOVED, one
value per window. H0-2 says that removal of the most-attended edge gives no
larger shift than removal of the nearest edge. The arm comparison is the
primary test.

WHY EVERY WINDOW STAYS IN THE PRIMARY TEST. On many windows the two arms pick
the SAME edge, so D is exactly 0. That zero is a structural tie and it is real
data: it says that attention adds nothing on that window. The Pratt rule ranks
such a tie and adds 0 to the signed sum, so the tie pulls the test toward the
null. A test that dropped those windows would report the effect of the
disagreement alone and would call it the effect of attention.
`h2_disagree_only` reports that second number on its own, as a supporting row,
so a reader sees both.

THE FOUR SUPPORTING ROWS.

    h2_disagree_only                    the same paired test on the windows
                                        where the two arms pick a different
                                        edge.
    h2_spearman_attn_dist               how strongly attention tracks distance.
                                        A strong negative rho says that
                                        attention is close to a proximity rule.
    h2_partial_attn_shift_given_dist    the link of attention with the
                                        single-edge shift, with distance held
                                        fixed.
    h2_attention_shuffle                the permutation control. It breaks the
                                        link of attention with the shift and
                                        holds the graph shape.

None of the four enters the primary family. `src/stats/report.py::build` puts
them under Benjamini-Hochberg and puts h1, h2 and h3 under Holm.

THE INTERVALS. Every interval comes from the pairs cluster bootstrap of
`src/stats/clustered.py`. The bootstrap takes a one-dimensional array, so a
correlation of two columns enters as the ROW POSITION and the statistic reads
the two columns back. That keeps the two columns of one row together in every
resample, which is the whole point of a pairs bootstrap.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config
from src.ablation.arms import order_edges
from src.attention import edges as attention_edges
from src.hypotheses import h1_morf_vs_lerf as h1
from src.stats import clustered, effects, permutation

# The statistic name of the attention shuffle test.
SHUFFLE_EFFECT_NAME = "median_d_top_minus_bottom_m"

# The sentence of each supporting correlation.
SPEARMAN_WHY = (
    "The Spearman rank correlation measures the monotone link of attention "
    "with distance, and the pairs cluster bootstrap gives the interval."
)
PARTIAL_WHY = (
    "The partial correlation holds the distance fixed, so it isolates the link "
    "of attention with the single-edge shift."
)
SHUFFLE_WHY = (
    "The shuffle test permutes the attention weight inside the scene, so it "
    "breaks the link of attention with the shift and holds the graph shape."
)
NO_ROW_WHY = "No window carries the rows that this test needs, so no test runs."


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value: float, fallback: float) -> float:
    """Return `value` when it is finite, else `fallback`.

    A p value of NaN breaks the sort of `src/stats/correction.py::holm` and of
    Benjamini-Hochberg, so every p value passes through this guard.
    """
    value = float(value)
    return value if np.isfinite(value) else float(fallback)


def _cluster_count(window_id: pd.Series, ego_id: pd.Series) -> int:
    """Return the component count of the rows that a test reads."""
    table = pd.DataFrame(
        {
            "window_id": pd.Series(window_id).astype(str).to_numpy(),
            "ego_id": pd.Series(ego_id).astype("int64").to_numpy(),
        }
    )
    return clustered.n_clusters(clustered.component_clusters(table))


def single_rows(curves: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Return the single-arm rows under the primary mask policy.

    Columns: window_id, ego_id, edge_src, shift. The single arm removes one
    named edge at a time, so one row names one edge. The frame holds one draw
    only. A table with more than one draw keeps the lowest draw_id, because
    `src/stats/permutation.py::attention_shuffle_p` needs one row per edge.
    """
    block = curves.loc[
        (curves["arm"].astype(str) == "single")
        & (curves["mask_policy"].astype(str) == str(cfg.PRIMARY_MASK_POLICY))
    ]
    out = block.loc[:, ["window_id", "ego_id", "edge_src", "draw_id", "shift"]].copy()
    out["window_id"] = out["window_id"].astype(str)
    out["ego_id"] = out["ego_id"].astype("int64")
    out["edge_src"] = out["edge_src"].astype("int64")
    out["shift"] = out["shift"].astype("float64")
    if out["draw_id"].nunique() > 1:
        keep = int(out["draw_id"].min())
        out = out.loc[out["draw_id"] == keep]
    return out.loc[:, ["window_id", "ego_id", "edge_src", "shift"]].reset_index(
        drop=True
    )


def top_edge_agreement(
    edges_primary: pd.DataFrame, window_ids, seed: int = config.SEED
) -> pd.Series:
    """Return one bool per window: the two arms pick the same first edge.

    `src/ablation/arms.py::order_edges` gives the removal order of MoRF and of
    Nearest. The first entry of each order is the edge that the arm removes at
    n_removed 1. The index of the result is the window id.
    """
    by_window = attention_edges.edges_by_window(edges_primary)
    labels: list[str] = []
    values: list[bool] = []
    for window_id in window_ids:
        block = by_window.get(str(window_id))
        if block is None or len(block) == 0:
            continue
        top_attn = order_edges(block, "morf", int(seed))[0]
        top_near = order_edges(block, "nearest", int(seed))[0]
        labels.append(str(window_id))
        values.append(bool(top_attn == top_near))
    return pd.Series(values, index=labels, dtype=bool)


# ---------------------------------------------------------------------------
# The three correlation rows
# ---------------------------------------------------------------------------


def _bootstrap_correlation(
    statistic, n_rows: int, clusters: np.ndarray, cfg
) -> tuple[float, float]:
    """Return the pairs cluster bootstrap interval of a two-column statistic.

    `src/stats/clustered.py::cluster_bootstrap_ci` ravels its `values`
    argument, so a two-column input cannot pass through it. The row POSITION
    passes through instead, and `statistic` reads the two columns back from
    that position. Every resample therefore keeps the two values of one row
    together, which is what a pairs bootstrap must do.
    """
    positions = np.arange(n_rows, dtype=np.float64)

    def on_rows(drawn: np.ndarray) -> float:
        """Return the statistic of the drawn rows, or NaN when it is undefined."""
        index = np.asarray(drawn, dtype=np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return float(statistic(index))
            except (ValueError, FloatingPointError):
                return float("nan")

    return clustered.cluster_bootstrap_ci(
        positions,
        clusters,
        stat=on_rows,
        n_boot=int(cfg.N_BOOT),
        seed=int(cfg.SEED),
    )


def _empty_supporting(hypothesis: str, effect_name: str, sided: str) -> dict:
    """Return the result dict of a supporting test that reads no row."""
    return {
        "hypothesis": hypothesis,
        "role": "supporting",
        "test_used": "none",
        "why": NO_ROW_WHY,
        "sided": sided,
        "effect": 0.0,
        "effect_name": effect_name,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": 1.0,
        "n_clusters": 0,
        "location_m": float("nan"),
        "location_ci_low": float("nan"),
        "location_ci_high": float("nan"),
        "n_windows": 0,
    }


def spearman_attn_dist(edges_primary: pd.DataFrame, cfg=config) -> dict:
    """Return the Spearman correlation of attention against distance.

    The rows are the primary edges. `dst` is the ego of the window, so it is
    the ego id that `component_clusters` needs. A strong negative rho says that
    attention mostly repeats the proximity order, and H2 is then weak by
    construction.
    """
    frame = edges_primary.loc[
        :, ["window_id", "dst", "attn", "dist_at_last_frame"]
    ].copy()
    if len(frame) < 3:
        return _empty_supporting("h2_spearman_attn_dist", "spearman_rho", "two")

    attn = frame["attn"].to_numpy(dtype=np.float64)
    distance = frame["dist_at_last_frame"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p_value = effects.spearman(attn, distance)

    clusters = clustered.component_clusters(
        pd.DataFrame(
            {
                "window_id": frame["window_id"].astype(str).to_numpy(),
                "ego_id": frame["dst"].astype("int64").to_numpy(),
            }
        )
    )
    ci_low, ci_high = _bootstrap_correlation(
        lambda index: effects.spearman(attn[index], distance[index])[0],
        len(frame),
        clusters,
        cfg,
    )

    return {
        "hypothesis": "h2_spearman_attn_dist",
        "role": "supporting",
        "test_used": "spearman rank correlation, pairs cluster bootstrap",
        "why": SPEARMAN_WHY,
        "sided": "two",
        "effect": _finite(rho, 0.0),
        "effect_name": "spearman_rho",
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_raw": _finite(p_value, 1.0),
        "n_clusters": clustered.n_clusters(clusters),
        # A correlation is a unit-free number, so it names no location in
        # metres. The three location keys stay NaN, so every result dict of
        # this project holds the same keys.
        "location_m": float("nan"),
        "location_ci_low": float("nan"),
        "location_ci_high": float("nan"),
        "n_windows": int(frame["window_id"].nunique()),
        "n_edges": int(len(frame)),
    }


def _edges_with_shift(
    edges_primary: pd.DataFrame, single: pd.DataFrame
) -> pd.DataFrame:
    """Return the primary edges that carry a single-arm shift.

    The join key is (window_id, src) against (window_id, edge_src). The result
    holds window_id, src, ego_id, attn, dist_at_last_frame and shift.
    """
    left = edges_primary.loc[
        :, ["window_id", "src", "dst", "attn", "dist_at_last_frame"]
    ].copy()
    left["window_id"] = left["window_id"].astype(str)
    left["src"] = left["src"].astype("int64")
    left["ego_id"] = left["dst"].astype("int64")

    right = single.rename(columns={"edge_src": "src"}).loc[
        :, ["window_id", "src", "shift"]
    ]
    joined = left.merge(right, on=["window_id", "src"], how="inner")
    return joined.loc[
        :, ["window_id", "src", "ego_id", "attn", "dist_at_last_frame", "shift"]
    ].reset_index(drop=True)


def partial_attn_shift_given_dist(with_shift: pd.DataFrame, cfg=config) -> dict:
    """Return the partial correlation of attention with the shift, given distance.

    `with_shift` is the output of `_edges_with_shift`. The covariate is the
    distance at the last observed frame. A correlation above 0 says that
    attention carries information about the shift that distance does not.
    """
    if len(with_shift) < 4:
        return _empty_supporting(
            "h2_partial_attn_shift_given_dist", "partial_corr", "two"
        )

    attn = with_shift["attn"].to_numpy(dtype=np.float64)
    shift = with_shift["shift"].to_numpy(dtype=np.float64)
    distance = with_shift["dist_at_last_frame"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate, p_value = effects.partial_corr(attn, shift, distance)

    clusters = clustered.component_clusters(
        with_shift.loc[:, ["window_id", "ego_id"]]
    )
    ci_low, ci_high = _bootstrap_correlation(
        lambda index: effects.partial_corr(
            attn[index], shift[index], distance[index]
        )[0],
        len(with_shift),
        clusters,
        cfg,
    )

    return {
        "hypothesis": "h2_partial_attn_shift_given_dist",
        "role": "supporting",
        "test_used": "partial correlation given distance, pairs cluster bootstrap",
        "why": PARTIAL_WHY,
        "sided": "two",
        "effect": _finite(estimate, 0.0),
        "effect_name": "partial_corr",
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_raw": _finite(p_value, 1.0),
        "n_clusters": clustered.n_clusters(clusters),
        # See spearman_attn_dist for why the location keys stay NaN.
        "location_m": float("nan"),
        "location_ci_low": float("nan"),
        "location_ci_high": float("nan"),
        "n_windows": int(with_shift["window_id"].nunique()),
        "n_edges": int(len(with_shift)),
    }


def attention_shuffle(
    edges_primary: pd.DataFrame,
    single: pd.DataFrame,
    with_shift: pd.DataFrame,
    cfg=config,
) -> dict:
    """Return the attention shuffle control of H2.

    `src/stats/permutation.py::attention_shuffle_p` permutes the attention
    weight inside the scene and reads the shift of the new top edge and of the
    new bottom edge out of the single arm. It needs no forward pass. The
    statistic is the median over windows of D = shift(top) - shift(bottom).

    The test needs two edges per window, so the cluster count below reads only
    the windows that clear that bar.
    """
    sided = "one"
    edges_in = edges_primary.loc[:, ["window_id", "src", "attn"]].copy()
    edges_in["window_id"] = edges_in["window_id"].astype(str)
    edges_in["src"] = edges_in["src"].astype("int64")

    counts = with_shift.groupby("window_id")["src"].transform("size")
    entered = with_shift.loc[counts >= 2]
    if len(entered) == 0:
        return _empty_supporting("h2_attention_shuffle", SHUFFLE_EFFECT_NAME, sided)

    result = permutation.attention_shuffle_p(
        edges_in,
        single.loc[:, ["window_id", "edge_src", "shift"]],
        n_shuffle=int(cfg.N_PERM),
        seed=int(cfg.SEED),
        alternative="greater",
    )
    per_window = entered.loc[:, ["window_id", "ego_id"]].drop_duplicates()

    return {
        "hypothesis": "h2_attention_shuffle",
        "role": "supporting",
        "test_used": "attention shuffle permutation inside the scene",
        "why": SHUFFLE_WHY,
        "sided": sided,
        "effect": _finite(result["stat_obs"], 0.0),
        "effect_name": SHUFFLE_EFFECT_NAME,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": _finite(result["p"], 1.0),
        "n_clusters": _cluster_count(per_window["window_id"], per_window["ego_id"]),
        # The statistic of this test IS a median difference of two shifts, so
        # it already carries metres. The interval needs a second permutation
        # scheme, which this test does not run, so the two bounds stay NaN.
        "location_m": _finite(result["stat_obs"], float("nan")),
        "location_ci_low": float("nan"),
        "location_ci_high": float("nan"),
        "n_windows": int(result["n_windows"]),
        "stat_obs": float(result["stat_obs"]),
    }


# ---------------------------------------------------------------------------
# The hypothesis
# ---------------------------------------------------------------------------


def run(curves: pd.DataFrame, edges: pd.DataFrame, cfg=config) -> dict:
    """Return the primary test of H2 and its four supporting tests.

    `curves` is the perturbation_curves table. `edges` is the attention_edges
    table. The function slices `edges` to cfg.PRIMARY_MODULE and
    cfg.PRIMARY_TIME_AGG, so a caller may pass the whole table or the primary
    slice.

    The primary test compares MoRF against Nearest over EVERY window, ties
    included. See the module docstring for why the ties stay in.

    Returns {"primary": dict, "supporting": [4 dicts], "d": DataFrame,
    "choice": the choose() output of the primary test, "share_agree": float}.
    `share_agree` is the share of windows where the two arms pick the same
    first edge. A share near 1 says that H2 is weak before any test runs.
    """
    edges_primary = attention_edges.primary(
        edges, str(cfg.PRIMARY_MODULE), str(cfg.PRIMARY_TIME_AGG)
    )
    n_removed = int(cfg.PRIMARY_N_REMOVED)
    policy = str(cfg.PRIMARY_MASK_POLICY)

    d_primary = h1.paired_shift(curves, "morf", "nearest", n_removed, policy)
    primary = h1.test_paired(d_primary, "h2", "primary", cfg)

    agreement = top_edge_agreement(
        edges_primary, d_primary["window_id"].astype(str), int(cfg.SEED)
    )
    share_agree = float(agreement.mean()) if len(agreement) else float("nan")

    disagree = set(agreement.index[~agreement.to_numpy()])
    d_disagree = d_primary.loc[
        d_primary["window_id"].astype(str).isin(disagree)
    ].reset_index(drop=True)
    only_disagree = h1.test_paired(d_disagree, "h2_disagree_only", "supporting", cfg)

    single = single_rows(curves, cfg)
    with_shift = _edges_with_shift(edges_primary, single)

    return {
        "primary": primary,
        "supporting": [
            only_disagree,
            spearman_attn_dist(edges_primary, cfg),
            partial_attn_shift_given_dist(with_shift, cfg),
            attention_shuffle(edges_primary, single, with_shift, cfg),
        ],
        "d": d_primary,
        "choice": primary["choice"],
        "share_agree": share_agree,
    }
