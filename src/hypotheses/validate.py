"""The sensitivity table. Every row is descriptive. No row is a new test.

See CONTRACT.md section 5.10 and docs/FINISH_PLAN.md section 3.7 for the
signature. Every number comes from config.py.

WHAT THIS FILE ANSWERS. The primary tests of H1 and H2 read one module, one
time collapse, one mask policy and one removal step. A reader must know
whether the answer survives another choice. This file makes that same
comparison again under every other choice and puts the answers in one table.

WHY NO P VALUE HERE ENTERS A FAMILY. `src/stats/correction.py` corrects the
three primary rows h1, h2 and h3 under Holm. A leave one scene out row is not
a sixth hypothesis. It is the same hypothesis on a smaller sample. A reader
looks at the DIRECTION and the SPREAD of these rows, never at one p value of
one scene.

WHY EVERY ROW GOES THROUGH `h1_morf_vs_lerf.test_paired`. One effect
definition must serve the whole study. `test_paired` reads the shape of D with
`choose_test.choose`, takes the p value from the cluster sign flip and takes
the interval from the pairs cluster bootstrap. This file builds a D table for
every row and hands it to that one function, so a row of this table and the
primary row of the same hypothesis carry the same effect, the same test and
the same cluster unit.

WHY THE SENSITIVITY ROWS COST NO FORWARD PASS. At config.PRIMARY_N_REMOVED of
1 every arm removes exactly one edge, and the `single` arm already measured
every edge of every window on its own. Another module, another time collapse
and another mask policy therefore only change WHICH edge the arm picks. The
shift of that edge is a lookup into the single rows. See
docs/FINISH_PLAN.md section 2.

WHICH INTERVAL SITS IN ci_low AND ci_high. The location interval, in metres,
of the same row. `src/hypotheses/figures.py::loso_figure` draws a forest plot
of `location_m` with a bar from `ci_low` to `ci_high`, so the two bounds must
belong to `location_m` and not to the unit-free effect. The effect keeps its
own interval inside the result dict of `test_paired`, under location free
names, and this table does not repeat it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.ablation.arms import order_edges
from src.attention import edges as attention_edges
from src.data import schema
from src.hypotheses import h1_morf_vs_lerf as h1
from src.stats import clustered, loso

# The columns of the table, in order. docs/FINISH_PLAN.md section 3.7 fixes
# the first eleven. test_used names the test that produced the row, so a
# reader sees when the shape of D changed the test between two variants.
COLUMNS = (
    "check",
    "variant",
    "hypothesis",
    "n_windows",
    "n_clusters",
    "effect_name",
    "effect",
    "location_m",
    "ci_low",
    "ci_high",
    "p_raw",
    "test_used",
)

# The steps of the n_removed sweep. Step 1 is the primary step, so the sweep
# shows how far the answer reaches beyond it.
N_REMOVED_VARIANTS = (1, 2, 3, 4, 5)

# The two attention modules that the primary tests do not read.
MODULE_VARIANTS = ("decoder", "cross")

# The two time collapses that the primary tests do not read.
TIME_AGG_VARIANTS = ("max", "last")

# The second mask policy of config.MASK_POLICIES.
SECOND_MASK_POLICY = "weight_zero"

# The two hypotheses that this table repeats.
HYPOTHESES = ("h1", "h2")

# The test_used text of the three rows that run no test.
NO_ROW_TEST = "none, the rows that this variant needs are absent"
HETEROGENEITY_TEST = "i squared over the per scene location, equal weights"
NOISE_FLOOR_TEST = "share of windows, no test"

# The effect name of the two rows that carry no paired test.
HETEROGENEITY_EFFECT = "i_squared"
NOISE_FLOOR_EFFECT = "share_morf_above_random"

# The effect name of a row that reads no data.
EMPTY_EFFECT = "none"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _row(check: str, variant: str, hypothesis: str, result: dict) -> dict:
    """Return one table row from the result dict of `test_paired`.

    `ci_low` and `ci_high` take the LOCATION interval in metres. See the
    module docstring for the reason.
    """
    return {
        "check": str(check),
        "variant": str(variant),
        "hypothesis": str(hypothesis),
        "n_windows": int(result["n_windows"]),
        "n_clusters": int(result["n_clusters"]),
        "effect_name": str(result["effect_name"]),
        "effect": float(result["effect"]),
        "location_m": float(result["location_m"]),
        "ci_low": float(result["location_ci_low"]),
        "ci_high": float(result["location_ci_high"]),
        "p_raw": float(result["p_raw"]),
        "test_used": str(result["test_used"]),
    }


def _blank_row(
    check: str,
    variant: str,
    hypothesis: str,
    test_used: str = NO_ROW_TEST,
    effect_name: str = EMPTY_EFFECT,
) -> dict:
    """Return one table row that reads no data.

    The row keeps its place in the table, so a reader sees that the variant
    ran and found nothing. A silent gap would look like a variant that nobody
    tried.
    """
    return {
        "check": str(check),
        "variant": str(variant),
        "hypothesis": str(hypothesis),
        "n_windows": 0,
        "n_clusters": 0,
        "effect_name": str(effect_name),
        "effect": float("nan"),
        "location_m": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": float("nan"),
        "test_used": str(test_used),
    }


def _print_row(row: dict) -> None:
    """Print one line that names the row and its two headline numbers."""
    print(
        f"  {row['check']:<14} {row['variant']:<12} {row['hypothesis']:<3} "
        f"n={_text(row['n_windows'])!s:>5} clusters={_text(row['n_clusters'])!s:>4} "
        f"{row['effect_name']}={_number(row['effect'])} "
        f"location_m={_number(row['location_m'])} "
        f"p={_number(row['p_raw'])}",
        flush=True,
    )


def _text(value) -> str:
    """Return a count as text. An absent count prints as a dash."""
    if value is None:
        return "-"
    return str(value)


def _number(value: float) -> str:
    """Return a float with five decimals, or a dash when it is not finite."""
    value = float(value)
    if not np.isfinite(value):
        return "-"
    return f"{value:.5f}"


# ---------------------------------------------------------------------------
# The single arm lookup
# ---------------------------------------------------------------------------


def single_lookup(
    curves: pd.DataFrame, mask_policy: str = config.PRIMARY_MASK_POLICY
) -> dict[tuple[str, int], float]:
    """Return (window_id, edge_src) -> shift for the single arm of one policy.

    The `single` arm removes one named edge at a time, so one row names one
    edge. `edge_src` holds the removed source id and `n_removed` holds the
    position in the sweep. See CONTRACT.md section 3 and
    `src/ablation/curves.py`.

    Every sensitivity variant at config.PRIMARY_N_REMOVED of 1 is a lookup
    into this map, because at step 1 an arm removes exactly one edge and the
    sweep already measured that edge.

    A table with more than one draw keeps the lowest draw_id, so the map holds
    exactly one shift per edge. Return an empty map when the policy wrote no
    row, which is what a CPU run with MockPredictor gives for weight_zero.
    """
    for column in ("window_id", "arm", "edge_src", "mask_policy", "shift"):
        if column not in curves.columns:
            raise KeyError(
                f"curves must carry a {column!r} column. See CONTRACT.md "
                "section 3, perturbation_curves."
            )

    block = curves.loc[
        (curves["arm"].astype(str) == "single")
        & (curves["mask_policy"].astype(str) == str(mask_policy))
    ]
    if len(block) == 0:
        return {}

    frame = block.loc[:, ["window_id", "edge_src", "draw_id", "shift"]].copy()
    if frame["draw_id"].nunique() > 1:
        frame = frame.loc[frame["draw_id"] == int(frame["draw_id"].min())]

    return {
        (str(window_id), int(edge_src)): float(shift)
        for window_id, edge_src, shift in zip(
            frame["window_id"], frame["edge_src"], frame["shift"]
        )
    }


def _ego_of_window(curves: pd.DataFrame) -> dict[str, int]:
    """Return window_id -> ego_id.

    `src/stats/clustered.py::component_clusters` needs the ego of every row,
    and the curves table carries it on every row of the window.
    """
    frame = curves.loc[:, ["window_id", "ego_id"]].copy()
    frame["window_id"] = frame["window_id"].astype(str)
    frame = frame.drop_duplicates(subset=["window_id"], keep="first")
    return {
        str(window_id): int(ego_id)
        for window_id, ego_id in zip(frame["window_id"], frame["ego_id"])
    }


# ---------------------------------------------------------------------------
# One sensitivity variant
# ---------------------------------------------------------------------------


def variant_d(
    curves: pd.DataFrame,
    edges: pd.DataFrame,
    hypothesis: str,
    module: str = config.PRIMARY_MODULE,
    time_agg: str = config.PRIMARY_TIME_AGG,
    mask_policy: str = config.PRIMARY_MASK_POLICY,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, int]:
    """Return the D table of one variant, and the count of skipped windows.

    The module check, the time_agg check and the mask_policy check share this
    function. All three ask the same question at n_removed 1: which edge does
    the arm pick under this choice, and how far did the forecast move when the
    sweep removed that edge.

    `edges` is the attention_edges table. The function slices it with
    `src/attention/edges.py::primary` to `module` and `time_agg`, so a caller
    passes the whole table. `src/ablation/arms.py::order_edges` names the top
    attention edge and the bottom attention edge of that slice, and it names
    the edge with rank_dist 1.

    For h1: d = shift(top attention edge) - shift(bottom attention edge).
    For h2: d = shift(top attention edge) - shift(edge with rank_dist 1).

    Every shift is a lookup into the single rows under `mask_policy`, so the
    function runs no forward pass. The function skips a window that lacks a
    single row for either edge, and it counts that window. The second element
    of the result holds that count. A skip marks a hole in the curves, so the
    caller reports it.

    The columns of the D table are the columns of
    `h1_morf_vs_lerf.PAIRED_COLUMNS`, so `test_paired` accepts it directly.
    """
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"hypothesis must be one of {HYPOTHESES}; got {hypothesis!r}")

    lookup = single_lookup(curves, mask_policy)
    ego_of = _ego_of_window(curves)
    slice_of_variant = attention_edges.primary(edges, module, time_agg)
    by_window = attention_edges.edges_by_window(slice_of_variant)

    rows: list[dict] = []
    skipped = 0
    for window_id in sorted(by_window):
        block = by_window[window_id]
        top = int(order_edges(block, "morf", int(seed))[0])
        if hypothesis == "h1":
            other = int(order_edges(block, "lerf", int(seed))[0])
        else:
            other = int(order_edges(block, "nearest", int(seed))[0])

        key_top = (window_id, top)
        key_other = (window_id, other)
        if key_top not in lookup or key_other not in lookup or window_id not in ego_of:
            skipped += 1
            continue

        shift_a = float(lookup[key_top])
        shift_b = float(lookup[key_other])
        rows.append(
            {
                "window_id": window_id,
                "ego_id": int(ego_of[window_id]),
                "shift_a": shift_a,
                "shift_b": shift_b,
                "d": shift_a - shift_b,
            }
        )

    table = pd.DataFrame(rows, columns=list(h1.PAIRED_COLUMNS))
    if len(table) == 0:
        table = table.astype(
            {
                "window_id": "object",
                "ego_id": "int64",
                "shift_a": "float64",
                "shift_b": "float64",
                "d": "float64",
            }
        )
    return table, int(skipped)


# ---------------------------------------------------------------------------
# The leave one scene out block
# ---------------------------------------------------------------------------


def _with_scene(d_table: pd.DataFrame) -> pd.DataFrame:
    """Return the D table with a scene column.

    `src/stats/loso.py::leave_one_scene_out` groups on that column, and
    `src/data/schema.py::split_window_id` reads the scene out of the key.
    """
    out = d_table.copy()
    out["scene"] = [
        schema.split_window_id(str(window_id))[0] for window_id in out["window_id"]
    ]
    return out


def _scene_estimate(hypothesis: str, cfg):
    """Return the function that `leave_one_scene_out` calls once per scene.

    The function runs `test_paired` on the rows of one scene alone and returns
    a flat dict of the numbers that the table needs. `n_clusters_test` carries
    the component count of `test_paired`, not the ego count that
    `leave_one_scene_out` adds beside it. The component is the frozen cluster
    unit of config.CLUSTER_UNIT, so it is the honest number to print.
    """

    def estimate(part: pd.DataFrame) -> dict:
        """Return the numbers of one held out scene."""
        result = h1.test_paired(
            part.loc[:, list(h1.PAIRED_COLUMNS)], hypothesis, "supporting", cfg
        )
        return {
            "n_windows": int(result["n_windows"]),
            "n_clusters_test": int(result["n_clusters"]),
            "effect_name": str(result["effect_name"]),
            "effect": float(result["effect"]),
            "location_m": float(result["location_m"]),
            "location_ci_low": float(result["location_ci_low"]),
            "location_ci_high": float(result["location_ci_high"]),
            "p_raw": float(result["p_raw"]),
            "test_used": str(result["test_used"]),
        }

    return estimate


def _loso_rows(d_table: pd.DataFrame, hypothesis: str, cfg) -> tuple[list[dict], list[float]]:
    """Return the loso rows of one hypothesis and the per scene locations.

    The second element feeds `src/stats/loso.py::heterogeneity`.
    """
    if len(d_table) == 0:
        return [], []

    estimates = loso.leave_one_scene_out(_with_scene(d_table), _scene_estimate(hypothesis, cfg))

    rows: list[dict] = []
    locations: list[float] = []
    for record in estimates.to_dict("records"):
        locations.append(float(record["location_m"]))
        rows.append(
            {
                "check": "loso",
                "variant": str(record["scene"]),
                "hypothesis": hypothesis,
                "n_windows": int(record["n_windows"]),
                "n_clusters": int(record["n_clusters_test"]),
                "effect_name": str(record["effect_name"]),
                "effect": float(record["effect"]),
                "location_m": float(record["location_m"]),
                "ci_low": float(record["location_ci_low"]),
                "ci_high": float(record["location_ci_high"]),
                "p_raw": float(record["p_raw"]),
                "test_used": str(record["test_used"]),
            }
        )
    return rows, locations


def _heterogeneity_row(locations: list[float], hypothesis: str) -> dict:
    """Return the heterogeneity row of one hypothesis.

    `effect` holds i_squared and `location_m` holds the range of the per scene
    location, in metres. Every other number stays NaN, because this row runs
    no test and names no interval. A NaN p value is safe here, because no row
    of this table enters a correction family.
    """
    spread = loso.heterogeneity(pd.Series(locations, dtype="float64"))
    return {
        "check": "heterogeneity",
        "variant": hypothesis,
        "hypothesis": hypothesis,
        "n_windows": None,
        "n_clusters": None,
        "effect_name": HETEROGENEITY_EFFECT,
        "effect": float(spread["i_squared"]),
        "location_m": float(spread["range"]),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": float("nan"),
        "test_used": HETEROGENEITY_TEST,
    }


# ---------------------------------------------------------------------------
# The n_removed sweep
# ---------------------------------------------------------------------------


def _windows_at_step(
    curves: pd.DataFrame, arm_a: str, arm_b: str, n_removed: int, mask_policy: str
) -> set[str]:
    """Return the windows that hold both arms at one step under one policy.

    An inner join of the two arms gives that set. `paired_shift` raises when a
    window holds one arm and not the other, so the sweep hands it the
    intersection and never the union. A window with E edges holds E steps, so
    a deep step reads fewer windows than a shallow one.
    """
    block = curves.loc[
        (curves["mask_policy"].astype(str) == str(mask_policy))
        & (curves["n_removed"] == int(n_removed))
    ]
    arms = block["arm"].astype(str)
    left = set(block.loc[arms == str(arm_a), "window_id"].astype(str))
    right = set(block.loc[arms == str(arm_b), "window_id"].astype(str))
    return left & right


def _n_removed_rows(curves: pd.DataFrame, cfg) -> list[dict]:
    """Return one row per step of the n_removed sweep, for H1."""
    policy = str(cfg.PRIMARY_MASK_POLICY)
    rows: list[dict] = []
    for step in N_REMOVED_VARIANTS:
        keep = _windows_at_step(curves, "morf", "lerf", step, policy)
        if not keep:
            rows.append(_blank_row("n_removed", str(step), "h1"))
            continue
        block = curves.loc[curves["window_id"].astype(str).isin(keep)]
        d_table = h1.paired_shift(block, "morf", "lerf", step, policy)
        result = h1.test_paired(d_table, "h1", "supporting", cfg)
        rows.append(_row("n_removed", str(step), "h1", result))
    return rows


# ---------------------------------------------------------------------------
# The noise floor
# ---------------------------------------------------------------------------


def _noise_floor_row(curves: pd.DataFrame, cfg) -> dict:
    """Return the noise floor row: does MoRF beat Random at step 1.

    `effect` is the share of windows where the MoRF shift exceeds the Random
    shift. `location_m` is the median MoRF shift at step 1, in metres.
    AgentFormer is deterministic at inference, so the noise floor of the draws
    is exactly 0 and any shift above 0 is a real change. See
    docs/FINISH_PLAN.md section 2. The row therefore names a share and a size,
    and it runs no test, so the interval and the p value stay NaN.

    If MoRF does not beat Random here, the answer of the study is "attention
    is not faithful". That is a result, not a failure. See CONTRACT.md
    section 5.10.
    """
    table = h1.paired_shift(
        curves,
        "morf",
        "random",
        int(cfg.PRIMARY_N_REMOVED),
        str(cfg.PRIMARY_MASK_POLICY),
    )
    if len(table) == 0:
        return _blank_row(
            "noise_floor", "morf_vs_random", "h1", NOISE_FLOOR_TEST, NOISE_FLOOR_EFFECT
        )

    morf = table["shift_a"].to_numpy(dtype=np.float64)
    random_arm = table["shift_b"].to_numpy(dtype=np.float64)
    labels = clustered.component_clusters(table)

    return {
        "check": "noise_floor",
        "variant": "morf_vs_random",
        "hypothesis": "h1",
        "n_windows": int(len(table)),
        "n_clusters": int(clustered.n_clusters(labels)),
        "effect_name": NOISE_FLOOR_EFFECT,
        "effect": float(np.mean(morf > random_arm)),
        "location_m": float(np.median(morf)),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": float("nan"),
        "test_used": NOISE_FLOOR_TEST,
    }


# ---------------------------------------------------------------------------
# The variant blocks
# ---------------------------------------------------------------------------


def _variant_rows(
    curves: pd.DataFrame,
    edges: pd.DataFrame,
    check: str,
    variant: str,
    module: str,
    time_agg: str,
    mask_policy: str,
    cfg,
) -> list[dict]:
    """Return the H1 row and the H2 row of one sensitivity variant."""
    rows: list[dict] = []
    for hypothesis in HYPOTHESES:
        d_table, skipped = variant_d(
            curves,
            edges,
            hypothesis,
            module=module,
            time_agg=time_agg,
            mask_policy=mask_policy,
            seed=int(cfg.SEED),
        )
        if skipped:
            print(
                f"note: the check {check!r}, variant {variant!r}, {hypothesis} "
                f"skips {skipped} windows, because the single arm under the "
                f"mask policy {mask_policy!r} holds no row for one of the two "
                "edges.",
                flush=True,
            )
        if len(d_table) == 0:
            rows.append(_blank_row(check, variant, hypothesis))
            continue
        result = h1.test_paired(d_table, hypothesis, "supporting", cfg)
        rows.append(_row(check, variant, hypothesis, result))
    return rows


def _mask_policy_rows(curves: pd.DataFrame, edges: pd.DataFrame, cfg) -> list[dict]:
    """Return the two rows of the second mask policy.

    A CPU run with MockPredictor writes no weight_zero row, because that model
    cannot tell the two policies apart. The function then writes both rows
    with n_windows 0 and NaN numbers and prints a note. It never raises,
    because a missing second policy is a known state of a CPU run and not a
    bug. See `src/ablation/driver.py::window_curves`.
    """
    if not single_lookup(curves, SECOND_MASK_POLICY):
        print(
            f"note: the curves hold no single row under the mask policy "
            f"{SECOND_MASK_POLICY!r}, so the mask_policy check writes an empty "
            "row for H1 and for H2. A CPU run with MockPredictor gives this "
            "state.",
            flush=True,
        )
        return [
            _blank_row("mask_policy", SECOND_MASK_POLICY, hypothesis)
            for hypothesis in HYPOTHESES
        ]

    return _variant_rows(
        curves,
        edges,
        "mask_policy",
        SECOND_MASK_POLICY,
        str(cfg.PRIMARY_MODULE),
        str(cfg.PRIMARY_TIME_AGG),
        SECOND_MASK_POLICY,
        cfg,
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def _as_table(rows: list[dict]) -> pd.DataFrame:
    """Return the rows as a frame with the exact columns of COLUMNS.

    `n_windows` and `n_clusters` take the nullable integer dtype, so the
    heterogeneity row holds a missing count and every other row keeps a whole
    number.
    """
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    for name in ("n_windows", "n_clusters"):
        frame[name] = pd.array(list(frame[name]), dtype="Int64")
    for name in ("effect", "location_m", "ci_low", "ci_high", "p_raw"):
        frame[name] = frame[name].astype("float64")
    for name in ("check", "variant", "hypothesis", "effect_name", "test_used"):
        frame[name] = frame[name].astype(str)
    return frame.reset_index(drop=True)


def run(
    curves: pd.DataFrame,
    edges: pd.DataFrame,
    cfg=config,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return the sensitivity table of docs/FINISH_PLAN.md section 3.7.

    `curves` is the perturbation_curves table. `edges` is the attention_edges
    table, with every module and every time collapse. Set `verbose` to True to
    print one line per row.

    The columns are exactly COLUMNS. The checks are these.

        pooled          H1 and H2 on every window. The reference line of the
                        forest plot.
        loso            H1 and H2 on one scene alone, one row per scene.
        heterogeneity   i_squared and the range of the per scene location.
        n_removed       H1 at the steps 1 to 5 of the removal sweep.
        module          H1 and H2 with the edge picked under the decoder
                        attention and under the cross attention.
        time_agg        H1 and H2 with the edge picked under the max collapse
                        and under the last collapse.
        mask_policy     H1 and H2 from the single rows under weight_zero.
        noise_floor     the share of windows where MoRF beats Random.

    Every paired row goes through `h1_morf_vs_lerf.test_paired`, so every row
    carries the same effect definition, the same cluster unit and the same
    test rule as the primary rows. No p value of this table enters a
    correction family. The table is descriptive.

    The heterogeneity row, the noise floor row and a mask_policy row without
    data hold a NaN p value on purpose. Those three rows run no test.
    """
    policy = str(cfg.PRIMARY_MASK_POLICY)
    n_removed = int(cfg.PRIMARY_N_REMOVED)
    rows: list[dict] = []

    if verbose:
        print("validate.run", flush=True)

    # 1. The pooled reference of each hypothesis.
    d_primary = {
        "h1": h1.paired_shift(curves, "morf", "lerf", n_removed, policy),
        "h2": h1.paired_shift(curves, "morf", "nearest", n_removed, policy),
    }
    for hypothesis in HYPOTHESES:
        table = d_primary[hypothesis]
        if len(table) == 0:
            rows.append(_blank_row("pooled", "all", hypothesis))
            continue
        rows.append(
            _row("pooled", "all", hypothesis, h1.test_paired(table, hypothesis, "primary", cfg))
        )

    # 2. One scene at a time, then the spread over the scenes.
    heterogeneity_rows: list[dict] = []
    for hypothesis in HYPOTHESES:
        scene_rows, locations = _loso_rows(d_primary[hypothesis], hypothesis, cfg)
        rows.extend(scene_rows)
        heterogeneity_rows.append(_heterogeneity_row(locations, hypothesis))
    rows.extend(heterogeneity_rows)

    # 3. The removal sweep of H1.
    rows.extend(_n_removed_rows(curves, cfg))

    # 4. Another module, then another time collapse.
    for module in MODULE_VARIANTS:
        rows.extend(
            _variant_rows(
                curves, edges, "module", module, module, str(cfg.PRIMARY_TIME_AGG),
                policy, cfg,
            )
        )
    for time_agg in TIME_AGG_VARIANTS:
        rows.extend(
            _variant_rows(
                curves, edges, "time_agg", time_agg, str(cfg.PRIMARY_MODULE), time_agg,
                policy, cfg,
            )
        )

    # 5. The second mask policy.
    rows.extend(_mask_policy_rows(curves, edges, cfg))

    # 6. The noise floor.
    rows.append(_noise_floor_row(curves, cfg))

    if verbose:
        for row in rows:
            _print_row(row)

    return _as_table(rows)
