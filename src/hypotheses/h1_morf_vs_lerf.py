"""H1: the most-attended edge moves the forecast more than the least-attended one.

See CONTRACT.md section 5.10 and docs/FINISH_PLAN.md sections 3.4 and 3.5 for
the signatures. Every number comes from config.py.

THE CLAIM. D_i = shift(MoRF) - shift(LeRF) at config.PRIMARY_N_REMOVED, one
value per window. H0-1 says that the location of D is at or below 0. H1-1 says
that the location of D is above 0. The primary test is the arm comparison.
MoRF against Random and MoRF against Weight-matched SUPPORT the claim. Neither
one enters the primary family.

WHY THE PAIR. Both arms run on the same window, the same ego, the same draws
and the same unmasked baseline. The difference removes all of that shared
structure, so only the choice of the removed edge is left.

WHY THREE TOOLS, NOT ONE. `src/hypotheses/choose_test.py` reads the SHAPE of D
and names the test. `src/stats/permutation.py` gives the p value from a sign
flip inside the cluster, because the rows are not independent.
`src/stats/clustered.py` gives the interval from the pairs cluster bootstrap,
for the same reason. This module joins the three and adds no statistic of its
own.

WHY THE MASS RULE FOR WEIGHT-MATCHED. MoRF removes the heaviest edge, so MoRF
also removes more attention mass than any count-matched control. The
weight_matched arm removes the LOWEST edges until the mass matches, so the
comparison holds the mass fixed and varies only the rank order. The partner of
MoRF at step n is therefore chosen by mass, never by n_removed. See
CONTRACT.md section 5.7 and `src/ablation/arms.py::weight_matched_set`.

ZEROS. A zero difference is real data and it stays in. The Pratt rule of
`src/stats/effects.py::rank_biserial` and of the signed_rank statistic gives a
zero its rank and adds 0 to the sum, so a mass of ties pulls the test toward
the null and never away from it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config
from src.hypotheses import choose_test
from src.stats import clustered, effects, permutation

# The columns of the paired difference table.
PAIRED_COLUMNS = ("window_id", "ego_id", "shift_a", "shift_b", "d")

# The tolerance of the mass match. Two float sums of the same attention values
# differ in the last bits, so a plain ">=" test would drop the right partner.
# The number is a float guard, not a statistical constant.
MASS_TOL = 1e-9

# The sentence that the degenerate branch reports.
NO_TEST_WHY = "Fewer than 3 windows hold a non-zero difference, so no test runs."

# The name that the degenerate branch gives to an effect of 0.
NO_TEST_EFFECT_NAME = "none"


# ---------------------------------------------------------------------------
# The two effect and location statistics
# ---------------------------------------------------------------------------


def share_positive(d: np.ndarray) -> float:
    """Return the share of positive differences among the non-zero ones.

    This is the effect size of the sign test. A value of 0.5 says that the two
    arms tie. A value of 1.0 says that arm A wins on every non-zero window.
    Return NaN when every difference is 0, because the share is then undefined.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    nonzero = d[d != 0.0]
    if nonzero.size == 0:
        return float("nan")
    return float(np.count_nonzero(nonzero > 0.0) / nonzero.size)


def median(d: np.ndarray) -> float:
    """Return the sample median. This is the location that the sign test claims."""
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        return float("nan")
    return float(np.median(d))


def alternative_of(cfg=config) -> str:
    """Return the alternative that `src/stats/permutation.py` accepts.

    A one-sided design tests "greater", because every hypothesis of this study
    names a direction. Any other design tests both tails.
    """
    return "greater" if str(cfg.SIDED) == "one" else "two-sided"


# ---------------------------------------------------------------------------
# The paired difference table
# ---------------------------------------------------------------------------


def _arm_rows(curves: pd.DataFrame, arm: str, mask_policy: str) -> pd.DataFrame:
    """Return the rows of one arm under one mask policy, with plain dtypes.

    The `arm` and `mask_policy` columns are categories on disk, so both sides
    of the comparison become plain strings first.
    """
    for column in ("window_id", "ego_id", "arm", "n_removed", "removed_mass",
                   "mask_policy", "shift"):
        if column not in curves.columns:
            raise KeyError(
                f"curves must carry a {column!r} column. See CONTRACT.md "
                "section 3, perturbation_curves."
            )
    block = curves.loc[
        (curves["arm"].astype(str) == str(arm))
        & (curves["mask_policy"].astype(str) == str(mask_policy))
    ]
    keep = ["window_id", "ego_id", "n_removed", "removed_mass", "shift"]
    out = block.loc[:, keep].copy()
    out["window_id"] = out["window_id"].astype(str)
    out["ego_id"] = out["ego_id"].astype("int64")
    out["n_removed"] = out["n_removed"].astype("int64")
    out["removed_mass"] = out["removed_mass"].astype("float64")
    out["shift"] = out["shift"].astype("float64")
    return out


def _one_row_per_window(frame: pd.DataFrame, arm: str, n_removed: int) -> pd.DataFrame:
    """Raise when a window holds more than one row of the same arm and step."""
    repeated = frame["window_id"][frame["window_id"].duplicated()].unique()
    if len(repeated):
        raise ValueError(
            f"the arm {arm!r} holds more than one row at n_removed {n_removed} "
            f"for these windows: {sorted(repeated)[:5]}. Slice the curves to one "
            "draw first."
        )
    return frame


def _mass_partner(
    targets: pd.DataFrame, matched: pd.DataFrame
) -> pd.DataFrame:
    """Return the weight-matched partner of every target row.

    `targets` holds window_id and the cumulative mass that MoRF removed at the
    step under test. `matched` holds every weight_matched row of the same mask
    policy. The partner is the row with the SMALLEST removed_mass that reaches
    the target mass.

    A window whose matched sets all fall below the target keeps the heaviest
    set, and the function warns. `src/ablation/arms.py::weight_matched_set`
    returns every edge when the target exceeds the total mass, so this branch
    reports a real hole in the curves and not a rounded float.
    """
    joined = targets.merge(
        matched.loc[:, ["window_id", "removed_mass", "shift"]],
        on="window_id",
        how="inner",
        suffixes=("_target", "_matched"),
    )
    reaches = joined["removed_mass_matched"] >= joined["removed_mass_target"] - MASS_TOL

    short = set(joined.loc[~joined["window_id"].isin(
        joined.loc[reaches, "window_id"]
    ), "window_id"])
    if short:
        warnings.warn(
            f"{len(short)} windows hold no weight-matched set that reaches the "
            f"MoRF mass, so the heaviest set stands in. First: "
            f"{sorted(short)[:3]}",
            stacklevel=3,
        )

    good = joined.loc[reaches].sort_values(
        ["window_id", "removed_mass_matched"], kind="stable"
    )
    poor = joined.loc[joined["window_id"].isin(short)].sort_values(
        ["window_id", "removed_mass_matched"], ascending=[True, False], kind="stable"
    )
    picked = pd.concat([good, poor], ignore_index=True)
    picked = picked.drop_duplicates(subset=["window_id"], keep="first")
    return picked.loc[:, ["window_id", "shift"]].rename(columns={"shift": "shift_b"})


def paired_shift(
    curves: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    n_removed: int = config.PRIMARY_N_REMOVED,
    mask_policy: str = config.PRIMARY_MASK_POLICY,
) -> pd.DataFrame:
    """Return one row per window: window_id, ego_id, shift_a, shift_b, d.

    `curves` is the perturbation_curves table of CONTRACT.md section 3. The
    function reads the rows of `arm_a` at `n_removed` under `mask_policy` and
    pairs each one with the row of `arm_b` on the same window. The difference
    is `d = shift_a - shift_b`.

    The partner of a normal arm is the row of `arm_b` at the SAME n_removed.
    The partner of `weight_matched` is the row with the smallest removed_mass
    that reaches the cumulative mass of `arm_a` at that step. The removed_mass
    column of the `arm_a` row already holds that cumulative mass, because
    `src/ablation/curves.py::curve` sums the attention along the removal order.

    Raise ValueError when a window carries `arm_a` and carries no partner under
    `arm_b`. A silent drop there would compare two different sets of windows.
    """
    left = _arm_rows(curves, arm_a, mask_policy)
    left = left.loc[left["n_removed"] == int(n_removed)]
    left = _one_row_per_window(left, arm_a, int(n_removed))

    if str(arm_b) == "weight_matched":
        matched = _arm_rows(curves, arm_b, mask_policy)
        targets = left.loc[:, ["window_id", "removed_mass"]]
        right = _mass_partner(targets, matched)
    else:
        other = _arm_rows(curves, arm_b, mask_policy)
        other = other.loc[other["n_removed"] == int(n_removed)]
        other = _one_row_per_window(other, arm_b, int(n_removed))
        right = other.loc[:, ["window_id", "shift"]].rename(
            columns={"shift": "shift_b"}
        )

        orphan = set(right["window_id"]) - set(left["window_id"])
        if orphan:
            raise ValueError(
                f"{len(orphan)} windows carry the arm {arm_b!r} at n_removed "
                f"{n_removed} and carry no {arm_a!r} row: {sorted(orphan)[:5]}"
            )

    missing = set(left["window_id"]) - set(right["window_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} windows carry the arm {arm_a!r} at n_removed "
            f"{n_removed} and carry no {arm_b!r} partner: {sorted(missing)[:5]}"
        )

    out = left.loc[:, ["window_id", "ego_id", "shift"]].rename(
        columns={"shift": "shift_a"}
    )
    out = out.merge(right, on="window_id", how="inner", validate="one_to_one")
    out["d"] = out["shift_a"] - out["shift_b"]
    out = out.sort_values("window_id", kind="stable").reset_index(drop=True)
    return out.loc[:, list(PAIRED_COLUMNS)]


# ---------------------------------------------------------------------------
# One paired test
# ---------------------------------------------------------------------------


def _degenerate(
    hypothesis: str,
    role: str,
    sided: str,
    n_windows: int,
    n_zero: int,
    n_cluster: int,
    location: float,
) -> dict:
    """Return the result dict of a sample that holds too few non-zero rows.

    `src/hypotheses/choose_test.py::choose` raises below 3 non-zero
    differences, because such a sample has no shape to read. The dict below
    carries the same keys as a real result, so `src/stats/report.py::build`
    still accepts it. p_raw is 1.0 and never NaN, because Holm sorts the p
    values and a NaN would break that sort.

    `location` is the plain sample median of D, in metres, and it is NaN when
    the table holds no row. No test runs here, so the branch reports the
    median that the rows show and never a number that it did not measure. A
    table of pure ties therefore still reports 0.0, and an empty table
    reports NaN.
    """
    return {
        "hypothesis": hypothesis,
        "role": role,
        "test_used": "none",
        "why": NO_TEST_WHY,
        "sided": sided,
        "effect": 0.0,
        "effect_name": NO_TEST_EFFECT_NAME,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_raw": 1.0,
        "n_clusters": int(n_cluster),
        "location_m": float(location),
        "location_ci_low": float("nan"),
        "location_ci_high": float("nan"),
        "n_windows": int(n_windows),
        "n_zero": int(n_zero),
        "stat_obs": 0.0,
        "choice": None,
    }


def test_paired(
    d_table: pd.DataFrame, hypothesis: str, role: str, cfg=config
) -> dict:
    """Return the result dict of one paired test on the differences of `d_table`.

    `d_table` is the output of `paired_shift`. The steps are these.

    1. `choose()` reads the shape of D and names the test.
    2. `component_clusters` labels every row with its connected component.
    3. `sign_flip_test` gives the p value from a sign flip inside the cluster.
    4. `cluster_bootstrap_ci` gives the interval of the effect and of the
       location.

    A Wilcoxon choice takes the signed_rank statistic, the rank biserial
    correlation as the effect, and the Hodges-Lehmann pseudomedian in metres as
    the location. A sign choice takes the sign statistic, the share of positive
    differences as the effect, and the median in metres as the location.

    The dict holds every key of docs/FINISH_PLAN.md section 3.4, plus
    location_m, location_ci_low, location_ci_high, n_windows, n_zero, stat_obs
    and choice. `src/stats/report.py::build` reads the first group and ignores
    the rest.
    """
    if "d" not in d_table.columns:
        raise KeyError("d_table must carry a 'd' column. Call paired_shift first.")

    d = np.asarray(d_table["d"], dtype=np.float64).ravel()
    clusters = clustered.component_clusters(d_table)
    n_cluster = clustered.n_clusters(clusters)
    n_windows = int(d.size)
    n_zero = int(np.count_nonzero(d == 0.0))
    sided = str(cfg.SIDED)

    try:
        choice = choose_test.choose(d)
    except ValueError:
        return _degenerate(
            hypothesis, role, sided, n_windows, n_zero, n_cluster, median(d)
        )

    if choice["test"] == choose_test.WILCOXON:
        stat_name = "signed_rank"
        effect_fn = effects.rank_biserial
        effect_name = "rank_biserial"
        location_fn = clustered.hodges_lehmann
    else:
        stat_name = "sign"
        effect_fn = share_positive
        effect_name = "share_positive"
        location_fn = median

    permuted = permutation.sign_flip_test(
        d,
        clusters,
        alternative=alternative_of(cfg),
        n_perm=int(cfg.N_PERM),
        seed=int(cfg.SEED),
        stat=stat_name,
    )
    ci_low, ci_high = clustered.cluster_bootstrap_ci(
        d, clusters, stat=effect_fn, n_boot=int(cfg.N_BOOT), seed=int(cfg.SEED)
    )
    location_low, location_high = clustered.cluster_bootstrap_ci(
        d, clusters, stat=location_fn, n_boot=int(cfg.N_BOOT), seed=int(cfg.SEED)
    )

    return {
        "hypothesis": hypothesis,
        "role": role,
        "test_used": (
            f"{choice['test']}, sign-flip permutation by component, "
            "pairs cluster bootstrap"
        ),
        "why": choice["why"],
        "sided": sided,
        "effect": float(effect_fn(d)),
        "effect_name": effect_name,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_raw": float(permuted["p"]),
        "n_clusters": int(n_cluster),
        "location_m": float(location_fn(d)),
        "location_ci_low": float(location_low),
        "location_ci_high": float(location_high),
        "n_windows": n_windows,
        "n_zero": n_zero,
        "stat_obs": float(permuted["stat_obs"]),
        "choice": choice,
    }


# ---------------------------------------------------------------------------
# The hypothesis
# ---------------------------------------------------------------------------


def run(curves: pd.DataFrame, cfg=config) -> dict:
    """Return the primary test of H1 and its two supporting tests.

    The primary test compares MoRF against LeRF at cfg.PRIMARY_N_REMOVED under
    cfg.PRIMARY_MASK_POLICY. The first supporting test compares MoRF against
    Random, so a reader sees whether attention beats a coin flip. The second
    compares MoRF against Weight-matched, so a reader sees whether the rank
    order matters once the removed mass is held fixed.

    Returns {"primary": dict, "supporting": [dict, dict], "d": DataFrame,
    "choice": the choose() output of the primary test}.
    """
    n_removed = int(cfg.PRIMARY_N_REMOVED)
    policy = str(cfg.PRIMARY_MASK_POLICY)

    d_primary = paired_shift(curves, "morf", "lerf", n_removed, policy)
    primary = test_paired(d_primary, "h1", "primary", cfg)

    d_random = paired_shift(curves, "morf", "random", n_removed, policy)
    against_random = test_paired(d_random, "h1_morf_vs_random", "supporting", cfg)

    d_matched = paired_shift(curves, "morf", "weight_matched", n_removed, policy)
    against_matched = test_paired(
        d_matched, "h1_morf_vs_weight_matched", "supporting", cfg
    )

    return {
        "primary": primary,
        "supporting": [against_random, against_matched],
        "d": d_primary,
        "choice": primary["choice"],
    }
