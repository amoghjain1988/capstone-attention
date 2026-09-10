"""The cluster sign-flip test and the attention shuffle test.

See CONTRACT.md section 5.11 and docs/FINISH_PLAN.md section 3.2 for the
signatures. Every number comes from config.py.

The sign-flip test is the source of the primary p value of H1 and H2. The
design is paired, so the null says that the sign of every difference is
arbitrary. The test flips the sign of every difference inside one cluster
together, because the rows of one cluster are dependent. The cluster unit is
`config.CLUSTER_UNIT`. `src.stats.clustered.cluster_labels` builds the
labels.

The attention shuffle test supports H2. It permutes the
attention weight inside a scene and reads the shift of the new top edge and
the new bottom edge out of the `single` arm. It needs no forward pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import config
from src.data import schema

# The statistics that sign_flip_test accepts.
STATS = ("signed_rank", "sign", "mean", "median")

# The alternatives that both tests accept. "two_sided" is the same rule as
# "two-sided".
ALTERNATIVES = ("greater", "less", "two-sided", "two_sided")

# A memory limit, not a statistical constant. The permutation runs in blocks
# so that one block holds at most this many floats. The number changes the
# speed and the peak memory only. It does not change a result.
_CELL_BUDGET = 2_000_000


def _sign_matrix(n_perm: int, n_groups: int, seed: int) -> np.ndarray:
    """Return an (n_perm, n_groups) matrix of independent -1 and +1 signs.

    One draw of this matrix serves the whole test. Row b holds the sign of
    every cluster in permutation b.
    """
    rng = np.random.default_rng(seed)
    return 1.0 - 2.0 * rng.integers(0, 2, size=(n_perm, n_groups)).astype(np.float64)


def _p_value(
    statistics: np.ndarray, observed: float, alternative: str, n_draw: int
) -> float:
    """Return the Monte Carlo p value, with the observed data added to the count.

    p = (1 + count(statistic at or beyond the observed one)) / (1 + n_draw).
    The added 1 keeps the p value above 0 and keeps the test valid.
    """
    if alternative == "greater":
        count = int(np.count_nonzero(statistics >= observed))
    elif alternative == "less":
        count = int(np.count_nonzero(statistics <= observed))
    elif alternative in ("two-sided", "two_sided"):
        count = int(np.count_nonzero(np.abs(statistics) >= abs(observed)))
    else:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    return (1.0 + count) / (1.0 + n_draw)


def _flip_contribution(d: np.ndarray, stat: str) -> np.ndarray | None:
    """Return the per-row term of a statistic that the sign flip multiplies.

    Three of the four statistics are a plain sum over the rows, so a sign
    flip of a cluster only flips the sum of that cluster. The function
    returns that per-row term. The median is not a sum, so the function
    returns None and the caller flips the values themselves.

    signed_rank follows the Pratt rule: the rank comes from the absolute
    value over ALL rows, and a zero keeps its rank but adds 0.
    """
    if stat == "signed_rank":
        return scipy_stats.rankdata(np.abs(d)) * np.sign(d)
    if stat == "sign":
        return np.sign(d)
    if stat == "mean":
        return d / d.size
    if stat == "median":
        return None
    raise ValueError(f"stat must be one of {STATS}; got {stat!r}")


def _median_under_flips(
    d: np.ndarray, signs: np.ndarray, group_index: np.ndarray
) -> np.ndarray:
    """Return the median of d under every sign flip, one value per permutation."""
    n_perm = signs.shape[0]
    block = max(1, int(_CELL_BUDGET // max(1, d.size)))
    out = np.empty(n_perm, dtype=np.float64)
    done = 0
    while done < n_perm:
        size = min(block, n_perm - done)
        flipped = d[None, :] * signs[done : done + size][:, group_index]
        out[done : done + size] = np.median(flipped, axis=1)
        done += size
    return out


def sign_flip_test(
    d: np.ndarray,
    clusters: np.ndarray,
    alternative: str = "greater",
    n_perm: int = config.N_PERM,
    seed: int = config.SEED,
    stat: str = "signed_rank",
) -> dict:
    """Return the cluster sign-flip test of the paired differences d.

    The null says that the sign of a difference is arbitrary. One draw
    assigns one sign to a whole cluster, so every row of that cluster flips
    together. That rule respects the dependence inside a cluster, which a
    row-wise flip would ignore.

    `stat` is one of signed_rank, sign, mean or median.
      signed_rank ranks the absolute differences over ALL rows under the
        Pratt rule and sums rank * sign(d). A zero keeps its rank and adds 0.
      sign counts the positive differences minus the negative ones.
      mean and median are the plain sample statistics.

    numpy draws the whole (n_perm, n_clusters) sign matrix once. For the
    first three statistics the permutation is one matrix product against the
    per-cluster sums, so no Python loop runs over the permutations.

    Returns {stat, stat_obs, p, n_perm, n_clusters, alternative}.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    clusters = np.asarray(clusters).ravel()
    if d.size != clusters.size:
        raise ValueError(
            f"d and clusters must match in length; got {d.size} and {clusters.size}"
        )
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}; got {stat!r}")
    if alternative not in ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {ALTERNATIVES}; got {alternative!r}"
        )
    if d.size == 0:
        raise ValueError("d holds no rows")
    if n_perm < 1:
        raise ValueError(f"n_perm must be at least 1; got {n_perm}")

    _, group_index = np.unique(clusters, return_inverse=True)
    group_index = np.asarray(group_index).ravel()
    n_groups = int(group_index.max()) + 1
    signs = _sign_matrix(n_perm, n_groups, seed)

    contribution = _flip_contribution(d, stat)
    if contribution is None:
        stat_obs = float(np.median(d))
        draws = _median_under_flips(d, signs, group_index)
    else:
        per_group = np.bincount(group_index, weights=contribution, minlength=n_groups)
        stat_obs = float(per_group.sum())
        draws = signs @ per_group

    return {
        "stat": stat,
        "stat_obs": stat_obs,
        "p": _p_value(draws, stat_obs, alternative, n_perm),
        "n_perm": int(n_perm),
        "n_clusters": int(n_groups),
        "alternative": alternative,
    }


def sign_flip_p(
    d: np.ndarray,
    clusters: np.ndarray,
    alternative: str = "greater",
    n_perm: int = config.N_PERM,
    seed: int = config.SEED,
    stat: str = "signed_rank",
) -> float:
    """Return the p value of `sign_flip_test`. This is the contract signature."""
    return sign_flip_test(
        d, clusters, alternative=alternative, n_perm=n_perm, seed=seed, stat=stat
    )["p"]


# ---------------------------------------------------------------------------
# The attention shuffle test
# ---------------------------------------------------------------------------


def _padded_edge_tables(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the padded per-window layout of the edge rows.

    `frame` sorts by window. The function returns the row index and the
    column index of every edge inside a (n_windows, max_edges) grid, the
    shift of every cell, a mask of the real cells, and the attention weight
    of every edge row in the order of `frame`.
    """
    window_ids = frame["window_id"].to_numpy()
    _, window_index = np.unique(window_ids, return_inverse=True)
    window_index = np.asarray(window_index).ravel()
    n_windows = int(window_index.max()) + 1

    counts = np.bincount(window_index, minlength=n_windows)
    max_edges = int(counts.max())
    position = frame.groupby("window_id").cumcount().to_numpy(dtype=np.int64)

    shift_grid = np.zeros((n_windows, max_edges), dtype=np.float64)
    valid = np.zeros((n_windows, max_edges), dtype=bool)
    shift_grid[window_index, position] = frame["shift"].to_numpy(dtype=np.float64)
    valid[window_index, position] = True

    attn = frame["attn"].to_numpy(dtype=np.float64)
    return window_index, position, shift_grid, valid, attn


def _gap_per_window(
    attn_grid: np.ndarray, shift_grid: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Return D per window: the shift of the top edge minus the shift of the bottom edge.

    `attn_grid` holds one block per permutation, or one single grid. The pad
    cells go to minus infinity for the maximum and to plus infinity for the
    minimum, so a pad cell never wins.
    """
    for_max = np.where(valid, attn_grid, -np.inf)
    for_min = np.where(valid, attn_grid, np.inf)
    top = np.argmax(for_max, axis=-1)
    bottom = np.argmin(for_min, axis=-1)
    grid = shift_grid if attn_grid.ndim == 2 else shift_grid[None, :, :]
    high = np.take_along_axis(grid, top[..., None], axis=-1)[..., 0]
    low = np.take_along_axis(grid, bottom[..., None], axis=-1)[..., 0]
    return high - low


def attention_shuffle_p(
    edges: pd.DataFrame,
    single: pd.DataFrame,
    n_shuffle: int = config.N_PERM,
    seed: int = config.SEED,
    alternative: str = "greater",
) -> dict:
    """Return the attention shuffle test of H2.

    `edges` holds the primary edges (window_id, src, attn), one row per edge
    into the ego. `single` holds (window_id, edge_src, shift) from the single
    arm under the primary mask policy.

    The observed statistic is the median over windows of
    D = shift(the edge with the largest attn) - shift(the edge with the
    smallest attn).

    The null permutes the attn column inside each scene, across every edge
    row of that scene. Each window keeps its edge count, so the graph shape
    survives. The code then reads the new top edge and the new bottom edge of
    every window and takes D again. No forward pass runs, because every D is
    a lookup into `single`.

    Returns {stat_obs, p, n_shuffle, n_windows}.
    """
    for column in ("window_id", "src", "attn"):
        if column not in edges.columns:
            raise KeyError(f"edges must carry a {column!r} column")
    for column in ("window_id", "edge_src", "shift"):
        if column not in single.columns:
            raise KeyError(f"single must carry a {column!r} column")
    if alternative not in ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {ALTERNATIVES}; got {alternative!r}"
        )
    if n_shuffle < 1:
        raise ValueError(f"n_shuffle must be at least 1; got {n_shuffle}")

    lookup = single[["window_id", "edge_src", "shift"]].rename(
        columns={"edge_src": "src"}
    )
    if lookup.duplicated(subset=["window_id", "src"]).any():
        raise ValueError(
            "single holds more than one row for an edge. Slice it to one mask "
            "policy and one draw before the call."
        )
    if edges.duplicated(subset=["window_id", "src"]).any():
        raise ValueError("edges holds more than one row for an edge")

    joined = edges[["window_id", "src", "attn"]].merge(
        lookup, on=["window_id", "src"], how="inner", validate="one_to_one"
    )
    counts = joined.groupby("window_id")["src"].transform("size")
    joined = joined.loc[counts >= 2].copy()
    if len(joined) == 0:
        raise ValueError("no window holds 2 edges with a single-arm shift")

    joined["scene"] = [
        schema.split_window_id(str(wid))[0] for wid in joined["window_id"]
    ]
    joined = joined.sort_values(["window_id", "src"], kind="stable").reset_index(
        drop=True
    )

    window_index, position, shift_grid, valid, attn = _padded_edge_tables(joined)
    n_windows = int(shift_grid.shape[0])

    observed_grid = np.zeros_like(shift_grid)
    observed_grid[window_index, position] = attn
    stat_obs = float(np.median(_gap_per_window(observed_grid, shift_grid, valid)))

    scene_codes = joined["scene"].to_numpy()
    scene_rows = [
        np.flatnonzero(scene_codes == scene) for scene in pd.unique(joined["scene"])
    ]

    rng = np.random.default_rng(seed)
    cells = max(1, shift_grid.size)
    block = max(1, int(_CELL_BUDGET // cells))
    draws = np.empty(n_shuffle, dtype=np.float64)
    done = 0
    while done < n_shuffle:
        size = min(block, n_shuffle - done)
        shuffled = np.empty((size, attn.size), dtype=np.float64)
        for rows in scene_rows:
            keys = rng.random((size, rows.size))
            shuffled[:, rows] = attn[rows][np.argsort(keys, axis=1)]
        grid = np.zeros((size, *shift_grid.shape), dtype=np.float64)
        grid[:, window_index, position] = shuffled
        draws[done : done + size] = np.median(
            _gap_per_window(grid, shift_grid, valid), axis=-1
        )
        done += size

    return {
        "stat_obs": stat_obs,
        "p": _p_value(draws, stat_obs, alternative, n_shuffle),
        "n_shuffle": int(n_shuffle),
        "n_windows": n_windows,
    }
