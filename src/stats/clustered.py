"""Cluster labels, the pairs cluster bootstrap and the wild cluster bootstrap.

See CONTRACT.md section 5.11 and docs/FINISH_PLAN.md section 3.1 for the
signatures. Every number comes from config.py.

The cluster unit of the study is `config.CLUSTER_UNIT`, the connected
component of the bipartite graph of ego pedestrians against time blocks
inside one scene. config.py holds the reason for that choice and the false
positive rate of each candidate unit.

Two different bootstraps live here. `cluster_bootstrap_ci` is the pairs
cluster bootstrap. It resamples whole clusters and it gives an interval for
H1 and H2. `wild_cluster_bootstrap` puts Rademacher weights on the
restricted residuals of a regression. It gives the p value of H3. The two
tools are not interchangeable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

import config
from src.data import schema

# The units that cluster_labels accepts.
UNITS = ("pedestrian", "time_block", "scene", "component")

# A memory limit, not a statistical constant. The bootstrap builds its row
# indices in blocks so that one block holds at most this many integers. The
# number changes the speed and the peak memory only. It does not change a
# result: the draws stay the same for the same seed and the same table.
_INDEX_BUDGET = 2_000_000


# ---------------------------------------------------------------------------
# Cluster labels
# ---------------------------------------------------------------------------


def _scene_and_block(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the scene and the time block of every row of `table`.

    The scene and the first observed frame come from
    `schema.split_window_id`. The time block is `t0 // config.WINDOW_LEN`.
    """
    if "window_id" not in table.columns:
        raise KeyError("table must carry a 'window_id' column")

    window_ids = table["window_id"].astype(str).to_numpy()
    pairs = [schema.split_window_id(str(wid)) for wid in window_ids]
    scenes = np.array([pair[0] for pair in pairs], dtype=object)
    t0 = np.array([pair[1] for pair in pairs], dtype=np.int64)
    blocks = t0 // int(config.WINDOW_LEN)
    return scenes, blocks


def _component_index(
    scenes: np.ndarray, blocks: np.ndarray, egos: np.ndarray
) -> np.ndarray:
    """Return the connected component of every row, as a global integer.

    The graph is bipartite. One side holds the pair (scene, ego_id). The
    other side holds the pair (scene, time block). Every row is one edge.
    The scene enters both node keys, so no component crosses a scene.
    """
    n_rows = len(scenes)
    ego_keys = np.array(
        [f"{scene}|ego|{ego}" for scene, ego in zip(scenes, egos)], dtype=object
    )
    block_keys = np.array(
        [f"{scene}|blk|{block}" for scene, block in zip(scenes, blocks)], dtype=object
    )

    nodes, inverse = np.unique(
        np.concatenate([ego_keys, block_keys]), return_inverse=True
    )
    ego_nodes = inverse[:n_rows]
    block_nodes = inverse[n_rows:]

    graph = sparse.coo_matrix(
        (np.ones(n_rows, dtype=np.int8), (ego_nodes, block_nodes)),
        shape=(nodes.size, nodes.size),
    )
    _, component_of_node = csgraph.connected_components(graph, directed=False)
    return component_of_node[ego_nodes]


def cluster_labels(table: pd.DataFrame, unit: str = config.CLUSTER_UNIT) -> np.ndarray:
    """Return one cluster label per row of `table`.

    `table` carries `window_id` and, for every unit except scene, `ego_id`.
    `unit` is one of pedestrian, time_block, scene or component. The scene
    and the first frame come from `schema.split_window_id(window_id)`. The
    time block is `t0 // config.WINDOW_LEN`. The component is the connected
    component of the bipartite graph of (scene, ego_id) against
    (scene, time block), with one edge per row.

    A label is a string. The scene name starts every label, so a label never
    joins two scenes. The component labels count from 0 inside each scene, in
    the order the components first appear, for example "zara1_c12".
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}; got {unit!r}")
    if len(table) == 0:
        return np.array([], dtype=str)

    scenes, blocks = _scene_and_block(table)

    if unit == "scene":
        return np.asarray([str(scene) for scene in scenes], dtype=str)

    if unit == "time_block":
        return np.asarray(
            [f"{scene}_t{block}" for scene, block in zip(scenes, blocks)], dtype=str
        )

    if "ego_id" not in table.columns:
        raise KeyError(f"unit {unit!r} needs an 'ego_id' column")
    egos = table["ego_id"].to_numpy()

    if unit == "pedestrian":
        return np.asarray(
            [f"{scene}_p{ego}" for scene, ego in zip(scenes, egos)], dtype=str
        )

    components = _component_index(scenes, blocks, egos)
    per_scene_index: dict[tuple, int] = {}
    next_index: dict[str, int] = {}
    labels: list[str] = []
    for scene, component in zip(scenes, components):
        key = (scene, int(component))
        if key not in per_scene_index:
            per_scene_index[key] = next_index.get(scene, 0)
            next_index[scene] = per_scene_index[key] + 1
        labels.append(f"{scene}_c{per_scene_index[key]}")
    return np.asarray(labels, dtype=str)


def component_clusters(table: pd.DataFrame) -> np.ndarray:
    """Return `cluster_labels(table, "component")`.

    `src/hypotheses/h3_context.py` imports this name. Do not rename it.
    """
    return cluster_labels(table, "component")


def n_clusters(labels: np.ndarray) -> int:
    """Return the count of distinct labels."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0
    return int(np.unique(labels).size)


# ---------------------------------------------------------------------------
# The pairs cluster bootstrap
# ---------------------------------------------------------------------------


def _group_layout(clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the row order, the group starts, the group sizes and the count.

    The rows sort by group. `starts[g]` is the first position of group g in
    that order and `counts[g]` is its size. Every group holds at least one
    row, so `np.add.reduceat` is safe on this layout.
    """
    _, group_index = np.unique(clusters, return_inverse=True)
    group_index = np.asarray(group_index).ravel()
    n_groups = int(group_index.max()) + 1 if group_index.size else 0
    order = np.argsort(group_index, kind="stable")
    counts = np.bincount(group_index, minlength=n_groups)
    starts = np.zeros(n_groups, dtype=np.int64)
    if n_groups > 0:
        starts[1:] = np.cumsum(counts)[:-1]
    return order, starts, counts, n_groups


def _ragged_arange(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return the concatenation of arange(s, s + n) over every (s, n) pair.

    The function builds every block at once with one cumulative sum, so no
    Python loop runs over the blocks.
    """
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    keep = lengths > 0
    starts = starts[keep]
    lengths = lengths[keep]
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)

    out = np.ones(total, dtype=np.int64)
    ends = np.cumsum(lengths)
    begins = ends - lengths
    out[begins] = starts
    if begins.size > 1:
        out[begins[1:]] -= starts[:-1] + lengths[:-1] - 1
    return np.cumsum(out)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    stat=np.median,
    n_boot: int = config.N_BOOT,
    level: float = 0.95,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Return the percentile interval of `stat` from the pairs cluster bootstrap.

    The bootstrap draws CLUSTERS with replacement, not rows. It pools every
    row of the drawn clusters, then it applies `stat` to that pool. A cluster
    that comes up twice contributes its rows twice. The interval is the
    percentile interval of the bootstrap statistics at `level`.

    numpy builds the row indices of a whole block of draws in one step. Only
    the call to `stat` runs once per draw, because `stat` is an arbitrary
    callable.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    clusters = np.asarray(clusters).ravel()
    if values.size != clusters.size:
        raise ValueError(
            f"values and clusters must match in length; got {values.size} and "
            f"{clusters.size}"
        )
    if values.size == 0:
        return (float("nan"), float("nan"))
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1; got {n_boot}")

    order, starts, counts, n_groups = _group_layout(clusters)
    rng = np.random.default_rng(seed)

    block = max(1, int(_INDEX_BUDGET // max(1, values.size)))
    statistics = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(block, n_boot - done)
        picks = rng.integers(0, n_groups, size=(size, n_groups))
        sizes = counts[picks]
        rows = order[_ragged_arange(starts[picks].ravel(), sizes.ravel())]
        cuts = np.cumsum(sizes.sum(axis=1))[:-1]
        for offset, part in enumerate(np.split(rows, cuts)):
            statistics[done + offset] = float(stat(values[part]))
        done += size

    good = statistics[np.isfinite(statistics)]
    if good.size == 0:
        return (float("nan"), float("nan"))
    low = 100.0 * (1.0 - level) / 2.0
    high = 100.0 * (1.0 + level) / 2.0
    return (float(np.percentile(good, low)), float(np.percentile(good, high)))


def hodges_lehmann(d: np.ndarray) -> float:
    """Return the pseudomedian of d, the median of the Walsh averages.

    A Walsh average is (d_i + d_j) / 2 over every pair with i <= j. The
    Wilcoxon signed-rank test estimates this location, not the plain median.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        return float("nan")
    walsh = (d[:, None] + d[None, :]) / 2.0
    rows, columns = np.triu_indices(d.size)
    return float(np.median(walsh[rows, columns]))


# ---------------------------------------------------------------------------
# The wild cluster bootstrap
# ---------------------------------------------------------------------------


def _default_restriction(n_columns: int) -> np.ndarray:
    """Return the matrix that selects every column except column 0."""
    return np.eye(n_columns, dtype=np.float64)[1:]


def _cluster_covariance(
    xtx_inv: np.ndarray,
    scores: np.ndarray,
    correction: float,
) -> np.ndarray:
    """Return the cluster-robust covariance from the per-cluster scores.

    `scores` holds one row per cluster and one column per coefficient. Each
    row is the sum of X_i * residual_i over the rows of that cluster.
    """
    meat = scores.T @ scores
    return correction * (xtx_inv @ meat @ xtx_inv)


def _wald(beta: np.ndarray, cov: np.ndarray, restriction: np.ndarray) -> float:
    """Return the Wald statistic of H0: restriction @ beta = 0."""
    r_beta = restriction @ beta
    middle = np.linalg.pinv(restriction @ cov @ restriction.T)
    return float(r_beta @ middle @ r_beta)


def _restricted_fit(
    y: np.ndarray,
    design: np.ndarray,
    xtx_inv: np.ndarray,
    restriction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the coefficients and the residuals of the fit under H0.

    The restricted least squares solution projects the unrestricted fit onto
    the subspace where `restriction @ beta` equals 0.
    """
    beta = xtx_inv @ (design.T @ y)
    r_beta = restriction @ beta
    middle = np.linalg.pinv(restriction @ xtx_inv @ restriction.T)
    beta_null = beta - xtx_inv @ restriction.T @ (middle @ r_beta)
    return beta_null, y - design @ beta_null


def _rademacher(n_boot: int, n_groups: int, seed: int) -> np.ndarray:
    """Return an (n_boot, n_groups) matrix of independent -1 and +1 weights."""
    rng = np.random.default_rng(seed)
    return 1.0 - 2.0 * rng.integers(0, 2, size=(n_boot, n_groups)).astype(np.float64)


def _null_walds(
    design: np.ndarray,
    xtx_inv: np.ndarray,
    fitted_null: np.ndarray,
    resid_null: np.ndarray,
    restriction: np.ndarray,
    signs: np.ndarray,
    group_index: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    correction: float,
) -> np.ndarray:
    """Return one Wald statistic per bootstrap draw, under H0.

    The weights multiply the restricted residuals cluster by cluster. numpy
    holds a block of draws at a time, so no Python loop runs over the draws
    times the coefficients.
    """
    n_rows, n_columns = design.shape
    n_boot = signs.shape[0]
    n_groups = starts.size
    projector = xtx_inv @ design.T

    block = max(1, int(_INDEX_BUDGET // max(1, n_rows)))
    walds = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(block, n_boot - done)
        weights = signs[done : done + size].T[group_index]  # (n_rows, size)
        y_star = fitted_null[:, None] + resid_null[:, None] * weights
        beta_star = projector @ y_star  # (n_columns, size)
        resid_star = y_star - design @ beta_star

        scores = np.empty((n_groups, n_columns, size), dtype=np.float64)
        for column in range(n_columns):
            contribution = design[:, column][:, None] * resid_star
            scores[:, column, :] = np.add.reduceat(contribution[order], starts, axis=0)

        meat = np.einsum("gpm,gqm->mpq", scores, scores)
        cov = correction * np.einsum("ab,mbc,cd->mad", xtx_inv, meat, xtx_inv)
        r_beta = (restriction @ beta_star).T  # (size, q)
        rvr = np.einsum("qa,mab,rb->mqr", restriction, cov, restriction)
        walds[done : done + size] = np.einsum(
            "mq,mqr,mr->m", r_beta, np.linalg.pinv(rvr), r_beta
        )
        done += size
    return walds


def wild_cluster_bootstrap(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = config.N_BOOT,
    seed: int = config.SEED,
    restriction: np.ndarray | None = None,
) -> dict:
    """Return the wild cluster bootstrap of an ordinary least squares fit.

    Column 0 of `X` is the intercept. `restriction` is a (q, p) matrix R and
    the null is R beta = 0. The default R selects every column except column
    0, so the default null says that every slope is 0.

    The steps are these. Fit the model without the restriction and take the
    cluster-robust covariance, with the usual small sample correction
    G / (G - 1) * (n - 1) / (n - p). Fit the model again under H0 and keep
    those restricted residuals. Draw a Rademacher weight per cluster and per
    replicate, multiply the restricted residuals by that weight, add the
    restricted fit, and refit. The p value is the share of bootstrap Wald
    statistics at or above the observed one.

    The returned `p_values` holds one p value per coefficient. Each one comes
    from its own single-row restriction, so each one imposes its own null on
    the fit. Every test reuses the one draw of weights.

    Returns {beta, se_cluster, wald_obs, joint_p, p_values, n_clusters,
    n_boot}.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    design = np.asarray(X, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    clusters = np.asarray(clusters).ravel()

    n_rows, n_columns = design.shape
    if y.size != n_rows:
        raise ValueError(
            f"y and X must match in length; got {y.size} and {n_rows}"
        )
    if clusters.size != n_rows:
        raise ValueError(
            f"clusters and X must match in length; got {clusters.size} and {n_rows}"
        )
    if n_rows <= n_columns:
        raise ValueError(
            f"the fit needs more rows than columns; got {n_rows} rows and "
            f"{n_columns} columns"
        )
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1; got {n_boot}")

    if restriction is None:
        restriction = _default_restriction(n_columns)
    restriction = np.atleast_2d(np.asarray(restriction, dtype=np.float64))
    if restriction.shape[1] != n_columns:
        raise ValueError(
            f"restriction must hold {n_columns} columns; got {restriction.shape[1]}"
        )

    _, group_index = np.unique(clusters, return_inverse=True)
    group_index = np.asarray(group_index).ravel()
    n_groups = int(group_index.max()) + 1
    if n_groups < 2:
        raise ValueError("the wild cluster bootstrap needs at least 2 clusters")
    order = np.argsort(group_index, kind="stable")
    counts = np.bincount(group_index, minlength=n_groups)
    starts = np.zeros(n_groups, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    correction = (n_groups / (n_groups - 1.0)) * ((n_rows - 1.0) / (n_rows - n_columns))

    scores = np.add.reduceat((design * resid[:, None])[order], starts, axis=0)
    cov = _cluster_covariance(xtx_inv, scores, correction)
    se_cluster = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    signs = _rademacher(n_boot, n_groups, seed)

    def one_test(rule: np.ndarray) -> tuple[float, float]:
        """Return the observed Wald and the bootstrap p value of one rule."""
        observed = _wald(beta, cov, rule)
        beta_null, resid_null = _restricted_fit(y, design, xtx_inv, rule)
        walds = _null_walds(
            design,
            xtx_inv,
            design @ beta_null,
            resid_null,
            rule,
            signs,
            group_index,
            order,
            starts,
            correction,
        )
        p_value = (1.0 + float(np.count_nonzero(walds >= observed))) / (1.0 + n_boot)
        return observed, p_value

    wald_obs, joint_p = one_test(restriction)

    identity = np.eye(n_columns, dtype=np.float64)
    p_values = np.empty(n_columns, dtype=np.float64)
    for column in range(n_columns):
        p_values[column] = one_test(identity[column : column + 1])[1]

    return {
        "beta": beta,
        "se_cluster": se_cluster,
        "wald_obs": float(wald_obs),
        "joint_p": float(joint_p),
        "p_values": p_values,
        "n_clusters": int(n_groups),
        "n_boot": int(n_boot),
    }
