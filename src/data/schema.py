"""Column names and dtypes for the seven tables.

See CONTRACT.md section 3 for the signatures.

Every producer calls validate() before it writes. Every reader calls read() and
gets the dtypes back. The schema is the single place that knows a column name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The join key
# ---------------------------------------------------------------------------


def window_id(scene: str, t0: int) -> str:
    """Return the join key of one window.

    The key is the scene name, an underscore, and the first observed frame in
    six digits.
    """
    return f"{scene}_{int(t0):06d}"


def split_window_id(wid: str) -> tuple[str, int]:
    """Return the scene and the first observed frame of a window key."""
    scene, _, t0 = wid.rpartition("_")
    return scene, int(t0)


# ---------------------------------------------------------------------------
# The seven tables
# ---------------------------------------------------------------------------

TRAJECTORIES: dict[str, str] = {
    "scene": "category",
    "frame": "int32",
    "agent_id": "int32",
    "x": "float64",
    "y": "float64",
    "vx": "float64",
    "vy": "float64",
    "speed": "float64",
    "heading": "float64",
    "nearest_dist": "float64",
    "density": "float64",
}
TRAJECTORIES_KEY = ["scene", "frame", "agent_id"]

# loader_ethucy.py writes only these columns. clean.py adds the rest.
TRAJECTORIES_RAW: dict[str, str] = {
    "scene": "category",
    "frame": "int32",
    "agent_id": "int32",
    "x": "float64",
    "y": "float64",
}

WINDOWS: dict[str, str] = {
    "window_id": "string",
    "scene": "category",
    "t0": "int32",
    "ego_id": "int32",
    "agent_order": "object",  # list[int32], length N
    "n_agents": "int16",
    "hist": "object",  # list[float64], flattened (N, 8, 2)
    "fut": "object",  # list[float64], flattened (N, 12, 2)
    "overlap_fraction": "float64",
    "eligible": "bool",
}
WINDOWS_KEY = ["window_id"]

ATTENTION_EDGES: dict[str, str] = {
    "window_id": "string",
    "src": "int32",
    "dst": "int32",
    "module": "category",
    "time_agg": "category",
    "attn": "float64",
    "row_entropy": "float64",
    "dist_at_last_frame": "float64",
    "closing_speed": "float64",
    "inv_ttc": "float64",
    "rank_attn": "int16",
    "rank_dist": "int16",
}
ATTENTION_EDGES_KEY = ["window_id", "src", "dst", "module", "time_agg"]

PERTURBATION_CURVES: dict[str, str] = {
    "window_id": "string",
    "ego_id": "int32",
    "arm": "category",
    "n_removed": "int16",
    "removed_mass": "float64",
    "edge_src": "int32",
    "draw_id": "int16",
    "mask_policy": "category",
    "run_id": "string",
    "shift": "float64",
}
PERTURBATION_CURVES_KEY = [
    "window_id",
    "arm",
    "n_removed",
    "draw_id",
    "mask_policy",
]

FAITHFULNESS: dict[str, str] = {
    "window_id": "string",
    "ego_id": "int32",
    "n_edges": "int16",
    "floor": "float64",
    "ceiling": "float64",
    "fi": "float64",
}
FAITHFULNESS_KEY = ["window_id"]

PREDICTIONS: dict[str, str] = {
    "window_id": "string",
    "ade": "float64",
    "fde": "float64",
    "minade_20": "float64",
    "minfde_20": "float64",
    "collision": "bool",
}
PREDICTIONS_KEY = ["window_id"]

VERDICTS: dict[str, str] = {
    "hypothesis": "string",
    "role": "category",
    "test_used": "string",
    "why": "string",
    "sided": "category",
    "effect": "float64",
    "effect_name": "string",
    "ci_low": "float64",
    "ci_high": "float64",
    "p_raw": "float64",
    "p_adj": "float64",
    "n_clusters": "int32",
    "verdict": "string",
}
VERDICTS_KEY = ["hypothesis"]

TABLES: dict[str, dict[str, str]] = {
    "trajectories": TRAJECTORIES,
    "windows": WINDOWS,
    "attention_edges": ATTENTION_EDGES,
    "perturbation_curves": PERTURBATION_CURVES,
    "faithfulness": FAITHFULNESS,
    "predictions": PREDICTIONS,
    "verdicts": VERDICTS,
}

KEYS: dict[str, list[str]] = {
    "trajectories": TRAJECTORIES_KEY,
    "windows": WINDOWS_KEY,
    "attention_edges": ATTENTION_EDGES_KEY,
    "perturbation_curves": PERTURBATION_CURVES_KEY,
    "faithfulness": FAITHFULNESS_KEY,
    "predictions": PREDICTIONS_KEY,
    "verdicts": VERDICTS_KEY,
}

# The allowed values of every category column.
CATEGORIES: dict[str, tuple[str, ...]] = {
    # "synthetic" is the planted scene of src/data/synthetic.py. It is a test
    # fixture. It never joins the five real scenes in a result table.
    "scene": ("eth", "hotel", "univ", "zara1", "zara2", "synthetic"),
    "module": ("encoder", "decoder", "cross"),
    "time_agg": ("mean", "max", "last"),
    "arm": ("morf", "lerf", "weight_matched", "nearest", "random", "single"),
    "mask_policy": ("logit_neg_inf", "weight_zero"),
    "role": ("primary", "supporting"),
    "sided": ("one", "two"),
}

# The reason column of every drops.csv file.
DROPS: dict[str, str] = {
    "stage": "string",
    "scene": "string",
    "key": "string",
    "reason": "string",
    "n_rows": "int64",
}


# ---------------------------------------------------------------------------
# Cast and check
# ---------------------------------------------------------------------------


def cast(df: pd.DataFrame, table: str, partial: bool = False) -> pd.DataFrame:
    """Return a copy of df with the dtypes of the named table.

    Set partial to True to cast only the columns that are present.
    """
    spec = TABLES[table]
    out = df.copy()
    for name, dtype in spec.items():
        if name not in out.columns:
            if partial:
                continue
            raise KeyError(f"{table}: the column {name!r} is missing")
        if dtype == "category":
            allowed = CATEGORIES.get(name)
            if allowed is None:
                out[name] = out[name].astype("category")
            else:
                out[name] = pd.Categorical(out[name], categories=list(allowed))
        elif dtype == "object":
            continue
        else:
            out[name] = out[name].astype(dtype)
    return out


def validate(df: pd.DataFrame, table: str, partial: bool = False) -> None:
    """Raise if df does not match the named table.

    The function checks three things. First, every column is present. Second,
    no key column holds a null. Third, the key is unique.
    """
    spec = TABLES[table]
    expected = set(spec) if not partial else set(spec) & set(df.columns)
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"{table}: these columns are missing: {sorted(missing)}")

    key = [column for column in KEYS[table] if column in df.columns]
    if not key:
        return

    null_key = df[key].isna().any(axis=1)
    if bool(null_key.any()):
        raise ValueError(f"{table}: {int(null_key.sum())} rows hold a null key")

    duplicated = int(df.duplicated(subset=key).sum())
    if duplicated:
        raise ValueError(f"{table}: the key {key} repeats on {duplicated} rows")

    for name in key + [c for c in df.columns if c in CATEGORIES]:
        allowed = CATEGORIES.get(name)
        if allowed is None:
            continue
        bad = set(pd.Series(df[name]).dropna().astype(str)) - set(allowed)
        if bad:
            raise ValueError(f"{table}: the column {name!r} holds {sorted(bad)}")


# ---------------------------------------------------------------------------
# Read and write
# ---------------------------------------------------------------------------


def path_of(table: str, processed_dir: Path) -> Path:
    """Return the parquet path of a table."""
    return Path(processed_dir) / f"{table}.parquet"


def write(df: pd.DataFrame, table: str, processed_dir: Path) -> Path:
    """Cast, check and write a table. Return the path."""
    out = cast(df, table)
    validate(out, table)
    target = path_of(table, processed_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return target


def read(table: str, processed_dir: Path) -> pd.DataFrame:
    """Read a table and cast it back to the schema dtypes."""
    frame = pd.read_parquet(path_of(table, processed_dir))
    return cast(frame, table, partial=True)


# ---------------------------------------------------------------------------
# The drop log
# ---------------------------------------------------------------------------


def empty_drops() -> pd.DataFrame:
    """Return an empty drop log with the right columns."""
    return pd.DataFrame({name: pd.Series(dtype=t) for name, t in DROPS.items()})


def drop_row(stage: str, scene: str, key: str, reason: str, n_rows: int) -> dict:
    """Return one row of the drop log."""
    return {
        "stage": stage,
        "scene": scene,
        "key": key,
        "reason": reason,
        "n_rows": int(n_rows),
    }


# ---------------------------------------------------------------------------
# Window payload helpers
# ---------------------------------------------------------------------------


def unpack_hist(row: pd.Series, n_hist: int = 8) -> np.ndarray:
    """Return the observed paths of one window, shape (N, n_hist, 2)."""
    n_agents = int(row["n_agents"])
    return np.asarray(row["hist"], dtype=np.float64).reshape(n_agents, n_hist, 2)


def unpack_fut(row: pd.Series, n_fut: int = 12) -> np.ndarray:
    """Return the true future paths of one window, shape (N, n_fut, 2)."""
    n_agents = int(row["n_agents"])
    return np.asarray(row["fut"], dtype=np.float64).reshape(n_agents, n_fut, 2)


def ego_index(row: pd.Series) -> int:
    """Return the position of the ego on the agent axis.

    Index 0 of agent_order is not the ego. Always call this function.
    """
    order = list(row["agent_order"])
    return order.index(int(row["ego_id"]))
