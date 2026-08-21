"""The brute-force single-edge sweep and the faithfulness index.

See CONTRACT.md section 5.9 for the signatures of brute_force_single and
faithfulness_index. Everything from build_faithfulness_table onward is this
module's own design: CONTRACT.md names the two output artefacts this module
must produce, but gives no signature for the driver that builds them. Confirm
the shape of that driver with the team before treating faithfulness.parquet
as final.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.ablation.arms import order_edges
from src.ablation.mask import build_mask, masked_predict
from src.ablation.shift import ego_path, shift
from src.data import schema


def brute_force_single(
    window_row: pd.Series,
    model,
    edges: pd.DataFrame,
    draws: np.ndarray,
) -> pd.DataFrame:
    """Remove each edge into the ego on its own. One row per edge.

    Columns: src, shift. Cost is E forward passes, one per edge in `edges`.
    Use the SAME `draws` for the unmasked baseline and every masked pass, so
    every shift is comparable.
    """
    hist = schema.unpack_hist(window_row, config.N_HIST)
    ego_pos = schema.ego_index(window_row)
    n_agents = int(hist.shape[0])
    agent_order = [int(a) for a in window_row["agent_order"]]
    index_of = {agent_id: position for position, agent_id in enumerate(agent_order)}

    base_pred = model.predict(hist, edge_mask=None, draws=draws)
    base_ego = ego_path(base_pred, ego_pos)

    src_ids = sorted(int(s) for s in edges["src"].unique())

    rows: list[dict] = []
    for src in src_ids:
        mask = build_mask(n_agents, [index_of[src]], ego_pos)
        masked_pred = masked_predict(model, hist, mask, draws, config.PRIMARY_MASK_POLICY)
        masked_ego = ego_path(masked_pred, ego_pos)
        rows.append({"src": src, "shift": shift(masked_ego, base_ego)})
    return pd.DataFrame(rows)


def faithfulness_index(single: pd.DataFrame, morf_edge: int, tol: float = 1e-6) -> float:
    """Return the faithfulness index for one window.

    `single` is the brute_force_single table: one row per edge, columns src
    and shift. `morf_edge` is the src id attention ranked strongest.

    floor   = single.shift.mean()   the expected shift of a random single edge
    ceiling = single.shift.max()    the truly strongest edge
    FI = (shift(morf_edge) - floor) / (ceiling - floor)

    FI = 1 means attention picked the strongest edge. FI = 0 means attention
    did no better than chance. FI below 0 means attention did worse than
    chance. Return NaN when ceiling - floor <= tol. config.MIN_EGO_EDGES
    already removes the E == 1 case that makes this exact, but the function
    stays defensive for any caller that skips that filter.
    """
    if len(single) == 0:
        raise ValueError("single must hold at least one edge")

    floor = float(single["shift"].mean())
    ceiling = float(single["shift"].max())
    if ceiling - floor <= tol:
        return float("nan")

    match = single.loc[single["src"] == int(morf_edge), "shift"]
    if match.empty:
        raise ValueError(f"morf_edge {morf_edge} is not a row of single")
    morf_shift = float(match.iloc[0])

    return (morf_shift - floor) / (ceiling - floor)


# ---------------------------------------------------------------------------
# The driver that produces the two output artefacts. This part is this
# module's own design -- see the module docstring.
# ---------------------------------------------------------------------------


def build_faithfulness_table(
    windows: pd.DataFrame,
    edges_by_window: dict[str, pd.DataFrame],
    model,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run brute_force_single on every window and score it.

    `windows` is the eligible slice of the windows table. `edges_by_window`
    maps window_id to that window's edge table (src, attn, ...), already
    filtered to config.PRIMARY_MODULE and config.PRIMARY_TIME_AGG.

    Returns (faithfulness, attrition). `faithfulness` matches
    schema.FAITHFULNESS. `attrition` holds one row per scene actually present
    in `windows`, with the eligible window count, how many have E == 1, how
    many have E <= 2, and how many gave NaN.
    """
    draws = np.arange(config.N_SAMPLES, dtype=np.int64) + int(seed)

    rows: list[dict] = []
    attrition_rows: list[dict] = []

    for scene in sorted(windows["scene"].astype(str).unique()):
        scene_windows = windows.loc[windows["scene"].astype(str) == scene]
        n_e1 = 0
        n_e_le2 = 0
        n_nan = 0

        for _, window_row in scene_windows.iterrows():
            window_id = window_row["window_id"]
            edges = edges_by_window.get(window_id)
            if edges is None or len(edges) == 0:
                continue  # defensive: every eligible window should carry edges

            n_edges = int(edges["src"].nunique())
            if n_edges == 1:
                n_e1 += 1
            if n_edges <= 2:
                n_e_le2 += 1

            single = brute_force_single(window_row, model, edges, draws)
            morf_edge = order_edges(edges, "morf", seed)[0]
            fi = faithfulness_index(single, morf_edge)
            if np.isnan(fi):
                n_nan += 1

            rows.append(
                {
                    "window_id": window_id,
                    "ego_id": int(window_row["ego_id"]),
                    "n_edges": n_edges,
                    "floor": float(single["shift"].mean()),
                    "ceiling": float(single["shift"].max()),
                    "fi": fi,
                }
            )

        attrition_rows.append(
            {
                "scene": scene,
                "n_eligible": int(len(scene_windows)),
                "n_e_eq_1": n_e1,
                "n_e_le_2": n_e_le2,
                "n_nan": n_nan,
            }
        )

    faithfulness = pd.DataFrame(rows)
    attrition = pd.DataFrame(attrition_rows)
    return faithfulness, attrition


def write_faithfulness_outputs(
    windows: pd.DataFrame,
    edges_by_window: dict[str, pd.DataFrame],
    model,
    seed: int = config.SEED,
) -> tuple[Path, Path]:
    """Build the table, then write both required output files."""
    faithfulness, attrition = build_faithfulness_table(windows, edges_by_window, model, seed)

    fi_path = schema.write(faithfulness, "faithfulness", config.PROCESSED_DIR)

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    attrition_path = config.TABLE_DIR / "fi_attrition.csv"
    attrition.to_csv(attrition_path, index=False)

    return fi_path, attrition_path
