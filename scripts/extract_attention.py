"""Run inference once and cache the attention maps.

See CONTRACT.md section 5.1 for the signatures.

This is the only stage that needs a GPU. It writes two files per scene.

    data/interim/attention_<scene>.parquet   one row per edge into the ego,
                                             for every module and time collapse
    data/interim/predict_<scene>.npz         the unmasked prediction of every
                                             window, (K, N, 12, 2) float32

After this stage every other person works from the cache on a CPU. A MASKED
re-run still needs the GPU, because a mask changes the forward pass. See
src/models/cached.py.

The script is idempotent. It records which windows it finished and skips them
on a second run. Pass --scene to do one scene, and --limit to do a few windows.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.attention import aggregate, rank  # noqa: E402
from src.data import schema  # noqa: E402


def attention_path(scene: str) -> Path:
    """Return the parquet path of the cached attention of one scene."""
    return config.INTERIM_DIR / f"attention_{scene}.parquet"


def predict_path(scene: str) -> Path:
    """Return the npz path of the cached predictions of one scene."""
    return config.INTERIM_DIR / f"predict_{scene}.npz"


def edges_of_window(
    maps: dict[str, np.ndarray],
    row: pd.Series,
    ego_index: int,
    n_agents: int,
) -> list[dict]:
    """Return one record per module, time collapse and edge into the ego."""
    order = [int(a) for a in row["agent_order"]]
    records: list[dict] = []

    for module, raw in maps.items():
        entropy = aggregate.agent_row_entropy(raw, n_agents)
        for time_agg in config.TIME_AGGS:
            graph = aggregate.collapse(raw, n_agents, time_agg)
            ranked = rank.rank_edges(graph, ego_index, order)
            for record in ranked.to_dict("records"):
                records.append(
                    {
                        "window_id": str(row["window_id"]),
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


def run_scene(
    scene: str, windows: pd.DataFrame, limit: int | None, force: bool
) -> None:
    """Cache the attention and the prediction of every eligible window."""
    from src.models.agentformer import AgentFormerPredictor

    block = windows.loc[(windows["scene"] == scene) & windows["eligible"]]
    block = block.sort_values("t0", kind="mergesort")
    if limit:
        block = block.head(int(limit))
    if block.empty:
        print(f"{scene}: no eligible window")
        return

    done: set[str] = set()
    old_edges = pd.DataFrame()
    old_pred: dict[str, np.ndarray] = {}
    if not force and attention_path(scene).exists():
        old_edges = pd.read_parquet(attention_path(scene))
        done = set(old_edges["window_id"].astype(str))
        if predict_path(scene).exists():
            with np.load(predict_path(scene)) as handle:
                old_pred = {key: handle[key] for key in handle.files}

    todo = block.loc[~block["window_id"].astype(str).isin(done)]
    print(
        f"{scene}: {len(block)} eligible, {len(done)} cached, {len(todo)} to run"
    )
    if todo.empty:
        return

    model = AgentFormerPredictor(scene, device="cuda", n_samples=config.N_SAMPLES)
    print(f"  {model}")

    records: list[dict] = []
    predictions: dict[str, np.ndarray] = dict(old_pred)
    start = time.time()

    for counter, (_, row) in enumerate(todo.iterrows(), 1):
        n_agents = int(row["n_agents"])
        hist = schema.unpack_hist(row, config.N_HIST)
        ego_index = schema.ego_index(row)

        maps = model.attention(hist)
        records.extend(edges_of_window(maps, row, ego_index, n_agents))
        predictions[str(row["window_id"])] = model.predict(hist).astype(np.float32)

        if counter % 50 == 0 or counter == len(todo):
            rate = counter / (time.time() - start)
            left = (len(todo) - counter) / rate if rate else 0.0
            print(
                f"  {counter}/{len(todo)}  {rate:.1f} windows/s  "
                f"{left / 60:.1f} min left",
                flush=True,
            )

    fresh = pd.DataFrame(records)
    edges = pd.concat([old_edges, fresh], ignore_index=True) if len(old_edges) else fresh
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(attention_path(scene), index=False)
    np.savez_compressed(predict_path(scene), **predictions)
    print(
        f"  wrote {attention_path(scene).name} ({len(edges):,} edge rows) and "
        f"{predict_path(scene).name} ({len(predictions):,} windows)"
    )


def main(scene: str | None = None, limit: int | None = None, force: bool = False) -> None:
    """Cache every scene, or the one that was named."""
    config.make_dirs()
    windows = schema.read("windows", config.PROCESSED_DIR)
    scenes = [scene] if scene else list(config.SCENES)

    for name in scenes:
        if name not in config.SCENES:
            raise SystemExit(f"{name!r} is not a scene. Choose from {config.SCENES}.")
        run_scene(name, windows, limit, force)

    print("\nThe cache is ready. Everybody else can now work without a GPU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache attention and predictions.")
    parser.add_argument("--scene", default=None, help="one scene, or every scene")
    parser.add_argument("--limit", type=int, default=None, help="only the first N windows")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    main(**vars(parser.parse_args()))
