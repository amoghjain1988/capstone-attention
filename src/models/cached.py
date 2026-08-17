"""Replay a stored forward pass. It cannot mask.

See CONTRACT.md section 5.5 for the signatures.

scripts/extract_attention.py runs the GPU once and writes the cache. This class
reads that cache, so every later analysis runs on a CPU.

CachedPredictor does NOT satisfy MaskablePredictor. A mask changes the forward
pass, and a stored answer cannot change. predict() therefore raises
NotImplementedError as soon as edge_mask is not None. Do not build the ablation
loop on this class. CONTRACT.md section 4.2 says the same.

The lookup key is the history itself. Every window carries a different history,
so the class hashes the observed positions and finds the stored answer. Nobody
has to pass a window_id, and the protocol of section 4.1 stays intact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data import schema
from src.models import base


def history_key(hist: np.ndarray) -> str:
    """Return a stable key for one window history.

    The positions are rounded to the nearest micrometre first, so a float that
    survived a parquet round trip still matches.
    """
    array = np.round(np.asarray(hist, dtype=np.float64), 6)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


class CachedPredictor:
    """Replay the cached unmasked prediction and the cached attention."""

    def __init__(
        self,
        parquet_dir: Path = config.INTERIM_DIR,
        processed_dir: Path = config.PROCESSED_DIR,
        n_hist: int = config.N_HIST,
        n_fut: int = config.N_FUT,
    ) -> None:
        """Load every cached scene. Raise when no scene is cached."""
        self.parquet_dir = Path(parquet_dir)
        self.n_hist = int(n_hist)
        self.n_fut = int(n_fut)

        windows = schema.read("windows", processed_dir)
        self.windows = windows.loc[windows["eligible"]].copy()

        self.predictions: dict[str, np.ndarray] = {}
        frames: list[pd.DataFrame] = []
        self.scenes: list[str] = []

        for scene in config.SCENES:
            edge_file = self.parquet_dir / f"attention_{scene}.parquet"
            pred_file = self.parquet_dir / f"predict_{scene}.npz"
            if not edge_file.exists() or not pred_file.exists():
                continue
            self.scenes.append(scene)
            frames.append(pd.read_parquet(edge_file))
            with np.load(pred_file) as handle:
                for window_id in handle.files:
                    self.predictions[window_id] = handle[window_id]

        if not self.scenes:
            raise FileNotFoundError(
                f"no cache under {self.parquet_dir}. "
                "Run scripts/extract_attention.py on a GPU machine first."
            )

        self.edges = pd.concat(frames, ignore_index=True)

        # Map the history of every cached window onto its key.
        self._by_key: dict[str, str] = {}
        cached = self.windows.loc[
            self.windows["window_id"].astype(str).isin(self.predictions)
        ]
        for _, row in cached.iterrows():
            hist = schema.unpack_hist(row, self.n_hist)
            self._by_key[history_key(hist)] = str(row["window_id"])

    # -- the protocol -----------------------------------------------------

    def predict(
        self,
        hist: np.ndarray,
        edge_mask: np.ndarray | None = None,
        draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the stored (K, N, 12, 2) prediction of this window.

        Raise NotImplementedError when edge_mask is not None, because a masked
        re-run needs the GPU.
        """
        if edge_mask is not None:
            raise NotImplementedError(
                "CachedPredictor cannot mask. A masked re-run changes the forward "
                "pass, so it needs the GPU. Use AgentFormerPredictor."
            )
        hist = base.check_history(hist, self.n_hist)
        window_id = self.window_id_of(hist)
        stored = self.predictions[window_id].astype(np.float64)
        return base.check_prediction(stored, int(hist.shape[0]), self.n_fut)

    def attention(self, hist: np.ndarray) -> dict[str, np.ndarray]:
        """Raise. The cache stores the collapsed edges, not the token map.

        Read the collapsed graph with edges_of() instead. The token map is far
        too large to keep for every window.
        """
        raise NotImplementedError(
            "the cache stores the collapsed edges, not the token map. "
            "Call edges_of(window_id), or re-run AgentFormerPredictor."
        )

    # -- the cache --------------------------------------------------------

    def window_id_of(self, hist: np.ndarray) -> str:
        """Return the window key of a history. Raise when it is not cached."""
        key = history_key(hist)
        if key not in self._by_key:
            raise KeyError(
                "this history is not in the cache. Run scripts/extract_attention.py "
                "for its scene."
            )
        return self._by_key[key]

    def edges_of(
        self,
        window_id: str,
        module: str = config.PRIMARY_MODULE,
        time_agg: str = config.PRIMARY_TIME_AGG,
    ) -> pd.DataFrame:
        """Return the cached edges into the ego of one window, strongest first."""
        block = self.edges.loc[
            (self.edges["window_id"] == str(window_id))
            & (self.edges["module"] == module)
            & (self.edges["time_agg"] == time_agg)
        ]
        return block.sort_values("rank_attn", kind="mergesort").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.predictions)

    def __repr__(self) -> str:
        return (
            f"CachedPredictor(scenes={self.scenes}, windows={len(self)}, "
            f"edges={len(self.edges):,})"
        )
