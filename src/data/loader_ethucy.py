"""Read the raw scene files into the trajectories table.

See CONTRACT.md section 5.2 for the signatures.

The raw file holds 17 space-separated columns. AgentFormer reads the frame from
column 0, the agent id from column 1, x from column 13 and y from column 15.
This module keeps that convention, so our positions and the model positions are
the same numbers.

The scene univ has two source files, students001 and students003. The two files
are separate recordings, and both start at frame 0 with overlapping agent ids.
The loader therefore adds an offset to the second file. The offset is much
larger than one window, so no window can span the two recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data import schema


def _raw_path(source: str, raw_dir: Path) -> Path:
    """Return the path of one downloaded source file."""
    return Path(raw_dir) / f"{source.replace('/', '__')}.txt"


def load_file(source: str, raw_dir: Path) -> pd.DataFrame:
    """Read one raw text file. Return frame, agent_id, x and y.

    The function does no cleaning. It only selects the four columns.
    """
    path = _raw_path(source, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/download_ethucy.py first."
        )

    table = np.genfromtxt(path, delimiter=" ", dtype=str)
    if table.ndim == 1:
        table = table.reshape(1, -1)

    frame = table[:, config.COL_FRAME].astype(np.float64)
    agent = table[:, config.COL_AGENT].astype(np.float64)
    x = table[:, config.COL_X].astype(np.float64)
    y = table[:, config.COL_Y].astype(np.float64)

    return pd.DataFrame(
        {
            "frame": frame.astype(np.int32),
            "agent_id": agent.astype(np.int32),
            "x": x,
            "y": y,
        }
    )


def load_scene(scene: str, raw_dir: Path) -> pd.DataFrame:
    """Read every source file of one scene into one table.

    Columns: scene, frame, agent_id, x, y. The rows are sorted by agent and
    then by frame.
    """
    if scene not in config.SCENE_SOURCES:
        raise KeyError(f"{scene!r} is not a known scene")

    parts: list[pd.DataFrame] = []
    for index, source in enumerate(config.SCENE_SOURCES[scene]):
        part = load_file(source, raw_dir)
        offset = index * config.SOURCE_FILE_OFFSET
        part["frame"] = (part["frame"] + offset).astype(np.int32)
        part["agent_id"] = (part["agent_id"] + offset).astype(np.int32)
        part["source"] = source
        parts.append(part)

    frame = pd.concat(parts, ignore_index=True)
    frame.insert(0, "scene", scene)
    frame = frame.sort_values(["agent_id", "frame"], kind="mergesort")
    return frame.reset_index(drop=True)


def load_all(raw_dir: Path) -> pd.DataFrame:
    """Read every scene. Return the raw trajectories table."""
    frames = [load_scene(scene, raw_dir) for scene in config.SCENES]
    out = pd.concat(frames, ignore_index=True)
    out = schema.cast(out, "trajectories", partial=True)
    schema.validate(out, "trajectories", partial=True)
    return out
