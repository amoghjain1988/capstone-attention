"""Build a scene with a known interaction graph.

See CONTRACT.md section 5.2 for the signatures.

The purpose is the gate of tests/test_synthetic.py. We plant a graph, we run
the whole ranking pipeline, and we check that the pipeline finds the graph
again. If the method cannot recover a graph that we planted ourselves, then no
number from the real data means anything.

The rule of the simulation is simple. An agent reacts only to the agents on its
own influencer list. Every other agent is invisible to it. Therefore an edge
that is not on the list carries no information about the future, and the true
answer is known for every pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema

SCENE = "synthetic"

# The strength of the reaction. A larger number gives a clearer signal.
REPULSION = 2.0
RELAX = 1.5
MIN_RANGE_M = 0.35


def _plant_graph(
    n_agents: int, n_real: int, generator: np.random.Generator
) -> np.ndarray:
    """Return the (N, N) boolean influence matrix.

    Entry (dst, src) is True when src influences dst. The diagonal is False.
    """
    influence = np.zeros((n_agents, n_agents), dtype=bool)
    for dst in range(n_agents):
        others = np.array([a for a in range(n_agents) if a != dst])
        chosen = generator.choice(others, size=min(n_real, others.size), replace=False)
        influence[dst, chosen] = True
    return influence


def _simulate(
    n_agents: int,
    n_frames: int,
    influence: np.ndarray,
    generator: np.random.Generator,
    dt: float,
) -> np.ndarray:
    """Return the positions, shape (n_agents, n_frames, 2).

    Every agent starts on a circle and walks toward the far side. The paths
    cross in the middle, so the reaction term is large where it matters.
    """
    radius = 8.0
    angle = np.linspace(0.0, 2.0 * np.pi, n_agents, endpoint=False)
    angle = angle + generator.normal(0.0, 0.05, size=n_agents)

    position = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    target = -position
    direction = target - position
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    desired = direction * generator.uniform(1.0, 1.6, size=(n_agents, 1))
    velocity = desired.copy()

    out = np.empty((n_agents, n_frames, 2), dtype=np.float64)
    for step in range(n_frames):
        out[:, step, :] = position

        delta = position[:, None, :] - position[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        distance = np.clip(distance, MIN_RANGE_M, None)
        push = delta / (distance**3)[:, :, None]
        push = np.where(influence[:, :, None], push, 0.0)
        force = REPULSION * push.sum(axis=1)

        velocity = velocity + (RELAX * (desired - velocity) + force) * dt
        position = position + velocity * dt

    return out


def make_scene(
    n_agents: int = 8,
    n_frames: int = 200,
    seed: int = config.SEED,
    n_real: int = 2,
    dt: float = config.DT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trajectories, true_edges).

    The table trajectories holds scene, frame, agent_id, x and y. Every agent
    is present in every frame, so every window holds every agent.

    The table true_edges holds window_id, src, dst and is_real. It carries one
    row for every ordered pair of agents in every window. Filter it on
    dst == ego_id to get the edges that point into one ego.
    """
    generator = np.random.default_rng(int(seed))
    influence = _plant_graph(int(n_agents), int(n_real), generator)
    path = _simulate(int(n_agents), int(n_frames), influence, generator, float(dt))

    agent_ids = np.arange(int(n_agents), dtype=np.int32)
    frames = np.arange(int(n_frames), dtype=np.int32)
    trajectories = pd.DataFrame(
        {
            "scene": SCENE,
            "frame": np.repeat(frames, int(n_agents)),
            "agent_id": np.tile(agent_ids, int(n_frames)),
            "x": path[:, :, 0].T.reshape(-1),
            "y": path[:, :, 1].T.reshape(-1),
        }
    )
    trajectories = schema.cast(trajectories, "trajectories", partial=True)

    length = config.N_HIST + config.N_FUT
    starts = range(0, int(n_frames) - length + 1, config.WINDOW_STRIDE)
    src, dst = np.meshgrid(agent_ids, agent_ids, indexing="ij")
    off_diagonal = src != dst
    pair_src = src[off_diagonal]
    pair_dst = dst[off_diagonal]
    is_real = influence[pair_dst, pair_src]

    edges = []
    for t0 in starts:
        edges.append(
            pd.DataFrame(
                {
                    "window_id": schema.window_id(SCENE, t0),
                    "src": pair_src.astype(np.int32),
                    "dst": pair_dst.astype(np.int32),
                    "is_real": is_real,
                }
            )
        )
    true_edges = pd.concat(edges, ignore_index=True)
    return trajectories, true_edges


def influence_matrix(n_agents: int = 8, seed: int = config.SEED, n_real: int = 2):
    """Return the planted matrix on its own, for a predictor that must obey it."""
    generator = np.random.default_rng(int(seed))
    return _plant_graph(int(n_agents), int(n_real), generator)
