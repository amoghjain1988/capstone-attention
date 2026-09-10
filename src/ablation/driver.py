"""The writer of the perturbation_curves table.

See CONTRACT.md section 3 for the table and docs/FINISH_PLAN.md section 3.10
for the signatures.

The module holds three functions.

    window_curves     every arm of one window
    run_scene         every window of one scene, with a checkpoint
    consistency_check the audit that proves the arms agree with the sweep

The audit is the reason this file exists as a separate step. At
config.PRIMARY_N_REMOVED of 1 every arm removes exactly one edge, and the
single arm already measured every single edge on its own. The two numbers must
therefore be the same float. When they differ, a mask, a seed or a draw array
drifted, and no result of the study is safe.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.ablation.arms import ARMS
from src.ablation.arms import order_edges
from src.ablation.curves import curve
from src.attention.edges import edges_by_window
from src.data import schema

# Write the checkpoint after this many fresh windows.
CHECKPOINT_EVERY = 20

# The four arms whose first removed edge comes straight from order_edges.
# weight_matched has no fixed order, so consistency_check names its edge from
# the removed mass instead.
ORDERED_ARMS = ("morf", "lerf", "nearest", "random")

# One (model, policy) pair per skip. The module warns once per pair, so a run
# over 480 windows prints the note once and not 480 times.
_POLICY_SKIPS: set[tuple[str, str]] = set()


# ---------------------------------------------------------------------------
# The unmasked baseline
# ---------------------------------------------------------------------------


class ReuseBaseline:
    """Serve the unmasked prediction of one window from memory.

    src/ablation/curves.py::curve computes its own unmasked baseline on every
    call, and window_curves calls it once per arm and once per extra mask
    policy. That costs 7 identical forward passes per window, and 6 of them
    are waste.

    This wrapper keeps the last unmasked answer and returns it again when the
    history and the draws repeat. Every masked call passes straight through.
    The project already asserts that this substitution is safe:
    src/faithfulness/sanity.py::zero_check demands that two unmasked calls
    with the same draws return the same array, exactly. A model that fails
    that check fails the study, not this wrapper.

    scripts/run_ablation.py runs zero_check against the raw model, never
    against this wrapper, so the check stays a true test.
    """

    def __init__(self, model) -> None:
        """Wrap one model."""
        self.model = model
        self._key: tuple | None = None
        self._value = None

    def predict(self, hist, edge_mask=None, draws=None, **extra):
        """Return (K, N, 12, 2). Reuse the unmasked answer of this window."""
        if edge_mask is not None:
            return self.model.predict(
                hist, edge_mask=edge_mask, draws=draws, **extra
            )

        key = (
            np.asarray(hist, dtype=np.float64).tobytes(),
            None if draws is None else np.asarray(draws, dtype=np.int64).tobytes(),
        )
        if key != self._key:
            self._value = self.model.predict(hist, edge_mask=None, draws=draws, **extra)
            self._key = key
        return self._value

    def attention(self, hist):
        """Pass the attention call straight through."""
        return self.model.attention(hist)

    def __repr__(self) -> str:
        return f"ReuseBaseline({self.model!r})"


# ---------------------------------------------------------------------------
# The cost model
# ---------------------------------------------------------------------------


def forward_passes_per_window(
    n_edges: int, n_policies: int = len(config.MASK_POLICIES)
) -> int:
    """Return how many forward passes one window costs.

    E is the edge count into the ego. One unmasked baseline serves the whole
    window. Under the primary policy the arms morf, lerf, nearest, random and
    single each cost E masked passes, and weight_matched costs at most E more.
    Every other mask policy adds the single arm alone, which is E more passes.

    The total is therefore 1 + 6E for one policy, and 7E + 1 for the two
    policies of config.MASK_POLICIES. The number is an upper bound, because
    weight_matched skips a step whose chosen set repeats the step before it.
    """
    edges = int(n_edges)
    policies = max(int(n_policies), 1)
    return 1 + 6 * edges + (policies - 1) * edges


def scene_forward_passes(
    windows: pd.DataFrame,
    edges_primary: pd.DataFrame,
    n_policies: int = len(config.MASK_POLICIES),
) -> int:
    """Return the forward pass estimate of a block of windows."""
    counts = (
        edges_primary.groupby(edges_primary["window_id"].astype(str), sort=False)["src"]
        .nunique()
        .to_dict()
        if len(edges_primary)
        else {}
    )
    total = 0
    for _, row in windows.iterrows():
        window_id = str(row["window_id"])
        n_edges = int(counts.get(window_id, int(row["n_agents"]) - 1))
        total += forward_passes_per_window(n_edges, n_policies)
    return int(total)


# ---------------------------------------------------------------------------
# One window
# ---------------------------------------------------------------------------


def policy_skips() -> tuple[tuple[str, str], ...]:
    """Return every (model, policy) pair that this module skipped."""
    return tuple(sorted(_POLICY_SKIPS))


def reset_policy_skips() -> None:
    """Forget every skip. A test calls this before it asserts on the set."""
    _POLICY_SKIPS.clear()


def _note_skip(model, policy: str, error: Exception) -> None:
    """Record one skipped mask policy and warn about it once."""
    name = type(model).__name__
    if (name, policy) in _POLICY_SKIPS:
        return
    _POLICY_SKIPS.add((name, policy))
    warnings.warn(
        f"{name} does not honour the mask policy {policy!r}, so the driver "
        f"skips it and writes only {config.PRIMARY_MASK_POLICY!r}. "
        f"src/ablation/mask.py saw the ReuseBaseline wrapper around {name} "
        f"and said: {error}",
        stacklevel=3,
    )


def window_curves(
    window_row: pd.Series,
    model,
    edges_window: pd.DataFrame,
    seed: int = config.SEED,
    policies: tuple[str, ...] = config.MASK_POLICIES,
) -> pd.DataFrame:
    """Return every curve row of one window.

    Under config.PRIMARY_MASK_POLICY the function runs all six arms of
    src.ablation.arms.ARMS. Under every other policy of `policies` it runs the
    single arm alone, because at config.PRIMARY_N_REMOVED of 1 every other arm
    is a lookup into the single rows and costs no forward pass.

    A model whose predict method does not accept mask_policy cannot tell the
    two policies apart. src/ablation/mask.py::masked_predict raises
    NotImplementedError for such a model on any policy but the primary one.
    MockPredictor is one such model. The function catches that error, drops
    the extra policy, and records the fact once. It never writes a silent copy
    of the primary policy under a second name.

    `edges_window` is one window's slice of the primary attention_edges table.
    It must carry window_id, src, attn and rank_dist.
    """
    if len(edges_window) == 0:
        raise ValueError(
            f"window {window_row['window_id']!r} carries no edge. "
            "Filter the windows on the attention table first."
        )

    reuse = ReuseBaseline(model)
    frames = [
        curve(window_row, reuse, edges_window, arm, seed, config.PRIMARY_MASK_POLICY)
        for arm in ARMS
    ]

    for policy in policies:
        if policy == config.PRIMARY_MASK_POLICY:
            continue
        try:
            frames.append(
                curve(window_row, reuse, edges_window, "single", seed, policy)
            )
        except NotImplementedError as error:
            _note_skip(model, policy, error)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# The checkpoint
# ---------------------------------------------------------------------------


def empty_curves() -> pd.DataFrame:
    """Return an empty perturbation_curves frame with plain dtypes."""
    plain = {
        "window_id": "string",
        "ego_id": "int64",
        "arm": "string",
        "n_removed": "int64",
        "removed_mass": "float64",
        "edge_src": "int64",
        "draw_id": "int64",
        "mask_policy": "string",
        "run_id": "string",
        "shift": "float64",
    }
    return pd.DataFrame({name: pd.Series(dtype=t) for name, t in plain.items()})


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with the dtypes that survive a parquet round trip.

    The checkpoint keeps arm and mask_policy as plain strings, not as a
    category. A category column that comes back from parquet carries its own
    category list, and a concatenation of two such columns falls back to
    object. Plain strings keep every read and every write identical, so the
    idempotence test compares like with like. schema.write casts to the
    contract dtypes at the end.
    """
    out = frame.copy()
    for name in ("window_id", "arm", "mask_policy", "run_id"):
        out[name] = out[name].astype(str)
    for name in ("ego_id", "n_removed", "edge_src", "draw_id"):
        out[name] = out[name].astype("int64")
    for name in ("removed_mass", "shift"):
        out[name] = out[name].astype("float64")
    return out.loc[:, list(schema.PERTURBATION_CURVES)]


def read_checkpoint(path: Path) -> pd.DataFrame:
    """Read one checkpoint. Return an empty frame when the file is absent."""
    path = Path(path)
    if not path.exists():
        return empty_curves()
    return _plain(pd.read_parquet(path))


def write_checkpoint(frame: pd.DataFrame, path: Path) -> Path:
    """Write one checkpoint with pandas. Return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _plain(frame).to_parquet(path, index=False)
    return path


def scene_checkpoint(scene: str, interim_dir: Path | None = None) -> Path:
    """Return the checkpoint path of one scene.

    The function reads config.INTERIM_DIR at call time, not at import time, so
    a test that points config at a temporary folder still gets the right path.
    """
    root = config.INTERIM_DIR if interim_dir is None else interim_dir
    return Path(root) / f"curves_{scene}.parquet"


def _join(done: pd.DataFrame, fresh: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the checkpoint and the fresh rows, empty frames aside."""
    parts = [part for part in [done, *fresh] if len(part)]
    if not parts:
        return empty_curves()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# One scene
# ---------------------------------------------------------------------------


def run_scene(
    scene: str,
    windows: pd.DataFrame,
    edges_primary: pd.DataFrame,
    model,
    checkpoint_path: Path,
    limit: int | None = None,
    force: bool = False,
    seed: int = config.SEED,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return every curve row of one scene. The function is idempotent.

    The function reads `checkpoint_path` first and skips every window that the
    file already holds. It appends to the file after every CHECKPOINT_EVERY
    fresh windows, and once more at the end. A second call with the same
    arguments therefore runs no forward pass and adds no row.

    Set `force` to True to ignore the file and run every window again. Set
    `limit` to run only the first few windows of the scene, in t0 order.
    Set `verbose` to True to print the rate and the time left.

    The driver takes whatever eligible windows it receives. It does not
    rebuild the windows table and it does not care about the stride.
    """
    target = Path(checkpoint_path)

    block = windows.loc[
        (windows["scene"].astype(str) == str(scene)) & windows["eligible"]
    ]
    block = block.sort_values("t0", kind="mergesort")
    if limit:
        block = block.head(int(limit))

    done_frame = empty_curves() if force else read_checkpoint(target)
    done = set(done_frame["window_id"].astype(str)) if len(done_frame) else set()

    by_window = edges_by_window(edges_primary)
    todo = [
        row
        for _, row in block.iterrows()
        if str(row["window_id"]) not in done
        and len(by_window.get(str(row["window_id"]), ())) > 0
    ]

    if verbose:
        print(
            f"{scene}: {len(block)} eligible, {len(done)} on the checkpoint, "
            f"{len(todo)} to run",
            flush=True,
        )

    fresh: list[pd.DataFrame] = []
    start = time.time()
    for counter, row in enumerate(todo, 1):
        edges_window = by_window[str(row["window_id"])]
        fresh.append(window_curves(row, model, edges_window, seed=seed))

        if counter % CHECKPOINT_EVERY == 0:
            write_checkpoint(_join(done_frame, fresh), target)
        if verbose and (counter % CHECKPOINT_EVERY == 0 or counter == len(todo)):
            elapsed = time.time() - start
            rate = counter / elapsed if elapsed > 0 else 0.0
            left = (len(todo) - counter) / rate if rate else 0.0
            print(
                f"  {counter}/{len(todo)}  {rate:.2f} windows/s  "
                f"{left / 60:.1f} min left",
                flush=True,
            )

    out = _join(done_frame, fresh)
    if len(out):
        write_checkpoint(out, target)
        return _plain(out)
    return empty_curves()


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

MISMATCH_COLUMNS = (
    "window_id",
    "arm",
    "mask_policy",
    "draw_id",
    "edge_src",
    "arm_shift",
    "single_shift",
    "difference",
    "reason",
)


def consistency_check(
    curves: pd.DataFrame, edges_primary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the rows where an arm at n_removed 1 disagrees with the sweep.

    At n_removed 1 an arm removes exactly one edge, and the single arm already
    measured that edge on its own, under the same draws and the same mask. The
    two shifts must be the same float, bit for bit. Any difference means a
    seed, a mask or a draw array drifted.

    Pass `edges_primary` to name the removed edge with
    src/ablation/arms.py::order_edges, which is what the arms themselves used.
    Without it the function names the edge from removed_mass, because at step
    1 the removed mass of an arm equals the attention of the one edge it took,
    and the single row of that edge carries the same number. The mass route is
    ambiguous only when two edges of one window tie on attention, so the
    function then accepts a match against any tied candidate.

    weight_matched carries no fixed order, so its step-1 row always takes the
    mass route. A policy whose single arm never ran, such as weight_zero
    against MockPredictor, holds no reference and is skipped.

    Return an empty frame when every arm agrees.
    """
    empty = pd.DataFrame({name: pd.Series(dtype="object") for name in MISMATCH_COLUMNS})
    if len(curves) == 0:
        return empty

    frame = curves.copy()
    frame["window_id"] = frame["window_id"].astype(str)
    frame["arm"] = frame["arm"].astype(str)
    frame["mask_policy"] = frame["mask_policy"].astype(str)

    single = frame.loc[frame["arm"] == "single"]
    shift_of: dict[tuple, float] = {}
    by_mass: dict[tuple, list[tuple[float, int, float]]] = {}
    for row in single.itertuples():
        group = (row.window_id, row.mask_policy, int(row.draw_id))
        shift_of[(*group, int(row.edge_src))] = float(row.shift)
        by_mass.setdefault(group, []).append(
            (float(row.removed_mass), int(row.edge_src), float(row.shift))
        )

    by_window = edges_by_window(edges_primary) if edges_primary is not None else {}

    rows: list[dict] = []
    step_one = frame.loc[(frame["arm"] != "single") & (frame["n_removed"] == 1)]
    for row in step_one.itertuples():
        group = (row.window_id, row.mask_policy, int(row.draw_id))
        if group not in by_mass:
            continue  # this policy never ran the single arm

        expected: int | None = None
        if row.arm in ORDERED_ARMS:
            edges_window = by_window.get(row.window_id)
            if edges_window is not None and len(edges_window):
                expected = int(order_edges(edges_window, row.arm, int(row.draw_id))[0])

        if expected is not None:
            reference = shift_of.get((*group, expected))
            if reference is None:
                rows.append(
                    _mismatch_row(
                        row,
                        expected,
                        float("nan"),
                        "the single arm holds no row for this edge",
                    )
                )
            elif float(row.shift) != reference:
                rows.append(_mismatch_row(row, expected, reference, "the two shifts differ"))
            continue

        candidates = [
            (src, value)
            for mass, src, value in by_mass[group]
            if mass == float(row.removed_mass)
        ]
        if not candidates:
            rows.append(
                _mismatch_row(
                    row, -1, float("nan"), "no single row carries this removed mass"
                )
            )
        elif not any(value == float(row.shift) for _, value in candidates):
            rows.append(
                _mismatch_row(
                    row,
                    int(candidates[0][0]),
                    float(candidates[0][1]),
                    "the two shifts differ",
                )
            )

    if not rows:
        return empty
    return pd.DataFrame(rows, columns=list(MISMATCH_COLUMNS))


def _mismatch_row(row, edge_src: int, reference: float, reason: str) -> dict:
    """Return one row of the mismatch table."""
    return {
        "window_id": row.window_id,
        "arm": row.arm,
        "mask_policy": row.mask_policy,
        "draw_id": int(row.draw_id),
        "edge_src": int(edge_src),
        "arm_shift": float(row.shift),
        "single_shift": reference,
        "difference": float(row.shift) - reference,
        "reason": reason,
    }
