# 1 Interface contract

DAMO 699 capstone. AgentFormer attention faithfulness.

This file is a agreed upon contract as per team meetings on Aug 9, 10, 15 and 16th and describes
the agreed upon input and output of every function. The goal is to build towards a  common interface
and then upon merge we do one more cleanup to ensure all kinks are removed.

This contract defines all the major interface functions and a comment on the parameters and well tables, 
orders or work, splitting of tasks, alongside definition of done.


All 4 of us have agreed upon this and we will start commiting as per this document and update as required.

Note : This document stays live throughout the develop cycle!


---


## 2. Conventions

| Item | Rule |
|---|---|
| Position | metres, world coordinates, `float64` |
| Frame rate | 2.5 Hz. One frame is 0.4 s |
| History `T_obs` | 8 frames, 3.2 s |
| Future `T_pred` | 12 frames, 4.8 s |
| Angles | radians |
| Speed | metres per second |
| Missing value | `NaN`. Never `0`, never `-1` |
| Array axes | always `(agent, time, coordinate)` |
| Randomness | every sampling function takes an explicit `seed: int` |
| Tables | `data/processed/<name>.parquet`, `index=False` |

Symbols used below:

- `N` = agents in a window
- `T` = 8 for history, 12 for future
- `K` = number of latent draws, `config.N_SAMPLES`, fixed at 20
- `E` = edges that point into the ego in one window

`window_id` is the string `f"{scene}_{t0:06d}"`. It is the join key everywhere.

Nothing under `data/`, `checkpoints/` or `outputs/` is committed.

## 3. The seven tables

Each table has one producer. Everybody else reads it and must not write it.

### `trajectories` — key `scene, frame, agent_id`

| Column | Type | Note |
|---|---|---|
| `scene` | category | eth, hotel, univ, zara1, zara2 |
| `frame` | int32 | original frame index |
| `agent_id` | int32 | unique inside a scene |
| `x`, `y` | float64 | metres |
| `vx`, `vy` | float64 | first difference divided by `DT` |
| `speed` | float64 | norm of the velocity |
| `heading` | float64 | `NaN` when speed is 0 |
| `nearest_dist` | float64 | metres to the closest other agent in that frame |
| `density` | float64 | agents inside `DENSITY_RADIUS_M` |

### `windows` — key `window_id`

| Column | Type | Note |
|---|---|---|
| `window_id` | string | |
| `scene` | category | |
| `t0` | int32 | first observed frame |
| `ego_id` | int32 | the one ego of this window |
| `agent_order` | list[int32] | length `N`. Defines the axis order of `hist` and `fut` |
| `n_agents` | int16 | `N` |
| `hist` | list[float64] | flattened `(N, 8, 2)` |
| `fut` | list[float64] | flattened `(N, 12, 2)` |
| `overlap_fraction` | float64 | share of frames shared with any other window |
| `eligible` | bool | `False` rows stay in the table and are skipped downstream |

Index 0 of `agent_order` is not the ego. Find the ego with
`agent_order.index(ego_id)`.

### `attention_edges` — key `window_id, src, dst, module, time_agg`

| Column | Type | Note |
|---|---|---|
| `window_id` | string | |
| `src`, `dst` | int32 | agent ids |
| `module` | category | encoder, decoder, cross |
| `time_agg` | category | mean, max, last |
| `attn` | float64 | in `[0, 1]`, the collapsed N by N value |
| `row_entropy` | float64 | entropy of the source row before collapsing |
| `dist_at_last_frame` | float64 | metres, at frame `t0 + 7` |
| `closing_speed` | float64 | positive means approaching |
| `inv_ttc` | float64 | `0` when the pair never closes |
| `rank_attn` | int16 | 1 is the strongest, among edges into the ego |
| `rank_dist` | int16 | 1 is the closest |

### `perturbation_curves` — key `window_id, arm, n_removed, draw_id, mask_policy`

| Column | Type | Note |
|---|---|---|
| `window_id` | string | |
| `ego_id` | int32 | the cluster key |
| `arm` | category | morf, lerf, weight_matched, nearest, random, single |
| `n_removed` | int16 | for `weight_matched` the mass is matched, not `n` |
| `removed_mass` | float64 | total attention removed. Compare arms on this |
| `edge_src` | int32 | filled only when `arm == "single"`, else `-1` |
| `draw_id` | int16 | index into the common random numbers |
| `mask_policy` | category | logit_neg_inf, weight_zero |
| `run_id` | string | git short hash plus the config hash |
| `shift` | float64 | metres. Section 5.7 |

The `single` arm is the brute force sweep. It removes one named edge at a time and is what
makes the faithfulness index bounded. See section 5.9.

### `faithfulness` — key `window_id`

| Column | Type | Note |
|---|---|---|
| `window_id` | string | |
| `ego_id` | int32 | |
| `n_edges` | int16 | `E`, how many single edges were tried |
| `floor` | float64 | mean shift over all single edges |
| `ceiling` | float64 | largest shift over all single edges |
| `fi` | float64 | section 5.9. `NaN` when `ceiling - floor` is below tolerance |

### `predictions` — key `window_id`

| Column | Type | Note |
|---|---|---|
| `window_id` | string | |
| `ade`, `fde` | float64 | metres, mean over the K draws |
| `minade_20`, `minfde_20` | float64 | metres, best of K draws |
| `collision` | bool | any pair closer than `COLLISION_RADIUS_M` |

Coverage and the PIT histogram are per scene. They go to
`outputs/tables/calibration.csv`.

### `verdicts` — key `hypothesis`

| Column | Type |
|---|---|
| `hypothesis` | string, h1 / h2 / h3 |
| `role` | category, primary / supporting |
| `test_used` | string |
| `why` | string, one sentence from `choose_test.py` |
| `sided` | category, one / two |
| `effect`, `effect_name` | float64, string |
| `ci_low`, `ci_high` | float64 |
| `p_raw`, `p_adj` | float64 |
| `n_clusters` | int32 |
| `verdict` | string, reject / do not reject |

## 4. The model seam

### 4.1 `src/models/base.py`

Nobody imports AgentFormer outside `src/models/agentformer.py`.

```python
from typing import Protocol
import numpy as np

class Predictor(Protocol):
    def predict(
        self,
        hist: np.ndarray,                     # (N, 8, 2) float64, metres
        edge_mask: np.ndarray | None = None,  # (N, N) bool. True keeps the edge
        draws: np.ndarray | None = None,      # (K,) int64 seeds. None draws fresh
    ) -> np.ndarray:                          # (K, N, 12, 2) float64, metres
        ...

class AttentionPredictor(Predictor, Protocol):
    def attention(
        self,
        hist: np.ndarray,                     # (N, 8, 2)
    ) -> dict[str, np.ndarray]:               # module name -> (N*8, N*8) float64
        """Rows sum to 1. Token order is agent-major:
        index = agent_index * 8 + time_index."""
        ...

class MaskablePredictor(AttentionPredictor, Protocol):
    """Honours edge_mask. cached.py does NOT satisfy this protocol."""
```

`edge_mask` is `(N, N)` because we rank on the collapsed map. `mask.py` expands entry
`(i, j)` to the whole 8 by 8 block in the `(N*8, N*8)` attention. A pair is a block, not
a cell.

### 4.2 Implementations

| Class | File | Ready | GPU |
|---|---|---|---|
| `MockPredictor` | `src/models/mock.py` | day one | no |
| `ConstantVelocity` | `src/models/cv.py` | day one | no |
| `ExtendedKalman` | `src/models/ekf.py` | week 1 | no |
| `CachedPredictor` | `src/models/cached.py` | after one extraction run | no |
| `AgentFormerPredictor` | `src/models/agentformer.py` | the swap point | yes |

`CachedPredictor.predict` raises `NotImplementedError` when `edge_mask` is not `None`.
Do not build the ablation loop on it.

`AgentFormerPredictor` is a conditional variational autoencoder with DLow sampling. Two
identical calls give different paths. Always pass `draws`.

### 4.3 Which checkpoint

AgentFormer released five checkpoints, one per held-out scene. Score a scene only with
the checkpoint that held that scene out. `config.SCENE_TO_CHECKPOINT` holds the map.

## 5. Module by module

### 5.1 `scripts/`

```python
# download_ethucy.py
def main() -> None
# setup_agentformer.py
def main() -> None          # writes checkpoints/manifest.json: sha256, url, commit
# extract_attention.py
def main(scene: str | None = None) -> None
# run_all.py
def main() -> None          # calls every stage in the order of section 7
```

Every script is idempotent. Running it twice changes nothing.

### 5.2 `src/data/`

```python
# loader_ethucy.py    writes: trajectories (raw columns only)
def load_scene(scene: str, raw_dir: Path) -> pd.DataFrame
def load_all(raw_dir: Path) -> pd.DataFrame

# clean.py            reads: trajectories   writes: trajectories, drops.csv
def drop_short_tracks(df: pd.DataFrame, min_frames: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]
    """Returns (kept, dropped). `dropped` carries a `reason` column."""
def add_kinematics(df: pd.DataFrame, dt: float) -> pd.DataFrame
    """Adds vx, vy, speed, heading, nearest_dist, density."""

# windows.py          reads: trajectories   writes: windows
def make_windows(df: pd.DataFrame, n_hist: int = 8, n_fut: int = 12) -> pd.DataFrame
def overlap_fraction(windows: pd.DataFrame) -> pd.Series
    """Share of the 20 frames also covered by another window of the same scene."""

# eligibility.py      reads: windows        writes: windows, drops.csv
def pick_ego(window_row: pd.Series, seed: int) -> int
    """Returns the ego agent_id. The ego must be present for all 20 frames and have
    at least one neighbour. Deterministic given the seed."""
def mark_eligible(windows: pd.DataFrame, seed: int) -> pd.DataFrame
    """Adds ego_id and eligible. Never deletes a row."""

# synthetic.py        writes: a synthetic scene plus its true edge list
def make_scene(n_agents: int, n_frames: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]
    """Returns (trajectories, true_edges). true_edges has window_id, src, dst, is_real."""
```

### 5.3 `src/features/`

All three read `trajectories` and `windows`. All three return one row per pair per window,
keyed `window_id, src, dst`, except `context.py` which returns one row per window.

```python
# proximity.py
def pair_distance(window_row: pd.Series, trajectories: pd.DataFrame) -> pd.DataFrame
    """Columns: window_id, src, dst, dist_at_last_frame, rank_dist."""

# kinematics.py
def pair_kinematics(window_row: pd.Series, trajectories: pd.DataFrame) -> pd.DataFrame
    """Columns: window_id, src, dst, closing_speed, inv_ttc.
    closing_speed is positive when the pair approaches.
    inv_ttc is 0 when they never close."""

# context.py
def window_context(window_row: pd.Series, trajectories: pd.DataFrame) -> dict
    """Keys: window_id, density, n_agents. Density uses DENSITY_RADIUS_M
    around the ego at the last observed frame."""
```

### 5.4 `src/eda/`

Every function returns a `pd.DataFrame` and writes one png to `outputs/figures/`.
No EDA function writes a processed table.

```python
def summary(trajectories, windows) -> pd.DataFrame        # counts per scene
def density_report(windows, trajectories) -> pd.DataFrame
def distributions(trajectories) -> pd.DataFrame           # step, speed, spacing
def compare_scales(values: np.ndarray) -> pd.DataFrame    # raw, z, log side by side
def missing_report(trajectories) -> pd.DataFrame
def design_effect(windows: pd.DataFrame) -> dict
    """Returns {icc, mean_cluster_size, deff, n_effective}.
    deff = 1 + (mean_cluster_size - 1) * icc
    n_effective = n_rows / deff
    design.py cannot compute power without this."""
```

### 5.5 `src/models/`

Every class implements the protocol of section 4.1.

```python
class MockPredictor:
    def __init__(self, seed: int = 0, n_samples: int = 20)
class ConstantVelocity:
    def __init__(self, dt: float)
class ExtendedKalman:
    def __init__(self, dt: float, q: float, r: float)
class CachedPredictor:
    def __init__(self, parquet_dir: Path)
class AgentFormerPredictor:
    def __init__(self, checkpoint: Path, device: str = "cuda", n_samples: int = 20)
```

`predict` always returns `(K, N, 12, 2)`, even for the deterministic models. There `K`
is 1 and every draw is identical.

### 5.6 `src/attention/`

```python
# extract.py
def attention_maps(model: AttentionPredictor, hist: np.ndarray) -> dict[str, np.ndarray]
    """module -> (N*8, N*8). Pass through, no collapsing."""

# aggregate.py
def collapse(raw: np.ndarray, n_agents: int, time_agg: str = "mean") -> np.ndarray
    """(N*8, N*8) -> (N, N). Collapse TIME first, then heads, then layers.
    time_agg is one of mean, max, last."""
def row_entropy(raw: np.ndarray) -> np.ndarray
    """(N*8, N*8) -> (N*8,). Shannon entropy of every source row, natural log."""

# rank.py
def rank_edges(attn: np.ndarray, ego_index: int, agent_order: list[int]) -> pd.DataFrame
    """Returns only the edges whose dst is the ego.
    Columns: src, dst, attn, rank_attn. Rank 1 is the strongest.
    Ties break by the lower src id, so the ranking is deterministic."""
```

### 5.7 `src/ablation/`

```python
# arms.py
ARMS = ("morf", "lerf", "weight_matched", "nearest", "random", "single")

def order_edges(edges: pd.DataFrame, arm: str, seed: int) -> list[int]
    """Returns the src ids in removal order. `single` is not ordered:
    the caller sweeps every edge on its own."""

def weight_matched_set(edges: pd.DataFrame, target_mass: float) -> list[int]
    """The smallest set of the LOWEST attention edges whose total attention
    reaches target_mass. Matches mass, not count."""

# mask.py
def build_mask(n_agents: int, remove: list[int], ego_index: int) -> np.ndarray
    """(N, N) bool. True keeps. Only entries whose dst is the ego are ever set False."""

def masked_predict(
    model: MaskablePredictor,
    hist: np.ndarray,
    edge_mask: np.ndarray,
    draws: np.ndarray,
    mask_policy: str,
) -> np.ndarray:
    """(K, N, 12, 2). mask_policy is logit_neg_inf or weight_zero.
    The same `draws` must be used for every arm of the same window."""

# shift.py
def shift(path_masked: np.ndarray, path_unmasked: np.ndarray) -> float:
    """Both are (K, 12, 2) for the EGO ONLY, over the SAME draws.
    Returns the mean over the 12 steps of the euclidean distance,
    then the mean over the K draws. Metres.
    The reference is the unmasked prediction, never the ground truth."""

# curves.py
def curve(window_row, model, edges, arm, seed) -> pd.DataFrame
    """One row per n_removed. Columns match perturbation_curves."""

# arm_overlap.py
def jaccard(order_a: list[int], order_b: list[int], k: int) -> float
    """Overlap of the top k of two arms. H2 is only meaningful on the part
    where MoRF and Nearest disagree."""
```

### 5.8 `src/metrics/`

```python
def ade(pred: np.ndarray, truth: np.ndarray) -> float      # pred (K,12,2), truth (12,2)
def fde(pred: np.ndarray, truth: np.ndarray) -> float
def min_ade(pred: np.ndarray, truth: np.ndarray) -> float  # best of K
def has_collision(paths: np.ndarray, radius: float) -> bool  # paths (N,12,2)
def coverage(pred: np.ndarray, truth: np.ndarray, level: float = 0.90) -> float
    """Share of the 12 steps where the truth falls inside the level region of the
    K draws. Report per scene."""
def pit(pred: np.ndarray, truth: np.ndarray) -> np.ndarray
    """Probability integral transform values for the histogram."""
```

Expected calibration error is a classifier metric. We forecast paths, so we use coverage
and PIT instead. Do not report ECE.

### 5.9 `src/faithfulness/`

```python
# index.py
def brute_force_single(window_row, model, edges, draws) -> pd.DataFrame
    """Removes each edge on its own. One row per edge. Column `shift`.
    Cost is E forward passes per window, and E is small."""

def faithfulness_index(single: pd.DataFrame, morf_edge: int, tol: float = 1e-6) -> float:
    """floor    = single.shift.mean()          expected shift of a random single edge
       ceiling  = single.shift.max()           the truly strongest edge
       FI       = (shift(morf_edge) - floor) / (ceiling - floor)

       FI = 1  attention picked the strongest edge
       FI = 0  attention is no better than chance
       FI < 0  attention is worse than chance
       Returns NaN when ceiling - floor <= tol. Those windows are dropped and counted."""

# validation.py
def auroc_against_truth(scores: np.ndarray, is_real: np.ndarray) -> float
    """THE GATE. Runs on the synthetic scene. Must exceed 0.9 before any real
    number is reported."""

# sanity.py
def zero_check(model, hist, draws) -> float
    """Fixed draws, mask nothing. Must return exactly 0.0."""
def noise_floor(model, hist, n_repeat: int, seed: int) -> float
    """Free draws, mask nothing. The shift produced by sampling alone.
    Every arm must beat this number or the study reports nothing."""
```

### 5.10 `src/hypotheses/`

```python
# design.py
def frozen_design(deff: float, n_rows: int) -> dict
    """Returns alpha, sided, primary_n_removed, minimum_effect, power, n_effective.
    Needs the design effect from src/eda/overlap.py. Run it AFTER the EDA and
    BEFORE the first ablation. Hash config.py and commit the hash."""

# choose_test.py
def choose(d: np.ndarray) -> dict
    """Input: the paired differences D_i.
    Returns {test, why, sided, shape_stats}.
    Rule:
      paired            -> yes, always. We compare two arms on the same window.
      shape of D        -> histogram, QQ, skew, Shapiro-Wilk. Report it, never assume it.
      symmetric         -> Wilcoxon signed-rank, one-sided.
                           It tests the PSEUDOMEDIAN. That equals the median only
                           when D is symmetric. Say which one you are claiming.
      skewed or heavy   -> sign test on the median, or bootstrap the median of D.
                           Never a plain t-test.
      independent rows  -> NO. Windows overlap, pedestrians repeat, 5 scenes.
                           p comes from the sign flip permutation inside the cluster.
                           CI comes from the cluster bootstrap.
    Rejected on purpose:
      paired t-test  D is long-tailed, the mean is not the target
      plain bootstrap  assumes independent rows
      OLS or ANOVA on raw rows  the same fault
      Bonferroni  Holm dominates it at no cost
      chi-square  the outcome is continuous, not a count"""

# data_checks.py
def check(df: pd.DataFrame) -> pd.DataFrame
    """Every window carries every arm. No missing pair. Zero differences kept and
    handled by the Pratt rule. Returns the attrition table and raises if a
    primary window is incomplete."""

# h1_morf_vs_lerf.py
def run(curves: pd.DataFrame, cfg) -> dict
    """D_i = shift(MoRF) - shift(LeRF) at PRIMARY_N_REMOVED, per window.
    H0-1: pseudomedian(D) <= 0.       H1-1: pseudomedian(D) > 0.
    Also reports MoRF against Random and MoRF against Weight-matched.
    Those two SUPPORT the claim. They are not in the primary family."""

# h2_beyond_proximity.py
def run(curves: pd.DataFrame, edges: pd.DataFrame, cfg) -> dict
    """D_i = shift(MoRF) - shift(Nearest).
    H0-2: removing high-attention edges does not give a larger shift than removing
    the nearest-neighbour edges.
    The arm comparison is the primary test.
    Spearman rho(rank_attn, rank_dist) and the partial correlation of attn with
    shift given distance SUPPORT it."""

# h3_context.py
def run(fi: pd.DataFrame, context: pd.DataFrame, cfg) -> dict
    """FI_i = b0 + b . context + scene fixed effects.
    context = density, n_agents, closing_speed, inv_ttc, all z-scored.
    H0-3: every b = 0 AND every scene term = 0. One joint test.
    Five scenes is too few for a random intercept, so the scenes are fixed effects.
    Check collinearity first. Cluster-robust errors by pedestrian.
    Report partial R squared."""

# validate.py
def run(...) -> pd.DataFrame
    """LOSO, the n_removed sweep, another module, another time collapse, both mask
    rules. LOSO shows heterogeneity. It is NOT five more tests.
    If MoRF does not beat Random above the noise floor, the answer is
    'attention is not faithful'. That is a result, not a failure."""
```

### 5.11 `src/stats/`

```python
# clustered.py
def cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray, stat=np.median,
    n_boot: int = 10_000, level: float = 0.95, seed: int = 0,
) -> tuple[float, float]:
    """Resamples PEDESTRIANS with replacement, not rows. This is the pairs
    cluster bootstrap. Use it for H1 and H2."""

def wild_cluster_bootstrap(
    y: np.ndarray, X: np.ndarray, clusters: np.ndarray,
    n_boot: int = 10_000, seed: int = 0,
) -> dict:
    """Rademacher weights on the cluster residuals of a regression.
    Use it for H3 only. It is not the same tool as the one above."""

# permutation.py
def sign_flip_p(
    d: np.ndarray, clusters: np.ndarray, alternative: str = "greater",
    n_perm: int = 10_000, seed: int = 0,
) -> float:
    """Flips the sign of every D_i inside a cluster together. Exact for a paired
    design and it needs no GPU. This is where the primary p value comes from."""

def attention_shuffle_p(...) -> float
    """Shuffles the attention weights inside the scene and re-ranks. It holds the
    graph shape, but it needs a new forward pass, so it is expensive. Secondary."""

# loso.py
def leave_one_scene_out(df, fn) -> pd.DataFrame     # one estimate per held-out scene
def heterogeneity(estimates: pd.Series) -> dict     # I squared and the range

# effects.py
def rank_biserial(d: np.ndarray) -> float           # the effect for Wilcoxon
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float
def spearman(a, b) -> tuple[float, float]
def partial_corr(x, y, covar) -> tuple[float, float]

# correction.py
def holm(pvals: dict[str, float]) -> dict[str, float]
    """Over the three PRIMARY tests only: h1, h2, h3."""
def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, float]
    """Over the exploratory family."""

# report.py
def build(results: list[dict]) -> pd.DataFrame      # writes verdicts
```

## 6. `config.py`

No magic number lives anywhere else.

```python
SEED = 0
N_HIST, N_FUT = 8, 12
DT = 0.4
N_SAMPLES = 20
ALPHA = 0.05
SIDED = "one"
PRIMARY_N_REMOVED = 1
PRIMARY_MODULE = "encoder"
PRIMARY_TIME_AGG = "mean"
PRIMARY_MASK_POLICY = "logit_neg_inf"
COLLISION_RADIUS_M = 0.2
DENSITY_RADIUS_M = 5.0
MIN_TRACK_FRAMES = 20
N_BOOT = 10_000
N_PERM = 10_000
SCENE_TO_CHECKPOINT = {...}
MINIMUM_EFFECT = None   # fill after the EDA
POWER = None            # fill after the EDA
```

`MINIMUM_EFFECT` and `POWER` need the design effect from `src/eda/overlap.py`. Fill them
after the exploratory analysis and before the first ablation run. Then hash `config.py`
and commit the hash. `run_id` records that hash. After that, nobody changes a number.

## 7. Order of work

1. `scripts/download_ethucy.py`
2. `src/data/` up to `eligibility.py`
3. `src/eda/`, including `overlap.py`, which produces the design effect
4. `src/features/`
5. Fill `config.py`, hash it, commit the hash
6. `scripts/extract_attention.py`, then `src/attention/`
7. `src/ablation/` against `MockPredictor`, then against `AgentFormerPredictor`
8. `src/faithfulness/`, `src/metrics/`
9. `src/hypotheses/` and `src/stats/`
10. `notebooks/capstone.ipynb`

Steps 2, 3, 4 and all of `src/stats/` need no GPU and no AgentFormer. They run against
`MockPredictor` from day one. That is the point of the seam.

## 8. Tests that must always pass

| Test | Checks |
|---|---|
| `tests/test_shapes.py` | every predictor returns `(K, N, 12, 2)` |
| `tests/test_ablation.py` | fixed draws and an all-True mask give a shift of exactly 0 |
| `tests/test_synthetic.py` | the planted graph returns AUROC above 0.9 |

`test_synthetic.py` is the gate. If the method cannot recover a graph we planted
ourselves, no number from the real data means anything.

## 9. Definition of done

A module is done when all four are true.

1. It writes its table and the table matches section 3 exactly.
2. Its test passes.
3. It reads every number from `config.py`.
4. It logs every row it drops, with a reason.

## 10. Work split

`README.png` marks the owner on every box. Four people, one branch each.

| Person | Modules | Needs a GPU |
|---|---|---|
| Amogh | `scripts/`, `src/models/`, `src/attention/`, `src/ablation/` | yes |
| P2 | `src/data/`, `src/features/`, `src/data/schema.py` | no |
| P3 | `src/eda/`, `src/metrics/`, `src/faithfulness/` | no |
| P4 | `src/hypotheses/`, `src/stats/`, `tests/` | no |

Amogh owns the seam, so the swap from `MockPredictor` to `AgentFormerPredictor` happens
in one branch and nobody else waits for it. Reassign P2 to P4 at the kickoff and edit the
`[ owner ? ]` labels in `README.mmd` in the same commit.

## 11. Rendering the diagram

```
npx @mermaid-js/mermaid-cli -i README.mmd -o README.png -b white -s 3
```

Commit `README.mmd` and `README.png` together. They must never drift apart.
