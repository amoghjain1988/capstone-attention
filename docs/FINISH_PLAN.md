# Plan to finish the code

This file is the work specification for the last build. Every agent reads
this file before it writes code. The rules of CONTRACT.md still apply. When
this file and CONTRACT.md disagree, this file wins, because it records the
decisions that the team took after the contract.

Write all text in ASD-STE100 Simplified Technical English. Use the present
tense. Do not use the passive voice in a procedure. Do not use "-ing" forms.

## 1. State of the code on 2026-09-03

Done and tested:

- Stages 1 to 3: download, `src/data/`, `src/eda/`. The tables
  `trajectories` and `windows` exist on disk. The windows table on disk is at
  stride 1 and MIN_EGO_EDGES 1. `scripts/run_all.py` rebuilds it at the
  frozen stride 5 and MIN_EGO_EDGES 2.
- The GPU cache: `data/interim/attention_<scene>.parquet` and
  `data/interim/predict_<scene>.npz` for all 2,841 stride-1 windows. The
  stride-5 windows are a subset with the same window ids and the same ego,
  so the cache stays valid. No new extraction is needed.
- `src/models/`: base, mock, cv, cached, agentformer. `ekf.py` is a stub.
- `src/attention/`, `src/features/`, `src/ablation/`, `src/faithfulness/`,
  `src/metrics/`.
- `src/hypotheses/design.py`, `data_checks.py`, `h3_context.py`.
- `src/stats/effects.py`, `loso.py`, `correction.py`, `report.py`.

Missing:

- `src/stats/clustered.py`, `src/stats/permutation.py`
- `src/hypotheses/choose_test.py`, `h1_morf_vs_lerf.py`,
  `h2_beyond_proximity.py`, `validate.py`, `figures.py`
- `src/attention/edges.py` (the writer of the `attention_edges` table)
- `src/ablation/driver.py` and `scripts/run_ablation.py` (the writer of the
  `perturbation_curves` table)
- `faithfulness_from_curves()` in `src/faithfulness/index.py`
- the `weight_zero` mask policy in `src/models/agentformer.py`
- `src/models/ekf.py`
- stages 4 to 10 in `scripts/run_all.py`
- `notebooks/capstone.ipynb`
- `docs/REPORT.md`

## 2. Facts that drive the design

- AgentFormer is deterministic at inference. `draws` changes nothing. The
  noise floor is exactly 0. A shift above 0 is therefore a real change.
- One masked forward pass takes 26 ms on the RTX 5080. A window with E edges
  into the ego costs about 7E + 1 passes for every arm and both policies.
  The full run over about 480 windows takes about 20 minutes.
- At `PRIMARY_N_REMOVED = 1` every arm removes exactly one edge. The `single`
  arm already holds the shift of every single edge. Therefore every
  sensitivity variant at n_removed 1 (another module, another time collapse,
  a shuffled attention) is a lookup into the `single` rows. It costs no
  forward pass. Only the second mask policy needs new passes, because the
  policy changes the forward pass itself.
- The cluster unit is the connected component of the bipartite graph of ego
  pedestrians against time blocks, inside a scene. See `config.CLUSTER_UNIT`.
- The perturbation_curves table (see `src/data/schema.py`) holds, per
  window: the arms morf, lerf, nearest, random (one row per n_removed step,
  cumulative removal, `edge_src` is -1), weight_matched (one row per
  distinct mass-matched set, `n_removed` is the set size), and single (one
  row per edge, `n_removed` is the position in the sweep, `edge_src` is the
  removed src id). `draw_id` equals the seed. `run_id` is the config hash.

## 3. Shared interfaces

Every agent codes against these signatures. Do not change them.

### 3.1 `src/stats/clustered.py`

```python
def cluster_labels(table: pd.DataFrame, unit: str = config.CLUSTER_UNIT) -> np.ndarray
    """One label per row of `table`. `table` carries window_id and ego_id.
    unit is one of pedestrian, time_block, scene, component.
    scene and t0 come from schema.split_window_id(window_id).
    time block = t0 // config.WINDOW_LEN.
    component = connected component of the bipartite graph
    (scene, ego_id) against (scene, time_block), one edge per row.
    Labels are strings such as "zara1_c12". Use scipy.sparse.csgraph."""

def component_clusters(table: pd.DataFrame) -> np.ndarray
    """cluster_labels(table, "component"). h3_context.py imports this name."""

def n_clusters(labels: np.ndarray) -> int

def cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray, stat=np.median,
    n_boot: int = config.N_BOOT, level: float = 0.95, seed: int = config.SEED,
) -> tuple[float, float]
    """Pairs cluster bootstrap. Resample clusters with replacement, pool the
    rows of the drawn clusters, apply stat, take the percentile interval."""

def hodges_lehmann(d: np.ndarray) -> float
    """The pseudomedian: the median of the Walsh averages (d_i + d_j) / 2
    over i <= j. This is what the Wilcoxon signed-rank test estimates."""

def wild_cluster_bootstrap(
    y: np.ndarray, X: np.ndarray, clusters: np.ndarray,
    n_boot: int = config.N_BOOT, seed: int = config.SEED,
    restriction: np.ndarray | None = None,
) -> dict
    """Rademacher weights on the cluster residuals of an OLS fit.
    Column 0 of X is the intercept. `restriction` is a (q, p) matrix R with
    H0: R beta = 0. The default R selects every column except column 0.
    Impose H0 on the fit (restricted residuals), then resample.
    Returns {beta, se_cluster, wald_obs, joint_p, p_values (one per
    coefficient, each from its own single-row restriction), n_clusters,
    n_boot}. Vectorise over n_boot with numpy. Do not loop in Python over
    n_boot times p."""
```

### 3.2 `src/stats/permutation.py`

```python
def sign_flip_test(
    d: np.ndarray, clusters: np.ndarray, alternative: str = "greater",
    n_perm: int = config.N_PERM, seed: int = config.SEED,
    stat: str = "signed_rank",
) -> dict
    """Flip the sign of every d inside a cluster together.
    stat is one of signed_rank, sign, mean, median.
    signed_rank: rank |d| over ALL rows (Pratt: a zero keeps its rank and
    contributes 0), statistic = sum(rank * sign(d)).
    sign: statistic = count(d > 0) - count(d < 0).
    p = (1 + count(stat_perm >= stat_obs)) / (1 + n_perm) for greater.
    less mirrors it. two-sided uses |stat| around 0.
    Returns {stat, stat_obs, p, n_perm, n_clusters, alternative}.
    Vectorise: draw an (n_perm, G) sign matrix once."""

def sign_flip_p(d, clusters, alternative="greater", n_perm=config.N_PERM,
                seed=config.SEED, stat="signed_rank") -> float
    """sign_flip_test(...)["p"]. The contract signature."""

def attention_shuffle_p(
    edges: pd.DataFrame, single: pd.DataFrame, n_shuffle: int = config.N_PERM,
    seed: int = config.SEED, alternative: str = "greater",
) -> dict
    """Shuffle the attention weight inside the scene and re-rank.
    `edges` holds the primary edges (window_id, src, attn), one row per edge
    into the ego. `single` holds (window_id, edge_src, shift) from the
    single arm under the primary policy.
    Observed statistic: the median over windows of
    D = shift(edge with the largest attn) - shift(edge with the smallest attn).
    Null: permute the attn column inside each scene across every edge row.
    Each window keeps its edge count. Recompute D per window. No forward
    pass is needed, because every D is a lookup into `single`.
    Returns {stat_obs, p, n_shuffle, n_windows}."""
```

### 3.3 `src/hypotheses/choose_test.py`

```python
def choose(d: np.ndarray) -> dict
    """Returns {test, why, sided, shape_stats}.
    test is wilcoxon_signed_rank when D is symmetric, else sign_test.
    Symmetric means: the plain bootstrap 95 percent interval of the sample
    skewness of the non-zero D contains 0, or |skew| < config.SYMMETRY_SKEW_TOL.
    Use config.N_BOOT resamples and config.SEED.
    shape_stats holds n, n_zero, share_positive, mean, median, pseudomedian,
    skew, skew_ci_low, skew_ci_high, kurtosis, shapiro_w, shapiro_p,
    symmetric. Shapiro-Wilk runs on at most 5000 rows.
    why is ONE sentence, present tense, and names the rule that fired.
    sided is config.SIDED."""
```

`config.py` holds, under Statistics, with a comment:

```python
SYMMETRY_SKEW_TOL = 0.5
```

### 3.4 The result dict of one test

Every hypothesis module returns dicts that `src/stats/report.py::build`
accepts. One dict holds exactly these keys:

```
hypothesis, role, test_used, why, sided, effect, effect_name,
ci_low, ci_high, p_raw, n_clusters
```

plus any extra keys that build() ignores. `role` is primary or supporting.
`n_clusters` is the component count of the rows that entered the test.
`hypothesis` strings: `h1`, `h2`, `h3` for the primary rows. Supporting rows
use `h1_morf_vs_random`, `h1_morf_vs_weight_matched`, `h2_disagree_only`,
`h2_spearman_attn_dist`, `h2_partial_attn_shift_given_dist`,
`h2_attention_shuffle`.

The effect of a Wilcoxon test is the rank-biserial correlation (Pratt) and
the point estimate for the interval is the Hodges-Lehmann pseudomedian of D in
metres. The effect of a sign test is the share of positive D among the
non-zero D, and the interval is the cluster bootstrap of the median of D.
State the effect_name plainly: `rank_biserial` or `share_positive`. Put the
pseudomedian or median in metres in an extra key `location_m`, with
`location_ci_low`, `location_ci_high`. `ci_low` and `ci_high` are the
interval of `effect` from the cluster bootstrap. Also add `n_windows`.

### 3.5 `src/hypotheses/h1_morf_vs_lerf.py`

```python
def paired_shift(curves: pd.DataFrame, arm_a: str, arm_b: str,
                 n_removed: int = config.PRIMARY_N_REMOVED,
                 mask_policy: str = config.PRIMARY_MASK_POLICY) -> pd.DataFrame
    """One row per window: window_id, ego_id, shift_a, shift_b, d.
    For weight_matched the partner of morf at step n is the weight_matched
    row with the smallest removed_mass that is >= the cumulative morf mass
    at step n. Raise when a window lacks either arm."""

def test_paired(d_table: pd.DataFrame, hypothesis: str, role: str,
                cfg=config) -> dict
    """choose() the test, run sign_flip_test on the component clusters,
    take the effect and the interval per section 3.4, return the dict."""

def run(curves: pd.DataFrame, cfg=config) -> dict
    """Returns {"primary": dict, "supporting": [dict, dict],
    "d": pd.DataFrame (the H1 D table), "choice": choose() output}.
    primary: morf against lerf. supporting: morf against random, morf
    against weight_matched."""
```

### 3.6 `src/hypotheses/h2_beyond_proximity.py`

```python
def run(curves: pd.DataFrame, edges: pd.DataFrame, cfg=config) -> dict
    """primary: morf against nearest, all windows (a structural zero where
    the two arms pick the same edge stays in and follows the Pratt rule).
    supporting:
      h2_disagree_only: the same test on the windows where the top morf edge
        and the top nearest edge differ.
      h2_spearman_attn_dist: Spearman rho of attn against dist_at_last_frame
        over the primary edges, interval from the cluster bootstrap.
      h2_partial_attn_shift_given_dist: partial correlation of attn with the
        single-edge shift, given dist_at_last_frame, over the edges that
        carry a single row.
      h2_attention_shuffle: permutation.attention_shuffle_p.
    Returns {"primary", "supporting", "d", "choice", "share_agree"}."""
```

### 3.7 `src/hypotheses/validate.py`

```python
def run(curves, edges, cfg=config) -> pd.DataFrame
    """Columns: check, variant, hypothesis, n_windows, n_clusters,
    effect_name, effect, location_m, ci_low, ci_high, p_raw.
    Rows:
      check=loso, variant=<scene>: H1 and H2 on that scene alone.
      check=heterogeneity, variant=h1 or h2: i_squared and range of the
        per-scene location_m, from src/stats/loso.py.
      check=n_removed, variant=1..5: H1 on the windows that hold that step.
      check=module, variant=decoder or cross: H1 and H2 at n_removed 1 with
        the morf and lerf edge chosen under that module, shift from the
        single rows.
      check=time_agg, variant=max or last: the same for the time collapse.
      check=mask_policy, variant=weight_zero: H1 and H2 from the single rows
        under weight_zero.
      check=noise_floor, variant=morf_vs_random: the share of windows where
        the morf shift exceeds the random shift, and the median morf shift.
    No p here enters a family. The table is descriptive."""
```

### 3.8 `src/hypotheses/figures.py`

One function per figure. Each returns the path. Use the house style of
`src/eda/plotting.py`. Write to `config.FIGURE_DIR`.

- `curves_figure(curves)`: mean shift against n_removed per arm, pooled,
  with a cluster bootstrap band, plus one small panel per scene.
- `paired_figure(d_h1, d_h2)`: the histogram of D for H1 and H2, the
  median line, the zero line, the count of zeros in the title.
- `fi_figure(faithfulness)`: the faithfulness index per scene.
- `h3_figure(h3_result)`: the coefficient plot with the cluster-robust
  interval, headline fit.
- `loso_figure(validate_table)`: a forest plot of location_m per scene for
  H1 and H2.

### 3.9 `src/attention/edges.py`

```python
def build_table(cache_edges: pd.DataFrame, windows: pd.DataFrame,
                trajectories: pd.DataFrame) -> pd.DataFrame
    """The attention_edges table. Keep only the eligible windows of
    `windows`. Join the cache rows (window_id, src, dst, module, time_agg,
    attn, row_entropy, rank_attn) with proximity.pair_distance_all and
    kinematics.pair_kinematics_all on (window_id, src, dst). Raise when a
    join leaves a null. Return the columns of schema.ATTENTION_EDGES."""

def primary(edges: pd.DataFrame, module=config.PRIMARY_MODULE,
            time_agg=config.PRIMARY_TIME_AGG) -> pd.DataFrame
    """The slice of one module and one time collapse."""

def edges_of_window(edges_primary: pd.DataFrame, window_id: str) -> pd.DataFrame
    """The rows of one window, for order_edges and curve."""
```

### 3.10 `src/ablation/driver.py` and `scripts/run_ablation.py`

```python
def window_curves(window_row, model, edges_window, seed=config.SEED,
                  policies=config.MASK_POLICIES) -> pd.DataFrame
    """Every arm of one window under the primary policy, plus the single
    arm under every other policy in `policies` that the model accepts.
    Concatenate the rows of src.ablation.curves.curve."""

def run_scene(scene, windows, edges_primary, model, checkpoint_path,
              limit=None, force=False) -> pd.DataFrame
    """Idempotent. Append to checkpoint_path (a parquet) after every 20
    windows. Skip the windows that the file already holds."""

def consistency_check(curves) -> pd.DataFrame
    """For every window and every arm at n_removed 1, the shift must equal
    the single row of the removed edge, exactly. The removed edge of morf,
    lerf, nearest and random at step 1 comes from order_edges on the
    window's edges. Return the mismatches."""
```

`scripts/run_ablation.py` loads the frozen windows table and the
`attention_edges` table, runs every scene with `AgentFormerPredictor`, runs
`zero_check` and `noise_floor` on the first window of every scene, then
writes `data/processed/perturbation_curves.parquet` with `schema.write` and
`outputs/tables/ablation_sanity.csv`. Flags: `--scene`, `--limit`, `--force`,
`--model mock` (for a CPU smoke run with `MockPredictor`).

### 3.11 `src/faithfulness/index.py`

```python
def faithfulness_from_curves(curves: pd.DataFrame, edges_primary: pd.DataFrame) -> pd.DataFrame
    """The faithfulness table from the single rows under the primary policy.
    morf_edge is the src with rank_attn 1 in edges_primary. No forward pass."""
```

### 3.12 `src/models/agentformer.py`, the `weight_zero` policy

Add the second policy. `logit_neg_inf` stays as it is. `weight_zero` keeps
the additive mask at 0 and multiplies the attention weights AFTER the
softmax by the tiled keep matrix, with no renormalisation. Implement it by a
patch of the module attribute `softmax` in
`third_party/AgentFormer/model/agentformer_lib.py` (it imports `softmax`
from `torch.nn.functional` at module level) during the `predict` call only.
The wrapper tiles the (N, N) keep matrix by `(rows // N, cols // N)`. Undo
the patch in a `finally` block. Add a GPU test that skips without CUDA: the
two policies give two different shifts on the same mask, and both give a
shift of exactly 0 under an all-True mask.

### 3.13 `src/models/ekf.py`

A constant-velocity Kalman filter on x, y, vx, vy with the position as the
measurement. Fit on the 8 observed frames, then predict 12 frames. K is 1.
Same protocol as `ConstantVelocity`. Read q and r from config
(`EKF_PROCESS_NOISE`, `EKF_MEASUREMENT_NOISE`).

## 4. Order of work

Round 1, in parallel, four agents:

- A: 3.1, 3.2, and tests `tests/test_clustered.py`, `tests/test_permutation.py`.
- B: 3.3, and tests in `tests/test_choose_test.py`.
- C: 3.9, 3.10, 3.11, and tests against `MockPredictor`.
- D: 3.12, 3.13, and tests.

Round 2, in parallel, two agents:

- E: 3.5, 3.6, and tests against curves from `MockPredictor`.
- F: 3.7, 3.8, and tests.

Round 3, the top level: `scripts/run_all.py` stages 4 to 10, the GPU run,
the notebook, the report skeleton, the diagram.

## 5. Definition of done for one agent

1. The code imports and every new test passes.
2. `python -m pytest tests/ -q` passes in full.
3. Every docstring is in the present tense with no "-ing" form.
4. The agent reports the files it wrote and the test count.
