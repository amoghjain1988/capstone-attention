# To What Extent Do AgentFormer Attention Weights Give Faithful Explanations of the Interactions That Influence Pedestrian Trajectory Predictions?

DAMO 699 Capstone Project. Final report.

**Authors:** Amogh, Catherine, Grace (git author Oluwatola Ukel) and Prem. The
4 names come from the TEAM box of `README.mmd`.

**Supervisor:** This value is not available. No file in the repository records a
supervisor name.

**Date:** 2026-09-03. The pipeline wrote the last table on that date.

---

## 1. Abstract

This study reads the attention weights of AgentFormer as a graph over the
pedestrians of one window. The study masks an edge into one ego pedestrian,
predicts again, and measures how far the ego forecast moves. The study reports
faithfulness, not causality. Every claim is a claim about the model, not a
claim about how people walk.

The study tests three primary hypotheses on the ETH and UCY benchmark, over 5
scenes and 5 released checkpoints, at inference only.

| Hypothesis | Claim in one line | Verdict |
|---|---|---|
| H1 | The removal of the strongest attention edge moves the forecast more than the removal of the weakest edge. | Reject the null. MoRF beats LeRF in 85.2 percent of 481 windows. Holm p 0.0003 over 72 components. |
| H2 | The removal of the strongest attention edge moves the forecast more than the removal of the nearest neighbour edge. | Reject the null. MoRF beats Nearest in 80.4 percent of the non-zero windows. Holm p 0.0003 over 72 components. |
| H3 | The context of a window explains the faithfulness index. | Do not reject the null. The wild cluster bootstrap joint p is 0.60 and the partial R squared is 0.017 over 62 components. |

The AgentFormer attention gives a faithful rank of the single edges that move
the forecast. The top edge beats both the weakest edge and the nearest
neighbour edge. The window context does not explain the faithfulness index, and
the attention mass does not predict the size of the shift.

---

## 2. Introduction

### 2.1 The question

AgentFormer forecasts the future path of every pedestrian in a scene. The model
makes the interaction graph explicit, because the attention map holds one
weight for every ordered pair of agents. A reader of the model wants to know
whether that weight names the interaction that actually drives the forecast.
This study answers one question. To what extent do the AgentFormer attention
weights give faithful explanations of the interactions that influence the
pedestrian trajectory predictions?

### 2.2 Faithfulness, not causality

A faithful explanation names the inputs that the model itself uses. A causal
claim names the factors that move a real pedestrian. This study measures the
first and never the second. The study masks an edge inside the model and
watches the model forecast move, so every result describes the model. The
study reports an association between the attention weight and the forecast
shift. The study never reports a cause of human motion.

The scope holds 5 ETH and UCY scenes, a frame rate of 2.5 Hz, the public
AgentFormer weights, and inference only. Perception, edge deployment and
orchestration stay out of scope.

### 2.3 The contribution

- A bounded faithfulness index per window. The brute force sweep over every
  single edge gives a floor and a ceiling, so the index of one window sits on a
  known scale between chance and the best possible edge.
- A design that respects the real dependence in the data. The windows overlap,
  the pedestrians repeat, and only 5 scenes exist. The connected component of
  the pedestrian by time-block graph is the cluster unit, and a simulation
  shows that this unit is the only one that holds the false positive rate.
- A pre-committed test plan with a gate. The synthetic scene must return an
  AUROC above 0.9 before any number from the real data is reported, and the
  primary family is exactly 3 tests under the Holm correction.

---

## 3. Data

### 3.1 ETH and UCY

The study uses the ETH and UCY pedestrian benchmark. AgentFormer released its 5
checkpoints on this benchmark, so the data and the model match. The data is
public, anonymised, and already in world coordinates. No perception step and no
camera model enter the study.

### 3.2 The 5 scenes

The 5 scenes are eth, hotel, univ, zara1 and zara2. The scene univ holds 2
source files, students001 and students003. A per-file offset of 100,000 keeps
every frame index and every agent id unique inside a scene, and the offset is
much larger than the window length, so no window spans 2 files.

| Scene | Rows | Share of rows | Pedestrians | Mean agents per window |
|---|---|---|---|---|
| eth | 5,492 | 0.082 | 360 | 2.59 |
| hotel | 6,543 | 0.098 | 389 | 3.50 |
| univ | 39,766 | 0.596 | 849 | 25.70 |
| zara1 | 5,153 | 0.077 | 148 | 3.74 |
| zara2 | 9,722 | 0.146 | 204 | 6.33 |
| Total | 66,676 | 1.000 | 1,950 | 11.85 |

Source: `outputs/tables/eda_summary.csv` and `outputs/tables/eda_density.csv`.
The scene univ holds 60 percent of the rows. Section 9 states the limit that
this imbalance creates.

### 3.3 Coordinates, the frame rate and the units

| Item | Value |
|---|---|
| Position | metres, world coordinates, float64 |
| Frame rate | 2.5 Hz. One frame is 0.4 s |
| Angles | radians |
| Speed | metres per second |
| Missing value | NaN, never 0 and never -1 |
| Array axes | agent, time, coordinate |

The loader reads the AgentFormer text format. The loader takes the frame from
column 0, the agent id from column 1, x from column 13 and y from column 15.

### 3.4 The rules that clean the data

The pipeline adds the kinematics before any drop, so the density and the
nearest distance see every pedestrian that is really there. The kinematics are
vx, vy, speed, heading, nearest_dist and density. The density counts the agents
inside a radius of 5.0 m. The heading is NaN when the speed is 0.

The flag `DROP_SHORT_TRACKS` is False. A track shorter than 20 frames stays in
the table as a neighbour, because such a track is a real pedestrian and it
carries a real attention edge. A short track can never become an ego, because
an ego must be present for all 20 frames. A separate check reports every track
that skips a frame, because the window builder assumes a contiguous track.

The table `outputs/tables/drops.csv` holds every drop. No rule drops a row and
no rule drops a track. One rule drops a window: a window that holds fewer than
2 agents has no ego. That rule drops 45 windows in eth, 54 in hotel, 65 in
zara1 and 34 in zara2, so 198 windows in total. The scene univ loses no window
to that rule.

The table `outputs/tables/eda_missing.csv` gives a known cause for every empty
cell. The columns vx, vy and speed hold 1,950 empty cells, one per track,
because the first frame of a track has no earlier frame. The column heading
holds 3,188 empty cells, because the speed is 0 or the frame is the first frame
of the track. The column nearest_dist holds 190 empty cells, because the agent
is alone in that frame. The column unexplained is 0 in every row.

No track skips a frame. A check over `data/processed/trajectories.parquet`
counts the tracks whose frame index jumps by more than 1. That count is 0 in
eth, 0 in hotel, 0 in univ, 0 in zara1 and 0 in zara2. Every track is
contiguous, so the window builder holds its assumption. The table
`outputs/tables/eda_frame_occupancy.csv` counts the empty frames of the span,
which are frames with no pedestrian at all: 285 in eth, 639 in hotel, 0 in
univ, 30 in zara1 and 0 in zara2. An empty frame is not a gap in a track.

### 3.5 The window build

One window holds 8 observed frames and 12 forecast frames, so 20 frames in
total. The 8 observed frames cover 3.2 s and the 12 forecast frames cover 4.8
s. This 8 in and 12 out protocol is the benchmark protocol that AgentFormer
reports on. The builder steps the window start by a stride of 5 frames. Every
agent of a window is present for all 20 frames. The window key is the string
`scene_t0`, and that key joins every table.

### 3.6 The ego pick

Each window carries exactly one ego pedestrian. One ego gives one row and one
cluster key, so no window contributes more than one observation to a test. The
pick is random but deterministic. The seed of a window comes from the global
seed and from the window key, so the pick does not depend on the row order and
does not change between runs.

### 3.7 The eligibility rule

The number of edges into the ego is E, and E equals the agent count minus 1.
The frozen rule sets `MIN_EGO_EDGES` to 2, not to 1. A window with E of 1 holds
one edge only, so MoRF, LeRF, Nearest and Random all pick the same edge. In
that case H1 gives a difference of exactly 0 by construction, H2 gives a
difference of exactly 0 by construction, and the floor of the faithfulness
index equals the ceiling, so the index is NaN. Such zeros are structural, not
observed ties, so the Pratt rule would rank them and would pull the test toward
the null.

E is the agent count minus 1, and the agent count is known before any model
runs. The rule therefore selects on a pre-treatment covariate and not on an
outcome. The rule drops 423 of the 2,841 eligible stride-1 windows and only 2.5
percent of the compute. All 5 scenes keep 20 windows or more. A stricter value
costs the scene eth: at a floor of 3 the scene eth falls from 32 windows to 7.

The frozen build gives 679 windows at stride 5, and 481 of those windows pass
the eligibility rule.

| Scene | Eligible windows | Mean agents per eligible window |
|---|---|---|
| eth | 8 | 3.13 |
| hotel | 42 | 4.05 |
| univ | 190 | 25.78 |
| zara1 | 74 | 4.70 |
| zara2 | 167 | 6.71 |
| Total | 481 | 13.64 |

Source: `data/processed/windows.parquet`, the eligible column. The scene eth
keeps only 8 windows, so section 9 states the limit that this count creates.

### 3.8 The overlap and the design effect

The proposal counts every window as one independent observation. Every window
is not independent. Two windows that start one frame apart share 19 of their 20
frames, and many windows share the same ego pedestrian. Both facts inflate the
sample and both shrink the true information.

The design effect measures the loss. The formula is
`deff = 1 + (mean_cluster_size - 1) * icc`, and the effective sample is
`n_effective = n_rows / deff`. The intraclass correlation needs an outcome, and
the real outcome is the shift, which does not exist before the ablation runs.
The exploratory module therefore reports the intraclass correlation of every
proxy that exists before the ablation and takes the largest one, because the
largest one gives the safest answer.

| Stride | Windows | Cluster unit | Design effect | Effective n | Forward passes |
|---|---|---|---|---|---|
| 1 | 2,841 | time block | 14.57 | 195 | 33,654 |
| 5 | 571 | time block | 3.18 | 179 | 6,741 |
| 5 | 571 | pedestrian | 1.55 | 369 | 6,741 |
| 20 | 149 | time block | 1.00 | 149 | 1,728 |

Source: `outputs/tables/eda_overlap.csv`. The mean overlap fraction stays above
0.99 at every stride, so the overlap never disappears.

### 3.9 The frozen design

The team froze the design after the exploratory analysis and before the first
ablation run. Nobody changes a number after that commit.

| Design item | Frozen value |
|---|---|
| Window stride | 5 frames |
| Windows at that stride | 571 |
| Headline cluster unit for the power calculation | time block |
| Design effect | 3.18 |
| Effective n | 179 |
| Alpha | 0.05 |
| Sided | one |
| Primary n_removed | 1 |
| Minimum detectable effect | 0.592 |
| Power | 0.80 |

The minimum effect of 0.592 is the share of windows in which MoRF gives the
larger shift. A share of 0.5 is the null. The number is pre-committed and the
team stated the number before it saw any p value. At the same effective n a
power of 0.90 needs a share of 0.608.

### 3.10 The cluster unit

The pedestrian and the time block both hold real dependence, and the two units
cross. 37 percent of the pedestrians appear in more than 1 time block, so
neither unit nests inside the other. A test that clusters on one unit alone
leaks the dependence of the other unit into the p value.

A pilot of 2,418 real windows measures the effect. The variance of
`D = shift(MoRF) - shift(LeRF)` splits 47 percent to the pedestrian, 20 percent
to the time block and 33 percent to the residual. A simulation under the null,
with that same crossed dependence, gives these false positive rates against a
target of 0.05.

| Cluster unit | False positive rate |
|---|---|
| No cluster unit | 0.245 |
| Pedestrian | 0.085 |
| Time block | 0.115 |
| Connected component | 0.030 |
| Scene | 0.020 |

The connected component is the coarsest unit that respects both. The pipeline
builds the bipartite graph of the ego pedestrians against the time blocks
inside a scene, then takes the connected component of each window. The pilot
gives 39 components, and the test keeps 0.875 power against a true shift of
0.03 m. The scene unit is also safe, but the scene unit loses half the power.
Every p value in this report carries the component count beside it.

---

## 4. Model

### 4.1 AgentFormer

AgentFormer is a transformer forecaster for multi-agent trajectories. The model
attends over the agents and over time in one operation, so the attention map
holds one weight for every ordered pair of tokens. The study trains nothing.
The study loads the public weights and runs inference only.

### 4.2 The 5 leave-one-scene-out checkpoints

AgentFormer released 5 checkpoints, one per held-out scene. The study scores a
scene only with the checkpoint that held that scene out, so no checkpoint ever
sees its own test scene. The map from the scene to the checkpoint lives in
`config.SCENE_TO_CHECKPOINT`. The setup script records the sha256 checksum, the
source URL and the commit of every checkpoint.

The file `checkpoints/manifest.json` records the source and every checkpoint.
The code comes from `https://github.com/Khrylx/AgentFormer.git` at the commit
`e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`, with 6 patches. The weights come
from one archive, `agentformer_models.zip`, at
`https://drive.google.com/file/d/1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB/view`. The
archive holds 517,512,541 bytes and its sha256 is
`f192aa9596c25ec115dbf646f96a03f5b17e2c969016cd1b9c7e753db30941f0`.

| Scene | Configuration | Epoch | sha256 of the checkpoint |
|---|---|---|---|
| eth | eth_agentformer | 5 | `1c6b2b01ec236863e44798a65d6c1ba137712695b914fbf1bdc9d8cf100fae79` |
| hotel | hotel_agentformer | 50 | `537edd5ca1b8149bdbf786ecedb7bb0c01dfa3da6c0a6456b4d77949022da715` |
| univ | univ_agentformer | 15 | `09d2bd05cb3f018ed09d58a061ca26bb38cc98fcf6ad135e8c975014e2f7e2d7` |
| zara1 | zara1_agentformer | 40 | `c3619e1ebb44e6af9098e3c1b57a1843de56b93e810c279cad0179ce608a4103` |
| zara2 | zara2_agentformer | 25 | `214717ebb465261a7c3ee09f9a82ee7770add1df45a83b0e0b23e8dbc31577d1` |

Every checkpoint file holds 7,110,791 bytes. The eth checkpoint is the epoch 5
release, and every other checkpoint sits at epoch 15 or later. Section 9 states
the limit that the eth release creates.

### 4.3 Three measured facts that contradict the contract

The team measured these 3 facts against the released code. All 3 contradict
CONTRACT.md, and all 3 change how a later stage reads the model.

| Fact | What the contract says | What the code does |
|---|---|---|
| Token order | The token index is `agent_index * 8 + time_index`, so the order is agent-major. | AgentFormer flattens `(T, N, 2)` with a plain view, so a token sits at `time_index * N + agent_index`. The order is time-major. The wrapper transposes both axes, so every caller still sees the agent-major order of the contract. |
| Determinism | The model is a conditional variational autoencoder with DLow, so 2 identical calls give 2 different paths. | `DLow.inference()` calls `main(mean=True)`, which sets z to b and never draws an epsilon. 2 identical calls give identical paths. The `draws` argument therefore selects nothing, and the noise floor is exactly 0. |
| Map shapes | Every attention map is `(N*8, N*8)`. | Only the encoder map is `(N*8, N*8)`. The decoder self-attention is `(N*12, N*12)` and the cross attention is `(N*12, N*8)`. Every reader takes the shape from the array and never assumes it. |

The determinism fact is a gift and a limit at the same time. The noise floor of
0 means that any shift above 0 is a real change of the forecast, and not the
product of a new sample draw. Section 9.1 states the limit.

### 4.4 The extraction of the attention and the collapse to N by N

A hook reads the raw attention map of every module in one unmasked forward
pass. The extraction passes the raw map through with no change. The aggregate
step then collapses time first, then the heads, then the layers, and returns an
N by N map. The available time collapses are the mean, the max and the last
frame. The step also records the Shannon entropy of every source row before the
collapse, in natural logarithm units.

The rank step keeps only the edges whose destination is the ego. Ties break by
the lower source id, so the rank order is deterministic.

### 4.5 The primary choice

The primary map is the encoder social attention with the mean over time. The
encoder is the module whose map already matches the `(N*8, N*8)` shape that the
mask expands to. The mean over time is the collapse that keeps every observed
frame. The decoder module, the cross module, the max collapse and the last
collapse enter section 8 as sensitivity checks and never as a primary result.

---

## 5. Method

### 5.1 The 6 arms

Every arm removes edges that point into the ego, and every arm removes them
under the same latent draws. The common random numbers make the arms
comparable.

| Arm | Removal order | Role |
|---|---|---|
| MoRF | The strongest attention edge first. | The treatment of H1 and H2. |
| LeRF | The weakest attention edge first. | The comparison arm of H1. |
| Weight-matched | The smallest set of the lowest attention edges whose total attention reaches the target mass. | The mass-matched control. |
| Nearest | The closest neighbour first, by rank_dist. | The comparison arm of H2. |
| Random | A shuffle seeded from the global seed and the window key. | The floor. |
| Single | One named edge at a time, no order. | The brute force sweep that bounds the faithfulness index. |

### 5.2 The mass-matched control

MoRF removes few heavy edges. A plain count-matched control removes the same
number of edges but a much smaller attention mass, so a count-matched control
confounds the mass with the rank. The weight-matched arm removes many light
edges whose total attention reaches the mass that MoRF removed. The arm matches
the mass, not the count, so the comparison isolates the rank.

### 5.3 The 2 mask policies

The mask is an `(N, N)` boolean matrix, and True keeps the edge. Only the
entries whose destination is the ego ever become False. The mask builder
expands the entry `(i, j)` to the whole 8 by 8 block of that pair in the
`(N*8, N*8)` attention, because a pair is a block and not a cell.

| Policy | Rule | Role |
|---|---|---|
| `logit_neg_inf` | Add minus infinity to the logits before the softmax. AgentFormer already carries an `(N, N)` additive matrix in `data['agent_mask']`, so the wrapper writes into that matrix and patches nothing. | The primary policy. |
| `weight_zero` | Keep the additive mask at 0 and multiply the attention weights after the softmax by the tiled keep matrix, with no renormalisation. | The check. Section 8.5 reports it. |

### 5.4 The shift metric

The shift is the mean over the 12 forecast steps of the euclidean distance
between the masked ego path and the unmasked ego path, then the mean over the K
draws. The unit is the metre. The reference is the unmasked forecast, never the
ground truth. The metric compares the ego path only, and it compares the 2
paths over the same draws.

### 5.5 The faithfulness index

The single arm removes each edge into the ego on its own, so the arm returns
one shift per edge. The cost is E forward passes per window, and E is small.

    floor   = the mean of the shift over all single edges
    ceiling = the largest shift over all single edges
    FI      = ( shift(MoRF edge) - floor ) / ( ceiling - floor )

The 3 values carry these meanings.

| Value | Meaning |
|---|---|
| FI = 1 | Attention picked the truly strongest edge. |
| FI = 0 | Attention does no better than chance. |
| FI below 0 | Attention does worse than chance. |

The index is NaN when the ceiling minus the floor falls to or below a tolerance
of 1e-6. Those windows leave the H3 sample and the pipeline counts them.

### 5.6 The gate

The gate runs before any real number appears. The synthetic module builds a
scene with a known true edge list. The validation module scores the edges
against that planted truth and returns the AUROC. The AUROC must exceed 0.9. If
the method cannot recover a graph that the team planted itself, no number from
the real data means anything. A reversed rank order serves as the negative
control, because a reversed rank order must fall below 0.5.

### 5.7 The sanity checks

| Check | Rule | Pass condition |
|---|---|---|
| Zero check | Fixed draws, mask nothing, run twice. | The distance is exactly 0.0. |
| Noise floor | Free draws, mask nothing, repeat. | Every arm beats the floor. On the released model the floor is exactly 0, because the model is deterministic. |

Both checks pass in every scene. The zero check returns exactly 0.0 in eth,
hotel, univ, zara1 and zara2. The noise floor returns exactly 0.0 in the same 5
scenes, because the released model is deterministic. Section 7.2 holds the full
table.

---

## 6. Hypotheses and Tests

### 6.1 H1, the rank order of attention

Claim: the removal of the edge that attention ranks strongest moves the ego
forecast more than the removal of the edge that attention ranks weakest.

    D_i  = shift(MoRF)_i - shift(LeRF)_i,  at n_removed = 1
    H0-1 : pseudomedian(D) <= 0
    H1-1 : pseudomedian(D) > 0

The comparison of MoRF against Random and the comparison of MoRF against
Weight-matched support the claim. Neither comparison enters the primary family.

### 6.2 H2, beyond proximity

Claim: the removal of the high-attention edge moves the ego forecast more than
the removal of the nearest-neighbour edge. Proximity is the obvious rival
explanation, so the arm comparison is the primary test.

    D_i  = shift(MoRF)_i - shift(Nearest)_i,  at n_removed = 1
    H0-2 : pseudomedian(D) <= 0
    H1-2 : pseudomedian(D) > 0

The Spearman correlation of rank_attn against rank_dist, the partial
correlation of the attention with the single-edge shift given the distance, the
same test on the windows where the 2 arms disagree, and the attention shuffle
all support the claim. None of the 4 enters the primary family.

### 6.3 H3, context

Claim: the context of a window explains the faithfulness index.

    FI_i    = b0 + b . context_i + scene fixed effects
    context = density, n_agents, closing_speed, inv_ttc, all z-scored
    H0-3    : every b = 0 AND every scene term = 0
    H1-3    : at least one of those terms is not 0

The test is one joint Wald test of every non-intercept term. 5 scenes is too
few for a random intercept, so the scene enters as a fixed effect. The module
checks the collinearity first with the variance inflation factor, because a
near-redundant covariate makes both the fit and the joint test unreliable. The
standard errors cluster on the connected component. The module also reports the
partial R squared of the context block over a scene-only model.

### 6.4 The two-edge trap

A window at the frozen floor of 2 edges gives a faithfulness index of exactly
+1 or exactly -1. The floor is the mean of 2 shifts and the ceiling is the max
of the same 2 shifts, so only 2 values are reachable and no middle value
exists. Those windows are about 12 percent of the sample and 78 percent of the
usable windows of the scene eth. Mixed in with the continuous index of the
larger windows, those windows inject a bimodal spike that reflects an E of 2
and not the context.

H3 therefore fits the model twice. The first fit keeps every window with a
defined index. The second fit drops the two-edge windows. The headline is the
second fit. The report states both fits and states which fit is which.

### 6.5 The paired design and the test chooser

The design is paired, always. The 2 arms run on the same window, under the same
draws, so the analysis works on the paired differences D_i. The chooser reports
the shape of D and never assumes it. The chooser inspects the histogram, the QQ
plot, the skew, the bootstrap interval of the skew and the Shapiro-Wilk
statistic.

The symmetry rule decides the test. D counts as symmetric when the plain
bootstrap 95 percent interval of the sample skewness of the non-zero D contains
0, or when the absolute skew falls below the tolerance of 0.5. A symmetric D
gets the one-sided Wilcoxon signed-rank test, which estimates the pseudomedian.
The pseudomedian equals the median only when D is symmetric, so the report
always says which quantity it claims. A skewed or heavy-tailed D gets the sign
test on the median instead.

The chooser picks the sign test for both primary hypotheses.

| Item | H1, MoRF against LeRF | H2, MoRF against Nearest |
|---|---|---|
| Windows | 481 | 481 |
| Zero differences | 0 | 134 |
| Share of positive D among the non-zero D | 0.852 | 0.804 |
| Mean D, metres | 0.0755 | 0.0420 |
| Median D, metres | 0.0391 | 0.0087 |
| Pseudomedian D, metres | 0.0548 | 0.0206 |
| Skewness | 1.79 | 2.12 |
| Bootstrap interval of the skewness | 0.85 to 2.50 | 0.78 to 3.05 |
| Kurtosis | 8.36 | 10.93 |
| Shapiro-Wilk W | 0.810 | 0.752 |
| Shapiro-Wilk p | 2.2e-23 | 1.8e-22 |
| Symmetric | False | False |
| Test used | sign test | sign test |

Source: `outputs/tables/hypotheses_extra.csv`. The reason is one sentence for
each hypothesis. For H1 the skewness interval excludes 0 and the skewness of
1.79 stays above the tolerance of 0.5. The sign test on the median therefore
applies. For H2 the skewness interval excludes 0 and the skewness of 2.12 stays
above the same tolerance. The sign test on the median applies again. The report
therefore claims a median and never a pseudomedian for H1 and H2.

### 6.6 Where the p value comes from

The rows are not independent, so no standard table gives the p value. The p
value comes from a sign-flip permutation. The permutation flips the sign of
every D_i inside one connected component together, and it repeats that flip
10,000 times. The flip is exact for a paired design and it needs no GPU. A zero
difference keeps its rank under the Pratt rule and contributes 0 to the
statistic.

### 6.7 Where the interval comes from

The interval comes from a cluster bootstrap. For H1 and H2 the pairs cluster
bootstrap resamples the connected components with replacement, pools the rows
of the drawn components, applies the statistic and takes the percentile
interval over 10,000 resamples. For H3 the wild cluster bootstrap puts
Rademacher weights on the cluster residuals of the restricted regression fit.
The 2 tools are not the same tool, and each hypothesis uses only its own tool.

### 6.8 The multiplicity correction

The primary family is exactly 3 tests: H1, H2 and H3. The Holm correction
covers those 3 tests and nothing else. The Benjamini-Hochberg correction covers
the supporting family. That family holds the MoRF against Random comparison,
the MoRF against Weight-matched comparison, the disagreement-only comparison,
the Spearman correlation, the partial correlation and the attention shuffle.

### 6.9 The rejected alternatives

| Rejected method | Why the study rejects it |
|---|---|
| Paired t-test | D is long-tailed. The mean is not the target quantity. |
| Plain bootstrap | It assumes independent rows. The rows are not independent. |
| OLS or ANOVA on the raw rows | The same independence fault. |
| Bonferroni | Holm dominates Bonferroni at no cost. |
| Chi-square | The outcome is continuous, not a count. |
| Expected calibration error | ECE is a classifier metric. The study forecasts a continuous path, so no confidence score and no correct or incorrect outcome exist to bin. Coverage and the PIT histogram replace it. |
| A random intercept per scene | 5 scenes is too few. The scene enters as a fixed effect instead. |

---

## 7. Results

Every number of this section comes from a table on disk. The run identifier of
every table is `db6e67b8e6e7`, the hash of `config.py`. Section 11.2 states the
same hash.

The realised sample differs from the frozen plan. Section 3.9 names 571 windows
at stride 5, and the team froze that plan before the edge floor rose to 2. The
frozen row of `outputs/tables/design.csv` gives a design effect of 3.06 and an
effective n of 157 by time block. The realised row of the same table gives 481
windows, 72 connected components, 322 ego pedestrians and 152 time blocks. That
row gives a design effect of 3.08 and an effective n of 156. The effective n
holds near the frozen value, so the power statement of section 3.9 still
stands. The scene eth keeps only 8 of the 481 windows.

### 7.1 Accuracy and calibration

The accuracy table holds the average displacement error, the final displacement
error, the best-of-20 version of both, the collision flag and the collision
share. The collision flag is True when a majority of the K draws hold a
collision inside a radius of 0.2 m. The collision share is the float that the
table trusts, and the flag exists only to satisfy the contract column.

| Scene | Windows | ADE, m | FDE, m | minADE20, m | minFDE20, m |
|---|---|---|---|---|---|
| eth | 8 | 2.491 | 5.497 | 0.703 | 1.217 |
| hotel | 42 | 0.833 | 1.893 | 0.147 | 0.209 |
| univ | 190 | 1.031 | 2.304 | 0.277 | 0.484 |
| zara1 | 74 | 0.682 | 1.540 | 0.142 | 0.249 |
| zara2 | 167 | 0.724 | 1.665 | 0.125 | 0.208 |
| Pooled | 481 | 0.878 | 1.982 | 0.199 | 0.340 |

Source: `data/processed/predictions.parquet`. The scene eth holds the worst
error of the 5 scenes. Two facts explain that error. The eth checkpoint is the
epoch 5 release, and every other checkpoint sits at epoch 15 or later. The
scene eth also keeps only 8 windows, so one window carries 0.125 of the eth
mean. A reader must treat every eth number as weak.

| Scene | Collision share | Collision flag rate | Coverage at the 0.90 level | Share of the top PIT bin |
|---|---|---|---|---|
| eth | 0.125 | 0.000 | 0.615 | 0.323 |
| hotel | 0.067 | 0.071 | 0.845 | 0.127 |
| univ | 0.811 | 0.863 | 0.816 | 0.148 |
| zara1 | 0.065 | 0.068 | 0.923 | 0.045 |
| zara2 | 0.173 | 0.168 | 0.907 | 0.066 |
| Pooled | 0.398 | 0.416 | | |

Source: `data/processed/predictions.parquet` for the 2 collision columns and
`outputs/tables/calibration.csv` for the coverage and the PIT histogram. The
collision share is the fraction of the 20 draws that hold a pair inside 0.2 m.
The scene univ holds a mean of 25.7 agents per window. Its collision share of
0.811 therefore follows the crowd density and not a model fault alone. The
calibration table holds no pooled row, so the 2 pooled cells stay empty.

The PIT histogram holds 21 bins per scene. The top bin holds 0.323 of the mass
in eth, 0.127 in hotel, 0.148 in univ, 0.045 in zara1 and 0.066 in zara2. A
flat histogram gives 0.048 per bin. The scenes eth, hotel and univ therefore
pile mass at the top bin. The truth then falls outside the spread of the 20
modes in many windows. The coverage of 0.615 in eth and 0.845 in hotel says
the same fact in one number.

The determinism limit applies to every number in this subsection. AgentFormer
is deterministic, so the K draws are K fixed DLow modes and not K samples from
a posterior. The coverage and the PIT histogram therefore describe how the K
fixed modes sit around the truth. Neither number is a probabilistic calibration
claim.

### 7.2 The gate and the sanity checks

The gate passes. The attention score reaches an AUROC of 0.9270 against the
planted synthetic graph at the global seed. The test scores 259 edges, of which
74 are real and 185 are fake. The threshold is 0.9, so the gate clears it.
The test repeats the gate over 5 seeds, and the AUROC runs from 0.9262 to
0.9649 over those seeds. Every seed clears 0.9.

The negative control passes as well. The reversed rank order gives an AUROC of
0.0730 at the global seed, and the value runs from 0.0351 to 0.0738 over the 5
seeds. Every value stays far below 0.5, so a reversed order recovers nothing.
The 2 AUROC numbers come from `tests/test_synthetic.py`.

| Scene | Zero check | Noise floor | Windows | Forward passes |
|---|---|---|---|---|
| eth | 0.0 | 0.0 | 8 | 127 |
| hotel | 0.0 | 0.0 | 42 | 938 |
| univ | 0.0 | 0.0 | 190 | 33,146 |
| zara1 | 0.0 | 0.0 | 74 | 1,992 |
| zara2 | 0.0 | 0.0 | 167 | 6,838 |
| Total | 0.0 | 0.0 | 481 | 43,041 |

Source: `outputs/tables/ablation_sanity.csv`. The zero check is exactly 0.0 in
every scene, so 2 identical calls give identical paths. The noise floor is
exactly 0.0 in every scene, so every arm beats the floor. Any shift above 0 is
therefore a real change of the forecast.

The consistency check finds 0 mismatched rows. The check compares every arm row
at n_removed 1 against the single row of the same edge, over the 38,413 rows of
`data/processed/perturbation_curves.parquet`. Every pair of shifts agrees, so
no seed, no mask and no draw array drifted between the arms and the sweep.

### 7.3 H1, MoRF against LeRF

The test runs on 481 windows over 72 connected components. The count of zero
differences is 0, because every eligible window holds 2 edges or more, so MoRF
and LeRF always pick 2 different edges.

The chooser picks the sign test. The skewness of D is 1.79, and the bootstrap
interval of the skewness runs from 0.85 to 2.50. The interval excludes 0 and
the skewness stays above the tolerance of 0.5. The report therefore claims the
median of D and not the pseudomedian.

MoRF beats LeRF in 85.2 percent of the 481 windows. The 95 percent pairs
cluster bootstrap interval of that share runs from 80.9 percent to 90.7
percent. The median D is 0.039 m, with a cluster bootstrap interval of 0.034 m
to 0.046 m. The effect name is share_positive.

The sign-flip permutation gives a raw p of 0.0001 over 10,000 permutations.
That value is the smallest value that 10,000 permutations allow, so the true p
is at most 0.0001. The Holm-adjusted p is 0.0003. **H1 rejects the null.**

| Supporting row | Test | Effect | 95 percent interval | Raw p | BH p | Verdict |
|---|---|---|---|---|---|---|
| MoRF against Random | Wilcoxon signed rank | rank-biserial 0.611 | 0.505 to 0.744 | 0.0001 | 0.0001 | reject |
| MoRF against Weight-matched | sign test | share positive 0.270 | 0.178 to 0.343 | 1.0 | 1.0 | do not reject |

The 2 supporting rows point in 2 directions. MoRF beats Random, so the rank
order carries information above the floor. MoRF beats the mass-matched set in
only 27.0 percent of the windows, so the mass-matched control fails. Section
9.6 states what that failure means.

Figure `outputs/figures/hyp_curves.png` shows the mean shift against n_removed
for every arm, pooled, with a cluster bootstrap band and one small panel per
scene. Figure `outputs/figures/hyp_paired.png` shows the histogram of D for H1
and for H2. Each panel holds the median line, the zero line and the zero count.

### 7.4 H2, beyond proximity

Attention and the nearest neighbour pick the same top edge in 27.9 percent of
the windows. That share is 134 of the 481 windows, and each of those 134
windows gives a difference of exactly 0. Those 134 zeros are structural. The
share comes from `outputs/tables/hypotheses_extra.csv`, column share_agree.

| Scene | Mean Jaccard of the top edge | Median | Windows |
|---|---|---|---|
| eth | 0.875 | 1.0 | 8 |
| hotel | 0.500 | 0.5 | 42 |
| univ | 0.111 | 0.0 | 190 |
| zara1 | 0.216 | 0.0 | 74 |
| zara2 | 0.413 | 0.0 | 167 |

Source: `outputs/tables/arm_overlap.csv`. The overlap falls as the crowd grows.
In the scene eth the 2 orders name the same top edge in 7 of the 8 windows.
That scene therefore carries almost no information for H2. In the scene univ
the 2 orders agree in 11 percent of the windows, so attention and proximity
mostly name different edges there.

MoRF beats Nearest in 80.4 percent of the non-zero windows. The 95 percent
cluster bootstrap interval of that share runs from 75.6 percent to 84.6
percent. The median D is 0.0087 m, with a cluster bootstrap interval of 0.0003
m to 0.020 m. The raw p is 0.0001 over 10,000 sign-flip permutations, and the
Holm-adjusted p is 0.0003, over 72 components. **H2 rejects the null.**

The disagreement-only row repeats the primary numbers. The share is again 0.804
and the raw p is again 0.0001, and only the bootstrap interval moves, from
0.751 to 0.839. This repeat is a fact of the sign test and not a second
result. The sign test drops every zero difference before it counts. The primary
test therefore already ignores the 134 windows of agreement. The 2 rows run on
the same 347 non-zero windows by construction.

The Spearman rho of the attention against the distance is -0.23, over 6,080
edges, with a cluster bootstrap interval of -0.35 to -0.07. The attention
therefore rises as the distance falls, but the link is weak. The p of 2.4e-72
is a nominal row-wise p, and it treats every edge as independent. The cluster
bootstrap interval is therefore the honest statement of the uncertainty.

The partial correlation of the attention with the single-edge shift, given the
distance, is 0.70, with a cluster bootstrap interval of 0.65 to 0.74. The
attention therefore predicts the shift when the distance holds fixed. That link
is much stronger than the link of the attention with the distance itself.

The attention shuffle gives a median D of 0.039 m between the top edge and the
bottom edge. The p is 0.0001 over 10,000 shuffles, and the Benjamini-Hochberg p
is 0.0001. The shuffle permutes the attention weight inside the scene and keeps
the graph shape. The shuffle therefore breaks only the link of the attention
with the shift.

Figure `outputs/figures/hyp_paired.png` holds the histogram of D for H2 beside
the histogram for H1. The H2 panel holds the median line, the zero line and the
count of 134 zeros.

### 7.5 H3, context

The module checks the collinearity first.

| Term | VIF, headline fit | VIF, fit with the two-edge windows |
|---|---|---|
| density | 3.40 | 3.79 |
| n_agents | 3.19 | 3.57 |
| closing_speed | 1.63 | 1.70 |
| inv_ttc | 1.81 | 1.90 |

Source: `outputs/tables/h3_vif.csv`. The largest value is 3.4, well below the
usual threshold of 10, so no collinearity problem harms the fit or the joint
test.

Every one of the 481 windows holds a defined faithfulness index, and the count
of NaN indices is 0 in every scene. 60 windows hold exactly 2 edges into the
ego. The headline fit drops those 60 windows and keeps 421 windows over 62
components. The two-edge count is 7 in eth, 12 in hotel, 0 in univ, 24 in
zara1 and 17 in zara2. Sources: `data/processed/faithfulness.parquet` and
`outputs/tables/fi_attrition.csv`.

| Scene | Windows | Median index | Share above 0 | Share at exactly 1 | Two-edge windows |
|---|---|---|---|---|---|
| eth | 8 | 1.000 | 0.625 | 0.625 | 7 |
| hotel | 42 | 1.000 | 0.738 | 0.643 | 12 |
| univ | 190 | 0.546 | 0.863 | 0.368 | 0 |
| zara1 | 74 | 1.000 | 0.757 | 0.608 | 24 |
| zara2 | 167 | 1.000 | 0.731 | 0.515 | 17 |
| Pooled | 481 | 0.941 | 0.786 | 0.484 | 60 |

The index sits above 0 in 78.6 percent of the windows and reaches exactly 1 in
48.4 percent. The attention therefore names the truly strongest edge in almost
half of the windows. The scene univ holds the only continuous distribution,
because univ holds no two-edge window and a mean of 24.8 edges into the ego.

The headline fit is the second fit, without the two-edge windows.

| Term | b | Cluster-robust SE | p |
|---|---|---|---|
| Intercept | -0.636 | 0.047 | 2.3e-41 |
| scene hotel | 1.428 | 0.056 | 1.4e-143 |
| scene univ | 1.100 | 0.093 | 2.6e-32 |
| scene zara1 | 1.309 | 0.092 | 3.9e-46 |
| scene zara2 | 1.236 | 0.043 | 2.1e-180 |
| density | -0.062 | 0.043 | 0.143 |
| n_agents | 0.135 | 0.046 | 0.004 |
| closing_speed | 0.029 | 0.039 | 0.458 |
| inv_ttc | 0.010 | 0.037 | 0.791 |

Source: `outputs/tables/h3_coefficients.csv`, fit without_two_edge, 421 windows
over 62 components. The joint test covers every non-intercept term. The wild
cluster bootstrap gives a joint p of 0.60 over 10,000 Rademacher draws. The
partial R squared of the 4 context terms over a scene-only model is 0.017. The
context block therefore explains 1.7 percent of the variance that the scene
leaves.

The asymptotic cluster-robust Wald test on the same fit gives a p of 4.0e-56,
which rounds to 0.0000. **That value is not trustworthy, and the bootstrap p is
the primary p.** The two-edge drop takes 7 of the 8 eth windows, so the scene
eth keeps 1 window in the headline fit. The eth intercept therefore rests on
that single window, and every other scene term is a contrast against it. A
contrast against 1 window carries an almost zero standard error, so the Wald
statistic inflates. The scene coefficients above show the same fault: each one
carries a standard error near 0.05 and a p below 1e-31. The wild cluster
bootstrap does not trust the asymptotic covariance, so the bootstrap p of 0.60
is the number that the pre-registered plan reports.

Among the single terms only n_agents holds a cluster-robust p below 0.05, at
0.004, with a b of 0.135 per standard deviation. A larger crowd therefore
raises the faithfulness index a little. That single term is not the
pre-registered test. The pre-registered test is the joint test, and the joint
test does not reject.

The first fit keeps the two-edge windows and holds 481 windows over 72
components. That fit gives a joint p of 0.10 from the asymptotic cluster-robust
Wald test, and a partial R squared of 0.014. The n_agents term of that fit is
0.147, with a cluster-robust p of 0.003. Every scene term of that fit sits
above a p of 0.48. The 2 fits therefore agree: the context block does not
explain the index.

The Holm-adjusted p of the headline fit is 0.60, because 0.60 is the largest
raw p of the primary family of 3. **H3 does not reject the null.**

Figure `outputs/figures/hyp_faithfulness.png` shows the faithfulness index per
scene. Figure `outputs/figures/hyp_h3_coefficients.png` shows the coefficient
plot of the headline fit, with the cluster-robust interval.

### 7.6 The verdicts

| Hypothesis | Role | Test used | Effect | 95 percent interval | Adjusted p | Components | Verdict |
|---|---|---|---|---|---|---|---|
| h1 | primary | sign test, sign-flip permutation, pairs cluster bootstrap | share positive 0.852 | 0.809 to 0.907 | 0.0003 | 72 | reject |
| h2 | primary | sign test, sign-flip permutation, pairs cluster bootstrap | share positive 0.804 | 0.756 to 0.846 | 0.0003 | 72 | reject |
| h3 | primary | OLS with scene fixed effects, joint Wald test, wild cluster bootstrap | partial R squared 0.017 | not defined | 0.60 | 62 | do not reject |

The Holm correction covers those 3 rows and nothing else. The 3 raw p values
are 0.0001, 0.0001 and 0.60.

| Supporting row | Test used | Effect | 95 percent interval | BH p | Components | Verdict |
|---|---|---|---|---|---|---|
| h1_morf_vs_random | Wilcoxon signed rank | rank-biserial 0.611 | 0.505 to 0.744 | 0.0001 | 72 | reject |
| h1_morf_vs_weight_matched | sign test | share positive 0.270 | 0.178 to 0.343 | 1.0 | 72 | do not reject |
| h2_disagree_only | sign test | share positive 0.804 | 0.751 to 0.839 | 0.0001 | 72 | reject |
| h2_spearman_attn_dist | Spearman rank correlation | rho -0.228 | -0.353 to -0.073 | 8.3e-72 | 72 | reject |
| h2_partial_attn_shift_given_dist | partial correlation given the distance | 0.695 | 0.654 to 0.738 | 0.0 | 72 | reject |
| h2_attention_shuffle | attention shuffle inside the scene | median D 0.039 m | not defined | 0.0001 | 72 | reject |

The Benjamini-Hochberg correction covers those 6 rows. The table
`outputs/tables/verdicts.csv` also holds a seventh supporting row,
`h3_with_two_edge`. That row gives a partial R squared of 0.014, a raw p of
0.10 and an adjusted p of 0.12. That row is the second H3 fit of section 7.5.

Figure `outputs/figures/hyp_summary.png` shows the summary of the 3 primary
effects in one panel.

---

## 8. Sensitivity

The sensitivity table is descriptive. No p value in this section enters a
family. At a primary n_removed of 1 every arm removes exactly one edge, and the
single arm already holds the shift of every single edge. Every variant at
n_removed 1 is therefore a lookup into the single rows and costs no forward
pass. Only the second mask policy needs new forward passes, because that policy
changes the forward pass itself.

### 8.1 Leave-one-scene-out and heterogeneity

The leave-one-scene-out check runs H1 and H2 on each scene alone. The check
reports the heterogeneity between the scenes. The check is not 5 more tests.

| Hypothesis | Scene | Windows | Components | Effect | Location, m | Interval, m | Raw p |
|---|---|---|---|---|---|---|---|
| H1 | eth | 8 | 5 | rank-biserial 0.667 | 0.058 | 0.030 to 0.251 | 0.066 |
| H1 | hotel | 42 | 16 | rank-biserial 0.730 | 0.090 | 0.041 to 0.123 | 0.002 |
| H1 | univ | 190 | 21 | share positive 0.958 | 0.036 | 0.028 to 0.042 | 0.0001 |
| H1 | zara1 | 74 | 22 | rank-biserial 0.622 | 0.083 | 0.030 to 0.124 | 0.0009 |
| H1 | zara2 | 167 | 8 | share positive 0.802 | 0.042 | 0.019 to 0.104 | 0.024 |
| H2 | eth | 8 | 5 | none | 0.000 | not defined | 1.0 |
| H2 | hotel | 42 | 16 | share positive 0.571 | 0.000 | 0.000 to 0.000 | 0.377 |
| H2 | univ | 190 | 21 | share positive 0.870 | 0.024 | 0.015 to 0.030 | 0.0001 |
| H2 | zara1 | 74 | 22 | rank-biserial 0.472 | 0.033 | 0.007 to 0.088 | 0.012 |
| H2 | zara2 | 167 | 8 | share positive 0.796 | 0.000 | 0.000 to 0.000 | 0.014 |

Source: `outputs/tables/validate.csv`, the rows with check=loso. The location is
the median of D for a sign test and the pseudomedian of D for a Wilcoxon test.
H1 points the same way in all 5 scenes, and the p falls below 0.05 in 4 of
them. The scene eth gives a p of 0.066 on 8 windows and 5 components. That
scene lacks the power to clear 0.05, so it does not contradict the pooled
result.

H2 holds in univ, zara1 and zara2. H2 gives a p of 0.38 in hotel. H2 is
degenerate in eth, because attention and proximity name the same top edge in 7
of the 8 eth windows. Only 1 non-zero difference remains there, so the test
returns no effect and a p of 1.0. The scenes hotel and zara2 hold a median D of
exactly 0. The scene hotel holds 21 ties in 42 windows, and the scene zara2
holds 69 ties in 167 windows. The share of positive differences among the
non-zero windows still reaches 0.571 in hotel and 0.796 in zara2.

The equal-weight I squared is 0.0 for H1 and 0.0 for H2, so no heterogeneity
appears between the scenes beyond the noise of the sample. The range of the
per-scene location is 0.054 m for H1 and 0.033 m for H2. Those 2 rows carry
check=heterogeneity in the same table.

Figure `outputs/figures/hyp_loso.png` holds the forest plot of the per-scene
location for H1 and H2.

### 8.2 The n_removed sweep

| n_removed | Windows | Components | Test | Effect | Median or pseudomedian D, m | Interval, m | Raw p |
|---|---|---|---|---|---|---|---|
| 1 | 481 | 72 | sign test | share positive 0.852 | 0.039 | 0.034 to 0.046 | 0.0001 |
| 2 | 481 | 72 | Wilcoxon | rank-biserial 0.724 | 0.054 | 0.038 to 0.065 | 0.0001 |
| 3 | 421 | 62 | sign test | share positive 0.889 | 0.054 | 0.028 to 0.074 | 0.0001 |
| 4 | 351 | 43 | Wilcoxon | rank-biserial 0.746 | 0.072 | 0.059 to 0.082 | 0.0001 |
| 5 | 320 | 41 | Wilcoxon | rank-biserial 0.758 | 0.070 | 0.054 to 0.083 | 0.0001 |

Source: `outputs/tables/validate.csv`, the rows with check=n_removed. H1 holds
at every step from 1 to 5. The location grows from 0.039 m at 1 edge to 0.070 m
at 5 edges, so more removed edges move the forecast further. The window count
falls with the step, because a window must hold at least n_removed edges into
the ego.

### 8.3 The other modules

| Module | Hypothesis | Windows | Test | Effect | Location, m | Interval, m | Raw p |
|---|---|---|---|---|---|---|---|
| decoder | H1 | 481 | Wilcoxon | rank-biserial 0.570 | 0.034 | 0.021 to 0.044 | 0.0001 |
| decoder | H2 | 481 | Wilcoxon | rank-biserial 0.393 | 0.013 | 0.006 to 0.018 | 0.0002 |
| cross | H1 | 481 | Wilcoxon | rank-biserial 0.422 | 0.025 | 0.015 to 0.037 | 0.0001 |
| cross | H2 | 481 | sign test | share positive 0.705 | 0.000 | 0.000 to 0.006 | 0.0001 |

Source: `outputs/tables/validate.csv`, the rows with check=module. Both
hypotheses hold under the decoder self-attention and under the cross attention,
with 72 components in every row. The effects fall against the primary encoder
map. The rank-biserial correlation of H1 is 0.57 under the decoder and 0.42
under the cross module. The encoder map therefore ranks the edges best, and
that fact supports the primary choice of section 4.5.

### 8.4 The other time collapses

| Collapse | Hypothesis | Windows | Effect | Median D, m | Interval, m | Raw p |
|---|---|---|---|---|---|---|
| max | H1 | 481 | share positive 0.875 | 0.043 | 0.036 to 0.050 | 0.0001 |
| max | H2 | 481 | share positive 0.814 | 0.011 | 0.001 to 0.023 | 0.0001 |
| last | H1 | 481 | share positive 0.771 | 0.033 | 0.024 to 0.042 | 0.0001 |
| last | H2 | 481 | share positive 0.787 | 0.008 | 0.000 to 0.018 | 0.0001 |

Source: `outputs/tables/validate.csv`, the rows with check=time_agg. The sign
test applies to all 4 rows, over 72 components. Both hypotheses hold under the
max collapse and under the last collapse. The max collapse gives a slightly
larger effect than the primary mean collapse, at 0.875 against 0.852 for H1.
The choice of the time collapse therefore does not drive the result.

### 8.5 The second mask policy

| Hypothesis | Windows | Effect | Median D, m | Interval, m | Raw p |
|---|---|---|---|---|---|
| H1 | 481 | share positive 0.854 | 0.082 | 0.067 to 0.097 | 0.0001 |
| H2 | 481 | share positive 0.821 | 0.024 | 0.003 to 0.050 | 0.0001 |

Source: `outputs/tables/validate.csv`, the rows with check=mask_policy, over 72
components. Both hypotheses hold under the `weight_zero` policy. The shares
almost match the primary policy, at 0.854 against 0.852 for H1 and 0.821
against 0.804 for H2. The median D more than doubles for H1, from 0.039 m to
0.082 m, and it grows for H2 from 0.0087 m to 0.024 m. The second policy
removes the attention weight after the softmax with no renormalisation. That
policy therefore removes mass from the row and moves the forecast further. The
rank order of the arms does not change.

### 8.6 The attention shuffle and the noise floor

MoRF beats Random in 63.8 percent of the 481 windows. The median single-step
MoRF shift is 0.074 m, against a noise floor of exactly 0.0 m in every scene.
The row carries check=noise_floor in `outputs/tables/validate.csv` and it holds
no p value, because the row is a share and not a test. The floor of 0.0 comes
from the determinism of the released model. A shift of 0.074 m is therefore a
real change of the forecast and not a new sample draw.

The attention shuffle gives a median D of 0.039 m between the top edge and the
bottom edge, over 10,000 shuffles. The p is 0.0001 and the Benjamini-Hochberg p
is 0.0001. The shuffle permutes the attention weight inside the scene, so it
keeps the graph shape and the edge count of every window. The shuffle therefore
breaks only the link of the attention with the shift.

If MoRF does not beat Random above the noise floor, the answer is that the
attention is not faithful. That is a result, not a failure.

---

## 9. Limits

### 9.1 The released model is deterministic

`DLow.inference()` sets z to b and never draws an epsilon, so 2 identical calls
give identical paths. The K draws are K fixed modes, not K samples from a
posterior. The noise floor is therefore exactly 0, which is a gain for the
shift measurement. The same fact is a loss for the calibration. The coverage
and the PIT histogram describe the spread of the K fixed modes and not a
probabilistic calibration claim. No experiment exists in which the model draws
a different set of K paths.

### 9.2 Only 5 scenes

The benchmark holds 5 scenes. 5 scenes is too few for a random intercept, so
the scene enters H3 as a fixed effect. The scene cluster unit is safe but it
loses half the power, so the study uses the connected component instead. Any
claim about a scene that this study does not see stays out of reach.

### 9.3 The two-edge trap

A window with exactly 2 edges into the ego gives a faithfulness index of
exactly +1 or exactly -1, and no middle value is reachable. The frozen sample
holds 60 such windows in 481 windows, so 12 percent. Those 60 windows hold 7 of
the 8 windows of the scene eth. The headline H3 fit therefore drops those
windows, and the report states both fits.

### 9.4 The scene univ holds 60 percent of the rows

The scene univ holds 39,766 of the 66,676 rows and a mean of 25.7 agents per
window, against a mean of 2.6 agents in the scene eth. The scene univ also
takes 33,146 of the 43,041 forward passes, so 77 percent of the compute. A
pooled result therefore leans on one
scene. The leave-one-scene-out check of section 8.1 exists for exactly this
reason, and a reader must read the pooled number beside the per-scene numbers.

### 9.5 Faithfulness is about the model, not about people

Every result of this study describes the AgentFormer model. The masked edge is
a change inside the model, not a change in the street. The study reports an
association between the attention weight and the forecast shift. The study
never claims that one pedestrian causes the motion of another pedestrian.

### 9.6 The attention mass is not a currency of influence

The mass-matched control is the one supporting comparison that fails. MoRF
beats the mass-matched set of low-attention edges in only 27.0 percent of the
windows, with a Benjamini-Hochberg p of 1.0. Several light edges that together
carry the same attention mass as the single top edge move the ego MORE than
that top edge alone. The shift therefore follows the count of removed edges at
least as much as it follows the attention mass.

This result bounds the word faithful in this report. The attention ranks single
edges well, and H1 and H2 both show that rank. The attention mass is not a
conserved quantity of influence. A reader must never add 2 attention weights
and expect the sum to predict the shift.

### 9.7 The scene eth keeps only 8 windows

The scene eth keeps 8 of the 481 windows, and 7 of those 8 windows hold exactly
2 edges into the ego. The eth checkpoint is also the epoch 5 release, and every
other checkpoint sits at epoch 15 or later. The eth accuracy is therefore the
worst of the 5 scenes, at an ADE of 2.49 m. Three consequences follow. The
per-scene H1 test of eth lacks power and gives a p of 0.066. The per-scene H2
test of eth is degenerate, because attention and proximity name the same top
edge in 7 of the 8 windows. The headline H3 fit keeps 1 eth window, and section
9.8 explains that fault.

### 9.8 The asymptotic Wald test of H3 is not trustworthy

The headline H3 fit drops the two-edge windows, so the scene eth falls to 1
window. The eth intercept then rests on that single window, and every other
scene term is a contrast against it. Such a contrast carries an almost zero
standard error, so the asymptotic cluster-robust Wald statistic inflates and
returns a p of 4.0e-56. The wild cluster bootstrap does not trust that
covariance, and it returns a joint p of 0.60. The report therefore takes the
bootstrap p as the primary p, exactly as section 6.7 states.

---

## 10. Conclusion

The AgentFormer attention gives a faithful rank of the single edges that move
its own forecast. The top attention edge beats the weakest edge in 85.2 percent
of 481 windows, with a Holm p of 0.0003. The same edge beats the nearest
neighbour edge in 80.4 percent of the non-zero windows, with the same Holm p.
Proximity therefore does not explain the rank of the attention. The context of
a window does not explain the faithfulness index, because the joint bootstrap p
is 0.60 and the partial R squared is 0.017. The mass-matched control marks the
limit of the claim, because MoRF beats an equal mass of light edges in only
27.0 percent of the windows. A rank of single edges is therefore faithful, and
a mass of attention is not. Every statement here describes the model and not
the street, because the study masks an edge inside AgentFormer and never
touches a real pedestrian.

The sensitivity table supports the same answer. H1 points the same way in all 5
scenes and at every n_removed from 1 to 5. H1 also holds under both other
attention modules, under both other time collapses and under the second mask
policy. H2 holds in univ, zara1 and zara2. The scene eth carries no information
for H2, because attention and proximity name the same edge in 7 of its 8
windows.

Practical advice for a person who wants to use an AgentFormer attention map as
an explanation:

- Read the map as a rank, not as a budget. The top edge is the edge that the
  model uses, and the study supports that read with H1 and H2.
- Do not add the weights of several light edges and expect the sum to carry the
  influence of one heavy edge. Section 9.6 shows that the sum does not.
- Read the map beside the edge count of the window. A window with 2 edges gives
  only 2 reachable index values, so it says almost nothing.
- Do not read the map as a statement about the street. The map explains the
  model.

The next steps that this study makes possible:

- A count-controlled arm beside the mass-matched arm. The mass-matched failure
  of section 9.6 says that the count of removed edges drives the shift. The
  next design must vary the count and the mass one at a time.
- A forecaster with a real posterior. Section 9.1 shows that the released model
  is deterministic, so no experiment here makes a probabilistic calibration
  claim. The same pipeline runs on a stochastic model with no change to the
  statistics.
- A test of the same method on another attention forecaster. The connected
  component design, the sign-flip permutation and the bounded faithfulness
  index need no AgentFormer. The 3 tools transfer to any model that exposes an
  agent by agent attention map.

---

## 11. Reproducibility

### 11.1 The commands, in order

```
python scripts/download_ethucy.py
python scripts/setup_agentformer.py
python scripts/extract_attention.py
python scripts/run_ablation.py
python scripts/run_all.py
```

Every script is idempotent. A second run changes nothing. The download script
fetches the 5 scenes and writes `data/raw/manifest.json`. The setup script
fetches the 5 checkpoints and writes `checkpoints/manifest.json` with the
sha256, the URL and the commit. The extraction script runs once per scene and
writes the GPU cache under `data/interim`. The ablation script runs every arm
of every window and writes `data/processed/perturbation_curves.parquet`. The
top-level script calls every stage in the order of CONTRACT.md section 7.

### 11.2 The config hash

No magic number lives outside `config.py`. The run identifier records the hash
of that file, so a change of any number changes the hash, and every result that
carries the old hash becomes easy to find.

The config hash of the reported run is `db6e67b8e6e7`. The call
`config.config_hash()` returns that same value today. Every one of the 38,413
rows of `data/processed/perturbation_curves.parquet` carries `db6e67b8e6e7` in
the run_id column, and the column holds no other value. The reported run
therefore comes from one frozen configuration.

### 11.3 The tests

The command `python -m pytest tests/ -q` passes 412 tests and fails 0 tests, in
19.29 s. The run reports 8 warnings and no error. Two warnings say that
`MockPredictor` does not honour the `weight_zero` policy, so the driver writes
only `logit_neg_inf` against that mock model.

The 3 tests that must always pass are `tests/test_shapes.py`, which checks that
every predictor returns a `(K, N, 12, 2)` array, `tests/test_ablation.py`,
which checks that fixed draws and an all-True mask give a shift of exactly 0,
and `tests/test_synthetic.py`, which is the gate of section 5.6.

### 11.4 The hardware

The GPU stages run on an RTX 5080. One masked forward pass takes 26 ms. A
window with E edges into the ego costs about `7E + 1` passes for every arm and
both mask policies. The estimate of 26 ms per pass gives about 19 minutes for
the 43,041 passes of the frozen sample. The data stages, the exploratory
stages and every statistical stage need no GPU and no AgentFormer.

The machine holds one NVIDIA GeForce RTX 5080 with 16 GB, torch 2.11 with CUDA
12.8, Python 3.12.13 and Windows 11. The GPU ablation stage takes about 35
minutes for the 43,041 forward passes of section 7.2. Every CPU stage together
takes about 3 minutes. The measured time exceeds the 19 minute estimate,
because the wall clock also holds the model load, the mask build and the
parquet write.

---

## 12. References

1. Yuan, Y., Weng, X., Ou, Y., and Kitani, K. 2021. AgentFormer: Agent-Aware
   Transformers for Socio-Temporal Multi-Agent Forecasting. IEEE International
   Conference on Computer Vision (ICCV).
2. Pellegrini, S., Ess, A., Schindler, K., and Van Gool, L. 2009. You Will
   Never Walk Alone: Modeling Social Behavior for Multi-Target Tracking. IEEE
   International Conference on Computer Vision (ICCV).
3. Lerner, A., Chrysanthou, Y., and Lischinski, D. 2007. Crowds by Example.
   Computer Graphics Forum (Eurographics).
4. Jain, S., and Wallace, B. C. 2019. Attention is not Explanation. Conference
   of the North American Chapter of the Association for Computational
   Linguistics (NAACL-HLT).
5. Wiegreffe, S., and Pinter, Y. 2019. Attention is not not Explanation.
   Conference on Empirical Methods in Natural Language Processing (EMNLP).
6. Pratt, J. W. 1959. Remarks on Zeros and Ties in the Wilcoxon Signed Rank
   Procedures. Journal of the American Statistical Association.
7. Cameron, A. C., Gelbach, J. B., and Miller, D. L. 2008. Bootstrap-Based
   Improvements for Inference with Clustered Errors. The Review of Economics
   and Statistics.
8. Holm, S. 1979. A Simple Sequentially Rejective Multiple Test Procedure.
   Scandinavian Journal of Statistics.
9. Benjamini, Y., and Hochberg, Y. 1995. Controlling the False Discovery Rate:
   A Practical and Powerful Approach to Multiple Testing. Journal of the Royal
   Statistical Society, Series B.

---

## 13. Appendix A. The Output Manifest

### 13.1 The processed tables

| File | What it holds | Section |
|---|---|---|
| `data/processed/trajectories.parquet` | One row per agent per frame: scene, frame, agent_id, x, y, vx, vy, speed, heading, nearest_dist, density. 66,676 rows. | 3 |
| `data/processed/windows.parquet` | One row per window: window_id, scene, t0, ego_id, agent_order, n_agents, hist, fut, overlap_fraction, eligible. | 3 |
| `data/processed/attention_edges.parquet` | One row per pair per module per time collapse: attn, row_entropy, dist_at_last_frame, closing_speed, inv_ttc, rank_attn, rank_dist. | 4, 5, 7.4 |
| `data/processed/perturbation_curves.parquet` | One row per window, arm, n_removed, draw and mask policy: removed_mass, edge_src, run_id, shift. | 5, 7.3, 7.4, 8 |
| `data/processed/faithfulness.parquet` | One row per window: n_edges, floor, ceiling, fi. | 5.5, 7.5 |
| `data/processed/predictions.parquet` | One row per window: ade, fde, minade_20, minfde_20, collision, collision_share. | 7.1 |
| `data/processed/verdicts.parquet` | One row per claim: hypothesis, role, test_used, why, sided, effect, effect_name, ci_low, ci_high, p_raw, p_adj, n_clusters, verdict. | 7.3 to 7.6 |

The GPU cache under `data/interim` holds `attention_<scene>.parquet` and
`predict_<scene>.npz` for every scene. The cache is an input to the processed
tables and it is not itself a processed table.

### 13.2 The output tables

| File | What it holds | Section |
|---|---|---|
| `outputs/tables/eda_summary.csv` | Counts per scene: rows, share of rows, pedestrians, frames, windows, eligible windows, egos, mean agents, edges into the ego. | 3.2 |
| `outputs/tables/eda_density.csv` | Per scene: the agent count per window, the edge count, the ego density, the ego nearest distance and the forward-pass share. | 3.2, 9.4 |
| `outputs/tables/eda_distributions.csv` | The shape of the step length, the speed and the spacing. | 3.4, 6.5 |
| `outputs/tables/eda_missing.csv` | The gaps and the bad frames, per scene. | 3.4 |
| `outputs/tables/eda_frame_occupancy.csv` | How many agents each frame holds. | 3.4 |
| `outputs/tables/eda_scaling.csv` | The raw, the z-scored and the log scale side by side, for the transform decision. | 3.4 |
| `outputs/tables/eda_overlap.csv` | The stride sweep: windows, clusters, the intraclass correlation, the design effect, the effective n and the forward-pass cost, for both cluster units. | 3.8, 3.9 |
| `outputs/tables/drops.csv` | One row per drop, with the stage, the scene, the key, the reason and the row count. | 3.4, 3.7 |
| `outputs/tables/calibration.csv` | Per scene: the level, the window count, the coverage and the PIT histogram bins. | 7.1 |
| `outputs/tables/ablation_sanity.csv` | The zero check, the noise floor and the consistency check, per scene. | 5.7, 7.2 |
| `outputs/tables/fi_attrition.csv` | Per scene: the eligible window count, the count with an E of 1, the count with an E of 2 or less, and the count of NaN indices. | 5.5, 7.5 |
| `outputs/tables/validate.csv` | The sensitivity rows: check, variant, hypothesis, n_windows, n_clusters, effect_name, effect, location_m, ci_low, ci_high, p_raw. | 8 |
| `outputs/tables/design.csv` | The frozen row and the realised row of the design: windows, components, pedestrians, time blocks, the design effect and the effective n. | 3.9, 7 |
| `outputs/tables/arm_overlap.csv` | Per scene: the Jaccard overlap of the top MoRF edge and the top Nearest edge. | 7.4 |
| `outputs/tables/attrition.csv` | Per scene: the window count, the complete count and the incomplete count of the ablation run. | 7.2 |
| `outputs/tables/verdicts.csv` | The csv copy of `data/processed/verdicts.parquet`. | 7.3 to 7.6 |
| `outputs/tables/hypotheses_extra.csv` | The shape statistics of D, the zero count, the location in metres and the share that the 2 arms agree. | 6.5, 7.3, 7.4 |
| `outputs/tables/h3_coefficients.csv` | Every coefficient of both H3 fits, with the cluster-robust standard error, the p, the joint p and the partial R squared. | 7.5 |
| `outputs/tables/h3_vif.csv` | The variance inflation factor of every context term, for both H3 fits. | 7.5 |

### 13.3 The figures

| File | What it holds | Section |
|---|---|---|
| `outputs/figures/eda_summary.png` | The scene counts. | 3.2 |
| `outputs/figures/eda_density.png` | The neighbour count per window. | 3.2 |
| `outputs/figures/eda_distributions.png` | The step, the speed and the spacing. The shift is long-tailed, so expect the skewed branch of the test chooser. | 3.4, 6.5 |
| `outputs/figures/eda_missing.png` | The gaps and the bad frames. | 3.4 |
| `outputs/figures/eda_scaling_ego_nearest_dist.png` | The 3 scales side by side. The rank tests need no transform, so H1 and H2 use the raw shift. | 3.4 |
| `outputs/figures/eda_overlap.png` | The effective sample size and the forward-pass cost against the stride. | 3.8 |
| `outputs/figures/hyp_curves.png` | The mean shift against n_removed per arm, pooled, with a cluster bootstrap band, plus one small panel per scene. | 7.3 |
| `outputs/figures/hyp_paired.png` | The histogram of D for H1 and H2, with the median line, the zero line and the zero count. | 7.3, 7.4 |
| `outputs/figures/hyp_faithfulness.png` | The faithfulness index per scene. | 7.5 |
| `outputs/figures/hyp_h3_coefficients.png` | The coefficient plot with the cluster-robust interval, headline fit. | 7.5 |
| `outputs/figures/hyp_loso.png` | The forest plot of the per-scene location for H1 and H2. | 8.1 |
| `outputs/figures/hyp_summary.png` | The summary of the 3 primary effects in one panel. | 7.6 |

The 6 hypothesis figure names now come from `src/hypotheses/figures.py`, which
writes each file through `src.eda.plotting.save`. The 5 names `hyp_curves`,
`hyp_paired`, `hyp_faithfulness`, `hyp_h3_coefficients` and `hyp_loso` match
the files on disk. The file `hyp_summary.png` is in production at the date of
this report, so a reader must confirm that one file before use.

### 13.4 The other artefacts

| File | What it holds | Section |
|---|---|---|
| `data/raw/manifest.json` | The source URL and the checksum of every raw scene file. | 11.1 |
| `checkpoints/manifest.json` | The sha256, the URL and the commit of the 5 released checkpoints. | 4.2, 11.1 |
| `README.mmd` and `README.png` | The pipeline diagram. The 2 files must never drift apart. | 2, 11 |
| `notebooks/capstone.ipynb` | The one notebook. It calls, it plots and it writes up. No logic lives in the notebook. | 7, 8 |
