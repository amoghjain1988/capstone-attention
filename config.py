"""The frozen numbers. Nothing else holds a constant.

See CONTRACT.md section 6.

Read every number from this file. Do not write a number into a module.
After the exploratory analysis, fill MINIMUM_EFFECT and POWER, then hash this
file with config_hash() and commit the hash. After that step, change nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------

SEED = 0

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

N_HIST = 8  # observed frames, 3.2 s
N_FUT = 12  # forecast frames, 4.8 s
DT = 0.4  # seconds in one frame. The rate is 2.5 Hz
WINDOW_LEN = N_HIST + N_FUT  # 20 frames

# The step between two window starts, in frames.
# A stride of 1 gives every possible window. The windows then overlap heavily.
# src/eda/overlap.py reports the design effect for each stride.
# Freeze this number in design.py, before the first ablation run.
WINDOW_STRIDE = 5  # Changed by prem from 1 to 5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

N_SAMPLES = 20  # K, the latent draws per forward pass
PRIMARY_MODULE = "encoder"
PRIMARY_TIME_AGG = "mean"
PRIMARY_MASK_POLICY = "logit_neg_inf"
MODULES = ("encoder", "decoder", "cross")
TIME_AGGS = ("mean", "max", "last")
MASK_POLICIES = ("logit_neg_inf", "weight_zero")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

ALPHA = 0.05
SIDED = "one"
PRIMARY_N_REMOVED = 1
N_BOOT = 10_000
N_PERM = 10_000

# Fill these two after the exploratory analysis and before the first ablation.
# src/eda/overlap.py gives the design effect that src/hypotheses/design.py needs.
MINIMUM_EFFECT = 0.592 #changed by prem from none to 0.592
POWER = 0.80 #changed by prem from none to 0.80

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

COLLISION_RADIUS_M = 0.2
DENSITY_RADIUS_M = 5.0

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

MIN_TRACK_FRAMES = 20

# The smallest number of edges that must point into the ego, E = N - 1.
#
# Set this to 2, not 1. A window with E of 1 holds one edge only, so MoRF, LeRF,
# Nearest and Random all pick THE SAME edge. Then:
#   H1: D = shift(MoRF) - shift(LeRF) is 0 by construction.
#   H2: D = shift(MoRF) - shift(Nearest) is 0 by construction.
#   H3: the floor equals the ceiling, so the faithfulness index is NaN.
# Such a window carries no information for any of the three primary tests. Those
# zeros are structural, not observed ties. The Pratt rule would rank them and
# would pull the test toward the null, so keeping them adds bias, not caution.
#
# E is the agent count minus 1. It is known before any model runs, so this rule
# selects on a pre-treatment covariate and not on an outcome.
#
# The rule drops 423 of 2,841 windows and only 2.5 percent of the compute.
# All 5 scenes keep 20 windows or more. A larger value costs the scene eth:
# at 3 the scene eth falls from 32 windows to 7. See src/eda/overlap.py
# min_edges_report() and docs/EDA_FINDINGS.md finding F3.
MIN_EGO_EDGES = 2

# Keep a short track in the table as a neighbour.
# The ego of a window is present for all 20 frames, so the ego always passes
# MIN_TRACK_FRAMES. A short track is still a real pedestrian that the model
# sees, and it carries a real attention edge. Set this flag to True only to
# measure the effect of the other policy. src/eda/summary.py reports both.
DROP_SHORT_TRACKS = False

# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

SCENES = ("eth", "hotel", "univ", "zara1", "zara2")

# The raw source files, in the AgentFormer leave-one-scene-out layout.
# The key is our scene name. The value is the list of source files.
# A scene with two files gets an offset per file. See src/data/loader_ethucy.py.
SCENE_SOURCES = {
    "eth": ("eth/biwi_eth",),
    "hotel": ("hotel/biwi_hotel",),
    "univ": ("univ/students001", "univ/students003"),
    "zara1": ("zara1/crowds_zara01",),
    "zara2": ("zara2/crowds_zara02",),
}

# The offset that separates two source files of the same scene.
# The offset keeps every frame index and every agent id unique inside a scene.
# The offset is much larger than WINDOW_LEN, so no window spans two files.
SOURCE_FILE_OFFSET = 100_000

# The column indices of the AgentFormer text format.
# AgentFormer reads x from column 13 and y from column 15.
# See data/preprocessor.py in the AgentFormer repository.
COL_FRAME = 0
COL_AGENT = 1
COL_CLASS = 2
COL_X = 13
COL_Y = 15

DATA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/Khrylx/AgentFormer/main/datasets/eth_ucy/{path}.txt"
)

# AgentFormer released one checkpoint per held-out scene.
# Score a scene only with the checkpoint that held that scene out.
# The value is the AgentFormer configuration name.
# scripts/setup_agentformer.py turns the name into a file and records the hash.
SCENE_TO_CHECKPOINT = {
    "eth": "eth_agentformer",
    "hotel": "hotel_agentformer",
    "univ": "univ_agentformer",
    "zara1": "zara1_agentformer",
    "zara2": "zara2_agentformer",
}

SCENE_TO_DLOW = {scene: f"{scene}_dlow" for scene in SCENES}


# ---------------------------------------------------------------------------
# The hash
# ---------------------------------------------------------------------------


def config_hash() -> str:
    """Return the sha256 of this file, first 12 characters.

    run_id records this hash. If a number changes, the hash changes, and every
    result that carries the old hash becomes easy to find.
    """
    text = Path(__file__).read_bytes()
    return hashlib.sha256(text).hexdigest()[:12]


def make_dirs() -> None:
    """Create every output directory. The function is idempotent."""
    for path in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        CHECKPOINT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
