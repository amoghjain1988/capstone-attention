# capstone-attention

DAMO 699 capstone. AgentFormer attention faithfulness.

See `README.mmd` (rendered in `README.png`) for the full pipeline diagram, and `CONTRACT.md` for the data and function contracts.

## Setup

The per-role setup docs (for example `docs/GRACE_AGENT.md`) give commands written for Windows. If you're on macOS or Linux, use the commands below instead — everything else in those docs (which files to touch, which tests to run) still applies.

### 1. Clone and branch

```bash
git clone <REPO_URL> capstone-attention
cd capstone-attention
git checkout -b your-branch-name
```

### 2. Create and activate a virtual environment

| | Windows | macOS / Linux |
|---|---|---|
| Create | `uv venv --python 3.12 .venv` | `python3 -m venv .venv` |
| Activate | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| Run Python | `.venv\Scripts\python.exe` | `python3` (once activated) |

Once activated, your terminal prompt should show `(.venv)` at the start of the line.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data and run the base tests

```bash
python3 scripts/download_ethucy.py
python3 -m pytest tests/ -q
```

This must report `at least 400 passed` (some tests skip without a GPU — see below).

## Run the study

Run the commands in this order from the repository root. Commands 3 and 5 need
a CUDA GPU. Every other command runs on a CPU.

```bash
python scripts/download_ethucy.py
python scripts/setup_agentformer.py
python scripts/extract_attention.py
python scripts/run_all.py --through 8
python scripts/run_ablation.py
python scripts/run_all.py
```

1. `download_ethucy.py` gets the 5 ETH / UCY scenes and writes the manifest.
2. `setup_agentformer.py` gets the 5 AgentFormer checkpoints, one per held-out
   scene, and records the checksum and the commit.
3. `extract_attention.py` runs the model once per window on the GPU and caches
   the attention and the unmasked prediction under `data/interim/`.
4. `run_all.py --through 8` runs stages 1 to 8. It builds the `trajectories`
   and `windows` tables, the exploratory tables, the `attention_edges` table,
   the frozen design and the `predictions` table.
5. `run_ablation.py` masks the edges on the GPU and writes
   `data/processed/perturbation_curves.parquet`.
6. `run_all.py` runs every stage to the end. It builds the `faithfulness`
   table, the `verdicts` table, the figures and `notebooks/capstone.ipynb`.

Every stage is idempotent. Run a command twice and nothing changes. Use
`--through N` to stop at stage N.

Read `docs/FINISH_PLAN.md` for the state of the code and the interface of every
module. Read `docs/REPORT.md` for the method and the results.

## macOS-specific notes

**SSL certificate error on `scripts/download_ethucy.py`.** If you installed Python from python.org, you may see `CERTIFICATE_VERIFY_FAILED` the first time you download anything. Fix it once with:

```bash
open "/Applications/Python 3.13/Install Certificates.command"
```

(adjust the version number to match your installed Python)

**No CUDA GPU.** `requirements-gpu.txt` and the AgentFormer setup stage (`scripts/setup_agentformer.py`) assume an NVIDIA GPU. On Apple Silicon or Intel Macs, `pip install torch` gives you a CPU build instead — the GPU-only tests in `tests/` are marked to skip automatically when no CUDA device is present, so `pytest tests/ -q` still passes cleanly without one.
