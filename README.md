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

This should report `57 passed` (some tests skip without a GPU — see below).

## macOS-specific notes

**SSL certificate error on `scripts/download_ethucy.py`.** If you installed Python from python.org, you may see `CERTIFICATE_VERIFY_FAILED` the first time you download anything. Fix it once with:

```bash
open "/Applications/Python 3.13/Install Certificates.command"
```

(adjust the version number to match your installed Python)

**No CUDA GPU.** `requirements-gpu.txt` and the AgentFormer setup stage (`scripts/setup_agentformer.py`) assume an NVIDIA GPU. On Apple Silicon or Intel Macs, `pip install torch` gives you a CPU build instead — the GPU-only tests in `tests/` are marked to skip automatically when no CUDA device is present, so `pytest tests/ -q` still passes cleanly without one.
