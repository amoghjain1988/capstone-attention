"""Fetch the AgentFormer code and the five checkpoints. Record the checksum.

See CONTRACT.md section 5.1 for the signatures.

The script does five things and every one of them is idempotent.

    1. Clone AgentFormer at a pinned commit into third_party/.
    2. Patch three PyTorch 2.x breakages. AgentFormer pins PyTorch 1.8.
    3. Download agentformer_models.zip, check the sha256, and unpack it.
    4. Link third_party/AgentFormer/results to checkpoints/results, because
       their utils/config.py reads a relative path from their own repository
       root.
    5. Write checkpoints/manifest.json with the sha256, the url and the commit.

Run it twice and nothing changes.

WHY THE PATCH EXISTS
AgentFormer was released for PyTorch 1.8 on Linux and MacOS. An RTX 50 series
card is compute capability 12.0, and PyTorch 1.8 cannot emit code for it. We
therefore run PyTorch 2.x and repair three small breakages. The repairs do not
touch the mathematics and they do not touch the weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

REPO_URL = "https://github.com/Khrylx/AgentFormer.git"
REPO_COMMIT = "e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5"
AGENTFORMER_ROOT = config.ROOT / "third_party" / "AgentFormer"

# The pretrained models, from the link in the AgentFormer README.
MODELS_DRIVE_ID = "1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB"
MODELS_ZIP_NAME = "agentformer_models.zip"
# Recorded from the download of 2026-08-17. A fresh clone checks against this
# value and downloads only when the file is absent or the hash differs.
MODELS_SHA256 = "f192aa9596c25ec115dbf646f96a03f5b17e2c969016cd1b9c7e753db30941f0"
MODELS_BYTES = 517_512_541

MANIFEST_NAME = "manifest.json"

# (relative path, what to find, what to write). Every entry is idempotent,
# because the script also recognises the patched text.
PATCHES: list[tuple[str, str, str]] = [
    (
        "model/agentformer_lib.py",
        "from torch.nn.modules.linear import Linear, _LinearWithBias",
        "from torch.nn.modules.linear import Linear",
    ),
    (
        "model/agentformer_lib.py",
        "self.out_proj = _LinearWithBias(embed_dim, embed_dim)",
        "self.out_proj = Linear(embed_dim, embed_dim, bias=True)",
    ),
    (
        "model/agentformer_lib.py",
        "from torch.overrides import has_torch_function, handle_torch_function",
        "from torch.overrides import has_torch_function, handle_torch_function\n"
        "\n"
        "# PyTorch 2.x compatibility. torch.nn.functional now defines __all__, so the\n"
        "# star import above no longer leaks these names into this module.\n"
        "from typing import List, Optional, Tuple\n"
        "from torch import Tensor\n"
        "from torch.nn.functional import dropout, linear, pad, softmax",
    ),
    ("data/preprocessor.py", "astype(np.int)", "astype(int)"),
    ("data/preprocessor.py", "dtype=np.int)", "dtype=int)"),
    ("data/convert_ethucy.py", "astype(np.str)", "astype(str)"),
]


def sha256_of(path: Path) -> str:
    """Return the sha256 of a file as a hexadecimal string."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run a command and raise when it fails."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 1. The source
# ---------------------------------------------------------------------------


def clone_source() -> str:
    """Clone AgentFormer at the pinned commit. Return the commit."""
    if (AGENTFORMER_ROOT / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=AGENTFORMER_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"  source present at {head[:12]}")
        return head

    AGENTFORMER_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(f"  clone {REPO_URL}")
    run(["git", "clone", REPO_URL, str(AGENTFORMER_ROOT)])
    run(["git", "checkout", REPO_COMMIT], cwd=AGENTFORMER_ROOT)
    print(f"  checked out {REPO_COMMIT[:12]}")
    return REPO_COMMIT


# ---------------------------------------------------------------------------
# 2. The patch
# ---------------------------------------------------------------------------


def apply_patches() -> int:
    """Apply every compatibility patch. Return how many files changed."""
    changed = 0
    for relative, old, new in PATCHES:
        path = AGENTFORMER_ROOT / relative
        if not path.exists():
            print(f"  skip {relative}, the file is absent")
            continue
        text = path.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(
                f"{relative}: cannot find {old!r}. The pinned commit may have moved."
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patched {relative}")
        changed += 1
    if changed == 0:
        print("  every patch is already applied")
    return changed


# ---------------------------------------------------------------------------
# 3. The checkpoints
# ---------------------------------------------------------------------------


def download_models(manifest: dict) -> Path:
    """Download and unpack the pretrained models. Return the zip path."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.CHECKPOINT_DIR / MODELS_ZIP_NAME
    known = manifest.get("models", {}).get("sha256") or MODELS_SHA256

    if archive.exists() and known and sha256_of(archive) == known:
        print(f"  {MODELS_ZIP_NAME} present, hash matches")
    else:
        try:
            import gdown
        except ImportError as error:
            raise SystemExit(
                "gdown is missing. Install it with:\n"
                "  uv pip install --python .venv/Scripts/python.exe -r requirements-gpu.txt"
            ) from error
        print(f"  download {MODELS_ZIP_NAME} from Google Drive, about 518 MB")
        gdown.download(id=MODELS_DRIVE_ID, output=str(archive), quiet=False)

    digest = sha256_of(archive)
    if known and digest != known:
        raise RuntimeError(
            f"{MODELS_ZIP_NAME} has sha256 {digest}, the manifest expects {known}"
        )

    results = config.CHECKPOINT_DIR / "results"
    wanted = [config.SCENE_TO_CHECKPOINT[scene] for scene in config.SCENES]
    if all((results / name / "models").is_dir() for name in wanted):
        print("  every scene checkpoint is already unpacked")
    else:
        print(f"  unpack into {results.relative_to(config.ROOT)}")
        shutil.unpack_archive(archive, config.CHECKPOINT_DIR)
    return archive


# ---------------------------------------------------------------------------
# 4. The link
# ---------------------------------------------------------------------------


def link_results() -> None:
    """Point third_party/AgentFormer/results at checkpoints/results.

    Their utils/config.py globs relative paths from their own repository root,
    so the checkpoints must appear there. A link keeps one copy on disk. If the
    platform refuses a link, the function copies instead.
    """
    target = config.CHECKPOINT_DIR / "results"
    link = AGENTFORMER_ROOT / "results"

    if link.is_dir() and any(link.iterdir()):
        print("  results already reachable from the AgentFormer root")
        return

    if link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link)

    try:
        os.symlink(target, link, target_is_directory=True)
        print("  linked results to checkpoints/results")
        return
    except (OSError, NotImplementedError):
        pass

    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  junction created for results")
            return

    print("  the platform refuses a link, so copy instead")
    shutil.copytree(target, link)


# ---------------------------------------------------------------------------
# 5. The manifest
# ---------------------------------------------------------------------------


def build_manifest(commit: str, archive: Path) -> dict:
    """Return the manifest and record every checkpoint file."""
    results = config.CHECKPOINT_DIR / "results"
    scenes = {}
    for scene in config.SCENES:
        name = config.SCENE_TO_CHECKPOINT[scene]
        models = sorted((results / name / "models").glob("model_*.p"))
        if not models:
            raise FileNotFoundError(f"no checkpoint under results/{name}/models")
        newest = models[-1]
        scenes[scene] = {
            "cfg": name,
            "epoch": int(newest.stem.split("_")[-1]),
            "path": str(newest.relative_to(config.ROOT)).replace("\\", "/"),
            "sha256": sha256_of(newest),
            "bytes": newest.stat().st_size,
        }
    return {
        "source": {"url": REPO_URL, "commit": commit, "patches": len(PATCHES)},
        "models": {
            "url": f"https://drive.google.com/file/d/{MODELS_DRIVE_ID}/view",
            "archive": MODELS_ZIP_NAME,
            "sha256": sha256_of(archive),
            "bytes": archive.stat().st_size,
        },
        "note": (
            "Score a scene only with the checkpoint that held that scene out. "
            "<scene>_agentformer is the DLow sampler and it loads "
            "<scene>_agentformer_pre, which is the variational autoencoder."
        ),
        "scenes": scenes,
    }


def main() -> None:
    """Set up AgentFormer and the checkpoints."""
    config.make_dirs()
    manifest_path = config.CHECKPOINT_DIR / MANIFEST_NAME
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    print("1. source")
    commit = clone_source()
    print("2. patch for PyTorch 2.x")
    apply_patches()
    print("3. checkpoints")
    archive = download_models(manifest)
    print("4. link")
    link_results()
    print("5. manifest")
    manifest = build_manifest(commit, archive)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {manifest_path.relative_to(config.ROOT)}")

    print("\nready:")
    for scene, record in manifest["scenes"].items():
        print(f"  {scene:6s} {record['cfg']:20s} epoch {record['epoch']:4d}")
    print("\nNext: python scripts/extract_attention.py")


if __name__ == "__main__":
    main()
