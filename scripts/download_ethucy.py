"""Download the five ETH/UCY scenes into data/raw.

See CONTRACT.md section 5.1 for the signatures.

The source is the AgentFormer repository. AgentFormer keeps the benchmark text
files in datasets/eth_ucy, in a leave-one-scene-out layout. For each scene we
take the file that carries no _train or _val suffix. That file is the held-out
test split, and it is the split the released checkpoint never saw.

The script is idempotent. It writes a manifest with the sha256 of every file.
A second run compares the hash and downloads nothing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

MANIFEST_NAME = "manifest.json"
RETRIES = 3
TIMEOUT_S = 120


def sha256_of(path: Path) -> str:
    """Return the sha256 of a file as a hexadecimal string."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    """Download one file. Retry up to RETRIES times.

    Write to a temporary name first, then rename. A broken download therefore
    never leaves a half file in place.
    """
    temporary = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
                payload = response.read()
            temporary.write_bytes(payload)
            temporary.replace(target)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            print(f"  attempt {attempt} of {RETRIES} failed: {error}")
    temporary.unlink(missing_ok=True)
    raise RuntimeError(f"the download of {url} failed") from last_error


def load_manifest(path: Path) -> dict:
    """Read the manifest. Return an empty record if no manifest exists."""
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Download every scene file and write the manifest."""
    config.make_dirs()
    raw_dir = config.RAW_DIR
    manifest_path = raw_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    files = manifest.get("files", {})

    wanted: list[tuple[str, str]] = []
    for scene, sources in config.SCENE_SOURCES.items():
        for source in sources:
            wanted.append((scene, source))

    for scene, source in wanted:
        name = f"{source.replace('/', '__')}.txt"
        target = raw_dir / name
        url = config.DATA_URL_TEMPLATE.format(path=source)

        record = files.get(name)
        if target.exists() and record and sha256_of(target) == record["sha256"]:
            print(f"{scene:6s} {name:34s} present, hash matches")
            continue

        print(f"{scene:6s} {name:34s} download")
        fetch(url, target)
        files[name] = {
            "scene": scene,
            "source": source,
            "url": url,
            "sha256": sha256_of(target),
            "bytes": target.stat().st_size,
        }

    manifest = {
        "dataset": "eth_ucy",
        "origin": "https://github.com/Khrylx/AgentFormer",
        "note": (
            "Held-out test split per scene. Column 0 is the frame, column 1 is "
            "the agent id, column 13 is x in metres and column 15 is y in metres."
        ),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(item["bytes"] for item in files.values())
    print(f"\n{len(files)} files, {total / 1e6:.1f} MB, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
