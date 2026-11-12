"""Copy aggregate evidence and freeze dependencies; never publish raw questions/IDs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ml.baselines import environment
from ml.common import read_json, sha256, write_json


def main() -> None:
    root = Path("artifacts/batch-a-v1")
    root.mkdir(parents=True, exist_ok=True)
    raw = Path("data/raw/mediq-v0.6.0")
    audit = Path("data/audited/mediq-v0.6.0-v2")
    prepared = Path("data/prepared/mediq-v0.6.0-v2")
    write_json(root / "source_manifest.json", read_json(raw / "source_manifest.json"))
    write_json(root / "data_audit.json", read_json(audit / "audit.json"))
    splits = read_json(prepared / "split_manifest.json")
    splits.pop("split_ids")
    write_json(root / "split_summary.json", {**splits, "full_manifest_sha256": sha256(prepared / "split_manifest.json")})
    export = read_json(Path("data/annotation/batch-a-v1/export_manifest.json"))
    export.pop("records")
    write_json(root / "annotation_summary.json", export)
    write_json(root / "environment.json", environment())
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True, check=True).stdout
    Path("requirements.lock.txt").write_text("# Resolved Batch A environment; Python 3.12 on Windows.\n" + lock, encoding="utf-8")
    print("Saved aggregate evidence and requirements.lock.txt; no raw text/IDs copied.")


if __name__ == "__main__":
    main()
