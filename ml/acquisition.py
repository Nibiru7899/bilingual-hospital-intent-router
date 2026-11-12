"""Acquire only the immutable MEDIQ release; never execute publisher code."""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

from ml.common import now, read_json, sha256, write_json

COMMIT = "c8226fc9662003270146e6130dccaf1d586d6eae"
VERSION = "v0.6.0"
DATA_BLOB_SHA1 = "d6c3ad25665c20b9da025f282e5e1e7b5dde5fa7"
DATA_SHA256 = "d68b19c8efc8376f2c03ac2bf0ad9ab3e43ba2b5ec33678e4d50e3ddca286aa4"
BASE = f"https://raw.githubusercontent.com/ProxyAyush/MEDIQ/{COMMIT}"


def download(url: str, target: Path, limit: int = 100_000_000) -> None:
    """Bound both socket inactivity and total download duration; leave no final partial file."""
    request = urllib.request.Request(url, headers={"User-Agent": "MedRoute-AI-research/0.1"})
    partial = target.with_suffix(target.suffix + ".partial")
    started = time.monotonic()
    size = 0
    with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > limit or time.monotonic() - started > 300:
                raise ValueError("Download exceeded the 100 MB / 300 second resource limit")
            stream.write(chunk)
    partial.replace(target)


def acquire(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["commit"] != COMMIT:
            raise ValueError("Source manifest does not match pinned MEDIQ commit")
        for item in manifest["files"]:
            if sha256(output / item["name"]) != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {item['name']}")
        if sha256(output / "mediq_full.json") != DATA_SHA256:
            raise ValueError("Dataset checksum differs from pinned release")
        return manifest

    files = []
    for remote, local in [
        ("LICENSE", "LICENSE"), ("DATASHEET.md", "DATASHEET.md"),
        ("README.md", "README.md"), ("data/mediq_full.json", "mediq_full.json"),
    ]:
        target = output / local
        # Existing files without a completed manifest are checked, never blindly overwritten.
        if not target.exists():
            download(f"{BASE}/{remote}", target)
        if local == "LICENSE" and "Attribution 4.0" not in target.read_text(encoding="utf-8"):
            raise ValueError("Expected CC BY 4.0 license not found")
        if local == "mediq_full.json":
            blob = hashlib.sha1(f"blob {target.stat().st_size}\0".encode())
            with target.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    blob.update(chunk)
            if blob.hexdigest() != DATA_BLOB_SHA1:
                raise ValueError("Dataset differs from the verified release Git blob")
            if sha256(target) != DATA_SHA256:
                raise ValueError("Dataset checksum differs from pinned release")
        files.append({"name": local, "url": f"{BASE}/{remote}", "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {
        "dataset": "MEDIQ", "version": VERSION, "commit": COMMIT,
        "retrieved_at_utc": now(), "license": "CC-BY-4.0",
        "attribution": "Dr. Ayush Yadav (2026), MEDIQ, https://github.com/ProxyAyush/MEDIQ",
        "data_git_blob_sha1": DATA_BLOB_SHA1, "files": files,
        "label_provenance": "publisher_ai_weak", "answers_verified": False,
    }
    write_json(manifest_path, manifest)
    return manifest
