"""All records in this directory are synthetic software fixtures, never research evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml.common import sha256, write_json, write_jsonl
from ml.data import normalize


@pytest.fixture
def raw_row() -> dict:
    return {"id": "fixture-1", "question": "Where is cardiology reception?", "timestamp": "2026-01-01T12:00:00Z", "consent": True, "consentTimestamp": "2026-01-01T11:59:00Z", "ai_intent": "navigation", "ai_language": "en", "language_detected": "en", "answer": "UNVERIFIED FIXTURE ANSWER"}


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    rows = []
    for split, n in [("train", 16), ("validation", 6), ("test", 6)]:
        for label, keyword in [("alpha", "registration administrative"), ("beta", "invoice payment")]:
            for index in range(n):
                text = f"{keyword} {split}onlyword row{index}"
                rows.append({"id": f"fixture-{split}-{label}-{index}", "question": text,
                             "normalized_question": normalize(text), "weak_intent": label,
                             "weak_language": "en" if index % 2 else "hi-Latn", "split": split,
                             "label_provenance": "publisher_ai_weak", "annotation_status": "pending",
                             "group_id": hashlib.sha256(text.encode()).hexdigest()[:20]})
    write_jsonl(root / "records.jsonl", rows)
    write_json(root / "split_manifest.json", {"records_sha256": sha256(root / "records.jsonl"), "source_sha256": "synthetic_fixture_only"})
    return root
