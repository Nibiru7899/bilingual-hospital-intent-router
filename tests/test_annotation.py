from __future__ import annotations

import csv

import pytest

from ml.annotation import FIELDS, export_annotations, import_annotations, safe_cell
from ml.common import read_jsonl, write_csv


def review_copy(export, path, complete=True):
    with (export / "annotation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if complete:
        for row in rows:
            row.update(status="reviewed", primary_reviewer="fixture-reviewer-a", reviewed_at="2026-01-02T12:00:00Z")
            prefixes = ["primary", "final"]
            if row["independent_review_required"] == "yes":
                row["secondary_reviewer"] = "fixture-reviewer-b"
                prefixes.append("secondary")
            for prefix in prefixes:
                row.update({f"{prefix}_intent": "OTHER_UNSUPPORTED", f"{prefix}_safety": "administrative", f"{prefix}_ambiguity": "clear", f"{prefix}_language": "en"})
    write_csv(path, rows, FIELDS)
    return rows


def test_pending_export_is_not_human_verified(prepared, tmp_path):
    export = tmp_path / "export"
    info = export_annotations(prepared, export, 20)
    assert info["independent_review_count"] == 4
    copy = tmp_path / "copy.csv"
    review_copy(export, copy, complete=False)
    report = import_annotations(copy, export, tmp_path / "import")
    assert report["human_reviewed_count"] == 0
    assert report["agreement"] == {}
    assert read_jsonl(tmp_path / "import" / "reviewed_records.jsonl") == []


def test_valid_review_and_agreement(prepared, tmp_path):
    export = tmp_path / "export"
    export_annotations(prepared, export, 20)
    review_copy(export, tmp_path / "copy.csv")
    result = import_annotations(tmp_path / "copy.csv", export, tmp_path / "import")
    assert result["human_reviewed_count"] == 20
    assert result["agreement"]["intent"]["paired_count"] == 4
    assert result["agreement"]["intent"]["raw_agreement"] == 1
    assert result["agreement"]["intent"]["cohen_kappa"] is None


@pytest.mark.parametrize("mutation,match", [
    ("text", "Immutable"), ("split", "Immutable"), ("identity", "different person"),
    ("missing_secondary", "review is missing"), ("invalid_label", "intent"),
    ("disagreement", "adjudication"), ("false_pending", "Pending rows"),
    ("missing_primary", "reviewer identity"), ("timestamp", "timezone"),
    ("duplicate", "exactly the original"),
])
def test_invalid_review_rejected(prepared, tmp_path, mutation, match):
    export = tmp_path / "export"
    export_annotations(prepared, export, 20)
    copy = tmp_path / "copy.csv"
    rows = review_copy(export, copy)
    row = next(item for item in rows if item["independent_review_required"] == "yes")
    if mutation == "text": row["question"] += " edited"
    elif mutation == "split": row["split"] = "unfrozen"
    elif mutation == "identity": row["secondary_reviewer"] = row["primary_reviewer"]
    elif mutation == "missing_secondary": row["secondary_reviewer"] = ""
    elif mutation == "invalid_label": row["final_intent"] = "fake_label"
    elif mutation == "disagreement": row["secondary_intent"] = "GREETING"
    elif mutation == "false_pending": row["status"] = "pending"
    elif mutation == "missing_primary": row["primary_reviewer"] = ""
    elif mutation == "timestamp": row["reviewed_at"] = "yesterday"
    elif mutation == "duplicate": rows.append(rows[0].copy())
    write_csv(copy, rows, FIELDS)
    with pytest.raises(ValueError, match=match):
        import_annotations(copy, export, tmp_path / "import")


def test_disagreement_requires_independent_adjudication(prepared, tmp_path):
    export = tmp_path / "export"
    export_annotations(prepared, export, 20)
    copy = tmp_path / "copy.csv"
    rows = review_copy(export, copy)
    row = next(item for item in rows if item["independent_review_required"] == "yes")
    row.update(status="adjudicated", secondary_intent="GREETING", adjudicator="fixture-reviewer-c", notes="Fixture resolution only")
    write_csv(copy, rows, FIELDS)
    assert import_annotations(copy, export, tmp_path / "import")["human_reviewed_count"] == 20


def test_formula_escape():
    assert safe_cell("=1+2") == "'=1+2"
    assert safe_cell("  @SUM(A1)").startswith("'")
    assert safe_cell("ordinary question") == "ordinary question"
