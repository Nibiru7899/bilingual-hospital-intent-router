"""Exportable human annotation workflow, isolated from publisher weak labels."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from ml.common import now, read_json, require_new, sha256, write_csv, write_json, write_jsonl
from ml.data import SEED, load_prepared

INTENTS = {
    "OPD_TIMINGS", "DOCTOR_AVAILABILITY", "APPOINTMENT_BOOKING", "APPOINTMENT_RESCHEDULING",
    "APPOINTMENT_CANCELLATION", "DEPARTMENT_NAVIGATION", "REPORT_STATUS", "BILLING_INSURANCE",
    "PRE_VISIT_INFORMATION", "FOLLOW_UP_INFORMATION", "HOSPITAL_SERVICES", "COMPLAINT",
    "GREETING", "OTHER_UNSUPPORTED",
}
SAFETY = {"administrative", "clinical_advice", "potential_emergency", "sensitive_personal_data", "uncertain"}
AMBIGUITY = {"clear", "underspecified", "multiple_intents", "uninterpretable"}
LANGUAGES = {"en", "hi", "hi-Latn", "hi-en", "hi-Latn-en", "other", "uncertain"}
DIMENSIONS = {"intent": INTENTS, "safety": SAFETY, "ambiguity": AMBIGUITY, "language": LANGUAGES}
BASE_FIELDS = ["id", "group_id", "split", "question", "independent_review_required"]
REVIEW_FIELDS = [f"{prefix}_{dimension}" for prefix in ("primary", "secondary", "final") for dimension in DIMENSIONS]
FIELDS = BASE_FIELDS + ["status", "primary_reviewer", "secondary_reviewer", "adjudicator", "reviewed_at", "notes"] + REVIEW_FIELDS


def safe_cell(text: str) -> str:
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r")) else text


def export_annotations(prepared: Path, output: Path, limit: int = 1000) -> dict:
    require_new(output)
    if not 1 <= limit <= 1000:
        raise ValueError("Annotation export size must be between 1 and 1000")
    records, source = load_prepared(prepared)
    rng = random.Random(SEED)
    # One representative per duplicate group, sampled round-robin across split/label/language.
    rng.shuffle(records)
    buckets: dict[tuple, list] = defaultdict(list)
    used = set()
    for row in records:
        if row["group_id"] not in used:
            buckets[(row["split"], row["weak_intent"], row["weak_language"])].append(row)
            used.add(row["group_id"])
    selected = []
    while buckets and len(selected) < limit:
        for key in sorted(list(buckets)):
            selected.append(buckets[key].pop())
            if not buckets[key]:
                del buckets[key]
            if len(selected) == limit:
                break
    second = set(rng.sample([row["id"] for row in selected], max(1, round(len(selected) * .2))))
    rows = []
    for item in selected:
        row = {field: "" for field in FIELDS}
        row.update({key: item[key] for key in ("id", "group_id", "split")})
        row.update(question=safe_cell(item["question"]), independent_review_required="yes" if item["id"] in second else "no", status="pending")
        rows.append(row)
    write_csv(output / "annotation.csv", rows, FIELDS)
    # Immutable metadata binds imports to this exact export, without trusting reviewer edits.
    manifest = {
        "created_at_utc": now(), "prepared_records_sha256": source["records_sha256"],
        "export_sha256": sha256(output / "annotation.csv"), "count": len(rows),
        "independent_review_count": len(second), "status": "pending_human_review",
        "records": {row["id"]: {key: row[key] for key in BASE_FIELDS} for row in rows},
        "taxonomy": {key: sorted(values) for key, values in DIMENSIONS.items()},
    }
    write_json(output / "export_manifest.json", manifest)
    return {key: manifest[key] for key in ("count", "independent_review_count", "status")}


def import_annotations(csv_path: Path, export_dir: Path, output: Path) -> dict:
    require_new(output)
    manifest = read_json(export_dir / "export_manifest.json")
    if sha256(export_dir / "annotation.csv") != manifest["export_sha256"]:
        raise ValueError("Original export changed; edit a separate review copy")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Annotation CSV columns do not match the workflow")
        rows = list(reader)
    if len({row["id"] for row in rows}) != len(rows) or {row["id"] for row in rows} != set(manifest["records"]):
        raise ValueError("Review copy must contain exactly the original IDs, without duplication")
    verified = []
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    status_counts: Counter = Counter()
    for row in rows:
        if any(row[key] != manifest["records"][row["id"]][key] for key in BASE_FIELDS):
            raise ValueError("Immutable annotation ID, text, split, group or review assignment changed")
        if row["status"] not in {"pending", "reviewed", "adjudicated"}:
            raise ValueError("Unknown annotation status")
        status_counts[row["status"]] += 1
        has_primary = bool(row["primary_reviewer"].strip())
        has_secondary = bool(row["secondary_reviewer"].strip())
        if row["status"] == "pending":
            if any(row[field] for field in REVIEW_FIELDS) or has_primary or has_secondary or row["adjudicator"] or row["reviewed_at"]:
                raise ValueError("Pending rows cannot contain completed review fields")
            continue
        if not has_primary or not row["reviewed_at"].strip():
            raise ValueError("Completed review requires reviewer identity and timestamp")
        from ml.data import timestamp_valid
        if not timestamp_valid(row["reviewed_at"]):
            raise ValueError("Review timestamp must include a timezone")
        required = row["independent_review_required"] == "yes"
        if required and not has_secondary:
            raise ValueError("Assigned independent review is missing")
        if has_secondary and row["secondary_reviewer"].strip().casefold() == row["primary_reviewer"].strip().casefold():
            raise ValueError("Independent reviewer must be a different person")
        disagreement = False
        for dimension, values in DIMENSIONS.items():
            primary, secondary, final = (row[f"{prefix}_{dimension}"] for prefix in ("primary", "secondary", "final"))
            if primary not in values or final not in values or (has_secondary and secondary not in values):
                raise ValueError(f"Invalid or missing {dimension} annotation")
            if not has_secondary and secondary:
                raise ValueError("Secondary label has no reviewer")
            if has_secondary:
                pairs[dimension].append((primary, secondary))
                disagreement |= primary != secondary
            if row["status"] == "reviewed" and (final != primary or (has_secondary and final != secondary)):
                raise ValueError("Disagreements require adjudication")
        if row["status"] == "adjudicated":
            if not has_secondary or not disagreement or not row["notes"].strip():
                raise ValueError("Adjudication requires two disagreeing reviews and resolution notes")
            if not row["adjudicator"].strip() or row["adjudicator"].strip().casefold() in {row["primary_reviewer"].strip().casefold(), row["secondary_reviewer"].strip().casefold()}:
                raise ValueError("Adjudication requires an independent third reviewer")
        elif row["adjudicator"]:
            raise ValueError("Adjudicator supplied without adjudicated status")
        verified.append({**row, "label_provenance": "human_reviewed", "review_count": 2 if has_secondary else 1})
    agreement = {}
    for dimension, pair in pairs.items():
        left, right = zip(*pair)
        kappa = float(cohen_kappa_score(left, right)) if len(set(left + right)) > 1 else None
        agreement[dimension] = {"paired_count": len(pair), "raw_agreement": float(np.mean(np.array(left) == np.array(right))), "cohen_kappa": kappa}
    report = {
        "created_at_utc": now(), "review_file_sha256": sha256(csv_path),
        "export_manifest_sha256": sha256(export_dir / "export_manifest.json"),
        "status_counts": dict(status_counts), "human_reviewed_count": len(verified), "agreement": agreement,
        "limitation": "Reviewer identities are declarations, not authenticated attestations. Independent annotations must be collected blind in separate copies before merging.",
    }
    write_jsonl(output / "reviewed_records.jsonl", verified)
    write_json(output / "review_report.json", report)
    return report
