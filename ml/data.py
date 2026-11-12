"""Schema auditing, cautious text preparation, duplicate grouping and frozen splits."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from ml.common import now, read_json, read_jsonl, require_new, sha256, write_json, write_jsonl

NEAR_THRESHOLD = 0.92
SEED = 42
LABEL_PROVENANCE = "publisher_ai_weak"
EXPECTED_INTENTS = {"information_seeking", "navigation", "scheduling", "complaint", "greeting", "feedback", "off_topic"}
EXPECTED_LANGUAGES = {"en", "hi", "hi-Latn", "hi-en", "hi-Latn-en"}
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
    "phone_or_identifier": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)|(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)"),
    "private_record_reference": re.compile(r"\b(?:aadhaar|aadhar|patient\s*id|uhid|mrn)\s*[:#-]?\s*\d{4,}\b", re.I),
    "self_identification": re.compile(r"\b(?:my name is|mera naam|मेरा नाम)\s+\S+", re.I),
}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def timestamp_valid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def audit(source: Path, output: Path) -> dict:
    require_new(output)
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("Expected MEDIQ JSON object with a data array or date-indexed arrays")
    container = payload.get("data")
    if isinstance(container, list):
        raw = container
        source_layout = "flat data array"
    elif isinstance(container, dict) and all(isinstance(rows, list) for rows in container.values()):
        raw = [row for day in sorted(container) for row in container[day]]
        source_layout = "date-indexed data arrays"
    else:
        raise ValueError("Expected MEDIQ JSON object with a data array or date-indexed arrays")
    ids = Counter(row.get("id") for row in raw if isinstance(row, dict) and isinstance(row.get("id"), str))
    reasons: Counter = Counter()
    pii_counts: Counter = Counter()
    fields: Counter = Counter()
    missing: Counter = Counter()
    field_types: dict[str, Counter] = defaultdict(Counter)
    labels: Counter = Counter()
    languages: Counter = Counter()
    heuristic_languages: Counter = Counter()
    categories: Counter = Counter()
    medical_relevance: Counter = Counter()
    timestamps = []
    clean, quarantine = [], []
    for index, row in enumerate(raw):
        failures = []
        if not isinstance(row, dict):
            quarantine.append({"source_row": index, "reasons": ["not_object"]})
            reasons["not_object"] += 1
            continue
        for key, value in row.items():
            fields[key] += 1
            field_types[key][type(value).__name__] += 1
        for key in ("id", "question", "timestamp", "consent", "consentTimestamp", "ai_intent", "ai_language"):
            if row.get(key) is None or row.get(key) == "":
                missing[key] += 1
        identifier = row.get("id")
        question = row.get("question")
        intent = row.get("ai_intent")
        language = row.get("ai_language")
        labels[str(intent)] += 1
        languages[str(language)] += 1
        heuristic_languages[str(row.get("language_detected"))] += 1
        categories[str(row.get("category"))] += 1
        medical_relevance[str(row.get("ai_medical_relevance"))] += 1
        if not isinstance(identifier, str) or not identifier.strip():
            failures.append("invalid_id")
        elif ids[identifier] != 1:
            failures.append("duplicate_id")
        if not isinstance(question, str) or not normalize(question) or len(question) > 10_000:
            failures.append("invalid_question")
        if not isinstance(intent, str) or not intent.strip():
            failures.append("missing_or_invalid_weak_intent")
        if row.get("consent") is not True:
            failures.append("consent_not_true")
        if not timestamp_valid(row.get("timestamp")) or not timestamp_valid(row.get("consentTimestamp")):
            failures.append("invalid_timestamp")
        else:
            interaction = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            consent = datetime.fromisoformat(row["consentTimestamp"].replace("Z", "+00:00"))
            timestamps.append(interaction.isoformat())
            if consent > interaction:
                failures.append("consent_after_interaction")
        pii = [name for name, pattern in PII_PATTERNS.items() if isinstance(question, str) and pattern.search(question)]
        pii_counts.update(pii)
        if pii:
            failures.append("potential_pii")
        if failures:
            reasons.update(failures)
            quarantine.append({"source_row": index, "id": identifier if isinstance(identifier, str) else None, "reasons": failures, "pii_flags": pii})
            continue
        clean.append({
            "id": identifier, "question": question, "normalized_question": normalize(question),
            "weak_intent": intent.strip(), "weak_language": language if isinstance(language, str) and language else "unknown",
            "heuristic_language": row.get("language_detected"), "timestamp": row["timestamp"],
            "label_provenance": LABEL_PROVENANCE, "annotation_status": "pending",
            "source_session_hash": hashlib.sha256(row["session_id"].encode()).hexdigest() if isinstance(row.get("session_id"), str) and row["session_id"].strip() else None,
        })
    duplicate_counts = Counter(row["normalized_question"] for row in clean)
    report = {
        "created_at_utc": now(), "source_sha256": sha256(source), "source_rows": len(raw),
        "publisher_total_entries": payload.get("totalEntries"), "eligible_rows": len(clean),
        "source_layout": source_layout,
        "quarantined_rows": len(quarantine), "quarantine_reasons": dict(reasons),
        "observed_fields": dict(fields), "field_types": {key: dict(value) for key, value in field_types.items()},
        "missing_required_or_research_fields": dict(missing), "weak_intent_counts": dict(labels),
        "weak_language_counts": dict(languages), "heuristic_language_counts": dict(heuristic_languages),
        "source_category_counts_unverified": dict(categories),
        "source_medical_relevance_counts_unverified": dict(medical_relevance),
        "hospital_subset_policy": "OPD/appointment/emergency source categories are candidate-only; no hospital-specific reference labels are inferred. Human review is required.",
        "undocumented_intents": sorted(set(labels) - EXPECTED_INTENTS),
        "undocumented_languages": sorted(set(languages) - EXPECTED_LANGUAGES),
        "potential_pii_flags": dict(pii_counts), "duplicate_id_values": sum(n > 1 for n in ids.values()),
        "exact_duplicate_extra_rows": sum(n - 1 for n in duplicate_counts.values()),
        "timestamp_range": [min(timestamps), max(timestamps)] if timestamps else [],
        "limitations": [
            "PII patterns are a screening aid, not proof of anonymization; names, spaced phones and clinical details may remain.",
            "Publisher AI labels and language categories are unverified weak supervision.",
            "Hospital relevance requires human annotation; medical-university records are not all hospital enquiries.",
            "No original answers are retained in research-ready records or treated as verified facts.",
        ],
    }
    write_jsonl(output / "clean_records.jsonl", clean)
    report["clean_records_sha256"] = sha256(output / "clean_records.jsonl")
    write_jsonl(output / "quarantine.jsonl", quarantine)
    write_json(output / "audit.json", report)
    return report


def group_duplicates(records: list[dict]) -> list[str]:
    """Label-blind hashing features avoid fitting a research vocabulary on held-out data."""
    texts = sorted({row["normalized_question"] for row in records})
    if not texts:
        return []
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # Nonnegative, normalized hashing yields sparse cosine similarity. Hash collisions and
    # transitive grouping are documented limitations; this is not semantic paraphrase detection.
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**18,
                                   alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32)
    matrix = vectorizer.transform(texts)
    for start in range(0, len(texts), 128):
        similarity = (matrix[start:start + 128] @ matrix.T).tocoo()
        for local, other, value in zip(similarity.row, similarity.col, similarity.data):
            current = start + int(local)
            if current < other and value >= NEAR_THRESHOLD:
                left, right = find(current), find(int(other))
                parent[max(left, right)] = min(left, right)
    group_by_text = {text: hashlib.sha256(texts[find(i)].encode("utf-8")).hexdigest()[:20] for i, text in enumerate(texts)}
    return [group_by_text[row["normalized_question"]] for row in records]


def session_split_groups(records: list[dict]) -> list[str]:
    """Keep both duplicate components and known source sessions inside one partition."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple, int] = {}
    for index, row in enumerate(records):
        keys = [("duplicate", row["group_id"])]
        if row.get("source_session_hash"):
            keys.append(("session", row["source_session_hash"]))
        for key in keys:
            if key in seen:
                left, right = find(index), find(seen[key])
                parent[max(left, right)] = min(left, right)
            seen[key] = index
    return [hashlib.sha256(records[find(i)]["id"].encode()).hexdigest()[:20] for i in range(len(records))]


def verify_partitions(records: list[dict]) -> None:
    if not records:
        raise ValueError("Empty prepared dataset")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Repeated source IDs in partitions")
    group_splits: dict[str, set] = defaultdict(set)
    text_splits: dict[str, set] = defaultdict(set)
    labels_by_split: dict[str, set] = defaultdict(set)
    session_splits: dict[str, set] = defaultdict(set)
    for row in records:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if row["label_provenance"] != LABEL_PROVENANCE:
            raise ValueError("Unexpected baseline label provenance")
        group_splits[row["group_id"]].add(row["split"])
        if row.get("split_group_id"):
            group_splits[row["split_group_id"]].add(row["split"])
        if row.get("source_session_hash"):
            session_splits[row["source_session_hash"]].add(row["split"])
        text_splits[row["normalized_question"]].add(row["split"])
        labels_by_split[row["split"]].add(row["weak_intent"])
    if any(len(splits) != 1 for splits in list(group_splits.values()) + list(text_splits.values()) + list(session_splits.values())):
        raise ValueError("Duplicate leakage across partitions")
    if not (labels_by_split["train"] == labels_by_split["validation"] == labels_by_split["test"]):
        raise ValueError("Class missing from one or more partitions")


def prepare(audit_dir: Path, output: Path) -> dict:
    require_new(output)
    records = read_jsonl(audit_dir / "clean_records.jsonl")
    audit_report = read_json(audit_dir / "audit.json")
    if audit_report.get("clean_records_sha256") and sha256(audit_dir / "clean_records.jsonl") != audit_report["clean_records_sha256"]:
        raise ValueError("Audited records checksum mismatch")
    if not records:
        raise ValueError("No eligible records after audit")
    records.sort(key=lambda row: row["id"])
    groups = group_duplicates(records)
    group_labels: dict[str, set] = defaultdict(set)
    for row, group in zip(records, groups):
        row["group_id"] = group
        group_labels[group].add(row["weak_intent"])
    conflicts = {group for group, labels in group_labels.items() if len(labels) > 1}
    label_groups: dict[str, set] = defaultdict(set)
    for row in records:
        if row["group_id"] not in conflicts:
            label_groups[row["weak_intent"]].add(row["group_id"])
    # Ten distinct groups permit seven folds and a subsequent two-fold holdout split.
    excluded = {label: len(group_ids) for label, group_ids in label_groups.items() if len(group_ids) < 10}
    selected, quarantine = [], []
    for row in records:
        if row["group_id"] in conflicts or row["weak_intent"] in excluded:
            quarantine.append({"id": row["id"], "group_id": row["group_id"], "reason": "conflicting_group_labels" if row["group_id"] in conflicts else "insufficient_class_groups"})
        else:
            selected.append(row)
    if len({row["weak_intent"] for row in selected}) < 2:
        raise ValueError("Fewer than two trainable weak-label classes remain")
    for row, split_group in zip(selected, session_split_groups(selected)):
        row["split_group_id"] = split_group
    # One of seven folds is test (~14%); one of six remaining folds is validation.
    y = np.array([row["weak_intent"] for row in selected])
    g = np.array([row["split_group_id"] for row in selected])
    remaining, test = next(StratifiedGroupKFold(7, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, g))
    train_local, validation_local = next(StratifiedGroupKFold(6, shuffle=True, random_state=SEED).split(np.zeros(len(remaining)), y[remaining], g[remaining]))
    for split, indices in [("train", remaining[train_local]), ("validation", remaining[validation_local]), ("test", test)]:
        for index in indices:
            selected[int(index)]["split"] = split
    verify_partitions(selected)
    write_jsonl(output / "records.jsonl", selected)
    write_jsonl(output / "quarantine.jsonl", quarantine)
    split_counts = Counter(row["split"] for row in selected)
    manifest = {
        "created_at_utc": now(), "seed": SEED, "audit_sha256": sha256(audit_dir / "audit.json"),
        "input_records_sha256": sha256(audit_dir / "clean_records.jsonl"),
        "records_sha256": sha256(output / "records.jsonl"), "source_sha256": read_json(audit_dir / "audit.json")["source_sha256"],
        "label_provenance": LABEL_PROVENANCE, "near_duplicate_threshold": NEAR_THRESHOLD,
        "grouping": "NFC/casefold/whitespace exact; hashed char 3-5 cosine >= 0.92 connected components",
        "grouping_limitations": "Lexical only; hash collisions and transitive chains possible. No semantic paraphrase independence claim. Known source sessions are isolated.",
        "split_grouping": "Connected components of duplicate groups and hashed source sessions",
        "target_fractions": {"train": .70, "validation": .15, "test": .15},
        "split_method": "First seeded StratifiedGroupKFold(7) test, then first StratifiedGroupKFold(6) validation",
        "split_counts": dict(split_counts), "actual_fractions": {key: value / len(selected) for key, value in split_counts.items()},
        "class_counts": dict(Counter(row["weak_intent"] for row in selected)),
        "class_split_counts": {split: dict(Counter(row["weak_intent"] for row in selected if row["split"] == split)) for split in ("train", "validation", "test")},
        "retained_groups": len({row["group_id"] for row in selected}), "conflicting_groups": len(conflicts),
        "retained_split_groups": len(set(g)), "largest_split_group_rows": max(Counter(g).values()),
        "excluded_class_group_counts": excluded, "quarantined_rows": len(quarantine),
        "quarantine_reason_counts": dict(Counter(row["reason"] for row in quarantine)),
        "split_ids": {split: [row["id"] for row in selected if row["split"] == split] for split in ("train", "validation", "test")},
    }
    write_json(output / "split_manifest.json", manifest)
    return manifest


def load_prepared(path: Path) -> tuple[list[dict], dict]:
    manifest = read_json(path / "split_manifest.json")
    if sha256(path / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("Frozen split checksum mismatch")
    records = read_jsonl(path / "records.jsonl")
    verify_partitions(records)
    return records, manifest
