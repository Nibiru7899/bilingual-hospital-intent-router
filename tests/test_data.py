from __future__ import annotations

import copy
import random
import string

import pytest

from ml.common import read_jsonl, write_json, write_jsonl
from ml.data import audit, group_duplicates, load_prepared, normalize, prepare, session_split_groups, verify_partitions


def test_normalization_preserves_hindi_and_is_idempotent():
    normalized = normalize("  OPD\tकहाँ  है? \n")
    assert normalized == "opd कहाँ है?"
    assert normalize(normalized) == normalized


@pytest.mark.parametrize("key,value,reason", [
    ("question", None, "invalid_question"), ("question", "  ", "invalid_question"),
    ("ai_intent", [], "missing_or_invalid_weak_intent"),
    ("id", 12, "invalid_id"), ("consent", "true", "consent_not_true"),
    ("consent", False, "consent_not_true"), ("timestamp", "bad", "invalid_timestamp"),
    ("timestamp", "2026-01-01T12:00:00", "invalid_timestamp"),
    ("consentTimestamp", "2026-01-02T12:00:00Z", "consent_after_interaction"),
    ("question", "Email me at fixture@example.invalid", "potential_pii"),
    ("question", "My number is 9876543210", "potential_pii"),
    ("question", "My name is Fixture Person", "potential_pii"),
])
def test_invalid_records_quarantined(tmp_path, raw_row, key, value, reason):
    raw_row[key] = value
    source = tmp_path / "source.json"
    write_json(source, {"data": [raw_row]})
    report = audit(source, tmp_path / "audit")
    assert report["eligible_rows"] == 0
    assert report["quarantine_reasons"][reason] == 1


def test_audit_duplicate_ids_and_weak_language_anomalies(tmp_path, raw_row):
    other = copy.deepcopy(raw_row)
    other["ai_language"] = "off_topic"
    source = tmp_path / "source.json"
    write_json(source, {"data": [raw_row, other]})
    report = audit(source, tmp_path / "audit")
    assert report["quarantined_rows"] == 2
    assert report["duplicate_id_values"] == 1
    assert "off_topic" in report["undocumented_languages"]


def test_clean_records_exclude_generated_answers(tmp_path, raw_row):
    source = tmp_path / "source.json"
    write_json(source, {"data": [raw_row]})
    report = audit(source, tmp_path / "audit")
    clean = read_jsonl(tmp_path / "audit" / "clean_records.jsonl")[0]
    assert "answer" not in clean
    assert clean["label_provenance"] == "publisher_ai_weak"
    assert clean["annotation_status"] == "pending"
    assert report["source_rows"] == 1
    assert "UNVERIFIED FIXTURE ANSWER" not in (tmp_path / "audit" / "audit.json").read_text()


def test_invalid_container_schema_rejected(tmp_path):
    source = tmp_path / "source.json"
    write_json(source, [])
    with pytest.raises(ValueError, match="data array"):
        audit(source, tmp_path / "audit")


def test_release_date_indexed_schema_and_session_provenance(tmp_path, raw_row):
    raw_row["session_id"] = "fixture-private-session"
    source = tmp_path / "source.json"
    write_json(source, {"data": {"2026-01-01": [raw_row]}})
    result = audit(source, tmp_path / "audit")
    assert result["source_layout"] == "date-indexed data arrays"
    rows = read_jsonl(tmp_path / "audit" / "clean_records.jsonl")
    assert rows[0]["source_session_hash"] != raw_row["session_id"]
    assert len(rows[0]["source_session_hash"]) == 64


def test_session_leakage_is_rejected(prepared):
    rows, _ = load_prepared(prepared)
    rows[0]["source_session_hash"] = "fixture-session"
    next(row for row in rows if row["split"] == "test")["source_session_hash"] = "fixture-session"
    with pytest.raises(ValueError, match="leakage"):
        verify_partitions(rows)


def test_session_and_duplicate_components_are_connected():
    rows = [{"id": "a", "group_id": "duplicate-a", "source_session_hash": "session-a"},
            {"id": "b", "group_id": "duplicate-b", "source_session_hash": "session-a"},
            {"id": "c", "group_id": "duplicate-b", "source_session_hash": "session-b"},
            {"id": "d", "group_id": "duplicate-d", "source_session_hash": "session-d"}]
    components = session_split_groups(rows)
    assert components[0] == components[1] == components[2]
    assert components[3] != components[0]


def test_exact_near_and_unrelated_groups():
    texts = ["What are the opening hours for the hospital outpatient registration counter on Monday?",
             "What are the opening hours for the hospital outpatient registration counter on Monday?!",
             "What are the opening hours for the hospital outpatient registration counter on Monday?",
             "नर्सिंग कोर्स में प्रवेश कैसे मिलेगा"]
    groups = group_duplicates([{"normalized_question": normalize(text)} for text in texts])
    assert groups[0] == groups[1] == groups[2]
    assert groups[0] != groups[3]
    assert groups == group_duplicates([{"normalized_question": normalize(text)} for text in texts])


def test_leakage_and_provenance_rejected(prepared):
    rows, _ = load_prepared(prepared)
    test = next(row for row in rows if row["split"] == "test")
    test["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="leakage"):
        verify_partitions(rows)
    test["group_id"] = "distinct"
    test["label_provenance"] = "human_verified"
    with pytest.raises(ValueError, match="provenance"):
        verify_partitions(rows)


def test_split_tampering_detected(prepared):
    with (prepared / "records.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(prepared)


def test_preparation_conflicts_rare_classes_and_reproducibility(tmp_path):
    rng = random.Random(9)
    records = []
    for label in ("label_a", "label_b"):
        for i in range(30):
            text = " ".join("".join(rng.choices(string.ascii_lowercase, k=14)) for _ in range(4))
            records.append({"id": f"{label}-{i}", "question": text, "normalized_question": text,
                            "weak_intent": label, "weak_language": "en", "label_provenance": "publisher_ai_weak"})
    conflict = {**records[0], "id": "conflict", "weak_intent": "label_b"}
    rare = {**records[-1], "id": "rare", "question": "rare fixture zz", "normalized_question": "rare fixture zz", "weak_intent": "rare"}
    records += [conflict, rare]
    write_jsonl(tmp_path / "audit" / "clean_records.jsonl", records)
    write_json(tmp_path / "audit" / "audit.json", {"source_sha256": "fixture"})
    first = prepare(tmp_path / "audit", tmp_path / "prepared1")
    second = prepare(tmp_path / "audit", tmp_path / "prepared2")
    assert first["records_sha256"] == second["records_sha256"]
    assert first["conflicting_groups"] == 1
    assert first["excluded_class_group_counts"] == {"rare": 1}
    assert first["quarantined_rows"] == 3
    load_prepared(tmp_path / "prepared1")
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path / "audit", tmp_path / "prepared1")
