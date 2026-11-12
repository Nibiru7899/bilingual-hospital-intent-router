"""Offline CLI. All displayed summaries are aggregates; no patient-facing service."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

RAW = Path("data/raw/mediq-v0.6.0")
AUDIT = Path("data/audited/mediq-v0.6.0-v2")
PREPARED = Path("data/prepared/mediq-v0.6.0-v2")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire", help="Fetch and verify pinned MEDIQ source")
    acquire.add_argument("--output", type=Path, default=RAW)
    audit = commands.add_parser("audit", help="Screen schema, consent and possible PII")
    audit.add_argument("--source", type=Path, default=RAW / "mediq_full.json")
    audit.add_argument("--output", type=Path, default=AUDIT)
    prepare = commands.add_parser("prepare", help="Group duplicates and freeze partitions")
    prepare.add_argument("--audit", type=Path, default=AUDIT)
    prepare.add_argument("--output", type=Path, default=PREPARED)
    verify = commands.add_parser("verify", help="Check frozen split checksums and isolation")
    verify.add_argument("--prepared", type=Path, default=PREPARED)
    export = commands.add_parser("annotation-export", help="Export pending reviews; no inferred human labels")
    export.add_argument("--prepared", type=Path, default=PREPARED)
    export.add_argument("--output", type=Path, default=Path("data/annotation/batch-a-v1"))
    export.add_argument("--limit", type=int, default=1000)
    ingest = commands.add_parser("annotation-import", help="Validate a separate, human-completed review copy")
    ingest.add_argument("--csv", type=Path, required=True)
    ingest.add_argument("--export-dir", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Select classical models on validation macro F1")
    train.add_argument("--prepared", type=Path, default=PREPARED)
    train.add_argument("--output", type=Path, default=Path("artifacts/batch-a-v1"))
    train.add_argument("--model", choices=["logistic", "svm", "both"], default="both")
    evaluate = commands.add_parser("evaluate", help="Evaluate a selected model once on frozen test")
    evaluate.add_argument("--prepared", type=Path, default=PREPARED)
    evaluate.add_argument("--run", type=Path, required=True)
    predict = commands.add_parser("predict", help="Predict broad source intent using a trusted local artifact")
    predict.add_argument("--run", type=Path, required=True)
    predict.add_argument("--text", required=True)
    return root


def execute(args: argparse.Namespace) -> dict:
    if args.command == "acquire":
        from ml.acquisition import acquire
        return acquire(args.output)
    if args.command == "audit":
        from ml.data import audit
        result = audit(args.source, args.output)
        return {key: result[key] for key in ("source_rows", "eligible_rows", "quarantined_rows", "quarantine_reasons", "weak_intent_counts", "weak_language_counts", "exact_duplicate_extra_rows")}
    if args.command == "prepare":
        from ml.data import prepare
        result = prepare(args.audit, args.output)
        return {key: result[key] for key in ("split_counts", "class_counts", "retained_groups", "conflicting_groups", "excluded_class_group_counts", "quarantined_rows", "records_sha256")}
    if args.command == "verify":
        from ml.data import load_prepared
        rows, manifest = load_prepared(args.prepared)
        return {"verified_rows": len(rows), "records_sha256": manifest["records_sha256"], "duplicate_group_leakage": False}
    if args.command == "annotation-export":
        from ml.annotation import export_annotations
        return export_annotations(args.prepared, args.output, args.limit)
    if args.command == "annotation-import":
        from ml.annotation import import_annotations
        return import_annotations(args.csv, args.export_dir, args.output)
    if args.command == "train":
        from ml.baselines import train
        from ml.common import require_new
        models = ("logistic", "svm") if args.model == "both" else (args.model,)
        for model in models:
            require_new(args.output / model)
        results = {}
        for model in models:
            manifest = train(args.prepared, args.output / model, model)
            results[model] = {key: manifest[key] for key in ("config", "training_seconds_including_selection", "artifact_bytes", "validation_trials")}
        return results
    if args.command == "evaluate":
        from ml.baselines import evaluate
        return evaluate(args.prepared, args.run)
    if args.command == "predict":
        from ml.baselines import predict
        return predict(args.run, args.text)
    raise ValueError("Unknown command")


def main() -> int:
    args = parser().parse_args()
    try:
        result = execute(args)
    except (ValueError, OSError, urllib.error.URLError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
