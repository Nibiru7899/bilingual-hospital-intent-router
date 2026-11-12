"""Verify the saved Batch A evidence chain without refitting or re-evaluating models."""

from __future__ import annotations

from pathlib import Path

from ml.acquisition import acquire
from ml.baselines import source_hashes
from ml.common import read_json, sha256
from ml.data import load_prepared


def main() -> None:
    root = Path("artifacts/batch-a-v1")
    acquire(Path("data/raw/mediq-v0.6.0"))
    prepared = Path("data/prepared/mediq-v0.6.0-v2")
    rows, splits = load_prepared(prepared)
    summary = read_json(root / "split_summary.json")
    if sha256(prepared / "split_manifest.json") != summary["full_manifest_sha256"]:
        raise ValueError("Aggregate evidence differs from frozen split manifest")
    for kind in ("logistic", "svm"):
        run = root / kind
        manifest = read_json(run / "experiment_manifest.json")
        if manifest["code_sha256"] != source_hashes():
            raise ValueError("Current ML code differs from trained source snapshot")
        if sha256(run / "model.joblib") != manifest["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        if manifest["prepared_records_sha256"] != splits["records_sha256"]:
            raise ValueError("Model data identity mismatch")
        if manifest["split_manifest_sha256"] != summary["full_manifest_sha256"]:
            raise ValueError("Model split identity mismatch")
        evaluation = read_json(run / "evaluation" / "evaluation_manifest.json")
        if evaluation["model_sha256"] != manifest["model_sha256"] or evaluation["records_sha256"] != splits["records_sha256"]:
            raise ValueError("Evaluation identity mismatch")
        for filename, checksum in evaluation["outputs_sha256"].items():
            if sha256(run / "evaluation" / filename) != checksum:
                raise ValueError("Evaluation output checksum mismatch")
    print(f"Verified pinned raw source, {len(rows)} prepared rows, trained code, both model identities, frozen split identity and every saved evaluation output checksum.")


if __name__ == "__main__":
    main()
