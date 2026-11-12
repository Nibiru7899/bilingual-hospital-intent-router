"""Build a human-readable comparison exclusively from completed, local run artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from ml.common import read_json, write_json


def main() -> None:
    root = Path("artifacts/batch-a-v1")
    splits = read_json(root / "split_summary.json")
    audit = read_json(root / "data_audit.json")
    source = read_json(root / "source_manifest.json")
    annotation = read_json(root / "annotation_summary.json")
    results = {}
    for kind in ("logistic", "svm"):
        run = root / kind
        results[kind] = {"manifest": read_json(run / "experiment_manifest.json"), "metrics": read_json(run / "evaluation" / "metrics.json"), "languages": read_json(run / "evaluation" / "language_metrics.json")}
    majority_label = max(splits["class_split_counts"]["train"], key=splits["class_split_counts"]["train"].get)
    majority_accuracy = splits["class_split_counts"]["test"][majority_label] / splits["split_counts"]["test"]
    summary = {"source_sha256": splits["source_sha256"], "prepared_records_sha256": splits["records_sha256"], "label_provenance": "publisher_ai_weak", "test_n": splits["split_counts"]["test"], "train_majority_label": majority_label, "constant_train_majority_test_accuracy": majority_accuracy, "models": {kind: {"selected_C": item["manifest"]["config"]["selected_C"], "model_sha256": item["manifest"]["model_sha256"], **item["metrics"]} for kind, item in results.items()}}
    write_json(root / "summary.json", summary)
    lines = [
        "# Batch A measured baseline report", "", "## Data and protocol", "",
        f"Source: MEDIQ {source['version']} at `{source['commit']}` (CC BY 4.0).",
        f"Source SHA-256: `{splits['source_sha256']}`.",
        f"Prepared records SHA-256: `{splits['records_sha256']}`.", "",
        f"Audited {audit['source_rows']:,} source rows; {audit['quarantined_rows']:,} excluded by schema/consent/PII screening.",
        f"Preparation excluded a further {splits['quarantined_rows']:,} rows: {splits['quarantine_reason_counts']}.",
        f"Retained {sum(splits['split_counts'].values()):,} rows, {splits['retained_groups']:,} duplicate components and {splits['retained_split_groups']:,} duplicate/session split components. Largest split component: {splits['largest_split_group_rows']} rows.",
        f"Train/validation/test: {splits['split_counts']['train']:,}/{splits['split_counts']['validation']:,}/{splits['split_counts']['test']:,}; seed 42. Both models use these identical frozen partitions.",
        f"Class exclusions for scarcity (remaining duplicate groups): `{splits['excluded_class_group_counts']}`. Other labels can disappear through conflicting-group exclusion; see the audit and retained class counts.", "",
        "All intent references and language subgroups are publisher AI weak labels. None is independently human-verified. Source answers are not inputs or factual references. C was selected on validation macro F1; held-out results below were not used to tune C.", "",
        "## Frozen test measurements", "",
        "| Model | C | Validation macro F1 | Test accuracy | Test macro F1 | Weighted F1 | Train + selection (s) | Saved bytes | CPU median / p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, item in results.items():
        m, manifest = item["metrics"], item["manifest"]
        selected = next(trial for trial in manifest["validation_trials"] if trial["C"] == manifest["config"]["selected_C"])
        latency = m["latency_batch1_cpu_ms"]
        lines.append(f"| {kind} | {manifest['config']['selected_C']:g} | {selected['macro_f1']:.4f} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {manifest['training_seconds_including_selection']:.2f} | {manifest['artifact_bytes']:,} | {latency['median']:.3f} / {latency['p95']:.3f} |")
    lines += ["", f"A constant predictor using the training-majority label `{majority_label}` would score **{majority_accuracy:.4f}** test accuracy. This is a class-frequency reference, not a third trained model. High accuracy alone obscures minority-class performance; Logistic Regression's accuracy is below this reference.", "", "The observed macro-F1 difference between the two models is small and does not establish superiority. Do not select a production model using this test comparison.", "", "| Model | Accuracy 95% interval | Macro F1 95% interval | Evaluation process lifetime peak working set (bytes) |", "| --- | --- | --- | ---: |"]
    for kind, item in results.items():
        m = item["metrics"]
        ci = m["confidence_intervals"]["95_percent"]
        lines.append(f"| {kind} | {ci['accuracy'][0]:.4f}–{ci['accuracy'][1]:.4f} | {ci['macro_f1'][0]:.4f}–{ci['macro_f1'][1]:.4f} | {m['process_lifetime_peak_working_set_bytes']:,} |")
    lines += ["", "Intervals use 200 duplicate/session-group bootstrap replicates, conditional on one fitted model/split. They do not capture label validity or training-seed variation. Peak memory is a process-lifetime working set, not isolated per-model memory. Latency uses 10 warm-ups and the first 200 sequential CPU batch-1 predictions, excluding model load and HTTP overhead.", "", "## Language-specific weak-label analysis", "", "Macro F1 uses all eight retained source classes, including classes absent from a subgroup. These figures are not human-verified language benchmarks. Preserve anomalous categories rather than merging them.", "", "| Source AI language | n | Distinct duplicate groups | Logistic macro F1 | SVM macro F1 | n < 30 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    svm_by_language = {item["language"]: item for item in results["svm"]["languages"]}
    for item in results["logistic"]["languages"]:
        other = svm_by_language[item["language"]]
        lines.append(f"| {item['language']} | {item['n']} | {item['distinct_groups']} | {item['macro_f1']:.4f} | {other['macro_f1']:.4f} | {item['small_subgroup']} |")
    lines += ["", "## Error analysis", "", "The table reports model disagreement with source labels, not confirmed human errors. Confusion matrices and ID-only disagreement files are saved in each evaluation directory; no raw questions are copied into this report.", "", "| Source class | Test support | Logistic F1 | SVM F1 |", "| --- | ---: | ---: | ---: |"]
    tables = {}
    for kind in results:
        with (root / kind / "evaluation" / "per_class_metrics.csv").open(newline="", encoding="utf-8") as stream:
            tables[kind] = {row["class"]: row for row in csv.DictReader(stream)}
    for label, row in tables["logistic"].items():
        lines.append(f"| {label} | {int(float(row['support']))} | {float(row['f1-score']):.4f} | {float(tables['svm'][label]['f1-score']):.4f} |")
    for kind in results:
        matrix = read_json(root / kind / "evaluation" / "confusion_matrix.json")
        confusions = sorted(((count, matrix["labels"][i], matrix["labels"][j]) for i, row in enumerate(matrix["counts"]) for j, count in enumerate(row) if i != j and count), reverse=True)[:3]
        lines += ["", f"Largest {kind} disagreements: " + "; ".join(f"`{reference}` → `{prediction}`: {count}" for count, reference, prediction in confusions) + "."]
    lines += ["", "Only two test examples carry the source `emergency` label; this is inadequate evidence for safety recognition. Imbalanced classes and low-support language categories limit interpretation. Source-label noise and duplicate-conflict exclusions change the population being evaluated. Learned behavior may include academic/university administration rather than hospital enquiries.", "", "## Annotation and remaining work", "", f"{annotation['count']} actual source questions exported pending review; {annotation['independent_review_count']} assigned independent review. **Zero human reviews were completed.** Importing the untouched pending export confirmed zero reviewed labels and no agreement estimate.", "", "Batch B must add multilingual comparisons, entity evaluation, calibration and safety routing with defensible evaluation data. Retrieval, hospital facts, full-stack application, ablations and deployment remain unimplemented. No confidence, safety, clinical validity, or algorithmic novelty claim follows from these scores.", ""]
    Path("docs/BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Generated docs/BASELINE_REPORT.md and aggregate summary.json from completed run artifacts.")


if __name__ == "__main__":
    main()
