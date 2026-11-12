"""Reproducible CPU baselines; no calibration, routing or hospital action claims."""

from __future__ import annotations

import ctypes
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ml.common import now, read_json, require_new, sha256, write_csv, write_json
from ml.data import SEED, load_prepared, normalize

MODELS = ("logistic", "svm")
C_GRID = (.5, 1., 2.)


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def peak_memory_bytes() -> int | None:
    """Process lifetime peak working set, not isolated per-model allocation."""
    if platform.system() != "Windows":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return None


def environment() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "matplotlib", "joblib", "pytest", "threadpoolctl")}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(), "packages": versions, "gpu_inventory_not_used_for_training": gpu, "training_device": "CPU", "thread_limit": 4}


def make_pipeline(kind: str, c: float = 1.) -> Pipeline:
    common = {"lowercase": False, "sublinear_tf": True, "dtype": np.float64}
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=60_000, **common)
    if kind == "logistic":
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=40_000, token_pattern=r"(?u)\b\w+\b", **common)),
            ("char", char),
        ])
        classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED)
    elif kind == "svm":
        features = char
        classifier = LinearSVC(C=c, class_weight="balanced", max_iter=10_000, random_state=SEED, dual="auto")
    else:
        raise ValueError("Unknown baseline model")
    return Pipeline([("features", features), ("classifier", classifier)])


def summarize(y: list[str], predictions: list[str], labels: list[str]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y, predictions, labels=labels, average="weighted", zero_division=0)[2]
    return {"n": len(y), "accuracy": float(accuracy_score(y, predictions)), "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1), "weighted_f1": float(weighted)}


def train(prepared: Path, output: Path, kind: str) -> dict:
    require_new(output)
    records, splits = load_prepared(prepared)
    training = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    x_train = [row["normalized_question"] for row in training]
    y_train = [row["weak_intent"] for row in training]
    x_val = [row["normalized_question"] for row in validation]
    y_val = [row["weak_intent"] for row in validation]
    labels = sorted(set(y_train))
    started = time.perf_counter()
    trials, best_f1, best = [], -1., None
    with threadpool_limits(limits=4):
        pipeline = make_pipeline(kind)
        # Vocabulary and IDF see training inputs only, including during C selection.
        features = pipeline.named_steps["features"]
        train_matrix = features.fit_transform(x_train)
        val_matrix = features.transform(x_val)
        for c in C_GRID:
            classifier = clone(pipeline.named_steps["classifier"]).set_params(C=c)
            trial_start = time.perf_counter()
            classifier.fit(train_matrix, y_train)
            scores = summarize(y_val, classifier.predict(val_matrix).tolist(), labels)
            trials.append({"C": c, "fit_seconds": time.perf_counter() - trial_start, **scores})
            if scores["macro_f1"] > best_f1:
                best_f1, best = scores["macro_f1"], classifier
        selected = Pipeline([("features", features), ("classifier", best)])
    duration = time.perf_counter() - started
    output.mkdir(parents=True)
    joblib.dump(selected, output / "model.joblib", compress=3)
    config = {"model": kind, "selected_C": float(best.C), "C_grid": list(C_GRID), "seed": SEED, "class_weight": "balanced", "feature_specification": str(features), "normalization": "NFC, casefold, collapsed whitespace", "model_selection": "validation macro F1, smallest C wins ties", "test_access_during_fit": False}
    manifest = {
        "created_at_utc": now(), "config": config, "label_provenance": "publisher_ai_weak",
        "intended_use": "Offline broad source-intent research only; not hospital routing",
        "source_sha256": splits["source_sha256"], "prepared_records_sha256": splits["records_sha256"],
        "split_manifest_sha256": sha256(prepared / "split_manifest.json"), "code_sha256": source_hashes(),
        "model_sha256": sha256(output / "model.joblib"), "artifact_bytes": (output / "model.joblib").stat().st_size,
        "training_seconds_including_selection": duration, "train_rows": len(training), "validation_rows": len(validation),
        "feature_count": int(train_matrix.shape[1]), "labels": labels, "validation_trials": trials,
        "process_lifetime_peak_working_set_bytes": peak_memory_bytes(), "environment": environment(),
    }
    write_json(output / "experiment_manifest.json", manifest)
    return manifest


def load_model(run: Path) -> tuple[Pipeline, dict]:
    manifest = read_json(run / "experiment_manifest.json")
    if sha256(run / "model.joblib") != manifest["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    # joblib is pickle-based: this command is only for locally trained, trusted artifacts.
    return joblib.load(run / "model.joblib"), manifest


def predict(run: Path, text: str) -> dict:
    if not text.strip() or len(text) > 10_000:
        raise ValueError("Prediction input must contain 1-10000 characters")
    model, manifest = load_model(run)
    normalized = normalize(text)
    with threadpool_limits(limits=4):
        predicted = str(model.predict([normalized])[0])
        if manifest["config"]["model"] == "logistic":
            scores = model.predict_proba([normalized])[0]
            score_type = "uncalibrated_probability"
        else:
            scores = model.decision_function([normalized])[0]
            if np.ndim(scores) == 0:
                scores = np.array([-float(scores), float(scores)])
            score_type = "decision_margin_not_probability"
    return {"predicted_source_intent": predicted, "scores": dict(zip(model.classes_.tolist(), np.asarray(scores).tolist())), "score_type": score_type, "model_version": manifest["model_sha256"], "label_provenance": manifest["label_provenance"], "hospital_routing_validated": False, "action_performed": False}


def bootstrap_intervals(rows: list[dict], predictions: list[str], labels: list[str]) -> dict:
    """Resample whole duplicate groups, preserving correlation within each group."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.get("split_group_id", row["group_id"]), []).append(index)
    group_indices = list(groups.values())
    rng = np.random.default_rng(SEED)
    samples = {"accuracy": [], "macro_f1": []}
    y = np.array([row["weak_intent"] for row in rows])
    p = np.array(predictions)
    for _ in range(200):
        indices = np.concatenate([group_indices[i] for i in rng.integers(0, len(groups), len(groups))])
        result = summarize(y[indices].tolist(), p[indices].tolist(), labels)
        for key in samples:
            samples[key].append(result[key])
    return {"method": "200 duplicate/session-group bootstrap percentile replicates; fixed model, not training-seed variability", "seed": SEED, "95_percent": {key: np.quantile(values, [.025, .975]).tolist() for key, values in samples.items()}}


def evaluate(prepared: Path, run: Path) -> dict:
    output = run / "evaluation"
    require_new(output)
    records, splits = load_prepared(prepared)
    model, manifest = load_model(run)
    if splits["records_sha256"] != manifest["prepared_records_sha256"] or sha256(prepared / "split_manifest.json") != manifest["split_manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from the training split")
    rows = [row for row in records if row["split"] == "test"]
    texts = [row["normalized_question"] for row in rows]
    y = [row["weak_intent"] for row in rows]
    labels = manifest["labels"]
    with threadpool_limits(limits=4):
        predictions = model.predict(texts).tolist()
        # Separate sequential batch-1 calls from whole-test batch timing.
        for text in texts[:min(10, len(texts))]:
            model.predict([text])
        latency = []
        for index, text in enumerate(texts[:min(200, len(texts))]):
            started = time.perf_counter_ns()
            model.predict([text])
            latency.append({"sample": index, "milliseconds": (time.perf_counter_ns() - started) / 1e6})
    metrics = {
        **summarize(y, predictions, labels), "label_provenance": "publisher_ai_weak", "split": "frozen_test",
        "macro_averaging": "All retained source classes, including zero-support subgroup classes",
        "confidence_intervals": bootstrap_intervals(rows, predictions, labels),
        "latency_batch1_cpu_ms": {"n": len(latency), "warmup_calls": min(10, len(texts)), "median": float(np.median([item["milliseconds"] for item in latency])), "p95": float(np.percentile([item["milliseconds"] for item in latency], 95)), "includes_model_load": False},
        "model_bytes": manifest["artifact_bytes"], "process_lifetime_peak_working_set_bytes": peak_memory_bytes(),
    }
    language_metrics = []
    for language in sorted({row["weak_language"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["weak_language"] == language]
        language_metrics.append({"language": language, "language_provenance": "publisher_ai_weak", "distinct_groups": len({rows[i]["group_id"] for i in indices}), "small_subgroup": len(indices) < 30, "observed_classes": len({y[i] for i in indices}), **summarize([y[i] for i in indices], [predictions[i] for i in indices], labels)})
    report = classification_report(y, predictions, labels=labels, output_dict=True, zero_division=0)
    per_class = [{"class": label, **report[label]} for label in labels]
    output.mkdir()
    write_json(output / "metrics.json", metrics)
    write_json(output / "language_metrics.json", language_metrics)
    write_csv(output / "per_class_metrics.csv", per_class, ["class", "precision", "recall", "f1-score", "support"])
    write_csv(output / "language_comparison.csv", language_metrics, list(language_metrics[0]))
    write_csv(output / "latency_results.csv", latency, ["sample", "milliseconds"])
    prediction_rows = [{"id": row["id"], "group_id": row["group_id"], "language": row["weak_language"], "weak_reference": row["weak_intent"], "prediction": p, "label_provenance": "publisher_ai_weak"} for row, p in zip(rows, predictions)]
    write_csv(output / "predictions.csv", prediction_rows, list(prediction_rows[0]))
    # No raw questions in reports; authorized local reviewers can join IDs to ignored data.
    errors = [row for row in prediction_rows if row["weak_reference"] != row["prediction"]]
    write_csv(output / "error_examples.csv", errors, list(prediction_rows[0]))
    matrix = confusion_matrix(y, predictions, labels=labels)
    write_json(output / "confusion_matrix.json", {"labels": labels, "counts": matrix.tolist()})
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
             xlabel="Predicted source intent", ylabel="Publisher weak reference", title=f"{manifest['config']['model']}: frozen weak-label test")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar([item["language"] for item in language_metrics], [item["macro_f1"] for item in language_metrics])
    axis.set(ylim=(0, 1), ylabel="Macro F1 (all retained classes)", title="Publisher AI language groups; not human-verified")
    for i, item in enumerate(language_metrics):
        axis.text(i, item["macro_f1"] + .02, f"n={item['n']}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "language_comparison.png", dpi=150)
    plt.close(fig)
    write_json(output / "evaluation_manifest.json", {
        "created_at_utc": now(), "model_sha256": manifest["model_sha256"], "records_sha256": splits["records_sha256"],
        "evaluation_code_sha256": source_hashes(), "environment": environment(),
        "outputs_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())},
        "limitations": ["Source-label agreement, not hospital intent correctness or safety", "Test results must not be used for model or threshold selection", "Calibration, risk-coverage and human-reviewed evaluation deferred to Batch B"],
    })
    return metrics
