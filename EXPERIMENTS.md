# Experiment register

## Protocol — Batch A

**Hypothesis:** word/character TF-IDF and character-only TF-IDF provide feasible CPU
baselines for MEDIQ's broad source intents. Their relative performance is unknown until
measured; neither is assumed to validate hospital-specific intent understanding.

Pinned source: MEDIQ v0.6.0, commit `c8226fc9662003270146e6130dccaf1d586d6eae`,
SHA-256 `d68b19c8efc8376f2c03ac2bf0ad9ab3e43ba2b5ec33678e4d50e3ddca286aa4`.
Only questions are model features. Labels and language subgroups have
`publisher_ai_weak` provenance. Human-reviewed data is not available at the outset.

Protocol and mathematical definitions: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
Seed 42; duplicate/session-aware frozen split; train-only features; class-balanced training;
C in {0.5, 1, 2}, selected on validation macro F1. Both selected configurations receive one
test evaluation. No thresholds or model choices are selected using test outcomes.

## Run register

Completed on 19 October 2026. Both saved pipelines used **20,510 training**, **4,103
validation**, and **4,103 test** rows across eight source classes. Prepared record checksum:
`29a34a6ce796e5896e04f475708ddd928b474e0f9c4e0ecf1ed8545cace37141`.

| Run | Validation-selected C | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `artifacts/batch-a-v1/logistic` | 1.0 | 0.5071 | 0.7926 | 0.4524 | 0.8118 |
| `artifacts/batch-a-v1/svm` | 0.5 | 0.4566 | 0.8389 | 0.4623 | 0.8393 |

Logistic model SHA-256: `fdf5ee0e689f99787aa04296941918e6e4ba0791e1cf02f38c6d4ff9bb66c7ee`.
SVM model SHA-256: `6eb45fc37bea7f4b88c3cc93480bd4a46d0668b55a5db75b1e47e63726a7e6f7`.

Training/selection took 32.03 s and 24.44 s, respectively. Compressed artifacts occupy
6,474,948 and 3,108,640 bytes. Warmed CPU batch-1 median/p95 prediction times were
0.931/1.102 ms and 0.405/0.570 ms, excluding model loading and HTTP overhead.

The training-majority constant label would yield 0.8155 test accuracy. Logistic accuracy
falls below that reference. Macro-F1 bootstrap intervals overlap, so the observed test
difference is not evidence of statistical superiority. Both models have zero F1 on the
two source-labeled emergency test records; no safety-recognition claim is defensible.

Full measured report, per-language analysis and error analysis:
[docs/BASELINE_REPORT.md](docs/BASELINE_REPORT.md). Under each run's `evaluation` directory:
`metrics.json`, `per_class_metrics.csv`, `language_metrics.json`, `language_comparison.csv`,
`predictions.csv`, `error_examples.csv`, `confusion_matrix.json/png`,
`language_comparison.png`, `latency_results.csv`, and a checksummed evaluation manifest.
Run manifests preserve C trials, environment, feature settings, source/split/code identities,
training duration, model size and process-lifetime peak working set.

The current annotation export contains 1,000 pending records with 200 independent-review
assignments. A real import validation of the untouched pending export returned **zero
human-reviewed records** and no agreement estimate. Tests exercise human-review imports
using synthetic fixtures only. None of those labels or fixture metrics enters this table.

## Planned beyond this batch

- Batch B: multilingual embeddings/XLM-R feasibility and comparison; entity extraction;
  trustworthy language/entity evaluation; calibration, selective prediction, safety routing.
- Batch C: verified-source retrieval and grounding benchmarks; product integration;
  appropriate ablations, security/integration checks and deployment evidence.

No corresponding model improvements, calibration curves, risk–coverage curves, retrieval
scores, ablations or deployment outcomes have been measured in Batch A.
