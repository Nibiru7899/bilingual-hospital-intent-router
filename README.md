# MedRoute AI

**An uncertainty-aware multilingual hospital assistant research project.**

Current deliverable: **Batch A, canonical Phases 1–2** — an offline, reproducible MEDIQ
audit, annotation workflow, and two classical weak-label classification baselines.
There is no patient-facing application, hospital knowledge integration, clinical validation,
appointment service, or public deployment in this batch.

Read [PROJECT_STATUS.md](PROJECT_STATUS.md) for measured outcomes and blockers,
[ARCHITECTURE.md](ARCHITECTURE.md) for boundaries, and [EXPERIMENTS.md](EXPERIMENTS.md)
for experiment provenance. Later batches require explicit authorization.

## Setup on Windows / PowerShell

Use Python 3.12 and an isolated project environment. On this machine Python was unavailable
on PATH; the installed Codex runtime supplied Python 3.12.14. The existing `.venv` is ready
after dependency verification. To recreate from scratch with your own Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
```

If the launcher is unavailable, replace `py -3.12` with the absolute path of an existing
Python 3.12 executable. Do not change unrelated environments. No CUDA/PyTorch, database,
Node packages, paid APIs, or Transformer downloads are required for Batch A.

## Reproduce the data and baseline workflow

Run from the repository root. Source files are acquired from one immutable release commit.
Acquisition may reuse a checksum-valid source; audit/preparation/annotation/training/evaluation outputs are write-once. If a directory
already exists, verify it or supply a new versioned `--output` and pass matching paths to
downstream commands. Do not delete research data to force a rerun.

```powershell
.\.venv\Scripts\python.exe -m ml.cli acquire
.\.venv\Scripts\python.exe -m ml.cli audit
.\.venv\Scripts\python.exe -m ml.cli prepare
.\.venv\Scripts\python.exe -m ml.cli verify
.\.venv\Scripts\python.exe -m ml.cli annotation-export
.\.venv\Scripts\python.exe -m ml.cli train --model both
.\.venv\Scripts\python.exe -m ml.cli evaluate --run artifacts/batch-a-v1/logistic
.\.venv\Scripts\python.exe -m ml.cli evaluate --run artifacts/batch-a-v1/svm
.\.venv\Scripts\python.exe -m ml.cli predict --run artifacts/batch-a-v1/logistic --text "Where is the outpatient registration counter?"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/capture_evidence.py
.\.venv\Scripts\python.exe scripts/report_batch_a.py
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

Defaults: raw `data/raw/mediq-v0.6.0`, final audit `data/audited/mediq-v0.6.0-v2`,
final prepared `data/prepared/mediq-v0.6.0-v2`, annotation `data/annotation/batch-a-v1`,
models and aggregate evidence `artifacts/batch-a-v1`. Earlier development audit/prepared
outputs are preserved locally and are not the selected experimental inputs.

`python -m ml.cli --help` lists commands and explicit input/output options. Downloads have
socket and total-duration bounds and require both pinned release identity and checksums.
Allow roughly 70 MB for source data plus environments, intermediate data and artifacts.
Grouping uses sparse 128-row blocks but can take minutes because its time cost is quadratic;
baseline feature counts and the C grid are bounded. Training is CPU-only with four native
threads. Actual duration/memory are recorded rather than promised in advance.

## Human annotation

Follow [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md). Preserve the original export,
edit a separate review copy, obtain real primary/independent/adjudication judgments, then:

```powershell
.\.venv\Scripts\python.exe -m ml.cli annotation-import --csv data/annotation/review-copy.csv --export-dir data/annotation/batch-a-v1 --output data/annotation/review-import-v1
```

The import requires exactly the original IDs/text/split assignments, valid labels, reviewer
declarations, timestamps, and independent review on assigned rows. It reports agreement only
for completed reviewer pairs. It cannot authenticate reviewers or enforce blind work: that is
a human process requirement. Pending rows never become human-reviewed labels automatically.
Reviewed imports do not alter existing weak-label training or test data.

## Results, privacy, and limits

- MEDIQ intent/language labels are AI-generated weak supervision. Benchmark scores measure
  source-label agreement, not hospital-intent correctness, language validation, or safety.
- Hospital intent taxonomy, human-reviewed test evaluation, multilingual Transformers,
  calibration, routing, retrieval, API, frontend, and deployment remain separate future work.
- No original chatbot answers are used as verified information. No patient data is sent to
  inference providers, and no hospital action is performed.
- PII screening is imperfect. Raw data, clean question text, annotations, quarantine records,
  models and ID-level predictions stay local and ignored by Git. Review local access and
  retention before sharing. Deletion is manual and requires explicit authorization.
- Only load trusted local `model.joblib` files: joblib/pickle can execute code. A checksum
  detects accidental corruption, not an attacker who changes the manifest as well.
- Logistic probabilities are uncalibrated; SVM margins are not probabilities. Neither drives
  automation. Safety-sensitive handling is not implemented in this research CLI.
- Tests use explicitly synthetic fixtures under temporary directories. Their metrics and
  fixture questions are not research results or a public demo.

## Data attribution

Dr. Ayush Yadav (2026), **MEDIQ**, v0.6.0, CC BY 4.0:
https://github.com/ProxyAyush/MEDIQ/tree/c8226fc9662003270146e6130dccaf1d586d6eae

License: https://creativecommons.org/licenses/by/4.0/

This project filters, normalizes, groups and partitions question records; it trains separate
classifiers. Source AI annotations remain weak labels. Source license/datasheet and exact
checksums are saved locally and source attribution is repeated in aggregate evidence.
