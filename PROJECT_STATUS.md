# MedRoute AI — project status

Date: 19 October 2026 (Asia/Kolkata). Authorized work: **Batch A, canonical Phases 1–2**.

## Batch and phase status

**Batch A delivered:** Phase 1 complete; Phase 2 complete for the authorized preliminary
weak-label foundation. Independently reviewed hospital-intent evaluation remains pending
actual human annotation. The overall capstone is **not complete**. Batches B/C were not started.

## Completed components and evidence identities

- Workspace inspected; only AGENTS.md existed. Git initialized; no commit or global Git
  configuration change. Original repository instructions preserved.
- Isolated Python 3.12.14 `.venv`, compatible dependencies and `requirements.lock.txt`.
  Node 22.13.0/npm 10.9.2 and RTX 4060 Laptop GPU (8,188 MiB, driver 617.42) observed;
  training used CPU, not CUDA. GPU framework readiness remains unverified.
- MEDIQ v0.6.0 at `c8226fc9662003270146e6130dccaf1d586d6eae` downloaded with license,
  datasheet/README, Git blob verification and SHA-256 identity
  `d68b19c8efc8376f2c03ac2bf0ad9ab3e43ba2b5ec33678e4d50e3ddca286aa4`.
- Source JSON is date-indexed and contains 30,587 rows/36 observed fields. Audit accepted
  30,380 and quarantined 207: 99 possible-PII flags and 108 consent-after-interaction
  timestamps. No missing audited core fields or repeated source IDs were found.
  Undocumented intent/language values and unexpected field types are recorded, not hidden.
- Duplicate grouping found 254 conflicting components, excluding 1,650 rows. Scarce classes
  excluded another 14 rows. Final dataset: **28,716 rows**, 26,580 duplicate groups,
  17,040 duplicate/session split groups; largest split group 241 rows.
- Frozen train/validation/test: **20,510 / 4,103 / 4,103**. Records checksum:
  `29a34a6ce796e5896e04f475708ddd928b474e0f9c4e0ecf1ed8545cace37141`.
  Final inputs are `data/audited/mediq-v0.6.0-v2` and `data/prepared/mediq-v0.6.0-v2`.
  Earlier audit/prepared outputs are preserved as development history.
- 1,000 real questions exported pending annotation; 200 require independent review.
  Untouched pending export import validated **zero human-reviewed labels**, with no agreement
  statistic. Primary/secondary/adjudicator identities remain human declarations, not authentication.
- Two genuine saved CPU classifiers with validation-selected C and exactly one completed
  frozen-test evaluation per selected configuration. No test-driven parameter changes.
- Architecture, methodology, annotation guidance, reproducible CLI, aggregate dataset/run
  manifests, error analysis, confusion/language plots and measured report delivered.

| Model | Selected C | Test accuracy | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Word + character TF-IDF Logistic Regression | 1.0 | 0.7926 | 0.4524 | 0.8118 |
| Character TF-IDF Linear SVM | 0.5 | 0.8389 | 0.4623 | 0.8393 |

Scores measure **publisher weak-label agreement**, not correct hospital routing. The constant
training-majority reference has test accuracy 0.8155. Both classifiers missed the two
source-labeled emergency test records. Full counts, bootstrap intervals, language limitations,
timing and source identities are in [docs/BASELINE_REPORT.md](docs/BASELINE_REPORT.md).

## Commands and observed results

Run in PowerShell from the repository root; all Python commands use `.\.venv\Scripts\python.exe`.

| Exact command (after the Python executable where shown) | Observed result |
| --- | --- |
| `git init` | Initialized empty Git repository; no commits made. |
| Bundled Python `-m venv .venv` | Created isolated Python 3.12.14 environment. |
| `-m pip install -e '.[test]'` | Installed project and Batch A ML/test dependencies successfully. |
| `-m ml.cli acquire` | Downloaded verified 69,138,580-byte JSON plus license/datasheet/README. |
| `-m ml.cli audit` | Final versioned audit: 30,587 source / 30,380 eligible / 207 quarantine. |
| `-m ml.cli prepare` | Final frozen dataset: 28,716 rows; eight retained source classes. |
| `-m ml.cli verify` | Verified all 28,716 rows; checksums and duplicate/session partition isolation passed. |
| `-m ml.cli annotation-export` | Exported 1,000 pending records / 200 independent-review assignments. |
| `-m ml.cli train --model both` | Six C-grid fits; selected logistic C=1, SVM C=0.5 on validation. |
| `-m ml.cli evaluate --run artifacts/batch-a-v1/logistic` | Completed test evaluation on 4,103 rows; metrics above. |
| `-m ml.cli evaluate --run artifacts/batch-a-v1/svm` | Completed test evaluation on same 4,103 rows; metrics above. |
| `-m ml.cli annotation-import --csv data/annotation/batch-a-v1/annotation.csv --export-dir data/annotation/batch-a-v1 --output data/annotation/pending-import-validation-v1` | 1,000 pending / 0 reviewed / no agreement estimate. |
| `scripts/capture_evidence.py` | Saved aggregate-only evidence and resolved dependency lock. |
| `scripts/report_batch_a.py` | Generated measured baseline report and summary from completed runs. |
| `scripts/verify_artifacts.py` | Verified pinned source, all 28,716 prepared rows, current trained ML code, both model identities, frozen splits and every evaluation output checksum. |
| `-m pip check` | No broken requirements found. |
| `-m pytest -q` | **43 passed in 10.06 s** on final expanded suite. |
| `-m compileall -q ml scripts tests` | Completed successfully after tests. |
| `-m ml.cli predict --run artifacts/batch-a-v1/logistic --text 'Where is the outpatient registration counter?'` | Synthetic I/O smoke probe: actual saved model returned source intent navigation and uncalibrated probabilities; no action claimed. |
| `-m ml.cli predict --run artifacts/batch-a-v1/svm --text 'Kal cardiology appointment chahiye'` | Synthetic I/O smoke probe: actual saved model returned scheduling and explicit margins; no action claimed. |

Initial tests were 38 passed in 7.66 s, then 40 passed in 3.25 s after schema/session changes.
Final suite adds release-checksum, download-bound and transitive session-group checks.
Software fixtures and the two synthetic I/O probes are **not** held-out research evidence.
Confusion matrix and language plot samples were visually inspected for legibility.

### Encountered failures and resolutions

- Initial `python`/Python launcher discovery found no PATH runtime. Used the installed bundled
  Python to create a project `.venv`; no unrelated environment was changed.
- Initial audit assumed a flat array and stopped with a schema error. Inspection found
  date-indexed arrays; loader corrected and covered with a regression test. The failed audit
  produced no eligible-data output, rather than a fabricated success.
- Normal sandbox process creation failed with `setup refresh had errors`; approved escalated
  commands were used. No privacy/security check was disabled to pass tests.
- Git status under the elevated user rejected ownership of sandbox-initialized `.git`.
  Verification used `git -c 'safe.directory=D:/Bilingual Hospital Intent Router' ...`, scoped
  to each invocation; no global safe-directory or author identity configuration was modified.
- `git ... check-ignore` confirmed raw data, annotations, `.venv`, model weights and individual
  prediction files are ignored. Aggregate metrics/manifests/figures remain reviewable.

## Explicit limits

No actual human reviews, verified hospital-intent test set, approved hospital knowledge,
Transformers, calibration, safety-routing service, API, database, UI, or deployment.
MEDIQ labels are weak and public-source anonymity claims are not independently established.
Normal sandbox execution is failing before process creation (`setup refresh had errors`);
necessary commands use approved escalation. Test/data outputs remain local.

## Next authorized work to request

> Implement Batch B only: canonical Phases 3–4. Read AGENTS.md, PROJECT_STATUS.md,
> ARCHITECTURE.md, EXPERIMENTS.md and the Batch A baseline report. Preserve the pinned source,
> frozen duplicate/session-aware partitions and weak-label provenance. Benchmark feasible
> multilingual models and entity extraction, then calibration, abstention and conservative
> safety routing. Obtain/record actual human reviews or explicitly retain evaluation blockers.
> Run tests and report measured results. Do not start retrieval, the application, or Batch C.
