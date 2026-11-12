# MedRoute AI architecture

## Implemented in Batch A

The executable system is an **offline research CLI**, not a hospital assistant service.
Its flow is:

`immutable MEDIQ release -> verified raw files -> schema/consent/PII audit -> clean questions -> duplicate groups -> session-aware frozen partitions -> validation-selected CPU baseline -> test evaluation`

Human annotation is a separate branch from the frozen partitions. It does not silently
replace publisher labels. Reviewed imports are stored separately and are not consumed by
the current baseline training command.

`ml/acquisition.py` verifies the pinned Git data blob and saves SHA-256 checksums and
attribution. `ml/data.py` performs screening, normalization, grouping, and splitting.
`ml/annotation.py` manages exports and declared human reviews. `ml/baselines.py`
implements fitting, model loading, evaluation, and local prediction. `ml/cli.py` is the
only public executable interface. All modules operate on explicit local paths.

### Interfaces and stored contracts

- Raw input: MEDIQ object with a `data` dictionary of date-indexed arrays. A flat `data`
  array is also accepted for software fixtures or compatible exports. Other layouts fail.
- Audited JSONL: source ID, original and normalized question, source weak intent/language,
  timestamp, hashed source session, `publisher_ai_weak` provenance, pending annotation status.
  Generated answers and raw source session IDs are omitted.
- Prepared JSONL adds `group_id` (duplicate component), `split_group_id` (duplicate/session
  component), and `split`. A checksum-bound manifest records source identity, memberships,
  grouping settings, exclusions, and actual partition sizes.
- Research run: selected scikit-learn pipeline, model checksum, source/split/code identities,
  environment, C-selection trials, and training measurements. The serialized pipeline is
  pickle-based: **load only trusted, locally trained artifacts**.
- Prediction JSON: `predicted_source_intent`, class scores, explicit `score_type`, model
  checksum/version, label provenance, and false flags for hospital routing validation and
  action execution. There is no booking, answer generation, or safety guarantee.

CLI data and run output directories are versioned and write-once. Reuse is rejected, except
acquisition, which validates an existing completed manifest. Evidence rendering scripts may
regenerate aggregate copies and reports from the saved runs. A failed run may leave diagnostic partial
outputs; preserve them and select a new output path. Raw data and annotation files are
ignored by Git. Aggregate research reports can be tracked; no commits are created here.

## Planned modular monolith (not implemented)

- `apps/api`: FastAPI/Pydantic boundary, ownership/authentication, application services,
  explicit conversation state, escalation, and inference dependency injection. Model loaded
  once per process. SQLAlchemy/Alembic with PostgreSQL; isolated SQLite tests.
- `apps/web`: Next.js/React/TypeScript/Tailwind patient and authorized staff experiences.
  Compact interfaces, actual records/artifacts, explicit empty states. No fabricated demo data.
- ML additions: compact multilingual comparisons and entity rules, followed by calibration,
  abstention and safety overrides (Batch B). Those require reliable evaluation annotations.
- Retrieval: versioned permitted administrative sources, BM25 first, citations, stale-source
  checks and unsupported-answer abstention (Batch C). No hospital sources are configured.

Future request flow: validate/authorize -> safety override -> conversation state -> intent
and entities -> uncertainty decision -> permitted retrieval -> supported administrative
answer or clarification/escalation. An intent prediction never confirms an external action.
Retrieved content is evidence, never operating instructions. No agent framework is required.

## Hardware and dependency decisions

Observed Windows environment: Python 3.12.14 available through the bundled runtime, Node
22.13.0/npm 10.9.2, NVIDIA RTX 4060 Laptop GPU reporting 8,188 MiB and driver 617.42.
GPU inventory is **not** evidence that PyTorch/CUDA training works. CPU/RAM discovery via
CIM was denied during planning; processor text and memory measurements have their own
provenance in experiment manifests. Batch A uses CPU only, four native-library threads,
bounded sparse features, a three-value C grid, and no model-weight downloads.

The project `.venv` is isolated. `requirements.lock.txt` records the actual resolved
dependencies after successful installation; `pyproject.toml` defines compatible bounds.
No existing CUDA installation, unrelated environment, or operating system setting is changed.

## Risk register and gates

| Risk | Current control and remaining limit |
| --- | --- |
| Weak labels mistaken for truth | Source label names and explicit weak provenance everywhere; independent hospital evaluation remains pending. |
| Leakage | Label-blind duplicate grouping, session-connected partition groups, frozen checksums, train-only vocabulary/IDF, validation-only C selection. Semantic paraphrases and unknown sessions can still leak. |
| PII in a public dataset | Quarantine suspicious questions; ignored local data; aggregate reports and ID-only prediction files. Pattern screening cannot establish anonymity. |
| Overstated multilingual support | Preserve AI language categories/anomalies and counts; no verified language claim. |
| Excessive resource use | ~69 MB source, sparse 128-row similarity blocks, bounded n-grams, CPU models; grouping remains worst-case quadratic in time. |
| Unsafe clinical use | Offline output only, no clinical or operational deployment; broad source labels cannot drive hospital automation. |
| Artifact execution | joblib is trusted-local only; checksum detects accidental changes, not a maliciously replaced manifest. |
| Dataset unavailable | Stop acquisition with a clear error; retain a testable pipeline, never invent data or scores. |

Batch A exits after a real audit, frozen leakage checks, annotation workflow, passing tests,
and two saved/evaluated classical baselines where labels support them. Missing human reviews
remain explicit. No public deployment or later-phase implementation is authorized by this exit.
