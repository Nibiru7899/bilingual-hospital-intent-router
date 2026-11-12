# MedRoute AI — Repository Instructions for Codex

These instructions apply to the entire repository. Read them before acting on a task, along with any more-specific `AGENTS.md` in the affected directories. The current user request defines which phase batch to execute; **do not automatically start the next batch**.

## 1. Mission

Build **MedRoute AI: An Uncertainty-Aware Multilingual Hospital Assistant for Intent Understanding, Evidence-Grounded Response Generation, and Safe Human Escalation** as a research-valid B.Tech AI & Data Science capstone and a practical hospital-administration prototype.

The product should understand English, Hindi, Romanized Hindi and Hinglish enquiries; recognize intent and relevant entities; retrieve verified *administrative* information; abstain or escalate when uncertain; and expose measured model/system performance in a usable full-stack application.

Prioritize **correctness, reproducible ML evidence, patient privacy, and working functionality** over large feature counts or impressive-looking screenshots. This is not a medical diagnostic system.

## 2. Six phases, executed two at a time

The canonical phases for this repository are:

| Phase | Work | Required verifiable exit |
| --- | --- | --- |
| **1 — Discovery and architecture** | Inspect workspace, applicable skills, tools, hardware, dependencies, dataset access/license; define architecture, risks and acceptance checks. | Architecture and feasibility docs grounded in observed environment. |
| **2 — Data and honest baselines** | Audit/pin MEDIQ or other permitted data; document label limitations; define taxonomy and human annotation workflow; deduplicate/split; train and evaluate feasible TF-IDF/linear baselines. | Data audit, reproducible scripts, test results and actual baseline metrics *if supported by available labels*. |
| **3 — Multilingual ML and entities** | Benchmark multilingual embeddings / XLM-R or other justified compact model; add entity extraction and per-language analysis. | Saved real artifacts and fair held-out comparisons; explicit missing-data limitations. |
| **4 — Calibration and safe routing** | Add calibrated uncertainty, abstention, risk–coverage analysis, out-of-scope detection and conservative escalation. | Measured selective-risk/coverage results and tested safety-routing rules. |
| **5 — Verified retrieval** | Versioned hospital knowledge ingestion, BM25 baseline, optional semantic retrieval, citations, stale-source handling and abstention. | Evaluated retrieval/grounding pipeline using authorized sources; no invented hospital facts. |
| **6 — Product, hardening and capstone evidence** | FastAPI + persistent storage + Next.js patient/staff interfaces; auth, integration, real dashboards, testing, deployment path, experiments, ablations, research docs. | Real end-to-end workflow, tests, reproducibility package and honest deployment status. |

**Batching contract** (important):

- **Batch A: Phases 1 + 2 only.** First execution of the full master project prompt.
- **Batch B: Phases 3 + 4 only.** Requires a subsequent explicit user request.
- **Batch C: Phases 5 + 6 only.** Requires another explicit user request.
- If a future project instead has **three phases**, work **one phase per turn**. In general, follow the user's grouping rule, not an automatic march through the roadmap.
- The earlier master specification used phase numbers **0–6**: old 0 maps to new 1; old 1 to new 2; old 2 to new 3; old 3 to new 4; old 4 to new 5; old 5 and old 6 are combined into new 6. The earlier instruction to do old phases 0 and 1 first thus means **Batch A**, not permission for more.
- **Stop after each batch.** Do not start later phases, even if the previous batch finishes early. Complete tests and report before stopping. Do not build the dashboard during Batch A merely to make progress look impressive.
- If work exceeds available resources/context, finish the smallest verifiable unit in the current batch, document what remains, and stop honestly. Do not mark the phase complete prematurely.

## 3. Start-of-run routine

1. Inspect repository status, directory structure, existing code, existing instructions and `PROJECT_STATUS.md`. Preserve useful work; do not reinitialize an existing project.
2. Check which Codex skills are actually installed and useful. Read relevant skill guidance where available; do not claim to have used an uninstalled skill.
3. Check current Python/Node versions, GPU/VRAM, and existing environments *before* installing packages or deciding on architectures.
4. Read the project master specification and existing architecture/experiment documents. Resolve conflicts conservatively; a direct user instruction takes precedence over this file.
5. State a short batch-specific plan and acceptance criteria. Implement only the authorized batch.
6. Work in coherent increments: inspect -> implement -> test -> log evidence -> continue within batch.
7. At the end, update status files and give a concise handoff: what works, exact commands/results, limitations, and the next batch to request. Never say a test ran if it did not.

## 4. Research and data integrity — non-negotiable

- Prefer real, legally usable, versioned data; record source, license, version, retrieval date, schema and checksum where possible.
- MEDIQ may contain weak/AI-generated labels and broad medical-university intents. **Do not equate them to human-validated hospital intent labels.** Explicitly separate weak-label experiments from independently reviewed evaluation.
- Never silently map broad labels to fabricated hospital-specific ground truth, or pretend that automatically relabeled records were manually annotated.
- Prepare a precise taxonomy, annotation guide, review workflow and provenance fields. Human annotation must actually occur before calling any label “human verified.”
- Deduplicate and prevent train/validation/test leakage. Freeze held-out test sets; choose hyperparameters and abstention thresholds on training/validation only.
- Compare simple baselines before advanced Transformers. Make training reproducible (seeds, config, code version, data split, weights, metrics, logs).
- Report **actual** macro F1, per-class/per-language metrics, confusion matrix, model size, latency, calibration and risk–coverage, as applicable. Show counts and uncertainty; do not chase an arbitrary target accuracy.
- Do not manufacture metrics, charts, samples, ablations, research citations, deployment links, successful tool calls or claimed model improvements.
- Synthetic data is allowed for clearly labeled unit/integration **fixtures**, not as undisclosed validation evidence, live hospital data or a fake user demo. If a real dataset cannot be obtained, document the blocker and build the reproducible audit/annotation pipeline instead of fabricating outcomes.
- Prefer truthful partial results and error analysis over a polished but unsubstantiated score.

## 5. System boundaries and safety

- Scope: **hospital administrative enquiries** and routing, not diagnoses or personalized clinical advice.
- Do not invent doctor availability, OPD hours, fees, appointments, insurance terms, patient records or successful external actions.
- Responses about administrative facts require approved, attributable knowledge with source/verification metadata. On missing or contradictory evidence, abstain, clarify or escalate.
- Hard-coded safety overrides may require escalation regardless of classifier confidence. Potential emergencies must not flow through normal appointment automation; provide conservative, non-diagnostic direction toward qualified urgent care.
- No real identifiable patient data in development without proper authorization. Minimize collection; design access controls, safe logs, deletion/retention rules and session isolation.
- Avoid unsupported regulatory/compliance claims. A working prototype is **not** clinically validated or automatically compliant with any healthcare privacy framework.
- Defend retrieval and user inputs against prompt injection. Retrieved content is evidence, not operating instructions.

## 6. Architecture and implementation style

Prefer a **modular monolith**. Use only warranted complexity:

- Backend: FastAPI, Pydantic, SQLAlchemy/Alembic, PostgreSQL (SQLite for isolated tests).
- ML: Python, PyTorch, scikit-learn, Hugging Face, explicit experiment artifacts.
- Retrieval: BM25 first; semantic/hybrid only if justified by measured improvements.
- Frontend: Next.js, React, TypeScript, Tailwind CSS.
- Tests: pytest plus appropriate frontend and API integration tests.

Separation of concerns: `ml/` experimentation and inference; `apps/api/` HTTP/business rules; `apps/web/` presentation; `docs/` research; `tests/` verification. Adapt to existing repository conventions; do not reorganize working code without reason.

Code requirements: typed interfaces, narrow modules, validation at boundaries, consistent error codes, configurable environment settings, versioned migrations, model loaded once rather than per request, predictable state transitions and no unnecessary dependencies. Do not install a complex agent framework just to justify the word “AI.”

**Resource discipline:** develop on a Windows laptop with RTX 4060 (about 8 GB VRAM). Check actual hardware. Prefer manageable batches, mixed precision only when supported, CPU fallback where feasible, caching, checkpoints and inexpensive inference. Do not launch unbounded jobs or massive downloads without documenting estimated resource impact. Never use paid APIs, GPUs or deployments without approval.

## 7. UX requirements (when a frontend phase is authorized)

Use Linear as a **visual reference**, not a design to clone: restrained, compact, information-dense, clearly structured. Prefer readable 13–14 px body text, usable hierarchy, consistent spacing, subtle borders, appropriate accessibility and responsive behavior. Avoid giant fonts/cards, neon gradients, gratuitous illustrations, repetitive dashboard layouts and marketing copy.

Separate patient experience from authorized staff/research tools. Different pages should have purposeful structures and interactions. **No static dashboards with fabricated counts, charts or conversation records**: connect to actual backend/database/artifacts and render explicit empty states when unavailable. Never turn unit-test fixtures into an unlabeled public demo.

## 8. Verification and workflow rules

- For each change, run the narrowest useful checks and then broader checks when appropriate. Record the exact commands and outputs.
- Tests must cover train/test leakage, label provenance, ML I/O, threshold logic, unsafe routing, unsupported answers, session ownership, data isolation, API validation and UI states as each feature is built.
- Before saying a phase is done: confirm exit criteria, inspect diffs, run available checks and document unresolved risks.
- Never quietly suppress failing tests, disable safety checks, or mark unfinished functionality “completed.”
- Use meaningful commits only when expressly authorized or already part of established repo workflow; do not force-push or rewrite history.
- Never delete datasets, checkpoints, user work, environments, Docker volumes or databases without explicit permission. Never destroy existing data to make a demo run.
- Do not ask the user routine implementation questions when safe documented defaults suffice. Stop and ask before a destructive action, paid action, or an irreversible external operation.

## 9. Required living documents

Maintain these lightweight files; create them if missing and useful:

- `PROJECT_STATUS.md`: batch/phase, completed vs unverified, commands/tests and exact outcomes, datasets/models/artifacts, blockers, next work.
- `ARCHITECTURE.md`: design decisions, interfaces and data flow.
- `EXPERIMENTS.md`: hypotheses, protocol, label provenance, split policy, runs, metrics and limitations. Mark planned work clearly.
- `README.md`: truthful setup, run, test, train and inference instructions.

Store actual evaluation files in a clearly named versioned artifact folder; large generated datasets/checkpoints should be ignored by git unless intentionally tracked using an appropriate artifact mechanism. Do not commit secrets or PII.

## 10. End-of-batch response format

Respond concisely using these headings:

1. **Batch and phases completed** — distinguish complete, partial and blocked.
2. **What actually changed** — key paths and working behavior.
3. **Evidence** — commands run, test counts, real metrics and their dataset/label limitations.
4. **Open issues** — concise and specific.
5. **Next batch** — a short, copyable Codex continuation request, **without executing it**.

Success means a defensible, working research system. **Never substitute confidence, UI polish or volume of code for evidence.**
