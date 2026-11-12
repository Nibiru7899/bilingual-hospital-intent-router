# Batch A methodology and references

## Research question and experiment boundary

This batch establishes a baseline for RQ1: how well do classical text classifiers reproduce
MEDIQ's broad, publisher-generated intent labels? It does **not** answer whether they
correctly route hospital enquiries. RQ2 language analysis is provisional because its
subgroup labels are also automated. Calibration/safety questions and retrieval questions
remain planned for later batches.

## Data protocol

Pin release v0.6.0 at commit `c8226fc9662003270146e6130dccaf1d586d6eae`. Verify the
69,138,580-byte JSON against Git blob `d6c3ad25665c20b9da025f282e5e1e7b5dde5fa7` and
record its SHA-256. Retain source attribution, license, datasheet and retrieval timestamp.
Independently inspect actual schema/counts instead of copying changing README statistics.
Treat publisher consent/anonymity statements as source assertions, not legal validation.

Require nonempty string ID/question/weak intent, explicit boolean consent, unique IDs,
timezone-aware interaction and consent timestamps, and consent no later than interaction.
Quarantine possible PII and invalid records. PII screening has false positives and negatives;
its accepted subset is not certified anonymous. No original chatbot answer is a response
reference or training input. No generated translation or AI metadata is a model feature.

Normalize questions with NFC Unicode normalization, casefolding and whitespace collapsing.
Preserve Hindi and punctuation. Use exact normalized duplicates plus label-blind hashed
character 3–5-gram cosine similarity >=0.92, with 262,144 nonnegative hash bins and sparse
128-row batches. Connected components prevent discovered transitive duplicates from crossing
partitions. The threshold is fixed before model evaluation. Hash collisions, semantic
paraphrases, and short texts remain limitations. Quarantine duplicate components with
conflicting weak labels instead of selecting a convenient label.

Classes with fewer than ten remaining duplicate groups are excluded from preliminary
training, with counts reported. Preserve original label names, including undocumented labels
if trainable. Connect retained duplicate components sharing a known hashed session into
split groups; mixed-intent sessions are legitimate and do not themselves imply label conflict.
Use seeded StratifiedGroupKFold: first of seven folds for test, then first of six remaining
folds for validation. This approximates 71/14/14; actual fractions vary because groups are
indivisible. Fail if any retained class is absent from a partition. Freeze source IDs,
memberships, hashes and settings. Same-group repeated rows remain within a single partition;
metrics are row-weighted and duplicate frequency may influence them.

## Models and selection

Both pipelines use normalized **question text only**, train-only vocabulary/IDF,
sublinear TF, L2 normalization, and float64 sparse matrices.

| Pipeline | Features | Classifier |
| --- | --- | --- |
| logistic | Word 1–2 grams (max 40,000) + character 3–5 grams (max 60,000) | Class-balanced LogisticRegression, max_iter=1000 |
| svm | Character 3–5 grams (max 60,000) | Class-balanced LinearSVC, max_iter=10,000 |

Use seed 42 and four native-library threads. Fit features once per model on training rows;
choose C from 0.5, 1, 2 by validation macro F1. Smaller C wins ties. Save that already fitted
classifier without refitting on validation/test. Evaluate each selected configuration once
on the identical frozen test partition. Results do not authorize choosing a winner on test.
Logistic scores are uncalibrated probabilities; SVM scores are margins. Neither is a safety
score, and no operational threshold is selected in this batch.

## Metric definitions and interpretation

For class c: precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2PR/(P+R).
Macro F1 is the unweighted mean across all retained classes; weighted F1 weights by support.
Undefined class precision/recall are set to zero and support is always reported. Accuracy is
the fraction of rows whose prediction matches the weak source label, not clinical accuracy.
Per-language macro scores use the same full class list, including zero-support classes;
also report observed classes, counts, groups and a flag for subgroups below 30 rows.
Do not merge anomalous source language strings into more favorable categories.

Uncertainty: 200 whole duplicate/session-group bootstrap replicates yield percentile 95%
intervals for accuracy and macro F1. These intervals condition on one fitted model and one
split; they do not measure repeated training-seed variability or weak-label correctness.
Latency measures up to 200 sequential, warmed CPU batch-1 `predict` calls, including feature
transform but excluding model loading and HTTP overhead. Artifact bytes measure compressed
joblib size. Windows peak working set is a **process-lifetime** measurement, not an isolated
per-model allocation. Training duration includes feature fitting and all C trials.

Confusion matrices and language charts come only from actual predictions. Error files store
IDs/labels, not question text; authorized reviewers may join IDs locally for error review.
Label disagreements with a weak source cannot automatically be called model mistakes.

## Later mathematical methodology (planned, not measured)

Softmax p(c|x)=exp(z_c)/sum_j exp(z_j); cross-entropy=-log p(y|x). These explain classifier
probabilities, but this batch does not train a Transformer. Calibration asks whether predicted
probabilities match empirical correctness. Multiclass Brier score is the mean sum_c
(p_c-1[y=c])². Selective risk is errors/accepted cases; coverage is accepted/all cases,
with risk undefined when no cases are accepted. Retrieval recall@k is relevant retrieved
documents/all relevant documents; precision@k is relevant retrieved documents/k. Later
experiments need trustworthy targets, validation-only choices, and explicit zero-denominator
behavior. No corresponding curves are generated for an unimplemented experiment.

## Source references

- Ayush Yadav, MEDIQ v0.6.0, pinned repository and included CC BY 4.0 license:
  https://github.com/ProxyAyush/MEDIQ/tree/c8226fc9662003270146e6130dccaf1d586d6eae
- Publisher datasheet (statements require independent scrutiny):
  https://github.com/ProxyAyush/MEDIQ/blob/c8226fc9662003270146e6130dccaf1d586d6eae/DATASHEET.md
- scikit-learn official API references for implemented methods:
  https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

This is a foundation for a later literature review, not a completed review of multilingual
learning, medical safety, calibration, or retrieval research. No algorithmic novelty is claimed.
