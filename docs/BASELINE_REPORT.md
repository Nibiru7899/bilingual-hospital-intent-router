# Batch A measured baseline report

## Data and protocol

Source: MEDIQ v0.6.0 at `c8226fc9662003270146e6130dccaf1d586d6eae` (CC BY 4.0).
Source SHA-256: `d68b19c8efc8376f2c03ac2bf0ad9ab3e43ba2b5ec33678e4d50e3ddca286aa4`.
Prepared records SHA-256: `29a34a6ce796e5896e04f475708ddd928b474e0f9c4e0ecf1ed8545cace37141`.

Audited 30,587 source rows; 207 excluded by schema/consent/PII screening.
Preparation excluded a further 1,664 rows: {'conflicting_group_labels': 1650, 'insufficient_class_groups': 14}.
Retained 28,716 rows, 26,580 duplicate components and 17,040 duplicate/session split components. Largest split component: 241 rows.
Train/validation/test: 20,510/4,103/4,103; seed 42. Both models use these identical frozen partitions.
Class exclusions for scarcity (remaining duplicate groups): `{'admission': 6, 'request': 2, 'contact': 4, 'problem_solving': 2}`. Other labels can disappear through conflicting-group exclusion; see the audit and retained class counts.

All intent references and language subgroups are publisher AI weak labels. None is independently human-verified. Source answers are not inputs or factual references. C was selected on validation macro F1; held-out results below were not used to tune C.

## Frozen test measurements

| Model | C | Validation macro F1 | Test accuracy | Test macro F1 | Weighted F1 | Train + selection (s) | Saved bytes | CPU median / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 1 | 0.5071 | 0.7926 | 0.4524 | 0.8118 | 32.03 | 6,474,948 | 0.931 / 1.102 |
| svm | 0.5 | 0.4566 | 0.8389 | 0.4623 | 0.8393 | 24.44 | 3,108,640 | 0.405 / 0.570 |

A constant predictor using the training-majority label `information_seeking` would score **0.8155** test accuracy. This is a class-frequency reference, not a third trained model. High accuracy alone obscures minority-class performance; Logistic Regression's accuracy is below this reference.

The observed macro-F1 difference between the two models is small and does not establish superiority. Do not select a production model using this test comparison.

| Model | Accuracy 95% interval | Macro F1 95% interval | Evaluation process lifetime peak working set (bytes) |
| --- | --- | --- | ---: |
| logistic | 0.7771–0.8119 | 0.3911–0.4950 | 306,409,472 |
| svm | 0.8214–0.8541 | 0.4013–0.4970 | 278,863,872 |

Intervals use 200 duplicate/session-group bootstrap replicates, conditional on one fitted model/split. They do not capture label validity or training-seed variation. Peak memory is a process-lifetime working set, not isolated per-model memory. Latency uses 10 warm-ups and the first 200 sequential CPU batch-1 predictions, excluding model load and HTTP overhead.

## Language-specific weak-label analysis

Macro F1 uses all eight retained source classes, including classes absent from a subgroup. These figures are not human-verified language benchmarks. Preserve anomalous categories rather than merging them.

| Source AI language | n | Distinct duplicate groups | Logistic macro F1 | SVM macro F1 | n < 30 |
| --- | ---: | ---: | ---: | ---: | --- |
| en | 2086 | 1863 | 0.4833 | 0.4917 | False |
| en-hi | 1 | 1 | 0.1250 | 0.1250 | True |
| en-hi-Latn | 1 | 1 | 0.1250 | 0.1250 | True |
| hi | 397 | 394 | 0.2557 | 0.2252 | False |
| hi-Latn | 900 | 877 | 0.3818 | 0.4376 | False |
| hi-Latn-en | 617 | 611 | 0.3287 | 0.3609 | False |
| hi-en | 101 | 100 | 0.2872 | 0.2989 | False |

## Error analysis

The table reports model disagreement with source labels, not confirmed human errors. Confusion matrices and ID-only disagreement files are saved in each evaluation directory; no raw questions are copied into this report.

| Source class | Test support | Logistic F1 | SVM F1 |
| --- | ---: | ---: | ---: |
| complaint | 118 | 0.5125 | 0.5441 |
| emergency | 2 | 0.0000 | 0.0000 |
| feedback | 12 | 0.6364 | 0.7619 |
| greeting | 27 | 0.3896 | 0.4889 |
| information_seeking | 3346 | 0.8734 | 0.9067 |
| navigation | 99 | 0.3636 | 0.3871 |
| off_topic | 32 | 0.2308 | 0.0000 |
| scheduling | 467 | 0.6132 | 0.6100 |

Largest logistic disagreements: `information_seeking` → `scheduling`: 342; `information_seeking` → `navigation`: 115; `information_seeking` → `complaint`: 101.

Largest svm disagreements: `information_seeking` → `scheduling`: 225; `scheduling` → `information_seeking`: 138; `information_seeking` → `complaint`: 58.

Only two test examples carry the source `emergency` label; this is inadequate evidence for safety recognition. Imbalanced classes and low-support language categories limit interpretation. Source-label noise and duplicate-conflict exclusions change the population being evaluated. Learned behavior may include academic/university administration rather than hospital enquiries.

## Annotation and remaining work

1000 actual source questions exported pending review; 200 assigned independent review. **Zero human reviews were completed.** Importing the untouched pending export confirmed zero reviewed labels and no agreement estimate.

Batch B must add multilingual comparisons, entity evaluation, calibration and safety routing with defensible evaluation data. Retrieval, hospital facts, full-stack application, ablations and deployment remain unimplemented. No confidence, safety, clinical validity, or algorithmic novelty claim follows from these scores.
