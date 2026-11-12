# Hospital-administration annotation protocol (v1)

## Status and purpose

The initial export contains **pending** examples from real MEDIQ questions. No human
annotation has been performed by Codex. Source intent and language labels are intentionally
hidden from reviewers; model predictions are not exported. The taxonomy below is proposed,
not validated. Reviewers should revise a future taxonomy version if actual data shows gaps;
never quietly alter an existing frozen test set or relabel old exports in place.

The source contains medical-university education enquiries as well as hospital enquiries.
Unsupported university admissions, examinations, jobs and academic fees must not be forced
into hospital-specific classes. The preliminary source-label benchmark uses its original
taxonomy and is independent of this annotation task.

## Intent definitions

| Label | Include | Exclude / distinguish |
| --- | --- | --- |
| OPD_TIMINGS | Opening days/hours of outpatient administrative services | A specific doctor's presence or a requested appointment time |
| DOCTOR_AVAILABILITY | Whether/when a named doctor or specialty is available | Booking intent without an availability question |
| APPOINTMENT_BOOKING | Requesting a new appointment or its booking procedure | Changing/cancelling an existing appointment |
| APPOINTMENT_RESCHEDULING | Moving an existing appointment | New bookings |
| APPOINTMENT_CANCELLATION | Cancelling an existing appointment | Complaints about cancellation without a requested action |
| DEPARTMENT_NAVIGATION | Where/how to reach a hospital department/counter | Academic department/course information |
| REPORT_STATUS | Availability/collection procedure of a medical report | Interpretation of report findings |
| BILLING_INSURANCE | Hospital payment, invoices, refunds, insurance process | College tuition or personalized coverage guarantees |
| PRE_VISIT_INFORMATION | Registration/documents/preparation procedures before a visit | Clinical advice or invented medical preparation instructions |
| FOLLOW_UP_INFORMATION | Administrative follow-up visit/process enquiries | Treatment changes or clinical follow-up advice |
| HOSPITAL_SERVICES | General hospital facilities/contact/service information | A more specific matching intent above |
| COMPLAINT | An explicit grievance about a hospital administrative service | Neutral status enquiries and academic grievances |
| GREETING | Greeting only | Greeting plus an actual request; label the request instead |
| OTHER_UNSUPPORTED | Clinical requests, education enquiries, unrelated topics, uninterpretable requests | Do not use merely because a supported request lacks a date/entity |

Annotation records describe the question, not whether a particular hospital can fulfill it.
No source answer is verified by assigning an intent. Do not add doctor names, schedules,
appointment confirmations, or other facts to complete an unclear question.

## Separate dimensions

- Safety: `administrative`, `clinical_advice`, `potential_emergency`,
  `sensitive_personal_data`, or `uncertain`. This is a review flag, not clinical triage.
  Emergency wording overrides routine administrative interpretation in a future router.
- Ambiguity: `clear`, `underspecified`, `multiple_intents`, `uninterpretable`.
  An obvious booking request with no date is underspecified; it can still have booking intent.
  For multiple requests choose the first explicit actionable administrative request and flag
  `multiple_intents`; if there is no defensible primary request use OTHER_UNSUPPORTED.
- Language: `en`, `hi` (Devanagari Hindi), `hi-Latn` (Romanized Hindi), `hi-en`
  (Hindi/English mixing with Devanagari), `hi-Latn-en` (Romanized Hindi/English mixing),
  `other`, or `uncertain`. Borrowed terms alone may be ambiguous; record uncertainty rather
  than mechanically counting ASCII characters. Reviewers need competence in both languages.

When potentially identifying or sensitive material escaped screening, flag it, stop wider
distribution, and request dataset review. Do not paste it into public issues or external APIs.

## Human workflow

1. Preserve `annotation.csv` and `export_manifest.json`. Work on a separate copy. Do not
   edit IDs, text, split, group or independent-review assignments. Questions beginning with
   spreadsheet formula characters are apostrophe-escaped for safe viewing.
2. Primary reviewer fills all `primary_*` fields, reviewer ID, and timezone-aware
   `reviewed_at`. Required independent review is assigned to 20% of the export. Collect the
   second person's judgments **blind in a separate copy** and merge only afterwards; a CSV
   tool cannot enforce blinding or authenticate a person's identity.
3. Fill `secondary_*` and `secondary_reviewer` for assigned rows. Distinct people are
   required. Fill `final_*` from the agreed labels and set `status=reviewed`.
4. On any disagreement, a third reviewer resolves it, fills `final_*`, supplies their
   `adjudicator` ID and explanatory `notes`, and sets `status=adjudicated`. Original judgments
   remain intact. A row assigned to independent review cannot finish without it.
5. Keep untouched rows `pending`. Import validates the complete export membership, field
   values, reviewer declarations and immutable text/splits; it writes a new reviewed artifact.
   It does not overwrite the weak-label dataset or train any model.
6. Review the paired count, raw agreement and Cohen's kappa per dimension. Kappa is undefined
   for constant-label pairs and reported as null, not invented. Stratified sampling means
   agreement describes this reviewed subset, not necessarily the source population.

Up to 1,000 unique duplicate groups are selected round-robin across existing split/source
intent/source-language strata. All retain their original partitions. Test reviewers must
remain blind to model predictions; training/tuning must never consume their held-out labels.
Until actual qualified reviews exist, there is **no independently reviewed hospital test set**.
