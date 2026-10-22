# Final reviewed pattern-recognition research project

**Title:** Feature Dropout for Subject-Independent Activity Recognition: Benefits and Failure Modes

This package contains the revised two-dataset empirical study, the complete implementation and data, measured results, editable IEEE conference LaTeX, compiled paper, and a critical review. No conference submission has been made. The manuscript is anonymous because author details and the target conference were not supplied.

## Start here

- `paper/final_paper.pdf`: canonical revised manuscript.
- `latex/main.tex`: complete editable IEEE conference source; `references.bib` contains 17 cited references.
- `review/PEER_REVIEW.md`: readiness assessment, resolved issues, and remaining limitations.
- `review/SUBMISSION_CHECKLIST.md`: actions requiring the actual authors and a named venue.
- `implementation/README_REVISION.md`: installation, reproduction, results, and inference instructions.
- `implementation/results_revision/`: final two-dataset evidence; the paper's numbers come from here.
- `implementation/results/`: historical UCI-only results required for reproducing the follow-up audit.
- `implementation/data/`: both public datasets, processed WISDM windows, checksums, and original notices.
- `implementation/paper/` and `implementation/walkthrough.ipynb`: historical single-dataset material, **not the final paper**.
- `MANIFEST_SHA256.json`: hashes of distributed files (excluding the manifest itself).

## What was completed

The follow-up added raw-data WISDM preprocessing, subject-disjoint validation, stronger regularization controls, ten corruption seeds, structured feature removal, raw-axis flatlining, paired subject-bootstrap intervals, and multiplicity-adjusted equal-subject comparisons. The full test suite passed 21 tests. Tables and the vector figure are generated from saved results, not manually invented values. The complete follow-up took about 3.9 minutes on the execution host using two numerical threads, in addition to the earlier UCI experiment.

## Honest submission status

The paper is a reproducible empirical study of an established augmentation method. Its suitability is strongest for a student conference, workshop, or a venue that welcomes careful empirical studies. It is not certified ready for an unnamed conference, and a high-selectivity main track may require substantially more novelty and evidence. Human review, author information, the venue's page limit/template, anonymity requirements, and AI-assistance disclosure must be resolved before submission.

The text was rewritten with source attribution. No Turnitin or other similarity report was obtained, and no score or detector outcome is guaranteed. Do not remove legitimate citations or the assistance disclosure to manipulate a detector.

Dataset rights remain with their providers; preserve all original notices when redistributing. The UCI website and legacy archive have different license statements, both documented in the implementation README. WISDM requests inclusion of its original readme, which is supplied under `data/wisdm_notices/`.
