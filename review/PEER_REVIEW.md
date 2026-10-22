# Research review and revision report

## Verdict

**Original version: major revision needed. Revised version: a substantially stronger, reproducible empirical manuscript, with conditional venue suitability.** This is an internal AI-assisted critical review, not an independent conference peer-review decision.

Yes, the work fits pattern recognition: it extracts representations from inertial signals and learns multiclass decision rules that generalize to unseen participants. The appropriate contribution is a controlled study of missing-feature robustness and its limits. It is not a novel dropout algorithm or a state-of-the-art HAR system.

For a workshop or student conference welcoming empirical studies, the revised manuscript is a credible candidate after human and venue checks. For a selective main research track, I would still expect concerns about incremental novelty, limited dataset breadth and split variability, and the absence of strong modern temporal or specialized missing-data comparators. Formatting and more references cannot by themselves resolve those concerns.

## Major issues addressed

| Initial concern | Revision and evidence | Residual limitation |
|---|---|---|
| Only one dataset | Added original WISDM v1.1 with 36 subjects, raw-to-feature preprocessing, 25/11 subject split | Two datasets do not support universal ranking |
| Weak missingness interpretation | Added feature-family removal and WISDM raw-axis flatlining before feature extraction | Synthetic faults; no physical device validation |
| Regularization confounding | Expanded UCI C grid; added WISDM clean LR matched at dropout's C | Modest search spaces; not exhaustive family tuning |
| Limited corruption sampling | Increased stochastic test conditions from three to ten paired seeds | One main WISDM augmentation training seed |
| Window-level uncertainty risk | 5,000 paired subject-cluster bootstrap replicates | Only nine/eleven held-out subjects; conditional intervals |
| Multiple claims without adjustment | Added exact subject sign-flip tests and Holm adjustment over nine conditions per dataset | Different estimand from pooled F1; structured tests exploratory |
| Missing provenance | Archive checksums, WISDM cleaning counts, subject IDs, source hashes and protocol snapshot | No external preregistration |
| Too little related work | Expanded to 17 relevant references; distinguishes finite augmentation, marginalized features, and neural dropout | Additional venue-specific literature review remains sensible |
| Reproducibility | Full data, fitted models, predictions, generated tables/figure, and 21 passing tests | Tests demonstrate consistency, not independent replication |
| Overstated submission readiness | Anonymous IEEE source, explicit contribution boundary, AI-assistance statement and venue checklist | Actual authors and target venue unknown |

## Findings that must not be hidden

At 50% missing entries, dropout LR improves pooled macro-F1 over LR by 1.13 percentage points on UCI HAR and 17.21 on WISDM. The corresponding subject-bootstrap intervals exclude zero; the separate equal-subject tests survive within-dataset Holm correction. This supports a narrow missing-feature robustness claim.

However, WISDM random forest scores 64.04% at the same condition versus dropout LR's 63.50%. Clean SVM is stronger on both datasets. WISDM Gaussian-noise results do not consistently improve. Raw-axis flatlining gives mixed differences: -0.22, +2.50 and +16.71 points for x, y and z. Under raw y flatlining, dropout LR achieves only 24.90% macro-F1. These findings prevent a broad claim of sensor-failure robustness or overall superiority.

## Methodological audit

- Scaling is fitted on the training fold only; participant IDs do not overlap across train/validation/test boundaries.
- The same clean-plus-30%-missing selection objective is used for all main model families. Clean-trained LR is not tuned on clean performance alone.
- Each augmented example pair has weights 0.5 and 0.5, preserving regularization scale at fixed C.
- WISDM timestamps are sorted within contiguous participant/activity recordings, deduplicated, and split at gaps over one second. The amendment was made before model fitting. Its irregular window durations and absence of resampling are explicitly reported.
- The bootstrap retains subject clusters and pairs model comparisons. Its point estimate is pooled macro-F1 averaged over corruption seeds.
- The exact sign-flip test uses equal-subject differences and labels observed per subject. The paper explicitly distinguishes this from the bootstrap estimand. Its symmetry/exchangeability assumption and small subject count limit inference.
- UCI follow-up does not regain an untouched test set simply because the expanded C selection uses training data. The manuscript correctly calls it exploratory.
- Ten corruption seeds are repeated perturbations of the same subjects, not ten independent datasets.

## Remaining work for stronger research claims

Repeated outer subject splits, several training seeds on WISDM, timestamp-preprocessing sensitivity, further independent datasets, specialized missing-feature comparators and temporal neural baselines would strengthen the empirical scope. Actual fault measurements are necessary for deployment claims. They are not silently represented as completed. The manuscript narrows its claims to the evidence instead.

## Originality and similarity review

The revised prose synthesizes the study and cites prior methods; it does not claim an established dropout mechanism as novel. References were checked against primary paper, publisher or dataset sources during preparation. Preprints are labelled as preprints. There is no fabricated acceptance, similarity score, external review, or independent reproduction.

No Turnitin access or report was used. A similarity report must be interpreted by a human: bibliography, standard methods and dataset names can legitimately match other documents. The objective is accurate attribution and honest authorship, not bypassing detection. The manuscript discloses AI assistance; actual authors remain responsible for reviewing and defending the work.
