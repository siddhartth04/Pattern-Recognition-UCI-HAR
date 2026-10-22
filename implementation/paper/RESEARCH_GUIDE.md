# How to explain and extend this project

## Suggested presentation

1. **Problem:** identify six daily activities from a phone's published accelerometer/gyroscope feature representation.
2. **Research question:** whether simple feature dropout improves recognition when feature entries are missing, and whether that benefit transfers to additive noise.
3. **Dataset choice:** UCI HAR is public, small enough for CPU training, has interpretable activity labels, and includes subject IDs needed for realistic generalization checks.
4. **Implementation:** checksum-verified download, train-only scaling, four classifiers, subject-disjoint hyperparameter selection, saved model inference, stress tests and uncertainty.
5. **Result:** use the measured tables in the paper. Explain both the clean-performance tradeoff and the uncertainty; never substitute validation scores for held-out test results.
6. **Contribution:** a controlled and reproducible robustness evaluation, with a matched-regularization ablation and subject-level uncertainty. Feature dropout itself is established prior work.

## Concepts to understand before a viva

- A **feature vector** describes one activity window with 561 numerical values. The model receives this vector, not a photograph or an activity name.
- **Standardization** subtracts each feature's training mean and divides by its training standard deviation. Zero in that space means the training mean.
- **Feature dropout** replaces a random subset of standardized entries by zero. The same labels are retained during augmentation.
- **Macro-F1** is the unweighted mean of the six activity-specific F1 scores. It prevents the most frequent activity from entirely determining the metric.
- **Subject-disjoint validation** prevents windows from one person appearing in both training and validation. The official test people are also separate.
- **Data leakage** occurs when information unavailable at prediction time influences training or selection. Fitting a scaler on all rows before splitting would be an example.
- A **paired bootstrap** resamples the same people for two models. This preserves the paired comparison and avoids pretending that overlapping windows are independent people.
- An interval that includes zero does not establish an improvement; an interval conditional on nine subjects is also not a universal population guarantee.

## Why not claim physical sensor-failure robustness?

The dropout masks act independently on derived feature entries. A failed gyroscope would remove many correlated measurements and derived features together, possibly across a long time interval. Gaussian noise added after feature extraction also does not recreate the effect of noise in raw motion signals. Such experiments require a different, explicit corruption and feature-extraction pipeline.

## Sensible further research

For a stronger conference submission, first select a target venue and inspect its scope. A useful next study would introduce raw-signal corruption before feature extraction, evaluate an additional HAR dataset under an explicitly comparable protocol, and compare against an appropriate recent robust-learning method. These would be new experiments, not completed results in this package. If designing new hyperparameters after studying the current test set, reserve a new untouched evaluation dataset; do not claim that repeated tuning preserves a fresh test.

## Submission tasks

- Add actual author names, affiliations and contact information.
- Move the manuscript into the exact conference template and page limit.
- Check references, mathematical definitions and all tables with the source results.
- Follow the venue's policies for AI assistance, authorship and disclosure.
- Have every listed author understand, verify and approve the work.

The supplied PDF is a completed empirical manuscript in a generic conference style. It is not an accepted paper or a claim that the method meets any particular conference's novelty threshold.
