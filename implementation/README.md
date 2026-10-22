# Historical UCI-only implementation notes

**For the final two-dataset study, read `README_REVISION.md` and the project-root README. The manuscript in this directory's `paper/` is the earlier draft. The canonical revised paper is `../paper/final_paper.pdf`. Numbers and test counts below describe the original run, not the final follow-up.**

## Feature-dropout robustness for smartphone activity recognition

A complete CPU-based pattern recognition research project on the public UCI HAR dataset. The package includes source code, prepared dataset files, fitted models, per-example predictions, validation records, stress tests, figures, an editable paper, and a conference-style PDF.

## Research question

Does training multinomial logistic regression on clean and randomly masked features improve recognition of unseen people's activities when extracted feature entries are unavailable? How does it compare with clean-trained logistic regression, an RBF SVM, and a random forest?

This is a reproducible empirical study of established methods. It does not claim a new dropout algorithm, state-of-the-art performance, or validation on physical sensor failures. The implementation begins with UCI's published 561-feature representation; it does not extract features from a live phone stream.

## Setup (Python 3.11)

Open a terminal in this folder. On Windows PowerShell:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
& .venv/Scripts/python.exe -m src.data
& .venv/Scripts/python.exe -m pytest -q
& .venv/Scripts/python.exe -m src.experiment
& .venv/Scripts/python.exe -m src.figures
& .venv/Scripts/python.exe -m src.build_paper
& .venv/Scripts/python.exe -m pytest -q
```

On macOS/Linux, replace `& .venv/Scripts/python.exe` with `.venv/bin/python`. No GPU or account is required. `requirements.txt` pins the main libraries; `requirements-lock.txt` records the complete tested environment. The initial download is approximately 58 MB. Allow at least 1 GB of free disk space for a virtual environment, dataset and results.

`python -m src.experiment --output results_reproduction` preserves the supplied results while repeating the full protocol. The figure and paper builders use the supplied `results/` directory by default. Rerunning the default experiment replaces those outputs. Numerical results should reproduce with the tested versions; wall-clock measurements vary by hardware and background load.

The supplied full run took 4.94 minutes on the execution host with a two-thread limit. All 12 tests passed. The five-page PDF was rendered and visually checked. The selected Dropout-LR model achieved 94.88% clean accuracy; the RBF SVM achieved the highest clean accuracy among the four main methods (95.22%). At 50% missing feature entries, Dropout-LR improved macro-F1 over the validation-selected clean LR comparator from 86.07% to 87.11%. The paired subject-bootstrap interval for the difference was +0.64 to +1.44 percentage points.

`walkthrough.ipynb` contains five executed code cells for loading the data, examining selection and metrics, inspecting uncertainty, and making predictions. Open it in a Jupyter-capable editor using the project environment; a notebook UI is optional and not required for any command above.

## Run prediction

```powershell
& .venv/Scripts/python.exe -m src.predict --model results/models/dropout-lr.joblib --input examples/features.csv --output examples/predictions.csv
```

Input is a headerless numeric CSV with exactly 561 columns in the order of `data/UCI HAR Dataset/features.txt`. Use `nan` for unavailable features. Missing values are replaced with training-set means. Infinite values and incorrect dimensions are rejected. Activity labels are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING and LAYING. Only load trusted joblib files.

## Evaluation protocol

- Official split: 7,352 training windows from 21 subjects; 2,947 test windows from 9 different subjects.
- Three subject-disjoint validation folds within the training split. Scalers are fit independently on each training fold.
- Every model is selected by the equal mean of clean macro-F1 and macro-F1 at 30% independent feature-entry masking. The latter averages three fixed corruption seeds.
- Sixteen configurations, 48 validation fits. All choices are saved before loading test arrays.
- The augmentation method uses one clean and one masked copy per training example, each weighted 0.5. This preserves total sample weight and the regularization comparison.
- Four main methods; a matched-C logistic-regression ablation; two additional augmentation training seeds.
- Clean evaluation plus four dropout rates and four Gaussian noise levels, each with three matched test corruption seeds. These are feature-space stress tests.
- 2,000 paired subject-cluster bootstrap replicates. The nine test subjects, not individual overlapping windows, are resampled. Intervals are conditional on fixed fitted models and corruption realizations.

## Files

| Location | Purpose |
|---|---|
| `config.json` | Fixed hyperparameter grids, seeds, corruption levels and selection rule |
| `walkthrough.ipynb` | Executed, beginner-friendly data-to-prediction walkthrough |
| `src/data.py` | Official downloader, pinned SHA-256 check, safe member extraction, loading |
| `src/models.py` | Models, augmentation, preprocessing and inference |
| `src/experiment.py` | Validation, training, evaluation, bootstrap and resource measurements |
| `src/predict.py` | CSV inference command |
| `src/figures.py` | Plots from measured results |
| `src/build_paper.py` | PDF and Markdown manuscript generation |
| `tests/` | Unit tests and result consistency checks |
| `results/cv_results.csv` | Every validation configuration and fold |
| `results/selection.json` | Validation-selected configurations and comparator |
| `results/test_metrics.csv` | All 175 model-condition evaluations |
| `results/predictions.npz` | Full predictions for independent verification |
| `results/paired_comparisons.csv` | Paired macro-F1 differences and uncertainty |
| `results/resources.csv` | Fit time, batch prediction time and compressed model size |
| `results/run_manifest.json` | Versions, configuration, source hashes and runtime |
| `results/models/` | Seven fitted model bundles |
| `paper/` | Final manuscript and research notes |

## Dataset provenance and attribution

Dataset: J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto and X. Parra, *Human Activity Recognition Using Smartphones*, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C54S4K.

Original paper: D. Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013, pp. 437-442, https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf.

The UCI landing page currently states CC BY 4.0. The legacy archive README also contains an older noncommercial-use statement; both original notices are preserved. This package is prepared for academic research. The data archive checksum and member checksums are recorded in `data/manifest.json`. The download command checks the archive against the exact SHA-256 used in the reported run.

## Paper status

The paper reports actual executed experiments, including limitations and unfavorable results. Its contribution is the controlled evaluation and reproducible artifact, not an asserted algorithmic novelty. Author names and affiliations are deliberately omitted. Before submission, add the actual authors, follow the chosen venue's template and disclosure rules, and review whether its novelty expectations require further datasets or experiments. No conference acceptance is implied.
