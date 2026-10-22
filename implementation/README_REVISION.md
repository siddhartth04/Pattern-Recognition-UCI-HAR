# Reproducing the revised two-dataset study

Use Python 3.11. Open a terminal **in this implementation directory**. The included results and models allow inspection and prediction without retraining. A GPU is unnecessary.

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
& .venv/Scripts/python.exe -m pytest -q
```

On macOS/Linux replace `& .venv/Scripts/python.exe` with `.venv/bin/python`. Allow roughly 1 GB for the data, working files and environment. Only load the bundled trusted joblib files or models you trained yourself.

## Full reproduction

These commands overwrite their generated results. Run them in a copy of the package to preserve the delivered evidence.

```powershell
& .venv/Scripts/python.exe -m src.data
& .venv/Scripts/python.exe -m src.experiment
& .venv/Scripts/python.exe -m src.wisdm
& .venv/Scripts/python.exe -m src.revision
& .venv/Scripts/python.exe -m src.revision_report
& .venv/Scripts/python.exe -m pytest -q
```

The historical UCI experiment must exist before `src.revision`, which reuses its models and validation records. Its outputs are already included. Download scripts verify pinned archive hashes; when archives are present they use local copies. `revision_protocol.json` documents the pre-WISDM-evaluation design and timestamp-cleaning amendment. `results_revision/protocol_snapshot.json` records the protocol used by the supplied run. Training used scikit-learn 1.5.2 and NumPy 1.26.4; complete versions are in `requirements-lock.txt`.

## Inspect final results

For each dataset in `results_revision/uci` and `results_revision/wisdm`:

- `selection.json`, `cv_results.csv`, `folds.json`: model selection and subject boundaries.
- `test_metrics.csv`: every clean/corrupted condition (504 UCI, 435 WISDM rows).
- `predictions.npz`: labels, subject IDs, and every model/condition prediction vector.
- `paired_comparisons.csv`: pooled macro-F1 differences and descriptive subject-bootstrap intervals.
- `subject_signflip_tests.csv`: the distinct equal-subject comparison, with Holm correction over nine conditions per dataset.
- `structured_groups.json`: exact feature indices replaced in the structured tests.
- `subject_metrics.csv`: per-subject scores; `observed_class_delta` uses only labels present for that subject.

WISDM's selected dropout model uses C=1 and p=0.3. The matched-C clean LR derives C from that selection and is saved as a separate model. UCI's expanded audit retains C=0.1 for both LR variants and p=0.1 for dropout. The expanded UCI model outputs are duplicates by design, confirming that the stronger regularization grid did not change the choice.

## Prediction

Original UCI CSV CLI:

```powershell
& .venv/Scripts/python.exe -m src.predict --model results/models/dropout-lr.joblib --input examples/features.csv --output examples/predictions.csv
```

WISDM inference in Python, using 200-sample windows with three axes in the released units and orientation:

```python
import joblib
import numpy as np
from src.wisdm import extract_features, ACTIVITIES
from src.models import predict
data = np.load('data/wisdm_prepared.npz')
model = joblib.load('results_revision/wisdm/models/dropout-lr.joblib')
features = extract_features(data['windows'][data['test'][:5]])
labels = predict(model, features)
print([ACTIVITIES[int(label)-1] for label in labels])
```

The generic predictor accepts feature arrays and replaces NaNs by training feature means. Raw WISDM windows must be finite before extraction. This is a research inference example, not an independently validated live-phone deployment.

## Build the final paper

`python -m src.revision_report` regenerates `../latex/generated/*.tex` and `../latex/figures/robustness.pdf` from final CSVs. Manuscript prose lives in `../latex/main.tex`; the historical `src.build_paper` generates the older UCI-only draft and is **not** the final manuscript builder.

From `../latex`, use a TeX installation with IEEEtran:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternatively `tectonic main.tex` builds automatically (first use may download TeX resources). The delivered PDF was compiled with Tectonic 0.17.0 and rendered for visual inspection. Keep the T1 font encoding declaration before the document class when using Tectonic. Source, bibliography, generated tables, vector figure and prebuilt PDF are all supplied.

## Protocol limitations

UCI follow-up is exploratory because its test results were previously inspected. WISDM uses one held-out subject split and one augmentation training seed. Subject bootstraps condition on the trained models and fixed corruptions. No deep model, device-energy measurement, physical failure trial, or repeated outer split is claimed. See the paper and review report for the consequences.
