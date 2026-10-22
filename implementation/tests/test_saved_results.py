"""Integration checks over actual experiment outputs; skip before first run."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score
from src.data import ROOT, load_split
from src.models import predict

pytestmark = pytest.mark.skipif(not (ROOT/'results/run_manifest.json').exists(), reason='Run experiment first')

def test_saved_metrics_recompute_from_predictions():
    values = np.load(ROOT/'results/predictions.npz')
    rows = pd.read_csv(ROOT/'results/test_metrics.csv')
    assert len(rows)==175
    for row in rows.itertuples():
        pred = values[f'{row.model}|{row.kind}|{row.severity}|{row.corruption_seed}']
        assert row.accuracy == pytest.approx(accuracy_score(values['y'],pred))
        assert row.macro_f1 == pytest.approx(f1_score(values['y'],pred,average='macro'))

def test_saved_model_matches_clean_test_predictions():
    X,_,_ = load_split('test')
    model = joblib.load(ROOT/'results/models/dropout-lr.joblib')
    values = np.load(ROOT/'results/predictions.npz')
    np.testing.assert_array_equal(predict(model,X),values['Dropout-LR|clean|0.0|0'])

def test_selection_matches_cv_maximum():
    rows=pd.read_csv(ROOT/'results/cv_results.csv')
    selected=json.loads((ROOT/'results/selection.json').read_text())
    assert len(rows)==48
    for model in ['LR','Dropout-LR','RBF-SVM','Random forest']:
        best=rows[rows.model==model].groupby('params').selection_score.mean().max()
        assert selected[model]['cv_score']==pytest.approx(best)

def test_bootstrap_outputs_are_valid():
    intervals=pd.read_csv(ROOT/'results/bootstrap_intervals.csv')
    assert len(intervals)==63
    assert np.isfinite(intervals[['estimate','low','high']]).all().all()
    assert (intervals.low <= intervals.high).all()
    assert ((intervals.low>=0)&(intervals.high<=1)).all()
