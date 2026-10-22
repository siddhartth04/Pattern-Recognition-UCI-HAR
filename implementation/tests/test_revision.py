import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from src.data import ROOT
from src.wisdm import extract_features,axis_group,FEATURES
from src.revision import exact_signflip

def test_features_constant_window_and_schema():
    W=np.ones((2,200,3)); f=extract_features(W)
    assert f.shape==(2,51) and len(FEATURES)==51 and len(set(FEATURES))==51
    assert np.isfinite(f).all()
    assert f[0,0]==1 and f[0,1]==0 and f[0,9]==0
    np.testing.assert_array_equal(f[:,-3:],0)

def test_raw_axis_failure_and_group_dependencies():
    W=np.random.default_rng(9).normal(size=(2,200,3))
    altered=W.copy(); altered[:,:,0]=0
    a,b=extract_features(W),extract_features(altered)
    changed=set(np.flatnonzero(np.any(np.abs(a-b)>1e-10,axis=0)))
    assert changed.issubset(set(axis_group('x')))
    assert len(axis_group('x'))==26
    assert set(range(12)).issubset(axis_group('x'))
    np.testing.assert_array_equal(a[:,12:36],b[:,12:36])

def test_invalid_feature_input_rejected():
    with pytest.raises(ValueError): extract_features(np.zeros((10,3)))
    with pytest.raises(ValueError): extract_features(np.full((1,200,3),np.nan))

def test_exact_subject_signflip():
    assert exact_signflip(np.ones(9))==pytest.approx(2/512)
    assert exact_signflip(np.zeros(9))==1
    assert exact_signflip(np.array([1.,-1.]))==1

def test_wisdm_integrity_and_subject_split():
    p=np.load(ROOT/'data/wisdm_prepared.npz')
    train,test=p['train'],p['test']
    assert len(train)+len(test)==4933
    assert set(p['subjects'][train]).isdisjoint(p['subjects'][test])
    assert len(np.unique(p['subjects'][train]))==25 and len(np.unique(p['subjects'][test]))==11
    assert set(p['y'][train])==set(p['y'][test])==set(range(1,7))
    np.testing.assert_allclose(p['X'][:10],extract_features(p['windows'][:10]))
    m=json.loads((ROOT/'data/wisdm_manifest.json').read_text())
    c=m['counts']
    assert c['valid_records']-c['duplicate_timestamps_removed']-c['discarded_tail_samples']==4933*200

@pytest.mark.parametrize('dataset',['uci','wisdm'])
def test_revision_scores_recompute(dataset):
    folder=ROOT/'results_revision'/dataset
    if not (folder/'test_metrics.csv').exists(): pytest.skip('Run revision experiments first')
    scores=pd.read_csv(folder/'test_metrics.csv'); arrays=np.load(folder/'predictions.npz')
    assert len(scores)==(6*84 if dataset=='uci' else 5*87)
    for row in scores.itertuples():
        pred=arrays[f'{row.model}|{row.kind}|{row.severity}|{row.seed}']
        assert row.macro_f1==pytest.approx(f1_score(arrays['y'],pred,labels=np.arange(1,7),average='macro'))

@pytest.mark.parametrize('dataset',['uci','wisdm'])
def test_revision_fold_separation_and_adjusted_tests(dataset):
    folder=ROOT/'results_revision'/dataset
    if not (folder/'subject_signflip_tests.csv').exists(): pytest.skip('Run revision experiments first')
    folds=json.loads((folder/'folds.json').read_text())
    for fold in folds: assert set(fold['train_subjects']).isdisjoint(fold['validation_subjects'])
    rows=pd.read_csv(folder/'subject_signflip_tests.csv').dropna(subset=['holm_p'])
    assert len(rows)==9
    assert (rows.holm_p>=rows.signflip_p-1e-12).all() and (rows.holm_p<=1).all()
