import joblib
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from src.data import load_split
from src.models import corrupt, fit_model, predict
from src.experiment import f1_from_cm, subject_bootstrap

@pytest.fixture
def toy():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 8))
    y = 1 + (X[:,0] > 0).astype(int)
    return X, y

def test_official_split_and_cv_are_subject_disjoint():
    X, y, g = load_split('train')
    _, _, h = load_split('test')
    assert set(g).isdisjoint(h)
    assert len(set(g)) == 21 and len(set(h)) == 9
    for train, val in GroupKFold(3).split(X,y,g):
        assert set(g[train]).isdisjoint(g[val])

def test_corruption_endpoints_and_reproducibility():
    Z = np.ones((100,20))
    np.testing.assert_array_equal(corrupt(Z,'dropout',0,1), Z)
    assert not corrupt(Z,'dropout',1,1).any()
    np.testing.assert_array_equal(corrupt(Z,'dropout',.3,1), corrupt(Z,'dropout',.3,1))
    assert .25 < (corrupt(Z,'dropout',.3,1) == 0).mean() < .35
    np.testing.assert_array_equal(corrupt(Z,'noise',0,1), Z)
    np.testing.assert_array_equal(Z, np.ones_like(Z))

def test_scaler_never_refits_during_inference(toy):
    X,y = toy
    model = fit_model(X,y,{'family':'dropout_lr','C':1,'p':.3})
    np.testing.assert_allclose(model['scaler'].mean_, X.mean(axis=0))
    original = model['scaler'].mean_.copy()
    predict(model, X+100)
    np.testing.assert_array_equal(model['scaler'].mean_, original)

def test_nan_imputation_and_shape_validation(toy):
    X,y = toy
    model = fit_model(X,y,{'family':'lr','C':1})
    missing = X.copy(); missing[:,2] = np.nan
    imputed = X.copy(); imputed[:,2] = model['scaler'].mean_[2]
    np.testing.assert_array_equal(predict(model,missing),predict(model,imputed))
    with pytest.raises(ValueError): predict(model, X[:,:-1])
    bad = X.copy(); bad[0,0] = np.inf
    with pytest.raises(ValueError): predict(model, bad)

def test_serialization_round_trip(toy, tmp_path):
    X,y = toy
    model = fit_model(X,y,{'family':'dropout_lr','C':1,'p':.3})
    joblib.dump(model,tmp_path/'model.joblib')
    np.testing.assert_array_equal(predict(model,X),predict(joblib.load(tmp_path/'model.joblib'),X))

def test_zero_dropout_has_same_regularization_as_clean(toy):
    X,y = toy
    baseline = fit_model(X,y,{'family':'lr','C':.1})
    augmented = fit_model(X,y,{'family':'dropout_lr','C':.1,'p':0})
    np.testing.assert_allclose(baseline['classifier'].coef_, augmented['classifier'].coef_, atol=1e-8)

def test_bootstrap_metric_matches_sklearn():
    y = np.tile(np.arange(1,7), 5)
    pred = y.copy(); pred[:4] = 1
    expected = f1_score(y,pred,labels=np.arange(1,7),average='macro')
    assert f1_from_cm(confusion_matrix(y,pred,labels=np.arange(1,7))) == pytest.approx(expected)
    groups = np.repeat(np.arange(5),6)
    values = subject_bootstrap(y,{'a':[pred], 'b':[pred]},groups,100,42)
    np.testing.assert_array_equal(values['a']['draws'],values['b']['draws'])
    assert values['a']['estimate'] == pytest.approx(expected)

def test_training_reproducibility(toy):
    X,y = toy
    params = {'family':'dropout_lr','C':1,'p':.3}
    a,b = fit_model(X,y,params,42),fit_model(X,y,params,42)
    np.testing.assert_allclose(a['classifier'].coef_,b['classifier'].coef_)
