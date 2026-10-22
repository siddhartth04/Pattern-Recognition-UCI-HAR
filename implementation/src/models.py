"""Training and inference; all learned preprocessing uses training rows only."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def corrupt(Z, kind, severity, seed):
    if not np.isfinite(Z).all():
        raise ValueError('Corruption expects finite standardized input')
    rng = np.random.default_rng(seed)
    if kind == 'clean':
        return Z.copy()
    if kind == 'dropout':
        if not 0 <= severity <= 1:
            raise ValueError('Dropout probability must be in [0, 1]')
        return Z * (rng.random(Z.shape) >= severity)
    if kind == 'noise':
        if severity < 0:
            raise ValueError('Noise scale must be nonnegative')
        return Z + rng.normal(0, severity, Z.shape)
    raise ValueError(f'Unknown corruption: {kind}')

def fit_model(X, y, params, seed=2026, threads=2):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    family = params['family']
    sample_weight = None
    if family in ['lr', 'dropout_lr']:
        clf = LogisticRegression(C=params['C'], max_iter=2000, tol=1e-4, solver='lbfgs')
        if family == 'dropout_lr':
            Z = np.vstack([Z, corrupt(Z, 'dropout', params['p'], seed)])
            y = np.tile(y, 2)
            # Keep total sample weight n so regularization is comparable to clean LR.
            sample_weight = np.full(len(y), 0.5)
    elif family == 'svm':
        clf = SVC(C=params['C'], gamma='scale', kernel='rbf', cache_size=512)
    elif family == 'rf':
        clf = RandomForestClassifier(n_estimators=params['trees'],
              max_features=params['max_features'], random_state=seed, n_jobs=threads)
    else:
        raise ValueError(f'Unknown family: {family}')
    clf.fit(Z, y, sample_weight=sample_weight)
    return {'scaler': scaler, 'classifier': clf, 'params': params, 'seed': seed,
            'feature_count': X.shape[1]}

def predict(bundle, X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != bundle['feature_count']:
        raise ValueError(f"Expected an n-by-{bundle['feature_count']} feature matrix")
    if np.isinf(X).any():
        raise ValueError('Infinite inputs are invalid; use NaN for missing features')
    # Missing extracted features are imputed with training feature means.
    Z = bundle['scaler'].transform(X)
    Z = np.nan_to_num(Z, nan=0.0)
    return bundle['classifier'].predict(Z)
