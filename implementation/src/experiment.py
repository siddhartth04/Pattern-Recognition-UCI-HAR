"""Run the fixed protocol. Selection finishes before test data are loaded."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .data import ROOT, download, load_split
from .models import fit_model, corrupt

LABELS = np.arange(1, 7)

def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0))}

def f1_from_cm(cm):
    numerator = 2 * np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0).mean(axis=-1)

def subject_bootstrap(y, prediction_sets, groups, B, seed):
    """Paired subject-cluster resampling; average F1 over fixed corruption seeds."""
    subjects = np.unique(groups)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(subjects), np.ones(len(subjects))/len(subjects), size=B)
    output = {}
    for name, predictions in prediction_sets.items():
        matrices = np.array([[confusion_matrix(y[groups == g], pred[groups == g], labels=LABELS)
                              for g in subjects] for pred in predictions])
        sampled = np.einsum('bg,sgij->bsij', counts, matrices)
        values = f1_from_cm(sampled).mean(axis=1)
        point = float(np.mean([metrics(y, p)['macro_f1'] for p in predictions]))
        output[name] = {'estimate': point, 'low': float(np.quantile(values, .025)),
                        'high': float(np.quantile(values, .975)), 'draws': values}
    return output

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')

def run(output, config):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / 'models'
    models_dir.mkdir(exist_ok=True)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    download()
    X, y, groups = load_split('train')
    folds = list(GroupKFold(config['cv_folds']).split(X, y, groups))
    fold_manifest = []
    for train, val in folds:
        assert set(groups[train]).isdisjoint(groups[val])
        fold_manifest.append({'train_subjects': sorted(set(groups[train].tolist())),
                              'validation_subjects': sorted(set(groups[val].tolist())),
                              'train_n': len(train), 'validation_n': len(val)})
    write_json(output / 'cv_folds.json', fold_manifest)
    candidates = {
        'LR': [{'family': 'lr', 'C': c} for c in config['C_grid']],
        'Dropout-LR': [{'family': 'dropout_lr', 'C': c, 'p': p}
                       for c in config['C_grid'] for p in config['train_dropout_grid']],
        'RBF-SVM': [{'family': 'svm', 'C': c} for c in config['svm_C_grid']],
        'Random forest': [{'family': 'rf', 'max_features': m, 'trees': config['rf_trees']}
                          for m in config['rf_max_features_grid']]}
    cv_rows, selection = [], {}
    for model_name, grid in candidates.items():
        candidate_scores = []
        for candidate_id, params in enumerate(grid):
            scores = []
            for fold, (train, val) in enumerate(folds):
                t0 = time.perf_counter()
                bundle = fit_model(X[train], y[train], params, config['seed']+fold, config['threads'])
                Z = bundle['scaler'].transform(X[val])
                clean = metrics(y[val], bundle['classifier'].predict(Z))['macro_f1']
                dropout = np.mean([metrics(y[val], bundle['classifier'].predict(
                    corrupt(Z, 'dropout', config['validation_dropout'], s)))['macro_f1']
                    for s in config['validation_corruption_seeds']])
                score = (clean + dropout) / 2
                scores.append(score)
                cv_rows.append({'model': model_name, 'candidate': candidate_id, 'fold': fold,
                    'params': json.dumps(params, sort_keys=True), 'clean_macro_f1': clean,
                    'dropout_macro_f1': dropout, 'selection_score': score,
                    'seconds': time.perf_counter()-t0})
            candidate_scores.append(float(np.mean(scores)))
            print(f'CV {model_name} {params}: {np.mean(scores):.5f}', flush=True)
            pd.DataFrame(cv_rows).to_csv(output / 'cv_results.csv', index=False)
        best = int(np.argmax(candidate_scores))
        selection[model_name] = {'params': grid[best], 'cv_score': candidate_scores[best]}
    baseline = max(['LR','RBF-SVM','Random forest'], key=lambda n: selection[n]['cv_score'])
    selection['primary_baseline'] = baseline
    selection['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(output / 'selection.json', selection)
    # The selection file above is frozen before loading the official test arrays.
    X_test, y_test, test_groups = load_split('test')
    assert set(groups).isdisjoint(test_groups)
    write_json(output / 'dataset_summary.json', {
        'train_n': len(y), 'test_n': len(y_test), 'features': X.shape[1],
        'train_subjects': sorted(set(groups.tolist())), 'test_subjects': sorted(set(test_groups.tolist())),
        'train_class_counts': np.bincount(y)[1:].tolist(),
        'test_class_counts': np.bincount(y_test)[1:].tolist()})
    conditions = [('clean', 0.0, 0)]
    conditions += [('dropout', p, s) for p in config['test_dropout_rates'] for s in config['test_corruption_seeds']]
    conditions += [('noise', p, s) for p in config['test_noise_sigmas'] for s in config['test_corruption_seeds']]
    all_models = {name: (selection[name]['params'], config['seed']) for name in candidates}
    all_models['LR matched C'] = ({'family': 'lr', 'C': selection['Dropout-LR']['params']['C']}, config['seed'])
    for seed in config['augmentation_sensitivity_seeds']:
        all_models[f'Dropout-LR seed {seed}'] = (selection['Dropout-LR']['params'], seed)
    metric_rows, predictions, resources = [], {'y': y_test, 'subjects': test_groups}, []
    cached_predictions = {}
    per_class = {}
    for model_name, (params, train_seed) in all_models.items():
        t0 = time.perf_counter()
        bundle = fit_model(X, y, params, train_seed, config['threads'])
        fit_seconds = time.perf_counter()-t0
        filename = model_name.lower().replace(' ','_') + '.joblib'
        joblib.dump(bundle, models_dir / filename, compress=3)
        Z = bundle['scaler'].transform(X_test)
        # Timed batches include scaling; median of five repetitions, with one warm-up.
        bundle['classifier'].predict(Z[:100])
        latency = []
        for _ in range(5):
            t0 = time.perf_counter()
            bundle['classifier'].predict(bundle['scaler'].transform(X_test))
            latency.append((time.perf_counter()-t0) * 1e6 / len(X_test))
        resources.append({'model': model_name, 'fit_seconds': fit_seconds,
            'predict_us_per_row': float(np.median(latency)),
            'model_kib': (models_dir / filename).stat().st_size/1024})
        for kind, severity, seed in conditions:
            pred = bundle['classifier'].predict(corrupt(Z, kind, severity, seed))
            key = f'{model_name}|{kind}|{severity}|{seed}'
            predictions[key] = pred
            cached_predictions.setdefault((kind, severity), {}).setdefault(model_name, []).append(pred)
            metric_rows.append({'model': model_name, 'kind': kind, 'severity': severity,
                                'corruption_seed': seed, **metrics(y_test, pred)})
            if kind == 'clean':
                per_class[model_name] = classification_report(y_test, pred, labels=LABELS, output_dict=True)
        print(f'Test completed: {model_name}; fit {fit_seconds:.2f}s', flush=True)
        pd.DataFrame(metric_rows).to_csv(output / 'test_metrics.csv', index=False)
    np.savez_compressed(output / 'predictions.npz', **predictions)
    pd.DataFrame(resources).to_csv(output / 'resources.csv', index=False)
    write_json(output / 'per_class.json', per_class)
    summaries = []
    comparisons = []
    for (kind, severity), prediction_sets in cached_predictions.items():
        boot = subject_bootstrap(y_test, prediction_sets, test_groups,
                                config['bootstrap_replicates'], config['bootstrap_seed'])
        for name, values in boot.items():
            summaries.append({'model': name, 'kind': kind, 'severity': severity,
                **{k: v for k,v in values.items() if k != 'draws'}})
        for reference in dict.fromkeys([baseline, 'LR', 'LR matched C']):
            delta = boot['Dropout-LR']['draws'] - boot[reference]['draws']
            comparisons.append({'kind': kind, 'severity': severity, 'reference': reference,
                'delta': boot['Dropout-LR']['estimate']-boot[reference]['estimate'],
                'low': float(np.quantile(delta, .025)), 'high': float(np.quantile(delta, .975))})
    pd.DataFrame(summaries).to_csv(output / 'bootstrap_intervals.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'paired_comparisons.csv', index=False)
    write_json(output / 'run_manifest.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
        'elapsed_seconds': time.perf_counter()-started, 'config': config,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))},
        'uncertainty': 'Paired percentile subject-cluster bootstrap, 9 held-out subjects, fixed corruption realizations; no refitting.'})
    print('Experiment complete.', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    with threadpool_limits(limits=config['threads']):
        run(args.output, config)

if __name__ == '__main__':
    main()
