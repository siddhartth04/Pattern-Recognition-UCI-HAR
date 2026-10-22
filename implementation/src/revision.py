"""Review-phase experiments. Original UCI findings remain explicitly historical."""
import json,time,hashlib,platform,itertools
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .data import ROOT,load_split
from .models import fit_model,corrupt
from .experiment import metrics,subject_bootstrap,write_json
from .wisdm import extract_features,axis_group

CONFIG=json.loads((ROOT/'revision_protocol.json').read_text())
OUT=ROOT/'results_revision'

def selection(X,y,g,grids,folder,prior=None):
    records=[] if prior is None else prior.to_dict('records')
    folds=list(GroupKFold(CONFIG['cv_folds']).split(X,y,g))
    write_json(folder/'folds.json',[{'train_subjects':np.unique(g[a]).tolist(),'validation_subjects':np.unique(g[b]).tolist()} for a,b in folds])
    for name,grid in grids.items():
        for params in grid:
            for fold,(train,val) in enumerate(folds):
                assert set(g[train]).isdisjoint(g[val])
                model=fit_model(X[train],y[train],params,CONFIG['train_seed']+fold,2)
                z=model['scaler'].transform(X[val]); classifier=model['classifier']
                clean=metrics(y[val],classifier.predict(z))['macro_f1']
                noisy=np.mean([metrics(y[val],classifier.predict(corrupt(z,'dropout',.3,s)))['macro_f1'] for s in CONFIG['validation_seeds']])
                records.append({'model':name,'params':json.dumps(params,sort_keys=True),'fold':fold,'clean_macro_f1':clean,
                                'dropout_macro_f1':noisy,'selection_score':(clean+noisy)/2})
            print('CV',name,params,'done',flush=True)
            pd.DataFrame(records).to_csv(folder/'cv_results.csv',index=False)
    scores=pd.DataFrame(records).groupby(['model','params'],sort=True).selection_score.mean()
    selected={}
    for name in grids:
        score=scores.loc[name]; params=score.idxmax()
        selected[name]={'params':json.loads(params),'cv_score':float(score.loc[params])}
    write_json(folder/'selection.json',selected)
    return selected

def exact_signflip(delta):
    # Test the mean equal-subject difference, under the symmetry/exchangeability null.
    signs=np.array(list(itertools.product([-1.,1.],repeat=len(delta))))
    observed=abs(np.mean(delta))
    return float(np.mean(np.abs(signs@delta/len(delta))>=observed-1e-14))

def evaluate(name,X,y,g,bundles,folder,groups,raw=None):
    rows=[]; predictions={'y':y,'subjects':g}; cache={}; costs=[]
    conditions=[('clean',0.,0)]+[(kind,v,s) for kind,vs in [('dropout',CONFIG['dropout_rates']),('noise',CONFIG['noise_sigmas'])]
                for v in vs for s in CONFIG['test_seeds']]
    conditions += [('group_'+key,1.,0) for key in groups]
    raw_features={}
    if raw is not None:
        for i,axis in enumerate(['x','y','z']):
            modified=raw.copy(); modified[:,:,i]=0
            raw_features['raw_'+axis+'_flatline']=extract_features(modified)
            conditions.append(('raw_'+axis+'_flatline',0.,0))
    for model_name,bundle in bundles.items():
        z=bundle['scaler'].transform(X)
        for kind,level,seed in conditions:
            if kind.startswith('group_'):
                altered=z.copy(); altered[:,groups[kind.removeprefix('group_')]]=0
            elif kind.startswith('raw_'): altered=bundle['scaler'].transform(raw_features[kind])
            else: altered=corrupt(z,kind,level,seed)
            pred=bundle['classifier'].predict(altered)
            predictions[f'{model_name}|{kind}|{level}|{seed}']=pred
            cache.setdefault((kind,level),{}).setdefault(model_name,[]).append(pred)
            rows.append({'dataset':name,'model':model_name,'kind':kind,'severity':level,'seed':seed,**metrics(y,pred)})
        print('Evaluated',name,model_name,flush=True)
    np.savez_compressed(folder/'predictions.npz',**predictions)
    pd.DataFrame(rows).to_csv(folder/'test_metrics.csv',index=False)
    intervals=[]; comparisons=[]; subject_rows=[]; tests=[]
    for (kind,level),sets in cache.items():
        boot=subject_bootstrap(y,sets,g,CONFIG['bootstrap_replicates'],CONFIG['bootstrap_seed'])
        for model,result in boot.items():
            intervals.append({'model':model,'kind':kind,'severity':level,**{k:v for k,v in result.items() if k!='draws'}})
        pairs=[('Dropout-LR','LR')]
        if 'Expanded Dropout-LR' in boot: pairs.append(('Expanded Dropout-LR','Expanded LR'))
        if 'LR matched C' in boot: pairs.append(('Dropout-LR','LR matched C'))
        for method,reference in pairs:
            draws=boot[method]['draws']-boot[reference]['draws']
            comparisons.append({'method':method,'reference':reference,'kind':kind,'severity':level,
                'delta':boot[method]['estimate']-boot[reference]['estimate'],
                'low':float(np.quantile(draws,.025)),'high':float(np.quantile(draws,.975))})
        deltas=[]
        for subject in np.unique(g):
            mask=g==subject
            scores={m:float(np.mean([f1_score(y[mask],p[mask],labels=np.arange(1,7),average='macro',zero_division=0) for p in ps])) for m,ps in sets.items()}
            # No absent-class penalty for subject-level inferential comparison: include only true classes observed for that subject.
            observed=np.unique(y[mask])
            a=np.mean([f1_score(y[mask],p[mask],labels=observed,average='macro',zero_division=0) for p in sets['Dropout-LR']])
            b=np.mean([f1_score(y[mask],p[mask],labels=observed,average='macro',zero_division=0) for p in sets['LR']])
            deltas.append(a-b)
            subject_rows.append({'subject':int(subject),'kind':kind,'severity':level,'observed_classes':len(observed),
                                 'observed_class_delta':a-b,**scores})
        tests.append({'kind':kind,'severity':level,'equal_subject_delta':float(np.mean(deltas)),
                      'positive_subjects':int(np.sum(np.array(deltas)>0)), 'n_subjects':len(deltas),
                      'signflip_p':exact_signflip(np.asarray(deltas))})
    # Family includes all nine clean/dropout/noise conditions; structured analyses are separately exploratory.
    family=[i for i,t in enumerate(tests) if t['kind'] in ['clean','dropout','noise']]
    ordered=sorted(family,key=lambda i:tests[i]['signflip_p']); running=0.
    for rank,index in enumerate(ordered):
        running=max(running,min(1.,tests[index]['signflip_p']*(len(ordered)-rank)))
        tests[index]['holm_p']=running
    pd.DataFrame(intervals).to_csv(folder/'bootstrap_intervals.csv',index=False)
    pd.DataFrame(comparisons).to_csv(folder/'paired_comparisons.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(folder/'subject_metrics.csv',index=False)
    pd.DataFrame(tests).to_csv(folder/'subject_signflip_tests.csv',index=False)
    write_json(folder/'structured_groups.json',groups)

def run():
    started=time.perf_counter(); OUT.mkdir(exist_ok=True)
    write_json(OUT/'protocol_snapshot.json',CONFIG)
    for dataset in ['wisdm','uci']:
        folder=OUT/dataset; folder.mkdir(exist_ok=True); (folder/'models').mkdir(exist_ok=True)
        if dataset=='wisdm':
            packed=np.load(ROOT/'data/wisdm_prepared.npz')
            train=packed['train']; X=packed['X'][train]; y=packed['y'][train]; g=packed['subjects'][train]
            grids={'LR':[{'family':'lr','C':c} for c in CONFIG['C_grid']],
                   'Dropout-LR':[{'family':'dropout_lr','C':c,'p':p} for c in CONFIG['C_grid'] for p in CONFIG['dropout_p_grid']],
                   'RBF-SVM':[{'family':'svm','C':c} for c in [1.,10.]],
                   'Random forest':[{'family':'rf','max_features':m,'trees':200} for m in ['sqrt',.2]]}
            selected=selection(X,y,g,grids,folder)
            fitted={}; timings=[]
            selected['LR matched C']={'params':{'family':'lr','C':selected['Dropout-LR']['params']['C']}}
            for model,entry in selected.items():
                t=time.perf_counter(); fitted[model]=fit_model(X,y,entry['params'],2026,2)
                timings.append({'model':model,'fit_seconds':time.perf_counter()-t})
                joblib.dump(fitted[model],folder/'models'/f'{model.lower().replace(" ","_")}.joblib',compress=3)
            pd.DataFrame(timings).to_csv(folder/'resources.csv',index=False)
            test=packed['test']
            evaluate(dataset,packed['X'][test],packed['y'][test],packed['subjects'][test],fitted,folder,
                     {axis:axis_group(axis) for axis in ['x','y','z']},packed['windows'][test])
        else:
            X,y,g=load_split('train')
            grids={'LR':[{'family':'lr','C':c} for c in [.001,.01]],
                   'Dropout-LR':[{'family':'dropout_lr','C':c,'p':p} for c in [.001,.01] for p in CONFIG['dropout_p_grid']]}
            prior=pd.read_csv(ROOT/'results/cv_results.csv')
            prior=prior[prior.model.isin(grids)]
            selected=selection(X,y,g,grids,folder,prior=prior)
            fitted={name:joblib.load(ROOT/'results/models'/file) for name,file in [('LR','lr.joblib'),
                   ('Dropout-LR','dropout-lr.joblib'),('RBF-SVM','rbf-svm.joblib'),('Random forest','random_forest.joblib')]}
            for name,entry in selected.items():
                fitted['Expanded '+name]=fit_model(X,y,entry['params'],2026,2)
                joblib.dump(fitted['Expanded '+name],folder/'models'/('expanded_'+name.lower()+'.joblib'),compress=3)
            Xtest,ytest,gtest=load_split('test')
            features=[line.split(maxsplit=1)[1] for line in (ROOT/'data/UCI HAR Dataset/features.txt').read_text().splitlines()]
            groups={key:[i for i,name in enumerate(features) if key in name] for key in CONFIG['structured_uci_groups']}
            evaluate(dataset,Xtest,ytest,gtest,fitted,folder,groups)
    write_json(OUT/'manifest.json',{'elapsed_seconds':time.perf_counter()-started,'python':platform.python_version(),
        'protocol_sha256':hashlib.sha256((ROOT/'revision_protocol.json').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
        'interpretation':'UCI follow-up exploratory; WISDM model selection excludes held-out subjects; no external preregistration'})
    print('Review experiments complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): run()
