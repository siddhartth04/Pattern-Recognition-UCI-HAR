"""WISDM v1.1: strict record parsing and non-overlapping, run-safe windows."""
from pathlib import Path
import hashlib, json, tarfile, urllib.request
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .data import ROOT

URL='https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz'
SHA='e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7'
ACTIVITIES=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
STATS=['mean','std','min','max','median','q25','q75','mean_abs','rms','mean_abs_diff','dominant_bin','spectral_entropy']
FEATURES=[f'{axis}_{stat}' for axis in ['x','y','z','magnitude'] for stat in STATS]+['corr_xy','corr_xz','corr_yz']

def extract_features(windows):
    W=np.asarray(windows,dtype=np.float64)
    if W.ndim!=3 or W.shape[2]!=3 or W.shape[1]<2 or not np.isfinite(W).all():
        raise ValueError('Expected finite windows of shape (n, samples>=2, 3)')
    W=np.concatenate([W,np.linalg.norm(W,axis=2,keepdims=True)],axis=2)
    output=[]
    for axis in range(4):
        a=W[:,:,axis]; centered=a-a.mean(axis=1,keepdims=True)
        power=np.abs(np.fft.rfft(centered,axis=1))**2
        power[:,0]=0
        total=power.sum(axis=1,keepdims=True)
        probabilities=np.divide(power,total,out=np.zeros_like(power),where=total>0)
        entropy=-(probabilities*np.log(np.maximum(probabilities,1e-300))).sum(axis=1)/np.log(power.shape[1]-1)
        values=[a.mean(1),a.std(1),a.min(1),a.max(1),np.median(a,axis=1),np.quantile(a,.25,axis=1),
                np.quantile(a,.75,axis=1),np.abs(a).mean(1),np.sqrt((a*a).mean(1)),
                np.abs(np.diff(a,axis=1)).mean(1),np.argmax(power,axis=1)/a.shape[1],entropy]
        output.extend(values)
    for i,j in [(0,1),(0,2),(1,2)]:
        a=W[:,:,i]-W[:,:,i].mean(1,keepdims=True); b=W[:,:,j]-W[:,:,j].mean(1,keepdims=True)
        denom=np.sqrt((a*a).sum(1)*(b*b).sum(1))
        output.append(np.divide((a*b).sum(1),denom,out=np.zeros(len(a)),where=denom>0))
    return np.stack(output,axis=1)

def axis_group(axis):
    return [i for i,name in enumerate(FEATURES) if name.startswith(axis+'_') or name.startswith('magnitude_')
            or (name.startswith('corr_') and axis in name.removeprefix('corr_'))]

def prepare():
    data=ROOT/'data'; archive=data/'wisdm_original.tar.gz'
    if not archive.exists(): urllib.request.urlretrieve(URL,archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!=SHA: raise ValueError('WISDM archive checksum differs from verified release')
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.txt'):
                name=Path(member.name).name
                destination=data/('wisdm_raw.txt' if name.endswith('_raw.txt') else 'wisdm_notices/'+name)
                destination.parent.mkdir(exist_ok=True,parents=True)
                destination.write_bytes(tar.extractfile(member).read())
    config=json.loads((ROOT/'revision_protocol.json').read_text())
    n=config['window_samples']; counts=Counter(); windows=[]; labels=[]; subjects=[]; duration=[]; starts=[]
    recording=[]; previous_key=None
    def flush_recording():
        if not recording: return
        counts['recordings']+=1
        times=np.array([r[0] for r in recording],dtype=np.int64)
        counts['nonincreasing_original_timestamps']+=int(np.sum(np.diff(times)<=0))
        order=np.argsort(times,kind='stable')
        times=times[order]; unique=np.r_[True,np.diff(times)>0]
        counts['duplicate_timestamps_removed']+=int(np.sum(~unique))
        order=order[unique]; times=times[unique]
        boundaries=np.r_[0,np.flatnonzero(np.diff(times)>config['max_timestamp_gap_ns'])+1,len(times)]
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            counts['run_boundaries']+=1
            counts['discarded_tail_samples']+=(right-left)%n
            for first in range(left,right-n+1,n):
                indices=order[first:first+n]
                windows.append([recording[index][1] for index in indices])
                labels.append(previous_key[1]); subjects.append(previous_key[0])
                duration.append((times[first+n-1]-times[first])/1e9)
                starts.append(min(recording[index][2] for index in indices))
    with (data/'wisdm_raw.txt').open(encoding='utf-8',errors='strict') as source:
        for lineno,line in enumerate(source,1):
            for record in line.strip().split(';'):
                if not record.strip(): continue
                counts['nonempty_records']+=1
                try:
                    fields=record.strip().split(',')
                    if len(fields)!=6: raise ValueError('field count')
                    subject=int(fields[0]); activity=ACTIVITIES.index(fields[1])+1; timestamp=int(fields[2])
                    xyz=np.array([float(v) for v in fields[3:]])
                    if not 1<=subject<=36 or timestamp<=0 or not np.isfinite(xyz).all(): raise ValueError('invalid value')
                except (ValueError,OverflowError):
                    counts['invalid_records']+=1
                    continue
                counts['valid_records']+=1
                if previous_key is not None and (subject,activity)!=previous_key:
                    flush_recording(); recording=[]
                previous_key=(subject,activity)
                recording.append((timestamp,xyz,lineno))
    flush_recording()
    W=np.asarray(windows,dtype=np.float32); y=np.array(labels); groups=np.array(subjects)
    X=extract_features(W)
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=config['wisdm_test_subject_fraction'],
                                      random_state=config['wisdm_split_seed']).split(X,y,groups))
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(y[train])==set(y[test])==set(range(1,7))
    np.savez_compressed(data/'wisdm_prepared.npz',X=X,y=y,subjects=groups,train=train,test=test,
                        windows=W,duration_seconds=duration,source_start_lines=starts)
    summary={'archive_url':URL,'archive_sha256':digest,'counts':{k:int(v) for k,v in counts.items()},'n_windows':len(y),'n_features':51,
        'n_train':len(train),'n_test':len(test),'train_subjects':sorted(set(groups[train].tolist())),
        'test_subjects':sorted(set(groups[test].tolist())),'train_class_counts':np.bincount(y[train],minlength=7)[1:].tolist(),
        'test_class_counts':np.bincount(y[test],minlength=7)[1:].tolist(),
        'duration_seconds_min_median_max':[float(np.min(duration)),float(np.median(duration)),float(np.max(duration))],
        'features':FEATURES,'class_names':ACTIVITIES,'window_samples':n,'overlap':0,
        'timestamp_rule':'Within contiguous subject/activity recordings, stable-sort timestamps; retain first duplicate; split gaps >1e9 ns; reject timestamp<=0; no resampling',
        'statistics':'Population std; FFT of demeaned series, DC excluded, dominant cycles/sample; normalized spectral entropy; zero correlation for constant axes'}
    (data/'wisdm_manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='features'},indent=2),flush=True)

if __name__=='__main__': prepare()
