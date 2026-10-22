"""Regenerate final manuscript tables and vector figures from measured CSV files."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT

DEST=ROOT.parent/'latex'
NAMES=['LR','Dropout-LR','RBF-SVM','Random forest']

def run():
    (DEST/'generated').mkdir(exist_ok=True)
    (DEST/'figures').mkdir(exist_ok=True)
    frames={d:pd.read_csv(ROOT/'results_revision'/d/'test_metrics.csv') for d in ['uci','wisdm']}
    for dataset in frames:
        folder=ROOT/'results_revision'/dataset
        predictions=np.load(folder/'predictions.npz')
        labels=(['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING'] if dataset=='uci'
                else ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing'])
        class_rows=[]
        for model in NAMES:
            pred=predictions[f'{model}|clean|0.0|0']
            report=classification_report(predictions['y'],pred,labels=np.arange(1,7),target_names=labels,output_dict=True,zero_division=0)
            class_rows.extend({'model':model,'class':label,**report[label]} for label in labels)
            pd.DataFrame(confusion_matrix(predictions['y'],pred,labels=np.arange(1,7)),index=labels,columns=labels).to_csv(folder/f'clean_confusion_{model.lower().replace(" ","_")}.csv')
        pd.DataFrame(class_rows).to_csv(folder/'clean_class_metrics.csv',index=False)
    def value(d,m,k,s):
        f=frames[d]; return 100*f[(f.model==m)&(f.kind==k)&(f.severity==s)].macro_f1.mean()
    rows=[]
    for d,label in [('uci','UCI HAR'),('wisdm','WISDM')]:
        for m in NAMES:
            rows.append(f'{label} & {m} & '+ ' & '.join(f'{value(d,m,k,s):.2f}' for k,s in [('clean',0),('dropout',.3),('dropout',.5),('noise',1)])+r' \\')
    (DEST/'generated/main_results.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    rows=[]
    for d in frames:
        p=pd.read_csv(ROOT/'results_revision'/d/'paired_comparisons.csv')
        q=pd.read_csv(ROOT/'results_revision'/d/'subject_signflip_tests.csv')
        for k,s in [('clean',0),('dropout',.5),('noise',1)]:
            r=p[(p.method=='Dropout-LR')&(p.reference=='LR')&(p.kind==k)&(p.severity==s)].iloc[0]
            t=q[(q.kind==k)&(q.severity==s)].iloc[0]
            condition={'clean':'Clean','dropout':'50\% missing','noise':'Noise $\\sigma=1$'}[k]
            rows.append(f'{"UCI" if d=="uci" else "WISDM"} & {condition} & {100*r.delta:+.2f} & [{100*r.low:+.2f}, {100*r.high:+.2f}] & {t.holm_p:.3f}'+r' \\')
    (DEST/'generated/paired.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    rows=[]
    for d in frames:
        conditions=['group_Gyro','group_BodyAcc','group_GravityAcc'] if d=='uci' else ['group_x','group_y','group_z','raw_x_flatline','raw_y_flatline','raw_z_flatline']
        for k in conditions:
            s=0 if k.startswith('raw') else 1
            label=k.replace('group_','Mean: ').replace('raw_','Raw: ').replace('_flatline',' flatline')
            a,b=value(d,'LR',k,s),value(d,'Dropout-LR',k,s)
            rows.append(f'{"UCI" if d=="uci" else "WISDM"} & {label} & {a:.2f} & {b:.2f} & {b-a:+.2f}'+r' \\')
    (DEST/'generated/structured.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.6),sharey='row',layout='constrained')
    for row,d in enumerate(frames):
        for col,k in enumerate(['dropout','noise']):
            ax=axes[row,col]
            for m in NAMES:
                sub=frames[d][(frames[d].model==m)&(frames[d].kind.isin(['clean',k]))]
                avg=sub.groupby('severity').macro_f1.mean()*100
                ax.plot(avg.index,avg.values,marker='o',markersize=3,label=m)
            ax.set_title(('UCI HAR' if d=='uci' else 'WISDM')+': '+('missing entries' if k=='dropout' else 'Gaussian noise'))
            ax.set_xlabel('Missing probability' if k=='dropout' else 'Noise standard deviation')
            if col==0: ax.set_ylabel('Macro-F1 (%)')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,loc='lower left')
    fig.savefig(DEST/'figures/robustness.pdf'); plt.close(fig)
    print('Regenerated manuscript tables and robustness.pdf')

if __name__=='__main__': run()

