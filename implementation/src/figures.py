"""Generate publication figures entirely from saved measured results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .data import ROOT

MAIN = ['LR','RBF-SVM','Random forest','Dropout-LR']
COLORS = ['#748293','#dd8547','#4f8c82','#214c91']

def main():
    folder = ROOT / 'figures'; folder.mkdir(exist_ok=True)
    metrics = pd.read_csv(ROOT/'results/test_metrics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':220})
    for kind, xlabel in [('dropout','Missing feature entries (%)'),
                         ('noise','Noise standard deviation (training SD)')]:
        fig, ax = plt.subplots(figsize=(3.4,2.7), layout='constrained')
        for model, color in zip(MAIN,COLORS):
            data = metrics[(metrics.model == model) & metrics.kind.isin(['clean',kind])]
            grouped = data.groupby('severity').macro_f1.agg(['mean','min','max'])
            x = grouped.index.to_numpy() * (100 if kind == 'dropout' else 1)
            ax.plot(x,100*grouped['mean'],marker='o',label=model,color=color,
                    linewidth=2.3 if model == 'Dropout-LR' else 1.5,markersize=4)
            ax.fill_between(x,100*grouped['min'],100*grouped['max'],alpha=.12,color=color)
        ax.set(xlabel=xlabel,ylabel='Macro-F1 (%)',ylim=(25 if kind=='dropout' else 65,100))
        ax.grid(alpha=.18)
        ax.legend(loc='lower left',ncol=1,frameon=False,fontsize=7)
        fig.savefig(folder/f'{kind}_robustness.png')
        fig.savefig(folder/f'{kind}_robustness.pdf')
        plt.close(fig)
    predictions = np.load(ROOT/'results/predictions.npz')
    labels = ['Walk','Upstairs','Downstairs','Sit','Stand','Lie']
    fig, axes = plt.subplots(2,1,figsize=(3.4,5.3),layout='constrained')
    for ax,kind,severity,seed,title in zip(axes,['clean','dropout'],[0.0,.5],[0,901],
                                          ['Clean test set','50% missing entries (seed 901)']):
        matrix = confusion_matrix(predictions['y'],predictions[f'Dropout-LR|{kind}|{severity}|{seed}'])
        norm = matrix / matrix.sum(axis=1,keepdims=True)
        ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,
                        color='white' if norm[i,j]>.5 else '#273748')
        ax.set_xticks(range(6),labels,rotation=50,ha='right',fontsize=7)
        ax.set_yticks(range(6),labels,fontsize=7)
        ax.set(title=title,xlabel='Predicted activity',ylabel='True activity')
        ax.title.set_fontsize(9)
    fig.savefig(folder/'confusion_matrices.png')
    fig.savefig(folder/'confusion_matrices.pdf')
    plt.close(fig)
    # Equal-subject view makes participant heterogeneity visible without treating windows as independent.
    fig, ax = plt.subplots(figsize=(6.7,3.0),layout='constrained')
    from sklearn.metrics import f1_score
    subjects = np.unique(predictions['subjects'])
    for model,color in [('LR',COLORS[0]),('Dropout-LR',COLORS[3])]:
        vals=[]
        for subject in subjects:
            mask = predictions['subjects']==subject
            vals.append(np.mean([f1_score(predictions['y'][mask],predictions[f'{model}|dropout|0.5|{seed}'][mask],
                                labels=np.arange(1,7),average='macro',zero_division=0) for seed in [901,902,903]]))
        ax.plot(subjects,100*np.array(vals),'o-',color=color,label=model)
    ax.set(xticks=subjects,xlabel='Held-out subject ID',ylabel='Macro-F1 (%)',ylim=(0,100))
    ax.legend(frameon=False); ax.grid(alpha=.18)
    fig.savefig(folder/'subject_results.png'); plt.close(fig)
    equations = [
        (r'$z_{ij}=(x_{ij}-\mu_j)/s_j$', 12, .35),
        (r'$m_{ij}\sim\mathrm{Bernoulli}(1-p),\quad\widetilde{z}_{ij}=m_{ij}z_{ij}$', 11, .35),
        (r'$\widetilde{z}_{ij}=z_{ij}+\epsilon_{ij},\quad\epsilon_{ij}\sim\mathcal{N}(0,\sigma^2)$', 11, .35),
        ('$J(W,b)=\\frac{1}{n}\\sum_i[\\frac{1}{2}L(z_i,y_i)+\\frac{1}{2}L(m_i\\odot z_i,y_i)]$\n'
         '$+\\frac{\\|W\\|_F^2}{2Cn}$', 10, .65),
        (r'$S=\mathrm{mean}_{f}\!\left[\frac{F_{f,\mathrm{clean}}+\mathrm{mean}_s F_{f,p=0.3,s}}{2}\right]$', 11, .5)
    ]
    for i,(text,size,height) in enumerate(equations,1):
        fig=plt.figure(figsize=(3.4,height))
        fig.text(.01,.5,text,va='center',ha='left',fontsize=size)
        fig.savefig(folder/f'equation_{i}.png',dpi=300)
        plt.close(fig)
    print(f'Figures written to {folder}')

if __name__ == '__main__':
    main()
