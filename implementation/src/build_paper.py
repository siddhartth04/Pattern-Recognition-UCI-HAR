"""Build an editable manuscript and conference-style PDF from measured results."""
import json
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image, FrameBreak, NextPageTemplate, KeepTogether)
from .data import ROOT

MAIN = ['LR','RBF-SVM','Random forest','Dropout-LR']
SHORT = {'LR':'LR','RBF-SVM':'RBF-SVM','Random forest':'RF','Dropout-LR':'Dropout-LR'}
TITLE = 'Feature Dropout for Robust Smartphone Activity Recognition: A Subject-Disjoint Empirical Study'

def main():
    out = ROOT/'paper'; out.mkdir(exist_ok=True)
    metric = pd.read_csv(ROOT/'results/test_metrics.csv')
    intervals = pd.read_csv(ROOT/'results/bootstrap_intervals.csv')
    comparisons = pd.read_csv(ROOT/'results/paired_comparisons.csv')
    resources = pd.read_csv(ROOT/'results/resources.csv').set_index('model')
    selection = json.loads((ROOT/'results/selection.json').read_text())
    manifest = json.loads((ROOT/'results/run_manifest.json').read_text())
    dataset = json.loads((ROOT/'results/dataset_summary.json').read_text())
    perclass = json.loads((ROOT/'results/per_class.json').read_text())
    cv = pd.read_csv(ROOT/'results/cv_results.csv')
    baseline = selection['primary_baseline']
    p = selection['Dropout-LR']['params']['p']
    c = selection['Dropout-LR']['params']['C']

    def value(model, kind='clean', severity=0, key='macro_f1'):
        return metric[(metric.model==model)&(metric.kind==kind)&(metric.severity==severity)][key].mean()
    def pct(x): return f'{100*x:.2f}'
    def diff(kind, level, reference=baseline):
        return comparisons[(comparisons.kind==kind)&(comparisons.severity==level)&(comparisons.reference==reference)].iloc[0]
    def interval(model,kind='clean',level=0):
        return intervals[(intervals.model==model)&(intervals.kind==kind)&(intervals.severity==level)].iloc[0]
    clean, half = diff('clean',0), diff('dropout',.5)
    noise = diff('noise',1)
    signword = 'higher' if half.delta >= 0 else 'lower'
    conclusion = ('The interval excludes zero in favor of augmentation for this condition.' if half.low > 0
                  else 'The interval excludes zero in favor of the baseline for this condition.' if half.high < 0
                  else 'The interval includes zero, so this experiment does not establish an advantage at this condition.')
    blocks=[]
    def para(text): blocks.append(('p',text))
    def heading(text): blocks.append(('h',text))
    def table(caption, headers, rows, widths=None): blocks.append(('t',(caption,headers,rows,widths)))
    def figure(filename, caption, ratio): blocks.append(('f',(filename,caption,ratio)))
    def equation(text): blocks.append(('e',text))

    heading('Abstract')
    para(f'Missing input features can change the behavior of activity recognizers even when clean-test accuracy is high. '
         f'This study evaluates a simple feature-dropout augmentation procedure for multinomial logistic regression on UCI HAR. '
         f'The official training and test partitions contain 21 and 9 distinct subjects, respectively. Three subject-disjoint '
         f'validation folds select all model configurations using an equal-weight clean-and-corrupted macro-F1 criterion. '
         f'Clean-trained logistic regression, an RBF support vector machine, and a random forest serve as baselines. '
         f'Dropout-LR achieves {pct(value("Dropout-LR",key="accuracy"))}% clean-test accuracy and '
         f'{pct(value("Dropout-LR"))}% macro-F1. With 50% independently missing feature entries, its mean macro-F1 is '
         f'{pct(value("Dropout-LR","dropout",.5))}%, {abs(100*half.delta):.2f} percentage points {signword} than the '
         f'validation-selected comparator ({baseline}); the paired subject-bootstrap 95% interval for the difference is '
         f'[{100*half.low:.2f}, {100*half.high:.2f}] points. We also report unseen additive-noise conditions, training-seed '
         f'sensitivity, class errors and computation costs. The contribution is a reproducible empirical evaluation of an '
         f'established augmentation principle, with explicit limits on generalization and physical-sensor interpretations.')
    para('Keywords: pattern recognition; human activity recognition; feature corruption; logistic regression; subject-disjoint validation; reproducibility.')
    heading('1. Introduction')
    para('A supervised activity recognizer maps a vector of measurements to a discrete behavior label. The central pattern '
         'recognition tasks are choosing a representation, learning a decision rule and estimating how that rule generalizes. '
         'Recognition under incomplete inputs adds another question: does a classifier rely on a small set of features that may '
         'be unavailable at prediction time? Clean accuracy alone cannot answer this question.')
    para('We investigate this issue using the published feature representation of UCI HAR [1,2]. This makes the study practical '
         'on a laptop and separates classifier behavior from the engineering of a raw-signal feature extractor. The pipeline '
         'begins with the released numerical vectors and ends with reproducible predictions, stress-test curves and uncertainty '
         'estimates. It is not an end-to-end neural model or a live mobile application.')
    para('The main question is whether adding one randomly masked copy of each training example improves a linear classifier '
         'under feature loss without an unacceptable loss on clean data. A second question asks whether any benefit transfers '
         'to additive feature noise, which is absent from model selection. A third asks whether the result changes when the '
         'augmentation seed changes or when the unaugmented model uses exactly the same regularization parameter.')
    para('The study contributes a common subject-disjoint selection protocol for four methods; matched corruption realizations '
         'and a regularization-controlled augmentation comparison; and an auditable implementation with per-example outputs, '
         'subject-cluster intervals and measured computational costs. These are experimental contributions. No new learning '
         'algorithm, broad superiority claim or state-of-the-art record is asserted.')
    heading('2. Related work and scope')
    para('Anguita et al. introduced the smartphone activity dataset and a multiclass SVM benchmark [1]. The repository provides '
         'the data and citation metadata [2]. Their work establishes the task; published headline scores are not treated as '
         'direct experimental baselines here because implementation and selection protocols can differ.')
    para('Subject identity is an important evaluation variable in activity recognition. Dehghani et al. discuss how ordinary '
         'sample-based cross-validation can overestimate performance when observations from the same subject are correlated [3]. '
         'We use subject-disjoint validation and retain the official held-out subjects. Uncertainty calculations also resample '
         'subjects rather than treating overlapping windows as independent experimental units.')
    para('Feature corruption during training is established prior work. Van der Maaten et al. optimize expected losses under '
         'corruption distributions in marginalized corrupted features [4]. Neural dropout is another influential application '
         'of random removal during training [5]. Our implementation uses a finite augmented dataset and a conventional '
         'multinomial logistic-regression solver. It does not implement marginalized optimization or neural-network dropout.')
    para('A more recent preprint by Saha et al. considers activity and subject recognition, missing data and imputation on UCI '
         'HAR [6]. This reinforces that missing-data robustness is not an unexplored problem. Our narrower study focuses on '
         'a matched comparison of classical classifiers, explicit subject separation and synthetic feature-entry corruption. '
         'The preprint is contextual related work; its numerical scores are not compared as though obtained under our protocol.')
    heading('3. Data and method')
    heading('3.1. Dataset and representation')
    para(f'The loaded dataset has {dataset["train_n"]+dataset["test_n"]:,} rows and {dataset["features"]} features. '
         f'The official training partition has {dataset["train_n"]:,} rows from 21 subjects; the test partition has '
         f'{dataset["test_n"]:,} rows from 9 other subjects. The six labels are listed in Table 1. Counts were verified '
         'from the downloaded files, and an explicit assertion checks that the subject sets do not intersect.')
    table('Table 1. Observed class counts in the official partitions.', ['Activity','Train','Test'],
          [[name,str(a),str(b)] for name,a,b in zip(['Walking','Walking upstairs','Walking downstairs','Sitting','Standing','Lying'],
               dataset['train_class_counts'],dataset['test_class_counts'])], [126,54,54])
    para('The original collection uses waist-mounted smartphone motion sensors at 50 Hz. Published features summarize '
         '2.56-second windows with 50% overlap [1,2]. We retain the original feature order and supplied preprocessing. '
         'An additional StandardScaler is learned only from the relevant training rows. The original data contain no '
         'missing values; all evaluated missingness is synthetic.')
    heading('3.2. Standardization and corruption')
    equation('z_ij = (x_ij - mu_j) / s_j')
    para('Here mu and s are the training feature mean and standard deviation; a constant feature uses the scaler\'s unit '
         'scale convention. For dropout, a Bernoulli mask is sampled independently for every row-feature entry. A removed '
         'standardized value is set to zero, which is equivalent to training-mean imputation in the supplied feature space.')
    equation('m_ij ~ Bernoulli(1 - p);   z_tilde_ij = m_ij z_ij')
    para('Retained entries are not divided by the retention probability. This is deliberate: the experiment models missing '
         'features imputed with their training mean, not expectation-preserving neural dropout. The inference command accepts '
         'NaN entries and performs the same mean imputation. Gaussian stress tests instead add independent zero-mean noise '
         'with standard deviation sigma in standardized feature units; these perturbed values are not clipped.')
    equation('z_tilde_ij = z_ij + epsilon_ij;   epsilon_ij ~ N(0, sigma^2)')
    heading('3.3. Classifiers and augmentation')
    para('The linear classifier is L2-regularized multinomial logistic regression with an intercept, optimized using L-BFGS. '
         'Its clean counterpart uses the original standardized training matrix. Dropout-LR concatenates that matrix with one '
         'masked copy and repeats the labels. Each of the two copies receives sample weight 0.5, so the total sample weight '
         'remains the original training size n.')
    equation('J(W,b) = (1/n) sum_i [0.5 L(z_i,y_i) + 0.5 L(m_i*z_i,y_i)] + ||W||^2 / (2 C n)')
    para('L is multiclass cross-entropy; the intercept is unpenalized. C is scikit-learn\'s inverse regularization parameter. '
         'The masks are sampled once per fit and remain fixed during optimization. Weighting prevents a doubled dataset '
         'size from silently changing the effective regularization. A zero-dropout test verifies equivalence to clean '
         'training at the same C. The trained augmented model has the same inference architecture as ordinary logistic regression.')
    para('The other classifiers are an RBF-kernel SVM and a 200-tree random forest. The SVM uses gamma="scale" and does '
         'not fit probability calibration. The forest uses bootstrap sampling and unrestricted tree depth; its feature '
         'subsampling setting is selected on validation folds. All classifiers receive the same train-fitted standardized '
         'representation. Implementations use scikit-learn [7].')
    heading('4. Experimental protocol')
    heading('4.1. Selection without test tuning')
    para('The protocol is specified in config.json before the model run. Three GroupKFold partitions split the 21 training '
         'subjects, and a new scaler and classifier are fit in each fold. The validation score for a configuration is the '
         'mean of clean macro-F1 and corrupted macro-F1 at p=0.3. Corrupted validation scores average seeds 101, 102 and '
         '103; the final selection score averages the three folds equally. Ties use grid order. The protocol is recorded '
         'locally and is not an external preregistration.')
    equation('S = mean_folds { [F1_clean + mean_seeds(F1_drop, p=0.3)] / 2 }')
    para('The candidate grids are C in {0.1,1,10} for both logistic models, training p in {0.1,0.3,0.5} for Dropout-LR, '
         'C in {1,10} for the SVM, and max_features in {sqrt,0.2} for the forest. Thus 16 configurations require 48 fits. '
         'The same scoring rule is applied to every model, although the augmented model has a larger search grid. '
         'The selected configurations are written to disk before the runner loads the official test arrays.')
    table('Table 2. Validation-selected parameters and mixed score S (%).', ['Model','Selected setting','S'],
          [[SHORT[m],f'C={selection[m]["params"]["C"]:g}' + (f', p={p:g}' if m=='Dropout-LR' else '')
            if m!='Random forest' else f'max_features={selection[m]["params"]["max_features"]}',
            pct(selection[m]['cv_score'])] for m in MAIN], [65,123,46])
    para(f'The baseline with the largest validation score is {baseline}, so it is the primary comparator before test '
         f'evaluation. Dropout-LR selects C={c:g} and p={p:g}. Model and augmentation seeds are 2026 for final fitting '
         'and 2026 plus the fold index during validation. The test set is not used to revise the grid, corruption levels '
         'or primary comparator. Validation estimates used for selection are not presented as unbiased final performance.')
    heading('4.2. Test conditions and metrics')
    para('Each selected model is retrained on all official training rows. The clean test condition is evaluated once. '
         'Feature-dropout rates are 0.1, 0.3, 0.5 and 0.7; noise standard deviations are 0.1, 0.3, 0.5 and 1.0. '
         'Every non-clean condition uses seeds 901, 902 and 903, matched across models. Curves report the mean across '
         'these three realizations; they are not averages over independently collected datasets. Gaussian corruption '
         'is not used during training or hyperparameter selection.')
    para('Accuracy counts correct windows. Macro-F1 is the arithmetic mean of the six classwise F1 scores, with absent '
         'denominators assigned zero. The main ablation trains clean logistic regression at the selected augmented C. '
         'Two further augmented fits use seeds 2027 and 2028 with unchanged hyperparameters. In total, seven fitted '
         'bundles are evaluated in 25 conditions each, giving 175 saved metric records.')
    heading('4.3. Uncertainty and computational measurements')
    para('For each condition, 2,000 bootstrap samples draw nine test subjects with replacement. All windows for each '
         'drawn subject are included, with multiplicity. Macro-F1 is computed from the pooled confusion matrix for '
         'each corruption realization and then averaged across the fixed realizations. The 2.5th and 97.5th percentiles '
         'form the interval. Paired differences use the same resampled subjects for both classifiers (bootstrap seed 777).')
    para('These intervals express variability under resampling the observed subject clusters, conditional on the fitted '
         'models and corruption draws. They do not include retraining uncertainty or represent a confidence guarantee '
         'for all smartphone users. Only nine test subjects are available. Multiple stress conditions are descriptive '
         'and no multiplicity-adjusted familywise significance claim is made.')
    para(f'The experiment ran on Windows with Python {manifest["python"]}, NumPy {manifest["numpy"]} and '
         f'scikit-learn {manifest["sklearn"]}. Computation was limited to two threads for numerical libraries and the '
         f'forest; no GPU was used. The recorded full-run time was {manifest["elapsed_seconds"]/60:.2f} minutes. '
         'Fit times include scaling and augmentation. Prediction costs include scaling, use all 2,947 test rows as '
         'one batch, and report the median of five repetitions after a warm-up. They are not single-request latency '
         'or device-energy measurements.')
    heading('5. Results')
    heading('5.1. Clean performance')
    rows=[]
    for m in MAIN:
        ci=interval(m)
        rows.append([SHORT[m],pct(value(m,key='accuracy')),pct(value(m)),f'{pct(ci.low)}-{pct(ci.high)}'])
    table('Table 3. Clean-test performance (%); intervals resample subjects.', ['Model','Accuracy','Macro-F1','95% interval'],rows,[65,49,49,71])
    para(f'Dropout-LR obtains {pct(value("Dropout-LR",key="accuracy"))}% accuracy and {pct(value("Dropout-LR"))}% '
         f'macro-F1 on clean inputs. Relative to {baseline}, its macro-F1 difference is {100*clean.delta:+.2f} percentage '
         f'points, with a paired 95% interval [{100*clean.low:.2f}, {100*clean.high:.2f}]. The evaluation objective '
         'balances clean and corrupted performance, so it does not select models solely to maximize clean accuracy. '
         'The marginal score intervals should not be substituted for the paired difference interval.')
    heading('5.2. Feature-loss stress test')
    table('Table 4. Macro-F1 (%) averaged over three missingness masks.', ['Model','10%','30%','50%','70%'],
          [[SHORT[m]]+[pct(value(m,'dropout',q)) for q in [.1,.3,.5,.7]] for m in MAIN],[66,42,42,42,42])
    figure('dropout_robustness.png','Figure 1. Feature-loss robustness. Lines show mean macro-F1 and shaded bands the min-max range over three corruption seeds, not confidence intervals.',2.7/3.4)
    para(f'At 50% missing entries, Dropout-LR reaches {pct(value("Dropout-LR","dropout",.5))}% macro-F1 versus '
         f'{pct(value(baseline,"dropout",.5))}% for {baseline}. The paired difference is {100*half.delta:+.2f} points '
         f'[{100*half.low:.2f}, {100*half.high:.2f}]. {conclusion} '
         'Table 4 reports every specified missingness level, including severe corruption, to avoid selecting a single '
         'favorable point after inspecting the test results.')
    para('The masks remove entries independently within each window. A classifier can still exploit correlated retained '
         'features, which may explain resilience under substantial entry loss. This interpretation is a hypothesis '
         'consistent with the representation; the experiment does not directly measure feature redundancy or causal '
         'sensor contributions. Model rank at one corruption level should not be assumed to persist at another.')
    heading('5.3. Transfer to additive noise')
    table('Table 5. Macro-F1 (%) under Gaussian feature noise.', ['Model','0.1 SD','0.3 SD','0.5 SD','1.0 SD'],
          [[SHORT[m]]+[pct(value(m,'noise',q)) for q in [.1,.3,.5,1.0]] for m in MAIN],[66,42,42,42,42])
    figure('noise_robustness.png','Figure 2. Additive-noise stress test. The noise distribution was not used for selection. Shading shows the range across three noise realizations.',2.7/3.4)
    para(f'At sigma=1.0, the augmented model has {pct(value("Dropout-LR","noise",1.0))}% macro-F1 and the primary '
         f'baseline has {pct(value(baseline,"noise",1.0))}%. Their paired difference is {100*noise.delta:+.2f} points '
         f'[{100*noise.low:.2f}, {100*noise.high:.2f}]. Dropout and additive noise have different effects: masking '
         'moves entries exactly to the training mean, whereas noise can move them in either direction and beyond '
         'the observed range. Consequently, success under one perturbation is insufficient evidence of general robustness.')
    heading('5.4. Ablations and class errors')
    matched=diff('dropout',.5,'LR matched C')
    same_c = selection['LR']['params']['C']==c
    para(('The selected clean and augmented classifiers use the same C. Consequently, the matched-C ablation is '
          'identical to the selected clean LR model and supplies no additional independent model comparison. ' if same_c else
          'The matched-C ablation isolates augmentation from the difference in selected regularization strength. ')+
         f'At 50% missingness, the augmented-minus-matched-C difference is {100*matched.delta:+.2f} points '
         f'[{100*matched.low:.2f}, {100*matched.high:.2f}]. Maintaining total sample weight makes this comparison '
         'interpretable without a hidden regularization change from duplicating examples.')
    seed_names=['Dropout-LR','Dropout-LR seed 2027','Dropout-LR seed 2028']
    table('Table 6. Augmentation-seed sensitivity; macro-F1 (%).', ['Train seed','Clean','50% missing','Noise 1 SD'],
          [[str(s),pct(value(m)),pct(value(m,'dropout',.5)),pct(value(m,'noise',1))]
           for s,m in zip([2026,2027,2028],seed_names)],[54,49,68,63])
    para('These additional seeds are sensitivity analyses, not a hyperparameter search. Seed 2026 remains the primary '
         'reported model. Three fits provide limited evidence about optimization and mask variability and do not '
         'replace repeated subject-split experiments. The subject bootstrap and training-seed table answer different '
         'uncertainty questions.')
    table('Table 7. Clean Dropout-LR class precision, recall and F1 (%).', ['Activity','P','R','F1'],
          [[name]+[pct(perclass['Dropout-LR'][str(i)][k]) for k in ['precision','recall','f1-score']]
           for i,name in enumerate(['Walking','Upstairs','Downstairs','Sitting','Standing','Lying'],start=1)],[96,46,46,46])
    figure('confusion_matrices.png','Figure 3. Dropout-LR confusion counts on clean inputs and one preselected 50% masking realization (seed 901). Color is normalized within each true class.',5.3/3.4)
    para('Class-level precision and recall identify which errors contribute to the aggregate score. The confusion '
         'matrices use the same underlying test windows; differences arise from the feature mask. The corrupted '
         'matrix is an illustration for the first fixed test seed, while the tables average all three seeds. A '
         'separate subject-results figure in the artifact exposes heterogeneity that aggregate window metrics can hide.')
    heading('5.5. Computational cost')
    table('Table 8. Local resource measurements (single final fit).', ['Model','Fit (s)','Batch us/row','Size (KiB)'],
          [[SHORT[m],f'{resources.loc[m,"fit_seconds"]:.2f}',f'{resources.loc[m,"predict_us_per_row"]:.2f}',
            f'{resources.loc[m,"model_kib"]:.1f}'] for m in MAIN],[65,48,66,55])
    para('The augmented model fits twice as many rows as clean LR but stores a linear decision function at inference. '
         'The serialized size includes the scaler and classifier and uses joblib compression level 3. All timings '
         'were measured on the same host, but background load and implementation-specific threading remain potential '
         'sources of variation. Peak memory, power use and mobile deployment were not measured.')
    heading('6. Limitations and reproducibility')
    para('External validity is limited by one benchmark, one official subject split, six activities and a small set '
         'of participants. The dataset\'s prepared representation and collection setup may not match other devices, '
         'body locations or free-living behavior. No inference should be made about unobserved activity classes, '
         'demographic groups, medical decisions or safety-critical deployment.')
    para('The perturbations operate after feature extraction and ignore time continuity and correlated sensor '
         'failure. A missing sensor may remove whole feature families over long intervals; independent feature '
         'entry masking does not simulate that mechanism. The method also assumes that missing values can be '
         'identified and imputed. Undetected corruption, biased missingness and adversarial perturbations were not tested.')
    para('Hyperparameter grids are deliberately small for laptop feasibility. The selected LR C is at a grid '
         'boundary, so a broader search could change the comparison; that search was not performed after seeing '
         'the test outcomes. The dropout method has more candidate settings than each baseline. Three validation '
         'folds and three corruption seeds are limited samples, and the nine-cluster percentile bootstrap may '
         'have unstable coverage. There is no formal power analysis or independent replication dataset.')
    para('The artifact records the dataset archive and member hashes, fold subject IDs, every validation score, '
         'selected parameters, fitted model bundles, all test predictions and metric tables. Configuration values '
         'and source hashes are included in the run manifest. Tests check train/test subject separation, corruption '
         'endpoints, training-only scaling, mean imputation, serialization, regularization equivalence, deterministic '
         'training and metric reconstruction from saved predictions. The implementation can therefore be audited '
         'without relying only on rounded manuscript numbers.')
    para('The public data are credited to their creators [1,2]; no new human data were collected. An original '
         'archive notice and the current repository licensing statement are both preserved in the project notes. '
         'Any claim about method novelty requires a broader literature assessment than this focused empirical '
         'study. The package is intended to make that further research easier, not to substitute for it.')
    heading('7. Conclusion')
    para(f'This study implements and evaluates feature-dropout training for smartphone activity recognition under '
         f'a subject-disjoint protocol. The selected model attains {pct(value("Dropout-LR"))}% clean macro-F1 and '
         f'{pct(value("Dropout-LR","dropout",.5))}% macro-F1 at 50% independently missing entries. Relative to '
         f'the validation-selected {baseline}, the latter difference is {100*half.delta:+.2f} percentage points '
         f'with interval [{100*half.low:.2f}, {100*half.high:.2f}]. {conclusion} The practical outcome is a '
         'small, reproducible comparison that makes robustness tradeoffs and uncertainty explicit. Additional '
         'datasets and structured raw-signal failures are needed before claiming broader deployment benefits.')
    heading('References')
    refs=[
      '[1] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," ESANN, pp. 437-442, 2013. https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf',
      '[2] J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, and X. Parra, "Human Activity Recognition Using Smartphones," UCI Machine Learning Repository, 2013. https://doi.org/10.24432/C54S4K',
      '[3] A. Dehghani, T. Glatard, and E. Shihab, "Subject Cross Validation in Human Activity Recognition," arXiv:1904.02666, 2019. https://doi.org/10.48550/arXiv.1904.02666',
      '[4] L. van der Maaten, M. Chen, S. Tyree, and K. Weinberger, "Learning with Marginalized Corrupted Features," ICML, PMLR 28(1), pp. 410-418, 2013. https://proceedings.mlr.press/v28/vandermaaten13.html',
      '[5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR, vol. 15, pp. 1929-1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html',
      '[6] D. Saha, P. Malik, and A. Saha, "Activity and Subject Detection for UCI HAR Dataset with & without missing Sensor Data," arXiv:2505.06730, 2025. https://doi.org/10.48550/arXiv.2505.06730',
      '[7] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825-2830, 2011. https://www.jmlr.org/papers/v12/pedregosa11a.html']
    for reference in refs: blocks.append(('r',reference))

    # Markdown is the editable paper source; values are generated from CSV/JSON.
    md=['# '+TITLE,'','Anonymous manuscript - reproducible empirical study','']
    for kind,content in blocks:
        if kind=='h': md += ['## '+content,'']
        elif kind in ['p','r']: md += [content,'']
        elif kind=='e': md += ['```text',content,'```','']
        elif kind=='f':
            name,caption,_=content; md += [f'![{caption}](../figures/{name})','',caption,'']
        elif kind=='t':
            caption,headers,rows,_=content
            md += [caption,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
            md += ['| '+' | '.join(row)+' |' for row in rows]; md += ['']
    (out/'research_paper.md').write_text('\n'.join(md),encoding='utf-8')

    width,height=A4; margin=40; gap=20; col=(width-2*margin-gap)/2
    body=ParagraphStyle('Body',fontName='Times-Roman',fontSize=9.3,leading=11.3,
                        alignment=TA_JUSTIFY,spaceAfter=5.5)
    head=ParagraphStyle('Head',parent=body,fontName='Times-Bold',fontSize=10.2,leading=12.5,
                        alignment=TA_LEFT,spaceBefore=8,spaceAfter=5,keepWithNext=True)
    caption_style=ParagraphStyle('Caption',parent=body,fontSize=8,leading=9.7,alignment=TA_LEFT,spaceAfter=7)
    refstyle=ParagraphStyle('Ref',parent=body,fontSize=8.2,leading=10,alignment=TA_LEFT,wordWrap='LTR')
    eqstyle=ParagraphStyle('Eq',parent=body,fontName='Courier',fontSize=7,leading=9,alignment=TA_LEFT,spaceBefore=4,spaceAfter=8)
    cellstyle=ParagraphStyle('Cell',fontName='Times-Roman',fontSize=7.8,leading=9.2)
    title_style=ParagraphStyle('Title',fontName='Times-Bold',fontSize=17,leading=19,alignment=TA_CENTER,spaceAfter=9)
    subtitle_style=ParagraphStyle('Sub',fontName='Times-Roman',fontSize=9,leading=11,alignment=TA_CENTER)
    def page(canvas,doc):
        canvas.saveState(); canvas.setFont('Times-Roman',8); canvas.setFillColor(colors.HexColor('#526071'))
        canvas.drawString(margin,24,'Feature dropout for smartphone activity recognition')
        canvas.drawRightString(width-margin,24,str(doc.page)); canvas.restoreState()
    firstframes=[Frame(margin,height-127,width-2*margin,89,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),
                 Frame(margin,43,col,height-181,id='firstleft',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),
                 Frame(margin+col+gap,43,col,height-181,id='firstright',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
    frames=[Frame(margin,43,col,height-83,id='left',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),
            Frame(margin+col+gap,43,col,height-83,id='right',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
    document=BaseDocTemplate(str(out/'research_paper.pdf'),pagesize=A4,title=TITLE,author='Anonymous',
                             leftMargin=margin,rightMargin=margin,topMargin=40,bottomMargin=43)
    document.addPageTemplates([PageTemplate(id='first',frames=firstframes,onPage=page),
                               PageTemplate(id='normal',frames=frames,onPage=page)])
    story=[Paragraph(escape(TITLE),title_style),Paragraph('Anonymous manuscript | Reproducible empirical study',subtitle_style),
           NextPageTemplate('normal'),FrameBreak()]
    eq_index=0
    for kind,content in blocks:
        if kind in ['h','p','r','e']:
            if kind=='e':
                eq_index+=1
                ratios=[.35/3.4,.35/3.4,.35/3.4,.65/3.4,.5/3.4]
                story.append(Image(str(ROOT/'figures'/f'equation_{eq_index}.png'),width=col,height=col*ratios[eq_index-1]))
                story.append(Spacer(1,6))
                continue
            text=escape(content)
            # Break the long objective at its natural additive boundary.
            if kind=='e' and len(content)>75:
                text=escape(content.replace(' + ||W||','\n+ ||W||')).replace('\n','<br/>')
            story.append(Paragraph(text,{'h':head,'p':body,'r':refstyle,'e':eqstyle}[kind]))
        elif kind=='t':
            caption,headers,rows,widths=content
            cells=[[Paragraph(escape(s),cellstyle) for s in row] for row in [headers]+rows]
            scale=col/sum(widths)
            tab=Table(cells,colWidths=[x*scale for x in widths],hAlign='LEFT')
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9edf3')),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#6b7c8e')),
                ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#9da7b4')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),
                ('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            preceding=[]
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='Head':
                preceding=[story.pop()]
            story.append(KeepTogether(preceding+[Paragraph(escape(caption),caption_style),tab,Spacer(1,9)]))
        elif kind=='f':
            filename,caption,ratio=content
            story.append(KeepTogether([Image(str(ROOT/'figures'/filename),width=col,height=col*ratio),
                                       Paragraph(escape(caption),caption_style)]))
    document.build(story)
    # Short result digest used by the handover and independently readable without the manuscript.
    digest={'title':TITLE,'primary_baseline':baseline,'clean_accuracy':value('Dropout-LR',key='accuracy'),
            'clean_macro_f1':value('Dropout-LR'),'dropout_50_macro_f1':value('Dropout-LR','dropout',.5),
            'dropout_50_delta':float(half.delta),'dropout_50_ci':[float(half.low),float(half.high)],
            'elapsed_minutes':manifest['elapsed_seconds']/60}
    (out/'result_digest.json').write_text(json.dumps(digest,indent=2))
    print(json.dumps(digest,indent=2))

if __name__ == '__main__':
    main()
