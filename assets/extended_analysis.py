"""Additional classroom exercises and validation, without changing the legacy split.

Run after analysis.py; inputs are locally saved fitted models and predictions.
All upstream metadata remain local. Public outputs contain aggregate statistics.
"""
from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict, train_test_split)

ROOT = Path(__file__).resolve().parent
FEATURES = ["age_month", "lab_5237_min", "lab_5227_min", "lab_5225_range", "lab_5235_max", "lab_5257_min"]
TARGET = "HOSPITAL_EXPIRE_FLAG"
SEED = 42
N_BOOT = 1000
CORE = ["LR", "RF_plain", "RF_missing"]
LABELS = {"LR":"普通 LR", "RF_plain":"RF 无指示", "RF_missing":"RF + 缺失指示"}
COLORS = {"LR":"#147d64", "RF_plain":"#b97835", "RF_missing":"#167a98"}
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei"], "axes.unicode_minus":False,
                     "pdf.fonttype":42, "ps.fonttype":42, "font.size":10, "savefig.dpi":200,
                     "axes.spines.top":False, "axes.spines.right":False})


def save(frame, stem):
    frame.to_csv(ROOT / "tables" / f"{stem}.csv", index=False, encoding="utf-8-sig")
    return frame


def figure(stem):
    for ext in ["png", "pdf"]:
        plt.savefig(ROOT / "figures" / f"{stem}.{ext}", bbox_inches="tight")
    plt.close("all")


def score(y, p, cutoff):
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= cutoff, labels=[0,1]).ravel()
    return {"AUROC":roc_auc_score(y,p), "AP":average_precision_score(y,p),
            "Brier":brier_score_loss(y,p), "sensitivity":tp/(tp+fn),
            "specificity":tn/(tn+fp), "accuracy":(tp+tn)/len(y),
            "FP":int(fp), "FN":int(fn), "threshold":float(cutoff)}


def cutoff_oof(y, p):
    fpr,tpr,t = roc_curve(y,p)
    eligible = np.flatnonzero((tpr >= .8) & np.isfinite(t))
    return float(t[eligible[np.argmin(fpr[eligible])]])


def cluster_indices(groups, y, seed=42):
    rng = np.random.RandomState(seed)
    ids, inverse = np.unique(groups, return_inverse=True)
    locations = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    result = []
    while len(result) < N_BOOT:
        ix = np.concatenate([locations[i] for i in rng.randint(0,len(ids),len(ids))])
        if np.unique(y[ix]).size == 2:
            result.append(ix)
    return result


def intervals(y, probabilities, indices, protocol):
    rows, differences = [], []
    measures = {"AUROC":roc_auc_score, "AP":average_precision_score, "Brier":brier_score_loss}
    samples = {}
    for name,p in probabilities.items():
        samples[name] = {}
        for metric,fn in measures.items():
            values = np.array([fn(y[i],p[i]) for i in indices])
            samples[name][metric] = values
            lo,hi = np.percentile(values,[2.5,97.5])
            rows.append({"protocol":protocol,"model":name,"metric":metric,"estimate":fn(y,p),"CI_low":lo,"CI_high":hi,"bootstrap_repeats":len(indices),"unit":"SUBJECT_ID"})
    if "RF_missing" in probabilities:
        for other in [n for n in probabilities if n!="RF_missing"]:
            for metric,fn in measures.items():
                v=samples['RF_missing'][metric]-samples[other][metric]
                lo,hi=np.percentile(v,[2.5,97.5])
                differences.append({"protocol":protocol,"reference":"RF_missing","comparison":other,"metric":metric,
                                    "estimate":fn(y,probabilities['RF_missing'])-fn(y,probabilities[other]),"CI_low":lo,"CI_high":hi})
    return rows,differences


def parameter_grid(name):
    # Four candidate configurations for every independently tuned algorithm.
    if name.startswith('LR'):
        return {"classifier__C":[.01,.1,1.,10.]}
    return {"classifier__max_depth":[5,None],"classifier__min_samples_leaf":[5,20]}


def fit_search(model, name, x, y, folds):
    return GridSearchCV(clone(model),parameter_grid(name),scoring="average_precision",cv=folds,
                        n_jobs=1,error_score="raise",refit=True).fit(x,y)


def run():
    started=time.time()
    raw=pd.read_excel(ROOT/'data/data.xlsx')
    data=raw.copy(); data.loc[data.age_month.lt(0),'age_month']=np.nan
    x=data[FEATURES]; y=data[TARGET].astype(int)
    membership=pd.read_csv(ROOT/'results/split_membership_private.csv')
    train,test=train_test_split(data.index.to_numpy(),test_size=.2,random_state=SEED,stratify=y)
    assert set(train)==set(membership.loc[membership.split.eq('train'),'source_row_index'])
    assert set(test)==set(membership.loc[membership.split.eq('test'),'source_row_index'])
    xt,yt=x.loc[train],y.loc[train]; xv,yv=x.loc[test],y.loc[test]
    primary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    models={name:joblib.load(ROOT/'models'/f'{name}.joblib') for name in ['LR','LR_balanced','LR_missing','RF_plain','RF_missing']}
    frozen=primary['thresholds']
    saved=pd.read_csv(ROOT/'results/test_predictions_private.csv')
    assert np.array_equal(saved.source_row_index.to_numpy(),test)
    probabilities={n:saved[f'{n}_probability'].to_numpy() for n in models}
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(xt,yt))

    # Task 22: prediction-score distribution, with the actual frozen threshold.
    histogram=[]
    edges=np.linspace(0,1,41)
    fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
    for ax,name in zip(axes,['LR','RF_missing']):
        p=probabilities[name]
        for outcome,color in [(0,'#6d8eab'),(1,'#c56543')]:
            values=p[yv.to_numpy()==outcome]
            counts,_=np.histogram(values,bins=edges)
            histogram.extend({'model':name,'outcome':outcome,'left':float(a),'right':float(b),'count':int(c),'within_outcome_fraction':float(c/len(values))} for a,b,c in zip(edges[:-1],edges[1:],counts))
            ax.hist(values,bins=edges,weights=np.ones(len(values))/len(values),histtype='step',linewidth=1.5,color=color,label=['存活','死亡'][outcome])
        ax.axvline(frozen[name],color='#213845',ls='--',label=f"OOF阈值 {frozen[name]:.4f}")
        ax.set(xlabel='预测分数',ylabel='组内记录比例',title=LABELS[name],xlim=(0,.45))
        ax.legend(fontsize=9)
    figure('14_score_distribution')
    save(pd.DataFrame(histogram),'prediction_score_histogram')

    # Median vs mean, changing only the imputation rule for each pair.
    imputation_rows=[]
    for name in ['LR','RF_missing']:
        imputation_rows.append({'model':name,'imputation':'median',**score(yv,probabilities[name],frozen[name])})
        mean=clone(models[name]).set_params(imputer__strategy='mean')
        oof=cross_val_predict(mean,xt,yt,cv=folds,method='predict_proba')[:,1]
        cut=cutoff_oof(yt,oof)
        mean.fit(xt,yt); p=mean.predict_proba(xv)[:,1]
        imputation_rows.append({'model':name,'imputation':'mean',**score(yv,p,cut)})
    save(pd.DataFrame(imputation_rows),'imputation_sensitivity')
    print('Extended: mean/median completed',flush=True)

    # Transparent standardised ordinary-LR formula and preprocessing constants.
    lr=models['LR']; clf=lr.named_steps['classifier']; scaler=lr.named_steps['scaler']
    coefficients=pd.DataFrame({'feature':FEATURES,'coefficient_standardized':clf.coef_[0],
                               'imputation_median':lr.named_steps['imputer'].statistics_,
                               'standardization_mean':scaler.mean_,'standardization_scale':scaler.scale_,
                               'intercept':float(clf.intercept_[0])})
    save(coefficients,'LR_formula_coefficients')
    processed=lr[:-1].transform(xv)
    assert np.allclose(expit(clf.intercept_[0]+processed@clf.coef_[0]),probabilities['LR'])

    # Train/test comparability is descriptive, not a proof of no bias.
    comparisons=[]
    for column in FEATURES:
        a,b=xt[column].dropna(),xv[column].dropna()
        pooled=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        comparisons.append({'variable':column,'train_n_observed':len(a),'test_n_observed':len(b),
                            'train_mean':a.mean(),'train_sd':a.std(),'test_mean':b.mean(),'test_sd':b.std(),
                            'train_missing_rate':xt[column].isna().mean(),'test_missing_rate':xv[column].isna().mean(),
                            'SMD_test_minus_train':(b.mean()-a.mean())/pooled})
    rate=(yt.mean()+yv.mean())/2
    comparisons.append({'variable':TARGET,'train_n_observed':len(yt),'test_n_observed':len(yv),
                        'train_mean':yt.mean(),'test_mean':yv.mean(),'train_sd':yt.std(),'test_sd':yv.std(),
                        'train_missing_rate':0,'test_missing_rate':0,'SMD_test_minus_train':(yv.mean()-yt.mean())/np.sqrt(rate*(1-rate))})
    save(pd.DataFrame(comparisons),'train_test_comparability')

    # Four-candidate matched-budget sensitivity on the original training set.
    budget=[]
    for name,model in models.items():
        search=fit_search(model,name,xt,yt,folds)
        budget.append({'model':name,'candidate_configurations':len(search.cv_results_['params']),'folds':len(folds),
                       'fit_budget':len(search.cv_results_['params'])*len(folds),'CV_AP_best':search.best_score_,
                       'best_params':json.dumps(search.best_params_)})
    save(pd.DataFrame(budget),'matched_budget_CV')
    print('Extended: matched-budget CV completed',flush=True)

    # Reconcile all retained rows before using upstream identifiers.
    upstream=ROOT/'reference/PIC_mortality_upstream'
    full=pd.read_csv(upstream/'icu_first24hours.csv',float_precision='round_trip')
    assert len(raw)==len(full)
    assert np.allclose(raw.to_numpy(float),full[raw.columns].to_numpy(float),rtol=1e-12,atol=1e-12,equal_nan=True)
    byte_identical=hashlib.sha256((ROOT/'data/data.xlsx').read_bytes()).digest()==hashlib.sha256((upstream/'tutorial/data.xlsx').read_bytes()).digest()
    assert byte_identical
    groups=full.SUBJECT_ID.to_numpy()
    shared=set(groups[train])&set(groups[test])
    source_audit={'n':len(full),'deaths':int(y.sum()),'csv_columns':len(full.columns),'unique_subjects':len(np.unique(groups)),
                  'unique_admissions':full.HADM_ID.nunique(),'excel_byte_identical':byte_identical,
                  'all_rows_retained_columns_match':True,'primary_shared_subjects':len(shared),
                  'primary_test_records_shared_subjects':int(np.isin(groups[test],list(shared)).sum()),
                  'source_sha256':hashlib.sha256((upstream/'icu_first24hours.csv').read_bytes()).hexdigest()}
    save(pd.DataFrame([source_audit]),'upstream_metadata_audit')
    bootstrap_rows,paired_rows=intervals(yv.to_numpy(),{n:probabilities[n] for n in CORE},cluster_indices(groups[test],yv.to_numpy()),'primary_row_split')
    save(pd.DataFrame(bootstrap_rows),'primary_patient_cluster_CI')
    save(pd.DataFrame(paired_rows),'primary_patient_cluster_differences')

    # Official random row split and an independently tuned patient-group split.
    shuffled=np.random.RandomState(0).permutation(len(raw)); cut=int(.8*len(raw))
    group_outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    gt,gv=next(group_outer.split(x,y,groups))
    protocols=[('official_seed0',shuffled[:cut],shuffled[cut:]),('patient_group',gt,gv)]
    protocol_rows=[]; validation_rows=[]; validation_ci=[]; validation_diff=[]
    for protocol,tr,te in protocols:
        xtr,ytr=x.iloc[tr],y.iloc[tr]; xte,yte=x.iloc[te],y.iloc[te]
        common=set(groups[tr])&set(groups[te])
        protocol_rows.append({'protocol':protocol,'train_n':len(tr),'test_n':len(te),'train_deaths':int(ytr.sum()),'test_deaths':int(yte.sum()),
                              'train_subjects':len(np.unique(groups[tr])),'test_subjects':len(np.unique(groups[te])),
                              'shared_subjects':len(common),'test_records_shared_subjects':int(np.isin(groups[te],list(common)).sum())})
        if protocol=='patient_group':
            assert not common
            inner=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(xtr,ytr,groups[tr]))
        else:
            inner=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(xtr,ytr))
        local_models={}
        for name in ['LR','RF_missing']:
            search=fit_search(models[name],name,xtr,ytr,inner)
            local_models[name]=search.best_estimator_
        local_models['RF_plain']=clone(local_models['RF_missing']).set_params(imputer__add_indicator=False).fit(xtr,ytr)
        predictions={}
        for name in CORE:
            model=local_models[name]
            oof=cross_val_predict(clone(model),xtr,ytr,cv=inner,method='predict_proba')[:,1]
            threshold=cutoff_oof(ytr,oof)
            predictions[name]=model.predict_proba(xte)[:,1]
            validation_rows.append({'protocol':protocol,'model':name,**score(yte,predictions[name],threshold),
                                    'parameters':json.dumps({k:v for k,v in model.get_params().items() if k.startswith('classifier__')},default=str)})
        a,b=intervals(yte.to_numpy(),predictions,cluster_indices(groups[te],yte.to_numpy()),protocol)
        validation_ci.extend(a); validation_diff.extend(b)
        print('Extended:',protocol,'completed',flush=True)
    save(pd.DataFrame(protocol_rows),'validation_protocols')
    save(pd.DataFrame(validation_rows),'validation_sensitivity_metrics')
    save(pd.DataFrame(validation_ci),'validation_patient_cluster_CI')
    save(pd.DataFrame(validation_diff),'validation_patient_cluster_differences')

    # Repeated nested CV on the ORIGINAL TRAINING SET only. No main-test reuse.
    # Five outer folds x two repeats; inside each training fold, three-fold search.
    nested=[]
    outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=SEED)
    for step,(itr,iva) in enumerate(outer.split(xt,yt)):
        inner=list(StratifiedKFold(3,shuffle=True,random_state=SEED+step).split(xt.iloc[itr],yt.iloc[itr]))
        fold_models={}
        for name in ['LR','RF_missing']:
            fold_models[name]=fit_search(models[name],name,xt.iloc[itr],yt.iloc[itr],inner).best_estimator_
        fold_models['RF_plain']=clone(fold_models['RF_missing']).set_params(imputer__add_indicator=False).fit(xt.iloc[itr],yt.iloc[itr])
        for name in CORE:
            p=fold_models[name].predict_proba(xt.iloc[iva])[:,1]
            nested.append({'repeat':step//5+1,'fold':step%5+1,'model':name,'n_validation':len(iva),
                           'AUROC':roc_auc_score(yt.iloc[iva],p),'AP':average_precision_score(yt.iloc[iva],p),
                           'Brier':brier_score_loss(yt.iloc[iva],p)})
        print('Extended: nested outer fold',step+1,'/10',flush=True)
    repeated=save(pd.DataFrame(nested),'repeated_nested_CV_folds')
    rows=[]
    for name,g in repeated.groupby('model',sort=False):
        for metric in ['AUROC','AP','Brier']:
            rows.append({'model':name,'metric':metric,'mean':g[metric].mean(),'SD_across_outer_folds':g[metric].std(),
                         'min':g[metric].min(),'max':g[metric].max(),'n_outer_folds':len(g),'outer_repeats':2,'inner_folds':3})
    save(pd.DataFrame(rows),'repeated_nested_CV_summary')
    fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
    for ax,metric in zip(axes,['AUROC','AP']):
        for j,name in enumerate(CORE):
            values=repeated.loc[repeated.model.eq(name),metric].to_numpy()
            ax.scatter(j+np.linspace(-.1,.1,len(values)),values,s=25,color=COLORS[name],alpha=.75)
            ax.plot([j-.15,j+.15],[values.mean()]*2,color='#213845',lw=2)
        ax.set(xticks=range(len(CORE)),xticklabels=[LABELS[n] for n in CORE],ylabel=metric)
        ax.tick_params(axis='x',labelsize=9)
    figure('15_repeated_nested_CV')

    # CV versus held-out values are a descriptive comparison, not an overfit test.
    cv=pd.read_csv(ROOT/'tables/CV_model_comparison.csv').set_index('model')
    save(pd.DataFrame([{'model':n,'CV_AUROC':cv.loc[n,'CV_AUROC_mean'],'test_AUROC':roc_auc_score(yv,probabilities[n]),
                        'CV_AP':cv.loc[n,'CV_AP_mean'],'test_AP':average_precision_score(yv,probabilities[n])} for n in CORE]),'CV_test_descriptive_comparison')
    summary={'source_audit':source_audit,'nested_CV_outer_folds':10,'nested_CV_repeats':2,'nested_CV_inner_folds':3,
             'matched_search_configurations':4,'patient_mapping_assumption':'Original row order is preserved; all retained columns were checked.',
             'main_split_unchanged':True,'inference_scope':'Conditional fixed-model cluster intervals; no external validation; upstream six-feature selection provenance unknown.',
             'runtime_seconds':time.time()-started}
    (ROOT/'results/extended_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Extended analysis finished in',round(time.time()-started),'seconds',flush=True)


if __name__=='__main__':
    run()
