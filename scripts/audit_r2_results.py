#!/usr/bin/env python3
"""Validate published predictions, query invariants, timing units and train-only transforms."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import run_full_experiment as base
ROOT=Path(__file__).resolve().parents[1]
def main():
    results=ROOT/'results';q1=results/'q1_upgrade';clean=pd.read_csv(q1/'q1_clean_by_seed.csv');adv=pd.read_csv(q1/'q1_adversarial_by_seed.csv')
    assert len(clean)==240 and len(adv)==2160
    for _,r in clean.iterrows():
        name=f'{r.dataset}_{r.split}_{r.seed}_{r.model}_{r.defense}.npz'
        with np.load(q1/'predictions'/name) as d:
            assert abs(f1_score(d['y'],d['clean_pred'])-r.f1)<1e-12
            subset=adv[(adv.dataset==r.dataset)&(adv.split==r.split)&(adv.seed==r.seed)&(adv.model==r.model)&(adv.defense==r.defense)]
            for _,v in subset.iterrows():
                key=f'{v.attack}_{v.epsilon:.2f}_pred'
                assert abs(f1_score(d['y'],d[key])-v.f1)<1e-12
    r2=results/'r2_checks';metrics=pd.read_csv(r2/'metrics.csv');thresholds=pd.read_csv(r2/'thresholds.csv');timing=pd.read_csv(r2/'latency_repetitions.csv')
    assert len(metrics)==144 and len(thresholds)==18 and len(timing)==1080
    for file in r2.glob('predictions_*.npz'):
        with np.load(file) as d:
            seed,model,defense=file.stem[len('predictions_'):].split('_',2);seed=int(seed)
            trace=d['query_trace'];assert trace.shape==(129,1000);assert np.all(np.diff(trace,axis=0)<=1e-7)
            manifest=np.load(r2/f'train_manifest_{seed}.npz');c={k:manifest[k] for k in ['low','high','perturb_low','perturb_high','immutable_idx','integer_idx','monotone_non_decrease_idx']}
            orig=d['original_features'];new=d['adversarial_features'];mask=d['y']==1
            assert np.array_equal(orig[~mask],new[~mask]);assert base.constraint_violations(new,orig,c,mask)==0
            span=np.maximum(c['perturb_high']-c['perturb_low'],1e-6);assert np.max(np.abs(new-orig)/span)<=.15001
            assert not np.intersect1d(manifest['fit_rows'],manifest['validation_rows']).size
            subset=metrics[(metrics.seed==seed)&(metrics.model==model)&(metrics.defense==defense)]
            for _,r in subset.iterrows():
                pred=d[r.attack]>=r.threshold;assert abs(f1_score(d['y'],pred)-r.f1)<1e-12
                eligible=(d['y']==1)&(d['clean']>=r.threshold)
                assert int(eligible.sum())==r.n_eligible
                assert int(np.sum(~pred[eligible]))==r.evasion_count
            assert np.mean(d['validation_scores'][d['validation_y']==0]>=d['threshold'])<=.05+1e-12
    assert timing.ms_per_flow.gt(0).all();assert timing.batch_size.eq(2000).all()
    assert timing.groupby(['seed','model','defense','mode']).size().eq(30).all()
    for file in (results/'revision_analyses/predictions').glob('bootstrap_*.npy'):assert np.load(file).shape==(2000,)
    for _,r in pd.read_csv(results/'additional_day/metrics.csv').iterrows():
        with np.load(results/'additional_day'/f'predictions_{r.seed}_{r.model}.npz') as d:assert abs(f1_score(d['y'],d[r.setting+'_pred'])-r.f1)<1e-12
    # Missing-value imputation and retained columns cannot depend on held-out values.
    train=pd.DataFrame({'a':[1.,3.,np.nan],'constant':[7.,7.,7.]})
    test=pd.DataFrame({'a':[np.nan,10000.],'constant':[1.,2.]})
    x,z=base.sanitize_train_test(train,test);assert list(x.columns)==['a'];assert z.iloc[0,0]==2.
    altered=test.copy();altered.iloc[1,0]=-1e20
    x2,z2=base.sanitize_train_test(train,altered);assert x.equals(x2) and z2.iloc[0,0]==2.
    report=dict(status='passed',clean_conditions=240,adversarial_conditions=2160,target_query_conditions=144,thresholds=18,timing_observations=1080,query_budget=128,benign_invariance=True,query_score_monotonicity=True,constraint_and_budget_checks=True,train_only_imputation_invariance=True)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
