#!/usr/bin/env python3
"""Held-out score-query stress test, validation-only thresholds and repeated timing.

Defaults are the reported design. All raw scores, row indices, attack queries,
training transforms and timing repetitions are retained. No test data tune models.
"""
import argparse, json, platform, time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run_q1_upgrade_experiment as q

ROOT = Path(__file__).resolve().parents[1]
SEEDS = q.SEEDS

def prepare(data, seed, out):
    tr = pd.read_csv(data/'raw/UNSW_NB15_training-set.csv')
    te = pd.read_csv(data/'raw/UNSW_NB15_testing-set.csv')
    fit, val = train_test_split(np.arange(len(tr)), test_size=.2, stratify=tr.label, random_state=seed)
    rng = np.random.default_rng(seed)
    fit = rng.choice(fit, 30000, replace=False)
    val = rng.choice(val, 6000, replace=False)
    # Sampling is per true class, independent of target-model correctness.
    test = np.concatenate([rng.choice(np.flatnonzero(te.label.to_numpy()==c),1000,replace=False) for c in [0,1]])
    drop=['id','attack_cat','label','proto','service','state']
    raw = tr.drop(columns=drop)
    X, V = q.BASE.sanitize_train_test(raw.iloc[fit], raw.iloc[val])
    _, T = q.BASE.sanitize_train_test(raw.iloc[fit], te.drop(columns=drop).iloc[test])
    cols=sorted(X.columns); X=X[cols].reset_index(drop=True); V=V[cols].reset_index(drop=True); T=T[cols].reset_index(drop=True)
    y=tr.label.to_numpy()[fit]; v=tr.label.to_numpy()[val]; t=te.label.to_numpy()[test]
    c=q.BASE.infer_constraints(X,cols)
    np.savez_compressed(out/f'train_manifest_{seed}.npz', fit_rows=fit,validation_rows=val,test_rows=test,columns=np.array(cols,dtype=str),medians=raw.iloc[fit][cols].median().to_numpy(),**c)
    return X,y,V,v,T,t,c,test

def project(z, x, c, eps):
    """Empirical manifest, original anchor and an explicit quantile-span budget.
    Unlike transferred attacks, allow either sign on non-monotonic coordinates.
    """
    span=np.maximum(c['perturb_high']-c['perturb_low'],1e-6)
    lo=np.maximum(np.minimum(c['low'],x),x-eps*span)
    hi=np.minimum(np.maximum(c['high'],x),x+eps*span)
    z=np.clip(z,lo,hi)
    ints=np.setdiff1d(c['integer_idx'],c['immutable_idx'])
    z[:,ints]=np.round(z[:,ints])
    mono=c['monotone_non_decrease_idx'];z[:,mono]=np.maximum(z[:,mono],x[:,mono])
    z[:,c['immutable_idx']]=x[:,c['immutable_idx']]
    invalid=(z<lo-1e-7)|(z>hi+1e-7)
    z[invalid]=x[invalid]
    return z.astype(np.float32)

def query_attack(pipe,T,t,c,direction,seed,budget):
    rng=np.random.default_rng(seed)
    x=T.to_numpy(np.float32); malicious=np.flatnonzero(t==1); orig=x[malicious]
    def score(z):return pipe.predict_proba(pd.DataFrame(z,columns=T.columns))[:,1]
    best=orig.copy();bs=score(best);clean=bs.copy()
    surrogate,_,_,_=q.BASE.make_adversarial(T,t,direction,c,.15,True)
    # Query 1 is the same baseline proposal, projected onto the strict budget.
    candidate=project(surrogate.to_numpy()[malicious],orig,c,.15)
    mutable=np.setdiff1d(np.arange(x.shape[1]),c['immutable_idx'])
    span=np.maximum(c['perturb_high']-c['perturb_low'],1e-6)
    trace=[bs.copy()]; checkpoints={}
    for k in range(1,budget+1):
        if k>1:
            candidate=best.copy()
            # Periodic full random restarts plus adaptive random-block moves.
            block=len(mutable) if k%16==0 else int(rng.integers(1,min(6,len(mutable)+1)))
            for j in rng.choice(mutable,size=block,replace=False):
                candidate[:,j]=orig[:,j]+rng.choice([-.15,-.075,-.03,.03,.075,.15],len(orig))*span[j]
            candidate=project(candidate,orig,c,.15)
        cs=score(candidate);better=cs<bs;best[better]=candidate[better];bs[better]=cs[better]
        trace.append(bs.copy())
        if k in [32,budget]:
            full=x.copy();full[malicious]=best;checkpoints[k]=full.copy()
    assert np.all(bs<=clean+1e-7)
    assert q.BASE.constraint_violations(best,orig,c)==0
    assert np.max(np.abs(best-orig)/span)<=.15001
    return checkpoints,np.array(trace)

def metrics(y,s,threshold):
    pred=(s>=threshold).astype(int)
    return dict(f1=float(f1_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),fpr=float(pred[y==0].mean()),recall=float(pred[y==1].mean()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=ROOT/'data');parser.add_argument('--out-dir',type=Path,default=ROOT/'results/r2_checks');parser.add_argument('--queries',type=int,default=128);parser.add_argument('--skip-latency',action='store_true')
    a=parser.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[];timings=[];calibrations=[]
    for seed in SEEDS:
        X,y,V,v,T,t,c,testidx=prepare(a.data_dir,seed,a.out_dir)
        direction=q.BASE.fit_surrogate_direction(X,y)
        for name in ['RandomForest','ExtraTrees','HistGB']:
            for defense in ['standard','adv_aug']:
                model=clone(q.models(seed)[name])
                if hasattr(model,'n_jobs'):model.set_params(n_jobs=4)
                train,labels=X,y
                if defense=='adv_aug':
                    aug,*_=q.BASE.make_adversarial(X,y,direction,c,.06,True)
                    ids=np.flatnonzero(y==1)[:12000];train=pd.concat([X,aug.iloc[ids]],ignore_index=True);labels=np.r_[y,y[ids]]
                pipe=Pipeline([('scaler',StandardScaler()),('model',model)]).fit(train,labels)
                vs=pipe.predict_proba(V)[:,1]
                # Lowest fixed grid threshold satisfying a prespecified 5% validation FPR.
                grid=np.arange(.05,.951,.01)
                valid=[z for z in grid if np.mean(vs[v==0]>=z)<=.05]
                threshold=float(valid[0]) if valid else 1.
                calibrations.append(dict(seed=seed,model=name,defense=defense,threshold=threshold,validation_fpr=float(np.mean(vs[v==0]>=threshold)),fallback=not bool(valid)))
                clean=pipe.predict_proba(T)[:,1]
                surrogate,*_=q.BASE.make_adversarial(T,t,direction,c,.15,True)
                ss=pipe.predict_proba(surrogate)[:,1]
                checkpoints,trace=query_attack(pipe,T,t,c,direction,seed+71,a.queries)
                evidence=dict(y=t,test_rows=testidx,validation_y=v,validation_scores=vs,clean=clean,surrogate=ss,query_trace=trace,threshold=threshold)
                for attack,score in [('clean',clean),('surrogate',ss)]+[(f'query_{k}',pipe.predict_proba(pd.DataFrame(z,columns=T.columns))[:,1]) for k,z in checkpoints.items()]:
                    evidence[attack]=score
                    for threshold_name,z in [('fixed_050',.5),('validation_fpr05',threshold)]:
                        eligible=(t==1)&(clean>=z)
                        rows.append(dict(seed=seed,model=name,defense=defense,attack=attack,threshold_rule=threshold_name,threshold=z,n_test=len(t),n_eligible=int(eligible.sum()),evasion_count=int(np.sum(score[eligible]<z)),asr=float(np.mean(score[eligible]<z)) if eligible.any() else np.nan,**metrics(t,score,z)))
                evidence['adversarial_features']=checkpoints[a.queries];evidence['original_features']=T.to_numpy();evidence['columns']=np.array(T.columns,dtype=str)
                np.savez_compressed(a.out_dir/f'predictions_{seed}_{name}_{defense}.npz',**evidence)
                # Timing is performed after training/attacks, with no concurrent workers in this script.
                joblib.dump(pipe,a.out_dir/f'model_{seed}_{name}_{defense}.joblib')
                scaled=pipe[0].transform(T)
                for mode in ([] if a.skip_latency else ['estimator_only','scale_and_predict']):
                    call=(lambda:pipe[-1].predict_proba(scaled)) if mode=='estimator_only' else (lambda:pipe.predict_proba(T))
                    for _ in range(5):call()
                    for repetition in range(30):
                        start=time.perf_counter_ns();call();elapsed=time.perf_counter_ns()-start
                        timings.append(dict(seed=seed,model=name,defense=defense,mode=mode,repetition=repetition,batch_size=len(T),ms_per_flow=elapsed/1e6/len(T)))
                pd.DataFrame(rows).to_csv(a.out_dir/'metrics.csv',index=False)
                pd.DataFrame(timings).to_csv(a.out_dir/'latency_repetitions.csv',index=False)
                pd.DataFrame(calibrations).to_csv(a.out_dir/'thresholds.csv',index=False)
                print(seed,name,defense,'done',flush=True)
    (a.out_dir/'metadata.json').write_text(json.dumps(dict(seeds=SEEDS,queries=a.queries,epsilon=.15,warmups=5,repetitions=30,batch_size=2000,threads=4,python=platform.python_version(),platform=platform.platform(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
