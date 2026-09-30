#!/usr/bin/env python3
"""Bounded second-day CICFlowMeter check; preserves sampled source-row identities."""
import json,gzip,shutil
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import run_q1_upgrade_experiment as q
ROOT=Path(__file__).resolve().parents[1]
URL='https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/'+quote('Processed Traffic Data for ML Algorithms/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
def main():
    out=ROOT/'results/additional_day';out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'data/external/cse_cic_ids2018_friday_2018_03_02_r2.csv'
    archived=ROOT/"inputs"/(cache.name+".gz")
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists() and archived.exists():
        with gzip.open(archived,"rb") as src,cache.open("wb") as dst:shutil.copyfileobj(src,dst)
    if not cache.exists():
        parts=[];counts={0:0,1:0};offset=0
        with requests.get(URL,stream=True,timeout=60) as r:
            r.raise_for_status()
            for chunk in pd.read_csv(r.raw,chunksize=50000,low_memory=False):
                chunk.columns=chunk.columns.str.strip();chunk['source_row']=np.arange(offset,offset+len(chunk));offset+=len(chunk)
                labels=chunk.Label.astype(str).str.strip(); chunk=chunk[labels.isin(['Benign','Bot'])]; labels=chunk.Label.astype(str).str.strip()
                for c,label in [(0,'Benign'),(1,'Bot')]:
                    candidates=chunk[labels.eq(label)];n=min(6000-counts[c],len(candidates))
                    if n>0:parts.append(candidates.sample(n=n,random_state=20260930+c));counts[c]+=n
                if all(v==6000 for v in counts.values()):break
        if min(counts.values())<6000:raise RuntimeError(counts)
        pd.concat(parts,ignore_index=True).to_csv(cache,index=False)
        (out/'sampling.json').write_text(json.dumps(dict(url=URL,rows_read=offset,per_class=6000,scheme='first chunks until class quotas filled; within-chunk seeded selection; not a representative reservoir'),indent=2))
    d=pd.read_csv(cache);y=d.Label.eq('Bot').astype(int).to_numpy()
    X=d.drop(columns=['Label','source_row']).apply(pd.to_numeric,errors='coerce')
    rows=[]
    for seed in q.SEEDS:
        tr,te=train_test_split(np.arange(len(d)),test_size=.3,stratify=y,random_state=seed)
        train,test=q.BASE.sanitize_train_test(X.iloc[tr],X.iloc[te]); cols=sorted(train.columns);train=train[cols];test=test[cols]
        c=q.BASE.infer_constraints(train,cols);direction=q.BASE.fit_surrogate_direction(train,y[tr]);adv,viol,*_=q.BASE.make_adversarial(test,y[te],direction,c,.15,True)
        for name in ['RandomForest','ExtraTrees','HistGB']:
            m=q.models(seed)[name]
            if hasattr(m,'n_jobs'):m.set_params(n_jobs=4)
            pipe=Pipeline([('scale',StandardScaler()),('model',m)]).fit(train,y[tr])
            evidence=dict(y=y[te],source_rows=d.source_row.to_numpy()[te],train_rows=d.source_row.to_numpy()[tr],columns=np.array(cols,dtype=str),medians=X.iloc[tr][cols].median().to_numpy())
            for setting,z in [('clean',test),('surrogate',adv)]:
                score=pipe.predict_proba(z)[:,1];pred=(score>=.5).astype(int);evidence[setting+'_pred']=pred;evidence[setting+'_score']=score
                rows.append(dict(seed=seed,model=name,setting=setting,n_train=len(tr),n_test=len(te),n_features=len(cols),violation_rate=viol if setting=='surrogate' else 0.,**q.metric_row(y[te],pred,score)))
            np.savez_compressed(out/f'predictions_{seed}_{name}.npz',**evidence)
            print(seed,name,'done',flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
