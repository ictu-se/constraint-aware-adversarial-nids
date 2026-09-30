#!/usr/bin/env python3
"""Measure saved stress-test models with no other experiment running."""
import argparse,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,default=ROOT/'results/r2_checks');a=ap.parse_args()
    out=ROOT/'results/r2_checks';rows=[]
    for f in sorted(a.model_dir.glob('model_*.joblib')):
        seed,name,defense=f.stem[len('model_'):].split('_',2)
        p=joblib.load(f);d=np.load(out/f'predictions_{seed}_{name}_{defense}.npz');X=pd.DataFrame(d['original_features'],columns=d['columns']);z=p[0].transform(X)
        for mode in ['estimator_only','scale_and_predict']:
            call=(lambda:p[-1].predict_proba(z)) if mode=='estimator_only' else (lambda:p.predict_proba(X))
            for _ in range(5):call()
            for repetition in range(30):
                t=time.perf_counter_ns();call();elapsed=time.perf_counter_ns()-t
                rows.append(dict(seed=int(seed),model=name,defense=defense,mode=mode,repetition=repetition,batch_size=len(X),ms_per_flow=elapsed/1e6/len(X)))
        print(seed,name,defense,flush=True)
    assert len(rows)==1080
    pd.DataFrame(rows).to_csv(out/'latency_repetitions.csv',index=False)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
