# ruff: noqa
# The simulation behind the "Test 1" arithmetic in 2026-11-09-redirect-plan-opus-check.md. Written and run by the
# Opus review agent (an AI model), 9 Nov 2026; filed unedited below this header by Fable. numpy only; no data.
import numpy as np
rng=np.random.default_rng(0)
def rk(X): return np.argsort(np.argsort(X,axis=0),axis=0).astype(float)
def stats(X):
    R=np.corrcoef(rk(X),rowvar=False)
    iu=np.triu_indices(5,1); pr=R[iu]
    ev=np.linalg.eigvalsh(np.corrcoef(X,rowvar=False))[::-1]
    return np.median(pr),(pr>0).sum(),ev[0]/5
def resid(X,c):
    C=np.column_stack([np.ones(len(c)),c]); B=np.linalg.lstsq(C,X,rcond=None)[0]; return X-C@B
def run(n=30,load=0.0,fam=0,famload=0.0,N=4000):
    hits=f40=nul=med3=0;meds=[]
    for _ in range(N):
        cap=rng.normal(size=n); g=rng.normal(size=n)
        X=0.5*cap[:,None]+load*g[:,None]+rng.normal(size=(n,5))
        if fam:
            fid=rng.integers(0,fam,size=n); fe=rng.normal(size=fam)*famload
            X=X+fe[fid][:,None]   # a family-level shared offset (e.g. lab safety policy), common to all measures
        Xr=resid(X,cap); m,pos,v=stats(Xr); meds.append(m)
        hits+= (m>=0.3 and pos>=8 and v>=0.4); f40+=(v>=0.4); nul+=(m<0.1); med3+=(m>=0.3)
    return dict(hit=hits/N,f40=f40/N,med03=med3/N,nul=nul/N,med_pct=np.round(np.percentile(meds,[5,50,95]),2))
for load in [0,0.4,0.6,0.8]:
    print('load',load,'true r~',round(load**2/(1+load**2),2),run(load=load))
print('no individual factor, 6 families with shared lab offset sd 0.7 (r~0.33 from family alone):',run(fam=6,famload=0.7))
print('n=12 effective:',run(n=12,load=0.0),run(n=12,load=0.6))
