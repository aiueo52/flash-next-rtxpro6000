# Replays FlashInfer TopPRenormProbKernel's ternary search (sampling.cuh ~1736-1822) on a top-k=40
# renormalised row, counting full-row passes. float32 throughout.
import numpy as np
f=np.float32
def rounds(probs,p=0.95):
    low=f(0); high=f(probs.max()); n=0
    while True:
        n+=1
        p0=f((high+f(2)*low)/f(3)); p1=f((f(2)*high+low)/f(3))
        a0=f(probs[probs>p0].sum(dtype=np.float32)); a1=f(probs[probs>p1].sum(dtype=np.float32))
        gt=probs[probs>low]; le=probs[probs<=high]
        mgl=gt.min() if gt.size else high; mlh=le.max() if le.size else low
        if a1>=p: low=p1
        elif a0>=p: low=p0; high=min(p1,mlh)
        else: high=min(p0,mlh)
        if not (mgl<mlh and np.nextafter(mgl,mlh)<mlh): break
    return n
rng=np.random.default_rng(0); V=248320; out={}
for name,scale in [("peaked(T=0.8, top1~0.9)",6.0),("medium",3.0),("flat",1.0)]:
    rs=[]
    for _ in range(200):
        lg=rng.gumbel(size=V).astype(np.float32)*f(1.0)
        lg[:40]+=np.sort(rng.exponential(scale,40))[::-1].astype(np.float32)*f(3)
        x=lg/f(0.8); x=x-x.max(); pr=np.exp(x).astype(np.float32); pr/=pr.sum(dtype=np.float32)
        idx=np.argsort(-pr)[:40]; q=np.zeros(V,np.float32); q[idx]=pr[idx]; q/=q.sum(dtype=np.float32)
        rs.append((rounds(q), float(np.sort(q)[::-1][0])))
    r=np.array([a for a,_ in rs]); t=np.array([b for _,b in rs])
    print(f"{name:26s} rounds median {np.median(r):.0f}  p10-p90 {np.percentile(r,10):.0f}-{np.percentile(r,90):.0f}  top1 prob median {np.median(t):.2f}")
print("zipf-shaped top-40 rows:")
for s in [0.5,1.0,1.5,2.0,3.0]:
    rs=[];t=[]
    for _ in range(200):
        v=(np.arange(1,41,dtype=np.float64)**-s)*np.exp(rng.normal(0,0.3,40)); v/=v.sum()
        q=np.zeros(V,np.float32); q[rng.choice(V,40,replace=False)]=v.astype(np.float32); q/=q.sum(dtype=np.float32)
        rs.append(rounds(q)); t.append(q.max())
    r=np.array(rs); print(f"  s={s:3.1f} top1~{np.median(t):.2f}: rounds median {np.median(r):.0f}, p10-p90 {np.percentile(r,10):.0f}-{np.percentile(r,90):.0f}, max {r.max()}")
