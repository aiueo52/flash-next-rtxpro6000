import gzip,json,os,sys,collections,statistics
if len(sys.argv) < 2 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: attribute.py <trace.json.gz>  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p=sys.argv[1]
ev=json.load(gzip.open(p,"rt"))["traceEvents"]
kern=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
rt=[e for e in ev if e.get("ph")=="X" and e.get("cat")=="cuda_runtime"]
ann=sorted([e for e in ev if e.get("ph")=="X" and e.get("cat")=="user_annotation"],key=lambda e:e["ts"])
# correlation -> runtime event ts
corr_ts={}
corr_name={}
for e in rt:
    c=(e.get("args") or {}).get("correlation")
    if c is not None: corr_ts[c]=e["ts"]; corr_name[c]=e["name"]
# annotation lookup: for a cpu ts find innermost annotation among a chosen set
names_of_interest=("draft","step[TARGET_VERIFY bs=1]","draft_extend","copy_result_to_cpu","scheduler.process_batch_result","scheduler.get_next_batch_to_run")
A=[a for a in ann if a["name"] in names_of_interest]
import bisect
starts=[a["ts"] for a in A]
def find(ts):
    i=bisect.bisect_right(starts,ts)-1
    best=None
    for j in range(i,max(-1,i-6),-1):
        a=A[j]
        if a["ts"]<=ts<=a["ts"]+a["dur"]:
            if best is None or a["dur"]<best["dur"]: best=a
    return best["name"] if best else "other"
per=collections.defaultdict(lambda:[0,0.0])
launch=collections.Counter()
for k in kern:
    c=(k.get("args") or {}).get("correlation")
    ts=corr_ts.get(c)
    lab=find(ts) if ts is not None else "no-corr"
    per[lab][0]+=1; per[lab][1]+=k["dur"]
    if ts is not None: launch[(lab,corr_name.get(c,"?")[:24])]+=1
nsteps=sum(1 for a in ann if a["name"]=="draft")
print(f"steps={nsteps}")
print(f"{'phase':36s} {'kernels/step':>12s} {'gpu_ms/step':>12s}")
for lab,(c,d) in sorted(per.items(),key=lambda kv:-kv[1][1]):
    print(f"{lab:36s} {c/nsteps:12.1f} {d/1e3/nsteps:12.2f}")
print("launch types (phase, api): ", [(k,v/nsteps) for k,v in launch.most_common(10)])
# top kernels within draft phase
top=collections.defaultdict(lambda:[0,0.0])
for k in kern:
    c=(k.get("args") or {}).get("correlation"); ts=corr_ts.get(c)
    if ts is None: continue
    lab=find(ts)
    if lab=="draft":
        t=top[k["name"][:90]]; t[0]+=1; t[1]+=k["dur"]
print("--- top kernels in 'draft' phase (per step)")
for n,(c,d) in sorted(top.items(),key=lambda kv:-kv[1][1])[:14]: print(f"  {d/1e3/nsteps:6.2f}ms {c/nsteps:6.1f}x {n}")
