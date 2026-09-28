#!/usr/bin/env python3
"""WS1 width-sweep report: per arm x workload, acceptance (tokens/verify, from the server's
windowed 'accept len' log lines that fall inside each measured request) and client decode t/s;
plus trimmed step time per arm from the sweep log (prof/trimmed_step.py output).

Only arms with runs/<prefix>-<arm>-<profile>.done (written by ws1_sweep.sh after a successful arm) are
reported; others are listed on stderr. --allow-unmarked accepts arms without it (sweeps that predate the
marker; their completeness is then unchecked).
Usage: ws1_report.py [--allow-unmarked] [--runs DIR] [--server-logs DIR] <prefix> <sweep log>
On the published data: ws1_report.py --allow-unmarked --runs results/runs ws11337 bench/calib/logs/ws1_sweep.log
(the server logs are not published, so the server-log acceptance cross-check then reads n/a)."""
import argparse, json, re, sys, glob, os, datetime as dt
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("prefix", help="label prefix, e.g. ws11337")
ap.add_argument("sweeplog", help="calib/logs/ws1_sweep.log")
ap.add_argument("--allow-unmarked", action="store_true")
ap.add_argument("--runs", default=os.path.expanduser('~/tools/flash-next-bench/runs'), help="fnbench JSONL directory")
ap.add_argument("--server-logs", default=os.path.expanduser('~/tools/sglang-rtxpro6000/logs'))
_a = ap.parse_args()
ALLOW_UNMARKED = _a.allow_unmarked
prefix = _a.prefix              # e.g. ws11337
sweeplog = _a.sweeplog          # calib/logs/ws1_sweep.log
RUNS = _a.runs; LOGS = _a.server_logs
if not os.path.isfile(sweeplog): sys.exit(f"ws1_report: no sweep log {sweeplog}")
ARMS = ['w4','w6','w8','w12','w16','w4b']
WL = ['code-edit','prose-en','prose-ja','agent-loop']
LOCAL = dt.datetime.now().astimezone().tzinfo

def accept_lines(log):
    out=[]
    for l in open(log):
        m=re.match(r'\[(\S+ \S+)\] Decode batch.*accept len: ([0-9.]+)', l)
        if m: out.append((dt.datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S').replace(tzinfo=LOCAL), float(m.group(2))))
    return out

# step times from sweep log
steps={}  # (arm, wl) -> (wall, trimmed)
cur=None
for l in open(sweeplog):
    m=re.search(r'traces/'+prefix+r'-(\w+)-w\d+-([\w-]+)/', l)
    if m: cur=(m.group(1), m.group(2)); continue
    m=re.search(r'step wall=\s*([0-9.]+)ms', l)
    if m and cur: steps.setdefault(cur,[None,None])[0]=float(m.group(1))
    m=re.search(r'TOTAL.*legacy_trimmed_ms=\s*([0-9.]+)', l)
    if m and cur: steps[cur][1]=float(m.group(1))

def marked(arm):
    prof='w16' if arm=='w16' else 'w4'
    return ALLOW_UNMARKED or os.path.exists(f'{RUNS}/{prefix}-{arm}-{prof}.done')
COMPLETE={arm for arm in ARMS if marked(arm)}
for arm in ARMS:
    if arm not in COMPLETE:
        print(f'ws1_report: skipping arm {arm}: no .done marker (arm failed, incomplete or not run)', file=sys.stderr)

rows={}
for arm in ARMS:
    prof='w16' if arm=='w16' else 'w4'
    f=f'{RUNS}/{prefix}-{arm}-{prof}.jsonl'
    if not os.path.exists(f) or arm not in COMPLETE: continue
    log=f'{LOGS}/serve-{prefix}-{arm}-{prof}.log'
    al=accept_lines(log) if os.path.isfile(log) else []   # server logs are not published
    recs=[json.loads(l) for l in open(f)]
    for wl in WL:
        rs=[r for r in recs if r['workload']==wl]
        tps=[r['client']['decode_tps'] for r in rs]; accs=[]; chunk_acc=[]
        for r in rs:
            end=dt.datetime.fromisoformat(r['timestamp']); start=end-dt.timedelta(seconds=r['client']['decode_seconds'])  # log lines are 1 s-granular: skip the first 1.5 s so the previous request's tail windows are excluded
            v=[a for t,a in al if start+dt.timedelta(seconds=1.5)<=t<=end+dt.timedelta(seconds=0.5)]
            if v: accs.append(sum(v)/len(v))
            chunks=len(r['client']['timeline']) if r['client'].get('timeline') else r['client']['sse_chunks']  # stripped records keep the count
            chunk_acc.append(r['client']['usage']['completion_tokens']/chunks)
        rows[(arm,wl)]=dict(tps=sum(tps)/len(tps), tps_min=min(tps), tps_max=max(tps), acc=sum(accs)/len(accs) if accs else float('nan'), nwin=len(accs), chunk=sum(chunk_acc)/len(chunk_acc))

if not rows:
    sys.exit(f"ws1_report: no complete arm with results for prefix {prefix} in {RUNS}; nothing to report (raw runs are not published in this repository; for the published records use --runs results/runs --allow-unmarked)")
print('| arm | ' + ' | '.join(f'{w} acc / t/s' for w in WL) + ' |')
print('|---|' + '---|'*len(WL))
for arm in ARMS:
    if (arm,WL[0]) not in rows: continue
    print(f'| {arm} | ' + ' | '.join(f"{rows[(arm,w)]['chunk']:.2f} / {rows[(arm,w)]['tps']:.0f} ({rows[(arm,w)]['tps_min']:.0f}-{rows[(arm,w)]['tps_max']:.0f})" for w in WL) + ' |')
print()
print('Cross-check: acceptance from the server log windows (40 verify steps each, first 1.5 s of each request excluded):')
print('| arm | ' + ' | '.join(WL) + ' |'); print('|---|' + '---|'*len(WL))
for arm in ARMS:
    if (arm,WL[0]) not in rows: continue
    print(f'| {arm} | ' + ' | '.join('n/a' if rows[(arm,w)]['acc'] != rows[(arm,w)]['acc'] else f"{rows[(arm,w)]['acc']:.2f}" for w in WL) + ' |')
print()
print('| arm | code-edit wall ms | code-edit trimmed ms | prose-en wall ms | prose-en trimmed ms |')
print('|---|---|---|---|---|')
for arm in ARMS:
    if (arm,'code-edit') not in steps or arm not in COMPLETE: continue
    c=steps[(arm,'code-edit')]; p=steps[(arm,'prose-en')]
    print(f'| {arm} | {c[0]:.2f} | {c[1]:.2f} | {p[0]:.2f} | {p[1]:.2f} |')
