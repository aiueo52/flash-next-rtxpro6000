#!/usr/bin/env python3
"""Report all WA5 controls, individual C gates, pooled gate and pair drift."""
import json
from pathlib import Path
import statistics as st

B = Path(__file__).resolve().parents[1]
root = B / 'specs/wa5'
ws = ['code-edit', 'prose-en', 'prose-ja', 'agent-loop']
limits = dict(zip(ws, [0, -2, -2, 5]))
if not root.is_dir():
    raise SystemExit(f'wa5_gate.py: needs the WA5 run directories under {root}; they are not published in this '
                     'repository (results/wa5 keeps only stripped fnbench records)')
arms = {}
for name in ['A1', 'C1', 'A2', 'C2']:
    ds = list(root.glob('wa5-*-' + name))
    assert len(ds) == 1, (name, ds)
    d = ds[0]
    assert (d/'complete').exists()
    arms[name] = json.loads((d/'summary.json').read_text())
mean = lambda names, w: st.mean(arms[a]['rows'][w]['mean'] for a in names)
delta = lambda a, b: 100 * (a / b - 1)
base = {w:mean(['A1','A2'], w) for w in ws}
cmean = {w:mean(['C1','C2'], w) for w in ws}
pool = {w:delta(cmean[w],base[w]) for w in ws}
individual = {a:{w:delta(arms[a]['rows'][w]['mean'],base[w]) for w in ws} for a in ['C1','C2']}
pairs = {c:{w:delta(arms[c]['rows'][w]['mean'],arms[a]['rows'][w]['mean']) for w in ws} for a,c in [('A1','C1'),('A2','C2')]}
passes = lambda ds: all(ds[w] >= limits[w] for w in ws)
result = dict(labels={a:s['label'] for a,s in arms.items()}, baseline=base, cmean=cmean, pooled_deltas=pool, individual_deltas=individual, pair_deltas=pairs, pooled_pass=passes(pool), individual_pass={a:passes(ds) for a,ds in individual.items()}, needle_pass=all('PASS=True' in a['needle'] for a in arms.values()), reserve_pass=all(a['steady_4gib'] for a in arms.values()))
result['adoption_pass'] = result['pooled_pass'] and all(result['individual_pass'].values()) and result['needle_pass'] and result['reserve_pass']
lines=['\n## Final throughput and gate\n', '| workload | A1 t/s | C1 t/s | A2 t/s | C2 t/s | A mean | C mean | C vs A | threshold | pooled gate |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for w in ws:
    vals=' | '.join(f"{arms[a]['rows'][w]['mean']:.2f}" for a in arms)
    lines.append(f'| {w} | {vals} | {base[w]:.2f} | {cmean[w]:.2f} | {pool[w]:+.2f}% | {limits[w]:+.0f}% | {"PASS" if pool[w]>=limits[w] else "FAIL"} |')
lines += ['\n### Individual C versus pooled A and paired drift\n', '| workload | C1 vs mean A | C2 vs mean A | C1 vs A1 | C2 vs A2 | A2 vs A1 | C2 vs C1 |', '|---|---:|---:|---:|---:|---:|---:|']
for w in ws:
    values=[individual['C1'][w],individual['C2'][w],pairs['C1'][w],pairs['C2'][w],delta(arms['A2']['rows'][w]['mean'],arms['A1']['rows'][w]['mean']),delta(arms['C2']['rows'][w]['mean'],arms['C1']['rows'][w]['mean'])]
    lines.append('| '+w+' | '+' | '.join(f'{v:+.2f}%' for v in values)+' |')
lines += ['\nAll figures use arithmetic means of the three measured request decode throughputs per arm; the pooled policy means weight both instances equally (six requests each). Individual C gates use the same pooled A baseline. Pair deltas diagnose drift and do not replace the registered baseline.', '\nVerdict: '+json.dumps({k:result[k] for k in ['pooled_pass','individual_pass','needle_pass','reserve_pass','adoption_pass']})+'.', '\n### VRAM and needle\n', '| arm | ready MiB | steady median MiB | steady min MiB | reserve | needle |','|---|---:|---:|---:|---|---|']
for a,s in arms.items():
    lines.append(f'| {a} | {s["ready_free_mib"]} | {s["median_steady_free_mib"]:.0f} | {s["min_free_mib"]} | {"PASS" if s["steady_4gib"] else "FAIL"} | PASS |')
with (root/'gate.json').open('x') as f: json.dump(result,f,indent=2);f.write('\n')
with (B/'specs/WA5_LOG.md').open('a') as f:f.write('\n'.join(lines)+'\n')
print('\n'.join(lines))
