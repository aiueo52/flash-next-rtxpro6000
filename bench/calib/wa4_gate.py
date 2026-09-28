#!/usr/bin/env python3
"""Gate completed WA4 arms against the mean of every supplied A arm."""
import json
from pathlib import Path
import statistics
import sys

B = Path(__file__).resolve().parents[1]
ROOT = B / 'specs/wa4'
ws = ['code-edit', 'prose-en', 'prose-ja', 'agent-loop']
def _require_runs(dirs, files, usage):
    """Refuse (clear message, no traceback) unless every directory holds the files this report reads."""
    missing = [str(Path(d) / f) for d in dirs for f in files if not (Path(d) / f).exists()]
    if not dirs or missing:
        sys.exit("wa4_gate.py: usage: " + usage + "; needs completed run directories with " + ", ".join(files)
                 + ("; missing: " + ", ".join(missing[:6]) if missing else "") + "; WA run directories are not published in this repository (results/wa5 keeps only stripped fnbench records)")
_require_runs(sys.argv[1:], ['complete', 'summary.json', 'command.json'], 'wa4_gate.py <run dir>...')
dirs = [Path(d) for d in sys.argv[1:]]
arms = []
for d in dirs:
    assert (d / 'complete').exists(), d
    a = json.loads((d / 'summary.json').read_text())
    a['arm'] = json.loads((d / 'command.json').read_text())['arm']
    arms.append(a)
controls = [a for a in arms if a['arm'] == 'A']
assert len(controls) >= 2
baseline = {w: statistics.mean(a['rows'][w]['mean'] for a in controls) for w in ws}
gates = dict(zip(ws, [0, -2, -2, 5]))
verdicts = {}
lines = [f'\n## Gate versus all A controls (n={3*len(controls)})\n',
         '| workload | A pooled mean | B t/s | B delta | B floor | C t/s | C delta | C gate | required C/D |',
         '|---|---:|---:|---:|---|---:|---:|---|---:|']
tests = {a['arm']: a for a in arms if a['arm'] != 'A'}
for arm, a in tests.items():
    deltas = {w:100*(a['rows'][w]['mean']/baseline[w]-1) for w in ws}
    passing = all(d>=-2 for d in deltas.values()) and any(d>2 for d in deltas.values()) if arm == 'B' else all(deltas[w]>=gates[w] for w in ws)
    verdicts[arm] = dict(passed=passing, deltas=deltas)
for w in ws:
    b, c = tests['B']['rows'][w]['mean'], tests['C']['rows'][w]['mean']
    db, dc = verdicts['B']['deltas'][w], verdicts['C']['deltas'][w]
    lines.append(f'| {w} | {baseline[w]:.2f} | {b:.2f} | {db:+.2f}% | {"PASS" if db>=-2 else "FAIL"} | {c:.2f} | {dc:+.2f}% | {"PASS" if dc>=gates[w] else "FAIL"} | {gates[w]:+.0f}% |')
if 'D' in tests:
    lines += ['\n| workload | A pooled mean | D t/s | D delta | gate |', '|---|---:|---:|---:|---|']
    for w in ws:
        delta=verdicts['D']['deltas'][w]
        lines.append(f'| {w} | {baseline[w]:.2f} | {tests["D"]["rows"][w]["mean"]:.2f} | {delta:+.2f}% | {"PASS" if delta>=gates[w] else "FAIL"} |')
lines += ['\nB gate: all deltas >=-2%, at least one >+2%. C/D gate: all four workload thresholds must pass.',
          '\nVerdicts: '+', '.join(f'{a} {"PASS" if v["passed"] else "FAIL"}' for a,v in verdicts.items())+'.',
          '\n### Control drift\n', '| workload | A1 mean | A2 mean | A2 vs A1 |'+ (' A3 mean | A3 vs A1 |' if len(controls)==3 else ''),
          '|---|---:|---:|---:|'+ ('---:|---:|' if len(controls)==3 else '')]
for w in ws:
    vals=[a['rows'][w]['mean'] for a in controls]
    row=f'| {w} | {vals[0]:.2f} | {vals[1]:.2f} | {100*(vals[1]/vals[0]-1):+.2f}% |'
    if len(vals)==3:
        row+=f' {vals[2]:.2f} | {100*(vals[2]/vals[0]-1):+.2f}% |'
    lines.append(row)
lines += ['\n### VRAM and needle\n', '| arm | ready MiB | steady median MiB | steady min MiB | >=4096 MiB | needle |', '|---|---:|---:|---:|---|---|']
for a in arms:
    lines.append(f'| {a["label"]} | {a["ready_free_mib"]} | {a["median_steady_free_mib"]:.0f} | {a["min_free_mib"]} | {"PASS" if a["steady_4gib"] else "FAIL"} | PASS |')
result=dict(arms=[a['label'] for a in arms], baseline=baseline, verdicts=verdicts,
            steady_reserve_pass=all(a['steady_4gib'] for a in arms))
output=ROOT/f'gate-{len(arms)}-arms.json'
with output.open('x') as f:
    json.dump(result,f,indent=2)
    f.write('\n')
with (B/'specs/WA4_LOG.md').open('a') as f:
    f.write('\n'.join(lines)+'\n')
print('\n'.join(lines))
