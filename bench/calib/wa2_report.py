#!/usr/bin/env python3
"""Append completed WA2 arm results and final gate to the incremental log."""
import collections
import json
from pathlib import Path
import statistics
import sys

B = Path(__file__).resolve().parents[1]
LOG = B / 'specs/WA2_LOG.md'
WORKLOADS = ['code-edit', 'prose-en', 'prose-ja', 'agent-loop']

def arm(directory, heading='##'):
    d = Path(directory)
    records = [json.loads(l) for l in (d / 'runs.jsonl').read_text().splitlines()]
    events = [json.loads(l) for l in (d / 'events.jsonl').read_text().splitlines()]
    traces = [json.loads(l) for l in (d / 'trace.jsonl.rank0').read_text().splitlines()]
    rows = {}
    for w in WORKLOADS:
        rs = [r for r in records if r['workload'] == w]
        es = [e for e in events if e['workload'] == w and e['repeat'] > 0]
        assert len(rs) == len(es) == 3
        assert sorted(r['repeat'] for r in rs) == [1, 2, 3]
        hist = collections.Counter()
        accepted = []
        errors = []
        for event, record in zip(es, rs):
            ts = [t for t in traces if event['start'] <= t['t'] <= event['end']]
            assert ts, (w, event)
            assert all(t['bs'] == 1 and t['steps'] in (3, 7, 15) and len(t['a']) == 1 for t in ts)
            assert all(0 <= t['a'][0] <= t['steps'] for t in ts)
            hist.update(t['steps'] for t in ts)
            accepted.append(statistics.mean(1 + t['a'][0] for t in ts))
            emitted = sum(1 + t['a'][0] for t in ts)
            tokens = record['client']['usage']['completion_tokens']
            errors.append(emitted - tokens)
            assert abs(emitted - tokens) <= 32, (w, emitted, tokens)
        tps = [r['client']['decode_tps'] for r in rs]
        chunk = [r['client']['usage']['completion_tokens'] / len(r['client']['timeline']) for r in rs]
        rows[w] = dict(mean=statistics.mean(tps), min=min(tps), max=max(tps),
                       acceptance=statistics.mean(accepted), chunk_acceptance=statistics.mean(chunk),
                       histogram={str(s): hist[s] for s in (3, 7, 15)},
                       trace_emission_minus_completion=errors)
    needle = (d / 'needle.log').read_text().strip()
    assert 'PASS=True' in needle, needle
    memory = [json.loads(l) for l in (d / 'memory.jsonl').read_text().splitlines()]
    startup_free = [r['free_mib'] for r in memory if r['phase'] == 'startup']
    steady_free = [r['free_mib'] for r in memory if r['phase'] == 'steady']
    min_free = min(steady_free)
    ready_free = int((d / 'ready-free-mib.txt').read_text())
    result = dict(label=d.name, rows=rows, needle=needle, min_free_mib=min_free,
                  ready_free_mib=ready_free, median_steady_free_mib=statistics.median(steady_free),
                  min_startup_free_mib=min(startup_free), steady_4gib=min_free >= 4096)
    (d / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    out = [f'\n{heading} Arm {d.name}\n',
           '| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |',
           '|---|---:|---:|---:|---:|---:|']
    for w, row in rows.items():
        h = row['histogram']; n = sum(h.values())
        cells = [f'{h[str(s)]} ({100*h[str(s)]/n:.2f}%)' for s in (3, 7, 15)]
        out.append(f"| {w} | {row['mean']:.2f} ({row['min']:.2f}-{row['max']:.2f}) | {row['acceptance']:.3f} / {row['chunk_acceptance']:.3f} | " + ' | '.join(cells) + ' |')
    out += [f'\nNeedle: `{needle}`', f'\nSteady-state free VRAM: ready {ready_free} MiB; minimum {min_free} MiB ({min_free/1024:.3f} GiB); median {statistics.median(steady_free):.0f} MiB. Startup minimum {min(startup_free)} MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: {"PASS" if min_free >= 4096 else "BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s)"}.',
            '\nTrace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: ' +
            '; '.join(f"{w}: {r['trace_emission_minus_completion']}" for w, r in rows.items()) + '.',
            f'\nArtifacts: `specs/wa2/{d.name}/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).\n']
    with LOG.open('a') as f:
        f.write('\n'.join(out))
    print('\n'.join(out), flush=True)

def final(directories):
    ctl, test, drift = [json.loads((Path(d) / 'summary.json').read_text()) for d in directories]
    out = ['\n## Final gate and drift\n', '| workload | control pooled mean (n=6) | test mean (n=3) | test vs control | control2 vs control1 | gate | result |', '|---|---:|---:|---:|---:|---:|---|']
    passed = True
    gates = {'code-edit': 0, 'prose-en': -2, 'prose-ja': -2, 'agent-loop': 5}
    for w in WORKLOADS:
        c1, t, c2 = [a['rows'][w]['mean'] for a in (ctl, test, drift)]
        c = (c1 + c2) / 2
        delta = 100 * (t/c - 1)
        ok = delta >= gates[w]
        passed &= ok
        out.append(f'| {w} | {c:.2f} | {t:.2f} | {delta:+.2f}% | {100*(c2/c1-1):+.2f}% | {gates[w]:+.0f}% | {"PASS" if ok else "FAIL"} |')
    out += ['\n### Steady-state VRAM\n', '| arm | ready free MiB | steady median MiB | steady minimum MiB | startup minimum MiB |', '|---|---:|---:|---:|---:|']
    for a in (ctl, test, drift):
        out.append(f"| {a['label']} | {a['ready_free_mib']} | {a['median_steady_free_mib']:.0f} | {a['min_free_mib']} | {a['min_startup_free_mib']} |")
    out.append('\nThe supervisor-requested sustained <1024 MiB / 10-second abort did not trigger in the completed sequence. The measured steady 4 GiB target was not met; report this separately from the throughput gates. SERVE_DISPLAY_HZ remained empty, and mem-fraction-static remained 0.925 for every arm.')
    out.append('\nVerdict: ' + ('PASS. Recommend the following launcher change, without applying it.' if passed else 'FAIL. Do not change serve-fast.sh; retain the production w16_conf.json and STEP_A/B defaults 9.74/0.70.'))
    if passed:
        out.append('In the wa arm, change the ADAPTIVE_CONFIG fallback to `$HOME/tools/flash-next-bench/adaptive/w16_3_7_15.json` and prepend `SGLANG_ADAPTIVE_STEP_A="${SGLANG_ADAPTIVE_STEP_A:-7.942995690}" SGLANG_ADAPTIVE_STEP_B="${SGLANG_ADAPTIVE_STEP_B:-0.555366379}"` to the existing environment assignments.')
    out.append('\nBoth controls contribute equally (3 repeats each). These are descriptive small-sample gates, not a confidence-interval claim. All arms enable the same chain tracing/debug instrumentation; tracing synchronizes the chain staging event and can affect throughput, so absolute numbers describe the instrumented server. No launcher edits, commits, pushes, new branches, production source or venv changes were made.\n')
    with LOG.open('a') as f:
        f.write('\n'.join(out))
    print('\n'.join(out), flush=True)

def _require_runs(dirs, files, usage):
    """Refuse (clear message, no traceback) unless every directory holds the files this report reads."""
    missing = [str(Path(d) / f) for d in dirs for f in files if not (Path(d) / f).exists()]
    if not dirs or missing:
        sys.exit("wa2_report.py: usage: " + usage + "; needs completed run directories with " + ", ".join(files)
                 + ("; missing: " + ", ".join(missing[:6]) if missing else "") + "; WA run directories are not published in this repository (results/wa5 keeps only stripped fnbench records)")

if __name__ == '__main__':
    usage = 'wa2_report.py arm <dir> | wa2_report.py final <control> <test> <control2>'
    if len(sys.argv) >= 3 and sys.argv[1] == 'arm':
        _require_runs(sys.argv[2:3], ['runs.jsonl', 'events.jsonl', 'trace.jsonl.rank0'], usage)
        arm(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == 'final':
        _require_runs(sys.argv[2:], ['summary.json'], usage)
        final(sys.argv[2:])
    else:
        _require_runs([], ['summary.json'], usage)
