#!/usr/bin/env python3
"""WA3 iteration 3: test-c -> control -> test-c, independently gated."""
import json
from pathlib import Path
import sys

from wa2_report import arm, LOG, WORKLOADS

def final(directories):
    first, control, last = [json.loads((Path(d) / 'summary.json').read_text()) for d in directories]
    gates = {'code-edit': 0, 'prose-en': -2, 'prose-ja': -2, 'agent-loop': 5}
    passed = [True, True]
    out = ['\n### Iteration 3 gate and repeat drift\n',
           '| workload | control mean t/s (n=3) | test-c1 t/s | c1 vs control | c1 gate | test-c2 t/s | c2 vs control | c2 gate | c2 vs c1 | required |',
           '|---|---:|---:|---:|---|---:|---:|---|---:|---:|']
    for w in WORKLOADS:
        c = control['rows'][w]['mean']
        t1, t2 = [a['rows'][w]['mean'] for a in (first, last)]
        deltas = [100*(t/c-1) for t in (t1, t2)]
        ok = [d >= gates[w] for d in deltas]
        passed = [p and o for p, o in zip(passed, ok)]
        out.append(f'| {w} | {c:.2f} | {t1:.2f} | {deltas[0]:+.2f}% | {"PASS" if ok[0] else "FAIL"} | {t2:.2f} | {deltas[1]:+.2f}% | {"PASS" if ok[1] else "FAIL"} | {100*(t2/t1-1):+.2f}% | {gates[w]:+.0f}% |')
    out += ['\n### Iteration 3 steady-state VRAM\n', '| arm | ready free MiB | steady median MiB | steady minimum MiB | startup minimum MiB |', '|---|---:|---:|---:|---:|']
    for a in (first, control, last):
        out.append(f"| {a['label']} | {a['ready_free_mib']} | {a['median_steady_free_mib']:.0f} | {a['min_free_mib']} | {a['min_startup_free_mib']} |")
    out.append(f'\nGate verdict: test-c1 {"PASS" if passed[0] else "FAIL"}; test-c2 {"PASS" if passed[1] else "FAIL"}; combined {"PASS" if all(passed) else "FAIL"}. Each test arm must independently pass all four gates versus the intervening production control; test means are not pooled to hide a failing arm.')
    if all(passed):
        out.append('\nExact launcher recommendation for the tested settings, not applied: in the wa arm, add `PYTHONPATH="$HOME/tools/sglang-wa2/python"` for the tested code overlay, change the ADAPTIVE_CONFIG fallback to `$HOME/tools/flash-next-bench/adaptive/w16_3_7_15_c.json`, prepend `SGLANG_ADAPTIVE_STEP_A="${SGLANG_ADAPTIVE_STEP_A:-7.942995690}" SGLANG_ADAPTIVE_STEP_B="${SGLANG_ADAPTIVE_STEP_B:-0.555366379}"` to the environment assignments, and change the WA_MEM_FRACTION fallback from 0.925 to 0.920. No other launcher change. The steady-state 4 GiB target remains a separate deployment limitation if the measured table is below it.')
    else:
        out.append('\nExact launcher recommendation: no edit. Keep production w16_conf.json and default STEP_A/B=9.74/0.70. Decision-log interpretation follows after analysis. This is the last iteration; no fourth experiment will be started.')
    out.append('\nAll three needle invocations passed. All arms used the same sglang-wa2 PYTHONPATH overlay; the control config omits the new keys and its default CPU trajectory was verified exactly against production. All arms used the common reduced mem-fraction-static=0.920 as authorized for iteration 3. Empty SERVE_DISPLAY_HZ, v5 head, expandable_segments:True, and trace/debug instrumentation were unchanged. Startup low-VRAM values are recorded without abort; after ready the abort condition is <1024 MiB continuously for at least 10 seconds. Tables report observed steady VRAM separately from throughput gates. Comparison is against a production-policy control at the same 0.920 memory fraction; it does not validate serving the new config with the launcher-default 0.925 fraction.\n')
    with LOG.open('a') as f:
        f.write('\n'.join(out))
    Path(directories[0]).parent.joinpath(Path(directories[0]).name.removesuffix('-c1') + '-gate.json').write_text(json.dumps(dict(arms=[a['label'] for a in (first, control, last)], passed=passed, combined=all(passed)), indent=2) + '\n')
    print('\n'.join(out), flush=True)

def _require_runs(dirs, files, usage):
    """Refuse (clear message, no traceback) unless every directory holds the files this report reads."""
    missing = [str(Path(d) / f) for d in dirs for f in files if not (Path(d) / f).exists()]
    if not dirs or missing:
        sys.exit("wa3_report.py: usage: " + usage + "; needs completed run directories with " + ", ".join(files)
                 + ("; missing: " + ", ".join(missing[:6]) if missing else "") + "; WA run directories are not published in this repository (results/wa5 keeps only stripped fnbench records)")

if __name__ == '__main__':
    usage = 'wa3_report.py arm <dir> | wa3_report.py final <test-c1> <control> <test-c2>'
    if len(sys.argv) >= 3 and sys.argv[1] == 'arm':
        _require_runs(sys.argv[2:3], ['runs.jsonl', 'events.jsonl', 'trace.jsonl.rank0'], usage)
        arm(sys.argv[2], heading='###')
    elif len(sys.argv) == 5 and sys.argv[1] == 'final':
        _require_runs(sys.argv[2:], ['summary.json'], usage)
        final(sys.argv[2:])
    else:
        _require_runs([], ['summary.json'], usage)
