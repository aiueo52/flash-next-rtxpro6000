#!/usr/bin/env python3
"""Extract switch/grace evidence from completed WA4 logs without GPU work."""
import datetime as dt
import json
from pathlib import Path
import re
import statistics
import sys

def _require_runs(dirs, files, usage):
    """Refuse (clear message, no traceback) unless every directory holds the files this report reads."""
    missing = [str(Path(d) / f) for d in dirs for f in files if not (Path(d) / f).exists()]
    if not dirs or missing:
        sys.exit("wa4_diagnose.py: usage: " + usage + "; needs completed run directories with " + ", ".join(files)
                 + ("; missing: " + ", ".join(missing[:6]) if missing else "") + "; WA run directories are not published in this repository (results/wa5 keeps only stripped fnbench records)")
_require_runs(sys.argv[1:], ['events.jsonl', 'server.log', 'command.json'], 'wa4_diagnose.py <run dir>...')
for directory in sys.argv[1:]:
    d = Path(directory)
    events = [json.loads(l) for l in (d / 'events.jsonl').read_text().splitlines()]
    grace, last_dir, batch = 40, 0, None
    arm = json.loads((d / "command.json").read_text())["arm"]
    cap = 2000 if arm in ("A", "B") else 80
    switches = []
    for line in (d / 'server.log').read_text().splitlines():
        m = re.search(r'\[adaptive-conf\].*batch=(\d+)', line)
        if m:
            batch = int(m[1])
            if switches and 'next_decision' not in switches[-1]:
                switches[-1]['next_decision'] = line
        m = re.search(r'^\[(.*?)\].*Adaptive spec params updated.*steps (\d+) -> (\d+)', line)
        if not m:
            continue
        direction = 1 if int(m[3]) > int(m[2]) else -1
        grace = min(cap, max(1, grace) * 2) if direction == -last_dir else 40
        last_dir = direction
        t = dt.datetime.fromisoformat(m[1]).replace(tzinfo=dt.timezone(dt.timedelta(hours=9))).timestamp() + .5
        event = next((e for e in events if e['start'] <= t <= e['end']), None)
        switches.append(dict(time=m[1], old=int(m[2]), new=int(m[3]), batch=batch,
                             grace=grace, until=batch+grace,
                             workload=event['workload'] if event else 'boundary/needle',
                             repeat=event['repeat'] if event else None, log=line))
    result = dict(label=d.name, timestamp_caveat='Workload attribution uses the midpoint of second-resolution log timestamps; near request boundaries treat it as approximate.', switches=switches)
    traces = [json.loads(l) for l in (d / 'trace.jsonl.rank0').read_text().splitlines()]
    records = [json.loads(l) for l in (d / 'runs.jsonl').read_text().splitlines()]
    result['effective_step_estimates'] = []
    for record in records:
        event = next(e for e in events if e['workload'] == record['workload'] and e['repeat'] == record['repeat'])
        measured = [t for t in traces if event['start'] <= t['t'] <= event['end']]
        acceptance = statistics.mean(1 + t['a'][0] for t in measured)
        result['effective_step_estimates'].append(dict(
            workload=record['workload'], repeat=record['repeat'],
            steps=sorted(set(t['steps'] for t in measured)),
            acceptance=acceptance,
            prefix_s3=statistics.mean(1 + min(t['a'][0], 3) for t in measured),
            effective_step_ms=1000 * acceptance / record['client']['decode_tps'],
        ))
    (d / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(d.name)
    for r in switches:
        print(f"  {r['time']} {r['old']}->{r['new']} batch={r['batch']} grace={r['grace']} until={r['until']} {r['workload']} repeat={r['repeat']}")
