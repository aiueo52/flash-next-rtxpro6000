#!/usr/bin/env python3
"""Assemble specs/Q1_QUALITY_AUDIT.md from the runs/ directory (analysis via analyze.py).

Only complete arms are reported, chosen by analyze.py's rule (runs/<arm>/DONE from the current run, see
analyze.py); skipped arms are listed on stderr.
Usage: write_report.py [--run-id ID|any] [--runs DIR] [--allow-unmarked] [--out FILE]
On the published results: write_report.py --runs results/quality-q1 --allow-unmarked --out /tmp/q1.md
(arm.log is not published, so timeline and env lines then read "not published")."""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs")
PY = sys.executable
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run-id", help="passed to analyze.py (default: runs/CURRENT_RUN)")
ap.add_argument("--runs", help="arm directory root, passed to analyze.py (default: runs/ next to this script)")
ap.add_argument("--allow-unmarked", action="store_true", help="passed to analyze.py (published results have no DONE markers)")
ap.add_argument("--out", help="report path (default: <bench>/specs/Q1_QUALITY_AUDIT.md, which is git-ignored)")
args = ap.parse_args()
if args.runs: R = os.path.abspath(args.runs)
sel = (["--run-id", args.run_id] if args.run_id else []) + (["--runs", R] if args.runs else []) \
    + (["--allow-unmarked"] if args.allow_unmarked else [])
arms = subprocess.run([PY, os.path.join(HERE, "analyze.py"), "--list-arms"] + sel,
                      stdout=subprocess.PIPE, text=True, check=True).stdout.split()
if "prod" not in arms: sys.exit("write_report.py: the base arm prod is not complete for this run; no report written"
                             " (raw runs are not published in this repository; for the published Q1 results use"
                             " --runs results/quality-q1 --allow-unmarked --out FILE)")
table = subprocess.run([PY, os.path.join(HERE, "analyze.py"), "prod"] + arms + sel,
                       stdout=subprocess.PIPE, text=True, check=True).stdout
def armlog(a):
    p = os.path.join(R, a, "arm.log")
    if not os.path.exists(p):
        return {k: "not published" for k in ("start", "server up", "benchmarks done", "stopped")}
    ls = [l.strip() for l in open(p) if l.startswith(f"[{a}]")]
    return {k: next((l for l in ls if k in l), "") for k in ("start", "server up", "benchmarks done", "stopped")}
times = "\n".join(f"- {a}: {armlog(a)['start'].split(' env')[0]} -> {armlog(a)['benchmarks done']} | {armlog(a)['stopped'].split(';')[0]}" for a in arms)
envs = "\n".join(f"- `{a}`: " + (armlog(a)["start"].split("env: ", 1)[1] if "env: " in armlog(a)["start"] else "") for a in arms)
summ = {a: json.load(open(os.path.join(R, a, "summary.json"))) for a in arms}
trunc = "\n".join(f"- {a}: " + ", ".join(f"{b} {summ[a]['bench'][b]['truncated']}" for b in ("gsm8k", "mmlu", "humaneval", "jcqa") if b in summ[a]["bench"]) for a in arms)
body = open(os.path.join(HERE, "report_template.md")).read()
out = body.replace("{{TABLE}}", table).replace("{{TIMES}}", times).replace("{{ENVS}}", envs).replace("{{TRUNC}}", trunc)
dest = args.out or os.path.join(HERE, "..", "..", "specs", "Q1_QUALITY_AUDIT.md")
os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
open(dest, "w").write(out)
print(out)
