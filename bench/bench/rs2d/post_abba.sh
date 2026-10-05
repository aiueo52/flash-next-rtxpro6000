#!/bin/bash
# post_abba.sh [dir] -- the package ABBA's analyses beyond abba_rs2d.sh's own pooled ANCOVA (RS2d spec 4 steps 3-4):
# per-workload ANCOVA (RS2 §6 needs code-edit and agent-loop), the clock-covariate refit pooled and per workload,
# clocks and display load per arm phase, and the greedy verify graph span per arm. Full output in <dir>/post.log;
# stdout: one line per fit (bench/rs2d/post_summary.py).
set -uo pipefail
cd ~/tools/flash-next-bench
DIR=${1:-runs/rs2d-abba}
LABELS="A1 B1 B2 A2 B3 A3 A4 B4"
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
SPLIT=$DIR/by_workload; rm -rf $SPLIT; mkdir -p $SPLIT
for L in $LABELS; do
  for S in lmstudio greedy; do
    $PY -c '
import json, sys
src, out, label = sys.argv[1:]
for line in open(src):
    with open(f"{out}/{json.loads(line)["workload"]}--{label}.jsonl", "a") as handle:
        handle.write(line)' $DIR/$L-$S.jsonl $SPLIT $L-$S
  done
done
WORKLOADS=$(ls $SPLIT | sed 's/--.*//' | sort -u)
{
  for S in lmstudio greedy; do
    for W in $WORKLOADS; do
      echo "#### ancova $S / $W"
      nice -n 19 $PY bench/stats/ancova_ab.py $(for L in $LABELS; do echo -n "$L=$SPLIT/$W--$L-$S.jsonl "; done) 2>&1
    done
  done
  for S in lmstudio greedy; do
    echo "#### clock ancova $S / pooled"
    nice -n 19 $PY bench/rs2d/ancova_clock.py $DIR/clocks.csv \
        $(for L in $LABELS; do echo -n "$L=$DIR/$L-$S.jsonl "; done) 2>&1
    for W in $WORKLOADS; do
      echo "#### clock ancova $S / $W"
      nice -n 19 $PY bench/rs2d/ancova_clock.py $DIR/clocks.csv \
          $(for L in $LABELS; do echo -n "$L=$SPLIT/$W--$L-$S.jsonl "; done) 2>&1
    done
  done
  echo "#### clocks per arm phase"
  $PY bench/rs2d/clock_phase.py $DIR/clocks.csv $DIR/abba.log 2>&1
  echo "#### display load per arm phase"
  $PY bench/rs2/pmon_phase.py $DIR/pmon.log $DIR/abba.log 2>&1
  echo "#### greedy code-edit graph spans"
  nice -n 19 $PY prof/graph_span.py $(for L in $LABELS; do echo -n "$L=$DIR/traces/$L-g-code-edit "; done) 2>&1 | tail -8
} > $DIR/post.log
$PY bench/rs2d/post_summary.py $DIR/post.log $DIR/abba.log
