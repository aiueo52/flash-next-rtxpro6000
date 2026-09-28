#!/bin/bash
set -euo pipefail
BENCH=$HOME/tools/flash-next-bench
PREFLIGHT=${1:?preflight prefix}
PROFILE=${2:?width}
LABEL=${3:?label}
# One blocking wait, no lock acquisition polling. GPU starts only after the
# numerical preflight artifact passes and our turn on the shared flock arrives.
while [ ! -s "$PREFLIGHT.json" ]; do
  if rg -q 'Traceback|AssertionError|CUDA error' "$PREFLIGHT.log"; then
    echo 'P3 preflight failed; capture not started'; exit 1
  fi
  sleep 15
done
python3 - "$PREFLIGHT.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert len(r)==7 and r[-1].get('recorder_graph_pass'),r
assert all(x['output_unchanged'] for x in r[:-1])
print('Validated P3 preflight; queueing capture',flush=True)
PY
exec flock -w 28800 $HOME/.gpu.lock "$BENCH/prof/p3_capture_run.sh" "$PROFILE" "$LABEL"
