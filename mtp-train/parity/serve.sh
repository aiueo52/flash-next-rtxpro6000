#!/bin/bash
# Start / wait for / stop a hook-enabled SGLang server for the MTP training pipeline.
#
# The launchers are this repository's launch/serve-local.sh and launch/serve-fast.sh; they run the
# SGLang checkout in $SGLANG_DIR (with its .venv). The MTP dump hook lives on the branch
# flash-next-mtp-dump, which patches/sglang/mtp-dump/apply.sh creates in a separate worktree of the
# serving clone; point SGLANG_DUMP_TREE at that worktree. Only PYTHONPATH changes, which is enough to make the
# worktree's patched model files the ones that get imported (the editable install registers its
# finder with sys.meta_path.append, i.e. after the standard PathFinder, so PYTHONPATH wins).
#
#   SGLANG_DIR=...        patched serving checkout with .venv (as for launch/serve-fast.sh)
#   SGLANG_DUMP_TREE=...  worktree of the same clone on branch flash-next-mtp-dump
#   TARGET_MODEL=...      checkpoint to serve (default: see launch/serve-local.sh)
#
#   ./serve.sh start-extract <logfile> [sglang args]  target dump, production w4 config.
#                                       Set SGLANG_MTP_DUMP_DIR (and SGLANG_MTP_DUMP_TOPK=8) first.
#   ./serve.sh start-prod    <logfile> [sglang args]  serve-fast.sh w4 as served (radix cache off)
#   ./serve.sh wait  <logfile> [timeout_s]
#   ./serve.sh stop  <pgidfile>          (written next to the log as <log>.pgid)
#   ./serve.sh gpu-idle                  exit 0 iff no compute process is running
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAUNCH="${LAUNCH_DIR:-$(cd "$HERE/../../launch" 2>/dev/null && pwd)}"
PORT="${PORT:-8001}"
READY="The server is fired up and ready to roll"

need_env() {
  : "${SGLANG_DIR:?set SGLANG_DIR to the patched sglang-rtxpro6000 checkout}"
  : "${SGLANG_DUMP_TREE:?set SGLANG_DUMP_TREE to a worktree on branch flash-next-mtp-dump}"
  [ -f "$LAUNCH/serve-local.sh" ] || { echo "launch scripts not found in $LAUNCH (set LAUNCH_DIR)" >&2; exit 2; }
  [ -d "$SGLANG_DIR" ] || { echo "SGLANG_DIR=$SGLANG_DIR: no such directory" >&2; exit 2; }
  # A PYTHONPATH entry that does not exist is silently skipped by Python: the
  # server would start without the hook and dump nothing.
  [ -d "$SGLANG_DUMP_TREE/python" ] || {
    echo "SGLANG_DUMP_TREE=$SGLANG_DUMP_TREE has no python/ (not a dump-branch worktree)" >&2; exit 2; }
  export SGLANG_DIR PORT
}

need_log() {                     # need_log <mode> <logfile>
  [ -n "${2:-}" ] || { echo "usage: $0 $1 <logfile> [sglang args]" >&2; exit 2; }
}

need_dump_dir() {                # need_dump_dir <VAR NAME> <value>
  [ -n "${2:-}" ] || { echo "set $1 to the dump output directory" >&2; exit 2; }
  if [ -e "$2" ] && [ ! -d "$2" ]; then echo "$1=$2 exists and is not a directory" >&2; exit 2; fi
}

gpu_idle() {
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
  [ -z "$apps" ]
}

launch() {                       # launch <logfile> <script> [args...]
  local log="$1"; shift
  if ! gpu_idle; then
    echo "REFUSING TO LAUNCH: GPU has compute processes:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    return 1
  fi
  rm -f "$log"
  setsid nohup "$@" > "$log" 2>&1 < /dev/null &
  local pid=$!
  sleep 2
  local pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  echo "$pgid" > "${log%.log}.pgid"
  echo "launched pid=$pid pgid=$pgid log=$log"
}

case "${1:-}" in
  gpu-idle)
    if gpu_idle; then echo "GPU idle"; exit 0; else
      nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv; exit 1; fi ;;

  start-prod)
    need_env; need_log start-prod "${2:-}"; LOG="$2"; shift 2
    PYTHONPATH="$SGLANG_DUMP_TREE/python" \
      launch "$LOG" "$LAUNCH/serve-fast.sh" w4 --disable-radix-cache "$@" ;;

  start-extract)
    need_env; need_log start-extract "${2:-}"; LOG="$2"; shift 2
    need_dump_dir SGLANG_MTP_DUMP_DIR "${SGLANG_MTP_DUMP_DIR:-}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}" MAMBA_SLOTS="${MAMBA_SLOTS:-10}" \
    PYTHONPATH="$SGLANG_DUMP_TREE/python" \
      launch "$LOG" "$LAUNCH/serve-fast.sh" w4 --disable-radix-cache "$@" ;;

  wait)
    need_log wait "${2:-}"
    case "${3:-600}" in ''|*[!0-9]*) echo "timeout must be whole seconds" >&2; exit 2 ;; esac
    LOG="$2"; DEADLINE=$(( $(date +%s) + ${3:-600} ))
    until grep -qF "$READY" "$LOG" 2>/dev/null; do
      if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "TIMEOUT waiting for server"; tail -30 "$LOG"; exit 1
      fi
      if grep -qiE "Traceback|CUDA out of memory|Error in|Initialization failed" "$LOG" 2>/dev/null; then
        echo "SERVER ERROR"; tail -40 "$LOG"; exit 1
      fi
      sleep 5
    done
    echo "server ready"; curl -s -m 10 "http://127.0.0.1:$PORT/health" && echo " /health ok" ;;

  stop)
    # Only the process group this script started is signalled.
    [ -f "${2:-}" ] || { echo "usage: $0 stop <pgidfile> (no such file: ${2:-})" >&2; exit 2; }
    PGID="$(cat "$2" 2>/dev/null)"
    [ -n "${PGID:-}" ] || { echo "no pgid in $2" >&2; exit 1; }
    kill -TERM -- "-$PGID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 -- "-$PGID" 2>/dev/null || break; sleep 2; done
    if kill -0 -- "-$PGID" 2>/dev/null; then
      echo "TERM did not stop process group $PGID; sending KILL"
      kill -KILL -- "-$PGID" 2>/dev/null
    fi
    echo "stopped process group $PGID" ;;

  *) echo "usage: $0 {gpu-idle|start-extract|start-prod|wait|stop}" >&2; exit 2 ;;
esac
