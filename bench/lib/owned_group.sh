# owned_group.sh -- sourced (not executed) by the bench scripts that start a GPU server.
#
# Only the process group that the calling script started is ever signalled. GPU processes this
# script did not start are never killed: if the GPU is not idle when a run needs it, the script
# refuses and lists them instead.
#
#   require_idle_gpu [wait_s] [settle_s]  wait up to wait_s for no compute processes, then settle_s
#                                         more and check again; otherwise list them and return 1
#   start_owned <log> <cmd> [args...]     run cmd under setsid (its own process group), output to
#                                         log; sets OWNED_PID / OWNED_PGID
#   stop_owned [term_timeout_s]           TERM that group, wait, then KILL the same group
#   owned_cleanup                         EXIT-trap handler: stop_owned if a group is still recorded
#   report_gpu_leftovers [wait_s]         after a stop: wait for the GPU to show no compute processes,
#                                         else warn and list them (never kills them)
#
# Typical use:
#   . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../lib/owned_group.sh"
#   trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM

OWNED_PID=""; OWNED_PGID=""

gpu_compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}

require_idle_gpu() {
  local wait_s=${1:-0} settle_s=${2:-0} deadline apps
  deadline=$(( $(date +%s) + wait_s ))
  while :; do
    apps=$(gpu_compute_apps) || { echo "nvidia-smi failed; refusing to start" >&2; return 1; }
    if [ -z "$apps" ]; then
      [ "$settle_s" -gt 0 ] || return 0
      sleep "$settle_s"
      apps=$(gpu_compute_apps) || { echo "nvidia-smi failed; refusing to start" >&2; return 1; }
      [ -z "$apps" ] && return 0
    fi
    [ "$(date +%s)" -lt "$deadline" ] || break
    echo "waiting for GPU (busy: $(echo "$apps" | tr '\n' ';'))" >&2
    sleep 10
  done
  echo "REFUSING TO START: the GPU has compute processes this script did not start" >&2
  echo "(they are NOT killed; stop them yourself or wait):" >&2
  echo "$apps" >&2
  return 1
}

start_owned() {
  local log=$1 pg="" i; shift
  [ -z "$OWNED_PGID" ] || { echo "start_owned: process group $OWNED_PGID is still recorded" >&2; return 1; }
  setsid "$@" > "$log" 2>&1 < /dev/null &
  OWNED_PID=$!
  # In a non-interactive script the background child is not a group leader, so setsid(1) calls
  # setsid() and execs without forking: the child's pid is its pgid. Confirm before relying on it.
  for i in $(seq 1 50); do
    pg=$(ps -o pgid= -p "$OWNED_PID" 2>/dev/null | tr -d ' ') || pg=""
    [ -z "$pg" ] && break                  # already exited; the caller notices via kill -0
    [ "$pg" = "$OWNED_PID" ] && break
    sleep 0.1
  done
  if [ -n "$pg" ] && [ "$pg" != "$OWNED_PID" ]; then
    echo "start_owned: pid $OWNED_PID did not become a process-group leader (pgid $pg); stopping it" >&2
    kill -TERM "$OWNED_PID" 2>/dev/null || true
    OWNED_PID=""; return 1
  fi
  OWNED_PGID=$OWNED_PID
}

stop_owned() {
  local t=${1:-30} pg=$OWNED_PGID i
  [ -n "$pg" ] || return 0
  if kill -0 -- "-$pg" 2>/dev/null; then
    kill -TERM -- "-$pg" 2>/dev/null || true
    for ((i = 0; i < t; i++)); do kill -0 -- "-$pg" 2>/dev/null || break; sleep 1; done
    if kill -0 -- "-$pg" 2>/dev/null; then
      echo "process group $pg still running ${t}s after TERM; sending KILL to that group" >&2
      kill -KILL -- "-$pg" 2>/dev/null || true
      for ((i = 0; i < 15; i++)); do kill -0 -- "-$pg" 2>/dev/null || break; sleep 1; done
    fi
  fi
  wait "$OWNED_PID" 2>/dev/null || true
  if kill -0 -- "-$pg" 2>/dev/null; then echo "process group $pg did not exit" >&2; return 1; fi
  OWNED_PID=""; OWNED_PGID=""
}

owned_cleanup() {
  local rc=$?
  trap - EXIT
  if [ -n "$OWNED_PGID" ]; then
    echo "cleanup (exit $rc): stopping owned process group $OWNED_PGID" >&2
    stop_owned 30 || true
  fi
  exit "$rc"
}

report_gpu_leftovers() {
  local wait_s=${1:-20} i apps=""
  for ((i = 0; i <= wait_s; i += 2)); do
    apps=$(gpu_compute_apps 2>/dev/null) || apps=""
    [ -z "$apps" ] && return 0
    sleep 2
  done
  echo "WARNING: GPU compute processes remain after stopping this script's server (not killed):" >&2
  echo "$apps" >&2
}
