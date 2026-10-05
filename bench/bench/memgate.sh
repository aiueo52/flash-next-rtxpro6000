#!/bin/bash
# memgate.sh <label> -- before a server start: wait until MemAvailable >= MEMGATE_GIB (default 110) GiB, polling every
# 30 s for up to MEMGATE_MIN (default 30) minutes, else exit 1. A server pins about 67 GB of host RAM (OOM, 10-01).
need_gib=${MEMGATE_GIB:-110}; end=$(( $(date +%s) + ${MEMGATE_MIN:-30} * 60 )); waited=0
while :; do
  avail_gib=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1048576 ))
  if [ "$avail_gib" -ge "$need_gib" ]; then
    [ $waited = 1 ] && echo "[$1] MemAvailable ${avail_gib} GiB, starting $(date +%T)"
    exit 0
  fi
  [ "$(date +%s)" -ge "$end" ] &&
    { echo "[$1] MemAvailable ${avail_gib} GiB < ${need_gib} GiB for ${MEMGATE_MIN:-30} min; not starting"; exit 1; }
  [ $waited = 0 ] && echo "[$1] MemAvailable ${avail_gib} GiB < ${need_gib} GiB; waiting $(date +%T)"
  waited=1; sleep 30
done
