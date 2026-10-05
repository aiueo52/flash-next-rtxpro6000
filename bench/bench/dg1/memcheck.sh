# Copy of bench/rs1/memcheck.sh (2026-10-01), sourced by arm_dg.sh before a server starts.
# A Flash-Next server pins ~67 GB of host RAM
# (--ple-offload-embedding, cudaHostAlloc: shows as shmem, never reclaimable) for its whole life;
# with 8 GB swap, starting one next to an uncapped compile froze the machine on 2026-10-01.
need_gb=${MIN_AVAIL_GB:-110}
for _i in $(seq 1 60); do
  avail_gb=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail_gb" -ge "$need_gb" ] && break
  echo "[$LABEL] waiting for memory: MemAvailable ${avail_gb}G < ${need_gb}G $(date +%T)"; sleep 30
done
if [ "$avail_gb" -lt "$need_gb" ]; then
  echo "[$LABEL] abort: MemAvailable ${avail_gb}G < ${need_gb}G after 30 min"; exit 1
fi
echo "[$LABEL] MemAvailable ${avail_gb}G ok"
