#!/usr/bin/env python3
"""Read-only VRAM telemetry after readiness; terminate only our server PGID."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
pgid=int(sys.argv[1]);log=Path(sys.argv[2]);fail=Path(sys.argv[3])
low=0
with log.open('a',buffering=1) as f:
    while True:
        try:
            os.kill(pgid,0)
        except ProcessLookupError:
            break
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
        free=int(p.stdout.strip().splitlines()[0])
        f.write(f'{time.time():.3f} free_MiB={free}\n')
        low=low+1 if free<4096 else 0
        if low>=3:
            fail.write_text(f'3 consecutive post-readiness samples below 4096 MiB; last={free}\n')
            os.killpg(pgid,signal.SIGTERM)
            break
        time.sleep(5)
