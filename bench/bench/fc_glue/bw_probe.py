"""FC-glue clock probe (read-only: changes no GPU setting).

Question: the memory clock is 14001 MHz (P0) with no CUDA context and 13365 MHz (P1) while any
CUDA context exists. Does DRAM bandwidth move with it, and does the P0->P1 switch happen at
context creation or only later? Samples nvidia-smi clocks in a thread while a read-bound
reduction (torch.sum over 2 GiB bf16) and a copy run in 50 ms windows, starting immediately
after the context is created.

Run: . bench/megakernel/env.sh; flock -w 28800 ~/.gpu.lock $PY bench/fc_glue/bw_probe.py
"""
from __future__ import annotations

import json, subprocess, sys, threading, time


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [x.strip() for x in out.splitlines()[0].split(",")]


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.s, self.ev, self.t0 = [], threading.Event(), time.time()

    def run(self):
        while not self.ev.is_set():
            self.s.append((round(time.time() - self.t0, 3), *smi()))
            self.ev.wait(0.05)


def main():
    pre = smi()
    print("[before CUDA context]", pre)
    sam = Sampler(); sam.start()
    import torch
    t_ctx = time.time() - sam.t0
    torch.cuda.init(); torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    t_ready = time.time() - sam.t0
    n = 1 << 30  # 1 Gi bf16 elements = 2 GiB
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(n // 2, dtype=torch.bfloat16, device="cuda")
    acc = torch.empty((), dtype=torch.float32, device="cuda")
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    rows = []
    t_end = time.time() + 6.0
    while time.time() < t_end:
        tw = time.time() - sam.t0
        s.record(); torch.sum(x, dim=0, dtype=torch.float32, out=acc); e.record(); e.synchronize()
        rd = 2 * n / (s.elapsed_time(e) / 1e3) / 1e9
        s.record(); y.copy_(x[: n // 2]); e.record(); e.synchronize()
        cp = 2 * n / (s.elapsed_time(e) / 1e3) / 1e9  # read n/2 + write n/2 elements, 2 B each
        rows.append((round(tw, 3), round(rd, 1), round(cp, 1)))
        time.sleep(0.03)
    sam.ev.set(); sam.join()
    del x, y
    print(f"[ctx] import+init {t_ctx:.3f}..{t_ready:.3f} s after sampler start")
    print("  t_s    read_GB/s  copy_GB/s   | nearest smi sample (t, sm, mem, pstate, W)")
    for r in rows:
        near = min(sam.s, key=lambda q: abs(q[0] - r[0]))
        print(f"  {r[0]:6.3f} {r[1]:9.1f} {r[2]:9.1f}   | {near}")
    mems = sorted({q[2] for q in sam.s})
    print("[mem clocks seen]", mems)
    out = {"pre": pre, "t_ctx": t_ctx, "t_ready": t_ready, "rows": rows, "smi": sam.s}
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
