"""Pin test v7: do persisting lines survive *another CUDA context* running on the GPU?
Same as v5's T16 (one touch per emulated step, 16 consumers/step, 3.5 GB verify stream) with and
without a second process launching a tiny kernel + sync every ~1 ms (a stand-in for the desktop
compositor's context switches).
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import os, statistics, subprocess, sys, time, torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from common import MB, PERSISTING, STREAMING, Graph, L2Ctl, banner, gemv_fns, make_weight, sector_range
torch.cuda.init(); ctl = L2Ctl(); gemv = gemv_fns()
w, s, x, y = make_weight(49152, 2560, 1); wb, wn = sector_range(w)
small = torch.empty(4 * 90 * MB, dtype=torch.uint8, device="cuda"); sb, _ = sector_range(small)
big = torch.empty(3500 * MB, dtype=torch.uint8, device="cuda")
ctl.set_persisting(80 * MB); WIN = (wb, wn, 1.0, PERSISTING, STREAMING)
A = Graph(ctl, lambda: ctl.touch(torch.cuda.current_stream(), w, grid=376, window=WIN))
def d_call():
    st = torch.cuda.current_stream()
    for i in range(4):
        ctl.touch_ptr(st, sb + i * 90 * MB, 90 * MB, grid=376); gemv(x, w, s, out=y)
D4 = Graph(ctl, d_call); V = Graph(ctl, lambda: ctl.touch(torch.cuda.current_stream(), big, grid=376))

NOISE = r'''
import torch, time
torch.cuda.init(); a = torch.ones(256, device="cuda"); t0 = time.time()
while time.time() - t0 < float(__import__("sys").argv[1]):
    a += 1; torch.cuda.synchronize(); time.sleep(0.001)
'''
def measure(tag, seq, per_step, noise=False):
    ctl.reset_persisting(); torch.cuda.synchronize()
    p = None
    if noise:
        p = subprocess.Popen([sys.executable, "-c", NOISE, "20"]); time.sleep(3.0)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for g in seq:
            g.replay()
        torch.cuda.synchronize()
    if p is not None:
        p.kill(); p.wait()
    ev = sorted([e for e in prof.events() if e.device_type == DeviceType.CUDA and "_w8a16_gemv_kernel" in e.key], key=lambda e: e.time_range.start)
    d = [round(e.device_time, 1) for e in ev]
    hot = sum(1 for v in d if v < 45); print(f"  {tag:40s} median {statistics.median(d):6.1f}  hot {hot}/{len(d)}", flush=True)
    for k in range(0, len(d), per_step):
        print(f"      step {k // per_step}: {d[k:k + per_step]}", flush=True)

banner("persistence vs a second CUDA context")
measure("T16 (A, D4x4, V) x3, no noise", ([A] + [D4] * 4 + [V]) * 3, 16)
measure("T24 same, with noise process", ([A] + [D4] * 4 + [V]) * 3, 16, noise=True)
measure("T25 (A, D4x4, V) x6, with noise", ([A] + [D4] * 4 + [V]) * 6, 16, noise=True)
measure("T16b no noise again", ([A] + [D4] * 4 + [V]) * 3, 16)
ctl.reset_persisting()
for g in (A, D4, V): g.close()
