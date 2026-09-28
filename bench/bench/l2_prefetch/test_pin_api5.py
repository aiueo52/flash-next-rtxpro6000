"""Pin test v5: decay of persisting lines over an emulated W16 step.
A = touch(win) of the 120 MiB head.  D = [stream 90 MB][head GEMV] (a draft forward: layer + experts + head).
V = [stream 3.5 GB] (the verify step).  E = [stream 90 MB][GEMV][touch(win)] (draft_extend with the touch at its end).
T16: (A, D x14, V) x3          one touch per step at the start of the draft phase
T17: A, (D x14, V) x3          one touch, then decay
T18: (D x14, V, E) x3          touch inside draft_extend (after verify) -- the implementation placement
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import statistics, torch
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
D4 = Graph(ctl, d_call)   # 4 draft forwards per graph
V = Graph(ctl, lambda: ctl.touch(torch.cuda.current_stream(), big, grid=376))
E = Graph(ctl, lambda: (ctl.touch_ptr(torch.cuda.current_stream(), sb, 90 * MB, grid=376), gemv(x, w, s, out=y), ctl.touch(torch.cuda.current_stream(), w, grid=376, window=WIN)))

def measure(tag, seq, per_step):
    ctl.reset_persisting(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for g in seq:
            g.replay()
        torch.cuda.synchronize()
    ev = sorted([e for e in prof.events() if e.device_type == DeviceType.CUDA and "_w8a16_gemv_kernel" in e.key], key=lambda e: e.time_range.start)
    d = [round(e.device_time, 1) for e in ev]
    steps = [d[i:i + per_step] for i in range(0, len(d), per_step)]
    print(f"  {tag:28s} median {statistics.median(d):6.1f}", flush=True)
    for k, st in enumerate(steps):
        print(f"      step {k}: {st}", flush=True)

banner("decay over emulated steps (D4 = 4 draft forwards per graph)")
measure("T16 (A, D4x4, V) x3", ([A] + [D4] * 4 + [V]) * 3, 16)
measure("T17 A, (D4x4, V) x3", [A] + ([D4] * 4 + [V]) * 3, 16)
measure("T18 (D4x4, V, E) x3", ([D4] * 4 + [V, E]) * 3, 17)
measure("T19 (D4x4, V) x2 cold", ([D4] * 4 + [V]) * 2, 16)
ctl.reset_persisting()
for g in (A, D4, V, E): g.close()
