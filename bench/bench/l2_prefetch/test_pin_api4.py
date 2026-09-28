"""Pin test v4: do persisting lines survive across *graph launches*?

A = graph [touch(win attr) of head]          B = graph [flush 320][stream 190][GEMV (no window)]
T12: A once, then B x 30 (CUPTI per-position medians of the GEMV over the 30 B launches, plus first/last)
T13: (A, B, B, B) x 10 -- one touch per 'step', three consumers in separate graph launches
T14: (A, B) x 30 -- touch before every consumer graph (the fallback design)
T15: B x 30 after cudaCtxResetPersistingL2Cache (control, cold)
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import statistics, torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from common import MB, PERSISTING, STREAMING, NORMAL, Graph, L2Ctl, banner, gemv_fns, make_weight, sector_range
torch.cuda.init(); ctl = L2Ctl(); gemv = gemv_fns()
w, s, x, y = make_weight(49152, 2560, 1); wb, wn = sector_range(w)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda"); other = torch.empty(200 * MB, dtype=torch.uint8, device="cuda")
ctl.set_persisting(80 * MB); WIN = (wb, wn, 1.0, PERSISTING, STREAMING)
A = Graph(ctl, lambda: ctl.touch(torch.cuda.current_stream(), w, grid=376, window=WIN))
B = Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), ctl.touch(torch.cuda.current_stream(), other, grid=376, nbytes=190 * MB), gemv(x, w, s, out=y)))

def measure(tag, seq):
    ctl.reset_persisting(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for g in seq:
            g.replay()
        torch.cuda.synchronize()
    ev = sorted([e for e in prof.events() if e.device_type == DeviceType.CUDA and "_w8a16_gemv_kernel" in e.key], key=lambda e: e.time_range.start)
    d = [round(e.device_time, 1) for e in ev]
    print(f"  {tag:34s} gemv median {statistics.median(d):6.1f}  first 4 {d[:4]}  last 2 {d[-2:]}", flush=True)

banner("persistence across graph launches (head 120 MiB, set-aside 80 MiB)")
measure("T15 B x30 (cold control)", [B] * 30)
measure("T12 A once, B x30", [A] + [B] * 30)
measure("T13 (A,B,B,B) x10", [A, B, B, B] * 10)
measure("T14 (A,B) x30", [A, B] * 30)
measure("T12b A once, B x30 (repeat)", [A] + [B] * 30)
ctl.reset_persisting(); A.close(); B.close()
