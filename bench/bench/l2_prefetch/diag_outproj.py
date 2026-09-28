"""Why does the out_proj GEMV (2560x6144, split-K 6) time 13 / 19 / 28 / 50 us in four contexts?"""
import os, sys, statistics
import torch
from common import MB, MODE_NORMAL, Graph, L2Ctl, banner, cupti_positions, gemv_fns, make_weight, sector_range, short
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "megakernel"))
from harness import cupti_by_kernel, make_copies, n_copies
from kernels import tiny

torch.cuda.init()
ctl = L2Ctl()
gemv = gemv_fns()
from sglang.srt.layers.quantization import w8a16_gemv as G
N, K, M = 2560, 6144, 4
w, s, x, y = make_weight(N, K, M)
print("plan:", G._plan(M, N, K, False, 1, G._num_sms(x.device)), "PDL", G.PDL, flush=True)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda")
buf = torch.zeros(20 * 256, device="cuda")

def show(tag, gr):
    pos, _ = cupti_positions(gr, reps=30)
    print(f"  {tag:40s} " + "  ".join(f"{short(n)}={t:.2f}" for n, t, _ in pos), flush=True)
    gr.close()

banner("A: megakernel-harness style, rotating copies, no evict kernel")
copies = make_copies(w, n_copies(w.numel()))
by, _ = cupti_by_kernel(lambda c: gemv(x, c, s, out=y), copies)
print("  ", {k[:30]: v for k, v in by.items()}, flush=True)
banner("B: my Graph, gemv alone (steady state = hot)")
show("gemv alone", Graph(ctl, lambda: gemv(x, w, s, out=y)))
banner("C: evict(touch 320MB) -> gemv")
show("evict->gemv", Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), gemv(x, w, s, out=y))))
banner("D: evict -> touch w -> gemv (hot)")
show("evict->touch->gemv", Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), ctl.touch(torch.cuda.current_stream(), w, grid=376), gemv(x, w, s, out=y))))
banner("E: evict via torch (evict.sum()) -> gemv")
show("sum->gemv", Graph(ctl, lambda: (evict.view(torch.int32).sum(), gemv(x, w, s, out=y))))
banner("F: tiny x5 -> gemv")
show("tiny->gemv", Graph(ctl, lambda: ([tiny(buf, 20) for _ in range(5)], gemv(x, w, s, out=y))))
banner("G: evict -> gemv qkvz -> tiny x5 -> gemv out_proj")
w1, s1, x1, y1 = make_weight(16384, 2560, M)
show("qkvz->tiny->out", Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), gemv(x1, w1, s1, out=y1), [tiny(buf, 20) for _ in range(5)], gemv(x, w, s, out=y))))
banner("H: counters after a run")
ws, cnt = G._workspace(x.device)
torch.cuda.synchronize()
print("  cnt nonzero:", int((cnt != 0).sum()), flush=True)
banner("I: evict -> gemv with evict grid 188 / 64")
for g in (188, 64):
    show(f"evict(grid {g})->gemv", Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=g), gemv(x, w, s, out=y))))
banner("J: evict -> gemv, split-K 1 tile (32,128,1)")
show("evict->gemv splits=1", Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), gemv(x, w, s, cfg=(32, 128, 1, True, None, 4, 3), out=y))))
show("evict->gemv (64,256,4,8w)", Graph(ctl, lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), gemv(x, w, s, cfg=(64, 256, 4, True, None, 8, 3), out=y))))
