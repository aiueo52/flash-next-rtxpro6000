"""Unit test for the fork's layers/l2_prefetch.py Triton touch-table kernel: does it compile,
does it touch every sector (GEMV hot afterwards), and how fast is it alone / under a tiny chain.

  PYTHONPATH=<L2 experiment worktree, not published>/python:$PYTHONPATH SGLANG_L2_PREFETCH=1 $PY test_prefetch_kernel.py
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import logging
import torch
logging.basicConfig(level=logging.INFO, format="%(message)s")
from common import MB, Graph, L2Ctl, banner, cupti_positions, gemv_fns, make_weight, sector_range, short, time_replay
from kernels import tiny
from sglang.srt.layers import l2_prefetch as P

torch.cuda.init()
ctl = L2Ctl()
gemv = gemv_fns()
w1, s1, x1, y1 = make_weight(16384, 2560, 4)
w2, s2, x2, y2 = make_weight(2560, 6144, 4)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda")
buf = torch.zeros(20 * 256, device="cuda")
pf = P.LayerPrefetcher(torch.device("cuda"))

class FakeLayer:  # mimics the attribute paths dense_tensors() walks
    pass
lin = FakeLayer(); lin.in_proj_qkvz = FakeLayer(); lin.in_proj_qkvz.weight = w1.t(); lin.out_proj = FakeLayer(); lin.out_proj.weight = w2.t()
layer = FakeLayer(); layer.linear_attn = lin
tab, n, total = pf.table_for(layer)
print("table:", n, "ranges", total / MB, "MB", tab.tolist(), flush=True)
print("spans:", sector_range(w1), sector_range(w2), flush=True)

def run(tag, call):
    gr = Graph(ctl, call)
    t = time_replay(gr)
    pos, _ = cupti_positions(gr, reps=30)
    gr.close()
    print(f"  {tag:44s} graph {t:8.2f} us   " + "  ".join(f"{short(nm)}={d:.2f}" for nm, d, _ in pos if 'tiny' not in nm), flush=True)

banner("prefetch kernel")
run("cold GEMVs (control)", lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), gemv(x1, w1, s1, out=y1), gemv(x2, w2, s2, out=y2)))
def pre_then_gemv():
    ctl.touch(torch.cuda.current_stream(), evict, grid=376)
    P._touch_table_kernel[(P._GRID,)](tab, n, pf.sink, BLOCK=P._BLOCK, num_warps=8)
    gemv(x1, w1, s1, out=y1); gemv(x2, w2, s2, out=y2)
run("table prefetch (main stream) then GEMVs", pre_then_gemv)
def forked():
    ctl.touch(torch.cuda.current_stream(), evict, grid=376)
    pf.issue(layer)
    for _ in range(40):
        tiny(buf, 20)
    pf.join()
    gemv(x1, w1, s1, out=y1); gemv(x2, w2, s2, out=y2)
run("fork: prefetch || 40 tiny, join, GEMVs", forked)
def chain_only():
    ctl.touch(torch.cuda.current_stream(), evict, grid=376)
    for _ in range(40):
        tiny(buf, 20)
    gemv(x1, w1, s1, out=y1); gemv(x2, w2, s2, out=y2)
run("no prefetch: 40 tiny, GEMVs (control)", chain_only)
