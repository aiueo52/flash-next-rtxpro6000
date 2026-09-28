"""Tune the fork's Triton touch-table kernel (BLOCK x num_warps x grid) on the 55 MB layer table."""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import itertools, torch
from common import MB, Graph, L2Ctl, banner, cupti_positions, make_weight, short
from sglang.srt.layers import l2_prefetch as P
torch.cuda.init(); ctl = L2Ctl()
w1, *_ = make_weight(16384, 2560, 4); w2, *_ = make_weight(2560, 6144, 4)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda")
pf = P.LayerPrefetcher(torch.device("cuda"))
class F: pass
lin = F(); lin.in_proj_qkvz = F(); lin.in_proj_qkvz.weight = w1.t(); lin.out_proj = F(); lin.out_proj.weight = w2.t()
layer = F(); layer.linear_attn = lin
tab, n, total = pf.table_for(layer)
sink = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
banner(f"touch-table kernel over {total/MB:.0f} MB, cold (after 320 MB flush)")
best = []
for block, warps, grid, unroll in itertools.product((512, 1024, 2048), (4, 8), (32, 64), (1, 2, 4, 8)):
    def call():
        ctl.touch(torch.cuda.current_stream(), evict, grid=376)
        P._touch_table_kernel[(grid,)](tab, n, sink, BLOCK=block, UNROLL=unroll, num_warps=warps)
    try:
        gr = Graph(ctl, call)
    except Exception as e:
        print(f"  BLOCK {block} warps {warps} grid {grid} unroll {unroll}: {str(e)[:60]}"); continue
    pos, _ = cupti_positions(gr, reps=20); gr.close()
    t = [d for nm, d, _ in pos if "touch_table" in nm][0]
    best.append((t, block, warps, grid, unroll))
    print(f"  BLOCK {block:5d} warps {warps} grid {grid:3d} unroll {unroll}: {t:7.2f} us = {total/t/1e3:5.0f} GB/s", flush=True)
best.sort(); print("best:", best[:6])
