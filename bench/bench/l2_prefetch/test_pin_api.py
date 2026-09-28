"""Unit test for the fork's layers/l2_pin.py: which accesses mark lines persisting?

T1 graph [flush][stream 190MB][GEMV] on a stream carrying the window (server code path as of 1f9c256ae1)
T2 as T1, after ONE out-of-graph touch of the head under the window (does persistence survive replays?)
T3 graph [flush][ctl touch of head (inherits window)][stream][GEMV]   (= bench_hit's swin-consumer, hr 1.0)
T4 graph [flush][Triton touch of head (inherits window)][stream][GEMV] (plain ld.global from Triton)
T5 graph [flush][GEMV][stream][GEMV] on the window stream: does the GEMV's own access set the property?

  PYTHONPATH=<L2 experiment worktree, not published>/python:$PYTHONPATH SGLANG_L2_PIN_DRAFT_HEAD=1 $PY test_pin_api.py
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import logging, os, sys
import torch
logging.basicConfig(level=logging.INFO, format="%(message)s")
from common import MB, Graph, L2Ctl, banner, cupti_positions, gemv_fns, make_weight, sector_range
from kernels import triton_touch
from sglang.srt.layers import l2_pin

torch.cuda.init()
ctl = L2Ctl()
gemv = gemv_fns()
w, s, x, y = make_weight(49152, 2560, 1)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda")
other = torch.empty(200 * MB, dtype=torch.uint8, device="cuda")
sink = torch.zeros(64 * 1024, dtype=torch.int32, device="cuda")
print("l2_pin enabled:", l2_pin.enabled(), "file:", l2_pin.__file__, flush=True)

def run(tag, stream, pre=None, post_gemv=False):
    def call():
        st = torch.cuda.current_stream()
        ctl.touch(st, evict, grid=376)
        if pre == "ctl":
            ctl.touch(st, w, grid=376)
        elif pre == "triton":
            triton_touch(w, sink)
        elif pre == "gemv":
            gemv(x, w, s, out=y)
        ctl.touch(st, other, grid=376, nbytes=190 * MB)
        gemv(x, w, s, out=y)
    gr = Graph(ctl, call, capture_stream=stream)
    nodes = [(n["name"][:22], n["win_bytes"] // MB, n["hit_prop"]) for n in gr.nodes() if n["type"] == 0]
    pos, _ = cupti_positions(gr, reps=30)
    gr.close()
    ts = [round(d, 2) for n, d, _ in pos if "_w8a16_gemv_kernel" in n]
    tt = [round(d, 2) for n, d, _ in pos if "touch" in n]
    print(f"  {tag:52s} gemv {ts}  touches {tt}  nodes {nodes}", flush=True)
    return ts[-1]

banner("draft head 49152x2560 M=1 after 190 MB streaming")
s0 = torch.cuda.Stream()
t1a = run("T0 no window", s0)
assert l2_pin.pin_tensor(w, "bench draft head")
sp = torch.cuda.Stream(); l2_pin.apply_to_stream(sp)
t1 = run("T1 window on stream, GEMV only (server 1f9c256)", sp)
ctl.reset_persisting()
with torch.cuda.stream(sp):
    ctl.touch(sp, w, grid=376)        # one touch under the window, outside any graph
torch.cuda.synchronize()
t2 = run("T2 after ONE out-of-graph touch under the window", sp)
t2b = run("T2b same again (steady state, no re-touch)", sp)
ctl.reset_persisting()
t3 = run("T3 ctl touch of head inside graph (inherits window)", sp, pre="ctl")
ctl.reset_persisting()
t4 = run("T4 Triton touch of head inside graph (inherits window)", sp, pre="triton")
ctl.reset_persisting()
t5 = run("T5 GEMV twice: [GEMV][stream][GEMV]", sp, pre="gemv")
ctl.reset_persisting()
t6 = run("T6 Triton touch, fresh stream without window (control)", torch.cuda.Stream(), pre="triton")
print(f"\nRESULT: T0 {t1a} T1 {t1} T2 {t2}/{t2b} T3 {t3} T4 {t4} T5 {t5} T6 {t6}", flush=True)
