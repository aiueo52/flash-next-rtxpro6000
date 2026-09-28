"""Pin test v3: consumer GEMV *without* any window; the persisting touch carries the window.

T7  graph [touch(win attr)][stream190][GEMV][stream190][GEMV][stream190][GEMV]: do persisting lines
    survive a normal-access GEMV + more streaming? (=> touch once per step is enough)
T8  ONE touch(win attr) outside the graph, then graph [flush][stream190][GEMV] replayed: survive replays?
T9  graph [set stream window -> Triton touch -> clear window][stream190][GEMV]: the implementation path
    (no cubin, no cuda-python launch), consumer node without window.
T10 as T9 but the touch is my CUDA kernel with the launch attribute (reference, = bench_hit persist+stream)

  PYTHONPATH=<L2 experiment worktree, not published>/python:$PYTHONPATH SGLANG_L2_PIN_DRAFT_HEAD=1 $PY test_pin_api3.py
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import logging
import torch
logging.basicConfig(level=logging.INFO, format="%(message)s")
from common import MB, PERSISTING, STREAMING, NORMAL, Graph, L2Ctl, banner, cupti_positions, gemv_fns, make_weight, sector_range
from kernels import triton_touch
from sglang.srt.layers import l2_pin

torch.cuda.init()
ctl = L2Ctl()
gemv = gemv_fns()
w, s, x, y = make_weight(49152, 2560, 1)
wb, wn = sector_range(w)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda")
other = torch.empty(200 * MB, dtype=torch.uint8, device="cuda")
sink = torch.zeros(64 * 1024, dtype=torch.int32, device="cuda")
ctl.set_persisting(80 * MB)
WIN = (wb, wn, 1.0, PERSISTING, STREAMING)

def run(tag, call, stream=None):
    gr = Graph(ctl, call, capture_stream=stream or torch.cuda.Stream())
    nodes = [(n["name"][:16], n["win_bytes"] // MB) for n in gr.nodes() if n["type"] == 0]
    pos, _ = cupti_positions(gr, reps=30)
    gr.close()
    ts = [round(d, 2) for n, d, _ in pos if "_w8a16_gemv_kernel" in n]
    tt = [round(d, 2) for n, d, _ in pos if "touch" in n]
    print(f"  {tag:58s} gemv {ts}  touches {tt}  nodes {nodes}", flush=True)
    return ts

banner("consumer without window; touch carries the persisting window")
def t7():
    st = torch.cuda.current_stream()
    ctl.touch(st, w, grid=376, window=WIN)
    for _ in range(3):
        ctl.touch(st, other, grid=376, nbytes=190 * MB)
        gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T7 touch(win) then 3x [stream190][GEMV]", t7)

ctl.reset_persisting()
st0 = torch.cuda.Stream()
ctl.touch(st0, w, grid=376, window=WIN); torch.cuda.synchronize()
run("T8 one out-of-graph touch(win); graph [flush][stream190][GEMV]", lambda: (ctl.touch(torch.cuda.current_stream(), evict, grid=376), ctl.touch(torch.cuda.current_stream(), other, grid=376, nbytes=190 * MB), gemv(x, w, s, out=y)))

def t9():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376)
    ctl.stream_window(st, wb, wn, 1.0, PERSISTING, STREAMING)   # what l2_pin would do around the touch
    triton_touch(w, sink)
    ctl.stream_window(st, 0, 0, 0.0, NORMAL, NORMAL)
    ctl.touch(st, other, grid=376, nbytes=190 * MB)
    gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T9 [flush][stream-window around Triton touch][stream190][GEMV]", t9)

def t10():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376)
    ctl.touch(st, w, grid=376, window=WIN)
    ctl.touch(st, other, grid=376, nbytes=190 * MB)
    gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T10 [flush][CUDA touch(win attr)][stream190][GEMV] (reference)", t10)

def t11():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376)
    ctl.touch(st, w, grid=376, window=WIN)
    for _ in range(3):
        ctl.touch(st, other, grid=376, nbytes=190 * MB)
        gemv(x, w, s, out=y)
    ctl.touch(st, evict, grid=376)
    gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T11 [flush][touch(win)] 3x[stream190][GEMV] [flush][GEMV]", t11)
ctl.reset_persisting()
