"""Pin test v6: the exact server path (l2_pin.pin_tensor + l2_pin.touch_now inside a graph capture,
cuda-python stream attribute set/cleared around the Triton touch), consumer without window.
T20 graph [flush][stream190][touch_now()][stream190][GEMV]      -> expect ~28
T21 graph [flush][stream190][ctl touch(win attr)][stream190][GEMV] (reference)
T22 as T20 but window set via ctl.stream_window (my .so) around l2_prefetch.touch_table
T23 graph [flush][touch_now()][stream 190 x3 + GEMV x3] : persistence within the graph
  PYTHONPATH=<L2 experiment worktree, not published>/python:$PYTHONPATH SGLANG_L2_PIN_DRAFT_HEAD=1 $PY test_pin_api6.py
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import logging, torch
logging.basicConfig(level=logging.INFO, format="%(message)s")
from common import MB, PERSISTING, STREAMING, NORMAL, Graph, L2Ctl, banner, cupti_positions, gemv_fns, make_weight, sector_range
from sglang.srt.layers import l2_pin, l2_prefetch
torch.cuda.init(); ctl = L2Ctl(); gemv = gemv_fns()
w, s, x, y = make_weight(49152, 2560, 1); wb, wn = sector_range(w)
evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda"); other = torch.empty(200 * MB, dtype=torch.uint8, device="cuda")
assert l2_pin.pin_tensor(w, "bench head")
print("limit now:", ctl.query()["persisting_limit_now"] // MB, "MiB; window", l2_pin.window()[1] // MB, flush=True)
WIN = (wb, wn, 1.0, PERSISTING, STREAMING)

def run(tag, call):
    gr = Graph(ctl, call)
    nodes = [(n["name"][:14], n["win_bytes"] // MB, n["hit_prop"]) for n in gr.nodes() if n["type"] == 0]
    pos, _ = cupti_positions(gr, reps=30); gr.close()
    ts = [round(d, 2) for n, d, _ in pos if "_w8a16_gemv_kernel" in n]; tt = [round(d, 2) for n, d, _ in pos if "touch" in n]
    print(f"  {tag:48s} gemv {ts}  touches {tt}  nodes {nodes}", flush=True)

def t20():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376); ctl.touch(st, other, grid=376, nbytes=190 * MB)
    l2_pin.touch_now()
    ctl.touch(st, other, grid=376, nbytes=190 * MB); gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T20 server path: l2_pin.touch_now() in capture", t20)
def t21():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376); ctl.touch(st, other, grid=376, nbytes=190 * MB)
    ctl.touch(st, w, grid=376, window=WIN)
    ctl.touch(st, other, grid=376, nbytes=190 * MB); gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T21 reference: CUDA touch with launch-attr window", t21)
def t22():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376); ctl.touch(st, other, grid=376, nbytes=190 * MB)
    ctl.stream_window(st, wb, wn, 1.0, PERSISTING, STREAMING)
    l2_prefetch.touch_table(l2_pin._table[0], l2_pin._table[1], l2_pin._sink)
    ctl.stream_window(st, 0, 0, 0.0, NORMAL, NORMAL)
    ctl.touch(st, other, grid=376, nbytes=190 * MB); gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T22 .so stream window around l2_prefetch touch", t22)
def t23():
    st = torch.cuda.current_stream()
    ctl.touch(st, evict, grid=376); l2_pin.touch_now()
    for _ in range(3):
        ctl.touch(st, other, grid=376, nbytes=190 * MB); gemv(x, w, s, out=y)
ctl.reset_persisting(); run("T23 touch_now then 3x[stream][GEMV]", t23)
ctl.reset_persisting()
