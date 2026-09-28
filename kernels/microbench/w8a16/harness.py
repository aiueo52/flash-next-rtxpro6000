"""CUDA-graph replay timing for the W8A16 skinny GEMV.

Every measurement rotates over enough distinct weight copies that the working set is
several times the 128 MB L2, so the timing reflects HBM traffic and not a cache hit.
"""

import torch

L2_BYTES = 128 << 20
WORKING_SET_TARGET = 4 * L2_BYTES  # 512 MB: ~4x L2
MEM_BUDGET = 6 << 30


def n_copies(weight_bytes: int, lo: int = 24, hi: int = 128) -> int:
    n = max(lo, -(-WORKING_SET_TARGET // weight_bytes))
    n = min(n, hi, max(2, MEM_BUDGET // weight_bytes))
    return int(n)


def make_copies(w, n):
    """n distinct byte-identical copies of w, preserving its stride layout."""
    out = [w]
    for _ in range(n - 1):
        c = torch.empty_strided(w.shape, w.stride(), dtype=w.dtype, device=w.device)
        c.copy_(w)
        out.append(c)
    return out


def time_graph(call, args_list, iters=50, warmup=2):
    """call(arg) once per element of args_list, captured in one graph, replayed `iters` times.

    Returns seconds per call. Raises whatever the callable raises during warmup.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for a in args_list:
                call(a)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for a in args_list:
            call(a)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    dt = start.elapsed_time(end) / 1e3 / (iters * len(args_list))
    del g
    return dt


def require_idle_gpu():
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(f"GPU busy (compute pids: {out.splitlines()}); refusing to benchmark.")
