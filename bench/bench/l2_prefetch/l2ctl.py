"""ctypes wrapper over l2ctl.cu (built on demand with nvcc from the env.sh toolchain).

    ctl = L2Ctl()
    ctl.query()                          -> dict of the device's L2 / persisting limits
    ctl.set_persisting(bytes)            -> actual carve-out
    ctl.touch(stream, tensor, mode=0, grid=64, block=256, window=None, prio=None)
    ctl.touch_table(stream, table_tensor, n, ...)
    ctl.stream_window(stream, base, bytes, hit_ratio, hit_prop, miss_prop)

`stream` is a torch.cuda.Stream (or its .cuda_stream int). `window` is
(base_ptr, bytes, hit_ratio, hit_prop, miss_prop). Access properties: NORMAL=0,
STREAMING=1, PERSISTING=2. Load modes: 0 normal, 1 ld.cs (evict-first),
2 evict_last cache hint, 3 evict_first cache hint.
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "libl2ctl.so")
CU = os.path.join(HERE, "l2ctl.cu")

NORMAL, STREAMING, PERSISTING = 0, 1, 2
MODE_NORMAL, MODE_CS, MODE_EVICT_LAST, MODE_EVICT_FIRST = 0, 1, 2, 3
SECTOR = 32
GRAPH_FLAG_USE_NODE_PRIORITY = 8  # cudaGraphInstantiateFlagUseNodePriority


def _build():
    if os.path.exists(SO) and os.path.getmtime(SO) >= os.path.getmtime(CU):
        return
    nvcc = os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc")
    if not os.path.exists(nvcc):
        nvcc = "nvcc"
    cmd = [nvcc, "-O3", "-std=c++17", "-arch=sm_120", "-Xcompiler", "-fPIC", "-shared",
           "-o", SO, CU, "-lcuda"]
    subprocess.run(cmd, check=True)


def _sid(stream):
    if isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    if stream is None:
        return torch.cuda.current_stream().cuda_stream
    return int(stream)


def sector_range(t: torch.Tensor):
    """(base, bytes) of a tensor's storage span rounded out to 32-byte sectors."""
    base = t.data_ptr()
    if t.numel() == 0:
        return base, 0
    # span = last element offset + element size, over the tensor's strides
    hi = 0
    for size, stride in zip(t.shape, t.stride()):
        if size > 1:
            hi += (size - 1) * stride
    nbytes = (hi + 1) * t.element_size()
    lo = base - (base % SECTOR)
    end = base + nbytes
    end = (end + SECTOR - 1) // SECTOR * SECTOR
    return lo, end - lo


class L2Ctl:
    def __init__(self):
        _build()
        self.lib = ctypes.CDLL(SO)
        L = self.lib
        L.l2_last_error.restype = ctypes.c_char_p
        L.l2_query.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        L.l2_set_persisting.argtypes = [ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong)]
        L.l2_stream_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong,
                                       ctypes.c_float, ctypes.c_int, ctypes.c_int]
        L.l2_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_float, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.l2_graph_instantiate.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_void_p)]
        L.l2_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.l2_graph_exec_destroy.argtypes = [ctypes.c_void_p]
        L.l2_graph_num_nodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        L.l2_graph_node_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                         ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong),
                                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
        L.l2_graph_node_set_window.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                               ctypes.c_ulonglong, ctypes.c_float, ctypes.c_int, ctypes.c_int]
        L.l2_graph_node_set_priority.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.sink = torch.zeros(4, dtype=torch.int32, device="cuda")

    def _ck(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: rc={rc} {self.lib.l2_last_error().decode()}")

    def query(self) -> dict:
        out = (ctypes.c_longlong * 8)()
        self._ck(self.lib.l2_query(out), "l2_query")
        return {
            "l2_bytes": out[0], "max_persisting_bytes": out[1], "max_window_bytes": out[2],
            "sms": out[3], "prio_least": out[4], "prio_greatest": out[5],
            "persisting_limit_now": out[6], "l2_bytes_driver": out[7],
        }

    def set_persisting(self, nbytes: int) -> int:
        actual = ctypes.c_ulonglong(0)
        self._ck(self.lib.l2_set_persisting(nbytes, ctypes.byref(actual)), "l2_set_persisting")
        return actual.value

    def reset_persisting(self):
        self._ck(self.lib.l2_reset_persisting(), "l2_reset_persisting")

    def stream_window(self, stream, base=0, nbytes=0, hit_ratio=1.0, hit_prop=PERSISTING, miss_prop=STREAMING):
        self._ck(self.lib.l2_stream_window(_sid(stream), base, nbytes, hit_ratio, hit_prop, miss_prop),
                 "l2_stream_window")

    def _launch(self, stream, kind, mode, ptr, n, grid, block, window, prio):
        if window is None:
            wb, wn, hr, hp, mp = 0, 0, 0.0, 0, 0
            won = 0
        else:
            wb, wn, hr, hp, mp = window
            won = 1
        pon = 0 if prio is None else 1
        self._ck(self.lib.l2_launch(_sid(stream), kind, mode, ptr, n, grid, block, won, wb, wn, hr, hp, mp,
                                    pon, 0 if prio is None else prio, self.sink.data_ptr()), "l2_launch")

    def touch(self, stream, t: torch.Tensor, mode=0, grid=64, block=256, window=None, prio=None, nbytes=None):
        base, nb = sector_range(t)
        if nbytes is not None:
            nb = min(nb, nbytes)
        self._launch(stream, 0, mode, base, nb // SECTOR, grid, block, window, prio)
        return base, nb

    def touch_ptr(self, stream, base: int, nbytes: int, mode=0, grid=64, block=256, window=None, prio=None):
        self._launch(stream, 0, mode, base, nbytes // SECTOR, grid, block, window, prio)

    def make_table(self, tensors) -> tuple[torch.Tensor, int]:
        """Device table of (ptr, bytes) rows (int64 pairs) for touch_table."""
        rows = []
        for t in tensors:
            b, n = sector_range(t)
            if n:
                rows.append((b, n))
        tab = torch.tensor(rows, dtype=torch.int64, device="cuda")
        return tab, sum(n for _, n in rows)

    def touch_table(self, stream, table: torch.Tensor, mode=0, grid=64, block=256, window=None, prio=None):
        self._launch(stream, 1, mode, table.data_ptr(), table.shape[0], grid, block, window, prio)

    # ---- graphs ----
    def instantiate(self, graph: torch.cuda.CUDAGraph, flags=0) -> int:
        e = ctypes.c_void_p(0)
        self._ck(self.lib.l2_graph_instantiate(graph.raw_cuda_graph(), flags, ctypes.byref(e)), "instantiate")
        return e.value

    def graph_launch(self, exec_handle: int, stream=None):
        self._ck(self.lib.l2_graph_launch(exec_handle, _sid(stream)), "graph_launch")

    def exec_destroy(self, exec_handle: int):
        self._ck(self.lib.l2_graph_exec_destroy(exec_handle), "exec_destroy")

    def nodes(self, graph: torch.cuda.CUDAGraph) -> list[dict]:
        g = graph.raw_cuda_graph()
        n = ctypes.c_ulonglong(0)
        self._ck(self.lib.l2_graph_num_nodes(g, ctypes.byref(n)), "num_nodes")
        out = []
        for i in range(n.value):
            typ = ctypes.c_int(0)
            name = ctypes.create_string_buffer(512)
            win = (ctypes.c_longlong * 4)()
            hr = ctypes.c_float(0)
            prio = ctypes.c_int(0)
            self._ck(self.lib.l2_graph_node_info(g, i, ctypes.byref(typ), name, 512, win, ctypes.byref(hr),
                                                 ctypes.byref(prio)), "node_info")
            out.append({"idx": i, "type": typ.value, "name": name.value.decode(errors="replace"),
                        "win_base": win[0], "win_bytes": win[1], "hit_prop": win[2], "miss_prop": win[3],
                        "hit_ratio": hr.value, "priority": prio.value})
        return out

    def node_set_window(self, graph, idx, base, nbytes, hit_ratio=1.0, hit_prop=PERSISTING, miss_prop=STREAMING):
        self._ck(self.lib.l2_graph_node_set_window(graph.raw_cuda_graph(), idx, base, nbytes, hit_ratio,
                                                   hit_prop, miss_prop), "node_set_window")

    def node_set_priority(self, graph, idx, prio):
        self._ck(self.lib.l2_graph_node_set_priority(graph.raw_cuda_graph(), idx, prio), "node_set_priority")
