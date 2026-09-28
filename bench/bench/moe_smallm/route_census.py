"""P1 route census: record every (row, expert) route with its weight.

``route_logger`` records only the *distinct-expert* structure (counts by
multiplicity).  The singleton-pruning question needs, per verify call, each
route's normalised routing weight and its rank inside the row, so that the
expected D reduction for a weight threshold tau can be computed offline.

Same constraints as ``route_logger``: verify and draft passes are CUDA-graph
replays, so everything here must be a graph-capturable op issued at capture
time (no ``.item()``, no ``unique``/``bincount``, no dynamic shapes).

Recorded per call, into a circular ring of the last ``SGLANG_MOE_CENSUS_RING``
calls (default 16384 ~ 256 decode steps at 64 MoE calls/step):

* ``ids``     -- int32 [max_rows], the flattened ``topk_ids`` (-1 padded)
* ``w``       -- float32 [max_rows], the matching ``topk_weights``
* ``T``, ``k``, ``call``

Everything else (multiplicity, singleton-ness, rank, tau sweeps, removed
routing mass) is derived offline by ``analyze_census.py`` -- keeping the
in-server work to two ``index_copy_``s per call.

The cumulative distinct-count histogram of ``route_logger`` is kept as well so
a census run can be cross-checked against section 8.2 of MOE_SMALLM_SPEC.md.

Arm with ``sitecustomize_census.py`` on PYTHONPATH and
``SGLANG_MOE_CENSUS_LOG=<prefix>``.  Dump with SIGUSR1, atexit, or the
autodump timer.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import threading
import time

_S: dict = {}


def install(num_experts: int = 512, max_rows: int = 256):
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fc

    if _S.get("installed"):
        return
    _S.update({
        "E": num_experts,
        "max_rows": int(os.environ.get("SGLANG_MOE_CENSUS_MAXROWS", max_rows)),
        "ring_n": int(os.environ.get("SGLANG_MOE_CENSUS_RING", "16384")),
        "per_t": {},
        "ready": False,
        "installed": True,
    })

    orig = fc._run_flashinfer_cutlass

    def patched(*, dispatch_output, quant_info, runner_config, output=None,
                enable_alltoall=False):
        try:
            tko = dispatch_output.topk_output
            if _ensure(tko.topk_ids.device):
                _record(tko.topk_ids, tko.topk_weights)
        except Exception as exc:  # never break inference
            print(f"[route_census] record failed: {exc!r}", file=sys.stderr)
        return orig(dispatch_output=dispatch_output, quant_info=quant_info,
                    runner_config=runner_config, output=output,
                    enable_alltoall=enable_alltoall)

    fc._run_flashinfer_cutlass = patched
    atexit.register(dump)
    try:
        signal.signal(signal.SIGUSR1, lambda *_: dump())
    except Exception:
        pass
    prefix = os.environ.get("SGLANG_MOE_CENSUS_LOG", "/tmp/moe_census")
    try:
        with open(prefix + ".pids", "a") as f:
            f.write(f"{os.getpid()}\n")
    except Exception:
        pass
    period = float(os.environ.get("SGLANG_MOE_CENSUS_AUTODUMP", "20"))
    if period > 0:
        def _loop():
            while True:
                time.sleep(period)
                try:
                    dump()
                except Exception as exc:
                    print(f"[route_census] autodump failed: {exc!r}",
                          file=sys.stderr)
        threading.Thread(target=_loop, daemon=True, name="route-census").start()
    print(f"[route_census] installed (E={num_experts}, ring={_S['ring_n']}, "
          f"max_rows={_S['max_rows']})", file=sys.stderr)


def _ensure(device) -> bool:
    import torch

    if _S.get("ready"):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    E, n, mr = _S["E"], _S["ring_n"], _S["max_rows"]
    _S["counts"] = torch.zeros(E, dtype=torch.int32, device=device)
    _S["ones_e"] = torch.ones(E, dtype=torch.int64, device=device)
    _S["one"] = torch.ones(1, dtype=torch.int64, device=device)
    _S["counter"] = torch.zeros(1, dtype=torch.int64, device=device)
    _S["ring_ids"] = torch.full((n, mr), -1, dtype=torch.int32, device=device)
    _S["ring_w"] = torch.zeros((n, mr), dtype=torch.float32, device=device)
    _S["ring_T"] = torch.zeros(n, dtype=torch.int32, device=device)
    _S["ring_k"] = torch.zeros(n, dtype=torch.int32, device=device)
    _S["ring_call"] = torch.full((n,), -1, dtype=torch.int64, device=device)
    _S["ready"] = True
    print(f"[route_census] buffers ready on {device}", file=sys.stderr)
    return True


def _slot(t: int):
    import torch

    if t not in _S["per_t"]:
        dev = _S["counts"].device
        _S["per_t"][t] = {
            "overlap": torch.zeros(t + 1, dtype=torch.int64, device=dev),
            "dist": torch.zeros(_S["E"] + 1, dtype=torch.int64, device=dev),
            "n": torch.zeros(1, dtype=torch.int64, device=dev),
        }
    return _S["per_t"][t]


def _record(topk_ids, topk_weights):
    import torch

    t = int(topk_ids.shape[0])          # static at capture time
    k = int(topk_ids.shape[1])
    sl = _slot(t)
    flat = topk_ids.reshape(-1).long()

    counts = _S["counts"]
    counts.zero_()
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    sl["overlap"].scatter_add_(0, counts.long().clamp_(0, t), _S["ones_e"])
    d = (_S["E"] - (counts == 0).sum(dtype=torch.int64)).reshape(1)
    sl["dist"].scatter_add_(0, d, _S["one"])
    sl["n"].add_(1)

    n = t * k
    if n <= _S["max_rows"]:
        idx = _S["counter"] % _S["ring_n"]
        mr = _S["max_rows"]
        dev = flat.device
        row_i = torch.full((1, mr), -1, dtype=torch.int32, device=dev)
        row_i[0, :n] = topk_ids.reshape(-1).to(torch.int32)
        row_w = torch.zeros((1, mr), dtype=torch.float32, device=dev)
        row_w[0, :n] = topk_weights.reshape(-1).to(torch.float32)
        _S["ring_ids"].index_copy_(0, idx, row_i)
        _S["ring_w"].index_copy_(0, idx, row_w)
        _S["ring_T"].index_copy_(
            0, idx, torch.full((1,), t, dtype=torch.int32, device=dev))
        _S["ring_k"].index_copy_(
            0, idx, torch.full((1,), k, dtype=torch.int32, device=dev))
        _S["ring_call"].index_copy_(0, idx, _S["counter"])
    _S["counter"].add_(1)


def _pct(hist, q):
    tot = sum(hist)
    if not tot:
        return None
    cum = 0
    for i, c in enumerate(hist):
        cum += c
        if cum >= q * tot:
            return i
    return len(hist) - 1


def dump(prefix: str | None = None):
    """Atomic (tmp + rename) so a copy taken between workloads is never torn."""
    import numpy as np

    if not _S.get("ready"):
        print("[route_census] nothing recorded", file=sys.stderr)
        return
    prefix = prefix or os.environ.get("SGLANG_MOE_CENSUS_LOG", "/tmp/moe_census")
    prefix = f"{prefix}.{os.getpid()}"
    E = _S["E"]
    out = {"num_experts": E, "by_T": {}}
    for t, sl in sorted(_S["per_t"].items()):
        dist = sl["dist"].cpu().tolist()
        ov = sl["overlap"].cpu().tolist()
        n = int(sl["n"].cpu().item())
        if not n:
            continue
        out["by_T"][str(t)] = {
            "calls": n,
            "mean_distinct": sum(i * c for i, c in enumerate(dist)) / n,
            "p50_distinct": _pct(dist, 0.5),
            "p90_distinct": _pct(dist, 0.9),
            "experts_by_multiplicity": {str(j): ov[j] / n for j in range(1, t + 1)},
            "distinct_histogram": {str(i): c for i, c in enumerate(dist) if c},
        }
    tmp = prefix + ".json.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, prefix + ".json")

    call = _S["ring_call"].cpu().numpy()
    keep = call >= 0
    order = np.argsort(call[keep], kind="stable")
    sel = np.nonzero(keep)[0][order]
    np.savez(
        prefix + ".npz.tmp.npz",
        call=call[keep][order].astype(np.int64),
        T=_S["ring_T"].cpu().numpy()[sel].astype(np.int16),
        k=_S["ring_k"].cpu().numpy()[sel].astype(np.int16),
        ids=_S["ring_ids"].cpu().numpy()[sel].astype(np.int16),
        w=_S["ring_w"].cpu().numpy()[sel].astype(np.float32),
    )
    os.replace(prefix + ".npz.tmp.npz", prefix + ".npz")
    total = sum(v["calls"] for v in out["by_T"].values())
    print(f"[route_census] dumped {total} calls ({len(sel)} in ring) to "
          f"{prefix}.{{json,npz}}", file=sys.stderr)
