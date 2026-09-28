"""Measure the REAL expert-routing structure per MoE call on a live server.

Nothing under ``python/sglang`` is modified: this monkeypatches
``sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._run_flashinfer_cutlass``
and is injected with a ``sitecustomize.py`` on ``PYTHONPATH`` (see
``sitecustomize_example.py``).

Everything is recorded with **CUDA-graph-capturable ops**, because the verify and
draft passes are graph replays: a plain python hook fires once at capture time and
never again, whereas the ops it *issues* at capture time run on every replay.
That rules out ``torch.unique`` / ``bincount`` (dynamic shapes) and ``.item()``.

Per call, with ``T = topk_ids.shape[0]`` (known at capture time, so the buffers can
be selected in python and the arithmetic stays static):

* ``counts[e]``  = how many of the T tokens routed to expert e
  (``counts.zero_(); counts.scatter_add_(0, ids.flatten(), ones)``)
* ``overlap[T][j] += #{e : counts[e] == j}`` for j = 0..T, in one
  ``scatter_add_``.  Bucket 0 is the untouched experts, so
  ``D = E - overlap[0]`` and ``overlap[T]`` is the count of experts shared by
  **all** T tokens.  This one histogram carries the whole overlap structure.
* ``dist_hist[T][D] += 1``  -- the per-call distribution of the distinct count,
  for p50/p90.
* optional ring of raw ids + the call index, for offline layer attribution and
  for ``bench_moe.py --routing replay``.

Output: ``$SGLANG_MOE_ROUTE_LOG.json`` (+ ``.routes.jsonl`` with the ring).
Dump with ``kill -USR1 <scheduler pid>`` or let ``atexit`` fire.
Analyse with ``python -m moe_smallm.route_logger --report <file>.json``.
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


def install(num_experts: int = 512, max_rows: int = 4096):
    """Patch the fused func.  CUDA buffers are allocated lazily (see ``_ensure``)
    because this runs at import time, when CUDA may not be up yet."""
    # Same import-order requirement as the benchmark harness.  See
    # runners.ensure_sglang_imports.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fc

    if _S.get("installed"):
        return
    _S.update({
        "E": num_experts,
        "max_rows": max_rows,
        "ring_n": int(os.environ.get("SGLANG_MOE_ROUTE_RING", "0")),
        "per_t": {},          # T -> {"overlap": [T+1], "dist": [E+1], "n": [1]}
        "ready": False,
        "installed": True,
    })

    orig = fc._run_flashinfer_cutlass

    def patched(*, dispatch_output, quant_info, runner_config, output=None,
                enable_alltoall=False):
        try:
            ids = dispatch_output.topk_output.topk_ids
            if _ensure(ids.device):
                _record(ids)
        except Exception as exc:  # never break inference
            print(f"[route_logger] record failed: {exc!r}", file=sys.stderr)
        return orig(dispatch_output=dispatch_output, quant_info=quant_info,
                    runner_config=runner_config, output=output,
                    enable_alltoall=enable_alltoall)

    fc._run_flashinfer_cutlass = patched
    atexit.register(dump)
    try:
        signal.signal(signal.SIGUSR1, lambda *_: dump())
    except Exception:
        pass
    prefix = os.environ.get("SGLANG_MOE_ROUTE_LOG", "/tmp/moe_routes")
    try:
        with open(prefix + ".pids", "a") as f:
            f.write(f"{os.getpid()}\n")
    except Exception:
        pass
    # Signals are unreliable here (SGLang installs its own handlers, and the
    # launcher shell dies if a broadcast reaches it), so also dump on a timer.
    period = float(os.environ.get("SGLANG_MOE_ROUTE_AUTODUMP", "20"))
    if period > 0:
        def _loop():
            while True:
                time.sleep(period)
                try:
                    dump()
                except Exception as exc:
                    print(f"[route_logger] autodump failed: {exc!r}",
                          file=sys.stderr)
        threading.Thread(target=_loop, daemon=True, name="route-logger").start()
    print(f"[route_logger] installed (E={num_experts}, "
          f"ring={_S['ring_n']})", file=sys.stderr)


def _ensure(device) -> bool:
    """Allocate the persistent buffers on the first *non-capturing* call.

    Allocating during a CUDA graph capture would put them in the graph's private
    pool, so skip those calls; SGLang runs eager warmup forwards before it
    captures, which is where this fires.
    """
    import torch

    if _S.get("ready"):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    E, ring_n = _S["E"], _S["ring_n"]
    _S["counts"] = torch.zeros(E, dtype=torch.int32, device=device)
    _S["ones_e"] = torch.ones(E, dtype=torch.int64, device=device)
    _S["one"] = torch.ones(1, dtype=torch.int64, device=device)
    _S["counter"] = torch.zeros(1, dtype=torch.int64, device=device)
    if ring_n:
        _S["ring"] = torch.full((ring_n, _S["max_rows"]), -1, dtype=torch.int32,
                                device=device)
        _S["ring_rows"] = torch.zeros(ring_n, dtype=torch.int32, device=device)
        _S["ring_call"] = torch.full((ring_n,), -1, dtype=torch.int64, device=device)
    _S["ready"] = True
    print(f"[route_logger] buffers ready on {device}", file=sys.stderr)
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


def _record(topk_ids):
    """CUDA-graph capturable only: no .item(), no unique(), no bincount."""
    import torch

    t = int(topk_ids.shape[0])          # static at capture time
    sl = _slot(t)
    flat = topk_ids.reshape(-1).long()

    counts = _S["counts"]
    counts.zero_()
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    # experts-by-multiplicity histogram over all E entries (bucket 0 = untouched)
    sl["overlap"].scatter_add_(0, counts.long().clamp_(0, t), _S["ones_e"])
    # per-call distinct count
    d = (_S["E"] - (counts == 0).sum(dtype=torch.int64)).reshape(1)
    sl["dist"].scatter_add_(0, d, _S["one"])
    sl["n"].add_(1)

    if _S["ring_n"] and flat.numel() <= _S["max_rows"]:
        idx = _S["counter"] % _S["ring_n"]
        n = flat.numel()
        row = torch.full((1, _S["max_rows"]), -1, dtype=torch.int32,
                         device=flat.device)
        row[0, :n] = topk_ids.reshape(-1).to(torch.int32)
        _S["ring"].index_copy_(0, idx, row)
        _S["ring_rows"].index_copy_(
            0, idx, torch.full((1,), t, dtype=torch.int32, device=flat.device))
        _S["ring_call"].index_copy_(0, idx, _S["counter"])
    _S["counter"].add_(1)


def _pct(hist: list[int], q: float):
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
    if not _S.get("ready"):
        print("[route_logger] nothing recorded (buffers never allocated)",
              file=sys.stderr)
        return
    prefix = prefix or os.environ.get("SGLANG_MOE_ROUTE_LOG", "/tmp/moe_routes")
    # several processes (tokenizer / scheduler / detokenizer) may be armed; only
    # the one holding the CUDA tensors has data, so tag the file with the pid.
    prefix = f"{prefix}.{os.getpid()}"
    E = _S["E"]
    out = {"num_experts": E, "by_T": {}}
    for t, sl in sorted(_S["per_t"].items()):
        dist = sl["dist"].cpu().tolist()
        ov = sl["overlap"].cpu().tolist()
        n = int(sl["n"].cpu().item())
        if not n:
            continue
        mean_d = sum(i * c for i, c in enumerate(dist)) / n
        out["by_T"][str(t)] = {
            "calls": n,
            "mean_distinct": mean_d,
            "p50_distinct": _pct(dist, 0.5),
            "p90_distinct": _pct(dist, 0.9),
            "min_distinct": next((i for i, c in enumerate(dist) if c), None),
            "max_distinct": next((i for i in range(len(dist) - 1, -1, -1)
                                  if dist[i]), None),
            # experts touched by exactly j of the T tokens, averaged per call
            "experts_by_multiplicity": {str(j): ov[j] / n for j in range(1, t + 1)},
            "shared_by_all_T": ov[t] / n,
            "distinct_histogram": {str(i): c for i, c in enumerate(dist) if c},
        }
    with open(prefix + ".json", "w") as f:
        json.dump(out, f, indent=2)
    if _S["ring_n"]:
        ring = _S["ring"].cpu()
        rows = _S["ring_rows"].cpu().tolist()
        calls = _S["ring_call"].cpu().tolist()
        recs = []
        for i, t in enumerate(rows):
            if t <= 0 or calls[i] < 0:
                continue
            ids = ring[i]
            ids = ids[ids >= 0].tolist()
            if not ids:
                continue
            recs.append({"call": calls[i], "T": t, "top_k": len(ids) // t,
                         "topk_ids": ids})
        recs.sort(key=lambda r: r["call"])
        with open(prefix + ".routes.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    total = sum(v["calls"] for v in out["by_T"].values())
    print(f"[route_logger] dumped {total} calls to {prefix}.json", file=sys.stderr)


def report(path: str):
    with open(path) as f:
        d = json.load(f)
    for t, v in sorted(d["by_T"].items(), key=lambda kv: int(kv[0])):
        print(f"T={t:>2}  calls={v['calls']:>7}  distinct mean={v['mean_distinct']:6.2f} "
              f"p50={v['p50_distinct']} p90={v['p90_distinct']} "
              f"range=[{v['min_distinct']},{v['max_distinct']}]  "
              f"shared_by_all_T={v['shared_by_all_T']:.2f}")
        m = v["experts_by_multiplicity"]
        cells = "  ".join(f"x{j}:{m[j]:.1f}" for j in sorted(m, key=int))
        print(f"      experts by #tokens routing to them: {cells}")
    return d


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    report(ap.parse_args().report)
