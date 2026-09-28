"""K2 experiment 1: what a dependent stage boundary costs.

Three primitives, all timed under CUDA-graph replay, all reported as the *slope*
of total time against the number of stages, so every fixed cost (graph replay
entry, kernel ramp, the counter reset) cancels out:

  launch   N chained trivial Triton kernels in one graph. Each stage reads the
           previous stage's output, so the chain is strictly dependent -- exactly
           the situation of the model's glue chain. Slope = cost of adding one
           dependent kernel to a graph.
  barrier  ONE persistent Triton kernel that performs N grid-wide barriers.
           Slope = cost of adding one dependent stage *inside* a kernel.
           Three barrier implementations x four grid sizes.
  cta_flag the same kernel, but stage k+1 waits only on ONE neighbouring CTA's
           stage-k flag (fine-grained per-tile dependency counters). Slope = cost
           of a stage boundary whose dataflow is CTA-local rather than grid-wide.

Every stage writes 4 B per lane, so a stage's own memory traffic is ~0 and what is
measured is the boundary itself.

Run:  flock -w 14400 ~/.gpu.lock python bench/megakernel/bench_barrier.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ab_rounds, emit, gpu_note  # noqa: E402

_FLAG_BASE = 4097  # counters live in [0, 4097); per-CTA flags after it


# --------------------------------------------------------------------------- #
# launch: N dependent trivial kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _stage_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, v + 1.0)


@triton.jit
def _stage_kernel_pdl(in_ptr, out_ptr, BLOCK: tl.constexpr):
    """Same stage under Programmatic Dependent Launch (what K1 is adding).

    The address arithmetic is the independent prologue; `gdc_wait` gates only the
    dependent load, and `gdc_launch_dependents` releases the successor as soon as
    this stage's store is visible.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    gdc_wait()
    v = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, v + 1.0)
    gdc_launch_dependents()


def launch_chain(n_stages: int, grid: int, block: int, bufs, pdl: bool = False):
    def call(_):
        for i in range(n_stages):
            if pdl:
                _stage_kernel_pdl[(grid,)](
                    bufs[i % 2], bufs[(i + 1) % 2], BLOCK=block, num_warps=4,
                    launch_pdl=True,
                )
            else:
                _stage_kernel[(grid,)](
                    bufs[i % 2], bufs[(i + 1) % 2], BLOCK=block, num_warps=4
                )

    return call


# --------------------------------------------------------------------------- #
# barrier: one persistent kernel with N stage boundaries
# --------------------------------------------------------------------------- #
@triton.jit
def _barrier_kernel(
    cnt_ptr,
    out_ptr,
    num_ctas,
    N: tl.constexpr,
    MODE: tl.constexpr,  # 0 rmw-poll, 1 load-poll, 2 arrive-only, 3 cta-flag
    BLOCK: tl.constexpr,
    FLAG_BASE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # the CTA this one consumes from: a real cross-CTA dependency, so a stage here
    # costs the same L2 round trip that a stage of the launch chain costs.
    dep = (pid + 1) % num_ctas
    dep_offs = dep * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(out_ptr + offs)

    for i in tl.static_range(N):
        # one trivial "stage": publish this CTA's slice ...
        acc = acc + 1.0
        tl.store(out_ptr + offs, acc, cache_modifier=".cg")
        tl.debug_barrier()
        if MODE == 0:
            tl.atomic_add(cnt_ptr + i, 1, sem="acq_rel", scope="gpu")
            while tl.atomic_add(cnt_ptr + i, 0, sem="acq_rel", scope="gpu") < num_ctas:
                pass
        elif MODE == 1:
            tl.atomic_add(cnt_ptr + i, 1, sem="release", scope="gpu")
            while tl.load(cnt_ptr + i, volatile=True) < num_ctas:
                pass
        elif MODE == 2:  # arrive, never wait -- lower bound on the atomic alone
            tl.atomic_add(cnt_ptr + i, 1, sem="release", scope="gpu")
        else:  # MODE == 3: publish own flag, wait on exactly one producer
            tl.atomic_xchg(cnt_ptr + FLAG_BASE + pid, i + 1, sem="release", scope="gpu")
            while tl.load(cnt_ptr + FLAG_BASE + dep, volatile=True) < i + 1:
                pass
        tl.debug_barrier()
        # ... and consume the producer's, which is what makes this a dependent stage
        acc = tl.load(out_ptr + dep_offs, cache_modifier=".cg")

    # (no trailing store: the in-loop store already keeps the whole dependency
    # chain live, and a post-loop store would race the last stage's loads)

    # Restore the counters so the next graph replay starts clean. This MUST be a
    # "last CTA out" ticket, not a barrier + reset by CTA 0: with the latter, CTA 0
    # can zero the counter while another CTA is still polling it, and that CTA then
    # spins forever. (The ticket is safe because ticket == num_ctas - 1 means every
    # other CTA has already left its last stage.)
    ticket = tl.atomic_add(cnt_ptr + 4096, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        reset = tl.arange(0, 128)
        zero = tl.zeros((128,), dtype=tl.int32)
        for r0 in range(0, N, 128):
            idx = r0 + reset
            tl.store(cnt_ptr + idx, zero, mask=idx < N)
        for r0 in range(0, num_ctas, 128):
            idx = r0 + reset
            tl.store(cnt_ptr + FLAG_BASE + idx, zero, mask=idx < num_ctas)
        tl.store(cnt_ptr + 4096, 0)


def barrier_call(n_barriers: int, grid: int, mode: int, block: int, cnt, out):
    def call(_):
        _barrier_kernel[(grid,)](
            cnt,
            out,
            grid,
            N=n_barriers,
            MODE=mode,
            BLOCK=block,
            FLAG_BASE=_FLAG_BASE,
            num_warps=4,
        )

    return call


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return a, my - a * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--min-seconds", type=float, default=0.3)
    ap.add_argument("--repeat", type=int, default=32,
                    help="constructs per captured graph; a graph holding one 2 us "
                         "kernel is host-launch-bound, not GPU-bound")
    ap.add_argument("--out", default="bench/megakernel/results/barrier.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    block = 256

    note = gpu_note()
    print(f"[gpu] {note}")
    rep = [None] * args.repeat
    payload = {"note": note, "sms": sms, "block": block,
               "repeat": args.repeat, "results": {}}

    # ---------------- launch chain ----------------
    stage_counts = [1, 2, 4, 8, 16]
    for grid in (1, sms):
        bufs = [
            torch.zeros(grid * block, dtype=torch.float32, device=dev) for _ in range(2)
        ]
        for pdl in (False, True):
            tag = "pdl" if pdl else "plain"
            print(f"[run] launch grid={grid} {tag} ...", flush=True)
            variants = {
                f"n{n}": (launch_chain(n, grid, block, bufs, pdl), rep)
                for n in stage_counts
            }
            try:
                res = ab_rounds(
                    variants, rounds=args.rounds, min_seconds=args.min_seconds
                )
            except Exception as exc:
                print(f"[launch] grid={grid} {tag}: FAILED {exc}")
                continue
            ys = [res[f"n{n}"]["median"] for n in stage_counts]
            a, b = slope(stage_counts, ys)
            payload["results"][f"launch_{tag}_grid{grid}"] = {
                "per_stage_us": round(a, 4),
                "intercept_us": round(b, 3),
                "raw": res,
            }
            print(
                f"[launch] grid={grid:4d} {tag:5s}: {a:.3f} us per dependent kernel "
                f"(b={b:.2f})  raw={[round(y, 2) for y in ys]}"
            )

    # ---------------- stage boundaries inside one kernel ----------------
    cnt = torch.zeros(_FLAG_BASE + sms, dtype=torch.int32, device=dev)
    out = torch.zeros(sms * block, dtype=torch.float32, device=dev)
    modes = {0: "rmw_poll", 1: "load_poll", 2: "arrive_only", 3: "cta_flag"}
    barrier_counts = [1, 2, 4, 8, 16]
    for grid in (8, 94, sms):
        for mode, mname in modes.items():
            print(f"[run] barrier grid={grid} mode={mname} ...", flush=True)
            variants = {
                f"n{n}": (barrier_call(n, grid, mode, block, cnt, out), rep)
                for n in barrier_counts
            }
            try:
                res = ab_rounds(
                    variants, rounds=args.rounds, min_seconds=args.min_seconds
                )
            except Exception as exc:  # pragma: no cover
                print(f"[barrier] grid={grid} {mname}: FAILED {exc}")
                continue
            ys = [res[f"n{n}"]["median"] for n in barrier_counts]
            a, b = slope(barrier_counts, ys)
            payload["results"][f"barrier_{mname}_grid{grid}"] = {
                "per_barrier_us": round(a, 4),
                "intercept_us": round(b, 3),
                "raw": res,
            }
            print(
                f"[barrier] grid={grid:4d} {mname:12s}: {a:.4f} us/stage "
                f"(kernel floor {b:.2f} us)  raw={[round(y, 2) for y in ys]}"
            )

    emit(args.out, payload)


if __name__ == "__main__":
    main()
