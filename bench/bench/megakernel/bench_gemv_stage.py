"""K2 experiment 2: does a W8A16 GEMV keep its bandwidth as a *stage* of a
persistent kernel?

A glue island can only absorb the fork's own Triton GEMVs if a GEMV run under a
fixed 188-CTA persistent grid streams weights as fast as it does under its
natural grid (512 CTAs at M=4, 256 at M=16 for the GDN qkvz projection). This
bench isolates exactly that: **one** Triton kernel, three launch shapes.

  native      grid = n_blocks, num_ctas = n_blocks -> the persistent loop runs
              once per CTA, i.e. it is the ordinary one-tile-per-CTA kernel.
  persistent  grid = 188, num_ctas = 188 -> each CTA walks n_blocks/188 tiles.
  island      grid = 188, plus a trivial predecessor stage and a grid barrier
              in front of the same loop -- the shape a fused island would have.

The machine code of the GEMV body is identical in all three; only the grid and
the trip count of the tile loop change. A `fork` row calls the production
`w8a16_gemv` at the same shape so the bench kernel can be checked against the
tuned one.

Shapes are the real ones (Qwen3.8-Flash-Next, hidden 2560, 16 key heads x 128,
48 value heads x 128):
  qkvz     N = 16384, K = 2560   (30.0 us in the W4 verify trace)
  out_proj N =  2560, K = 6144   (13.4 us)
  gate_up  N =  1280, K = 2560   (shared expert)

Run:  flock -w 14400 ~/.gpu.lock python bench/megakernel/bench_gemv_stage.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ab_rounds, emit, gpu_note, make_copies, n_copies  # noqa: E402

SHAPES = {
    "qkvz": (16384, 2560),
    "out_proj": (2560, 6144),
    "gate_up": (1280, 2560),
}


@triton.jit
def _gemv_stage_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    cnt_ptr,
    pre_ptr,
    M,
    N,
    K,
    n_blocks,
    num_ctas,
    M_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRESTAGE: tl.constexpr,
):
    pid = tl.program_id(0)

    if PRESTAGE:
        # A trivial predecessor stage plus a grid barrier: the island shape.
        p = tl.load(pre_ptr + pid * 32 + tl.arange(0, 32))
        tl.store(pre_ptr + pid * 32 + tl.arange(0, 32), p + 1.0, cache_modifier=".cg")
        tl.debug_barrier()
        tl.atomic_add(cnt_ptr, 1, sem="release", scope="gpu")
        while tl.load(cnt_ptr, volatile=True) < num_ctas:
            pass
        tl.debug_barrier()

    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)

    for nb in range(pid, n_blocks, num_ctas):
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            xt = tl.load(
                x_ptr + offs_m[:, None] * K + k[None, :], mask=m_mask[:, None], other=0.0
            )
            wt = tl.load(
                w_ptr + offs_n[:, None] * K + k[None, :], mask=n_mask[:, None], other=0.0
            )
            acc = tl.dot(xt, tl.trans(wt.to(tl.bfloat16)), acc)
        acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    if PRESTAGE:
        # Restore the barrier counter for the next replay with a "last CTA out"
        # ticket. A barrier followed by a reset from CTA 0 deadlocks: CTA 0 can zero
        # the counter while another CTA is still polling it.
        ticket = tl.atomic_add(cnt_ptr + 1, 1, sem="acq_rel", scope="gpu")
        if ticket == num_ctas - 1:
            tl.store(cnt_ptr, 0)
            tl.store(cnt_ptr + 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--min-seconds", type=float, default=0.3)
    ap.add_argument("--rows", default="4,16")
    ap.add_argument("--shapes", default="qkvz,out_proj,gate_up")
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=128)
    ap.add_argument("--out", default="bench/megakernel/results/gemv_stage.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    note = gpu_note()
    print(f"[gpu] {note}")
    payload = {"note": note, "sms": sms, "results": {}}

    from sglang.srt.layers.quantization.w8a16_gemv import prealloc, w8a16_gemv

    prealloc(dev)

    for shape in args.shapes.split(","):
        N, K = SHAPES[shape]
        w_bytes = N * K
        ncp = n_copies(w_bytes)
        w0 = (torch.randn(N, K, device=dev) / 8).to(torch.float8_e4m3fn)
        ws = make_copies(w0, ncp)
        s = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.01
        cnt = torch.zeros(8, dtype=torch.int32, device=dev)
        pre = torch.zeros(sms * 32, dtype=torch.float32, device=dev)

        for M in [int(v) for v in args.rows.split(",")]:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) / 8
            y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            bn, bk = args.block_n, args.block_k
            n_blocks = triton.cdiv(N, bn)

            def mk(grid, prestage):
                def call(w):
                    _gemv_stage_kernel[(grid,)](
                        x, w, s, y, cnt, pre, M, N, K, n_blocks, grid,
                        M_PAD=16, BLOCK_N=bn, BLOCK_K=bk, PRESTAGE=prestage,
                        num_warps=4, num_stages=3,
                    )

                return call

            def fork_call(w):
                w8a16_gemv(x, w, s, out=y)

            variants = {
                "fork": (fork_call, ws),
                "native": (mk(n_blocks, False), ws),
                "persistent": (mk(sms, False), ws),
                "island": (mk(sms, True), ws),
            }
            res = ab_rounds(variants, rounds=args.rounds, min_seconds=args.min_seconds)
            gb = w_bytes / 1e9
            row = {
                tag: {
                    "us": r["median"],
                    "GBps": round(gb / (r["median"] * 1e-6), 1),
                    "rounds": r["rounds"],
                }
                for tag, r in res.items()
            }
            row["n_blocks"] = n_blocks
            row["waves_native"] = round(n_blocks / sms, 2)
            row["weight_MB"] = round(w_bytes / 2**20, 1)
            row["copies"] = ncp
            payload["results"][f"{shape}_M{M}"] = row
            print(f"\n[{shape} N={N} K={K} M={M}] n_blocks={n_blocks} "
                  f"({n_blocks / sms:.2f} waves), weight {w_bytes / 2**20:.1f} MB")
            for tag in ("fork", "native", "persistent", "island"):
                print(f"    {tag:<11s} {row[tag]['us']:7.2f} us  {row[tag]['GBps']:7.1f} GB/s")
            print(f"    persistent-native = {row['persistent']['us'] - row['native']['us']:+.2f} us"
                  f"   island-persistent = {row['island']['us'] - row['persistent']['us']:+.2f} us")

        del ws, w0
        torch.cuda.empty_cache()

    emit(args.out, payload)


if __name__ == "__main__":
    main()
