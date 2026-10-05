# RT1: a faster bit-exact router top-k (FC_fc-moe #3, step 1)

## 0. Result (read this first)

- **Bit-exact on the GPU**: production's weights and ids bit for bit, 12 cases x 65536 rows plus the server
  shapes M=1/4/16, for both key widths and both routed-sum spellings (`runs/rt1/check.jsonl`).
- **Router share of the GEMV + router chain: 3.8 µs → 1.5-1.6 µs** with 32-bit keys (int64 keys: 2.4 µs).
  That is **-2.3 µs per call**, about **-0.13 ms/step at wa** (55 calls), -0.12 at W4 (51), -0.145 at W16 (63),
  roughly 1% of a step.
- Folding the top-k into the GEMV (FC_fc-moe #3 as first proposed) can win at most about 0.8 µs more per call:
  not pursued.
- Server: only the 32-bit kernel ships (the router GEMV writes bf16 and the bias is zero); routed sum as
  production's `tl.sum`. Worktree commit e959fa3b99, `SGLANG_ROUTER_FAST_TOPK=1`, default off.
- Server module check, nan-rank case and in-server smoke: PASS (§7). PENDING: the ABBA (`bench/rt1/abba_rt.sh`).

## 1. What is being replaced

Production router path for every MoE layer (alt stream, on the critical path; FC_fc-moe §3):

1. `SGLANG_ROUTER_GEMV=1`: `bf16_gemv` [32 CTAs, split-K 10] writes the bf16 logits [M, 512] (about 4.4 µs).
2. `fused_topk` → `moe_fused_gate(..., scoring_func="softmax")` → `_router_triton_kernel`
   (`sglang/kernels/ops/moe/moe_fused_gate.py`): grid (M,), 1 warp, BLOCK_N 512, BLOCK_K 16, top-10,
   renormalize, PDL from `is_arch_support_pdl()`.

In-trace median of `_router_triton_kernel` (wa): 4.26 µs at M=8 (verify, 49 calls per step) and 4.69 µs at
M=1 (draft, 6 per step), 55 calls per step.

Production picks each of the 10 experts with three dependent warp reductions:

- the max of the biased logits (16 in-thread steps + a 5-level shuffle butterfly);
- the lowest lane among the lanes equal to the max (`redux.sync.min`);
- a masked sum that fetches the winner's softmax weight (16 in-thread steps + a 5-level butterfly).

PTX per kernel (offline compile at the server's specialization, `scratchpad/rt1/*.ptx`):

| kernel | shfl.sync | redux.sync | setp | PTX lines |
|---|--:|--:|--:|--:|
| production | 127 | 10 | 341 | 2830 |
| fast (int64 keys) | 127 | 0 | 165 | 2125 |
| fast32 (32-bit keys) | 27 | 10 | 175 | 1742 |

The shuffle count of `fast` equals production's, but its pick is one dependent chain (an int64 max) instead of
three. `fast32`'s pick is 15 in-thread `max.s32` and one `redux.sync.max.s32`.

## 2. Design

### 2a. int64 keys (`_router_softmax_fast_kernel`, any float logits)

```
key = (order-preserving int32 of the fp32 biased logit) << 32 | (BLOCK_N - 1 - lane)
```

- One int64 max per pick gives the winner and the tie-break (the lowest lane wins, as in production).
- The winner's weight is recomputed from the float decoded out of the key with the same instructions
  production applies to every lane (fsub, fmul by log2e + ex2.approx, div.full by the row sum).
- A tree reduction for the in-thread part (tried as "fast2") compiles to the same PTX: LLVM already
  reassociates the int64 max chain into a depth-4 tree.

### 2b. 32-bit keys (`_router_softmax_fast32_kernel`, bf16 logits and a zero bias)

`fused_topk` passes a zero fp32 bias, and the router GEMV writes bf16, so `biased = float(logit) + 0.0` is a
bf16 value (-0.0 becomes +0.0). Its fp32 bits have 16 zero low bits, and the order of the logits is the
order of the high halves:

```
h = bits >> 16;  s = h ^ ((h >> 15) & 0x7FFF)       # order-preserving int16
v = 2 * s   (NaN: v = -58005)
key = v << 9 | (511 - lane)                          # 17 + 9 bits
```

- NaN: production floors NaN to -1e30 for the ranking. -1e30 (fp32 0xF149F2CA) lies strictly between bf16
  0xF14A (-1.0003e30) and 0xF149 (-9.95e29). The order-preserving int16 of 0xF149 is -29002, so the odd
  v = 2 * -29002 - 1 sits exactly in that slot.
- Decode: `wv = kmax >> 9`, `ws = wv >> 1`, `wh = ws ^ ((ws >> 15) & 0x7FFF)`, `wb = bitcast(wh << 16)`,
  replaced by -1e30 when `wv` is odd; `win_lane = 511 - (kmax & 511)`.
- Range: |v << 9| ≤ 2^25, so the sentinel INT32_MIN is below every key.

### 2c. The routed sum

- `tl.sum` over the [1, BLOCK_K] slot tensor, as production does.
- `EXPLICIT_SUM=1` spells production's order out for K=10 (slots 0..7 in sequence, plus slots 8 + 9, as the
  production LLIR shows). That order matches the GPU layout only; under `TRITON_INTERPRET=1` it does not,
  which is expected.

## 3. Why the outputs are identical

- Everything up to `row_sum` is production's code (same layout, so the same in-thread chain and xor 16/8/4/2/1
  butterfly for the row max and the row sum).
- The winner set and order are production's: the key order is the float order of `biased`, with ties broken
  by the lower lane; -0.0 never occurs because of the `+ bias` (+0.0).
- The winner's weight is production's `activated[win]`, recomputed from the identical float.
- A NaN logit makes the row sum, and so every weight of the row, NaN in production and here; the ids rank the
  NaN at production's -1e30 floor.

## 4. Offline evidence (before the GPU)

- IR: the int64 kernel's row max / row sum section matches production's LLIR.
- `TRITON_INTERPRET=1` (CPU, `scratchpad/rt1/interp_check.py`): ids equal for fast / fast32 / fast32x,
  weights equal for fast / fast32; fast32x weight mismatches are the interpreter's own sum order (§2c).

## 5. GPU check and chain timing (`bench/rt1/gpu1.sh`, 11:39-11:41)

Method:

- Check: 65536 rows per case (random scales, ties, all-equal, ±0, spiky, NaN, values around the -1e30 floor,
  real router-GEMV logits), plus the server's launch shapes M=1, 4, 16 (M is a Triton specialization key).
  Bit for bit, NaN == NaN.
- Timing: CUDA graphs of R back-to-back (router GEMV + router) chains at M=1, 4, 16; `minus_gemv_us` is the
  router's share. `gemv+consume` (a kernel that only reads the logits behind the GEMV, with PDL) is the floor
  for any separate dependent kernel, so `gemv+consume - gemv` bounds what folding the top-k into the GEMV
  could still save beyond the faster kernel.

Check (`runs/rt1/check.jsonl`): `PASS`, 0 bad rows for fast, fastx, fast32 and fast32x in every case.

Timing (`runs/rt1/time1.jsonl`; CUDA graphs of 32 chained calls, 7 interleaved rounds, median µs per call):

| M | gemv | +prod | +fast | +fastx | +fast32 | +fast32x | +consume |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 2.47 | 6.28 | 4.82 | 4.83 | **3.97** | 3.98 | 3.24 |
| 4 | 2.91 | 6.75 | 5.40 | 5.37 | **4.50** | 4.48 | 3.76 |
| 16 | 3.27 | 7.13 | 5.70 | 5.69 | **4.85** | 4.85 | 4.10 |

Router share (minus `gemv`), µs at M=1/4/16:

- production 3.81 / 3.84 / 3.86;
- int64 keys 2.35 / 2.49 / 2.43;
- **32-bit keys 1.50 / 1.59 / 1.58**;
- dependency floor (`consume`) 0.76 / 0.85 / 0.82.

Readings:

- The explicit routed-sum order costs and wins nothing; the server keeps production's `tl.sum`.
- The in-trace durations of `_router_triton_kernel` (4.3-4.7 µs) are longer than its 3.8 µs share because a PDL
  kernel starts early and waits at `gdc_wait` for the GEMV.
- The router is on the alt stream, which is the MoE layer's critical path (the shared expert on the main stream
  is about 10 µs against 50-120 µs routed), so the share comes off the step.

Added after session 1 (`runs/rt1/check2.jsonl`, with the in-server smoke):

- `server`: the worktree's `moe_router_softmax_fast.py` itself, loaded by path.
- `nan-rank`: fewer than 10 lanes above the -1e30 floor, so the picks reach the NaN lanes and their bf16
  neighbours (the older `nan-floor` case never got past its -1e29 lanes). On the CPU interpreter, 23 of the 24
  NaN rows reach a NaN lane, and the server kernel matches production.

## 6. Server integration (worktree `~/tools/sglang-rt1`, branch `opus/router-fast-topk`)

- `python/sglang/kernels/ops/moe/moe_router_softmax_fast.py` (commit e959fa3b99): the 32-bit kernel only,
  `covered()` (bf16 logits, fp32 bias, N ≤ 512) and `route_softmax_fast(scores, zero_bias, topk, renormalize)`.
  The int64 kernel (15c62f647e) is dropped: no model this fork serves has non-bf16 router logits.
- `python/sglang/srt/layers/moe/topk.py`: `SGLANG_ROUTER_FAST_TOPK=1` (default off) routes the CUDA softmax
  path of `fused_topk` through it when `covered()`; everything else still goes to `moe_fused_gate`.
- `bench/rt1/arm_rt.sh` (one `serve-fast.sh wa` arm, port 8021) and `abba_rt.sh` (A off, B on), with
  `router_trace.py` counting the router kernels in a 20-step trace.

## 7. In-server smoke and A/B

Check 2 and smoke (`bench/rt1/gpu2.sh`, 12:05-12:19, `runs/rt1/check2.jsonl`, `runs/rt1/smoke-rt.log`):

- `PASS`: 0 bad rows in every case, including `server` (the worktree module, all 13 cases), `nan-rank` and the
  server shapes M=1/4/16.
- Smoke (`SGLANG_ROUTER_FAST_TOPK=1`, `serve-fast.sh wa`): server up, 0 Traceback/Error lines; in the profiled
  20-step greedy code-edit trace `_router_softmax_fast32_kernel` n=1100 (55 per step) with a median of
  2.37 µs, and `_router_triton_kernel` n=0, so every router call took the new kernel. Production's in-trace
  median was 4.26-4.69 µs.

ABBA: PENDING (`bench/rt1/abba_rt.sh`, queued by `bench/stack/chain_late.sh` behind the RQ2 ABBA).
