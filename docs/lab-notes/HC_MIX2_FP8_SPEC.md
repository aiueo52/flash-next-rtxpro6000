# HC mix2: FP8 weights for the low-rank mix (spec)

## Target
`python/sglang/srt/layers/hc_mix2_triton.py` (`hc_norm_mix2`, three kernels: K0 stats/normalize,
K1 split-K down-proj `normed[M,10240] @ w_down^T[10240,320]` with fp32 atomics into `t_raw`,
K2 up-proj `silu(t)[M,320] @ w_up^T[320,10240]` + sigmoid gate + mean over the 4 branches).
Called from `layers/hyperconnection.py GatedResidual.mix()` when `SGLANG_HC_MIX2=1` (serving default).
Weights per call: `w_down` [320, 10240] bf16 = 6.5 MB, `w_up` [10240, 320] bf16 = 6.5 MB.
In-server (W16 code-edit, CUPTI medians): K0 1.6 us + K1 6.3 us + K2 6.4 us = 14.3 us/call, ~97 calls
per decode step (48 layers x {attn, mlp} + final mixer) => ~1.4 ms/step. The two 6.5 MB reads are the
floor (measured ~11.5 us for 13 MB cold). FP8 weights halve the bytes: goal <= ~9-10 us/call, i.e.
-0.4..-0.5 ms/step at W16, ~-0.25 ms at W4.

## Design
- Quantize once at weight-load time (in `GatedResidual`, when a new env flag `SGLANG_HC_MIX2_FP8=1`
  is set): per-output-row FP8 e4m3 with fp32 scales (`w_down`: 320 scales; `w_up`: 10240 scales).
  There is prior art in `layers/hc_mix_triton.py::quantize_hc_mix_weights_fp8` (used by the older
  fused path) — reuse or mirror it. Keep the bf16 originals out of GPU memory after quantization.
- K1: load the fp8 tile, `.to(tl.bfloat16)`, tl.dot as today; multiply the fp32 partial by the
  per-n scale BEFORE the atomic add (linear, so split-K stays exact w.r.t. the scaling).
- K2: same, scale per output column j (the 10240 axis) applied to the accumulator.
- Keep the bf16 path intact (flag off => bit-identical to today). Config knobs via the existing
  `HCMix2Config` (block sizes may want retuning for the narrower tiles: re-sweep K1/K2 geometry).

## Bench methodology (mandatory)
Use CUPTI kernel durations from `torch.profiler` over CUDA-graph replay with a rotating working set
>= 4x L2 (128 MB): see `bench/w8a16v2/harness.py::cupti_kernel_us` in the worktree
`~/tools/sglang-qsa-ring` (branch opus/gemv3). CUDA-event timing includes launch gaps (+25%) and is
NOT comparable to the server numbers. Reproduce today's bf16 numbers (K1 6.3 / K2 6.4 / K0 1.6 us,
M=16 and M=1) within +-25% before trusting any FP8 number. Re-measure winners head-to-head in one
process (cross-process drift is ~6%).

## Quality gates
1. Unit test vs the bf16 kernel with the same fp8-dequantized weights: mixed/normed max rel err
   consistent with fp8 weight quantization only (normed must stay bit-identical; mixed err vs the
   bf16-weight reference <= ~1e-2 relative, report the actual distribution).
2. The supervisor will validate in-server: trimmed kernel time, acceptance length unchanged
   (greedy W4 3.7-3.85 / 2.2-2.3 / 3.1; W16 code 9-10.8 / prose 2.4-2.5), needle 18.5k PASS.
   If acceptance drops measurably, the change is rejected (this weight set gates the residual mix of
   every layer, so precision matters more than for an ordinary linear).

## Where to work
Worktree `~/tools/sglang-qsa-ring` (create branch `opus/hc-fp8` from `codex/perf-v1` HEAD in the
main tree; do NOT edit `~/tools/sglang-rtxpro6000`). Bench in `bench/hc_mix2/` (existing
`bench_hc_mix2.py` is the geometry sweeper). Tests in `test/srt/layers/test_hc_mix2.py`.
GPU rule: the GPU is shared with server measurements; only use it when the marker file named by the
supervisor exists, and never run two benchmarks concurrently.

## Addendum (2026-09-03 21:30): second item for the same agent
**Shared-expert gate_up + silu_and_mul fusion.** In every layer the shared expert runs
`w8a16_gemv` for gate_up (N=1280 = [gate 640 | up 640], K=2560, fp8 [N,K]-contiguous) followed by
`sglang::act_and_mul_kernel` (`python/sglang/srt/layers/activation.py:92`, 48 launches/step, 98 us,
plus 14 more in the draft at W16). Add a kernel variant (or an epilogue mode) to
`python/sglang/srt/layers/quantization/w8a16_gemv.py` that computes silu(gate) * up directly, i.e.
each CTA owns columns n and n+640 (same k loop, two accumulators) and writes y[M, 640]; plumb it from
the shared-expert MLP (`python/sglang/srt/models/qwen2_moe.py` / `qwen3_5.py`, find the shared
expert forward) behind `SGLANG_SHARED_GATEUP_FUSED=1`. Numerics: silu in fp32 then bf16 store,
compare against the two-kernel path (expect <= 1 bf16 ulp). Bench per the methodology above at
M=1, 4, 16 with split-K where it helps (the current gate_up tiles are in `_BY_SHAPE*`).

## GPU protocol (strict, supersedes the line above)
The GPU is currently owned by an extraction job. Do all reading/writing/CPU work now. Before ANY GPU
use wait until `$SCRATCH/GPU_FREE_FOR_KERNELS`
exists (`until [ -f <path> ]; do sleep 60; done`), then wrap EVERY GPU command in
`flock $SCRATCH/gpu.lock <cmd>`
(another kernel agent shares the GPU under the same lock). Never start an SGLang server.
Worktree: `~/tools/sglang-qsa-ring`, branch `opus/hc-fp8` created from commit 61cf5390bc
(`git -C ~/tools/sglang-qsa-ring checkout -B opus/hc-fp8 61cf5390bc`; the bench dir bench/w8a16v2 is
already tracked there). Env: `. bench/w8a16v2/env.sh` (PYTHONPATH already points at this worktree).
