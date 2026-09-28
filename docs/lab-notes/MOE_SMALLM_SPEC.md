# Routed-expert grouped GEMM at small M: call path, bytes model, and what is left to win

Date: 2026-09-05. CPU-only analysis. No GPU used, no server started, nothing under
`python/sglang` modified. Sources: the chrome traces in `prof/traces/`, the checkpoint
headers under `~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/`, the SGLang fork at
`~/tools/sglang-rtxpro6000` (branch `codex/perf-v1`), FlashInfer 0.6.17 as installed in that
fork's `.venv`.

Code and harness: `bench/moe_smallm/`. Run instructions in section 7.

---

> **Status note (2026-09-05, after the GPU runs): section 8 supersedes parts of sections
> 2-5.** The distinct-expert count has since been *measured* in-server (28.4 at W4, 69.4 at
> W16), the harness was found to be running the wrong CUTLASS tactic, and the conclusion
> changed: the GEMM is wave-quantisation-limited at ~70 % of peak, not bandwidth-saturated
> at 88 %. Read section 8 first; sections 1-5 remain correct on the call path, the byte
> accounting and the FlashInfer knob space.

## TL;DR

1. **The claim "70-75 % of memory bandwidth at W4" cannot be verified from the trace alone,
   because the number of *distinct* experts touched per layer per step is not measured.**
   What the trace does prove:
   * Independent top-10 routing is an **upper bound** on the distinct count, so it gives an
     upper bound on achieved bandwidth: **≤ 1586 GB/s at W4 (≤ 88.5 % of the 1792 GB/s peak).**
   * At W16 the same assumption implies **2711 GB/s, i.e. 151 % of peak — impossible.**
     Routing in a speculative chain is therefore *provably* correlated: at T=16 at most
     **91.6** distinct experts are touched (66 % of the independent prediction of 138.6), and
     at the 1461 GB/s copy roof measured on this box, at most **74.7**.
   * The two widths are consistent with **one** efficiency: `t(16)/t(4) = 2.087`, so
     `D16 = 2.087 · D4`. The "70-75 %" figure corresponds to `D4 ≈ 31-33`, `D16 ≈ 65-69`.
   * The one experiment that settles it is a **kernel-time-vs-D sweep** (`--sweep-distinct`),
     which measures the slope `dt/dD` directly. Bandwidth = `bytes_per_expert / slope`, with
     no knowledge of the real routing needed. This is deliverable 4 and should be run first.

2. **The kernel is close to roofline and has little left in it.** At 90 % of peak the floor is
   3.20 ms/step (W4) and 6.66 ms/step (W16) *if* D matches the independent bound, against
   3.30 / 6.78 ms measured (outlier-robust). Kernel-side headroom is **0.1-0.7 ms at W4 and
   0.1-1.3 ms at W16**, the range being exactly the uncertainty in D.

3. **The two large, cheap, low-risk wins are not in the GEMM at all:**
   * **A0 — FlashInfer's fused routing prologue is disabled for this model on two counts**
     (top-k = 10 is not in the `{1,2,4,6,8}` switch, and 512 experts give `expert_log = 10`
     against a `≤ 9` gate). The 3-kernel fallback costs **0.34 ms/step at W4 and 0.41 ms/step
     at W16**; the fused path is one ~2 µs kernel. **Saving ≈ 0.23 / 0.29 ms/step**, ~5 lines
     in the FlashInfer csrc (not in `python/sglang`), and the shared-memory arithmetic says it
     will fit (64 KB of 99 KB at `BLOCK_SIZE=32`).
   * **A1 — the >200 µs outliers.** 1-5 MoE GEMM calls per step run 5-10x long, scattered
     randomly across layers and steps (44 of 96 call positions hit at least once, none every
     time). That is external preemption, not a kernel property, and it costs **0.46 ms/step at
     W4 and 1.13 ms/step at W16 of MoE GEMM time alone.**

4. **Expert dedup across the T chain tokens is already done** — the grouped GEMM reads each
   distinct expert once regardless of how many of the T·10 rows land on it. `OVERHEAD_REPORT.md`
   §6 F1 ("expert dedup across the T chain tokens") is not an available lever. The only way to
   move D is to change routing, which changes the model.

---

## 1. The exact call path at decode/verify time

### 1.1 Python

`qwen4_exp.py` does not define its own MoE method; the routed experts go through the generic
`FusedMoE` layer, and the NVFP4 quant method is what selects the runner. Chain (verified
file:line in `~/tools/sglang-rtxpro6000`):

```
FusedMoE.forward                       python/sglang/srt/layers/moe/fused_moe_triton/layer.py:1421
 └ forward_impl                                                                          :1462
    ├ dispatcher.dispatch(...)                                                           :1475
    ├ run_moe_core -> quant_method.apply(layer, dispatch_output)                     :1530/:1532
    │   └ ModelOptNvFp4FusedMoEMethod.apply    python/sglang/srt/layers/quantization/modelopt_quant.py:2990
    │      └ if self.enable_flashinfer_cutlass_moe:                                      :3098
    │         FlashInferCutlassMoeQuantInfo(quant_type="fp4", ...)                        :3106
    │         quant_scales = [w13_input_scale_quant, w13_blockscale_swizzled, g1_alphas,
    │                         w2_input_scale_quant,  w2_blockscale_swizzled,  g2_alphas]  :3111
    │         return self.runner.run(dispatch_output, quant_info)                         :3125
    │            └ MoeRunner.run              python/sglang/srt/layers/moe/moe_runner/runner.py:143
    │               └ fused_experts_none_to_flashinfer_cutlass
    │                                   python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:247
    │                  └ _run_flashinfer_cutlass                                          :178
    │                     ├ w13/w2 .view(torch.long); scales[1]/[4].view(torch.int32)  :209-220
    │                     └ flashinfer.fused_moe.cutlass_fused_moe(...)                :222-240
    └ dispatcher.combine(...)                                                             :1501
```

Weight preparation is `ModelOptNvFp4FusedMoEMethod.create_weights` (`modelopt_quant.py:2436`)
and `process_weights_after_loading` (`:2593`). For the `flashinfer_cutlass` branch the input
scales collapse to a scalar (`layer.w13_input_scale.max()`), `g1_alphas = w13_input_scale *
w13_weight_scale_2[:, 0]` (`_compute_gemm1_alphas`, `:2330`) and the block scales are swizzled
into the 128x4 layout by `swizzle_blockscale` (`python/sglang/srt/layers/quantization/utils.py:597`).

Arguments SGLang actually passes (`flashinfer_cutlass.py:222-240`): `output, input,
token_selected_experts, token_final_scales, fc1/fc2_expert_weights, output_dtype=bfloat16,
input_sf=None, quant_scales, ep/tp size+rank=1/0, tune_max_num_tokens=next_power_of_2(M),
activation_type=Swiglu, enable_alltoall=False, use_fused_finalize=SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`
(default True). It does **not** pass `workspace_buffer`, `enable_pdl`, `swizzled_input_sf` or
`profile_ids`.

### 1.2 Kernels in the trace, per MoE call

All from `flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`.
Counts are per decode step in `prof-base32k-*-code-edit` (48 verify calls + S draft calls +
1 draft_extend call).

| # | kernel (trace name) | grid / block | µs/call W4 | µs/call W16 | what it does |
|---|---|---|--:|--:|---|
| 1 | `blockExpertPrefixSumKernel<512>` | [512,1,1] / 512 | 2.53 | 2.40 | 3-step routing prologue, step 1 |
| 2 | `globalExpertPrefixSumKernel<512>` | [1,1,1] / 512 | 1.73 | 1.66 | step 2 |
| 3 | `mergeExpertPrefixSumKernel` | [512,1,1] / 512 | 1.89 | 1.92 | step 3 |
| 4 | `expandInputRowsKernel<__nv_bfloat16, __nv_fp4_e2m1, FpXBlockScalingType(1), false, false, ...>` | **[T·10,1,1]** / 256 | 4.19 | 4.54 | gather + **BF16→NVFP4 activation quant** |
| 5 | `computeStridesTmaWarpSpecializedKernel<fp4, fp4, bf16, bf16>` | [16,1,1] / 32 | 2.85 | 2.78 | per-expert TMA descriptors, 1 thread/expert over all 512 |
| 6 | `cutlass::device_kernel<GemmUniversal<GroupProblemShape, MainloopSm120ArrayTmaWarpSpecializedBlockScaled<S,3,(1,1,1)>, KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>, TileShape, (float_e2m1_t, float_ue4m3_t), ...>>` | **[1,188,1]** / 384 | 42.4 | 87.3 | **GEMM1 = gate_up** |
| 7 | `doActivationKernel<__nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16, GLUAdaptor<SiLu>, ...>` | [T·10,1,1] / 256 | 3.65 | 5.41 | SwiGLU + **BF16→NVFP4 re-quant** |
| 8 | `cutlass::device_kernel<...>` (2nd instantiation) | [1,188,1] / 384 | 25.3 | 54.0 | **GEMM2 = down**, finalize fused in the epilogue |

There is **no separate `finalizeMoeRoutingKernel`** — `use_fused_finalize` defaults to True, so
the top-k weighted scatter-add is the GEMM2 epilogue
(`ScaledAccPerRowBiasPerColScaleScatter`, `moe_gemm_tma_ws_launcher.inl:415-431`).

`expandInputRows`'s grid **is** the expanded row count and is the cleanest confirmation of the
routing shape in the trace: 40 at W4 verify, 160 at W16 verify, 10 in each draft iteration,
= `T · top_k` with `top_k = 10`.

### 1.3 Tile config chosen for small M

Grid `[1, 188, 1]` = a **persistent** kernel, one CTA per SM (188 SMs). Block 384 = 3 warpgroups
(cooperative: 2 math + 1 producer). Cluster is hard-wired `1x1x1` on sm120
(`moe_gemm_template_dispatch_tma_ws.h:300-303`).

Decoding the mangled names against FlashInfer's tactic table
(`cutlass_heuristic.cpp:595-638 get_candidate_configs_sm120`; base tiles are quoted in *bytes*
of K, so element-K is 2x for FP4):

| call | tile (M,N,K elements) | stages | tactic | note |
|---|---|--:|---|---|
| verify GEMM1, W4 and W16 | **128 x 32 x 256** | 3 | 16 = base 6 + `swap_ab` | |
| verify GEMM2, W4 | **128 x 32 x 128** | 6 | ~57 = base 7 + FINALIZE + `swap_ab` | |
| verify GEMM2, W16 | **128 x 64 x 256** | 2 | ~58 = base 8 + FINALIZE + `swap_ab` | |
| draft GEMM1 (M=10) | **128 x 32 x 128** | 6 | 17 | |
| draft GEMM2 (M=10) | **256 x 128 x 128** | 2 | 52 = base 2 + FINALIZE + `swap_ab` | |

These match the tuned entries already sitting in
`~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/*/rank_tp0_pp0_dp0.json` for exactly this
layer's shapes (gemm1 16/17/19, gemm2 47/49/52/57 depending on M; the spread at M=1-8 is
timing noise, not a real preference).

**There is no small-M tile on sm120.** The enumerator is a flat 10-entry list with no
dependence on M; M-tile is only 128 or 256, minimum N-tile is 32, `max_split_k = 1` is pinned
(`moe_gemm_template_dispatch.h:593`), the tile scheduler is `{1, AlongN}` with no stream-K, and
there is no cluster or stage choice. **The only small-M lever in the whole knob space is
`swap_ab`, which puts the token count on the N axis (tiles 32/64/128/256) instead of the
128-row M axis — and the autotuner already picks it.** Two of the 10 base tiles (indices 4, 5)
have no compiled kernel and throw; the sm120 occupancy pre-filter cannot catch them
(`moe_gemm_template_dispatch.h:1070-1072` returns 1 unconditionally).

### 1.4 Token grouping per expert

`threeStepBuildExpertMapsSortFirstToken` (`cutlass_fused_moe_kernels.cuh:898`) builds
`permuted_row_to_unpermuted_row` and `expert_first_token_offset` — a counting sort of the
`T · 10` (token, slot) pairs by expert id. `computeStridesTmaWarpSpecialized` then turns the
offsets into one `GroupProblemShape` entry per expert. **Experts with zero rows produce zero
tiles and their weights are never read** — this is why kernel time scales with the distinct
expert count and not with 512.

The three-step path is the *fallback*. See section 6, item A0: the fused single-kernel prologue
is silently disabled for this model.

### 1.5 Activation quantisation and accumulation

* **The activations are quantised to NVFP4 inside the runner, twice per call, fused into the
  glue kernels — there is no separate quant launch.**
  * before GEMM1, in `expandInputRowsKernel` (`:1455`, `:1536-1543`), using
    `quant_scales[0] = layer.w13_input_scale_quant = 1 / max(w13_input_scale)`;
  * between GEMM1 and GEMM2, in `doActivationKernel` (`:2274-2283`), using
    `quant_scales[3] = w2_input_scale_quant`.
  * Both write per-16 e4m3 block scales directly in the `SWIZZLED_128x4` layout
    (`cvt_quant_get_sf_out_offset<..., QuantizationSFLayout::SWIZZLED_128x4>`, `:1044-1049`).
  * The checkpoint's `input_activations.dynamic = false` refers only to the *global* fp32 scale
    (`input_scale`); the per-16-block e4m3 scales are computed at runtime, every call.
* **Accumulate is FP32**, unconditionally (`ElementAccumulator = float`,
  `moe_gemm_tma_ws_launcher.inl:302`). GEMM1 output is BF16, GEMM2 output is BF16
  (`ElementD = cutlass::bfloat16_t`). The block-scale element is `float_ue4m3_t` with vector
  size 16.
* The runner variant is `CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16,
  __nv_bfloat16, __nv_bfloat16, false>` — i.e. W4A4 with `NeedQuant = true`, selected because
  SGLang passes raw BF16 with `input_sf = None`
  (`flashinfer_cutlass_fused_moe_binding.cu:163-175`).

---

## 2. Bytes model

### 2.1 Per-expert weight bytes (read off the checkpoint header, not assumed)

`layer-00004-experts-0000-0127.safetensors`, expert 0:

| tensor | dtype | shape | bytes |
|---|---|---|--:|
| `gate_proj.weight` | uint8 (2 x fp4) | [640, 1280] | 819 200 |
| `gate_proj.weight_scale` | float8_e4m3 | [640, 160] | 102 400 |
| `up_proj.weight` / `.weight_scale` | | same | 921 600 |
| `down_proj.weight` | uint8 | [2560, 320] | 819 200 |
| `down_proj.weight_scale` | float8_e4m3 | [2560, 40] | 102 400 |
| `*.weight_scale_2`, `*.input_scale` | f32 scalars | [] | 24 |

`160 = 2560/16` and `40 = 640/16` confirm **block size 16** (matches `quantization_config.
config_groups.group_0.weights.group_size = 16`). Effective **4.5 bits per weight**.

```
bytes_gemm1_per_expert (gate+up) = 2*640*2560 * 9/16 = 1 843 200 B
bytes_gemm2_per_expert (down)    =   2560*640 * 9/16 =   921 600 B
bytes_per_expert                 =                     2 764 800 B = 2.7648 MB
all 512 experts, one layer       =                     1.416 GB
```

Shapes in SGLang's stacked layout (`create_weights`, `modelopt_quant.py:2458-2528`):
`w13_weight [512, 1280, 1280] u8`, `w13_weight_scale [512, 1280, 160] e4m3`,
`w2_weight [512, 2560, 320] u8`, `w2_weight_scale [512, 2560, 40] e4m3`.
The swizzle is shape-preserving here (1280 % 128 == 0, 160 % 4 == 0, 2560 % 128 == 0,
40 % 4 == 0), so no padding bytes.

### 2.2 Expected distinct experts per layer per step

For one token, `P(expert e not in its top-10) = 1 - 10/512` exactly (exchangeability of a
top-k draw). Assuming the T tokens route **independently**:

```
E[D(T)] = 512 * (1 - (1 - 10/512)^T)
```

| T | E[D] independent | bytes/layer | bytes/step (48 layers) |
|--:|--:|--:|--:|
| 1 | 10.00 | 27.6 MB | 1.33 GB |
| 2 | 19.80 | 54.8 MB | 2.63 GB |
| **4** | **38.84** | 107.4 MB | **5.155 GB** |
| 8 | 74.74 | 206.6 MB | 9.92 GB |
| **16** | **138.57** | 383.1 MB | **18.39 GB** |

The T tokens of a speculative chain are *not* independent — they share a prefix and are
adjacent positions — so this is an **upper bound**. Section 2.4 shows the trace falsifies it
outright at T=16.

### 2.3 Achieved bandwidth from the trace

Device: RTX PRO 6000 Blackwell Max-Q, GDDR7, 512-bit, `clocks.max.memory = 14001 MHz` =
28 Gbps effective, so **peak = 28e9 · 512/8 = 1792 GB/s** (this is the correct figure; it is not
a 6 TB/s HBM part). L2 = 128 MiB, 188 SMs, power-capped at 325 W. The measured *copy* roof on
this box is about **1461 GB/s** (re-measured at 1456 GB/s with `bench/moe_smallm/roofline.py`, see
`G1_LOG.md`); a pure streaming read should sit somewhere between that and peak.

Per-layer verify GEMM time, taken as the **sum of per-call-position medians** over the 19 steps
of `prof/traces/prof-base32k-{w4,w16}-code-edit` (this removes the preemption outliers of
section 6/A1; the mean-based numbers used in `OVERHEAD_REPORT.md` §5.1 are 3.761 / 7.902 ms):

| | GEMM1 µs/layer | GEMM2 µs/layer | total µs/layer | ms/step (x48) | mean-based ms/step |
|---|--:|--:|--:|--:|--:|
| **W4** (T=4) | 42.4 | 25.3 | **67.7** | **3.302** | 3.761 |
| **W16** (T=16) | 87.3 | 54.0 | **141.3** | **6.776** | 7.902 |

Achieved bandwidth = `D · 2.7648 MB / t_layer`:

| | assumed D | achieved GB/s | % of 1792 peak |
|---|--:|--:|--:|
| W4, independent | 38.84 | **1586** | **88.5 %** |
| W16, independent | 138.57 | **2711** | **151 %** — impossible |

Inverting instead — the maximum D compatible with a given bandwidth:

| | D at 1792 GB/s (peak) | D at 1613 GB/s (90 % peak) | D at 1461 GB/s (measured copy roof) |
|---|--:|--:|--:|
| W4 | 43.9 | 39.5 | 35.8 |
| W16 | 91.6 | 82.4 | 74.7 |

**Theoretical floor at 90 % of peak** (i.e. what the same D would cost on a perfect kernel):
W4 `5.155 GB / 1613 GB/s = 3.196 ms/step`; W16 `18.39 GB / 1613 GB/s = 11.40 ms/step` — the
W16 figure exceeds the measured 6.78 ms, which is the same contradiction stated differently.

### 2.4 What the trace actually proves

* `t(16) / t(4) = 141.3 / 67.7 = 2.087`. Both widths are weight-bandwidth-bound (M ≤ 16 against
  a 128-row tile means arithmetic is negligible, and per-tile bytes are fixed), so
  **`D16 = 2.087 · D4`** to within the kernel's efficiency ratio.
* Independence predicts `D16 / D4 = 3.57`. It is wrong by 71 %.
* Hard bound: `D16 ≤ 91.6` (at literal peak). So **at T=16 at most 66 % of the independently
  predicted expert set is actually touched.** Routing along a speculative chain is strongly
  correlated. This is a measured result, not a model.
* Self-consistent solutions, parameterised by the one unknown:

  | D4 | D16 | implied bandwidth | % of peak |
  |--:|--:|--:|--:|
  | 38.8 (independent) | 81.0 | 1586 GB/s | 88.5 % |
  | 35.8 | 74.7 | 1461 GB/s | 81.5 % |
  | **32.0** | **66.8** | **1307 GB/s** | **73 %** |
  | 30.0 | 62.6 | 1225 GB/s | 68 % |

  The earlier "70-75 % at W4" note corresponds to `D4 ≈ 31-33`, i.e. about 80-85 % of the
  independent prediction. That is plausible but **unverified**.

### 2.5 How to measure the real distribution

Two ways, both implemented.

**(a) Cheap and exact, no server: sweep D directly.** `bench_moe.py --sweep-distinct` builds
routings with an *exactly* specified distinct count (`routing.with_distinct`) and times the
production kernel. Kernel time is `t = a + b·D`; the slope gives
`BW = bytes_per_expert / b` with no assumption about real routing at all, and the intercept `a`
is the fixed per-call cost. Then the production D is read straight off the fit using the
trace's 67.7 / 141.3 µs. **This is the first thing to run.**

**(b) Log the real routing from a live server.** `bench/moe_smallm/route_logger.py` monkeypatches
`_run_flashinfer_cutlass` (nothing under `python/sglang` is edited) and records, per MoE call,
the number of distinct experts — using **CUDA-graph-capturable ops only** (`occ.zero_()`,
`occ.scatter_()`, `occ.sum()`, `hist.scatter_add_()`; no `torch.unique`, which has a dynamic
shape and would not capture). A plain python hook would fire once at graph capture and never
again during replay; this one records on every replay. It keeps a `[513]` int64 histogram at
zero host cost, plus an optional ring of raw ids for `--routing replay`. Arm it with the
`sitecustomize` shim (see the module docstring); dump with `kill -USR1 <scheduler pid>`.
A few hundred steps gives the full distribution for both widths and both workloads.

---

## 3. Where the rest of the MoE time goes (glue), per step

From the same traces, summed over verify + all draft iterations + draft_extend:

| group | W4 µs/step | W16 µs/step |
|---|--:|--:|
| `expandInputRows` + `doActivation` (gather, act-quant, SwiGLU) | 473 | 626 |
| **3-step routing prologue** (`blockExpertPrefixSum` + `globalExpertPrefixSum` + `mergeExpertPrefixSum`) | **335** | **415** |
| `computeStridesTmaWarpSpecialized` | 154 | 222 |
| **glue total** | **962** | **1262** |
| MoE GEMM (robust) | 3 302 | 6 776 |
| MoE GEMM (mean, incl. preemption outliers) | 3 761 | 7 902 |

The prologue and `computeStrides` are pure launch-latency-bound fixed cost: 1.7-2.9 µs each,
essentially independent of T, run 52 (W4) or 64 (W16) times per step. Together they are
**0.49 ms/step at W4 and 0.64 ms/step at W16 — 13 % and 8 % of the whole MoE cost — to
bookkeep at most 160 rows.**

---

## 4. The preemption outliers

1-5 MoE GEMM calls per step take 200-630 µs instead of 40-110 µs. They are **not** cold-start
and **not** layer-specific: over 20 steps of `prof-base32k-w16-code-edit`, 44 of the 96 call
positions are hit at least once and none is hit every time; the affected step numbers are
uniform. This is the same external preemption already identified as `OVERHEAD_REPORT.md` F4
(Max-Q power cap / desktop compositor), now quantified on the single largest kernel:

| | outlier cost, ms/step | as % of MoE GEMM |
|---|--:|--:|
| W4 code-edit | 0.46 | 12 % |
| W4 prose-en | 0.44 | 12 % |
| W16 code-edit | 1.13 | 14 % |
| W16 prose-en | 0.90 | 12 % |

Any A/B on this kernel that does not run headless is measuring this instead.

---

## 5. Alternatives, ranked

Savings are per decode step, against the outlier-robust baseline. "W4" = `prof-base32k-w4-*`,
"W16" = `prof-base32k-w16-*`.

| rank | item | W4 | W16 | effort | risk |
|---|---|--:|--:|---|---|
| **A0** | Enable FlashInfer's fused routing prologue (top-k=10, E=512) | **−0.23** | **−0.29** | ~5 lines + JIT rebuild, 1 day incl. validation | low-med |
| **A1** | Run headless / keep `nvidia-smi -pl 325`; stop the compositor | −0.46 | −1.13 | ops only | none |
| **A2** | Measure D (`--sweep-distinct`, then `route_logger`) | — | — | 2 h GPU | none |
| **A3** | Fold `computeStridesTmaWarpSpecialized` into the prologue | −0.13 | −0.18 | same patch site as A0 | med |
| **A4** | FlashInfer tactic/knob hygiene (pin tactic, blocklist the 2 uncompiled ones, confirm the tune bucket covers T=16) | 0 … −0.1 | 0 … −0.2 | 2 h | low |
| **A5** | Reduce D (verify-time top-k reduction / restrict to the draft's expert union) | −0.33 per 10 % of D | −0.68 per 10 % of D | high | **accuracy** |
| **A6** | MXFP4 (block 32) instead of NVFP4 (block 16): 4.5 → 4.25 bits/weight | −0.18 | −0.37 | high (requantise) | **accuracy** |

### A0 — the fused routing prologue is silently disabled (best value/effort)

`fusedBuildExpertMapsSortFirstToken` (`cutlass_fused_moe_kernels.cuh:557`) returns `false` for
this model on **two** independent gates, so every MoE call falls back to
`threeStepBuildExpertMapsSortFirstToken` (`:898`):

1. `int expert_log = (int)log2(num_experts_per_node + 1) + 1;` → `log2(513) = 9.003` → `9 + 1 =
   10`, and the dispatch array only has entries for `expert_log <= 9` (`:565-578`).
   **512 experts misses by exactly one.**
2. `fusedBuildExpertMapsSortFirstTokenBlockSize` switches on `experts_per_token` with cases
   `{1, 2, 4, 6, 8}` and `default: return false` (`:524-550`). **top-k = 10 is not there.**

The fix is to add `fusedBuildExpertMapsSortFirstTokenBlockSize<10>` to the array and a
`case 10:` to the switch. Feasibility check: at T ≤ 16 the dispatch picks `BLOCK_SIZE = 32`
(`block_size = num_tokens`, `:494`), so the shared memory is
`cub::BlockRadixRank<32, 10, false>::TempStorage` = `COUNTER_LANES(256) · BLOCK_THREADS(32) ·
PACKING_RATIO(4) · 2 B = 64 KiB`, under the ~99 KiB `cudaDevAttrMaxSharedMemoryPerBlockOptin`
on sm120 — and there is a runtime gate at `:471` that fails safe back to the 3-step path if it
does not fit. `BINS_TRACKED_PER_THREAD = 1024/32 = 32` covers `num_experts + 1 = 513`.

This file is inside the **FlashInfer** package
(`.venv/lib/python3.12/site-packages/flashinfer/data/csrc/...`), not `python/sglang`, so the
"don't touch `python/sglang`" constraint is respected. Patch a vendored copy of the csrc tree
and point the JIT at it; validate by asserting the fused and 3-step paths produce identical
`permuted_row_to_unpermuted_row` / `expert_first_token_offset`.

### A1 — headless

See section 4. Nothing to build; the caveat is that part of this may be genuinely present in
production if the desktop is running during serving, in which case it is a real 1.1 ms/step at
W16, not just measurement noise.

### A2 — measure D

Section 2.5. Everything below is conditioned on the answer:

* if `D4 ≳ 37` the kernel is at ≥ 85 % of peak and **there is no kernel work worth doing** —
  go straight to A0/A1 and then to A5/A6;
* if `D4 ≲ 32` the kernel is at ≤ 73 % and there is 0.7 ms (W4) / 1.3 ms (W16) of headroom,
  which would justify looking harder at A4.

### A3 — fold `computeStrides` into the prologue

`computeStridesTmaWarpSpecializedKernel` runs one thread per expert over all 512 experts
(grid 16 x 32) on every call, 2.8 µs, to fill in TMA descriptors for at most 160 rows spread
over ≤ 140 experts. The prologue already computes `expert_first_token_offset`; merging the two
removes one launch per MoE call. Same patch site and the same validation harness as A0, so do
them together. Medium risk: the descriptor layout is what the persistent GEMM's group scheduler
reads.

### A4 — FlashInfer knobs

The whole sm120 NVFP4 knob space is `{8 usable CTA tiles} x {swap_ab} x {NONE|FINALIZE epilogue,
GEMM2 only}` = 20 GEMM1 + 40 GEMM2 tactics. **No cluster shape, no split-K, no stream-K, no
stage count, no schedule, and no M-tile below 128.** Autotune buckets are powers of two from 1
(`get_hybrid_num_tokens_buckets`, `flashinfer/fused_moe/utils.py:184-242`), so `M ∈ {1,2,4,8,16}`
*is* covered, and the on-disk cache for this box already holds tuned entries for exactly this
layer's shapes. The picks are already the narrow-N + `swap_ab` corner. Concretely worth doing:

* confirm `tune_max_num_tokens = next_power_of_2(M)` (`flashinfer_cutlass.py:236`) leaves T=16
  inside the tuned set — the ceiling is set by the warmup decode batch size, and anything above
  it falls back to tactic `-1` = base tile 0 with no `swap_ab`, a bad default at small M;
* pin the chosen tactic by writing the autotune JSON, to stop the run-to-run churn at M=1-8
  (the cache shows 5 different GEMM2 tactics chosen across runs at M=1 — pure timing noise, and
  it makes every A/B noisier);
* `FLASHINFER_TACTICS_BLOCKLIST` out base tiles 4 and 5 (and their FINALIZE/`swap_ab`
  duplicates), which have no compiled kernel and throw during profiling;
* note that flipping `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` renumbers every GEMM2 tactic
  (40 → 20), invalidating any hand-written cache.

Expected saving is small precisely because the autotuner is already at the good corner. Do it
for measurement stability more than for throughput.

### A5 — reduce D (the only large lever, and it changes the model)

Time is linear in D with slope `48 · 2.7648 MB / BW`, i.e. **~85 µs of step time per distinct
expert per layer, at either width** (3.302 ms / 38.8 experts at W4 and 6.776 ms / 81 at W16 both
land there — as they must, since both widths share one bandwidth). Options: lower top-k on the T−1 speculative positions;
restrict the verify's expert set to the union of what the draft chose; prune the bank. All of
them change the target distribution, therefore change acceptance and quality — gate any of them
on a quality check as well as acceptance.

Note also that **expert dedup across the chain tokens is already free**: an expert is streamed
once per layer per call however many of the `T · 10` rows land on it. The lever is the *number
of distinct experts*, not redundancy.

Framing: `OVERHEAD_REPORT.md` §3 puts the true accepted length at W16 code-edit at ~11
tokens/step, so the MoE GEMM costs ~0.62 ms per *accepted* token there. Because D grows only
~2.1x from T=4 to T=16 while the accepted length grows much faster, **W16 is already the
efficient regime for this kernel** — it costs 2.05x the time of W4 for well over 2x the output.
Cutting MoE cost helps the shallow widths most.

### A6 — fewer bytes per weight

NVFP4 is 4 bits + one e4m3 per 16 = 4.5 bits/weight. MXFP4 (e8m0 scale per 32) is 4.25 —
a flat **5.6 % of the MoE GEMM time**. FlashInfer has an sm120 MXFP4 MoE path
(`FlashInferCutlassMxfp4MoeQuantInfo`, MXFP8 activations). Requires requantising the checkpoint
and a full quality re-run; the scale-block coarsening is the accuracy risk.

### Explicitly ruled out

| hypothesis | why not |
|---|---|
| dedup the T chain tokens' experts before the GEMM | the grouped GEMM already reads each distinct expert exactly once |
| split-K / stream-K for tiny M | `max_split_k = 1` is pinned on sm120 (`moe_gemm_template_dispatch.h:593`); stream-K exists only in FlashInfer's *dense* FP4 GEMM |
| a smaller M-tile (64 / 16 / 8) | not enumerated for sm120; `CtaShape128x8x256B`/`128x16x128B` are SM100-only and FP8-only |
| cluster / 2SM tuning | cluster forced `1x1x1`, `Is2SM` false on sm120 |
| a separate `finalize` kernel to remove | already fused into the GEMM2 epilogue (`use_fused_finalize` defaults True) |
| autotune does not reach M ≤ 16 | it does — buckets start at 1, and the on-disk cache has M=1,2,4,8,16 for these shapes |
| `min_latency_mode` | throws in FlashInfer 0.6.17 on this path |

---

## 6. Harness design

`bench/moe_smallm/` (CPU-safe to import; every GPU entry point is marked `GPU-ONLY`).

| file | role |
|---|---|
| `model_shapes.py` | shapes + byte accounting read from `config.json`; `expected_distinct_experts`, `achieved_gbps`, `implied_distinct_experts`; device constants |
| `weights.py` | loads ONE layer's real experts from the per-layer safetensors shards into SGLang's stacked layout; CPU-capable `swizzle_blockscale_cpu`; `prepare_runtime_scales` reproduces `process_weights_after_loading`; `synth_layer` for checkpoint-free runs |
| `routing.py` | `independent`, `correlated` (chain carry), `hot_mixture`, `with_distinct` (exact D — the sweep workhorse), `load_replay`, `disjointify` (spreads the working set past L2) |
| `runners.py` | candidate registry. `flashinfer_cutlass` calls SGLang's own `_run_flashinfer_cutlass` with a duck-typed dispatch output, so what is timed is the production path; `flashinfer_direct` exposes `tune_max_num_tokens` / `use_fused_finalize` / `workspace_buffer`; `triton_grouped` is a stub |
| `harness.py` | CUDA-graph capture + replay timing, clock sampling, idle-GPU guard, working-set check |
| `refs.py` | BF16 dequant reference (e2m1 LUT, low nibble = even k) + error metrics |
| `route_logger.py` | graph-capturable routing histogram + raw-id ring for a live server |
| `sitecustomize_example.py` | arms the logger without editing `python/sglang` |
| `tests_cpu.py` | 34 CPU tests: bytes model, routing invariants, swizzle, dequant, real-checkpoint slice load, CLI, SGLang import-order regression |

### SGLang import order (do not "simplify" `runners.ensure_sglang_imports`)

Importing `sglang.srt.layers.moe.moe_runner.*` standalone hits a circular import that the
server never sees:

```
moe_runner/flashinfer_trtllm.py:110   if is_flashinfer_available():
                                          from ...quantization.fp4_utils import fp4_quantize
  -> layers/quantization/__init__.py:31 -> compressed_tensors/... -> modelopt_quant.py:38
  -> layers/quantization/fp8.py:38      from ...moe_runner.flashinfer_trtllm import
                                            FlashInferTrtllmFp8MoeQuantInfo
  => ImportError: partially initialized module ...flashinfer_trtllm (circular import)
```

`srt/models/*.py` import `layers.quantization.*` first (e.g. `qwen4_exp.py:46-47`), so by the
time any `moe_runner` module is touched the quantization package is already initialising and
`fp4_utils` is just a submodule load. `runners.ensure_sglang_imports()` reproduces that order
and is called from every candidate factory, from `_runner_config`, and up front in
`cmd_run` / `cmd_check`.

The trap is that the cycle is **guarded by `is_flashinfer_available()`**, so it only fires on a
machine with a GPU — a plain CPU smoke test cannot see it. `tests_cpu.py` therefore forces
`sglang.srt.utils.common.is_flashinfer_available` to `True` in a subprocess and asserts both
that the naive order still breaks and that `ensure_sglang_imports()` fixes it, all without
touching CUDA.

Timing methodology (mirrors `sglang-rtxpro6000/bench/w8a16v2/harness.py`):

* **Working set.** One T=4 call touches ~39 experts x 2.7648 MB = 108 MB, which *fits* in the
  128 MiB L2 — replaying it would measure L2, not DRAM. Each measurement replays a rotation of
  `--rotation` (default 16) calls whose expert sets are offset from one another, and
  `check_working_set` refuses to report a clean number unless the union is ≥ 4x L2 (512 MiB).
  Measured at the default settings: 657-1258 MiB.
* **Clocks.** Spin-up before the timed region, replay for ≥ `--min-seconds`, and sample
  `clocks.sm` during it (`--clocks`) so an idle-clock run is visible rather than silent.
* **Guard.** `require_idle_gpu()` refuses to run while any compute app holds a context, and
  honours a `MOE_BENCH_MARKER` handover file. Override with `MOE_BENCH_FORCE=1`.

What is already tested on CPU (32/32 passing): argument parsing, all four modes' dispatch,
real-checkpoint tensor loading with `device="cpu"`, every routing generator's invariants
(per-token uniqueness, exact distinct counts, rejection of impossible D), the swizzle
permutation and its 128x4 padding, the e2m1 unpack nibble order, the dequant reference on a
tiny real slice, and the bytes-model identities including the falsification of independence at
T=16. **GPU-only:** `--mode run`, `--mode check`, everything in `harness.py`, and `route_logger`.

---

## 7. Commands to run when the GPU is free

All from `$HOME/tools/flash-next-bench`, with SGLang's venv (it has torch, flashinfer
and sglang; this repo's own `.venv` does not):

```bash
cd $HOME/tools/flash-next-bench
export P=$HOME/tools/sglang-rtxpro6000/.venv/bin/python
export PYTHONPATH=$HOME/tools/flash-next-bench/bench:$HOME/tools/sglang-rtxpro6000/python
export MOE_BENCH_MARKER=/tmp/gpu-handover        # touch this when the GPU is yours
touch $MOE_BENCH_MARKER
nvidia-smi -pl 325                                # re-apply after any reboot
```

**Step 0 — CPU sanity (safe to run right now, GPU untouched):**

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=bench $P -m pytest bench/moe_smallm/tests_cpu.py -q
CUDA_VISIBLE_DEVICES="" PYTHONPATH=bench $P -m moe_smallm.bench_moe --mode model
CUDA_VISIBLE_DEVICES="" PYTHONPATH=bench $P -m moe_smallm.bench_moe --mode dryrun --widths 4,16
```

**Step 1 — correctness against the BF16 dequant reference (~1 min):**

```bash
$P -m moe_smallm.bench_moe --mode check --num-experts 16 --widths 4
# expect max_rel_err at the FP4 activation-quantisation noise floor (a few %), cosine > 0.99
```

**Step 2 — THE measurement: kernel time vs distinct-expert count (~10 min).**
This is what settles the bandwidth question:

```bash
$P -m moe_smallm.bench_moe --mode run --widths 4  --clocks \
   --sweep-distinct 10,15,20,25,30,35,40 --json runs/moe-sweep-w4.json
$P -m moe_smallm.bench_moe --mode run --widths 16 --clocks \
   --sweep-distinct 20,40,60,80,100,120,139,160 --json runs/moe-sweep-w16.json
```

The tool prints a `{"fit": true, ...}` line per width with
`us_per_call = a + b*D`, `streaming_GBps_from_slope` and `pct_peak`. Then read the production D
off the fit: `D_prod = (67.7 - a)/b` at W4 and `(141.3 - a)/b` at W16.

**Step 3 — baseline at the production widths:**

```bash
$P -m moe_smallm.bench_moe --mode run --widths 4,8,16 --clocks --json runs/moe-base.json
```

**Step 4 — knob sweeps (A4):**

```bash
# does the tuned bucket really cover T=16?
$P -m moe_smallm.bench_moe --mode run --candidate flashinfer_direct --widths 16 \
   --tune-max-num-tokens 16   --json runs/moe-tune16.json
$P -m moe_smallm.bench_moe --mode run --candidate flashinfer_direct --widths 16 \
   --tune-max-num-tokens 8192 --json runs/moe-tune8192.json
# cost of the fused-finalize epilogue
$P -m moe_smallm.bench_moe --mode run --candidate flashinfer_direct --widths 4,16 \
   --no-fused-finalize --json runs/moe-nofinalize.json
```

**Step 5 — real routing from a live server (A2b), only if step 2 leaves ambiguity:**

```bash
mkdir -p /tmp/moe-route-hook
cp bench/moe_smallm/sitecustomize_example.py /tmp/moe-route-hook/sitecustomize.py
export PYTHONPATH=/tmp/moe-route-hook:$HOME/tools/flash-next-bench/bench:$PYTHONPATH
export SGLANG_MOE_ROUTE_LOG=/tmp/routes SGLANG_MOE_ROUTE_RING=2048
./serve-fast.sh          # in the sglang fork, W4 then W16
# ... drive a few hundred decode steps with fnbench, then:
kill -USR1 $(pgrep -f 'sglang.*scheduler' | head -1)
$P -m moe_smallm.route_logger --report /tmp/routes.hist.json
# replay the real routing through the kernel:
$P -m moe_smallm.bench_moe --mode run --widths 16 --routing replay \
   --replay /tmp/routes.routes.jsonl --json runs/moe-replay-w16.json
```

**Notes**

* Always run with the desktop compositor stopped (or on a headless X) — see section 4; a
  contaminated run inflates the MoE GEMM by 12-14 %.
* `--rotation` controls the working set; the default 16 gives 0.65-1.3 GB, comfortably past
  4x L2. Lowering it below ~8 will trip the `check_working_set` warning.
* Loading all 512 experts of one layer reads 1.4 GB from disk (~2 min cold on this box) and
  needs 1.4 GB of VRAM. Use `--num-experts N` or `--synthetic` for quick checks.
* The first `--mode run` may trigger FlashInfer JIT compilation of the sm120 grouped-GEMM
  module; that is a one-off few minutes, and it is cached under `~/.cache/flashinfer`.

---

# 8. Measured routing, corrected timing model, and a go/no-go on a custom kernel

Added 2026-09-05 after the GPU sweeps and two instrumented server runs. **Sections 2-5
above were written before D was measured; where they disagree with this section, this
section wins.** The headline correction: the routed-expert GEMM is *not* bandwidth-bound
at these widths, and it is not the "70-88 % of peak" section 2.3 bracketed. It runs at
**69-72 % of peak on the marginal expert** and loses another 8-25 % to **wave
quantisation**, because the real number of distinct experts is far below the independent
prediction.

## 8.1 A harness bug found on the way (read this before trusting any earlier sweep)

FlashInfer looks the CUTLASS tactic up in an **in-process** `AutoTuner` cache. The server
populates it from
`$SGLANG_CACHE_DIR/flashinfer/autotune/<ver>/sm120/<cfg-hash>/rank_tp0_pp0_dp0.json`
(`model_executor/runner/flashinfer_autotune.py:160-172`); a standalone harness starts
empty and silently runs **tactic -1** — the default tile, no `swap_ab`, and no FINALIZE
epilogue, so a separate `finalizeMoeRoutingKernel` appears that the server does not have.

Measured cost of getting this wrong, T=4 at D=28: **117.6 → 96.1 µs per MoE call**
(+1.03 ms/step). `runners.load_autotune_cache()` now loads it and `bench_moe.py`
calls it by default (`--autotune-cache auto|<path>|none`). **The `moe-sweep-w4/w16`
fits of 2026-09-05 morning (`51.1 + 2.92·D`, `57.3 + 2.73·D`) were taken without it and
should be discarded.** The two profiles are also distinguishable by eye: with the cache
loaded there is no `finalizeMoeRoutingKernel` line, matching the server trace.

The right cache per profile (both already on this box):
`c200dfb50995faa8` = W4 (buckets M=1,2,4,32), `990c98ce8356ecfe` = W16 (M=1,2,4,8,16,32).
Tactics: **gemm1 17 at M≤4 / 16 at M≥8; gemm2 57** — exactly the tile shapes read out of
the server trace in section 1.3.

## 8.2 Measured distinct-expert counts (task 1)

`route_logger.py` armed via the `sitecustomize` shim, cumulative histograms snapshotted
between workloads, per-workload figures by differencing (`analyze_routes.py`). Two server
runs (`serve-fast.sh w4`, `w16`), fnbench `--repeats 1 --sampling greedy` on the four
workloads. Raw data: `runs/routes-{w4,w16}-{baseline,<workload>}.json`,
`runs/routes-{w4,w16}-ring.jsonl`.

Columns `x1/x2/>=3/xT` are the mean number of experts reached by exactly 1, exactly 2,
3-or-more, and by **all** T tokens of the verify chain — the overlap structure.

**W4 verify (T=4), 197 470 calls. Independent routing would give E[D] = 38.84.**

| workload | calls | mean | p50 | p90 | min | max | x1 | x2 | ≥3 | all-4 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| code-edit | 63 504 | **30.95** | 31 | 37 | 12 | 40 | 24.23 | 4.89 | 1.83 | 0.50 |
| prose-en | 46 501 | **27.41** | 27 | 35 | 10 | 40 | 18.91 | 5.44 | 3.05 | 1.04 |
| agent-loop | 36 456 | **29.00** | 29 | 36 | 11 | 40 | 21.15 | 5.39 | 2.46 | 0.70 |
| prose-ja | 51 009 | **25.73** | 25 | 33 | 10 | 40 | 16.78 | 5.19 | 3.76 | 1.56 |
| **ALL** | 197 470 | **28.41** | 28 | 36 | 10 | 40 | 20.49 | 5.19 | 2.73 | 0.94 |

**W16 verify (T=16), 135 534 calls. Independent routing would give E[D] = 138.57.**

| workload | calls | mean | p50 | p90 | min | max | x1 | x2 | ≥3 | all-16 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| code-edit | 26 215 | **75.88** | 74 | 101 | 27 | 144 | 40.99 | 16.34 | 18.55 | 0.03 |
| prose-en | 38 269 | **68.53** | 67 | 91 | 21 | 140 | 33.64 | 15.58 | 19.30 | 0.05 |
| agent-loop | 25 529 | **72.68** | 71 | 96 | 20 | 141 | 37.55 | 16.13 | 19.00 | 0.03 |
| prose-ja | 45 521 | **64.53** | 63 | 88 | 19 | 141 | 31.72 | 13.97 | 18.84 | 0.28 |
| **ALL** | 135 534 | **69.39** | 68 | 94 | 19 | 144 | 35.15 | 15.29 | 18.94 | 0.12 |

Draft calls (T=1) are exactly 10 distinct experts, always, as they must be.

What this says:

* **D is 73 % of independent at T=4 and 50 % at T=16.** Section 2.4's bound
  (`D16 ≤ 91.6`) was correct and not tight; the measured 69.4 is well inside it.
* **`D16/D4 = 2.44`**, against a kernel-time ratio of 2.09 — so the per-expert cost is
  *lower* at W16 than at W4 (better wave filling, section 8.3), not higher.
* **The correlation is diffuse, not a shared core.** At T=16 only **0.12 experts** are
  common to all 16 tokens, and half the touched experts (35 of 69) are reached by a single
  token. The "hot expert" model of section 3 is wrong; what actually happens is that each
  token's top-10 is drawn from a *narrow, context-dependent* subset, so the union grows
  sub-linearly without any expert being universal.
* Workload ordering is stable and matches acceptance: code-edit (most tokens accepted,
  most diverse chain) has the highest D, prose-ja the lowest, spread ±10 %.
* **D does not vary by layer.** From `runs/routes-w16-ring.jsonl` (last 4096 calls, layer
  identified by call index modulo the 64 MoE calls per step), the per-layer mean ranges
  54.7-60.1 with no trend — so a per-layer optimisation has nothing to exploit.

## 8.3 Corrected timing model (task 2a)

Per-kernel profiles with the production tactic cache (`--mode profile`), µs per MoE call,
at the ~2300 MHz the card sustains under a 100 %-duty MoE loop at its 325 W cap:

| T | D | GEMM1 | GEMM2 | GEMM total | glue (6 kernels) | total |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 10 | 23.85 | 19.63 | 43.48 | 21.33 | 65.79 |
| 4 | 20 | 38.09 | 21.18 | 59.27 | 21.54 | 80.81 |
| 4 | **28** | 42.45 | 23.85 | **66.30** | 21.77 | 88.07 |
| 4 | 40 | 74.67 | 34.41 | 109.08 | 21.07 | 130.15 |
| 16 | 20 | 38.48 | 22.05 | 60.53 | 19.83 | 80.36 |
| 16 | 40 | 62.22 | 36.08 | 98.30 | 19.53 | 117.83 |
| 16 | **69** | 112.27 | 56.58 | **168.85** | 20.68 | 189.53 |
| 16 | 100 | 152.27 | 81.75 | 234.02 | 24.00 | 258.02 |
| 16 | 139 | 212.75 | 112.01 | 324.76 | 22.51 | 347.27 |

Least squares:

```
GEMM(T=4)  =  17.5 + 2.124 * D  us      marginal 1302 GB/s  (73 % of 1792 peak)
GEMM(T=16) =  13.0 + 2.233 * D  us      marginal 1238 GB/s  (69 % of peak)
glue       =  21.5 us, flat in D and in T
```

**So the 51-57 µs "latency-bound intercept" is mostly the six glue kernels (21.5 µs),
not GEMM launch latency (13-18 µs for the two GEMMs together).** The rest of the earlier
intercept was the missing tactic cache. And the marginal bandwidth is 1240-1300 GB/s,
not 946-1013.

Cross-check against the server at the measured D:

| | bench model | server trace (robust) | server trace (mean) |
|---|--:|--:|--:|
| W4 GEMM, D=28.4 | 68.0 µs/layer | **67.7** | 78.4 |
| W16 GEMM, D=69.4 | 168.0 µs/layer | **141.3** | 164.6 |

W4 agrees to 0.5 %. W16 does not: the bench is 19 % above the trace's robust number and
2 % above its mean. Two unresolved candidates, both worth one measurement before anyone
optimises against a W16 number: (i) the trace is `prof-base32k-w16`, i.e. the 32768-entry
draft token map, while the routing run used the default `hot2_49152` — a different draft
map proposes different tokens, so the verify chain and therefore D differ; (ii) SM clock,
though this is weakened by W4 agreeing exactly. **Treat the server trace as authoritative
for absolute ms/step and the bench for structure.**

### Where the D-independent time goes, and the CTA/wave arithmetic

Tiles per expert, with `swap_ab` (which puts the ≤16 tokens on the 32-wide N axis, so
that axis is always one tile, and the 128-wide M axis walks the weight's output dim):

| | tile (M,N,K) | tiles / expert | K-loop trips | weight bytes / tile |
|---|---|--:|--:|--:|
| GEMM1 gate_up (N=1280, K=2560) | 128x32x256 (T=16) / x128 (T=4) | **10** | 10 / 20 | 184 320 |
| GEMM2 down (N=2560, K=640) | 128x32x128 | **20** | 5 | 46 080 |

10 x 184 320 = 1 843 200 B and 20 x 46 080 = 921 600 B, i.e. exactly the per-expert weight
bytes of section 2.1 — the tiling reads each expert's weights once, nothing is re-read.

The kernel is persistent with **grid = [1, 188, 1]** = one CTA per SM, so it processes
`ceil(tiles / 188)` waves:

| | tiles | waves | last-wave fill | wasted µs/call |
|---|--:|--:|--:|--:|
| W4 D=28, GEMM1 | 280 | 2 | 74.5 % | 10.8 |
| W4 D=28, GEMM2 | 560 | 3 | 99.3 % | 0.2 |
| W16 D=69, GEMM1 | 690 | 4 | 91.8 % | 9.2 |
| W16 D=69, GEMM2 | 1380 | 8 | 91.8 % | 4.6 |
| draft D=10, GEMM1 | 100 | 1 | **53 %** | — |

**Wave quantisation costs 11.0 µs/call at W4 (0.53 ms/step) and 13.8 µs/call at W16
(0.66 ms/step).** Inside a full wave the kernel streams at 188 x 184 320 B / ~22 µs =
**~1575 GB/s = 88 % of peak** — the kernel body is fine; it is the tail that is not.

Fraction of the D-independent term that is the two launches plus the GEMM1→GEMM2
dependency: the GEMM intercept is 13-18 µs for *both* GEMMs, i.e. ~7-9 µs each, which is
the persistent kernel's ramp (TMA descriptor fetch + filling a 3-stage pipeline over a
10-20 trip K loop) plus drain. The glue's 21.5 µs is six strictly dependent launches at
1.5-7 µs each, of which the 3-kernel routing prologue is 8.4 µs and `computeStrides`
2.9-4.5 µs. **Everything in the intercept is serial dependency, not per-CTA latency.**

## 8.4 A split-K, expert-parallel NVFP4 grouped GEMM (task 2b)

### Sketch

* **Grid** `(D, N_tiles, K_splits)` instead of a 188-CTA persistent loop. `N_tiles` = 10
  (gemm1) / 20 (gemm2) as today; `K_splits = clamp(round(188·w / (D·N_tiles)), 1, 8)`
  chosen on the host from the already-computed expert count, so the grid is a whole
  number of waves.
* **Inner loop**: an ALU bit-trick NVFP4 dequant (the technique credited in `N1_LOG.md`), not a
  16-entry LUT, over `int32`-wide loads, accumulating in FP32.
* **Structure grouped, not route-parallel**: one program per (expert, N-tile, K-split),
  reading `expert_first_token_offset` from the existing prologue, so each distinct expert's
  weights are read once (`D` times per layer) rather than once per route (`T·10` times).
* **Reduction**: FP32 partials, `[D·rows, N]`, at most 69 x 16 x 2560 x 4 B = 11 MB;
  a deterministic tree-reduce kernel costs ~2-3 µs, or non-deterministic `atomicAdd`.
* **Fusing silu*up between the GEMMs is not possible in this layout.** An N-tile of 128
  over the 1280 gate_up columns holds either gate (cols 0-639) or up (cols 640-1279),
  never the pair. It becomes possible only if gate/up are **interleaved** so that
  `(g_i, u_i)` are adjacent — then a 128-wide tile covers 64 channels' gate *and* up and
  the epilogue can emit `silu(g)*u` directly as FP4 + block scale, deleting
  `doActivationKernel`. SGLang already has the repack machinery
  (`deinterleave_w13`, `modelopt_quant.py:2594-2606`), so this is a checkpoint-layout
  change, not new math. Worth **−0.21 ms/step (W4) / −0.31 (W16)** on its own.

### Estimated time

`K_splits` only changes the wave count, so time scales as `ceil(tiles·s/188)/s`:

| | today | best `s` | ratio | GEMM saving |
|---|--:|--:|--:|--:|
| W4 D=28 GEMM1 | 2 waves | s=2 → 3/2 | 0.75 | −10.6 µs/call |
| W4 D=28 GEMM2 | 3 waves | s=1 | 1.00 | 0 |
| W16 D=69 GEMM1 | 4 waves | s=3 → 11/3 | 0.92 | −9.3 µs/call |
| W16 D=69 GEMM2 | 8 waves | s=2 → 15/2 | 0.94 | −3.5 µs/call |

**Ceiling if a custom kernel matched CUTLASS's in-wave streaming rate and fixed only the
tail: −0.51 ms/step at W4, −0.62 ms/step at W16.**

But it will not match that rate:

* If a Triton kernel on this box streamed at no more than about 1150 GB/s (an assumption: no
  Triton NVFP4 grouped kernel was measured in this project), the same
  work costs **77.4 µs at W4 D=28** and **190.8 µs at W16 D=69** — *worse* than CUTLASS's
  66.3 and 168.9. It only breaks even at ~1300 GB/s, which is the marginal rate CUTLASS
  already achieves.
* At W16 a non-tensor-core kernel is **compute**-bound, not bandwidth-bound: 16 rows x
  2 FLOP per weight = 57 FLOP per weight byte, so sustaining 1500 GB/s needs 85 TFLOP/s
  of FP32 FMA against a ~125 TFLOP/s peak. Reaching the target would require `tl.dot` on
  dequantised BF16, i.e. re-implementing the CUTLASS mainloop in Triton without TMA.
  (At W4, 14 FLOP/byte, this is not a constraint — a gemv is viable *only* at W4.)

## 8.5 Can CUTLASS be coaxed into smaller tiles or split-K on sm120? (task 2c)

Re-checked directly in the installed FlashInfer 0.6.17 source. **No, on both counts.**

* `get_candidate_configs_sm120` (`nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:595-637`)
  returns a flat 10-entry list, every one with `ClusterShape_1x1x1`,
  `MainloopScheduleType::AUTO`, `EpilogueScheduleType::AUTO`:
  `128x128x128B, 128x128x64B, 256x128x64B, 128x256x64B, 128x128x256B, 256x128x128B,
  128x32x128B, 128x32x64B, 128x64x128B, 128x64x64B`.
  **The smallest M is 128 and the smallest N is 32** — the tile the autotuner already
  picks. There is no 64-, 16- or 8-row M tile for sm120 at all (`CtaShape128x8x256B` /
  `128x16x128B` exist only in the SM100 list, and only for FP8).
* **Split-K is unreachable.** `SplitKStyle::SPLIT_K_SERIAL` is only generated in the
  legacy pre-Hopper branch of `get_candidate_configs` (`:671-685`); `sm >= 120` returns
  from `get_candidate_configs_sm120` before reaching it, and both MoE dispatch sites
  hard-code `int const max_split_k = 1;`
  (`moe_gemm_template_dispatch.h:565` and `:593`). Stream-K exists in FlashInfer only in
  the *dense* FP4 GEMM, not the grouped one.
* **The sm90/sm100 paths do not compile for sm120.** Dispatch is by SM value
  (`cutlass_heuristic.cpp:653-660`), and the sm120 grouped kernel is built from
  `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` +
  `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120` with the
  `SM120::BLOCKSCALED::SM120_16x8x64_TN_VS` MMA atom. The sm100 collectives use
  `tcgen05` instructions that consumer/workstation Blackwell (sm_120) does not implement,
  and sm90 uses `SM90_64x*` wgmma, also absent. Forcing either would not compile, let
  alone run.
* The only sm120 knob left is `swap_ab`, and the autotuner already selects it.

**Therefore the wave-quantisation loss cannot be recovered inside CUTLASS.** It can only
be recovered by a custom kernel, and section 8.4 says that kernel would have to beat
1300 GB/s of NVFP4 streaming in Triton to break even.

## 8.6 Go / no-go

**NO-GO on the custom expert-parallel split-K NVFP4 kernel.** The ceiling is
−0.51 ms/step (W4) and −0.62 ms/step (W16) — 4 % and 3 % of the step — and it is reachable
only if a hand-written Triton kernel matches a TMA + tensor-core CUTLASS mainloop at
≥1300 GB/s, which no measurement in this project has shown for a Triton kernel on sm_120. Estimated cost is
2-3 weeks with a real chance of landing *slower* than today. Revisit only if someone
independently demonstrates >1300 GB/s NVFP4 grouped streaming in Triton on sm_120.

**GO on the three cheap, structural items**, none of which touch the GEMM:

| item | W4 | W16 | effort | risk |
|---|--:|--:|---|---|
| **A0** enable FlashInfer's fused routing prologue (top-k=10 absent from the `{1,2,4,6,8}` switch; 512 experts give `expert_log = 10` vs a `≤9` gate) — section 5/A0 | **−0.23** | **−0.29** | ~5 lines in the FlashInfer csrc + JIT rebuild | low-med |
| **A3** fold `computeStridesTmaWarpSpecialized` into that prologue | −0.13 | −0.18 | same patch site | med |
| **A8** interleave gate/up so the GEMM1 epilogue can emit `silu(g)*u` as FP4, deleting `doActivationKernel` | −0.21 | −0.31 | checkpoint repack + CUTLASS epilogue | med-high |
| **total** | **−0.57** | **−0.78** | | |
| **A1** run headless (this is measurement contamination, and possibly production too) | −0.46 | −1.13 | ops only | none |

Against a 11.95 ms (W4) / 20.47 ms (W16) step, A0+A3+A8 is **−4.8 % / −3.8 %**, and they
are additive with A1. That is the same order as the custom kernel's ceiling at a fraction
of the risk, which is the whole argument.

**Do not pursue** further D-reduction (section 5/A5) on the strength of these numbers
either: at 28.4 (W4) and 69.4 (W16) the model is *already* routing far more narrowly than
independent selection, so the remaining headroom before quality damage is small — and the
overlap profile shows there is no shared core of experts to exploit.

## 8.7 Reproducing section 8

```bash
cd $HOME/tools/flash-next-bench
# routing (per profile): arms route_logger via sitecustomize, snapshots between workloads
mkdir -p /tmp/moe-route-hook
cp bench/moe_smallm/sitecustomize_example.py /tmp/moe-route-hook/sitecustomize.py
export PYTHONPATH=/tmp/moe-route-hook:$PWD/bench
export SGLANG_MOE_ROUTE_LOG=/tmp/routes-w16 SGLANG_MOE_ROUTE_RING=4096
# ... start serve-fast.sh w16, run fnbench per workload, kill -USR1 <scheduler pid> between them
.venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
  --workloads code-edit --repeats 1 --sampling greedy --allow-proc sglang \
  --label routelog-w16-code-edit --out runs/routelog-w16-code-edit.jsonl
PYTHONPATH=bench python -m moe_smallm.analyze_routes --profile w16
PYTHONPATH=bench python -m moe_smallm.analyze_routes --ring runs/routes-w16-ring.jsonl \
  --T 16 --calls-per-step 64

# per-kernel timing model (GPU; ALWAYS pass the matching autotune cache)
C16=~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/990c98ce8356ecfe/rank_tp0_pp0_dp0.json
python -m moe_smallm.bench_moe --mode profile --widths 16 \
  --sweep-distinct 20,40,69,100,139 --rotation 16 --autotune-cache "$C16"
```
