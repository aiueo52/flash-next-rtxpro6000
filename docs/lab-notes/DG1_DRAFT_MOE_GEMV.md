# DG1: draft MoE one-token GEMV (W4A16 NVFP4, Triton)

Status (2026-10-01 11:55 JST): implemented on `opus/draft-gemv` in `~/tools/sglang-dg` (base 7b4d539f9b), commit
`5155fc2354`. GPU correctness and microbench on the real mtpft5 MTP layer pass (§3, §4; `runs/dg1/job1.*`).
In-server smoke: CLEAN (§6.1, 12:19-12:23). **The server A/B is queued** (`bench/stack/chain_late.sh`, after the RT1 ABBA).

## 0. Summary

- **What:** the draft (MTP) model's one-token MoE call (T=1, top-10 of 512 experts) runs two Triton kernels
  instead of FlashInfer's CUTLASS NVFP4 chain (routing prologue, FP4 GEMM1, doActivation, FP4 GEMM2).
- **Switch:** `SGLANG_OPT_DRAFT_MOE_GEMV=1` (EnvBool, default off). Only NEXTN (draft) MoE layers opt in.
- **Per call** (CUDA-graph replay, weights streamed from DRAM):
  - 49.4 → 29.8 µs back to back (−19.6 µs, −40%);
  - 54.5 → 30.4 µs after an L2 flush (−24.1 µs), the case that matches in-server CUTLASS timings (§4.1).
- **Per verify step:**
  - −0.27 to −0.34 ms at 14 one-token calls (15 draft steps: W16, wa high);
  - −0.12 to −0.15 ms at 6 calls, −0.04 to −0.05 ms at 2 (W4);
  - FC's −0.38 ms estimate assumed the prototype's 23.2 µs per call (§4.2).
- **Output:** target kernels are unchanged (T=1 occurs only in draft calls). The draft's activations stay bf16
  (W4A16) instead of FP4 (W4A4), so drafts, and with them tok/step, change.
  - The GEMV matches an fp32 model of its own arithmetic to 1.4e-4 mean rel L2.
  - Its gap to CUTLASS (0.161 mean) equals that of the W4A4 model: it comes from CUTLASS's FP4 activations (§3).
  - Greedy text can still differ between arms: the verify batch holds the draft tokens, and P2's singleton
    prune of the target MoE depends on the other rows of that batch (pre-existing, not new to DG1).

## 1. Design (`python/sglang/srt/layers/moe/draft_moe_gemv.py`)

| kernel | grid (DEFAULT) | work | bytes/call |
|---|---|---|--:|
| k1 `_draft_moe_up_gate_kernel` (:125) | (I/8, R) = (80, 10) | up and gate rows of one route: fp32 dot with x; act = silu(g·gate)·(g·up), stored bf16 [R, I] | 18.4 MB |
| k2 `_draft_moe_down_kernel` (:172) | (H/16, S2) = (160, 10) | down rows of routes r ≡ s (mod S2), × w_r·g2_r, fp32 partials; the last-arriving split sums the S2 partials in fixed order, bf16 [1, H] | 9.2 MB |

- Dequant: `_unpack_dequant` (w4a16_nvfp4_gemv.py:71, N1's shipped lm_head GEMV) places the e2m1 bits into an
  fp16 (value × 2^-14), times the e4m3 block scale in bf16 (exact: at most 6 significant bits); the global
  scale × 2^14 (`_FP4_TRICK`, :57) is applied once per row sum. bf16 weight × fp32 x, fp32 sums.
- Scales are read in place from the 128×4 swizzled layout: a CTA takes BN/4 adjacent rows from each 32-row
  group of one 128-row tile, so each 16 B scale word (4 rows × 4 blocks) is read whole by one CTA
  (`_swizzled_rows` :95, `_swizzled_scales` :114).
- Gone: routing prologue (expert maps, expand, FP4 quant of x), doActivation (FP4 requant of act), finalize.
  topk_ids/topk_weights are read directly; ids < 0 are skipped.
- Workspace per layer (`DraftMoeWorkspace` :85, about 26 KB): act [R, I] bf16, partials [H·S2] fp32,
  counters [H/BN2] int32. The last arriver resets its counter to 0, so graph replays need no memset.
- Allocated on the first eligible call; if that call is inside a CUDA-graph capture, the layer keeps CUTLASS
  for that graph and logs one warning (allocating in the graph pool is avoided).
- PDL: both kernels `pdl_wait` before the first dependent load and `pdl_trigger` at the end; `launch_pdl`
  follows `SGLANG_TRITON_PDL` (production 1).
- Deterministic: the split sum has a fixed order; reruns and PDL on/off are bitwise equal (§3).
- `DEFAULT_CONFIG` (:72): k1 BN 8, BK 512, 4 warps, 2 stages; k2 BN 16, BK 128, S2 10, 2 warps, 3 stages (§4.3).

## 2. Layout facts (verified)

| fact | source (worktree `python/sglang/srt/`, FlashInfer 0.6.17 csrc) | GEMV |
|---|---|---|
| w13 rows [0, I) = up, [I, 2I) = gate | modelopt_quant.py:2925-2930 (`load_up_proj_weight_first` for flashinfer_cutlass); fused_moe_triton/layer.py:653-658 (w1 goes to start = I) | q_up = row, q_gate = row + I |
| act = silu(gate)·up, gate = second half | cutlass_fused_moe_kernels.cuh:2911-2912 (fc1_value at +inter_size), :2920-2926 (linear = first half), GLUAdaptor :2650-2661 | same |
| e2m1: low nibble = even k | w4a16_nvfp4_gemv.py:71-87 | same helper |
| 128×4 swizzle: off(m, kb) = (m/128)·128·KP + (kb/4)·512 + (m%32)·16 + ((m%128)/32)·4 + kb%4 | quantization/utils.py:616-617; applied for CUTLASS at modelopt_quant.py:2829-2835 | `_swizzled_rows`, `_swizzled_scales` |
| no swizzle padding at H=2560, I=640 | utils.py:608-612 (M to 128, K/16 to 4) | `unsupported_reason` (:232) refuses padded shapes |
| ws2 = amax/(448·6) decode scale; gate and up quantized together (one ws2) | nvfp4_online.py:294-322, :549-551 | g = w13_weight_scale_2[e, 0] ([E] or [E, 2], by stride) |
| CUTLASS GEMM1 uses the gate column for both halves; alphas = input_scale·ws2, x quant = 1/input_scale | modelopt_quant.py:2346-2354, :2683-2706 | input scale cancels; x never quantized |
| draft input scale = 1.0 (modelopt_fp4 online) | modelopt_quant.py:2576-2596 | n/a |
| router weights applied once per route; fp4 path applies no routed_scaling_factor | flashinfer_cutlass.py:250-252; modelopt_quant.py:3133 | w_r in k2 |
| T=1 is never pruned (P2 min_rows 2; P1 off under P2) | cuh:1878, :4565; prune_singleton.py:124; serve-fast.sh:56-57 | no pruning, same routes |
| draft MoE = NEXTN layer | qwen4_exp_mtp.py:157 → qwen3_5.py:1818 → :1381-1391 → qwen2_moe.py:431 | hook |
| draft MoE backend = target's unless `--speculative-moe-runner-backend`; AUTO + modelopt_fp4 + SM120 → flashinfer_cutlass | moe/utils.py:328-331, :639-650; arg_groups/overrides.py:2822-2831 | hook sits in the CUTLASS branch |
| graph capture runs 2 eager warm-ups first | full_cuda_graph_backend.py:105-111; breakable_cuda_graph_backend.py:116-121 | workspace allocated eagerly |

## 3. Correctness (GPU, job1)

Setup (`bench/dg1/dg1_bench.py`; results `runs/dg1/job1.json`, gitignored):
- Weights: the mtpft5 MTP MoE layer (`mtp.layers.0.mlp.experts`, bf16 checkpoint), quantized like the server's
  online path and stored in the CUTLASS format: w13 [up; gate], swizzled block scales (`bench/dg1/layer.py`).
- Rows: all 3626 one-token routings of the W16 code-edit census (recorded ids and weights); x random bf16.
- fp32 references per row (`layer.py`):
  - dq: dequantized weights, fp32 activations;
  - emulation: dq with act and output rounded to bf16 (the GEMV's arithmetic up to fp32 summation order);
  - a4: dq with x and act fake-quantized to NVFP4 (CUTLASS's W4A4 math; a model, not bit-exact);
  - bf: the original bf16 weights.

| pair (rel L2 per row) | max | mean | p99 |
|---|--:|--:|--:|
| GEMV vs emulation | 1.7e-3 | 1.4e-4 | 8.2e-4 |
| GEMV vs dq | 2.8e-3 | 2.3e-3 | 2.5e-3 |
| CUTLASS vs a4 | 0.106 | 0.050 | 0.090 |
| CUTLASS vs dq | 0.191 | 0.161 | 0.180 |
| a4 vs dq | 0.193 | 0.161 | 0.180 |
| GEMV vs CUTLASS | 0.192 | 0.161 | 0.179 |
| GEMV vs bf | 0.189 | 0.162 | 0.178 |
| CUTLASS vs bf | 0.263 | 0.227 | 0.251 |

- GEMV vs emulation: 99.04% of the 9.28 M output elements are bitwise equal; 0.12% differ by more than 1 bf16 ulp.
- GEMV vs CUTLASS (0.161) equals a4 vs dq (0.161): the gap is CUTLASS's FP4 activations, not the weight decode.
- Against the bf16 checkpoint: GEMV 0.162 (weight quantization only); CUTLASS 0.227 ≈ √(0.162² + 0.161²)
  (weights and activations).
- All outputs finite; topk ids and weights unchanged; split counters back at 0.
- Determinism (256 rows): a GEMV rerun and a PDL-off run are bitwise equal to the first run. A CUTLASS rerun
  differs in 256/256 rows (rel max 5.7e-3): today's draft MoE is not run-to-run deterministic.

Dispatch, through the real `ModelOptNvFp4FusedMoEMethod.apply` (CUTLASS branch) with a runner attached:

| case | result |
|---|---|
| eager, 64 one-token calls | 64 GEMV calls, all bitwise equal to the direct GEMV |
| CUDA graph of 4 calls (eager warm-up first), 50 replays on new inputs | 4 GEMV calls captured; 200/200 rows bitwise equal; counters 0 |
| first one-token call inside a capture | 0 GEMV calls, no workspace, one warning ("layer 0 keeps CUTLASS for this graph"); output is CUTLASS's (rel 5.0e-3 to an eager CUTLASS run, its run-to-run noise) |
| two-token call | CUTLASS (0 GEMV calls), finite |
| `enable_draft_moe_gemv` | silu layer accepted; gelu layer refused ("needs gated silu"); non-FusedMoE module returns False |

CPU (Triton interpreter, `bench/dg1/selftest_cpu.py`, 12-expert slice, `runs/dg1/selftest_cpu.log`): 3 tile configs,
including S2 = 1 and an id of -1, match the emulation with the interpreter's truncating casts to at most 4.4e-4
rel L2 (max 5 ulp; at most 4 of 2560 elements above 1 ulp); counters 0.

## 4. Timing (GPU, job1)

Method (`dg1_bench.py`, phase `time`):
- One CUDA graph of 48 one-token calls, each on its own 10 experts (480 of 512). Every call streams its 27.65 MB
  from DRAM (all experts 1.4 GB, L2 128 MB).
- "After flush": a 256 MB read precedes each call; the flush-only graph's time is subtracted. The CUTLASS kernels
  then match the in-server FC traces (§4.1, last column).
- Per call = replay time / 48, median of 3 rounds. Kernel times: medians over 240 calls in a profiler trace.
- `SGLANG_TRITON_PDL=1` (production); triton 3.7.1, FlashInfer 0.6.17, torch 2.13.0+cu130; GPU otherwise idle.

### 4.1 Per call, T=1 (µs)

| | CUTLASS chain | GEMV (DEFAULT) | saved |
|---|--:|--:|--:|
| per call, back to back | 49.38 | 29.75 | **19.63 (−40%)** |
| per call, after L2 flush | 54.48 | 30.37 | **24.11 (−44%)** |
| kernel span, back to back | 43.79 | 25.50 | 18.29 |
| kernel span, after flush | 49.46 | 28.19 | 21.27 |
| bandwidth (27.65 MB per call, back to back) | 560 GB/s | 929 GB/s | |

| kernel (µs) | back to back | after flush | in-server (FC traces) |
|---|--:|--:|--:|
| CUTLASS prologue (`fusedBuildExpertMapsSortFirstTokenAndStrides`) | 7.49 | 10.78 | 11.5-12.3 |
| CUTLASS GEMM1 (FP4) | 17.34 | 17.70 | 17.7-18.5 |
| CUTLASS doActivation | 3.49 | 3.58 | 3.5 |
| CUTLASS GEMM2 (FP4; finalize fused, no separate kernel) | 16.54 | 16.96 | 17.0 |
| GEMV k1 (up and gate, 18.4 MB) | 16.03 | 16.35 | |
| GEMV k2 (down and route sum, 9.2 MB) | 9.54 | 9.60 | |

- PDL off: 29.97 µs per call (PDL saves 0.2 µs).
- k1 alone: 18.67 µs per call (kernel 16.03 µs, 1149 GB/s). The FC prototype's k1-only graph gave 13.8 µs per call.
  - Prototype differences: linear scales, random codes, contiguous rows per CTA, no PDL, 7 short replays.
  - Not isolated. If the gap is in the kernel, 2-5 µs per call (0.03-0.07 ms per 15-step verify) remain available.
- Per call exceeds the kernel span by 2-6 µs in both paths (launch gaps; kernel times come from a profiled run).

### 4.2 Per verify step

One-token draft calls per verify step = draft steps − 1 (census: 13.98 at 15 steps).

| one-token calls | draft steps | back to back | after flush |
|--:|---|--:|--:|
| 2 | 3 (W4; wa low) | −0.039 ms | −0.048 ms |
| 6 | 7 (wa mid) | −0.118 ms | −0.145 ms |
| 13.98 | 15 (W16 code-edit census) | −0.274 ms | −0.337 ms |
| 14 | 15 (W16; wa high) | −0.275 ms | −0.338 ms |

- In-server CUTLASS costs about 50 µs (FC traces), the after-flush case: expect about −0.33 ms per 15-step verify.
  wa's mean saving depends on its 3/7/15 mix.
- FC's −0.38 ms estimate assumed the prototype's 23.2 µs per call. k1 accounts for 4.9 of the 6.5 µs
  difference (§4.1).

### 4.3 Tile sweep (`--sweep`; back to back, shorter windows, about 0.7 µs above §4.1)

- k1 alone, 36 configs (BN 4/8/16 × BK 256/512 × 2/4/8 warps × 2/3 stages):
  - best (16, 512, 8 w, 2 s) 18.91 µs;
  - DEFAULT (8, 512, 4 w, 2 s) 19.14 µs, 5th;
  - worst 33.6 µs.
- k2 alone, 24 configs (BN 8/16/32 × BK 64/128 × S2 5/10 × 2/4 warps, 3 stages):
  - DEFAULT (16, 128, S2 10, 2 w) is best at 11.23 µs; the next is 11.71;
  - worst 31.0 µs.
- Joint top 3 × top 3: best is k1 (8, 512, 4 w, 3 s) with DEFAULT k2, at 30.41 µs.
- Best vs DEFAULT, alternated 3 times: 30.55 vs 30.60 µs median (−0.05 µs, below noise), so **DEFAULT is kept**.
  The best config's output equals DEFAULT's bitwise (same tiles, same summation order).

## 5. Dispatch change (worktree, 4 files)

- `environ.py:1120-1123`: `SGLANG_OPT_DRAFT_MOE_GEMV = EnvBool(False)`.
- `models/qwen2_moe.py:431-434`: `if is_nextn and envs.SGLANG_OPT_DRAFT_MOE_GEMV.get(): enable_draft_moe_gemv(self.experts)`.
- `enable_draft_moe_gemv` (:427) attaches a `DraftMoeGemvRunner` to the layer's quant method only if
  - the layer is a FusedMoE whose quant method is `ModelOptNvFp4FusedMoEMethod` (the online subclass included);
  - EP = TP = 1, gated silu, router weights on the output, no alpha/beta/clamp/swiglu limit;
  - the shapes need no swizzle padding.
  Otherwise it logs `SGLANG_OPT_DRAFT_MOE_GEMV ignored for layer N: <reason>`; on success
  `draft MoE GEMV enabled for layer N`.
- `modelopt_quant.py:2383-2384` (runner slot, default None) and `:3106-3112`: in the FlashInfer CUTLASS branch of
  `apply`, `maybe_apply` returns a `StandardCombineInput`, or None to take CUTLASS. It returns None unless:
  - standard dispatch, exactly 1 token, bf16, contiguous, no pre-quantized input (`hidden_states_scale` None);
  - contiguous topk ids and weights.
- Target layers, T > 1 draft calls (draft extend, batch > 1) and every other backend are untouched.

## 6. Server A/B (scripts ready; smoke run, ABBA queued)

Setup once (copies only; production files and the shared FlashInfer caches stay untouched):

```bash
WT=~/tools/sglang-dg; P=~/tools/sglang-rtxpro6000
cp $P/serve-fast.sh $WT/
sed "s|\$REPO/.venv/bin|$P/.venv/bin|" $P/serve-local.sh > $WT/serve-local.sh && chmod +x $WT/serve-local.sh  # lines 9, 40
mkdir -p $WT/.cache && cp -a $P/.cache/{jit,triton,sglang,xdg} $WT/.cache/   # ~2.3 GB, almost all triton
bash ~/tools/flash-next-bench/bench/jitcache/jitprobe.sh $WT                 # CPU; must report no work
```

| item | choice |
|---|---|
| arms | A = off (CUTLASS draft MoE), B = `SGLANG_OPT_DRAFT_MOE_GEMV=1`; same worktree and commit, `serve-fast.sh wa` |
| order | ABBA: `bench/dg1/abba_dg.sh` (A1 B1 B2 A2), each arm a fresh server in its own GPU-lock hold (`arm_dg.sh`, MemoryMax=110G, memcheck ≥ 110G) |
| grid | code-edit, prose-en, prose-ja, agent-loop × greedy and sampling (LM-Studio parameters) × N=4 requests × 600 tokens (`prof/fc_sampling_probe.py`) |
| primary metric | ms per output token per request = ms/step ÷ tok/step (the draft change moves both) |
| secondary | ms/step (expected change: §4.2) and tok/step (acceptance) |
| evidence the path ran | server log: `draft MoE GEMV enabled for layer N` (1 line: one MTP MoE layer) and 0 fallback warnings; 20-step trace (`dg_trace.py`): k1 = k2 count = one-token draft calls, CUTLASS prologue count down by the same |
| table | `bench/dg1/dg_table.py A1 B1 B2 A2`: per workload × mode change with bootstrap CI, block ratios, drift (A2/A1), pooled |

Decision rule. Ship (turn the flag on in serve-fast.sh) only if all of these hold:

1. pooled greedy ms/token is lower with the 95% CI excluding 0, and pooled sampling ms/token is not worse
   (CI upper bound < +0.5%);
2. pooled ms/step falls by roughly the §4.2 saving times the mean number of one-token calls per step;
3. no workload × mode loses more than 2% tok/step with a CI excluding 0;
4. the evidence row above holds in both B arms and the A arms show no GEMV kernels;
5. drift |A2/A1 − 1| is smaller than the measured effect; otherwise rerun the ABBA.

Effort: about 1.5 h of GPU lock (4 server starts of 8-12 min plus the requests).

### 6.1 In-server smoke (`bench/dg1/smoke_dg.sh`, 12:19-12:23, `runs/dg1/smoke-dg.log`)

- `SGLANG_OPT_DRAFT_MOE_GEMV=1`, `serve-fast.sh wa`, worktree commit 5155fc2354: server up, 0 Traceback/Error lines,
  `draft MoE GEMV enabled` 1 line (the one MTP MoE layer), 0 fallback warnings.
- Profiled 20-step greedy code-edit trace: k1 and k2 120 calls each (6 one-token draft calls per step, so the
  request ran at 7 draft steps), CUTLASS prologue 980 calls (49 per step, the verify layers only).
- In-server kernel medians: k1 15.92 µs, k2 9.86 µs (bench §4.1: 16.03 / 9.54); CUTLASS prologue 12.58 µs and
  doActivation 3.62 µs on the verify layers (FC traces: 11.5-12.3 / 3.5).
- Unprofiled: greedy prose-en 18.21 ms/step at 5.13 tok/step, sampling code-edit 13.52 ms/step at 6.25 tok/step
  (single requests; no comparison arm, so these say nothing about the effect).

## 7. Verified vs assumed; open risks

Verified:
- On the GPU (job1): §3 numerics on every census routing; the GEMV-CUTLASS gap equals the W4A4-vs-W4A16 gap of
  the same bytes (so the GEMV decodes the weights as CUTLASS does); bitwise determinism; eager, captured and
  fallback dispatch; timings and the tile sweep.
- On the CPU (Triton interpreter, `bench/dg1/selftest_cpu.py`): the kernels against the emulation (at most
  4.4e-4), 3 tile configs including S2 = 1 and an id of -1.
- By reading code: every row of §2.

Assumed:
- The server's online quantizer (FlashInfer `nvfp4_quantize`) writes the same format as the bench's torch
  `quantize_nvfp4`; both follow the ModelOpt convention (§2). Either way the GEMV reads the bytes as CUTLASS does.
- x is random bf16; the server feeds the post-norm hidden state.
- In-server, the first one-token call is an eager warm-up (code read, §2 last row), so draft graphs capture the
  GEMV. The A/B log and trace check this.
- The in-server CUTLASS draft call costs what the bench measures. FC traces give about 50 µs (prologue 11.5-12.3,
  GEMM1 17.7-18.5, doAct 3.5, GEMM2 17.0).

Open risks:
- Acceptance: the draft numerics change. Only the server A/B (tok/step) settles it.
- PDL: k1 waits for its predecessor with `griddepcontrol.wait`. This is the same contract as the fork's other
  PDL Triton kernels (production `SGLANG_TRITON_PDL=1`).
- One workspace per layer assumes one stream per layer call: no two-batch overlap or concurrent use of the
  draft MoE layer.
- Not exercised: fused shared experts (R = 11), EP/TP > 1 (refused at enable), shapes with swizzle padding
  (refused), and `FLASHINFER_MOE_PRUNE_MIN_ROWS=1` (CUTLASS would prune T=1; the GEMV never prunes).
- The tile config is tuned for E=512, H=2560, I=640 on the RTX PRO 6000 Max-Q. Other shapes that pass
  `unsupported_reason` are untested and untuned.
- Performance follow-up: k1 is 4.9 µs per call slower than the FC prototype's k1 (§4.1); the cause is not isolated.

Enable in a server (after the §6 setup):

```bash
cd ~/tools/sglang-dg && PYTHONPATH=$PWD/python SGLANG_OPT_DRAFT_MOE_GEMV=1 PORT=8021 ./serve-fast.sh wa
# serve-fast.sh already exports SGLANG_TRITON_PDL=1; expect "draft MoE GEMV enabled for layer" in the log
```
