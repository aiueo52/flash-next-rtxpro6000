# G2 log — deleting launches from the FlashInfer CUTLASS fused-MoE chain

Task: cut the number of kernel launches in the per-MoE-call chain for
Qwen3.8-Flash-Next (SGLang fork `~/tools/sglang-rtxpro6000`, FlashInfer 0.6.17 with
A0 + A3 already applied).  Everything below was done in the private copy
`~/tools/flashinfer-g2/flashinfer` with `SGLANG_CACHE_DIR=~/.cache/sglang-g2`; the
production venv, `flashinfer-a8`, `flashinfer-g1` and the main worktree were never
touched.

Date: 2026-09-05.  Author: G2 agent.

## 0. The chain as it stands (measured, from the production traces)

`prof/phase_kernels.py prof/traces/final0905-w4-code-edit/*.gz "step[TARGET_VERIFY bs=1]"`,
one MoE layer (kernels 50-55 of the step):

```
 50   5.8   fusedBuildExpertMapsSortFirstTokenAndStridesKernel   (A0+A3 prologue)
 51   4.8   expandInputRowsKernel
 52  53.4   CUTLASS GEMM1
 53   3.9   doActivationKernel
 54   1.0   memset32                       (zeroes the fused-finalize output)
 55  30.5   CUTLASS GEMM2                  (finalize fused into the epilogue)
```

There is nothing else between the prologue and GEMM2 — item 4 of the brief is
answered by the trace above: no hidden launches, and every gap is <= 0.2 us, i.e.
the chain is strictly serial with no PDL overlap to recover.

## 1. What G2 does

| item | env kill switch | what it deletes |
|---|---|---|
| G2-1 | `FLASHINFER_MOE_FOLD_MEMSET=0` | the `memset32`: `doActivationKernel` zeroes the fused-finalize output in its own grid |
| G2-2 | `FLASHINFER_MOE_FOLD_EXPAND=0` | the `expandInputRowsKernel` launch: the row permute + NVFP4 re-quantisation moves into the A0/A3 prologue kernel |

Both default ON in the private copy and fail safe (any unsupported configuration
falls back to the upstream launch, nothing throws).

(sections filled in as measurements land)

## 2. Items 3 and 4 of the brief: nothing left to hoist

**Item 3 (per-call constant bookkeeping that could move to graph capture).**  Everything the
descriptor phase writes is routing-dependent:

* `computeStridesTmaWarpSpecializedForExpert` writes `problem_shapes[expert]` for *all* 512
  experts, but the value contains `gemm_m = expert_first_token_offset[e+1] - [e]`, which
  changes every call.  The constant-looking case (an inactive expert, `gemm_m = 0`) cannot be
  pre-filled at capture time either: an expert that was active on the previous call leaves a
  stale non-zero `M` in the buffer, so clearing it needs the same per-call pass.
* The rest of the per-expert work (alpha-scale pointers, FP4 block-scale descriptors, strides,
  A/B/D pointers) is already **skipped for inactive experts** by the `if (gemm_m == 0) return;`
  early-out, so at W4 only ~28 of 512 experts pay it.
* A3 already removed the only launch this phase used to cost.  The remaining 512 problem-shape
  writes are one 4-byte-triple per thread across 512 threads — latency, not work, and it is
  inside a kernel that has to run anyway.

So there is no launch and no measurable work to recover here, and G2 does not touch it.

**Item 4 (anything else in the timeline).**  `prof/phase_kernels.py` on
`final0905-w4-code-edit` and `-w16-code-edit` shows exactly the six kernels listed in §0
between the router and the post-MoE gate, with gaps of -1 to +0.2 us.  There is no hidden
launch, no memcpy, and no idle window that a reordering could exploit.

## 3. First result (single run, same process/build, folds on vs the previous build)

`bench/moe_smallm/bench_moe.py --mode profile --widths 4,16 --sweep-distinct 28 --rotation 16
--min-seconds 2.0` with the production autotune cache, per-kernel `us/call`:

| kernel | before (A0+A3) | after (A0+A3+G2) |
|---|--:|--:|
| `fusedBuildExpertMapsSortFirstTokenAndStridesKernel` | 5.39 | **6.56** |
| `expandInputRowsKernel` | 4.54 | *(deleted)* |
| `doActivationKernel` | 3.59 | **3.53** |
| `memset32` | 0.98 | *(deleted)* |
| **glue total** | **14.50** | **10.09** |

So the prologue absorbs a 4.54 us launch for 1.17 us of extra time, and `doActivationKernel`
absorbs the 0.98 us memset for **nothing measurable** (3.59 -> 3.53, inside the noise).
`us_per_call_sustained` moved 72.56 -> 67.89 (T=4) and 72.59 -> 68.31 (T=16) in the same
back-to-back pair of builds.

**Four glue kernels become two.**  Per step that is 2 x 51 = **102 fewer launches at W4** and
2 x 63 = **126 fewer at W16**.

The properly alternated A/B (4 rounds x 4 env combinations, one process each) is in §4.

## 4. Bench A/B (4 alternating rounds, one process per env combination)

`bench_moe --mode profile --widths 4,16 --sweep-distinct 28 --rotation 16 --min-seconds 2.0`
with the production autotune cache, GPU held under `~/.gpu.lock`.  Metric is
`us_per_call_sustained` (CUDA-graph wall time per MoE call).

| combination | T=4 median | vs off | T=16 median | vs off |
|---|--:|--:|--:|--:|
| both folds off (= A0+A3 upstream) | 74.92 | -- | 74.91 | -- |
| G2-1 only (`FOLD_MEMSET=1`) | 72.70 | **-2.22** | 73.00 | **-1.91** |
| G2-2 only (`FOLD_EXPAND=1`) | 72.08 | **-2.84** | 72.36 | **-2.55** |
| both | **70.05** | **-4.87** | **70.50** | **-4.41** |

Round-to-round spread is 0.03-0.4 us, i.e. every one of these deltas is 5-100x the noise, and
the two folds are additive to within 4 %.

Per-kernel medians (`us/call`) from the same runs:

| kernel | off T=4 | both T=4 | off T=16 | both T=16 |
|---|--:|--:|--:|--:|
| routing prologue (A0+A3, +G2-2) | 5.42 | **6.57** | 5.45 | **6.42** |
| `expandInputRowsKernel` | 4.54 | -- | 4.44 | -- |
| `doActivationKernel` (+G2-1) | 3.65 | 3.92 | 3.61 | 3.56 |
| `memset32` | 0.98 | -- | 0.96 | -- |
| **glue total** | **14.59** | **10.49** | **14.46** | **9.98** |
| GEMM1 | 39.27 | 38.27 | 38.00 | 37.70 |
| GEMM2 | 23.59 | 22.96 | 22.48 | 24.38 |

The GEMMs are untouched (their spread across combinations is the usual +-1 us tactic/clock
noise).  Note the wall-clock saving (-4.87 / -4.41 us) is slightly **larger** than the deleted
kernel time (-4.10 / -4.48 us): the inter-kernel gaps go with the launches.

Per step, at 51 MoE calls (W4) and 63 (W16):

* **W4: -0.248 ms/step**  (-2.1 % of the 11.95 ms step)
* **W16: -0.278 ms/step** (-1.4 % of the 20.47 ms step)

and **102 / 126 fewer kernel launches per step**.

## 5. Numerics

### 5.1 Deterministic mode -- bit-identical

`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 --autotune-cache none` removes the GEMM2 finalize
epilogue's non-deterministic scatter-add (and the cache must go with it: the production
autotune cache indexes tactics by position, and dropping the finalize tactics makes
`Invalid gemm2 profile id: 57` -- that mistake cost one 45-minute run).
`moe_smallm/ab_prologue.py`, real 512-expert layer, T = 4 and 16, 4 calls each:

```
det_off2  vs det_off : RESULT: PASS (bit-identical)   <- control, same config twice
det_memset vs det_off: RESULT: PASS (bit-identical)
det_expand vs det_off: RESULT: PASS (bit-identical)
det_both   vs det_off: RESULT: PASS (bit-identical)
```

**G2-2 is bit-identical**, as designed: the fold changes which threads do the permute and
re-quantisation, not the arithmetic, and the (lane, lane^1) pairing that feeds
`cvt_warp_fp16_to_fp4`'s amax shuffle covers the same 16-element scale-factor groups as the
upstream 256-thread layout.

### 5.2 Production epilogue -- inside the control's own envelope

With the fused-finalize epilogue on, the path is non-deterministic **against itself**.  Same
build, same seed, `ff1_off` run twice:

| comparison | max_abs | n_diff (T=4) | n_diff (T=16) |
|---|--:|--:|--:|
| **control** `ff1_off2` vs `ff1_off` | 2.441e-04 | 2709-3154 / 10240 | 3000-5420 / 40960 |
| `ff1_memset` vs `ff1_off` | 2.441e-04 | 2792-3261 / 10240 | 2869-4926 / 40960 |
| `ff1_expand` vs `ff1_off` | 2.441e-04 | 2354-3228 / 10240 | 2829-5168 / 40960 |
| `ff1_both` vs `ff1_off` | 2.441e-04 | 2586-3201 / 10240 | 3193-4848 / 40960 |

Every patched combination sits in **exactly** the control's envelope -- same maximum absolute
difference to three digits, same fraction of differing elements.  This is the only statement
available for **G2-1**, because the epilogue it feeds is itself non-deterministic; but it is a
sharp one: a zeroing that was skipped, raced, or short would leave the previous call's
accumulated activations in `final_output`, an O(1) error, not 2.4e-04.


## 6. In-server A/B (same build, folds off vs on, `prof/validate.sh`)

Four server runs per width -- one off/on pair in each order, so thermal or clock drift cannot
alias into the result.  `PYTHONPATH=~/tools/flashinfer-g2 SGLANG_CACHE_DIR=~/.cache/sglang-g2`.

### 6.1 Kernel launches per step -- exactly as predicted

| profile | folds off | folds on | delta |
|---|--:|--:|--:|
| W4 total | 1698 | 1596 | **-102** |
| " verify (48 layers) | 1495 | 1399 | -96 |
| " draft / draft_extend | 113 / 90 | 109 / 88 | -4 / -2 |
| W16 total | 2337 | 2211 | **-126** |
| " verify | 1543 | 1447 | -96 |
| " draft / draft_extend | 703 / 91 | 675 / 89 | -28 / -2 |

`-2 x (MoE calls per step)` in every phase, i.e. every MoE call in the model lost both launches.

### 6.2 Per-MoE-call medians from the traces

`specs/G2_moe_chain.py`-style walk of the trace (all 51 / 63 MoE calls per step x 20 steps),
`code-edit`, medians in us:

| | W4 off A | W4 off B | W4 on A | W4 on B | W16 off A | W16 off B | W16 on A | W16 on B |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| routing prologue | 5.63 | 5.49 | 6.69 | 6.72 | 5.38 | 5.38 | 6.56 | 6.62 |
| `expandInputRows` | 4.06 | 4.06 | -- | -- | 4.19 | 4.26 | -- | -- |
| `doActivation` | 3.65 | 3.62 | 3.65 | 3.62 | 3.74 | 3.71 | 3.78 | 3.81 |
| `memset32` | 0.99 | 0.99 | -- | -- | 0.96 | 0.96 | -- | -- |
| **glue** | **14.34** | **14.16** | **10.34** | **10.34** | **14.27** | **14.30** | **10.34** | **10.43** |
| GEMM1 | 41.19 | 41.33 | 41.38 | 41.31 | 73.70 | 75.46 | 76.66 | 74.50 |
| GEMM2 | 25.12 | 25.25 | 25.38 | 25.34 | 41.63 | 42.18 | 43.12 | 41.54 |
| **chain** | **80.64** | **80.74** | **77.09** | **76.99** | **129.60** | **131.94** | **130.11** | **126.46** |

* **The glue result is exactly reproducible**: 10.34 us in *every* folds-on run against
  14.16-14.34 folds-off, i.e. **-3.90 us per MoE call**, at both widths, in both orders.
* **The GEMMs are untouched.**  At W4 they move by +0.3 % (66.31/66.58 -> 66.75/66.66).  At W16
  the two GEMMs vary by +-2 % *between server processes of the same configuration* because a
  separate server accepts a slightly different token chain and therefore routes to a different
  number of distinct experts (`gdn_decode` fires 39.6 vs 41.4 times/step across the pair) --
  that is workload variation, not an effect of the patch.
* Chain: **-3.65 us/call at W4**; at W16 the same -3.90 us of glue is buried in +-2 % of GEMM.

Per step: **-0.19 ms/step (W4, 51 calls)** and **-0.25 ms/step (W16, 63 calls)**.

### 6.3 What the step-level `busy_ms` does *not* show

| profile | workload | off A | off B | on A | on B |
|---|---|--:|--:|--:|--:|
| W4 | code-edit | 10.16 | 10.04 | 9.95 | 10.04 |
| W4 | prose-en | 9.81 | 9.73 | 9.64 | 9.61 |
| W16 | code-edit | 16.83 | 16.87 | 16.88 | 16.56 |
| W16 | prose-en | 16.86 | 16.54 | 16.85 | 16.92 |

W4 leans the right way (-0.10 / -0.15 ms), W16 is a wash.  **Between two server processes the
same configuration reproduces `busy_ms` only to about +-0.3 ms**, because the accepted token
chain -- and with it the number of distinct experts each MoE call touches -- differs; a
-0.19/-0.25 ms effect does not clear that.  This is a limit of the metric, not evidence against
the patch: §6.2 measures the same 1020-1260 MoE calls in those very traces and finds the glue
saving present, identical, and with the GEMMs unchanged, in all four pairings.

## 7. Acceptance and needle (`MODE=full`, folds on)

| profile | code-edit | prose-en | agent-loop | expected | needle |
|---|--:|--:|--:|---|---|
| W4 accept len | ~3.9 | ~2.5 | ~3.2 | 3.8 / 2.6 / 3.2 | **PASS** |
| W16 accept len | ~9 (6.5-14.8) | ~3.0 | ~4.9 (4.2-5.5) | 8.6-11 / 3.0 / 4.6-5.7 | **PASS** |

Throughput in the same runs: W4 377/378, 252/273, 312/328 t/s; W16 636/506, 176/165,
281/372 t/s (code-edit / prose-en / agent-loop, 2 repeats).  Needle at 18.5 k context returned
`AURORA-CEDAR-7319` in both, PASS.

## 8. Conclusion and caveats

**Result.** Two of the four glue kernels are gone from every MoE call.  Per call the glue drops
**14.2-14.3 -> 10.34 us (-3.90 us)**, reproducibly, with the GEMMs untouched; per step that is
**-0.19 ms at W4** and **-0.25 ms at W16** and **102 / 126 fewer launches**.  Against an
11.95 / 20.47 ms step that is **-1.6 % / -1.2 %**.  In the isolated bench (`us_per_call_sustained`,
4 alternating rounds) the same change is **-4.87 / -4.41 us per call**, of which G2-1 is
-2.22 / -1.91 and G2-2 is -2.84 / -2.55.

**Caveats.**

1. The saving is ~1.5 % of the step and does not clear the +-0.3 ms process-to-process spread of
   the step-level `busy_ms` metric.  Judge it on the per-call medians (§6.2), which are taken
   over 1020-1260 calls inside those same traces.
2. G2-1's numerics cannot be proven bit-identical end to end, because the fused-finalize
   epilogue it feeds is non-deterministic against itself (§5.2).  What is shown is that its diff
   envelope is indistinguishable from the control's, and that in the deterministic mode nothing
   changes.  The zeroing writes the same zeros over the same range at the same point in the
   stream; only the kernel doing it changed.
3. The fused prologue's grid grows from `ceil(E/32)` to `ceil(E/32) + num_tokens*k` blocks
   (16 -> 56 at W4, 16 -> 176 at W16).  Every block re-runs the (cheap) rank.  This showed no
   measurable cost -- the prologue grew 5.4 -> 6.6 us while absorbing a 4.2 us launch -- but it
   is the thing to look at first if a future model has a much larger `num_tokens * k`.
4. G2-2 costs the prologue +1.2 us.  That is one extra dependent memory round trip: the rank has
   to `__syncthreads()` before a row block knows which row to load.  Removing it would need the
   row load to issue before the rank completes, which is not expressible without a second
   launch -- so 10.34 us is the floor for this structure.
5. `doActivationKernel` remains, and A8 (fusing it into the GEMM1 epilogue) is still NO-GO.
   Two glue kernels is the floor for G2.
6. Nothing was applied to the production venv.  The two patches under
   `bench/moe_smallm/patches/` apply cleanly on top of `a0` + `a3` and were verified to
   reproduce the private copy byte for byte.

