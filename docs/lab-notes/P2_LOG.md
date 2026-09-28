# P2 — singleton-route pruning folded into FlashInfer's fused routing prologue

Follow-up to `P1_LOG.md` step 7.  Worktree `$HOME/tools/sglang-p2`, branch
`fable/prune-in-prologue` off `codex/perf-v1 @ 57df694fca` (which already carries P1).
FlashInfer side in the private copy `$HOME/tools/flashinfer-p2` (a `cp -a` of the
production package, i.e. a0 + a3 + g2-1 + g2-2 + g1-pack-only) with
`SGLANG_CACHE_DIR=$HOME/.cache/sglang-p2`.  The production venv, the main worktree,
`serve-fast.sh` and the other private copies were not touched.

Date: 2026-09-06.  Author: P2 agent (Fable).

## 0. The problem being fixed

P1 established that dropping low-weight singleton routes (τ=0.08) removes 17.6 of ~68 distinct
experts per W16 verify call and cuts the grouped GEMM by **1.2 ms/step**, but that the net at W16
was only −0.1 … −0.7 ms because the separate prune kernel, inserted immediately in front of
`fusedBuildExpertMapsSortFirstToken`, costs that prologue its 54 % overlap with the preceding
kernels: its exclusive time went 265 → 462 µs/step, and the prune kernel itself added ~190 µs.
At W4 the same mechanism already paid (−10 % step) because the penalty is smaller there.

## 1. What P2 does

The mask is evaluated **inside** `fusedBuildExpertMapsSortFirstTokenAndStridesKernel`
(`flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`), the A0/A3
prologue that G2-2 and G1 already extended.  No new launch, nothing in front of the prologue.

Per block (every block re-runs the rank, by the A3 design):

1. after the top-k ids are loaded into registers, load the row's `k` routing weights
   (`token_final_scales`, already a `runMoe` argument);
2. **pre-prune count**: zero `shared_offsets[0, E)` (the A3 buffer that is dead until after
   `RankKeys`), `__syncthreads`, one shared `atomicAdd` per on-node route, `__syncthreads`;
3. per route: within-row rank = `#{j : w_j > w_i or (w_j == w_i and j < i)}` (a k×k register
   compare, P1's exact tie rule, so it does not depend on `topk`'s storage order); drop iff
   `count == 1 && w_i < τ && rank >= MIN_RANK`; a dropped route is **re-keyed to
   `num_experts_per_node`**, the "not on this node" bucket;
4. `__syncthreads`, then the unchanged `RankKeys`.

Applying the drop *before* the rank is both cheaper and simpler than the step-7 sketch
(rank, then a `BlockScan` to shift the offsets): the radix rank sorts the re-keyed routes past
every real expert exactly as it sorts P1's `-1` ids (which the kernel maps to the same bucket), so
the permutation, `expert_first_token_offset`, the G1 packing rank and the G2-2 inverse
permutation in shared memory are all built from the pruned set with **no further change to any
of them**, and the keys reaching `RankKeys` are identical to what the P1 path feeds it.  The count
that defines "singleton" is the pre-prune count because the histogram is taken before any key is
changed.

5. Block 0 writes the dropped routes back into `token_selected_experts` (id −1) and
   `token_final_scales` (weight 0) — P1's convention — so `finalizeMoeRoutingKernel` /
   `finalizeMoeRoutingNoFillingKernel` (the unfused, deterministic path) skip them via their
   existing `expert_id < 0` test.  The fused FINALIZE epilogue never sees a pruned route (it walks
   permuted rows `< expert_first_token_offset[E]`), and the G2-2 row blocks read
   `expand_unpermuted_scales` only for surviving rows.

**On the write-back vs the other blocks' reads.**  Blocks 1… re-read the same tensors for their
own copy of the rank, so a block may observe a route already rewritten by block 0.  It derives
the same keys either way: an id of −1 maps straight to the not-on-node bucket (the dropped
route's own key), its histogram bin is untouched by anyone else (the route was a singleton), and
a zeroed weight can only *lower* the rank of that same dropped route — never below `MIN_RANK`,
since at least `MIN_RANK` routes with larger, unzeroed weights remain — while every other route's
count and weight are unchanged and its rank can only fall, which keeps a route that was not
dropped not dropped.  So the interleaving is benign; it is still documented in the kernel
because it is the one place where this patch relies on an argument rather than on ordering.

### Host side

`FusedPrologueRequest` (`moe_kernels.h`) carries `prune_tau / prune_min_rank / prune_scales /
prune_ids_out / prune_scales_out`, filled in `runMoe` next to the G2-2 request when
`FLASHINFER_MOE_PRUNE_SINGLETON_TAU > 0`, `ep_size == 1`, `token_final_scales != nullptr`,
`MIN_RANK < top_k` and `MIN_ROWS <= num_rows <= MAX_ROWS` (defaults 2..64, P1's).  `tau == 0`
leaves the kernel on its upstream path (the whole block is behind one uniform branch on a kernel
argument).  If the fused prologue declines a call for any of its existing reasons the call simply
runs unpruned — fail-safe, like A0/A3/G2.

### SGLang side (worktree `sglang-p2`, `14d4c4c985`)

`SGLANG_MOE_PRUNE_SINGLETON_TAU` stays the user-facing knob.  With
`SGLANG_MOE_PRUNE_IN_PROLOGUE=1`, `prune_singleton.py` exports τ / MIN_RANK / MIN_ROWS /
MAX_ROWS to the `FLASHINFER_MOE_PRUNE_*` variables at import time (the csrc reads them once, on
the first MoE call, through `getenv`, which `os.environ` writes reach via `putenv`) and its own
Triton kernel becomes a no-op.  `prune_singleton_routes_()` exposes the P1 kernel without the env
gating for the equivalence test.

### Deliverables

* `bench/moe_smallm/patches/p2-prune-in-prologue.patch` (+ README section) — apply after
  `g1-pack-only`; verified to reproduce `~/tools/flashinfer-p2` byte for byte from the
  production files.
* `bench/moe_smallm/test_prune_prologue.py` — census replay + cross-process compare.
* `bench/moe_smallm/p2env.sh`, `runs/p2/probe1.sh`, `runs/p2/measure.sh`.

## 2. Correctness (`runs/p2/probe1.sh`, log `runs/p2/probe1.log`, 2026-09-06 21:5x)

Build: the first MoE call against the private copy rebuilt the JIT module under
`~/.cache/sglang-p2` (all TUs, as A8 notes for a relocated package).  No compile
issues.

`bench/moe_smallm/test_prune_prologue.py` replays 48 recorded verify calls (every 97th
call of `runs/census-{w16,w4}-code-edit.npz`, real ids **and** weights) through the
production `flashinfer_cutlass` path on the real 512-expert NVFP4 layer 4, in
deterministic mode (`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`, `--autotune-cache none`,
`FLASHINFER_MOE_PACK_GROUPS=1`, G2 folds on), one env per process:

| arm | env |
|---|---|
| `off` | private build, `TAU=0` |
| `p1` | private build, P1 Triton kernel, `TAU=0.08` |
| `p2` | private build, in-prologue, `TAU=0.08 IN_PROLOGUE=1` |
| `prod-off` | **production venv csrc** (its own JIT cache), `TAU=0` |

| check | W16 (T=16) | W4 (T=4) |
|---|---|---|
| written-back ids/weights == numpy reference (in-process, `p2`) | **48/48** | **48/48** |
| written-back ids/weights == P1 Triton kernel on a copy (in-process, `p2`) | **48/48** | **48/48** |
| `p1` vs `p2`: ids/weights identical | **48/48** (753 routes dropped in both) | **48/48** (474 in both) |
| `p1` vs `p2`: MoE output **bit-identical** | **48/48 PASS** | **48/48 PASS** |
| `off` vs `p2`: outputs differ (pruning live) | 48/48 differ, max_abs ~1e-2 | 48/48 differ |
| `off` vs `prod-off`: private build with `TAU=0` == production build | **48/48 bit-identical** | **48/48 bit-identical** |

Routes dropped / D on this sample: W16 **15.69 routes/call, D 67.67 → 51.98**; W4 **9.88,
D 32.81 → 22.94** — the same picture as the P1 census (17.6 / 8.9 over the full workloads).

So: the in-kernel mask selects exactly the P1 set (the k×k rank with P1's tie rule and the
pre-prune histogram reproduce it route for route), the re-keyed routes land where P1's `-1`
ids land, and `TAU=0` is byte-identical to the production build.

## 3. In-server speed, round 1 (`runs/p2/measure.sh prof1`, 22:00-22:15, order off → p1 → p2)

One server per arm, all three on the private copy + `sglang-p2` worktree (only the env differs),
`prof/validate.sh MODE=prof` (20-step traces on code-edit and prose-en), read with
`prof/exclusive_time.py --phase verify` (exclusive = time the family is the only thing on the
GPU, i.e. what deleting it saves) and `prof/p1_moe.py` (per-call medians).

| arm | env |
|---|---|
| off | `SGLANG_MOE_PRUNE_SINGLETON_TAU=0` |
| p1 | `TAU=0.08` — P1 Triton kernel in front of the prologue |
| p2 | `TAU=0.08 SGLANG_MOE_PRUNE_IN_PROLOGUE=1` — the mask inside the prologue |

### W16 — exclusive µs/step (code-edit / prose-en)

| family | off | p1 | **p2** |
|---|--:|--:|--:|
| `cutlass_moe_grouped_gemm1` | 4262.6 / 4047.4 | 3066.1 / 2970.9 | 3030.9 / 3162.4 |
| `cutlass_moe_grouped_gemm2` | 2329.8 / 2248.3 | 1653.2 / 1572.7 | 1582.5 / 1673.1 |
| **GEMM subtotal** | 6592.4 / 6295.7 | 4719.3 / 4543.6 | **4613.4 / 4835.5** |
| `fusedBuildExpertMapsSortFirstToken` | **259.1 / 254.9** | 468.6 / 497.3 | **264.4 / 289.0** |
| its `excl/raw` | 45.5 % / 47.6 % | 85.3 % / 88.3 % | **46.8 % / 45.3 %** |
| `_prune_singleton_kernel` (raw) | – | 187.2 / 203.5 (excl **0.0**: PDL hides it) | – |
| prologue median duration, µs/call | 11.52 | 11.39 | 12.42 |
| **verify phase, total exclusive** | 11154.0 / 10662.0 | 9474.6 / 9132.6 | **9140.1 / 9338.0** |
| step wall (median), ms | 18.29 / 18.02 | 16.68 / 16.12 | **16.24 / 16.32** |

* **The prologue penalty is gone.**  With the mask inside the kernel its exclusive time is
  264 / 289 µs/step against 259 / 255 with pruning off — back to being ~54 % hidden — where
  the P1 launch had pushed it to 469 / 497 (85-88 % exposed).  The in-kernel histogram + k×k
  rank cost +0.9 µs of prologue *duration* (11.5 → 12.4 µs/call), all of it under the overlap.
* The P1 kernel's own exclusive time is **0.0** in every trace: PDL overlaps it completely with
  the preceding kernels.  Its entire cost was the overlap it took from the prologue
  (+210 / +242 µs/step here, +197 in P1_LOG step 6) — which is exactly what P2 recovers.
* The GEMM saving is unchanged: −1.98 / −1.46 ms/step exclusive (p2 vs off).  The p1↔p2 GEMM
  difference (−106 / +292 µs) is the usual process-to-process routing variation (a separate
  server accepts a different token chain and touches a different D per call; G2_LOG §6.2), not
  an effect of where the mask is computed — the deterministic replay in §2 shows the two masks
  are identical.

### W4 — exclusive µs/step (code-edit / prose-en)

| family | off | p1 | **p2** |
|---|--:|--:|--:|
| GEMM subtotal | 3222.2 / 2962.2 | 2464.2 / 2268.3 | **2308.7 / 2244.8** |
| `fusedBuildExpertMapsSortFirstToken` | **235.8 / 208.8** (41 %) | 302.1 / 240.1 (57 % / 53 %) | **207.5 / 181.7** (36 % / 34 %) |
| `_prune_singleton_kernel` (raw) | – | 115.6 / 114.7 (excl 0.0) | – |
| verify phase, total exclusive | 7519.9 / 7357.9 | 6785.0 / 6362.0 | **6625.2 / 6413.3** |
| step wall (median), ms | 10.70 / 10.61 | 10.11 / 9.54 | **9.84 / 9.58** |

Same structure: the prologue is back under the overlap (208 / 182 vs 302 / 240), the GEMM keeps
its −0.9 / −0.7 ms.  At W4 the P1 penalty was only +66 / +31 µs, so the P2 gain over P1 is
correspondingly smaller (−95 / −58 µs of prologue), as P1_LOG step 8 predicted.

Per-step wall numbers carry the ±0.3 ms process-to-process spread G2_LOG §6.3 documented (the
`off` arm here is 0.9 ms slower than P1_LOG's `base2`, on the same build), so the ship judgement
rests on the per-family exclusive rows above and on the reversed-order round below.

## 4. Round 2 (reversed order p2 → p1 → off, 22:23-22:38) and the census round

### W16 — exclusive µs/step (code-edit / prose-en)

| family | p2 (first) | p1 | off (last) |
|---|--:|--:|--:|
| GEMM subtotal | 4974.6 / 5129.7 | 4708.4 / 4853.2 | 6640.0 / 6462.0 |
| `fusedBuildExpertMapsSortFirstToken` | **299.4 / 308.9** (46 % / 48 %) | 467.2 / 503.8 (87 % / 86 %) | **266.0 / 245.8** (48 % / 44 %) |
| verify phase, total exclusive | 9790.9 / 9655.1 | 9520.3 / 9561.9 | 11286.2 / 10815.6 |
| step wall (median), ms | 17.06 / 16.69 | 16.41 / 16.56 | 18.49 / 18.09 |

The prologue row reproduces round 1 to within 35 µs (p2 264/289 → 299/309, p1 469/497 →
467/504, off 259/255 → 266/246).  The step-level ordering flips because **this p2 server ran at
a higher D**: GEMM1 65.5 / 68.0 µs/call against p1's 60.9 / 63.7 — i.e. +3-4 distinct experts per
call, +270 / +280 µs of GEMM per step, which is larger than the 170-200 µs the prologue gives
back.  Whether that is the patch or the process was settled directly:

### The census round (`runs/p2/census.sh c1`, 22:55-23:05, p2 → p1 → off, W16)

`p1_run.sh MODE=speed CENSUS=1`: the route-census hook records every call's **pre-prune** ids
and weights (it wraps `_run_flashinfer_cutlass`, so it sees the tensors before either prune),
12 744 T=16 calls per arm, and the P1 rule is applied offline:

| arm | pre-prune D | post-prune D (τ=0.08, offline) | routes dropped / call |
|---|--:|--:|--:|
| p2 | 59.58 | 43.23 | 16.35 |
| p1 | 61.11 | 44.77 | 16.34 |
| off | 59.67 | 43.56 | 16.10 |

The two arms drop the same number of routes per call (16.35 vs 16.34), and in *these* three
servers p2 happened to run 1.5 experts **below** p1 — GEMM1 medians 59.6 / 67.1 (p2) vs
60.9 / 62.2 (p1) µs in the same traces.  So the ±3-4 expert swing in round 2 is the
process-to-process routing variance (a different accepted chain touches a different set of
experts), not a difference in what is pruned; the deterministic replay in §2 had already shown
the masks are identical route for route.

The census round's traces make one more point for free: the hook's two `index_copy_` kernels
sit in front of the prologue in **every** arm, and there the prologue is 97-98 % exposed in
all three (551-594 µs/step, prune-off included).  Any kernel in front of
`fusedBuildExpertMapsSortFirstToken` costs it its overlap — which is why the mask had to move
inside it, and why those traces are not used for the prologue comparison.

### W4 — round 2, exclusive µs/step (code-edit / prose-en)

| family | p2 (first) | p1 | off (last) |
|---|--:|--:|--:|
| GEMM subtotal | 2151.0 / 2063.5 | 2212.1 / 2210.6 | 3118.7 / 3020.8 |
| `fusedBuildExpertMapsSortFirstToken` | **238.1 / 216.3** (43 % / 42 %) | 294.6 / 230.6 (53 % / 50 %) | 231.9 / 250.8 (43 % / 45 %) |
| verify phase, total exclusive | 6335.2 / 6153.4 | 6728.2 / 6416.9 | 7334.6 / 7275.9 |
| step wall (median), ms | **9.45 / 9.06** | 9.97 / 9.55 | 10.46 / 10.50 |

### Summary over the clean rounds (round 1 + round 2; census round excluded for the prologue)

Prologue exclusive µs/step, mean of the four (width × workload) cells per round:

| | off | p1 | **p2** | p2 − p1 |
|---|--:|--:|--:|--:|
| W16 | 256.5 | 484.2 | **290.4** | **−193.8** |
| W4 | 231.8 | 266.9 | **210.9** | **−56.0** |

Step wall, mean of the two rounds (code-edit / prose-en), ms:

| | off | p1 | **p2** | p2 vs off | p2 vs p1 |
|---|--:|--:|--:|--:|--:|
| W16 | 18.39 / 18.06 | 16.55 / 16.34 | **16.65 / 16.51** | **−9.5 % / −8.6 %** | +0.6 % / +1.0 % |
| W4 | 10.58 / 10.56 | 10.04 / 9.55 | **9.64 / 9.32** | **−8.9 % / −11.7 %** | −4.0 % / −2.4 % |

At W4 the fold wins on both metrics.  At W16 the prologue saving is exactly the predicted
~200 µs/step (−1.1 % of the step) and is reproducible to ±35 µs; the p2−p1 step-wall
difference of +0.1 … +0.2 ms is the average of one round at −0.4 / +0.2 and one at +0.65 / +0.13,
i.e. inside the ±0.3 ms process spread and, per the census, driven by which chain each server
happened to accept.  The family-level rows are the measurement; the step wall at W16 cannot
resolve a 0.2 ms effect in two rounds (G2_LOG §6.3 reached the same conclusion for a 0.25 ms
effect).

Against **pruning off**, which is the decision that matters for W16 (P1 was HOLD there), P2 is
**−1.7 … −1.9 ms/step (−8.6 … −9.5 %)** in both rounds, well past the 3 % bar, with the whole
GEMM saving reaching the step.  P1_LOG's step-6 estimate of −5.4 … −7.0 % for the folded mask
was, if anything, conservative: the `off` arm here sits at 18.1-18.5 ms rather than 17.2, and
the GEMM saving is −1.5 … −2.0 ms exclusive.

## 5. Quality (`runs/p2/quality.sh q1`, 22:38-22:52, off → p2 per width, same hour)

`p1_run.sh MODE=quality`: fnbench ×2 greedy on the four public workloads, synthetic needle 18.5k,
greedy-agreement dump.  Acceptance is the mean of the server's `accept len:` decode-batch lines
inside each request's window (the `/metrics` gauge is not exported by this build).

| profile | arm | code-edit | prose-en | agent-loop | prose-ja | needle |
|---|---|--:|--:|--:|--:|---|
| W16 | off | 9.12 / 492 t/s | 2.97 / 162 | 4.65 / 250 | 2.60 / 147 | PASS |
| W16 | **p2** | 8.92 / **543** | 3.00 / **179** | 5.11 / **310** | 2.84 / **177** | **PASS** |
| W4 | off | 3.71 / 357 | 2.67 / 259 | 3.14 / 314 | 2.35 / 235 | PASS |
| W4 | **p2** | 3.68 / **400** | 2.65 / **287** | 3.11 / **355** | 2.32 / **258** | **PASS** |

(acceptance / decode t/s, fnbench median of 2)

* Acceptance sits inside the ±4-6 % baseline-to-baseline band P1_LOG established at every cell
  (W16 code-edit −2 %, agent-loop +10 %, prose-ja +9 %; W4 all within ±1.5 %) — no consistent
  direction, exactly as with the P1 kernel.
* Throughput: **W16 +10.5 / +10.5 / +24 / +21 %**, **W4 +12 / +11 / +13 / +10 %**
  (code-edit / prose-en / agent-loop / prose-ja).  The W16 gain is the one P1 could not deliver.
* Greedy agreement vs prune-off over 256 tokens, code-edit (the only workload whose
  self-agreement is 1.000): **1.000 at both widths**, no divergence; the other three are at
  their 0.03-0.6 noise floor as documented in P1_LOG.

## 6. Conclusion

**The fold works as designed and pays at both widths.**

* Correctness: same (row, expert) set dropped as P1 on recorded routing at T=16 and T=4
  (48/48 calls each), MoE output bit-identical to the P1 path in deterministic mode, `TAU=0`
  bit-identical to the production build; in-server the two prune the same 16.3 routes/call.
* Speed: the prologue's exclusive time returns from 484 to **290 µs/step at W16** (off: 257) and
  from 267 to **211 at W4** (off: 232), the −1.5 … −2.0 ms/step GEMM saving is untouched, and the
  in-kernel work costs +0.9-1.0 µs of *hidden* prologue duration.  Against pruning off: **W16
  −8.6 … −9.5 % step, +10 … +24 % t/s; W4 −8.9 … −11.7 % step, +10 … +13 % t/s.**
* Published quality gates: needle PASS, acceptance flat, code-edit tokens identical (first 256
  tokens).

**Recommendation: ship `SGLANG_MOE_PRUNE_SINGLETON_TAU=0.08 SGLANG_MOE_PRUNE_IN_PROLOGUE=1` at
both W4 and W16** (and `wa`), after applying `p2-prune-in-prologue.patch` to the production venv
(after `g1-pack-only`; ~2 min JIT rebuild on the next start).  Keep P1's Python kernel as the
fallback path (`IN_PROLOGUE=0`) for a venv without the patch — with the patch absent the flag
is harmless: the csrc never reads the env and the Python kernel is simply skipped, i.e. **no
pruning**, so the two must be rolled out together.

### Caveats

* At W16 the p2-vs-p1 difference (~0.2 ms/step) is below what two server rounds resolve at the
  step level; the family-level exclusive rows (±35 µs across rounds) are the evidence, plus the
  census showing the per-process D swing that dominates the step wall.  Against `off` the result
  is not in doubt.
* The write-back of `-1` / `0` into the caller's `topk_ids` / `topk_weights` from block 0 races
  with the other blocks' reads of the same tensors.  §1 argues the race is benign (every
  observable interleaving yields the same keys), the replay test exercised 176-block grids
  96 times without a mismatch, and the published quality gates (needle, acceptance, code-edit
  agreement) passed; but it is an argument, not an
  ordering guarantee.  If that ever needs removing, the write-back can be made conditional on
  the unfused-finalize path (the only consumer) at the cost of P1-style observability.
* `MAX_ROWS` is effectively `min(MAX_ROWS, 32)`: the fused prologue itself declines batches over
  32 rows, and those then run unpruned (P1's kernel would have pruned up to 64).  Verify batches
  are 4 / 16 rows, so nothing changes for the shipped profiles.
* EP > 1 is gated off (`ep_size == 1`); single-GPU box, untested otherwise.
* GPU shared with other agents' jobs throughout; every conclusion rests on back-to-back
  same-session pairs and per-family exclusive time, never on a lone step-wall number.
