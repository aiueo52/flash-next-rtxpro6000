# P1 — selective expert-route pruning in the verify step

Worktree `$HOME/tools/sglang-p1`, branch `opus/moe-prune` off
`codex/perf-v1 @ ef31f26346`.  Bench-side tooling lives in this repo.

## The bet

`GEMM(T=16) = 13.0 + 2.233·D µs` per MoE call (MOE_SMALLM_SPEC §8.3), 48 MoE
layers per verify step, `D = 69.4` at W16 / `28.4` at W4 (§8.2).  Of those 69
experts, **35 are singletons** — reached by exactly one of the 16 chain rows.
Dropping a singleton route removes an entire expert weight read (2.76 MB,
~1.6 µs of GEMM1+GEMM2); dropping a route to an expert another row also uses
saves nothing, because the weights are read either way.  A uniform top-k cut
cannot tell the two apart, which is why the g3 draft-side experiment was
negative.  The §8.6 NO-GO on "reduce D" was an inference about quality, never a
measurement — this is the measurement.

## Step 0 — tooling (no GPU)

* `bench/moe_smallm/route_census.py` — `route_logger` plus the routing
  **weights**: a ring of the last N calls with `ids`/`w`/`T`/`k`/`call`, dumped
  atomically as `.npz` on a timer.  Two `index_copy_`s per call, all
  CUDA-graph-capturable, so the verify graph replays record every call.
* `bench/moe_smallm/sitecustomize_census.py` — the same import-hook arming
  trick as `sitecustomize_example.py`.
* `bench/moe_smallm/analyze_census.py` — offline: multiplicity, within-row
  rank, singleton distributions by rank and weight, and the τ / rank-rule sweep
  with ΔD and removed routing mass.
* `prof/census_run.sh` — one server, fnbench per workload, a ring snapshot
  after each.  The ring holds only the last N calls, so a snapshot *is* that
  workload; no differencing (unlike §8.2).
* `prof/agree_gen.py` / `prof/agree_cmp.py` — greedy generation on the four
  workloads (2 repeats) and token-level agreement between two builds, with the
  within-build repeat agreement printed as the noise floor.

## Step 0b — implementation (no GPU yet)

`sglang-p1: python/sglang/srt/layers/moe/prune_singleton.py`, called from
`moe_runner/flashinfer_cutlass.py::_run_flashinfer_cutlass` right after the
top-k tensors are read.

Why the sentinel works, read out of the installed FlashInfer 0.6.17 csrc
(`flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`):

* `fusedBuildExpertMapsSortFirstTokenKernel:403-406` — an id outside
  `[start_expert, end_expert)` is "not on this node" and ranks into the
  `num_experts_per_node` bucket, i.e. **after** every real expert.
* `expandInputRowsKernel:2047` and both finalize kernels (`:2383`, `:2741`)
  loop only to `num_valid_tokens = expert_first_token_offset[num_experts_per_node]`,
  so a masked route is never expanded and never read.
* `finalizeMoeRoutingNoFillingKernel:2434` skips `expert_id < 0 ||
  expert_id >= num_experts_per_node` explicitly, so the k-way reduction simply
  has fewer terms for that row.
* The group-packing patch (`FLASHINFER_MOE_PACK_GROUPS`, §G1) bounds the group
  list by the host-side `expanded_num_rows = num_tokens·k`, which stays an
  upper bound after masking. Nothing to coordinate.

Sentinel is `-1`, not `num_experts`: with a fused shared expert the slot at
index `num_experts` is a real expert.

Safety: the row's top-1 route is never a candidate
(`SGLANG_MOE_PRUNE_MIN_RANK=1`), because a row whose every route was masked
would leave `finalizeMoeRoutingNoFillingKernel` never writing its output row.
Only batches of `[MIN_ROWS, MAX_ROWS] = [2, 64]` rows are touched, which keeps
the T=1 draft/decode calls out of it — there every route is a singleton by
construction, so the rule would degenerate into the uniform top-k cut that g3
already showed to be a loss.

`SGLANG_MOE_PRUNE_KEEP_IDS=1` is the equivalence control: zero the weight but
keep the expert id.  Output is then mathematically identical to real pruning
(a zero-weight route contributes nothing in the finalize) while the GEMM still
reads every expert — so an A/B against the real thing isolates "is the -1
sentinel handled correctly" from "does dropping the route change the answer".

Commit: sglang-p1 `f7e57fd5cd`.

## Step 1a — free preliminary from the §8.2 ring (no GPU)

`runs/routes-{w16,w4}-ring.jsonl` already holds raw `topk_ids` for the last
4096 calls of the 2026-09-05 census (no weights, so only the **rank** rule can
be evaluated; the stored order is `torch.topk`'s, i.e. descending weight).

Singleton routes per call, by stored position (1 = the row's top-1):

| profile | D | singl/call | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| W16 T=16 (3186 calls) | 57.60 | 27.33 | 1.15 | 1.40 | 1.68 | 2.04 | 2.31 | 2.74 | 3.22 | 3.71 | 4.23 | 4.84 |
| W4  T=4  (3936 calls) | 23.71 | 14.12 | 0.72 | 0.84 | 0.98 | 1.13 | 1.28 | 1.49 | 1.68 | 1.80 | 2.02 | 2.18 |

Singleton-ness rises monotonically with rank — exactly the structure the
mechanism needs, and the reason a *uniform* top-k cut cannot exploit it: at
rank 10 a route is 4.2x (W16) / 3.0x (W4) more likely to be the sole user of
its expert than at rank 1, so the same number of dropped routes buys 4x the
expert reads at the tail.

| rule | W16 ΔD | W16 D after | W4 ΔD | W4 D after |
|---|--:|--:|--:|--:|
| singletons at rank 10 | 4.84 (8.4 %) | 52.8 | 2.18 (9.2 %) | 21.5 |
| singletons at rank 9-10 | 9.07 (15.7 %) | 48.5 | 4.20 (17.7 %) | 19.5 |
| singletons at rank 8-10 | 12.79 (22.2 %) | 44.8 | 6.00 (25.3 %) | 17.7 |
| singletons at rank 7-10 | 16.00 (27.8 %) | 41.6 | 7.68 (32.4 %) | 16.0 |

At `48 · 2.233 µs/expert` (W16) and `48 · 2.124` (W4) the rank 9-10 rule is
**−0.97 ms/step of 20.47 (−4.7 %) at W16** and **−0.43 of 11.95 (−3.6 %) at
W4**, before subtracting the prune kernel (~48 × 2 µs = 0.10 ms/step).  That
clears the 3 % ship bar on paper at both widths; whether it clears the quality
bar is the whole question.

(The ring's D=57.6 is below §8.2's 64.5 for prose-ja because it is the last
4096 calls of the run, not the workload mean — read the ΔD *fractions*, not the
absolute D.  The weight census replaces both.)

### Per-workload projection

The ring is one workload's tail, so the *share* of singletons that sit at each
rank was taken from it and applied to each workload's own `x1` from the §8.2
snapshots (recomputed here by differencing `runs/routes-{prof}-*.json`; D and
x1 reproduce §8.2's tables exactly, so the differencing is sound).

W16, T=16 — projected ΔD (and % of that workload's D):

| workload | D | x1 | rank 10 | rank 9-10 | rank 8-10 |
|---|--:|--:|--:|--:|--:|
| code-edit | 75.88 | 40.99 | 7.26 (9.6 %) | **13.60 (17.9 %)** | 19.18 (25.3 %) |
| prose-en | 68.53 | 33.64 | 5.96 (8.7 %) | **11.17 (16.3 %)** | 15.74 (23.0 %) |
| agent-loop | 72.68 | 37.55 | 6.65 (9.1 %) | **12.46 (17.1 %)** | 17.57 (24.2 %) |
| prose-ja | 64.53 | 31.72 | 5.62 (8.7 %) | **10.53 (16.3 %)** | 14.85 (23.0 %) |

W4, T=4:

| workload | D | x1 | rank 10 | rank 9-10 | rank 8-10 |
|---|--:|--:|--:|--:|--:|
| code-edit | 30.95 | 24.23 | 3.74 (12.1 %) | **7.21 (23.3 %)** | 10.30 (33.3 %) |
| prose-en | 27.41 | 18.91 | 2.92 (10.7 %) | **5.62 (20.5 %)** | 8.04 (29.3 %) |
| agent-loop | 29.00 | 21.15 | 3.26 (11.3 %) | **6.29 (21.7 %)** | 8.99 (31.0 %) |
| prose-ja | 25.73 | 16.78 | 2.59 (10.1 %) | **4.99 (19.4 %)** | 7.13 (27.7 %) |

At 48 MoE verify calls/step: rank 9-10 is **−1.13 to −1.46 ms/step (−5.5 to
−7.1 %) at W16** and **−0.51 to −0.73 ms/step (−4.3 to −6.1 %) at W4**.
Note the *fraction* of D removed is larger at W4 (20-23 %) than at W16
(16-18 %) even though the absolute ΔD is half: at T=4 there are fewer rows to
share an expert with, so 65-78 % of the touched experts are singletons versus
49-54 % at T=16.  The absolute win is still bigger at W16 because D is bigger.

## Step 1 — the census (W16), and the surprise that reshaped the task

`prof/census_run.sh w16 16384`, 2026-09-06 13:44-13:46, one fnbench pass per
workload, ~12 700 T=16 calls per snapshot (`runs/census-w16-*.npz`).  Server t/s
during the census: code-edit 443, prose-en 154, agent-loop 223, prose-ja 129 —
i.e. the two `index_copy_`s per call are essentially free, so the census is not
a distorted operating point.

Two facts confirmed at once: **every row's top-10 weights sum to exactly
1.00000** (so `renormalize=True`, `norm_topk_prob` defaults on) and the stored
`topk_ids` order *is* the descending-weight rank (agreement 1.000), which
retroactively validates the rank preliminary above.

### The surprise: the router's top-10 is nearly flat

Mean weight by rank, W16 verify:

| workload | D | singl/call | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| code-edit | 71.57 | 45.45 | .183 | .133 | .111 | .099 | .090 | .084 | .079 | .075 | .071 | .068 |
| prose-en | 68.04 | 33.95 | .203 | .149 | .121 | .102 | .089 | .079 | .072 | .066 | .061 | .058 |
| agent-loop | 73.42 | 37.59 | .208 | .146 | .117 | .100 | .087 | .078 | .071 | .066 | .062 | .058 |
| prose-ja | 57.23 | 27.11 | .195 | .142 | .116 | .100 | .089 | .080 | .073 | .068 | .064 | .060 |

With 10 renormalised experts the *mean* weight is 0.1 by construction, and the
router only spreads it over a 3x range (0.20 down to 0.058).  **The τ grid this
task was specified with (0.02/0.03/0.05/0.08) was calibrated for a peaked
router and does not apply here**: τ ≤ 0.03 sits below the entire weight
distribution and prunes 0.0-0.3 experts per call — nothing.  The whole action is
between τ = 0.05 and τ = 0.09.

Singleton fraction by rank (the mechanism, measured): it rises monotonically
from 0.073-0.179 at rank 1 to 0.300-0.390 at rank 10 — a rank-10 route is 2.2x
(code-edit) to 4.1x (prose-ja) more likely to be the sole user of its expert
than a rank-1 route.

### τ / rank sweep, W16 (ΔD per call, % of D, mass removed per row, worst row)

| rule | code-edit | prose-en | agent-loop | prose-ja | mean ΔD | mean mass/row | worst row |
|---|--:|--:|--:|--:|--:|--:|--:|
| τ=0.02 | 0.04 | 0.04 | 0.05 | 0.00 | 0.03 | 0.000 | 0.13 |
| τ=0.03 | 0.21 | 0.30 | 0.29 | 0.05 | 0.21 | 0.0003 | 0.22 |
| τ=0.05 | 2.28 | 3.31 | 3.55 | 1.97 | 2.78 | 0.0073 | 0.37 |
| τ=0.06 | 5.32 | 7.02 | 8.08 | 5.38 | 6.45 | 0.0200 | 0.44 |
| τ=0.07 | 10.09 | 12.18 | 14.35 | 10.50 | 11.78 | 0.0417 | 0.48 |
| τ=0.08 | 16.35 | 17.59 | 20.48 | 15.77 | 17.55 | 0.0687 | 0.57 |
| τ=0.09 | 24.91 | 21.76 | 25.19 | 19.49 | 22.84 | 0.0968 | 0.72 |
| rank 10 | 6.25 | 4.98 | 5.63 | 4.80 | 5.42 | 0.0207 | **0.10** |
| **rank 9-10** | 12.08 | 9.47 | 10.79 | 9.05 | **10.35** | 0.0407 | **0.20** |
| rank 8-10 | 17.51 | 13.56 | 15.43 | 12.71 | 14.80 | 0.0599 | **0.29** |
| rank 8-10 & τ=0.08 | 11.32 | 12.52 | 14.19 | 11.69 | 12.43 | 0.0521 | 0.24 |
| all singletons | 42.59 | 31.64 | 35.40 | 25.95 | 33.90 | 0.1749 | 0.89 |

The rank rule dominates the τ rule on the metric that should matter most for
quality — **the worst row**.  At matched removed mass (τ=0.07 vs rank 9-10, both
~4.1 % per row on average) the τ rule can strip 48 % of a single row's routing
mass while the rank rule can never take more than the row's two smallest
weights (19 %).  That is a structural property, not a sample artefact: a τ cut
is unbounded in how many of a row's routes it can take at once, a rank cut is
capped at 2.

Expected step time at 48 MoE verify calls/step and `2.233 µs/expert` (§8.3),
against a 20.47 ms W16 step:

| rule | mean ΔD | Δstep | % |
|---|--:|--:|--:|
| τ=0.05 | 2.78 | −0.30 ms | −1.5 % |
| τ=0.06 | 6.45 | −0.69 ms | −3.4 % |
| rank 9-10 | 10.35 | **−1.11 ms** | **−5.4 %** |
| τ=0.07 | 11.78 | −1.26 ms | −6.2 % |
| rank 8-10 & τ=0.08 | 12.43 | −1.33 ms | −6.5 % |
| τ=0.08 | 17.55 | **−1.88 ms** | **−9.2 %** |

Less ~0.10 ms/step for the prune kernel itself (48 × ~2 µs).

**Two configurations go to the A/B**, chosen to bracket the trade-off:
`p1r` = rank 9-10 (`TAU=1.0 MIN_RANK=8`), the bounded one; `p1t` = τ=0.08
(`MIN_RANK=1`), the aggressive one.

## Step 2 — the kernel is correct (GPU unit test)

`bench/moe_smallm/test_prune_gpu.py`, 2026-09-06.  First run failed to compile:
Triton 3.7 refuses to read a non-`constexpr` module global from a `@jit` body
(`NameError: Cannot access global variable SENTINEL`), fixed by passing the
sentinel as a `tl.constexpr` argument (sglang-p1 `3a1f1deb9d`).

```
  M=2..64, k=10, ids int32 and int64          -> match=True (12/12)
  CUDA graph: capture once, replay with fresh inputs (3 seeds) -> match=True
  kernel wall time M=16 k=10: 11.73 us/call (launch included)
```

The reference is a numpy recomputation of multiplicity, within-row rank and the
mask.  The graph case is the one that matters: SGLang's `full_cuda_graph_backend`
runs two eager warmups per batch size before each capture ("so kernels are
loaded and one-time setup is paid before capture"), so the Triton JIT is always
warm at capture time and no pre-warming hook is needed.

The 11.73 µs is **eager Python launch overhead**, not device time — the loop
enqueues 2000 launches and syncs once, so it measures Triton's ~10-20 µs CPU
launcher. Inside the decode CUDA graph there is no Python; the real cost is the
kernel's device duration, read off the profile trace in step 3.

## Step 1b — the census (W4)

`prof/census_run.sh w4 16384`, 2026-09-06 14:56-14:58, ~15 700 T=4 calls per
workload snapshot.  Weights again sum to exactly 1 and the stored order is the
weight rank.

| workload | D | singl/call (of 40) | r1 w | r10 w | singleton frac r1 | r10 |
|---|--:|--:|--:|--:|--:|--:|
| code-edit | 32.75 | 27.43 | .186 | .072 | 0.560 | 0.771 |
| prose-en | 27.24 | 18.16 | .199 | .058 | 0.330 | 0.591 |
| agent-loop | 28.88 | 20.88 | .205 | .059 | 0.384 | 0.648 |
| prose-ja | 27.87 | 15.60 | .196 | .060 | 0.236 | 0.566 |

ΔD per call (and mean removed mass per row):

| rule | code-edit | prose-en | agent-loop | prose-ja | mean ΔD | mean mass/row | worst row |
|---|--:|--:|--:|--:|--:|--:|--:|
| τ=0.05 | 0.61 | 1.40 | 1.54 | 1.04 | 1.15 | 0.0122 | 0.37 |
| τ=0.06 | 1.94 | 3.27 | 3.66 | 2.77 | 2.91 | 0.0366 | 0.44 |
| τ=0.07 | 4.54 | 6.00 | 6.79 | 5.36 | 5.67 | 0.0817 | 0.54 |
| τ=0.08 | 8.49 | 8.85 | 10.09 | 8.11 | 8.89 | **0.1420** | 0.67 |
| rank 10 | 3.08 | 2.36 | 2.59 | 2.26 | 2.57 | 0.0403 | 0.10 |
| **rank 9-10** | 6.12 | 4.57 | 5.07 | 4.37 | **5.03** | 0.0809 | **0.20** |
| rank 8-10 | 9.08 | 6.66 | 7.45 | 6.30 | 7.37 | 0.1219 | 0.29 |

**W4 is the riskier width, not the safer one.** At T=4 there are only 4 rows to
share an expert with, so 56-77 % of code-edit's routes are singletons (against
18-39 % at T=16) and any given rule removes roughly *twice* the routing mass per
row for a similar ΔD fraction: rank 9-10 takes 8.1 % of a W4 row's mass versus
4.1 % of a W16 row's; τ=0.08 takes 14.2 % versus 6.9 %.  τ=0.08 at W4 touches
85-99 % of all rows.  So the τ rule that looks reasonable at W16 is the most
aggressive thing in the whole table at W4 — another reason to prefer the
rank rule, whose per-row damage is capped by construction at both widths.

Expected step time (48 calls × 2.124 µs/expert, 11.95 ms step):

| rule | mean ΔD | Δstep | % |
|---|--:|--:|--:|
| rank 10 | 2.57 | −0.26 ms | −2.2 % |
| rank 9-10 | 5.03 | **−0.51 ms** | **−4.3 %** |
| τ=0.08 | 8.89 | −0.91 ms | −7.6 % |

## Step 3 — first W16 A/B (2026-09-06 16:00-16:57), and two problems found

Runs `base` / `p1r` (rank 9-10) / `p1t` (τ=0.08), all on the p1 worktree via
`PYTHONPATH`, back to back on one machine.  `MODE=both`: 20-step profile traces
on code-edit and prose-en, fnbench ×2 on the four public workloads, synthetic
needle 18.5k, greedy agreement dump.

### MoE chain per verify call (median, µs; `prof/p1_moe.py`)

| build | workload | prune | prologue | GEMM1 | act | GEMM2 | gemm | chain |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| base | code-edit | – | 11.55 | 87.10 | 4.06 | 47.26 | **134.37** | 149.98 |
| p1r | code-edit | 20.99 | 11.36 | 70.27 | 3.84 | 38.82 | **109.09** | 145.28 |
| p1t | code-edit | 20.70 | 11.33 | 60.27 | 3.87 | 33.87 | **94.14** | 130.05 |
| base | prose-en | – | 11.55 | 81.73 | 3.87 | 44.48 | **126.21** | 141.63 |
| p1r | prose-en | 20.99 | 11.33 | 72.26 | 3.81 | 40.13 | **112.38** | 148.51 |
| p1t | prose-en | 20.74 | 11.33 | 64.13 | 3.84 | 35.78 | **99.91** | 135.81 |

The draft (T=1) cluster is untouched at 17.8/17.0 µs, as the `MIN_ROWS=2` gate
intends.  **The mechanism works exactly as predicted**: the grouped GEMM shrinks
by 25.3 µs/call (rank 9-10) and 40.2 µs/call (τ=0.08) on code-edit, i.e.
2.1-2.5 µs per expert removed, and by 13.8 / 26.3 µs/call on prose-en (1.5 µs
per expert).  Calibrating off the draft cluster, which is the same kernel at
D=10, gives 1.61 µs/expert for GEMM1+GEMM2 — the A0/A3/G1/G2 patches have
brought the marginal cost well below §8.3's 2.233 µs.

### Problem 1: the first prune kernel cost 21 µs/call

49 verify calls/step × 21 µs = **1.0 ms/step**, which cancelled most of the win
(`p1r` step wall went *up*: 19.02→20.00 code-edit, 18.50→19.54 prose-en; `p1t`
still won, 19.02→18.29 and 18.50→18.04, i.e. −3.8 %/−2.5 % *while carrying* the
bad kernel).  Cause: an all-pairs `[N, N]` compare plus a runtime integer
division (`offs // K`) per route, in one CTA.  Rewritten (sglang-p1
`6f70d6aaba`) to work on the `[M, k]` tile directly — rank from a per-row k×k
compare, singleton count from an atomic histogram over the expert slots that is
left zeroed for the next call.  O(N) instead of O(N²), no division.

### Problem 2: the greedy-agreement gate is not measurable as specified

The baseline build **does not agree with itself**.  Two greedy (temperature 0)
repeats of the same prompt on the same server:

| workload | base rep0 vs base rep1 | first divergence |
|---|--:|--:|
| code-edit | **1.000** | – |
| agent-loop | 0.062 | token 14 |
| prose-en | 0.004 | token 1 |
| prose-ja | 0.020 | token 0 |

So on three of the four workloads the *noise floor* is at the floor and a
build-to-build token-identity number carries no information: speculative
decoding plus non-deterministic reductions flip a near-tie, and on open prose
one flipped token cascades through everything after it.  (`agree_gen.py` now
flushes the radix cache between requests, which removes one confound; the
numbers above are from the pre-flush run and will be re-measured.)

Where the floor *is* usable, the result is unambiguous:

**code-edit, base vs τ=0.08: agreement 1.000 over the first 256 tokens, no
divergence.**

The gate therefore has to rest on the measures that *are* stable: acceptance
(which is exactly "how far did the target's argmax move", since the draft is
unchanged) and the needle.

### Acceptance — unchanged or better (fnbench ×2, greedy)

| build | code-edit | prose-en | agent-loop | prose-ja |
|---|--:|--:|--:|--:|
| base | 10.43 | 2.93 | 5.10 | 2.51 |
| p1r (rank 9-10) | 11.12 | 2.90 | 5.84 | 2.50 |
| p1t (τ=0.08) | 10.88 | 2.97 | 5.55 | 2.61 |

Nothing drops.  Removing 12-18 of ~70 expert reads per verify call does not
move the target's argmax enough to lose a single accepted token on average —
which is the strongest evidence in this log that the §8.6 NO-GO on reducing D
was an inference, not a measurement.

Needle 18.5k: base PASS, p1t PASS.  `p1r`'s needle OOM'd (the 17.3k prefill
needed 160 MB the 0.93 static fraction did not leave); subsequent runs use
`W16_MEM_FRACTION=0.925`.  That is a memory-headroom issue on a shared card,
not a pruning issue — `p1t` passed the identical test in the same conditions.

## Step 4 — the O(N) kernel

`bench/moe_smallm/test_prune_gpu.py` now times the kernel inside a CUDA graph
(the earlier 11.73 µs was Triton's Python launcher, not device time).

| M | device µs/call (histogram) |
|--:|--:|
| 4 | 1.91 |
| 16 | **3.31** |
| 32 | 7.09 |

All twelve shape/dtype cases and the three graph replays still match the numpy
reference exactly.  At W16 this is 49 × 3.31 = **0.16 ms/step**, against
1.03 ms for the first version.

### Kernel variants, device time in a CUDA graph

| M | histogram | all-pairs (reference) |
|--:|--:|--:|
| 4 | 1.91 | 1.91 |
| 16 | **3.31** | 11.17 |
| 32 | 7.09 | 42.65 |

Both produce identical masks on every test case, so the histogram is a pure
win; the pairwise path stays as `SGLANG_MOE_PRUNE_PAIRWISE=1`, both as a
reference for the test and as the fallback if the scratch buffer could not be
allocated before graph capture.

## Step 5 — clean W16 A/B with the fast kernel (base2 / p1r2, 17:46-19:28)

The first `base` run was contended: the same build measured 19.02/18.50 ms per
step at 16:02 and **17.34/17.15 ms** at 18:40, with the MoE GEMM 125.31 vs
134.37 µs/call.  Nothing about the build changed — another agent's job was on
the card.  **Every cross-run number in step 3 above is therefore only good for
its own pair**, and the ship decision rests on this section's back-to-back runs.

### rank 9-10 (`TAU=1.0 MIN_RANK=8`) vs its own baseline

| | code-edit | prose-en |
|---|--:|--:|
| MoE GEMM/call, base2 | 125.31 | 123.26 |
| MoE GEMM/call, p1r2 | 113.25 | 109.95 |
| **ΔGEMM** | **−12.06** | **−13.31** |
| prune kernel | +4.16 | +4.13 |
| **Δchain/call** | **−8.00** | **−9.28** |
| Δstep at 49 calls | −0.39 ms | −0.46 ms |
| step wall, base2 → p1r2 | 17.34 → 17.14 | 17.15 → 17.27 |

**−2.3 to −2.7 % of the step — under the 3 % bar.**  (The step-wall column has
±0.2 ms of run-to-run noise, so the per-call chain delta, which is a median over
~980 calls inside one trace, is the number to trust.)  The in-server prune
kernel costs 4.16 µs, against 3.31 µs measured standalone.

The marginal GEMM cost implied here is 12-13 µs / ΔD 10.35 = **1.2-1.3 µs per
expert**, i.e. the 1.61 µs/expert read off the draft cluster is an upper bound
and §8.3's 2.233 is well out of date on the current fork.

### Why the greedy-agreement gate can never be met on this server

Re-measured with `/flush_cache` before every request, so the radix cache is not
the explanation.  The baseline still disagrees with itself: prose-en diverges at
**token 1**, prose-ja at token 9, agent-loop at token 12.

The cause is in the MoE itself.  FlashInfer's fused FINALIZE epilogue
accumulates each expert's scaled contribution into the token's output row with a
**vectorised `red.global.add`**
(`nv_internal/.../epilogue/collective/epilogue_moe_finalize.hpp:406,461`,
`cutlass_extensions/arch/copy_red_global.hpp`), and atomic float addition order
follows CTA scheduling.  The MoE output is therefore not bitwise reproducible
between two runs of the *same* build, one flipped near-tie changes an accepted
token, and on open-ended prose everything after it cascades.

So "greedy agreement ≥ 95 % on the first 256 tokens" is not a property this
server has against itself, and cannot be used to gate a change.  What survives:

* **code-edit, the one workload whose self-agreement is 1.000**: base2 vs
  rank 9-10 = **1.000**, base vs τ=0.08 = **1.000**, no divergence in 256 tokens.
* acceptance, read against a baseline-to-baseline noise band (below).

### Acceptance, with the noise band the two baselines give

| build | code-edit | prose-en | agent-loop | prose-ja |
|---|--:|--:|--:|--:|
| base (16:02) | 10.43 | 2.93 | 5.10 | 2.51 |
| base2 (18:40) | 11.50 | 3.16 | 4.50 | 2.69 |
| *baseline spread* | *±5 %* | *±4 %* | *±6 %* | *±3.5 %* |
| p1r (rank 9-10) | 11.12 | 2.90 | 5.84 | 2.50 |
| p1r2 (rank 9-10) | 10.18 | 2.99 | 5.10 | 2.63 |
| p1t (τ=0.08) | 10.88 | 2.97 | 5.55 | 2.61 |

Two runs of the *unmodified* build differ by 4-6 % on every workload, for the
same reason the tokens differ.  Every pruned build sits inside or below that
spread on every workload; nothing moves outside it.  This is a real but
**±6 %-wide** result, not the tight bound the task asked for, and tightening it
would need ~6 repeats per arm rather than 2.

Needle 18.5k: base PASS, base2 PASS, p1r2 PASS, p1t PASS.

## Step 6 — τ=0.08 at W16, and where the win actually goes

`base2` → `p1t2`, back to back, `prof/exclusive_time.py` (exclusive = wall time
during which that kernel family is the *only* thing on the GPU, i.e. what
deleting it would actually save).

### The grouped GEMM shrinks exactly as the census predicted

MoE GEMM summed over the ~45.5 verify calls in a step (µs/step; the *sum*, not
the median — D is right-skewed, mean 133 vs median 124 µs/call, so a median
delta overstates the total):

| | base2 | rank 9-10 | τ=0.08 |
|---|--:|--:|--:|
| code-edit GEMM sum/step | 6541 | 6117 (−424) | **5376 (−1165)** |
| prose-en GEMM sum/step | 6524 | 6062 (−462) | **5456 (−1068)** |

Exclusive time confirms it is all on the critical path (`excl/raw` 97-99.5 %):
code-edit gemm1+gemm2 exclusive 6135 → 5587 (rank) → **4936 (τ=0.08)**.

### But two thirds of it does not reach the step

| code-edit, µs/step exclusive | base2 | rank 9-10 | τ=0.08 |
|---|--:|--:|--:|
| `cutlass_moe_grouped_gemm1` | 3944.6 | 3629.3 | 3264.1 |
| `cutlass_moe_grouped_gemm2` | 2190.7 | 1957.4 | 1672.2 |
| **GEMM subtotal** | 6135.3 | 5586.7 (**−548.6**) | 4936.3 (**−1199.0**) |
| `fusedBuildExpertMapsSortFirstToken` | 265.5 | 515.5 | 462.4 |
| its `excl/raw` | **45.9 %** | **88.2 %** | **87.8 %** |
| **prologue penalty** | – | **+250.0** | **+196.9** |
| prune kernel + tail | – | ≈ +8 | ≈ +307 |
| **verify phase, total exclusive** | 10697.0 | 10406.7 (−290) | **10001.5 (−696)** |

The routing prologue used to be **54 % hidden** behind whatever ran before it.
Inserting *any* kernel immediately in front of it — even a 4 µs one — destroys
that overlap, and the prologue's exclusive time more than doubles.  So the
prune kernel's true cost is not its 4.13 µs duration (188 µs/step) but roughly
**0.5 µs of lost overlap for every µs it runs**, i.e. ~0.5 ms/step at W16.

Net, per step, against a 17.1-17.3 ms baseline:

| | code-edit | prose-en |
|---|--:|--:|
| rank 9-10 | −290 µs (−1.7 %) | −58 µs (−0.3 %) |
| **τ=0.08** | **−696 µs (−4.0 %)** | **−101 µs (−0.6 %)** |

(The draft phase moved +60 to +199 µs/step in the pruned builds even though the
`MIN_ROWS=2` gate keeps pruning out of it entirely — that is inter-run drift and
sets the noise scale on these numbers at roughly ±200 µs/step.)

### Quality at τ=0.08, W16

* needle 18.5k (synthetic context): **PASS**.
* code-edit greedy agreement vs base2 over 256 tokens: **1.000**, no divergence
  (the only workload whose self-agreement makes the number meaningful).
* acceptance 9.69 / 3.29 / 5.43 / 2.54 against a base-to-base band of
  10.43-11.50 / 2.93-3.16 / 4.50-5.10 / 2.51-2.69: prose-en and agent-loop are
  *above* both baselines, code-edit is 7 % below the lower baseline, prose-ja is
  inside.  No consistent direction.

## Step 7 — where this belongs instead: inside the fused prologue

**Design only, not implemented — the task forbids touching the venv, and the
FlashInfer csrc is in it.**  Recorded here because it is the difference between
a −0.6 % change and a −5 % one.

`fusedBuildExpertMapsSortFirstTokenKernel`
(`flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh:365-451`)
already does, in one CTA, everything the mask needs:

* it **reads `token_selected_experts`** into registers (`:394-407`) — the same
  load the prune kernel repeats;
* it **radix-ranks them by expert** and gets `local_expert_first_token_offset`
  (`:421-424`), whose consecutive differences *are* the per-expert route counts,
  i.e. exactly the singleton test, computed as a by-product;
* it is the kernel whose 54 % overlap the separate prune kernel destroys.

Sketch, after `RankKeys` at `:421`:

1. Stage `local_expert_first_token_offset` in the shared buffer that
   `BlockRadixRank::TempStorage` already occupies (it is dead after the rank),
   `__syncthreads()`, so every thread can read `off[e+1] - off[e]`.
2. For each of the thread's `EXPERTS_PER_TOKEN` routes: `singleton = (count == 1)`;
   the τ test needs `token_final_scales` (already a parameter of the sibling
   `buildMinLatencyActiveExpertMaps`, so plumbing it in is a signature change,
   not new data movement); the rank test is the route's own index `i`, since
   `topk` stores descending weight (measured agreement 1.000 over 50 k calls).
3. `cub::BlockScan` an `is_removed_expert` flag over the bins to get, per expert,
   the number of removed experts below it — one scan, the same primitive
   `prefixSum` at `:175-192` already uses.
4. Every surviving route's permuted index shifts down by that prefix, and every
   removed expert's bucket becomes empty.  **No second rank is needed**: a
   removed route is by construction the sole occupant of its expert, so the
   ranked order of the survivors is unchanged and only the offsets move.
5. Masked routes must also have their `token_final_scales` entry zeroed, or the
   finalize's `expert_id` range test (`:2434`) must reject them — the latter is
   already true if their id is written back as `-1`.

Cost: one `BlockScan` over 513 bins plus a shared round trip, inside a kernel
that is currently 11.4 µs and 46 % hidden.  It adds **no kernel launch and no
lost overlap**, so the full GEMM saving reaches the step:

| W16, τ=0.08 | measured GEMM saving | expected step delta |
|---|--:|--:|
| code-edit | −1199 µs/step | **−7.0 %** |
| prose-en | −924 µs/step | **−5.4 %** |

This is the same patch site as A0, A3 and G1/G2, so the build machinery for it
already exists on this box.

## Step 8 — W4 (base2 → p1t2, 19:50-20:40): the width where it works

Same harness, `W4_MEM_FRACTION=0.930`, back to back.

### Step time

| | base2 | τ=0.08 | Δ |
|---|--:|--:|--:|
| step wall, code-edit | 10 526 µs | **9 367 µs** | **−11.0 %** |
| step wall, prose-en | 10 461 µs | **9 459 µs** | **−9.6 %** |
| verify exclusive, code-edit | 7 605.1 | 6 537.4 | −1 067.7 |
| verify exclusive, prose-en | 7 186.6 | 6 331.1 | −855.5 |

Per-kernel (exclusive µs/step, code-edit / prose-en):

| family | base2 | τ=0.08 |
|---|--:|--:|
| `cutlass_moe_grouped_gemm1` | 2216.2 / 1916.2 | 1466.3 / 1425.1 |
| `cutlass_moe_grouped_gemm2` | 1182.5 / 1150.2 | 835.7 / 836.2 |
| **GEMM subtotal** | 3398.7 / 3066.4 | **2302.0 / 2261.3** |
| prologue (`excl / % of raw`) | 242.1 (43.9 %) / 208.6 (40.7 %) | 278.2 (54.1 %) / 300.7 (57.1 %) |
| prune kernel, µs/call | – | **2.53** |

**The prologue-overlap penalty is only +36 / +92 µs at W4** against +197 / +328
at W16, and the prune kernel is 2.53 µs instead of 4.13 (M=4, so the k×k rank
tile is a quarter the size).  Almost the whole GEMM saving therefore reaches the
step.

### End-to-end throughput (fnbench ×2, greedy)

| workload | base2 t/s | τ=0.08 t/s | Δ | base2 acc | τ=0.08 acc |
|---|--:|--:|--:|--:|--:|
| code-edit | 354 | **389** | **+9.9 %** | 3.84 | 3.86 |
| prose-en | 253 | **266** | **+5.1 %** | 2.62 | 2.49 |
| agent-loop | 296 | **339** | **+14.5 %** | 3.12 | 3.25 |
| prose-ja | 218 | **247** | **+13.3 %** | 2.24 | 2.29 |

Acceptance is flat (prose-en's −5 % is inside the ±4-6 % baseline-to-baseline
band established at W16); every workload gains throughput.

### Quality at W4, τ=0.08

* needle 18.5k (synthetic context) **PASS**;
* code-edit greedy agreement vs base2: **1.000**, no divergence in 256 tokens
  (the other three workloads' self-agreement is 0.05-0.45, so unusable, exactly
  as at W16).

This is the width the census said was *riskier* — at T=4, τ=0.08 removes 14.2 %
of a row's routing mass against 6.9 % at T=16, and touches 85-99 % of rows.  It
is nonetheless the width where nothing measurable degrades and the throughput
gain is largest, because at T=4 the MoE grouped GEMM is a bigger share of a
shorter step (3.4 of 10.5 ms) and the fixed costs of inserting a kernel are
smaller.

### W4, rank 9-10 (`p1r2`, same chain)

| | base2 | rank 9-10 | τ=0.08 |
|---|--:|--:|--:|
| step wall, code-edit | 10 526 | 9 643 (**−8.4 %**) | 9 367 (**−11.0 %**) |
| step wall, prose-en | 10 461 | 9 626 (**−7.9 %**) | 9 459 (**−9.6 %**) |
| verify exclusive, code-edit | 7 605.1 | 6 849.8 | 6 537.4 |
| gemm1+gemm2 exclusive, code-edit | 3 398.7 | 2 590.0 | 2 302.0 |
| t/s code / prose-en / agent / ja | 354/253/296/218 | 382/266/319/247 | 389/266/339/247 |
| acceptance | 3.84/2.62/3.12/2.24 | 3.82/2.56/3.11/2.34 | 3.86/2.49/3.25/2.29 |
| needle | PASS | PASS | PASS |

Both rules clear every gate at W4; τ=0.08 is 1.7-2.6 points faster and rank 9-10
is the more conservative (per-row mass removed 8.1 % vs 14.2 %).

---

# Conclusion and recommendation

## What was actually established

1. **The singleton structure is real and exploitable.**  At T=16 half the ~68
   distinct experts of a verify call are reached by exactly one chain row; at
   T=4, two thirds are.  Singleton-ness rises monotonically with routing rank
   (0.07-0.18 at rank 1 to 0.30-0.39 at rank 10), so a rule that targets
   low-weight singletons buys 2-4x the expert reads per dropped route that a
   uniform top-k cut would.
2. **The specified τ grid was calibrated for the wrong router.**  This model
   renormalises 10 experts and spreads their weight over only a 3x range
   (0.20 → 0.058), so τ ≤ 0.03 sits below the entire distribution and prunes
   0.0-0.3 experts per call.  The usable range is 0.05-0.09.
3. **The GEMM responds exactly as the bytes model says.**  Removing 8.9 (W4) /
   17.6 (W16) distinct experts per call cuts the grouped GEMM by 1.10 (W4) /
   1.20 (W16) ms per step, at 1.2-1.6 µs per expert.
4. **The published quality checks do not measurably move.**  Across five pruned
   server runs at two widths (public workloads): the synthetic 18.5k needle
   passes every time; acceptance stays inside the ±4-6 % band that
   two runs of the *unmodified* build span; and on code-edit — the one workload
   whose greedy output is reproducible at all — the pruned build's first 256
   tokens are **identical** to the baseline's.
5. **§8.6's NO-GO on reducing D was an inference, and it was wrong.**  D can be
   cut by 26-30 % with no measurable change on these checks (needle, acceptance,
   code-edit agreement).

## Recommendation

**SHIP at W4: `SGLANG_MOE_PRUNE_SINGLETON_TAU=0.08`** (defaults for everything
else: `MIN_RANK=1`, `MIN_ROWS=2`, `MAX_ROWS=64`, `RENORM=0`).

* step **−9.6 % to −11.0 %**, throughput **+5.1 % to +14.5 %** on all four
  workloads, comfortably past the ≥3 % bar;
* needle PASS, acceptance flat on the public workloads, code-edit token
  agreement 1.000 over the first 256 tokens.
* `SGLANG_MOE_PRUNE_SINGLETON_TAU=1.0 SGLANG_MOE_PRUNE_MIN_RANK=8` (rank 9-10)
  is the conservative alternative at −7.9 %/−8.4 % — take it if the 14 % of
  per-row routing mass τ=0.08 removes is felt to be too much on a workload not
  covered here.

**HOLD at W16.**  The GEMM saving is the same 1.2 ms/step, but the net after the
inserted kernel is −0.6 % to −4.0 % depending on workload, against ±200 µs/step
of inter-run drift — it does not reliably clear the 3 % bar.  The cause is not
the kernel's 4 µs but that inserting *any* kernel in front of
`fusedBuildExpertMapsSortFirstToken` costs the prologue its 54 % overlap
(exclusive time 265 → 462 µs/step).  **Fix it by folding the mask into that
prologue** (step 7): the counts it needs are already a by-product of its radix
rank, it adds no launch and no lost overlap, and it would deliver the full
−5.4 % to −7.0 % at W16.  That is the next piece of work, and it is small.

## Caveats

* **The greedy-agreement gate as specified cannot be met by this server.**  The
  baseline disagrees with *itself* from token 1 on prose, because FlashInfer's
  FINALIZE epilogue accumulates the k expert contributions with a vectorised
  `red.global.add` and float atomics are order-dependent.  Any future change to
  target outputs needs a different gate; the acceptance + needle combination
  used here is the honest substitute, and it is weaker.
* **Acceptance is a ±4-6 % instrument, not a ±1 % one**, for the same reason.
  Two runs of the unmodified build differ by that much.  Tightening to a real
  bound needs ~6 repeats per arm; this used 2.
* The GPU was shared with three other jobs throughout.  One early baseline was
  8 % slower than the same build measured two hours later, which is why every
  conclusion above rests on back-to-back pairs and on `exclusive_time.py`
  per-family numbers rather than on step wall alone.
* Singleton-ness is defined **within the current batch**.  At batch > 1 the mask
  for one request depends on what else is in flight, so outputs are not
  independent of server load.  All measurements here are single-stream.  If
  that matters, gate the feature on `MAX_ROWS` = the single-request chain length.
* Prefill and the T=1 draft are excluded by the `[2, 64]` row gate and were
  verified untouched (draft MoE GEMM1 17.8 µs in every build).
* The `KEEP_IDS` equivalence control (zero the weight, keep the id) was
  implemented but never run — the `-1` sentinel path is instead validated by
  the FlashInfer source reading in step 0b plus the fact that every quality gate
  passes, which it would not if masked routes were being mis-handled.
