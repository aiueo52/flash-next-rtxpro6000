# P3 — contribution-aware expert pruning

Status: **P3 capture and CPU analysis complete, 2026-09-07**. Eight unpruned
captures contain 316,272 target-layer calls / 27,868,800 routes. Joint singleton
plus count-two removal has the smallest p95 at the requested additional-D target
in every workload/width. A train-only norm-table proxy with one threshold per
width also clears the D targets on all eight temporal holdouts while keeping
p95 below tau=.08. **GO for a prologue prototype and fresh quality/speed gates;
not a production shipping approval or a measured +3% speed result.**

Successful capture/analysis label: `p3-20260907-2300`. Full tables and proxy
results are below; failed attempts are retained as dated execution evidence.

## Scope and isolation

- Worktree: `$HOME/tools/sglang-p3`, branch `codex/p3-contrib`, from
  `codex/perf-v1` (`14d4c4c985`). No commit or push.
- Production fork, launcher and venv remain unmodified; use the production interpreter
  with the worktree `python` directory on `PYTHONPATH` (C1 overlay).
- Each GPU/server run owns `flock -w 28800 $HOME/.gpu.lock` and releases
  it on completion. Cleanup targets only that run's process group.
- All capture arms: `SGLANG_MOE_PRUNE_SINGLETON_TAU=0`,
  `FLASHINFER_MOE_PRUNE_SINGLETON_TAU=0`, `MEM_FRACTION=0.920`,
  `W16_MEM_FRACTION=0.920`, `WA_MEM_FRACTION=0.920`, `SERVE_DISPLAY_HZ=`.
- Capture four census workloads at W4 and W16. Timestamp all new output labels.
  Keep at least 4 GiB free VRAM at steady state; loading transients are not a start failure.

## Required evidence

1. Flag-off recorder path and numerical isolation validation.
2. Per-route NPZ captures with explicit target-verify/layer attribution, immutable
   ids/weights, expert output L2 norm before weight, pre-MoE input/residual norm,
   and original-call P2 singleton eligibility.
3. True-score singleton/global, layer-equalized and joint count-two frontiers against
   the protected-top-1 P2 tau=0.08 baseline; contribution sum mean/p95/p99 per row.
4. Held-out cheap-proxy recall and actual contribution tails; calibrated layer tables.
5. Implementation specification and a non-inferiority quality gate before shipping.

Timing projections use delta-D times **69.9 microseconds per step**, with no call
multiplier. Norm-sum scores estimate local perturbation size; they do not measure
acceptance, downstream error, or task quality.

## Initial checklist (before execution)

At creation, recorder, correctness tests, all eight GPU captures and CPU frontier/proxy analysis were pending. Final completeness is at the end of this report.

## Recorder implementation (2026-09-07, before GPU results)

The chosen debug path is **same-shape one-hot-finalize replay**, a route-unmixing
variant of unfused debug execution. For each original target-verify MoE call, run
that original call first and retain its output unchanged. Then run the same
quantized expert kernel k additional times on private routing/output buffers,
with one routing slot's final weight set to 1 and all others set to 0. The ids,
[T,k] geometry, expert multiplicities, weights/scales, activation quantization,
and GEMM shapes stay the same. Each auxiliary output is one F_e(h), rounded to
BF16 at the normal output boundary **before the original router weight**.
This avoids a separate dequantized Torch implementation's NVFP4 activation-error
mismatch and avoids materializing dequantized expert banks. It costs k extra
MoE evaluations, so capture timing is not a speed benchmark. The cheaper expanded
[T*k,1] dispatch was not selected because it changes tactic/prologue geometry.

`sglang-p3/python/sglang/srt/layers/moe/contrib_recorder.py` is imported only
with `SGLANG_MOE_CONTRIB_LOG` set; the model's explicit forward mode restricts
recording to target verification, not draft extension or draft decoding. Normal
calls use the unchanged implementation and return their own output. Auxiliary
calls use clones of ids and private weights/output; they never feed the model.
Every call checks that the original output and routing tensors remain unchanged.
The recorder also measures the relative L2 residual of reconstructing the
original MoE output as sum(w*F), to quantify BF16 finalize/rounding differences.
A 2.5% maximum relative-L2 fail-closed guard is a predeclared instrumentation
sanity bound, not a claim that this much error is harmless for pruning.

The input to the routed experts is an H-wide HC-mixed vector; the actual
pre-MoE residual is the HC-expanded tensor in `residual[0]`. These are **different
vectors** in Qwen4-Exp. Primary scores divide by the full L2 norm of that
pre-MoE residual. `input_norm` separately records the H-wide expert-input norm.
Consequently these scores are local pre-HC-injection estimates; they do not
include the learned HC injection/combination gates or cancellation between
expert vectors. Sum(w*||F||)/||residual|| bounds the uncombined removed vector
norm by the triangle inequality, not final model/output error.

### NPZ schema v1

The filenames are `runs/census-p3-<timestamp>-w{4,16}-<workload>.npz`, alongside
existing census NPZs. `allow_pickle=False` is sufficient. The compatible
`call/T/k/ids/w` keys use the existing census convention, without padding beyond
T*k. The recorder is reset while idle immediately before each workload and
dumped while idle immediately after it, preventing startup/previous-workload
contamination. Capacity is 262,144 target-layer calls; overflow fails capture
validation, rather than silently treating a ring tail as a full workload.

| key | dtype / shape | meaning |
|---|---|---|
| schema_version, role, method | scalar integer / Unicode | 1, target_verify, method name |
| call | int64 [N] | contiguous target-layer call counter, reset per workload |
| T, k | int16 [N] | verify row count and routes per row |
| layer | int16 [N] | explicit target layer id, 0–47 |
| ids | int32 [N,T*k] | immutable original expert ids, row-major |
| w | float32 [N,T*k] | original router weights; no renormalization |
| expert_norm | float32 [N,T*k] | L2 of unweighted auxiliary BF16 F_e(h) |
| h_norm | float32 [N,T] | L2 of full pre-MoE HC residual, the score denominator |
| input_norm | float32 [N,T] | L2 of actual H-wide input to MoE |
| singleton | bool [N,T*k] | original-call multiplicity exactly 1 |
| p2_drop | bool [N,T*k] | singleton, not stable first argmax, float32 w < float32(.08) |
| reconstruction_rel | float32 [N,T] | ||sum(w F)-served_MoE|| / ||served_MoE|| |
| unchanged | bool [N] | served output and original ids/weights unchanged after debug calls |
| total_calls | int64 scalar | calls since reset; must equal stored N |
| metadata | Unicode JSON scalar | pruning, memory, display and model env |

The CPU loader requires contiguous calls, complete ordered 48-layer steps,
all expected fields, finite positive denominator norms, finite nonnegative
expert norms, correct singleton/P2 flags and passing reconstruction/isolation.
Unlike the old width-only census's 49-call burst, P3 excludes the draft-extend
MoE explicitly. Delta-D is computed against tau=.08 on **this same P3 sample**.

### Replay semantics and planned analysis

All policies preserve each row's stable first maximum router-weight route.
Global singleton cuts use s < threshold. Joint cuts consider original
multiplicity one or two, rank an entire expert by sum(s) over its routes, and
remove it only if none of its routes is protected. No incremental recount or
survivor renormalization is allowed. Layer cuts use per-layer thresholds that
spend a common training mean sum(s_removed) per layer/token row; ties are kept
whole, and saturated layers may underspend.

Threshold grids and expert-norm tables train on the first half of complete
verify steps of all four workloads at each width. The second half is held out.
Rows with zero removal remain in mean/p95/p99. The best qualifying test-frontier
point per workload is an **oracle screening choice**; selecting it on those
same test samples does not make it a validated deployable threshold. Full CSV
frontiers and threshold tables permit independent choices and later validation.
The baseline uses an explicitly float32 .08 threshold to match C++; the older
simulator's float64 conversion treats exactly-stored float32(.08) differently.
Its existing behavior/selftest remain unchanged.

CPU check: original `prune_policy_sim.py --selftest` PASS and seven P3 unit tests
PASS, using the production interpreter read-only (`PYTHONDONTWRITEBYTECODE=1`).
The benchmark venvs and system Python have no NumPy; no packages were installed
or changed in production. GPU execution requires the host permission path
because the sandbox exposes no `/dev/nvidia*`; the initial sandbox lock waiter
was stopped and replaced with the same locked preflight on the host.


## Prologue implementation specification

This is a proposed implementation, not a shipped kernel change. No true
expert output is available before dispatch, so **the oracle score cannot run
in the pruning prologue**. The minimum runtime inputs for a calibrated proxy
are immutable `ids[T,k]`, original FP32 `w[T,k]`, active-row validity mask,
layer id, width/policy id, a read-only per-layer/expert mean-norm table, and
per-row inverse pre-MoE residual norm. The table is 48*512 values: 48 KiB in
FP16 or 96 KiB in FP32. Start in FP32 and separately validate table compression.
The current layer needs only 2 KiB in FP32. Scores are
`w * table[layer,expert] * inv_residual_norm[row]`. Thus f(w)=w in the first
prototype; predicting from router-logit features should be a separately
held-out comparison if this simple table misses the oracle ranking.

The HC normalization producer already reads the expanded residual and
computes row statistics. Export its inverse L2 norm as one FP32 scalar per
row (derive from the same sum of squares, accounting for epsilon and vector
length); do not add a full H*hc_count read in front of the routing prologue.
This output needs producer/consumer ordering compatible with PDL. If that
scalar cannot be exposed cheaply, a calibrated mean inverse norm per layer
is a distinct, less input-sensitive proxy that must be evaluated separately.
A table of mean ||F||/||h|| is not algebraically identical to dividing a mean
||F|| by the current ||h||; do not silently interchange them.

Extend the prologue request with the proxy table pointer, per-row inverse
norm pointer, threshold(s), policy enum and **separate output** routing/mask
buffers. Each CTA must compute original multiplicities from immutable routes.
Singleton: protect stable top-1 and compare proxy score to a global/layer cut.
Count-two: accumulate the two scores and an OR of route protection into
shared expert bins; all routes of the expert receive one common decision.
Deterministic addition/tie handling is required. No read may observe a
partially pruned id/weight, and no candidate may become a singleton because
another route was already removed. A per-row cap, if added later, requires
a deterministic joint-group conflict rule across both rows; independent
row-local cap checks cannot implement all-or-none pair deletion.

P2 block-0 in-place writes are **not valid by inheritance** for this policy.
All blocks read the original ids/weights. Re-key only registers for local
ranking, write a distinct effective id/weight or keep-mask buffer, and make
unfused finalize consume that effective routing only after kernel completion.
Fused finalize continues to use the compacted surviving route permutation.
An output buffer of T*k ids plus FP32 weights costs at most 1,280 bytes at
W16/k10; mask-only variants need careful finalize integration. Inputs remain
const across CTAs and graph replays. Do not assume CUDA block execution order.

Fail closed to unpruned execution for unsupported EP/TP, invalid-row metadata,
missing calibration, nonfinite scores or unrecognized shapes. Initially
support EP=TP=1 and explicitly verified W4/W16 only, with the P2 prologue's
existing <=32-row eligibility. Preserve at least one route per valid row,
original surviving weights, shared experts, HC residual/gates and all GDN/QSA
state updates. A genuine short verify excludes invalid rows before histogram
construction. Separate concurrent requests change unions and need validation.

The scoring overhead budget is whatever remains after `69.9*delta_D` exceeds
the measured 3% throughput target; there is no free extra call multiplier.
W4 delta_D=3.7 implies 258.63 us/step and W16 6.7 implies 468.33 us/step.
The requested W4 rounded target is slightly below the review's exact ~3.72
expert/260-us threshold, so merely reaching 3.7 is not a measured +3% result.

## Quality and speed gate before shipping

The checked Q1 report has no literal section named "gate"; its audit, caveats,
and the review's Finding 3 shipping gate are the relevant source. The current
`bench/quality/analyze.py` already contains Q2's corrected comparison sign
and one-sided non-inferiority reporting. P3 does not modify that concurrent
work. (Its Wald interval, which collapses at zero discordance, has since been
replaced by Tango score bounds, which stay valid there.)

1. Freeze calibration and one candidate before a fresh holdout. Include the
   actual deployment request mix, code correctness, tool/JSON schema validity,
   English/Japanese text, long context, and reasoning if served. These four
   census prompts are a screening dataset, not a generalization certificate.
2. Compare candidate versus current tau=.08; include tau=0 and repeated .08
   anchors. Explicitly exercise bs=1 at W4 and W16 and every adaptive policy
   intended for deployment. Larger request unions may prune less; they do not
   conservatively cover bs=1. Fix the MTP model and width for each comparison.
3. Predeclare a 0.5 percentage-point broad-correctness non-inferiority margin
   (or an explicitly approved replacement) and require the paired one-sided
   95% lower bound of candidate minus baseline to exceed -0.5 pp. Correctness
   losses, schema failures and needle regressions cannot be rescued by an
   insignificant two-sided p-value. INCONCLUSIVE remains unshippable.
4. Score all prompts at the intended generation cap. Diagnose and rerun
   cap-limited cases with enough allowance; never decide from only the
   both-finished subset. Include needle 18.5k at depths .1/.5/.9 before/after
   and long-generation degeneration checks. A quality audit of this size can reject a
   bad policy; it cannot certify a tight accuracy margin.
5. Validate immutable-routing outputs against a CPU oracle and existing
   tau=.08 behavior, across ties, singletons, count-two groups, protected top-1,
   invalid/padded rows and changing CUDA graph inputs. Check actual removed
   output-vector errors, same-prefix logits/argmax flips, and proxy tail drift.
6. Run matched repeated throughput arms with recorder disabled, same memory
   fractions .920, `SERVE_DISPLAY_HZ=`, same lock and cleanup discipline.
   Require >=3% emitted tokens / total decode time versus .08, including all
   scoring/dispatch, acceptance, recovery and switching costs. Capture-step
   t/s from this debug recorder is not admissible speed evidence.

GPU preflight/captures and CPU numerical tables remain pending as of this
incremental specification update. No production pruning is enabled by P3.

Additional CPU verification: an eight-file **synthetic-only temporary fixture**
passed the complete loader → temporal split → calibration → 2,448-point sweep →
proxy evaluation → CSV/Markdown writer pipeline. All eight synthetic workloads
had a selected point; fixture outputs were removed with the temporary directory.
`runs/p3-cpu-integration.log` is explicitly synthetic and is not evidence about
the model. Post-readiness VRAM telemetry samples every 5 seconds; three
consecutive samples below 4 GiB terminate only the run's own server process group.

First host GPU preflight reached the real weight loader, then stopped before
kernel execution because `ninja` was absent from PATH. The preflight environment
now includes the existing production venv's `bin` directory (read-only), matching
the launcher's environment; retry uses a new timestamped label.

The layer calibration implementation caches each layer's sorted cost curve
once for the entire threshold sweep. `layer-error.csv` will retain both
training calibration means and held-out per-layer mean/p95/p99, making the
claimed equalization and its holdout drift inspectable. Norm-table training
and threshold calibration pool calls across workloads (they do not give each
workload equal statistical weight). Fnbench runs one warmup generation plus
one measured generation per workload, just as the existing census launcher;
both belong to the captured workload. Temporal holdout therefore tests new
serving states/repeats of the same four prompts, not unseen-prompt generalization.

Second GPU preflight (22:39 JST) completed the six W4/W16 real-weight unmix,
reconstruction, independent single-route and fresh-input graph assertions,
then found a recorder-specific CUDA graph error: constructing a CUDA tensor
from the host .08 scalar during capture. Replaced it with PyTorch's scalar
comparison (cast to the FP32 tensor dtype in the kernel). No capture was allowed
past that failed preflight. The full preflight, including actual ring recording
and dump, is repeated rather than treating the helper checks as sufficient.

Reproducible CPU commands (NumPy comes from the unchanged production interpreter):

```bash
PYTHONDONTWRITEBYTECODE=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python bench/quality/prune_policy_sim.py --selftest
PYTHONDONTWRITEBYTECODE=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python -m unittest discover -s bench/quality -p 'test_contrib_policy_sim.py' -v
PYTHONDONTWRITEBYTECODE=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python bench/quality/p3_fixture_selftest.py
```

## GPU preflight PASS (22:48 JST)

`runs/p3-preflight-20260907-2240.json` and `.log`: six real NVFP4 layer-4
cases (three recorded route samples each at W4/W16) passed reconstruction,
independent k=1 route isolation, unchanged output/routing and fresh-input CUDA
graph norm checks. The actual recorder graph/ring/dump test also passed:
1 stored call, unchanged=true, reconstruction relative L2 max=0.0049356651.
The dummy HC residual was refreshed before graph replay and its recorded L2
norm matched the refreshed values. Its temporary NPZ was a preflight fixture,
not a workload capture, and was removed. W4 and W16 full capture runs are now
queued separately; each acquires/releases the shared lock independently.

## First W4 start: steady-state memory guard worked; capture rejected

Run `p3-20260907-2240-w4` reached readiness at 22:53:28 with 4,904 MiB free,
then code-edit prefill raised workspace reservation: post-readiness samples
were 4,858 / 3,415 / 3,489 / 3,446 MiB. Three below-4-GiB samples triggered
termination of **only that run's own process group** at 22:53:43; no workload
NPZ was accepted. This was a steady decode-window observation, not a loading
transient. See its memory logs and `memory-failed.txt`.

The W4 launcher defaults to 24 Mamba slots, whereas W16 already defaults to
10. P3 now explicitly uses **MAMBA_SLOTS=10 for both widths**, enough for the
single fnbench request and identical to W16's existing setting. This reduces
unused state-pool allocation; all memory fractions remain .920, all served
MoE math and recording logic remain the same. W16 had already started with
10 slots and continues through the same readiness/continuous memory guard.
A fresh timestamped W4 retry is queued, rather than overwriting failed logs.

The first W16 run (`p3-20260907-2249-w16`) reached readiness but its controller
then hit a shell parse error. Cause: I edited the shared shell script while
that shell was executing it, invalidating its buffered file position. This was
an orchestration mistake, not a model/kernel failure. The EXIT trap cleaned up
that run's process group and no workload NPZ was produced. The current script
passes `bash -n`; the W16 retry executes the frozen, read-only snapshot
`runs/p3-capture-script-20260907-2257.sh`. Active run scripts are no longer edited.
The already-running W4 retry uses the validated script with 10 Mamba slots.

## Memory diagnosis and fixed capture capacity

The 10-slot W4 retry also tripped the steady guard (4,549 MiB at readiness,
then 3,127 / 3,137 / 3,137 MiB). The detailed allocation log explains why the
slot reduction did not help: automatic KV sizing filled the newly available
budget, growing the target cache to **677,184 tokens / 7.76 GiB**, plus the
draft cache. The first real code prefill reserved another ~1.4 GiB. Merely
changing the state pool could not leave the needed workspace reserve.

Final capture configuration therefore adds **MAX_TOTAL_TOKENS=131072 identically
at W4 and W16**, while retaining all three .920 fractions and MAMBA_SLOTS=10.
The largest census prompt+generation is far smaller than this capacity; this
caps unused KV storage without changing attention context, pruning or model
math. The immutable controller is `runs/p3-capture-script-20260907-2300.sh`.
The superseded W16 start was stopped using only its own process group to apply
this known steady-state fix; this was not a reaction to loading-time VRAM dips.
No workload NPZ has been accepted from any failed/superseded run.

Capture landed: w4 / code-edit: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w4-code-edit.npz`, 59088 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.007611.

Capture landed: w4 / prose-en: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w4-prose-en.npz`, 45072 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.005758.

Capture landed: w4 / prose-ja: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w4-prose-ja.npz`, 47856 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.006300.

Capture landed: w4 / agent-loop: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w4-agent-loop.npz`, 37440 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.006151.

All four W4 captures completed (23:04:39–23:08:15 JST) with the fixed KV
capacity; code-edit/prose-en/prose-ja/agent-loop contain respectively
59,088 / 45,072 / 47,856 / 37,440 target-layer calls, no ring loss and all
48 layers. The stricter CPU loader independently accepted the first two
files, including ordered complete steps, norms and exact P2 flag recomputation.
The analysis additionally evaluates norm tables trained with the evaluated
workload entirely excluded, to expose the optimistic same-prompt repeat case.
This is a norm-ranking diagnostic; frontier-based threshold selection remains
an oracle screen requiring a fresh final holdout.

Capture landed: w16 / code-edit: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w16-code-edit.npz`, 21168 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.007048.

The final oracle comparison refines each family's target crossing in addition
to the 101-point training grid: singleton/joint use the smallest strict cut
on captured true scores that reaches the requested mean D reduction; layer
budgets are bisected over the training-calibrated layer cost curves. This
avoids choosing a needlessly aggressive grid point. **These refined cuts use
test data and are explicitly oracle diagnostics**, whereas norm tables and
layer equalization curves remain trained on the first half only. An additional
`common-policy.csv` chooses a single training-grid setting per width meeting
the D target in all four test workloads, minimizing the worst p95/baseline
ratio. That common-setting selection also requires fresh holdout validation.
Eight CPU unit tests, including strict target-crossing ties, pass; the complete
synthetic pipeline now produces 2,472 points including the refinements.

Capture landed: w16 / prose-en: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w16-prose-en.npz`, 37872 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.005939.

Capture landed: w16 / prose-ja: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w16-prose-ja.npz`, 44496 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.006982.

Capture landed: w16 / agent-loop: `$HOME/tools/flash-next-bench/runs/census-p3-20260907-2300-w16-agent-loop.npz`, 23280 target-layer calls; layers 0–47; unchanged=True; max reconstruction relative L2=0.006923.

## Captured workload coverage

| W | workload | steps | train / test steps | reconstruction p99 / max relative L2 |
|---|---|---:|---:|---:|
| 4 | agent-loop | 780 | 390 / 390 | 0.005248 / 0.006151 |
| 4 | code-edit | 1231 | 615 / 616 | 0.005294 / 0.007611 |
| 4 | prose-en | 939 | 469 / 470 | 0.005166 / 0.005758 |
| 4 | prose-ja | 997 | 498 / 499 | 0.005200 / 0.006300 |
| 16 | agent-loop | 485 | 242 / 243 | 0.005257 / 0.006923 |
| 16 | code-edit | 441 | 220 / 221 | 0.005273 / 0.007048 |
| 16 | prose-en | 789 | 394 / 395 | 0.005192 / 0.005939 |
| 16 | prose-ja | 927 | 463 / 464 | 0.005214 / 0.006982 |

## Test frontier: baseline and best qualifying point per policy family

The base threshold grid, layer calibration and norm tables use the temporal training half. Additional target-crossing thresholds are refined on the test data; the displayed best point is selected on that test frontier and is an **oracle screen**, not a validated production setting. All values below are target-verify layers 0–47 only.

| W | workload | policy | D | delta D vs .08 | score sum mean | p95 | p99 | estimated us/step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | agent-loop | baseline | 18.706 | +0.000 | 0.071919 | 0.238725 | 0.428573 | +0.00 |
| 4 | agent-loop | singleton | 15.006 | +3.700 | 0.068679 | 0.173480 | 0.218809 | +258.63 |
| 4 | agent-loop | layer | 15.006 | +3.700 | 0.069785 | 0.172348 | 0.226432 | +258.63 |
| 4 | agent-loop | joint | 15.006 | +3.700 | 0.061955 | 0.146541 | 0.183291 | +258.63 |
| 4 | code-edit | baseline | 23.641 | +0.000 | 0.082174 | 0.326319 | 0.531735 | +0.00 |
| 4 | code-edit | singleton | 19.941 | +3.700 | 0.053397 | 0.159934 | 0.202423 | +258.63 |
| 4 | code-edit | layer | 19.941 | +3.700 | 0.056094 | 0.168326 | 0.246496 | +258.63 |
| 4 | code-edit | joint | 19.941 | +3.700 | 0.049419 | 0.141630 | 0.176569 | +258.63 |
| 4 | prose-en | baseline | 18.221 | +0.000 | 0.061546 | 0.200785 | 0.354053 | +0.00 |
| 4 | prose-en | singleton | 14.521 | +3.700 | 0.067074 | 0.167111 | 0.210378 | +258.63 |
| 4 | prose-en | layer | 14.521 | +3.700 | 0.067991 | 0.164749 | 0.211889 | +258.63 |
| 4 | prose-en | joint | 14.521 | +3.700 | 0.060829 | 0.144644 | 0.179328 | +258.63 |
| 4 | prose-ja | baseline | 17.353 | +0.000 | 0.057307 | 0.186420 | 0.330160 | +0.00 |
| 4 | prose-ja | singleton | 13.653 | +3.700 | 0.068990 | 0.172573 | 0.221992 | +258.63 |
| 4 | prose-ja | layer | 13.653 | +3.700 | 0.069637 | 0.171553 | 0.224305 | +258.63 |
| 4 | prose-ja | joint | 13.653 | +3.700 | 0.062751 | 0.147722 | 0.186423 | +258.63 |
| 16 | agent-loop | baseline | 53.914 | +0.000 | 0.038192 | 0.143183 | 0.285213 | +0.00 |
| 16 | agent-loop | singleton | 47.214 | +6.700 | 0.036605 | 0.119912 | 0.174047 | +468.33 |
| 16 | agent-loop | layer | 47.214 | +6.700 | 0.037148 | 0.120594 | 0.179594 | +468.33 |
| 16 | agent-loop | joint | 47.214 | +6.700 | 0.031488 | 0.096082 | 0.134957 | +468.33 |
| 16 | code-edit | baseline | 61.876 | +0.000 | 0.040478 | 0.181328 | 0.378607 | +0.00 |
| 16 | code-edit | singleton | 55.176 | +6.700 | 0.028760 | 0.109370 | 0.158994 | +468.33 |
| 16 | code-edit | layer | 55.176 | +6.700 | 0.029621 | 0.110438 | 0.168396 | +468.34 |
| 16 | code-edit | joint | 55.176 | +6.700 | 0.025263 | 0.091583 | 0.130087 | +468.33 |
| 16 | prose-en | baseline | 49.878 | +0.000 | 0.031132 | 0.118720 | 0.231847 | +0.00 |
| 16 | prose-en | singleton | 43.178 | +6.700 | 0.036334 | 0.121294 | 0.175861 | +468.33 |
| 16 | prose-en | layer | 43.178 | +6.700 | 0.036742 | 0.120481 | 0.177893 | +468.33 |
| 16 | prose-en | joint | 43.178 | +6.700 | 0.031631 | 0.097596 | 0.138673 | +468.33 |
| 16 | prose-ja | baseline | 46.640 | +0.000 | 0.028934 | 0.110087 | 0.210907 | +0.00 |
| 16 | prose-ja | singleton | 39.940 | +6.700 | 0.035736 | 0.119863 | 0.175349 | +468.33 |
| 16 | prose-ja | layer | 39.940 | +6.700 | 0.036140 | 0.119392 | 0.179495 | +468.33 |
| 16 | prose-ja | joint | 39.940 | +6.700 | 0.031419 | 0.097792 | 0.138636 | +468.33 |

## Smallest p95 meeting the requested additional-D target

| W | workload | winner | delta D | p95 | baseline p95 | p95 / baseline | us/step |
|---|---|---|---:|---:|---:|---:|---:|
| 4 | agent-loop | joint | 3.700 | 0.146541 | 0.238725 | 0.614 | 258.63 |
| 4 | code-edit | joint | 3.700 | 0.141630 | 0.326319 | 0.434 | 258.63 |
| 4 | prose-en | joint | 3.700 | 0.144644 | 0.200785 | 0.720 | 258.63 |
| 4 | prose-ja | joint | 3.700 | 0.147722 | 0.186420 | 0.792 | 258.63 |
| 16 | agent-loop | joint | 6.700 | 0.096082 | 0.143183 | 0.671 | 468.33 |
| 16 | code-edit | joint | 6.700 | 0.091583 | 0.181328 | 0.505 | 468.33 |
| 16 | prose-en | joint | 6.700 | 0.097596 | 0.118720 | 0.822 | 468.33 |
| 16 | prose-ja | joint | 6.700 | 0.097792 | 0.110087 | 0.888 | 468.33 |

## One common grid setting per width across all four workloads

Among training-grid settings that meet the additional-D target in every test workload, minimize the worst workload p95/baseline ratio. These share one policy/threshold configuration per width; selection is still an oracle screen. Per-workload refined cuts above are excluded.

| W | workload | common policy | grid index | delta D | p95 / baseline |
|---|---|---|---:|---:|---:|
| 4 | agent-loop | joint | 160 | 4.728 | 0.686 |
| 4 | code-edit | joint | 160 | 6.595 | 0.596 |
| 4 | prose-en | joint | 160 | 4.514 | 0.782 |
| 4 | prose-ja | joint | 160 | 3.896 | 0.808 |
| 16 | agent-loop | joint | 156 | 9.598 | 0.781 |
| 16 | code-edit | joint | 156 | 12.183 | 0.686 |
| 16 | prose-en | joint | 156 | 7.229 | 0.846 |
| 16 | prose-ja | joint | 156 | 6.930 | 0.901 |

## Held-out calibrated norm-table proxy

Proxy = w * shrunk training mean expert norm[layer, expert] / current residual norm. Eight layer-mean pseudo-observations smooth each expert. No test norms enter fitting. Recall counts whole experts (a count-two group counts once). Matched-D cuts are test-time ranking diagnostics only.

| W | workload | policy | same-cut expert recall | same-cut delta D | same-cut true p95 | matched-D recall | matched-D true p95 / oracle p95 | unseen routes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | agent-loop | singleton | 0.946 | 3.325 | 0.175030 | 0.962 | 1.059 | 0.0002 |
| 4 | agent-loop | joint | 0.928 | 3.152 | 0.148397 | 0.950 | 1.071 | 0.0002 |
| 4 | agent-loop | layer | 0.941 | 3.304 | 0.171715 | — | — | 0.0002 |
| 4 | code-edit | singleton | 0.929 | 3.168 | 0.159430 | 0.953 | 1.066 | 0.0003 |
| 4 | code-edit | joint | 0.924 | 3.121 | 0.140694 | 0.950 | 1.061 | 0.0003 |
| 4 | code-edit | layer | 0.900 | 2.858 | 0.154799 | — | — | 0.0003 |
| 4 | prose-en | singleton | 0.950 | 3.446 | 0.172688 | 0.961 | 1.067 | 0.0000 |
| 4 | prose-en | joint | 0.934 | 3.390 | 0.150000 | 0.947 | 1.069 | 0.0000 |
| 4 | prose-en | layer | 0.945 | 3.435 | 0.168086 | — | — | 0.0000 |
| 4 | prose-ja | singleton | 0.954 | 3.523 | 0.178871 | 0.963 | 1.063 | 0.0001 |
| 4 | prose-ja | joint | 0.932 | 3.398 | 0.151693 | 0.946 | 1.061 | 0.0001 |
| 4 | prose-ja | layer | 0.951 | 3.488 | 0.175209 | — | — | 0.0001 |
| 16 | agent-loop | singleton | 0.946 | 5.886 | 0.118278 | 0.963 | 1.044 | 0.0000 |
| 16 | agent-loop | joint | 0.915 | 5.274 | 0.094266 | 0.944 | 1.059 | 0.0000 |
| 16 | agent-loop | layer | 0.943 | 5.786 | 0.118023 | — | — | 0.0000 |
| 16 | code-edit | singleton | 0.918 | 5.401 | 0.105487 | 0.947 | 1.052 | 0.0001 |
| 16 | code-edit | joint | 0.896 | 4.908 | 0.086788 | 0.938 | 1.052 | 0.0001 |
| 16 | code-edit | layer | 0.900 | 4.996 | 0.102719 | — | — | 0.0001 |
| 16 | prose-en | singleton | 0.960 | 6.501 | 0.124439 | 0.965 | 1.042 | 0.0000 |
| 16 | prose-en | joint | 0.934 | 6.362 | 0.101604 | 0.941 | 1.061 | 0.0000 |
| 16 | prose-en | layer | 0.958 | 6.368 | 0.122200 | — | — | 0.0000 |
| 16 | prose-ja | singleton | 0.957 | 6.307 | 0.120685 | 0.967 | 1.040 | 0.0000 |
| 16 | prose-ja | joint | 0.923 | 5.912 | 0.098399 | 0.942 | 1.052 | 0.0000 |
| 16 | prose-ja | layer | 0.954 | 6.171 | 0.118798 | — | — | 0.0000 |

### Proxy generalization diagnostic: excluded workload

Each norm table below trains on the temporal training halves of the other three workloads only. Threshold choice still uses the oracle frontier; this isolates norm-prediction generalization rather than validating a deployable policy.

| W | workload | policy | excluded-workload same-cut recall | matched-D recall | matched-D p95 / oracle |
|---|---|---|---:|---:|---:|
| 4 | agent-loop | singleton | 0.942 | 0.959 | 1.066 |
| 4 | agent-loop | joint | 0.921 | 0.945 | 1.080 |
| 4 | agent-loop | layer | 0.937 | — | — |
| 4 | code-edit | singleton | 0.889 | 0.931 | 1.106 |
| 4 | code-edit | joint | 0.862 | 0.923 | 1.125 |
| 4 | code-edit | layer | 0.814 | — | — |
| 4 | prose-en | singleton | 0.950 | 0.959 | 1.073 |
| 4 | prose-en | joint | 0.935 | 0.943 | 1.075 |
| 4 | prose-en | layer | 0.946 | — | — |
| 4 | prose-ja | singleton | 0.955 | 0.959 | 1.075 |
| 4 | prose-ja | joint | 0.935 | 0.939 | 1.067 |
| 4 | prose-ja | layer | 0.952 | — | — |
| 16 | agent-loop | singleton | 0.945 | 0.960 | 1.048 |
| 16 | agent-loop | joint | 0.911 | 0.939 | 1.064 |
| 16 | agent-loop | layer | 0.942 | — | — |
| 16 | code-edit | singleton | 0.896 | 0.938 | 1.063 |
| 16 | code-edit | joint | 0.858 | 0.923 | 1.080 |
| 16 | code-edit | layer | 0.869 | — | — |
| 16 | prose-en | singleton | 0.961 | 0.963 | 1.045 |
| 16 | prose-en | joint | 0.934 | 0.938 | 1.065 |
| 16 | prose-en | layer | 0.958 | — | — |
| 16 | prose-ja | singleton | 0.956 | 0.964 | 1.047 |
| 16 | prose-ja | joint | 0.925 | 0.935 | 1.058 |
| 16 | prose-ja | layer | 0.954 | — | — |

Artifacts: bench/quality/runs/p3-contrib-20260907-2300. `frontier.csv` holds every threshold, `pareto.csv` the D/p95 non-dominated subset, `calibration.json` every layer threshold, `proxy-evaluation.json` precision and route/expert recall, and `proxy-w*.npz` runtime table candidates. `layer-error.csv` gives per-layer calibration means and held-out contribution tails for each selected policy family.


## Train-only common proxy threshold: actual temporal holdout

One joint-removal proxy threshold per width is the maximum of the four training-workload target-crossing thresholds. Both the norm table and threshold are fit without holdout data. The table below evaluates the fixed setting on the held-out halves; matched-D true-score recall is evaluation only. This still covers repeats of the same four prompts, not a new-prompt quality gate.

| W | workload | proxy threshold | D | delta D | true score mean | p95 | p99 | p95 / baseline | estimated us/step | oracle expert recall at matched D |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | agent-loop | 0.04199233 | 13.989 | 4.717 | 0.073366 | 0.174315 | 0.222663 | 0.730 | 329.71 | 0.955 |
| 4 | code-edit | 0.04199233 | 17.103 | 6.539 | 0.075659 | 0.203790 | 0.251604 | 0.625 | 457.05 | 0.957 |
| 4 | prose-en | 0.04199233 | 13.558 | 4.664 | 0.071872 | 0.169839 | 0.215580 | 0.846 | 325.99 | 0.952 |
| 4 | prose-ja | 0.04199233 | 13.300 | 4.053 | 0.068114 | 0.163202 | 0.212336 | 0.875 | 283.27 | 0.948 |
| 16 | agent-loop | 0.04226924 | 44.369 | 9.544 | 0.039145 | 0.117591 | 0.168331 | 0.821 | 667.15 | 0.951 |
| 16 | code-edit | 0.04226924 | 50.179 | 11.697 | 0.037281 | 0.126583 | 0.181324 | 0.698 | 817.60 | 0.945 |
| 16 | prose-en | 0.04226924 | 41.903 | 7.975 | 0.035827 | 0.111347 | 0.160916 | 0.938 | 557.43 | 0.945 |
| 16 | prose-ja | 0.04226924 | 39.427 | 7.214 | 0.033637 | 0.105979 | 0.152412 | 0.963 | 504.25 | 0.944 |


## Selected implementation candidate and verdict

**Advance joint whole-expert removal using a calibrated norm-table proxy.**
In the true-score oracle, it is best in all eight cases: at additional D=3.7
(W4) / 6.7 (W16), p95 is 0.434–0.792 / 0.505–0.888 of tau=.08. Layer equalization
does not beat joint selection here. It sometimes improves the singleton tail
slightly, but does not recover the benefit of including cheap count-two groups.

The practical training-only common proxy thresholds are:

| width | global proxy group threshold | held-out additional D range | estimated saving range (us/step) | p95 / .08 range | oracle group recall at matched actual D |
|---|---:|---:|---:|---:|---:|
| W4 | 0.04199233497 | 4.053–6.539 | 283.27–457.05 | 0.625–0.875 | 0.948–0.957 |
| W16 | 0.04226923952 | 7.214–11.697 | 504.25–817.60 | 0.698–0.963 | 0.944–0.951 |

Use `proxy-w4.npz` / `proxy-w16.npz` with the `norm_table` field and
`proxy-train-calibration.json` for these candidates. Tables/thresholds are
width-specific, but no workload classifier is used at runtime. The score
accumulation and table fitting here use NumPy float64; the proposed FP32
kernel/table conversion requires explicit parity/tie checks before use.

**Replace the P2 .08 decision, do not run P3 after the P2 mask.** Contribution
selection can restore a low-weight, high-norm route that P2 would drop, then
remove other lower-contribution groups. Running P2 first destroys exactly the
immutable routes and choices this frontier assumes. Enable one policy at a
time; a missing/invalid P3 calibration must select an explicit baseline or
unpruned fallback, not a partially applied mixture of two policies.

The cheap table retains substantial ranking information. The workload-excluded
joint proxy reaches 92.3–94.5% recall at matched D, with p95 5.8–12.5% above
its corresponding oracle (the code workload generalizes least well). These
are informative screening results, not a quality certificate. The narrowest
train-only proxy p95 advantage is just 3.73% on W16 Japanese prose, so
calibration drift and downstream HC sensitivity deserve particular attention.

This is a tail-focused tradeoff, not a reduction of every error statistic.
For the train-only proxy, mean sum(s_removed) rises **16.8% / 18.9%** on W4
English/Japanese prose and **15.1% / 16.3%** on W16 English/Japanese prose,
even though p95 falls. Agent-loop mean rises about 2–2.5%; code mean falls
about 7.9%. More distributed small perturbations could still harm quality or
accumulate across layers/tokens. The requested p95 winner must therefore pass
the fresh end-to-end non-inferiority gate; these statistics do not prove safety.

### Critical-path overhead budget

At the fixed train-only proxy's weakest speed point (Japanese prose), the
conditional savings are 283.27 us W4 and 504.25 us W16. Exactly +3% throughput
at unchanged acceptance requires `T * (1 - 1/1.03)` = 259.78 / 468.44 us for
the review's 8,919 / 16,083 us step times. Thus the total added critical-path
budget is only **23.49 us/step W4 / 35.81 us/step W16**, before any acceptance
loss. Do not multiply the 69.9 coefficient by calls/step again. The first GPU
prototype must measure the score-table loads, pair aggregation, immutable mask
outputs and HC norm export together, including lost PDL overlap. Kernel work
can consume this margin; the norm frontier alone does not establish +3%.

![P3 target-verify frontier](../bench/quality/runs/p3-contrib-20260907-2300/frontier.png)

The figure zooms around the requested budgets and clips very high-error
points outside its axes. The full un-clipped grid is `frontier.csv` (2,472
points) and the D/p95 non-dominated subset is `pareto.csv` (910 points).
`frontier.svg` provides a scalable export. The PNG was visually inspected;
labels, legend and all eight panels are readable.

## Final completeness and limits

| item | status / evidence |
|---|---|
| Dedicated worktree / branch | `$HOME/tools/sglang-p3`, `codex/p3-contrib`, base `14d4c4c985` |
| Default-off recorder | Implemented; production MoE implementation AST is identical after the function-name change; no recorder import on the flag-off model path |
| GPU preflight | Six real-weight W4/W16 cases plus actual recorder graph/dump PASS; independent k=1 outputs and graph norms bit-identical in tested cases |
| Workload NPZ captures | 8/8 complete, 6,589 complete verify steps, 316,272 layer calls, 27,868,800 routes; full ordered layers 0–47 and no overflow |
| Same-call numerical isolation | Every captured call's served output and original ids/weights unchanged after auxiliary work; global max reconstruction relative L2 0.00761103 |
| Capture configuration | All NPZ metadata verified: both prune knobs 0; MEM_FRACTION/W16_MEM_FRACTION/WA_MEM_FRACTION .920; SERVE_DISPLAY_HZ empty; MAMBA_SLOTS 10; MAX_TOTAL_TOKENS 131072 |
| Successful-run VRAM | W4 44 post-readiness samples, minimum 10,399 MiB; W16 70 samples, minimum 8,119 MiB; loading transients excluded |
| CPU validation | Original selftest PASS; 8 new unit tests PASS; temporary eight-capture end-to-end fixture PASS; real eight-capture loader/frontier/proxy pipeline PASS |
| Artifact integrity | `runs/p3-capture-manifest-20260907-2300.json`: capture/source/analysis SHA256, metadata and memory summary |
| Simulator and plots | `prune_policy_sim.py --contribution`, supporting policy/calibration modules, full CSVs, calibration tables, PNG/SVG frontier |
| Prologue implementation | Specification complete; new pruning kernel, immutable output ABI and HC norm export are **not implemented** |
| Quality and real speed | **Not measured** for any contribution policy; no new policy was applied to served routes; no shipping claim |
| Cleanup / isolation | Successful server controllers exited 0 and cleaned up their own groups; lock released per run; production tracked diff empty; no production fork/launcher/venv edits, no commit/push |

This experiment records all target-verify rows in both fnbench warmup and
measured generations, including draft proposals later rejected. It excludes
prefill, draft decode and draft extension. It samples four prompts and their
repeats, so neither the temporal nor workload-exclusion diagnostic establishes
broad prompt/quality generalization. The score ignores inter-expert vector
cancellation, HC injection gains, layer sensitivity and downstream state
propagation. Expert vectors are observed at a BF16 debug output boundary;
reconstruction measures the associated finalization/rounding discrepancy.
No separate recorder-on/off greedy A/B was performed: same-call output/routing
isolation is the numerical evidence, against a nondeterministic production
finalize. All speed values are projections, not capture throughput claims.

### Reproduction of the final CPU artifacts

```bash
PYTHONDONTWRITEBYTECODE=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python bench/quality/prune_policy_sim.py --contribution runs/census-p3-20260907-2300-w*.npz --output /tmp/P3_replay.md --artifact-dir /tmp/P3_replay_artifacts
PYTHONDONTWRITEBYTECODE=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python bench/quality/calibrate_contrib_proxy.py runs/census-p3-20260907-2300-w*.npz --artifact-dir /tmp/P3_replay_artifacts --report /tmp/P3_replay.md
MPLCONFIGDIR=$HOME/.cache/p3-matplotlib uv run --isolated --no-project --with matplotlib==3.10.6 --with numpy python bench/quality/plot_contrib_frontier.py /tmp/P3_replay_artifacts
```

Fresh capture reproduction uses a new timestamped label and the frozen
`runs/p3-capture-script-20260907-2300.sh`, one
`flock -w 28800 $HOME/.gpu.lock <script> <w4|w16> <new-label>` per run.
The script refuses existing output paths. Do not edit a script while it runs.
