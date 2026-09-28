# HC1 bottleneck screen — complete

Status: valid capture complete; 5/5 evaluation arms complete.

## Funding decision

**Do not fund a full trained HC simplification project on this evidence alone.
No cheap untrained variant has established the 0.5 pp non-inferiority target.**
If the HC research line is continued, the only recommended next allocation is
a tightly capped **rank-160 recovery plus packed-shape feasibility pilot**,
with a stop gate before a days-to-weeks implementation/training commitment.
This is a research recommendation, not a claim that recovery will succeed;
no recovery training or physically packed GPU kernel was run in HC1.

| Candidate | Untrained quality decision | Funding decision for standalone >=3% |
|---|---|---|
| hc256 | MMLU FAIL (-2.00 pp); other three NI results inconclusive | Stop this standalone proposal: even perfect proportional packing only gives .156/.232 ms at unchanged acceptance. Recovery could address quality, but does not create the missing structural time budget. |
| hc160 | All four FAIL; MMLU -7.71 pp, HumanEval -14.02 pp | Only conditional, capped recovery research. It has theoretical time budget and substantially less code-quality loss than hc128, but needs both quality/acceptance recovery and a viable 160-wide packed configuration. |
| hc128 | All four FAIL; HumanEval -32.32 pp | Stop as the low-cost candidate. The extra ideal saving over hc160 is only .078/.116 ms, while code-generation damage is much larger. This is not proof that an extensively retrained rank-128 architecture is impossible. |
| gate-const | All four inconclusive; GSM8K -1.36 pp, HumanEval -.61 pp | Closest to prod overall, but preservation is unproven and even the loose structural saving bound is below 3% at unchanged acceptance. Stop the standalone >=3% proposal; training necessity is not established by an inconclusive NI result. |

Any rank-160 pilot must recover all four fixed-battery NI gates, retain
acceptance on held-out prompts, and demonstrate >=3% end-to-end improvement
with physically packed matrices under the intended W4/W16 serving profile.
First check the packed kernel floor: current 64-unit tiling pads 160 to 192,
whose ideal W16 saving (.464 ms) already falls just below the .468 ms target.
The linear .390/.580 ms projection requires a more suitable shape/configuration.
Do not bank the prose acceptance gains as restored-model gains: these came
from the quality-degraded, untrained ablations and one restart per arm.

Even for W4 code-edit, the observed acceptance losses consume the apparent
time margin: hc160 would need .423 ms using the historical step denominator,
above its .390 ms linear budget; hc128 needs .467 ms against a .468 ms budget,
leaving essentially no fixed-cost margin. With the fresh prod effective-time
proxy, those requirements are .561/.620 ms. These are planning calculations,
not new packed latency measurements or proof of performance after training.

## Completion audit

- Full fixed battery: 20,010 graded rows, exact original problem IDs, zero
  transport errors. All 16 candidate/benchmark NI rows match the unchanged
  corrected analyzer's final NI section. No candidate/benchmark NI PASS.
- BN1: 160 valid requests, eight per domain per arm; one valid server per arm.
  Needle: all 30 pre/post cases pass, each with 17,307 actual input tokens.
- Calibration: 100 raw-file hashes match the manifest; all top-rank masks,
  lowest-variance gate IDs, and constant means match the saved statistics.
  See `hc1/selection-audit.json`.
- Total occupied lock time, including every invalid attempt: 15,984.6 seconds
  (4.440 hours), excluding lock waits. Valid calibration: 587.5 seconds.
- Minimum observed steady free VRAM: capture 7,972 MiB; prod 7,430;
  hc256 7,858; hc160 7,243; hc128 8,235; gate-const 7,322.
- Final frozen-input/model-inventory check passed. Worktree HEAD remains
  `7b4d539f9bd896265f498fabccc6466e45ffe818` on the requested branch, with
  only the three intended HC source files changed. Production tracked diff
  is empty; its pre-existing untracked launcher/log files remain unchanged
  in status, and the frozen launcher hashes match. No commit or push.
- At completion, `nvidia-smi --query-compute-apps` returned no compute
  processes and `lslocks` showed no owner/waiter for `.gpu.lock`.
  Detailed counts, memory minima and per-attempt times are in
  `hc1/final-verification.json`.

Worktree `$HOME/tools/sglang-hc1`, branch `codex/hc1-bottleneck-screen`,
base `codex/perf-v1` at `7b4d539f9b`. No commits or pushes. Production launcher,
source and venv are read-only at runtime; only PYTHONPATH overlays the worktree.
Private copied cache is bound over the launcher's fixed cache path.

## Frozen experiment

- W4, BS=1 throughout capture, acceptance and quality; production P2 tau=.08,
  v5 MTP and standard serving flags inherited from the frozen launcher.
- Every server enclosed by `flock -w 28800 $HOME/.gpu.lock`; one lock per
  arm, including teardown. Budget approximately 24 occupied GPU hours.
- Memory fraction .900, `SERVE_DISPLAY_HZ=`, >=4096 MiB steady-state free VRAM watchdog.
  No unowned process signaled; owned process groups only. MAX_TOTAL_TOKENS=131072.
- Calibration: all 32 BN1 v1 prompts, 512-token output cap; one short eager W4
  server after an unrecorded 1024-token non-holdout warmup, thinking disabled (matching the quality battery). BN1 acceptance
  retains its original client/server thinking behavior; this is an additional
  calibration-to-acceptance distribution difference. Observe real K1 split-K outputs every eighth HC call, max 32 rows per
  module per prompt. Prefill rows >16 and nonfinite rows are excluded; nonfinite rejections are counted per module/prompt. This samples both target
  verify rows (including subsequently rejected rows) and recursive draft rows.
- Unit contribution: equal-parent mean abs(BF16(SiLU(K1/hc))) times L2 norm of
  the FP8-dequantized up column. Gate variance/mean: FP32 replay of up projection
  from actual K1 activations, original FP8 bytes and scales, then sigmoid.
  Gate replay is a diagnostic approximation to Triton accumulation, not a claim
  of bit identity. Low variance is a heuristic, not a proven sensitivity measure.
- Keep top 256/160/128 independently per module; zero dropped down rows and up
  columns in both BF16 originals and already-quantized FP8 copies. Original FP8
  survivor bytes and scales are preserved. Matrix shapes and dense kernels stay
  unchanged; flag `SGLANG_HC1_ARM` defaults to `off`.
- Gate-const: lowest-variance 25% of input-mix gate coordinates per module are
  replaced by each coordinate's calibration mean. Zero corresponding up rows
  gives .5; an additive mixed-output correction supplies mean-.5. All combine/
  block-injection gates are unchanged. Correction incurs diagnostic overhead and
  an additional BF16 output rounding, which is not physical-packing performance.
- Evaluation: one fresh server per arm (`prod`, `hc256`, `hc160`, `hc128`,
  `gate-const`); fixed non-holdout warmup, BN1 32 requests at original full output
  caps, one repeat; needle with nominal 18.5k approximation at .1/.5/.9 before and after quality
  (the existing script produces **17,307 actual prompt tokens**, confirmed in
  prod logs; this is not a verified 18,500-token input).
- Quality uses unchanged copies of `bench/quality/run_bench.py`, `he_exec.py`,
  corrected `analyze.py` and the same datasets: GSM8K 1319, MMLU 1400,
  HumanEval 164, JCQA 1119. BS=CONC=1. The standard battery's launcher uses wa;
  HC1 holds W4 fixed for all five matched arms to meet the one-restart design.
- Primary NI difference is **candidate minus prod**, margin .5 percentage points,
  paired Tango score one-sided 95% bounds (the table below was
  recomputed from its discordant counts; the original Wald bounds changed no verdict). The corrected analyzer's
  NI uses BASE minus other: invoke with each candidate as BASE and prod as other.
  Non-significance is not NI.
- BN1 prompts are reused for calibration and acceptance per task; acceptance is
  in-calibration informational evidence, not held-out generalization.

## Startup guard correction

Initial capture startup stopped at 3982 MiB transient free VRAM because the
driver incorrectly enforced the steady-state 4096 MiB requirement during weight
loading as well. No prompt ran. Logs are retained in
`hc1/capture-startup-guard-attempt1`. The guard now enforces the requested floor
once ready; startup free memory continues to be logged. Fraction remains .900.

## Capture finite-value correction

First full capture completed in 629.927 lock seconds but was INVALIDATED:
only the first BN1 parent contained nonfinite accumulators in
`mtp.layers.0.mlp_hyper_connection` and `mtp.hyper_connection_mixer`.
The other 31 parents were finite. Initial/lazy MTP setup rows are suspected;
this is not proven from aggregate-only data. All original data and invalid masks
are retained with `-invalid-capture1` suffixes. No candidate used those masks.
The subsequent prod startup was stopped through its owned arm runner before
quality evaluation so all final arms can use one corrected source freeze.

The recorder now runs one unrecorded non-holdout warmup, excludes and counts
nonfinite diagnostic rows, and the builder fails closed on any nonfinite
aggregate and requires exactly 100 modules including all 3 MTP modules.
JSON output rejects NaN. The new validator correctly rejects the original
corrupt capture. CPU algebra/survivor/kernel tests still pass.

## Acceptance warmup correction

The first evaluated prod startup completed one BN1 request, but BN1 rejected
its acceptance as `missing_or_changed_series`. The HC1 non-streaming warmup
had initialized different Prometheus labels from the streaming BN1 request.
This run is INVALID and retained at `hc1/prod-invalid-metrics1`; its prompt is
not pooled into final results. The warmup now uses the exact BN1
`OpenAIStreamClient.complete()` contract (streaming, 1024 tokens, fixed
non-holdout prompt), initializing the same counter series before measurement.
This affects the evaluation client only; the validated capture and model
source are unchanged.

## Reproduction and evidence paths

Gate-const startup correction: the first gate-const startup failed before any
evaluation request because the model loader sets the default torch dtype to
BF16. The mean-value tensor therefore did not match the FP32 correction buffer
in indexed assignment. The runtime change explicitly sets that constructor to
FP32. All other runtime source bytes are unchanged from the preceding freeze;
the changed line is only reachable for gate-const, so capture/prod/unit masks
are unaffected. A CPU regression test reproduces the BF16 default and verifies
FP32 means, zero gate rows, and surviving rows. All four CPU tests pass.
The unmeasured 57.4-second startup and original source are retained in
`hc1/gate-const-invalid-dtype1`; the previous freeze is retained in
`hc1/frozen-before-gate-dtype-fix.json` with a scoped amendment in `frozen.json`.
The final gate-const measurement uses one valid fresh server after this fix.

- Driver: `bench/hc1/run_screen.py sequence`; resumes completed arms, acquires
  the shared flock independently per arm. Run on the GPU-visible host using
  the existing benchmark `.venv-review/bin/python -B`.
- Model changes: `sglang-hc1/python/sglang/srt/layers/hc1_screen.py` plus Python
  hooks in `hyperconnection.py` and `hc_mix2_triton.py`; dense Triton kernel
  bodies are unchanged.
- Immutable selection: `specs/hc1/masks.json`; per-unit/per-gate summary:
  `specs/hc1/activation-statistics.json`; raw per-parent tensor sums:
  `specs/hc1/capture-data/*.pt`.
- Quality rows: `bench/hc1/quality/runs/<arm>/*.jsonl`; acceptance, needle,
  server-info, VRAM samples and events: `specs/hc1/<arm>/`.
- `bench/hc1/report.py` refreshes this incremental report and
  `specs/hc1/results.json`. It audits candidate load logs, actual W4/BS1/.900
  server settings, full paired IDs and steady VRAM for completed arms.
- No dataset, checkpoint, production launcher or venv is modified. No commit
  or push is performed.

## CPU validation

Four tests passed: FP8 survivor-byte preservation and masked/packed algebra for
all ranks, gate replacement algebra, gate initialization under a BF16 default,
and AST identity of all HC dense kernels.
No GPU correctness or quality claim follows from these CPU checks.
The copied HumanEval executor also ran a known-correct CPU program successfully
on the GPU-visible host (`hc1/humaneval-executor-preflight.json`); sandbox
startup failure is therefore not being treated as model failure.

## Calibration interpretation

Contribution retention is the fraction of the sum of individual unit L2 norms;
it is not retained output energy or a bound on post-sigmoid/whole-model error.
Target (97 modules) mean retention for 256/160/128: 97.98% / 89.63% / 84.98%.
MTP (3 modules): 96.01% / 84.69% / 79.57%. The weakest rank-160/128 retention
is target layer 35 attention (78.17% / 71.51%).

Even the selected least-variable gate quartile is not nearly constant in all
layers: module-average RMS standard deviation is .0591 target / .0972 MTP,
with maximum module values .1172 / .1429 on a [0,1] sigmoid output scale.
These statistics select the ablation; only the independent quality battery
can establish its quality outcome.

## Planning arithmetic — NOT measured packed latency

Historical full step baselines: W4 8.919 ms, W16 16.083 ms. At unchanged
acceptance, +3% throughput needs 0.260/0.468 ms saving (rounded requirement
0.26/0.47 ms). Review §4.3 supplies K1+K2 budgets 0.78/1.16 ms per step.
Dense FP8 shapes: down [320,10240], up [10240,320] for target HC; MTP HC has the same [320,10240] / [10240,320] shapes, confirmed from
capture: 97 target modules and 3 MTP modules, all hc=4, hs=2560. Stats/normalization and combine stay fixed.

| Variant | Rank-byte reduction | Ideal K1+K2 saving W4 / W16 | Ideal throughput W4 / W16 | Clears 3% ideally |
|---|---:|---:|---:|---|
| hc256 | 20% | .156 / .232 ms | 1.78% / 1.46% | No / No |
| hc160 | 50% | .390 / .580 ms | 4.57% / 3.74% | Yes / Yes |
| hc128 | 60% | .468 / .696 ms | 5.54% / 4.52% | Yes / Yes |
| gate-const | 25% of up rows only | <=.195 / <=.290 ms (loose K2<=K1+K2 bound) | <=2.24% / <=1.84% | No / No |

These assume perfect proportional packing, no launch floor or added scatter,
and unchanged acceptance. Rank 160 pads to 192 at the current 64-unit tile size;
actual packing requires a different shape/configuration to realize the linear
50% bound. With current 64-unit tiling its ideal tile reduction is only 40%
(0.312 / 0.464 ms), which barely misses the historical W16 0.468 ms gate even
before fixed launch cost. K1 grid [20,5] becomes [20,4]/[20,3]/[20,2]; K2
keeps 160 output CTAs but takes 4/3/2 rank tiles instead of 5. Total FP8 matrix
bytes, excluding scales: 6.554 MB at 320, 5.243 MB at 256, 3.277 MB at 160,
2.621 MB at 128. Gate-const packs only 25% of up rows: 5.734 MB total (12.5%
weight-byte reduction). Using the pinned K1/K2 3.81/4.64 us ratio gives a
more specific gate-const projection of 0.107 / 0.159 ms (1.22% / 1.00%
throughput); the table retains a looser upper bound to avoid treating that
microbenchmark split as current step attribution.
Gate replacement still needs all down units for the other 75% of gate outputs;
it cannot save a quarter of the whole HC chain.

The requested 11.5%/9.3% HC shares require 25.3%/31.3% HC time reduction to
clear +3%. The current `EXCLUSIVE_TIME_MAP_0907.md` §1.2 instead lists
12.1%/10.3% (1.082/1.649 ms); these are different historical inputs and are not
silently mixed. New masked t/s is not packed HC time.

## Fresh W4 effective-time cross-check

The completed prod BN1 arm gives the following **derived decode time per verify**
(client decode seconds / completed verify calls). This includes effective
serving overhead; it is not a new exclusive kernel profile or physical HC timing.

| Domain | Effective ms/verify | Saving for +3% at unchanged acceptance |
|---|---:|---:|
| code-edit | 11.8194 | .3443 ms |
| prose-en | 10.7341 | .3126 ms |
| prose-ja | 10.6141 | .3091 ms |
| agent-loop | 10.9574 | .3191 ms |

These observed W4 denominators are higher than the historical 8.919 ms planning
baseline. Holding the historical ideal HC saving fixed would leave less margin
for acceptance loss or packing overhead. No W16 effective time was measured in
HC1; its .47 ms requirement remains the requested historical planning value.

## HC256 error-cohort diagnostic (primary NI unchanged)

| Benchmark | Candidate wins / losses vs prod | Wins with only prod truncated | Losses with only hc256 truncated |
|---|---:|---:|---:|
| GSM8K | 60 / 53 | 49 | 39 |
| MMLU | 38 / 66 | 1 | 15 |
| HumanEval | 2 / 7 | 1 | 2 |
| JCQA | 8 / 17 | 0 | 0 |

GSM8K's positive point delta is strongly associated with shifts around the
512-token cap, not established as improved uncapped reasoning. In MMLU,
15 of the 66 losses accompany a candidate-only cap hit; 15 losses have an
empty extracted choice. Many other losses remain, so the full deficit is not
explained by truncation. JCQA has no truncation in either arm. These are
post-hoc descriptive cohorts, not a revised quality gate or a causal proof.
Raw counts: `hc1/hc256/error-cohorts.json`.

## Results and verdict

Execution: capture: complete, prod: complete, hc256: complete, hc160: complete, hc128: complete, gate-const: complete.

Calibration captured 100 HC modules, 102400 module-token rows across 32 parents.

| Kept rank | Mean retained contribution | Min / max across modules |
|---|---:|---:|
| 256 | 97.92% | 93.64% / 99.63% |
| 160 | 89.48% | 78.17% / 96.62% |
| 128 | 84.82% | 71.51% / 94.17% |

Excluded nonfinite diagnostic rows: {"mtp.hyper_connection_mixer": 1, "mtp.layers.0.mlp_hyper_connection": 1}. All exported activation/gate aggregates passed finite-value checks.

### Quality (BS=1, W4, full fixed battery)

| Benchmark | prod | hc256 | hc160 | hc128 | gate-const |
|---|---|---|---|---|---|
| gsm8k | 1168/1319 (88.55%) | 1175/1319 (89.08%) | 1132/1319 (85.82%) | 1132/1319 (85.82%) | 1150/1319 (87.19%) |
| mmlu | 1173/1400 (83.79%) | 1145/1400 (81.79%) | 1065/1400 (76.07%) | 1055/1400 (75.36%) | 1171/1400 (83.64%) |
| humaneval | 158/164 (96.34%) | 153/164 (93.29%) | 135/164 (82.32%) | 105/164 (64.02%) | 157/164 (95.73%) |
| jcqa | 1089/1119 (97.32%) | 1080/1119 (96.51%) | 1060/1119 (94.73%) | 1048/1119 (93.66%) | 1084/1119 (96.87%) |

### Quality generation integrity

| Arm | Benchmark | Mean tokens | Length-capped | Transport errors |
|---|---|---:|---:|---:|
| prod | gsm8k | 345.9 | 158 | 0 |
| prod | mmlu | 2.1 | 13 | 0 |
| prod | humaneval | 231.8 | 4 | 0 |
| prod | jcqa | 2.0 | 0 | 0 |
| hc256 | gsm8k | 322.1 | 128 | 0 |
| hc256 | mmlu | 2.4 | 36 | 0 |
| hc256 | humaneval | 234.9 | 6 | 0 |
| hc256 | jcqa | 2.0 | 0 | 0 |
| hc160 | gsm8k | 323.3 | 158 | 0 |
| hc160 | mmlu | 2.4 | 36 | 0 |
| hc160 | humaneval | 276.1 | 14 | 0 |
| hc160 | jcqa | 2.0 | 0 | 0 |
| hc128 | gsm8k | 314.8 | 141 | 0 |
| hc128 | mmlu | 2.3 | 26 | 0 |
| hc128 | humaneval | 255.8 | 11 | 0 |
| hc128 | jcqa | 2.0 | 0 | 0 |
| gate-const | gsm8k | 349.0 | 181 | 0 |
| gate-const | mmlu | 2.0 | 0 | 0 |
| gate-const | humaneval | 237.2 | 4 | 0 |
| gate-const | jcqa | 2.0 | 0 | 0 |

HumanEval stderr audit: no bwrap startup errors in completed HumanEval rows. Fixed token caps are part of the primary task; cap-conditioned comparisons cannot replace the fixed-battery NI gate.

| Arm | HumanEval failure kinds |
|---|---|
| prod | AssertionError: 2, SyntaxError: 4 |
| hc256 | SyntaxError: 7, AssertionError: 4 |
| hc160 | SyntaxError: 20, AssertionError: 8, TypeError: 1 |
| hc128 | SyntaxError: 51, AssertionError: 7, AttributeError: 1 |
| gate-const | AssertionError: 3, SyntaxError: 4 |

These are failures of the generated/extracted program under the unchanged executor, not an attribution of each failure to model reasoning versus formatting.

### Candidate-minus-prod non-inferiority (0.5 pp)

| Candidate | Benchmark | Delta pp | Lower / upper one-sided 95% pp | NI | Discordant wins/losses |
|---|---|---:|---:|---|---:|
| hc256 | gsm8k | +0.53 | -0.80 / +1.87 | INCONCLUSIVE | 60/53 |
| hc256 | mmlu | -2.00 | -3.23 / -0.81 | FAIL | 38/66 |
| hc256 | humaneval | -3.05 | -6.65 / -0.05 | INCONCLUSIVE | 2/7 |
| hc256 | jcqa | -0.80 | -1.61 / -0.07 | INCONCLUSIVE | 8/17 |
| hc160 | gsm8k | -2.73 | -4.36 / -1.12 | FAIL | 66/102 |
| hc160 | mmlu | -7.71 | -9.35 / -6.15 | FAIL | 42/150 |
| hc160 | humaneval | -14.02 | -19.23 / -9.78 | FAIL | 1/24 |
| hc160 | jcqa | -2.59 | -3.76 / -1.52 | FAIL | 14/43 |
| hc128 | gsm8k | -2.73 | -4.42 / -1.06 | FAIL | 72/108 |
| hc128 | mmlu | -8.43 | -10.16 / -6.76 | FAIL | 50/168 |
| hc128 | humaneval | -32.32 | -38.57 / -26.64 | FAIL | 0/53 |
| hc128 | jcqa | -3.66 | -4.95 / -2.50 | FAIL | 14/55 |
| gate-const | gsm8k | -1.36 | -2.63 / -0.13 | INCONCLUSIVE | 40/58 |
| gate-const | mmlu | -0.14 | -1.29 / +1.00 | INCONCLUSIVE | 45/47 |
| gate-const | humaneval | -0.61 | -3.43 / +2.03 | INCONCLUSIVE | 2/3 |
| gate-const | jcqa | -0.45 | -1.15 / +0.21 | INCONCLUSIVE | 7/12 |

Only the copied analyzer's final BASE-minus-other non-inferiority section is used for the decision. Its earlier legacy significance VERDICT tests degradation in the opposite comparison direction when BASE is a candidate and is not the HC1 verdict.

### W4 BN1 acceptance and decode t/s (informational, one restart)

| Arm | Domain | Tokens/verify (pooled) | Decode t/s (pooled) | Equal-parent accept delta | Equal-parent t/s delta | Requests |
|---|---|---:|---:|---:|---:|---:|
| prod | code-edit | 3.907 | 330.5 | +0.00% | +0.00% | 8 |
| prod | prose-en | 2.387 | 222.4 | +0.00% | +0.00% | 8 |
| prod | prose-ja | 2.467 | 232.4 | +0.00% | +0.00% | 8 |
| prod | agent-loop | 3.153 | 287.7 | +0.00% | +0.00% | 8 |
| hc256 | code-edit | 3.896 | 379.6 | -0.29% | +14.84% | 8 |
| hc256 | prose-en | 2.802 | 293.7 | +17.35% | +32.30% | 8 |
| hc256 | prose-ja | 3.000 | 316.9 | +21.83% | +36.72% | 8 |
| hc256 | agent-loop | 3.293 | 346.3 | +4.47% | +20.36% | 8 |
| hc160 | code-edit | 3.834 | 344.0 | -1.89% | +4.20% | 8 |
| hc160 | prose-en | 3.374 | 309.3 | +38.16% | +37.93% | 8 |
| hc160 | prose-ja | 3.479 | 317.9 | +40.91% | +36.92% | 8 |
| hc160 | agent-loop | 3.647 | 331.0 | +15.46% | +15.07% | 8 |
| hc128 | code-edit | 3.813 | 346.4 | -2.40% | +4.98% | 8 |
| hc128 | prose-en | 3.401 | 325.9 | +41.58% | +44.00% | 8 |
| hc128 | prose-ja | 3.375 | 329.8 | +36.76% | +42.15% | 8 |
| hc128 | agent-loop | 3.597 | 347.6 | +13.54% | +20.47% | 8 |
| gate-const | code-edit | 3.907 | 324.0 | -0.01% | -1.96% | 8 |
| gate-const | prose-en | 2.376 | 212.9 | -0.51% | -4.26% | 8 |
| gate-const | prose-ja | 2.605 | 236.1 | +5.78% | +1.92% | 8 |
| gate-const | agent-loop | 3.132 | 280.2 | -0.69% | -2.55% | 8 |

### Acceptance-adjusted planning (W4, not a packed measurement)

| Candidate | Domain | Required saving at observed acceptance (historical / current effective T) | Linear packed saving budget | Implied optimistic delta at current effective T |
|---|---|---:|---:|---:|
| hc256 | code-edit | 0.285 / 0.377 ms | 0.156 ms | +1.05% |
| hc256 | prose-en | -1.243 / -1.496 ms | 0.156 ms | +19.09% |
| hc256 | prose-ja | -1.631 / -1.941 ms | 0.156 ms | +23.65% |
| hc256 | agent-loop | -0.127 / -0.156 ms | 0.156 ms | +5.98% |
| hc160 | code-edit | 0.423 / 0.561 ms | 0.390 ms | +1.46% |
| hc160 | prose-en | -3.045 / -3.664 ms | 0.390 ms | +43.37% |
| hc160 | prose-ja | -3.283 / -3.907 ms | 0.390 ms | +46.29% |
| hc160 | agent-loop | -1.079 / -1.325 ms | 0.390 ms | +19.72% |
| hc128 | code-edit | 0.467 / 0.620 ms | 0.468 ms | +1.63% |
| hc128 | prose-en | -3.341 / -4.021 ms | 0.468 ms | +48.03% |
| hc128 | prose-ja | -2.923 / -3.479 ms | 0.468 ms | +43.07% |
| hc128 | agent-loop | -0.913 / -1.121 ms | 0.468 ms | +18.61% |
| gate-const | code-edit | 0.261 / 0.346 ms | 0.195 ms | +1.66% |
| gate-const | prose-en | 0.304 / 0.366 ms | 0.195 ms | +1.33% |
| gate-const | prose-ja | -0.240 / -0.286 ms | 0.195 ms | +7.76% |
| gate-const | agent-loop | 0.320 / 0.393 ms | 0.195 ms | +1.11% |

This is a counterfactual using historical HC-saving budgets, the prod decode-time/verify proxy, and noisy single-restart acceptance; it is not a packed end-to-end speedup measurement. Gate-const uses the deliberately loose K2<=K1+K2 bound.


Needle results: prod: 6/6, hc256: 6/6, hc160: 6/6, hc128: 6/6, gate-const: 6/6.

### Current decision

- hc256: Untrained variant fails NI on mmlu; recovery would be required. This alone does not prove training can recover quality. At unchanged acceptance, standalone >=3% is excluded by the ideal time budget before implementation overhead; any claimed acceptance-driven rescue needs a separate replicated measurement.
- hc160: Untrained variant fails NI on gsm8k, mmlu, humaneval, jcqa; recovery would be required. This alone does not prove training can recover quality.
- hc128: Untrained variant fails NI on gsm8k, mmlu, humaneval, jcqa; recovery would be required. This alone does not prove training can recover quality.
- gate-const: NI remains inconclusive; untrained preservation is not established. At unchanged acceptance, standalone >=3% is excluded by the ideal time budget before implementation overhead; any claimed acceptance-driven rescue needs a separate replicated measurement.

Raw masked throughput includes all original dense work (and gate-const correction overhead); it is not a packed speedup. Single-restart acceptance cannot resolve restart variance. No training recovery or packed latency was measured.
