# E1 — expert intermediate width quality screen

Date: 2026-09-08. Status: COMPLETE; evidence audit PASS; verdict NEEDS RECOVERY TRAINING.
Worktree `$HOME/tools/sglang-e1`, branch `codex/e1-expert-width-screen`,
base `codex/perf-v1` at `7b4d539f9b`. No commits or pushes.
Production source/launcher/venv/checkpoint remain unchanged; private frozen
launcher snapshots redirect mutable runtime caches and use the worktree
`PYTHONPATH` overlay following C1. Existing production Python is read-only.

## Design and predeclared gates

Read: ASTRA_REVIEW_2026-09-08 rank 8, ASTRA_REVIEW_2026-09-07 Finding 6,
S1_SPARSE_FP4 Part A, MOE_SMALLM_SPEC, EXCLUSIVE_TIME_MAP_0907, BN1 and the
existing BS=1 quality battery. This experiment changes routed target and MTP
expert weights (48 target layers + 1 draft layer); shared experts are excluded.

`SGLANG_E1_WIDTH=off|512|576`, default off; mask file supplied through
`SGLANG_E1_MASK_PATH`. Transform after gate/up deinterleaving and before
backend swizzling. Zero corresponding full gate/up rows and down columns,
retain the original dense 640 shape and every original scale byte. Dense
kernels, production launcher and venv are not edited.

Choose whole contiguous block16 groups: rank the sum of 16 channel contribution
scores, retain the best 32/40 blocks for 512 and 36/40 for 576. Ties favor
lower original block indices. These are optimal top blocks, not unconstrained
top individual channels; preserving scale blocks is the constraint. The masks
are nested. Each down block's 8 packed bytes is either entirely retained or
positive-zeroed; gate/up rows are wholly removed. No partial scales or
requantization. Future physical packing must preserve within-block order and
copy the corresponding scales. Retained intermediate activation scale blocks
also remain independent of dropped whole blocks.

One W4 recorder server on the complete BN1 32-prompt sets. Read actual BF16
GEMM1 scratch retained after the existing fused-finalize GEMM2, using an
explicit workspace and an additional graph-capturable reduction kernel.
No BF16 replay approximation and no dense-kernel modifications. The recorder
observes the selected GEMM2 tactic and skips unfused-finalize calls whose
scratch is overwritten (including untuned startup calls); this selection and
all observed/skipped layer coverage are retained in the dump metadata. Restrict to
M<=16 decode/verify/draft-extend calls, excluding large prefills. Accumulate
per-layer/expert/channel mean `abs(silu(g)*u)` before second activation
quantization, and routed contribution norm `mean(abs(route_weight*h_NVFP4)*
L2(W_down[:,channel]))`. The second statistic reads the actual packed
activation and swizzled block16 scale from the workspace, including the
original global activation scale. It is the magnitude of each individual
channel's actual down-input contribution; it is not a signed
cancellation-aware output error or ablation gradient. P2-removed routes are
excluded by the actual compacted workspace row count. Down norms use the
checkpoint's actual block16/global scales; draft norms use online-quantized
weights. Zero-count experts use down norm as a disclosed fallback.
Warmup/capture/autotuning are excluded with a device enable scalar. Statistics
are dumped only by signaling a verified PID in the owned server group.

BN1 prompts are used for channel selection as explicitly requested, so later
BN1 acceptance is a **calibration-set diagnostic**, not a held-out claim.
The independent quality battery remains untouched. No BN1 prompt/data will
be fed into a future recovery-training pipeline.

Arms: recorder -> prod -> w512 -> w576, each holding its own
`flock -w 28800 $HOME/.gpu.lock` through owned process-group cleanup
and GPU-context drain. Memory fraction **0.900**, empty `SERVE_DISPLAY_HZ`,
BS=CONC=1, max requests 1, token pool 131072, Mamba slots 10, require >=4096 MiB
free at steady state. ~24 occupied GPU-hours maximum; per-arm limit 28000 s.
GPU lock waiting does not count as occupied time. HC1 may interleave arms.

Each evaluation arm: same non-holdout BN1 warmup; eight frozen prompts/domain,
one measured repeat, greedy; pre/post 18.5k needle at depths .1/.5/.9; quality
smoke then full fixed GSM8K/MMLU/HumanEval/JCQA plus five sanity generations.
Needle outcomes do not depend on the script's exit status alone. NI compares
candidate minus prod, margin .5 pp, paired one-sided 95% bounds. FAIL if
upper < -.5 pp, PASS if lower > -.5 pp, otherwise INCONCLUSIVE. Any NI FAIL
rejects the untrained mask; >3% pooled acceptance loss in a domain is another
screen failure. INCONCLUSIVE never establishes quality equivalence.
Throughput with zero-masked dense weights is informational, not a physical
packing speedup. No training or full-model packing is run in E1.

## Physical packing arithmetic (planning, not measured packed kernels)

Per expert, gate+up 1,843,200 B + down 921,600 B = 2,764,800 B (including
block16 scales, excluding unchanged global scalars). 512 keeps 2,211,840 B,
saving 552,960 B (20%); 576 saves 276,480 B (10%).
EXCLUSIVE_TIME_MAP_0907's D-dependent coefficient is 69.9 us per unit D,
already aggregated across profiled target calls; **do not multiply by 48 again**.
D=18.8/52.3 gives 1314.12/3655.77 us at historical W4/W16 step 8919/16083 us.
No extra draft-layer saving is credited to this target-only model.

Let q=retained width/640 and eta=new/old bandwidth efficiency. Saving is
`S=Dterm*(1-q/eta)`, comprising ideal `Dterm*(1-q)` minus efficiency loss
`Dterm*q*(1/eta-1)`. Throughput ratio with acceptance ratio r is
`r*T/(T-S)`. >=3% requires `S>=T*(1-r/1.03)`.
| Width | Profile | Efficiency eta | Ideal us | Efficiency loss us | Net saving us | Throughput at unchanged acceptance |
|---|---|---:|---:|---:|---:|---:|
| 512 | W4 | 1.00 | 262.8 | 0.0 | 262.8 | +3.04% |
| 512 | W4 | .95 | 262.8 | 55.3 | 207.5 | +2.38% |
| 512 | W4 | .90 | 262.8 | 116.8 | 146.0 | +1.66% |
| 512 | W16 | 1.00 | 731.2 | 0.0 | 731.2 | +4.76% |
| 512 | W16 | .95 | 731.2 | 153.9 | 577.2 | +3.72% |
| 512 | W16 | .90 | 731.2 | 325.0 | 406.2 | +2.59% |
| 576 | W4 | 1.00 | 131.4 | 0.0 | 131.4 | +1.50% |
| 576 | W16 | 1.00 | 365.6 | 0.0 | 365.6 | +2.33% |

+3% requires 259.8/468.4 us W4/W16 at unchanged acceptance. For 512, the
minimum eta is .9971/.9176; at eta=.95 W16 tolerates only .697% acceptance
loss. Ideal W4 tolerates only .035%. The matched W4 effective step and
acceptance are available in `analysis.json`; no W16 acceptance was measured.
Do not transfer the W4 acceptance ratio to W16.

A 5% relative efficiency loss is a planning sensitivity, not an established
prediction. The small-M wave model in MOE_SMALLM_SPEC section 8.4 gives a
stronger caution: at D52 with 128-output tiles, gate/up goes from 520 to 416
tiles but still occupies three 188-SM waves. Down retains its output tiles
while K-loop work decreases. Weighting old gate/up/down wave bytes yields
only ~6.67% of the D term saved (about 244 us W16), an eta~.857 case and only
~1.54% ideal step throughput. At D19 gate/up crosses a wave boundary and the
same coarse model predicts an efficiency improvement. These discontinuities
mean an average D and a blanket 5% penalty are insufficient: actual route
histograms and packed-shape timings are required. The wave model is a
scheduler sensitivity, not a measured packed kernel or guaranteed latency.
576 cannot reach +3% through the target-only byte term even ideally.

## Matched W4 economics (informational dense-mask runs)

The historical D-term table is not combined with current acceptance as a
measured forecast. From these matched arm summaries, let T0 and Tc be the
observed client-effective microseconds/verify, r=Ac/A0. The additional saving
needed from the current masked arm for +3% throughput is `Tc-T0*r/1.03`.
A negative number is current observed headroom, not extra kernel savings.
These are one-run calibration-set results with changed generations, not
restart-controlled adoption evidence. No estimate of the candidates' actual
D histogram or packed kernel cost is supplied by this calculation.

| Workload | Arm | Prod effective us/verify | Candidate effective us/verify | Observed dense t/s change | Additional us/verify saving needed for +3% |
|---|---|---:|---:|---:|---:|
| code-edit | w512 | 10745.0 | 11529.9 | -7.64% | +1190.8 |
| code-edit | w576 | 10745.0 | 11078.4 | -2.27% | +567.2 |
| prose-en | w512 | 9626.5 | 10750.8 | +3.14% | -15.0 |
| prose-en | w576 | 9626.5 | 10254.6 | -0.02% | +300.4 |
| prose-ja | w512 | 9739.3 | 11063.0 | +18.05% | -1616.3 |
| prose-ja | w576 | 9739.3 | 10282.4 | +14.12% | -1110.4 |
| agent-loop | w512 | 10215.5 | 10941.8 | -3.33% | +672.7 |
| agent-loop | w576 | 10215.5 | 10420.6 | +2.06% | +95.5 |

This reinforces the need for a new matched physical-packing experiment:
masked code-edit currently needs much more additional saving than the old
ideal byte estimate, while observed prose acceptance/throughput improvements
cannot rescue a candidate that fails the quality screen. W16 remains unmeasured.

## Recovery budget basis

MT1 uses V5's 1250 optimizer steps, 8192 tokens x accumulation 3, K=3,
initializing from V5; 30.72M optimizer token presentations. The MT1 capture
is private data; neither its size nor MT1's training time is published. MT1
trains a single MTP module, not the 48-layer target, so E1 recovery (target
repair as well as draft alignment) cannot take its cost from MT1.
MT1's training log (not published) reports 2.604201B trainable parameters. The
retained 48 target expert banks alone contain 96.637B parameters at width512,
37.1 times that trainable count (before other target modules). BF16 weights
alone require 193.27 GB; conventional full Adam states/gradients require
well over 1 TB, so the existing single-module recipe cannot be run as a
full-target optimizer on this GPU.

Planning reference: a future research stage would start with a **pilot on
one or two representative layers** and, only if it transfers, a
**layerwise/distillation program** around 2.5–4.5M fresh parent-disjoint
tokens and up to 30.72M token presentations/layer. Its GPU-time cost has to
be measured in the pilot; nothing here predicts it.
A global adapter/offload method requires a new measured memory/throughput
preflight and cannot inherit these estimates. BN1 and the quality holdouts
must be excluded. No recovery training is authorized or performed by E1.

## Implementation validation

Three tests pass: deterministic block selection/ties/nesting/invalid scores;
full 512-expert packed-byte mask invariants and partial-block rejection;
default-off hook accepts an unrelated object without accessing weights.
GPU preflight PASS at M1/4/8/16: actual BF16 GEMM1 values match the independent
constant-weight reference exactly; actual NVFP4 down-input contributions and
down-column norms match their references; two CUDA graph replays increment
the expected counts in every condition. Final output uses BF16 atomic
finalization and is not bitwise deterministic at M4/8/16. Maximum differences
with explicit workspace were 0.0009765625/0.00390625/0.00390625, exactly the
same maxima as ordinary-control versus repeated-control runs. The test uses
a two-BF16-ulp relative bound and also reconstructs the synthetic down output
from the quantized intermediate. M1 was bit-identical. Evidence:
`sglang-e1/bench_e1/recorder-preflight.json`.

Preflight failures are preserved: missing existing venv `ninja` on PATH;
untuned generic tactic overwriting scratch; inappropriate bit-exact final
output assertion under BF16 atomics; Triton FP8 masked-load zero literal type.
All were resolved before any E1 server arm. Production kernels were not
changed; the observer verifies fused-finalize tactic selection.

Supplemental GPU layout test PASS at M1/4/8/16 with ten shared experts,
multiple rows/expert, random signed nibbles and row/channel-varying block
scales. The independent reference uses the production scale swizzle rather
than the recorder scalar-index expression. Maximum contribution absolute
error was 9.54e-7; maximum SwiGLU sum error was 1.15e-5. Evidence:
`sglang-e1/bench_e1/workspace-row-preflight.json`. This ran under its own
shared GPU lock before the prod arm.

## Final decision and evidence audit

**Verdict: NEEDS RECOVERY TRAINING; untrained narrowing is not viable.**
Width512 fails non-inferiority on GSM8K, MMLU and HumanEval. Width576 fails
MMLU and HumanEval. Neither candidate establishes non-inferiority on all four tasks.
Both pass the calibration-set acceptance-loss screen and all six needles,
which does not override the independent quality failures. No recovery success
has been demonstrated, so this verdict means recovery is necessary, not that
recovery is known to be sufficient.

The primary 512 proposal remains a conditional research candidate rather than
an established speed optimization. W4 has almost no bandwidth-efficiency or
acceptance-loss headroom at the historical +3% threshold. W16 has some ideal
headroom, but the small-M wave sensitivity can erase it. Before any substantial
recovery program, gate it on packed-shape timing with representative route
histograms; then use the layer-local pilot described above.
Width576 is a quality sensitivity arm: at unchanged acceptance and efficiency,
its target-byte saving alone cannot reach +3% in either profile. The measured
W4 acceptance increases do not establish an independent held-out improvement,
and cannot be transferred to W16 or promised after training.

Evidence audit `sglang-e1/bench_e1/results/e1-20260908-a/audit.json`: PASS.
This is an experimental-integrity result, not a quality PASS. Verified all
source/config/data hashes and checkpoint metadata frozen for the study,
49-layer mask hashes/selection/nesting, actual BS1/W4 settings, 32 identical
BN1 prompt identities per arm, 4002 scored questions per evaluation arm,
zero recorded transport errors, and all 18 evaluation needles. Minimum
steady free VRAM across all servers: 8432 MiB. Total server lock occupancy:
9666.03 s = 2.685 hours, excluding separate GPU preflights and shared-lock
waiting. All four owned server process groups are gone after cleanup.
No commit or push; production tracked source and frozen launchers unchanged.

One recorder captured 25,025 of 25,088 layer-experts (99.749%); 63 zero-count
experts use the explicitly disclosed down-norm fallback. It captured all 512
MTP experts. No eligible small-M calls were skipped for overwritten scratch;
large prefills were excluded by design. Full explicit means and observation
flags are in `activation-means.pt`, and every fallback is in `mask-summary.json`.

Mask SHA256:

- w512: `70cb6e9d10177f73dc0b626a170a44747dcadd841b82867d6769d58c35cb2405`
- w576: `e771bd8c8ef5437e030f149f47afd878abf18f94a18b6461dc14b325f4aae55b`

## Output diagnostics (same fixed scoring; no post-hoc rescoring)

The existing generation caps and grader were preserved. Truncated responses
remain in the denominator and are graded by the existing harness. In MMLU,
86 of w512's 169 prod-correct/candidate-wrong cases reached the generation
cap; 85 had no parsable predicted letter. For w576 the corresponding counts
are 23 of 83, with 23 empty predictions. Inspected examples start explaining
instead of obeying the requested letter-only answer. This is partly a response
format/instruction-following regression; it is not evidence that all lost
answers represent missing factual knowledge. No longer-output rescue was run.

| Truncated scored responses | prod | w512 | w576 |
|---|---:|---:|---:|
| gsm8k | 152 | 190 | 159 |
| mmlu | 12 | 144 | 45 |
| humaneval | 3 | 6 | 5 |
| jcqa | 0 | 0 | 0 |

Full five-prompt sanity battery (prompts in `bench/bench/quality/run_bench.py`; tokens / finish / repeated-4gram fraction):

| Sanity prompt | prod | w512 | w576 |
|---|---|---|---|
| prose-en | 1500 / length / 0.023 | 1436 / stop / 0.039 | 1273 / stop / 0.030 |
| prose-ja | 1061 / stop / 0.036 | 1500 / length / 0.477 | 1048 / stop / 0.011 |
| code-edit | 1500 / length / 0.763 | 1500 / length / 0.763 | 1500 / length / 0.763 |
| agent-loop | 275 / stop / 0.423 | 517 / stop / 0.456 | 688 / stop / 0.572 |
| essay | 1500 / length / 0.010 | 1500 / length / 0.008 | 1500 / length / 0.004 |

Direct inspection of the full Japanese sanity generations confirms a long
repeated clause loop at the end of w512, ending at the 1500-token cap; prod
and w576 reach a narrative ending. Its repeated-4gram fraction rises from
.036 to .477. High code-edit repetition is present in prod as well and is
not uniquely attributed to pruning. The BN1 acceptance gain therefore must
not be interpreted as a quality improvement. Raw generations and diagnostics
are retained; these five samples do not estimate a population failure rate.

Artifacts: `sglang-e1/bench_e1/README.md` describes implementation and replay;
`sglang-e1/bench_e1/results/e1-20260908-a/` contains recorder statistics, masks,
freeze manifest, per-arm logs, memory series, `analysis.json`, `audit.json`,
`output-diagnostics.json` and `verdict.json`. Full scored generations are at
`flash-next-bench/bench/quality/runs/e1-20260908-a-{prod,w512,w576}/`.

## Incremental measurements

Run `e1-20260908-a`. Missing entries are not passes.

| Quality benchmark | prod | w512 | w576 |
|---|---:|---:|---:|
| gsm8k | 1162/1319 (88.10%) | 1129/1319 (85.60%) | 1162/1319 (88.10%) |
| mmlu | 1189/1400 (84.93%) | 1049/1400 (74.93%) | 1141/1400 (81.50%) |
| humaneval | 158/164 (96.34%) | 153/164 (93.29%) | 154/164 (93.90%) |
| jcqa | 1088/1119 (97.23%) | 1081/1119 (96.60%) | 1086/1119 (97.05%) |

| Candidate | Benchmark | Difference (pp) | One-sided 95% lower / upper (pp) | NI at 0.5 pp | Discordances win / loss |
|---|---|---:|---:|---|---:|
| w512 | gsm8k | -2.50 | -3.91 / -1.13 | FAIL | 45 / 78 |
| w512 | mmlu | -10.00 | -11.64 / -8.44 | FAIL | 29 / 169 |
| w512 | humaneval | -3.05 | -6.13 / -1.38 | FAIL | 0 / 5 |
| w512 | jcqa | -0.63 | -1.40 / +0.08 | INCONCLUSIVE | 8 / 15 |
| w576 | gsm8k | +0.00 | -1.29 / +1.29 | INCONCLUSIVE | 52 / 52 |
| w576 | mmlu | -3.43 | -4.74 / -2.18 | FAIL | 35 / 83 |
| w576 | humaneval | -2.44 | -5.32 / -0.78 | FAIL | 0 / 4 |
| w576 | jcqa | -0.18 | -0.79 / +0.41 | INCONCLUSIVE | 6 / 8 |

Candidate minus prod; paired Tango score bounds, recomputed from the discordant counts shown (they replace the original Wald bounds, which are unreliable with few or zero discordances; w576 HumanEval moved from INCONCLUSIVE to FAIL). These are separate one-sided 95% bounds, not a two-sided 95% interval. INCONCLUSIVE is not PASS.

| BN1 W4 workload | prod acceptance / t/s | w512 acceptance / t/s | w512 acceptance change | w576 acceptance / t/s | w576 acceptance change |
|---|---:|---:|---:|---:|---:|
| code-edit | 3.8940 / 362.40 | 3.8593 / 334.72 | -0.89% | 3.9235 / 354.16 | +0.76% |
| prose-en | 2.3830 / 247.55 | 2.7450 / 255.33 | +15.19% | 2.5381 / 247.51 | +6.51% |
| prose-ja | 2.4120 / 247.66 | 3.2343 / 292.35 | +34.09% | 2.9062 / 282.63 | +20.49% |
| agent-loop | 3.1733 / 310.64 | 3.2856 / 300.28 | +3.54% | 3.3036 / 317.02 | +4.11% |

Acceptance is pooled completed tokens / exact verify-counter delta. Throughput is pooled tokens / client decode seconds. One run/arm, eight BN1 parents/domain, one repeat; informational, without a restart-controlled or held-out acceptance guarantee. Recorder run excluded.

| Arm | Completion | Needle pass / count | Minimum steady free MiB | Lock hours |
|---|---|---:|---:|---:|
| recorder | Complete | 0 / 0 | 9298 | 0.277 |
| prod | Complete | 6 / 6 | 8432 | 0.794 |
| w512 | Complete | 6 / 6 | 9072 | 0.820 |
| w576 | Complete | 6 / 6 | 8771 | 0.794 |

Recorder coverage: **25025/25088 layer-experts**, 132,664,970 retained-route observations. Zero-count experts use down-column norm as an explicitly uncalibrated fallback.

Removed summed routed contribution norm: w512 **16.759%**, w576 **8.079%**. This is not output error or task-quality loss.

Final verdict: **NEEDS RECOVERY TRAINING. Both untrained masks fail the quality screen; physical-packing speed and successful recovery remain unverified.**
