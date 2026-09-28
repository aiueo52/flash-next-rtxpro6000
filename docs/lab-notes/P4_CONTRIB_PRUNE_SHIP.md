# P4 contribution pruning ship candidate

Status: **NO-SHIP**. Implementation, CPU/GPU correctness, and all six W4/W16 control/contrib/control arms are complete. All four traced speed cells miss the >=3% gate; seven of eight acceptance cells exceed +/-1%. The conditional BS=1 wa quality battery was not run. Production remains unchanged. Finalized 2026-09-08 JST.

## Frozen scope

- Base: `codex/perf-v1` at `446c801189165380465a73782c93658576283372`.
- Worktree `$HOME/tools/sglang-p4`, branch `codex/p4-contrib-prune`.
- Private FlashInfer package copied with `cp -a --reflink=auto` from the production venv into `$HOME/tools/flashinfer-p4`; no hardlinks to production.
- Overlay and JIT cache: `flashinfer-p4:sglang-p4/python`, `SGLANG_CACHE_DIR=$HOME/.cache/sglang-p4`.
- Production launcher, tree and venv are read-only. No commit, push or merge.
- Every GPU/server command owns `flock -w 28800 $HOME/.gpu.lock`; one ownership period per arm, cleanup only of its own process group. All arms memory .920, display override empty, at least 4096 MiB free at steady state.

## Policy and calibration

P3's `joint` simulator uses the **sum of both route scores** below the threshold, with neither route protected. P4 preserves this exact group rule (which also implies each nonnegative route score is below threshold), not the weaker independent-per-route rule. Stable first router-weight maximum is protected. Multiplicity is computed from immutable original routing; no recount or weight renormalization.

P3 fitted a separate 48x512 table at each width. Preserve both tables and their thresholds rather than pooling widths without recalibration: W4 0.04199233496772392, W16 0.04226923952110562. Training consists of the first floor(steps/2) complete 48-layer verify steps in each of the four workloads at that width. The temporal second halves are evaluation only. Tables use eight layer-mean pseudo-observations per expert. Runtime storage is FP32. Calibration provenance and hashes are saved in `prune/p4-manifest.json`.

## Predeclared gates

1. CPU decisions agree with P3; GPU immutable-input prologue agrees with CPU, including graph replay; policy-unset tau .08 retains P2 outputs.
2. W4 and W16: control .08 -> contrib -> control, 3 workload repeats, code-edit/prose-en/prose-ja/agent-loop. Trimmed step time at least 3% faster at both widths, acceptance change within +/-1%; report measured D, GEMM1/2 per-call medians and tokens/s. CUPTI code-edit/prose-en; needle per arm.
3. Only if speed passes: BS=1 wa quality arms prod/contrib/noprune/prod2, corrected analyzer, margin .5 pp, one-sided paired lower bound above -.5 pp, significance verdict and needle 6/6. Inconclusive is not shippable.

## Execution evidence

- 2026-09-07 23:35 JST locked host preflight: RTX PRO 6000 Blackwell Max-Q, 97887 MiB total, 93327 MiB free, no compute processes. Sandbox has no NVIDIA devices; GPU commands require host execution.

### Interrupted session and recovery (2026-09-08 00:01 JST)

The user reported a system-wide desktop freeze at 23:51 JST, killing this process; the desktop was restarted. The original GPU probe log is empty: no server arm or GPU result was completed, and no partial speed table existed. Worktree diff check passes; all six recorded production file/launcher hashes remain unchanged. Locked recovery preflight: 93646 MiB free, no compute or SGLang processes. Restarted correctness probe with fresh label `gpu-probe-20260908-0001`; no old arm result is reused.

Timing definition is predeclared as the 10% trimmed mean of **whole** draft-to-next-draft wall intervals, then median across three independent traces. `trimmed_step.py` explicitly warns its old per-kernel clipping metric is biased; that legacy statistic will not be used for the gate. Require >=3% saving for both traced workloads at both widths and +/-1% acceptance in each of the four workload/width cells. The control comparator is the mean of the initial and final control arm medians. Effective D is recorded in a separate diagnostic server during the same arm's lock ownership, to keep census kernels out of the timing traces.

### FP32 strict-cut representation

The original FP64 P3 tables/thresholds have two singleton decisions exactly at FP32 rounding boundaries (both W16: one training route, one held-out route). Nearest FP32 threshold representation keeps these routes. Use a uniform numerical conversion rule at both widths: `nextafter(float32(P3_threshold), +inf)`, while retaining the unmodified FP32-rounded norm tables. Runtime cuts are W4 **0.04199234023690224**, W16 **0.04226924479007721**; the manifest records both original and runtime thresholds. This restores **all original P3 joint decisions on all eight captures** in the full replay check; it is a representation change of ~5e-9, not a retuned accuracy/performance threshold. The diagnostic explicitly used the original FP64 tables, not only a simulator with FP32 tables. Complete count-two off/on replay validation is recorded separately.

### CPU verification

`runs/p4/cpu-policy-20260908-0004.log`: PASS on **55,737,600 route comparisons** (27,868,800 captured routes, joint disabled and enabled), with **0 differences against the original P3 FP64 table/threshold decisions** after the documented strict-cut conversion. Independent CPU reference uses immutable ids/weights, original group counts, stable first top-1, whole-pair protection, deterministic FP32 sums, and whole-call unpruned fallback for invalid inputs. Synthetic checks cover protected top-1, sum-vs-individual pair cutoff, count-two off, nonfinite denominator, and invalid id.

The first post-restart GPU attempt compiled the private csrc module, then stopped at an FFI argument-count mismatch (26 old public arguments versus 32 supplied). No model server ran. The public TVM-FFI lambda was extended to forward the six new typed arguments; the fresh validation run is `runs/p4/validation-20260908-0006.log`.

## Deployment recipe (review only; NO-SHIP, do not apply or merge)

The candidate requires both the SGLang worktree changes and the private FlashInfer patch. The current production chain is `a0 -> a3 -> g2-1 -> g2-2 -> g1-pack-only -> p2`; the proposed extension is `-> p4-contrib-prune.patch`, applied from the venv `site-packages` directory with `patch -p1`. The P4 patch includes the Python FlashInfer API and typed FFI bridge as well as the CUDA prologue/header, so applying only the `.cuh` is insufficient. After a SHIP verdict only, integrate the uncommitted changes on `codex/p4-contrib-prune` into `codex/perf-v1`, retain the calibration manifest and both 48x512 FP32 tables, and add these defaults beside the P2 settings in `serve-fast.sh`:

```bash
export SGLANG_MOE_PRUNE_POLICY=${SGLANG_MOE_PRUNE_POLICY:-contrib}
export SGLANG_MOE_PRUNE_NORM_MANIFEST=${SGLANG_MOE_PRUNE_NORM_MANIFEST:-$HOME/tools/flash-next-bench/prune/p4-manifest.json}
export SGLANG_MOE_PRUNE_CONTRIB_W4=${SGLANG_MOE_PRUNE_CONTRIB_W4:-0.04199234023690224}
export SGLANG_MOE_PRUNE_CONTRIB_W16=${SGLANG_MOE_PRUNE_CONTRIB_W16:-0.04226924479007721}
export SGLANG_MOE_PRUNE_CONTRIB_JOINT=${SGLANG_MOE_PRUNE_CONTRIB_JOINT:-1}
```

No production edit, merge, commit or push has been performed. Policy unset retains P2 singleton tau behavior; `contrib` disables P2, and missing/invalid calibration or unsupported execution falls back to unpruned computation. The experiment uses private launcher copies whose interpreter paths point to the production venv read-only; all experiment logs and JIT outputs stay private. The production rollout recipe remains conditional on every gate, including non-inferiority.

### GPU correctness (2026-09-08 00:13–00:14 JST)

`runs/p4/validation-20260908-0006.log`: **P4_VALIDATION_COMPLETE**.

| Check | W4 | W16 |
|---|---:|---:|
| Captured-routing graph replays, four workloads | 96/96 | 96/96 |
| GPU effective ids/weights equal CPU reference | 96/96 | 96/96 |
| Original ids/weights unchanged | 96/96 | 96/96 |
| Deterministic output equal explicitly prepruned reference | 96/96 bit-identical | 96/96 bit-identical |
| Dropped routes across replay sample | 1452 | 3084 |
| Nonfinite denominator / invalid-id whole-call fallback | PASS | PASS |
| Existing HC K0/K2 inverse L2 export, apply off/on, changed graph inputs | PASS, rtol 2e-6 | PASS, rtol 2e-6 |
| Policy unset: output vs existing P2 private build | 48/48 bit-identical | 48/48 bit-identical |
| Policy unset: routing vs existing P2 and P1 Triton reference | 48/48 | 48/48 |

The P2 private build's core Python, prologue and header are byte-identical to the production copies. Production JIT caches were not used by the test. HC normalized tensors were bit-identical with/without the extra norm-statistics output. All correctness GPU commands owned the shared lock. Speed sequence queued under label `p4-20260908-0014`; each arm owns the lock independently.

Arm `p4-20260908-0014-w4-control` complete at Tue Sep  8 00:28:14 2026: needle 3/3; BS=1, memory .920, private logs `runs/p4/p4-20260908-0014-w4-control`.

Additional locked GPU check `runs/p4/joint-off-20260908-0019.log`: count-two disabled passes another **96/96 captured calls at each width**, including changed graph inputs, immutable originals, exact effective routing, deterministic output equality and invalid-input fallback. Dropped routes: W4 1148, W16 2328. With the earlier joint-enabled check, both switch states total **384 captured GPU calls**, all passing.

### Acceptance measurement correction, before any candidate result

The current server exports a per-finished-request `sglang:spec_verify_calls_total` counter, unlike the older P2 log-only measurement. Before W4 contrib acquired its lock or produced any result, the decision metric was fixed to `delta(generation_tokens_total) / delta(spec_verify_calls_total)` for each **measured** fnbench request, then median of three requests. This is exactly the current tokenizer manager's `spec_accept_length` definition (includes bonus token). Assert the token-counter delta equals that request's completion token usage; a stale/misaligned counter fails analysis rather than silently passing the gate. The original windowed log averages (which include each client's warmup) remain as diagnostics only. W4 first-control exact medians: code-edit 3.864734, prose-en 2.580645, prose-ja 2.366864, agent-loop 3.260870. The +/-1% gate and all other thresholds are unchanged.

Arm `p4-20260908-0014-w4-contrib` complete at Tue Sep  8 00:47:52 2026: needle 3/3; BS=1, memory .920, private logs `runs/p4/p4-20260908-0014-w4-contrib`.

Arm `p4-20260908-0014-w4-control2` complete at Tue Sep  8 00:57:11 2026: needle 3/3; BS=1, memory .920, private logs `runs/p4/p4-20260908-0014-w4-control2`.

Arm `p4-20260908-0014-w16-control` complete at Tue Sep  8 01:09:44 2026: needle 3/3; BS=1, memory .920, private logs `runs/p4/p4-20260908-0014-w16-control`.

### Measurement scope and preservation

Controls use the same private P4 overlay with the policy unset and singleton tau .08; the unchanged P2 policy was separately checked for bit-identical routing and deterministic outputs against the production-identical private P2 build. Thus these are production-policy controls, not measurements of the untouched production process. No production server is started.

There are three independent measured fnbench requests for each arm/workload and three CUPTI captures for each traced workload. Each fnbench invocation performs its own warmup. The separate D census is one additional warmup-plus-measured workload replay per arm; D is the mean distinct effective expert count over its recorded target layer calls, not a paired counterfactual using identical accepted token paths. Acceptance and D can therefore change together when pruning changes the computation. The 3-repeat timing requirement does not imply three independent D-only server runs.

An additional source snapshot, `runs/p4/sglang-p4-working-tree.patch`, preserves the five SGLang integration files including the two untracked modules (SHA256 `5946b8f20e826f45ac38510f6516cbc3444efc88bc04fa92e2fcdcc4255b4923`). No commit was created; a future merge must first carry these working-tree changes into a reviewable commit or equivalent patch integration. A merge of the current branch reference alone would not include uncommitted files.

Arm `p4-20260908-0014-w16-contrib` complete at Tue Sep  8 01:22:08 2026: needle 3/3; BS=1, memory .920, private logs `runs/p4/p4-20260908-0014-w16-contrib`.

## Reproduction entry points

Run from `$HOME/tools/flash-next-bench`. Before repeating the archived GPU validation wrapper, copy it and change its fixed `D` output directory to a fresh label so the original evidence is preserved. These commands record the original entry points; the calibration and runtime candidate stayed frozen throughout the measurements. The table builder is CPU-only and asserts equality to the original P3 FP64 tables before saving FP32 files.

```bash
source bench/moe_smallm/p4env.sh
python bench/quality/build_p4_norm_table.py
python bench/quality/test_p4_policy.py
flock -w 28800 $HOME/.gpu.lock runs/p4/validate.sh
flock -w 28800 $HOME/.gpu.lock runs/p4/joint-off-20260908-0019.sh
```

For a new single arm, use a fresh label and retain the per-arm lock (the arm wrapper starts both the timing and diagnostic servers under that one ownership period):

```bash
LABEL="p4-repro-$(date +%Y%m%d-%H%M%S)"
flock -w 28800 $HOME/.gpu.lock runs/p4/run_arm.sh "$LABEL" w4 control
flock -w 28800 $HOME/.gpu.lock runs/p4/run_arm.sh "$LABEL" w4 contrib
flock -w 28800 $HOME/.gpu.lock runs/p4/run_arm.sh "$LABEL" w4 control2
```

Repeat that sequence at W16 with the same fresh label for a complete new comparison. `bench/moe_smallm/p4_report.py LABEL` requires all six completed arms, validates finished-request acceptance counters, and writes the speed JSON and report tables. `bench/moe_smallm/p4_audit.py LABEL` audits completeness, needles, memory minima, production-file preservation, calibration hashes and byte-identical reconstruction of the private FlashInfer package from the delivered patch. The recorded master script `runs/p4/measure-20260908-0014.sh` contains the executed sequence and the conditional BS=1 wa prod/contrib/noprune/prod2 quality sequence; its existing label must not be reused.

Arm `p4-20260908-0014-w16-control2` complete at Tue Sep  8 01:46:37 2026: needle 3/3; BS=1, memory .920, private logs `runs/p4/p4-20260908-0014-w16-control2`.

## In-server repeated speed results

Label `p4-20260908-0014`. 10% trimmed mean of full draft-to-next-draft wall intervals; median across 3 traces; not legacy per-kernel clipping. Controls: mean of the two control arm medians.

| W | workload | arm | trimmed step ms | GEMM1 us/call | GEMM2 us/call | measured D | accept length | t/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | code-edit | control | 9.994 | 31.488 | 18.240 | 22.980 | 3.865 | 374.375 |
| 4 | prose-en | control | 9.475 | 26.656 | 16.608 | 18.398 | 2.581 | 267.148 |
| 4 | prose-ja | control | — | — | — | 17.680 | 2.367 | 248.637 |
| 4 | agent-loop | control | — | — | — | 18.583 | 3.261 | 334.975 |
| 4 | code-edit | contrib | 9.953 | 24.896 | 16.096 | 16.574 | 3.768 | 390.569 |
| 4 | prose-en | contrib | 9.286 | 19.840 | 14.272 | 13.713 | 2.542 | 276.090 |
| 4 | prose-ja | contrib | — | — | — | 13.430 | 2.444 | 266.160 |
| 4 | agent-loop | contrib | — | — | — | 14.332 | 3.252 | 350.644 |
| 4 | code-edit | control2 | 9.733 | 31.552 | 18.368 | 22.802 | 3.709 | 376.999 |
| 4 | prose-en | control2 | 9.396 | 26.688 | 16.672 | 18.168 | 2.626 | 281.562 |
| 4 | prose-ja | control2 | — | — | — | 17.220 | 2.317 | 251.138 |
| 4 | agent-loop | control2 | — | — | — | 18.871 | 3.380 | 357.671 |
| 16 | code-edit | control | 16.810 | 63.392 | 34.912 | 59.557 | 10.169 | 565.268 |
| 16 | prose-en | control | 16.554 | 66.176 | 36.416 | 50.559 | 2.920 | 177.193 |
| 16 | prose-ja | control | — | — | — | 47.314 | 2.614 | 162.167 |
| 16 | agent-loop | control | — | — | — | 53.307 | 5.021 | 297.250 |
| 16 | code-edit | contrib | 16.749 | 57.088 | 30.496 | 47.993 | 9.091 | 540.021 |
| 16 | prose-en | contrib | 16.227 | 56.288 | 30.000 | 42.025 | 3.217 | 200.012 |
| 16 | prose-ja | contrib | — | — | — | 40.129 | 2.620 | 165.243 |
| 16 | agent-loop | contrib | — | — | — | 43.816 | 4.819 | 294.268 |
| 16 | code-edit | control2 | 16.822 | 63.264 | 34.928 | 59.165 | 11.483 | 639.580 |
| 16 | prose-en | control2 | 16.579 | 66.672 | 36.544 | 50.891 | 3.226 | 193.808 |
| 16 | prose-ja | control2 | — | — | — | 45.676 | 2.709 | 169.000 |
| 16 | agent-loop | control2 | — | — | — | 52.995 | 6.417 | 371.505 |

| W | workload | step faster % | acceptance change % | t/s change % | additional D removed |
|---|---|---:|---:|---:|---:|
| 4 | code-edit | -0.913 | -0.513 | +3.961 | +6.317 |
| 4 | prose-en | +1.587 | -2.338 | +0.632 | +4.570 |
| 4 | prose-ja | — | +4.367 | +6.512 | +4.020 |
| 4 | agent-loop | — | -2.064 | +1.248 | +4.395 |
| 16 | code-edit | +0.400 | -16.030 | -10.359 | +11.368 |
| 16 | prose-en | +2.049 | +4.699 | +7.823 | +8.700 |
| 16 | prose-ja | — | -1.559 | -0.206 | +6.366 |
| 16 | agent-loop | — | -15.732 | -11.995 | +9.335 |

Speed gate: **FAIL**. >=3% faster for code-edit and prose-en at each width, acceptance within +/-1% for all four workloads at each width.

## Final diagnosis

The norm-table proxy reduces measured D by **4.020 to 11.368 experts per target layer call** beyond tau .08, and reduces both GEMM medians in every traced cell. The extra prologue work offsets much of that saving. Its exclusive GPU time rises by **0.265 to 0.293 ms per step**, whereas exporting the row norm in existing HC kernels adds at most 0.144 us to the HC-up per-call median. The following attribution is diagnostic; exclusive time is the full-trace per-step mean, then median across three captures, and is not substituted for the predeclared trimmed whole-step decision metric.

| W | workload | prologue us/call control -> contrib | prologue exclusive us/step control -> contrib | extra exclusive us/step | HC-up us/call control -> contrib |
|---|---|---:|---:|---:|---:|
| 4 | code-edit | 12.384 -> 17.984 | 230.086 -> 519.024 | +288.938 | 4.944 -> 5.088 |
| 4 | prose-en | 10.608 -> 14.720 | 215.677 -> 480.219 | +264.542 | 4.928 -> 5.056 |
| 16 | code-edit | 12.624 -> 18.401 | 306.380 -> 587.908 | +281.528 | 5.136 -> 5.184 |
| 16 | prose-en | 12.592 -> 18.336 | 300.515 -> 593.190 | +292.675 | 5.072 -> 5.120 |

The initial and final controls themselves show acceptance variability (e.g. W16 code-edit 10.169 -> 11.483 and agent-loop 5.021 -> 6.417). All individual arm values are retained above. This limits causal claims from the acceptance comparison; it does not produce a passing gate. W16 contrib is below both controls for those two workloads. Tokens/s alone would be misleading because acceptance changes with the target computation. No benchmark-accuracy degradation or statistical non-inferiority is claimed from these speed measurements.

## Conditional quality battery and non-inferiority

The master printed `P4_SPEED_GATE_FAIL: quality battery skipped per predeclared condition` and `P4_MEASURE_COMPLETE` with exit code 0. Per the requested conditional execution, **no BS=1 wa prod/contrib/noprune/prod2 quality arm was started**. Existing P3/Q1 quality outputs were not reused as P4 evidence.

| Benchmark | P4 evaluated n | contrib - prod pp | One-sided lower bound pp | NI margin | Significance verdict |
|---|---:|---:|---:|---:|---|
| GSM8K | Not run | N/A | N/A | 0.5 pp | Not evaluated |
| MMLU | Not run | N/A | N/A | 0.5 pp | Not evaluated |
| HumanEval | Not run | N/A | N/A | 0.5 pp | Not evaluated |
| JCQA | Not run | N/A | N/A | 0.5 pp | Not evaluated |

Quality needle 6/6 per wa arm: **not evaluated**, since those arms were not started. Speed-arm needle is a separate result: 3/3 in each of six arms, **18/18 total**. The unchanged corrected analyzer and its candidate-minus-production orientation are wired in `bench/moe_smallm/p4_quality_report.py`, but have not been exercised on P4 quality samples.

## Final gates and preservation audit

| Gate | Result | Evidence |
|---|---|---|
| CPU reference equals original P3 decisions | PASS | 55,737,600 route comparisons, zero mismatches; joint off/on |
| GPU prologue equals CPU, immutable inputs, graph replay | PASS | 384 captured calls across both widths and joint states |
| Policy unset P2 compatibility | PASS | 96 calls total, routing and deterministic output bit-identical |
| Existing-kernel inverse norm | PASS | W4/W16, HC apply off/on, changed graph inputs, rtol 2e-6 |
| Trimmed step >=3% faster | FAIL | W4 -0.913% / +1.587%; W16 +0.400% / +2.049%; 0/4 pass |
| Acceptance within +/-1% | FAIL | 1/8 cells pass |
| Speed-arm needle | PASS | 18/18 |
| BS=1 wa quality NI and significance | NOT RUN | Conditional speed prerequisite failed |
| Memory and isolation | PASS | minimum 8545 MiB free, all fractions .920, display override empty, private JIT |
| Production files and patch reconstruction | PASS | six recorded hashes unchanged; both patches reconstruct their matching private files |
| Final verdict | **NO-SHIP** | Do not apply the proposed production patch-chain/launcher changes or merge the candidate |

VRAM monitor minima (MiB), measured from readiness through each stage:

| W | arm | timing minimum | census minimum | needle |
|---|---|---:|---:|---|
| 4 | control | 8545 | 9037 | 3/3 |
| 4 | contrib | 11056 | 11810 | 3/3 |
| 4 | control2 | 11396 | 11810 | 3/3 |
| 16 | control | 10195 | 10194 | 3/3 |
| 16 | contrib | 9440 | 10194 | 3/3 |
| 16 | control2 | 9600 | 10194 | 3/3 |

Completeness: **72 measured requests, 36 CUPTI traces, 24 D census files**, all six `COMPLETE` markers. The acceptance analyzer validated each measured request token-counter delta against its returned completion token count. No traceback, runtime CUDA error, out-of-memory exception, or assertion was found in the 12 server logs. Routine process-group teardown messages occur after completed stage measurements. All 12 owned server process groups were absent in the final host process check; other jobs were not modified.

Evidence: `runs/p4/p4-20260908-0014-speed.json`, `runs/p4/p4-20260908-0014-audit.json`, `runs/p4/p4-20260908-0014-process-cleanup.json`, `runs/p4/p4-20260908-0014-exception-scan.json`, and `runs/p4/sglang-patch-reconstruction.json`.

## Delivered state

- FlashInfer patch: `bench/moe_smallm/patches/p4-contrib-prune.patch`, SHA256 `9726fd244fe62a2329e090e82a51826e5983da35b2afa2f7fa0412ce67384422`.
- Calibration: `prune/p4-manifest.json`, `prune/p4-norm-w4.npy`, `prune/p4-norm-w16.npy`; each table is FP32 48x512.
- Private package: `$HOME/tools/flashinfer-p4/flashinfer`, copied from the production package, no production hardlinks.
- SGLang: `$HOME/tools/sglang-p4`, branch `codex/p4-contrib-prune`, base `446c801189165380465a73782c93658576283372`, changes uncommitted. Source snapshot `runs/p4/sglang-p4-working-tree.patch` includes all five integration files.
- Private launchers and `bench/moe_smallm/p4env.sh`; CPU/table/GPU/arm/report/audit scripts remain available at the paths listed above.
- No commit, push, production patch application, or merge. No branch beyond the requested candidate branch was created.
