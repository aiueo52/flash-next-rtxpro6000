# E0 — MTP fc_embedding / fc_hidden FP8

2026-09-08. Completed incremental record. **Final gate: FAIL; keep SGLANG_MTP_FC_FP8 default off.** All six valid arms and 30 CUPTI traces completed; see Final E0 gate below.
Worktree `$HOME/tools/sglang-e0`, branch `codex/e0-mtp-fc-fp8`, base `codex/perf-v1` at `7b4d539f9b`. No commits, pushes, production source / launcher / venv edits.

## Source audit and predeclared gate

FP8C §1 is correct: the selector includes `mtp.fc_embedding` / `mtp.fc_hidden`, but `_init_linear_projections` constructs plain `nn.Linear`, bypassing quantization. Each checkpoint matrix is BF16 [2560,2560], 13,107,200 bytes. The new tensor-returning `MTPFcLinear(ReplicatedLinear)` exposes both to the existing `Qwen4ExpDenseFp8LinearMethod`. Its inherited post-load conversion stores E4M3FN weights (transposed view), one FP32 scale per output row, and uses the unchanged W8A16 GEMV for BF16 flattened M <=16; larger M uses the existing FP8 Linear fallback. HC inputs are reshaped to [M*hc_count,K] and outputs restored. Flag `SGLANG_MTP_FC_FP8=1` requires `mtp_dense` and `SGLANG_FP8_W8A16_GEMV=1`; default off retains the original nn.Linear construction.

Important production context: common exports already enable `SGLANG_MTP_EMBED_TABLE=1` (R3). Thus fc_embedding has **zero projection launches per steady draft forward** when table build succeeds. Its FP8 conversion changes the precomputed table; the table must remain BF16. E0 routes table construction through the identical live M=1 FP8 GEMV, including `apply_into` output rounding. fc_hidden operates on four HC rows per bs=1 forward. The 09-07 map reports 14 draft forwards at W16 / 2 at W4 and 220/228 us per W16 forward, but does not explicitly identify fc_hidden; stored trace ordering will be audited before attributing a named family.

A/B: W16 (3 repetitions per workload and 3 traces per code-edit/prose-en per arm), W4 (2 each), control → FP8 → control, lock held per arm using `flock -w 28800 $HOME/.gpu.lock`. Fixed four workloads: code-edit, prose-en, prose-ja, agent-loop; greedy. Memory fraction 0.920, SERVE_DISPLAY_HZ empty, steady free VRAM >=4096 MiB. No unrelated process termination; own server process groups only.

Gate: acceptance change within ±0.5% relative on **every** workload/width cell, compared with mean of both controls; trimmed whole-step wall >=0.5% faster at W16 for each traced workload. Predeclared timing: 10% trimmed mean of complete draft-to-next-draft wall intervals, median across repeats, comparator mean of two control arm medians. Also report requested historical `legacy_trimmed_ms` (per-kernel clipping; biased when kernel populations change), physical wall/busy, and draft-forward GPU span / union. Speedup arithmetic `(control_mean - fp8) / control_mean * 100`. Needle at 18,500 nominal tokens, depth 0.4, each arm. No launcher change unless gates pass.

## Validation (incremental)

CPU: 3 unit tests PASS. Real ReplicatedLinear weight loader, non-square shapes, 1-D / 2-D / HC 3-D / empty / non-contiguous / M>16 inputs, BF16 dtype and torch.nn.functional.linear equality; quant-method spy verifies flattening; table `apply_into` and fallback preserve destination.
GPU access: initial sandbox hides `/dev/nvidia*`; a read-only host execution confirms RTX PRO 6000 Blackwell Max-Q, 93050 MiB free / 97887 MiB total. GPU tests and arms use host execution under the required lock.

### GPU small test completed

Actual mtpft5 checkpoint weights; seed 20260908, BF16 Gaussian activations. Both layers select Qwen4ExpDenseFp8LinearMethod, FP8 weight and FP32 row-scale shape checks pass; scales agree with row amax/448. M=1/4/16 execute W8A16; M=64 exercises the inherited large-M FP8 fallback. All outputs finite and BF16; relative L2 <6% guard passes. M<=16 CUDA graph capture/replay is bit-identical to eager FP8, and M=1 table apply_into is bit-identical. These inputs are synthetic, not captured hidden activations.

| layer | M | max abs | max rel | relative L2 | W8A16 |
|---|---:|---:|---:|---:|---|
| fc_embedding | 1 | 0.10156250 | 72.66666 | 0.0278995 | True |
| fc_embedding | 4 | 0.08593750 | 207.00000 | 0.0260806 | True |
| fc_embedding | 16 | 0.08203125 | 573.43903 | 0.0265069 | True |
| fc_embedding | 64 | 0.14843750 | 33203.12500 | 0.0373534 | False |
| fc_hidden | 1 | 0.03320312 | 63.59535 | 0.0111492 | True |
| fc_hidden | 4 | 0.03906250 | 114.26060 | 0.0111117 | True |
| fc_hidden | 16 | 0.05468750 | 81.08696 | 0.0112483 | True |
| fc_hidden | 64 | 0.12500000 | 29663.08594 | 0.0288067 | False |

Max relative error is max(abs(FP8-BF16)/max(abs(BF16),1e-6)); near-zero references make it very large. Relative L2 is also reported to avoid disguising this denominator effect. Raw results: `specs/e0/gpu-results.json`; micro CUPTI traces adjacent. Initial standalone test setup attempts failed before comparison because CUDA_HOME, then venv PATH/ninja were missing; `bench/e0/gpu_test.sh` uses the existing launcher toolchain paths, with no installation.

### Historical fc attribution

The stored X2 trace sequence is table gather → full-HC RMSNorm → BF16 WMMA GEMM `[8,10,20]` → splitKreduce `[80,1,1]` → add → HC mixing → attention. The WMMA family labelled QSA q/k in the printed map is therefore fc_hidden, not the later attention path. Kernel medians: W16 11.008 us GEMM + 1.568 us reduction; median paired sums 12.608 / 12.576 us (code/prose). W4 paired sums 12.783 / 12.736 us. fc_embedding contributes 0 launches/forward with the R3 table. See `specs/e0/historical.json` and `bench/e0/analyze.py --historical`. This is not a pure 2-us launch-only operator: 13.1 MB BF16 matrix reads imply 8.19 us at the assumed 1.6 TB/s streaming roof, plus 1.6 us reduction and scheduling. FP8 halves matrix bytes and the existing GEMV finishes split-K reduction inside the same launch (workspace plus completion counters). The small warm-cache GPU trace confirms one `_w8a16_gemv_kernel` at `[80,5,1]`, 4.624 us for fc_hidden M=4. This micro value has a warm 6.55 MB matrix and is not a cold-streaming or server speedup measurement. End-to-end A/B decides the actual benefit.

### A/B started

`bench/e0/run_all.sh` holds the lock separately for each arm. To avoid even cache/log writes in the production tree, launchers are snapshots under `sglang-e0/bench/e0/launch`; only venv path references become absolute paths to the existing production venv, while caches are a private copied directory. All logs/traces/results go to `flash-next-bench/specs/e0`. No production launcher modification.

Timing definitions: the map's ~220/228 us W16 forward is **exclusive** GPU time; the same historical traces give ~265/264 us between consecutive routing prologues and ~274/269 us phase GPU-busy per forward. E0 records all three explicitly rather than comparing exclusive against inclusive time.

### Invalid initial control and corrected memory envelope

The initial W16 control startup was within the ready-state floor (4380 MiB), but generation lowered free memory to **2956 MiB**. It was stopped immediately on discovery by validating the E0 PYTHONPATH and terminating only its server process group. No A/B gate uses this partial arm. The allocator had made an oversized 672896-token KV pool (target K/V 7.70 GiB + draft 0.64 GiB) despite the 262144 context limit.

All replacement arms retain memory fraction **0.920** and now set **MAX_TOTAL_TOKENS=262144**, equal to the original context limit and well above all benchmarks/needle. This saves roughly 5 GiB of unused KV capacity without changing kernels at the measured sequence lengths. A separate host watchdog samples free VRAM every 2 seconds from ready to completion and stops only its own server group if free memory falls below 4096 MiB. Initial partial evidence is preserved in `specs/e0/invalid-vram-w16-control1`.

CPU validation extended to **6 tests PASS**: actual MTP default-off construction stays exactly nn.Linear; missing category/GEMV prerequisites reject; flag-on CPU construction selects both real Qwen4ExpDenseFp8LinearMethod instances with [64,64] BF16 source-load shapes. Conversion itself is validated separately on GPU. See `specs/e0/cpu-tests.log`.

### W16 control1 completed

Needle PASS; minimum monitored free VRAM 7904 MiB. n=3 workload repetitions and n=3 traces for each traced workload.

| workload | mean tokens/verify | mean t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|
| code-edit | 11.42110 | 633.51718 | 16.77203 | 16.07014 | 227.23165 | 12.67200 |
| prose-en | 3.35997 | 207.93838 | 16.48315 | 15.70155 | 234.03705 | 12.65550 |
| prose-ja | 2.81822 | 173.66530 | — | — | — | — |
| agent-loop | 5.15031 | 302.54481 | — | — | — | — |

Acceptance is unrounded completion_tokens / delta(sglang:spec_verify_calls_total), averaged across timed repeats (same definition as runs/acc.py). This counts delivered tokens per verify, including the bonus token; the ±0.5% gate is relative change of this length, not percentage points of a Bernoulli acceptance rate. Throughput is the arithmetic mean of client decode_tps outside profiling. The early control is not enough to decide a gate.

### w16-control1 measured results

Needle PASS; minimum monitored free VRAM **7904 MiB**.

| workload | tokens/verify | t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | draft period us | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 11.421104 | 633.517185 | 16.772033 | 16.070139 | 227.231647 | 267.679688 | 12.672000 |
| prose-en | 3.359968 | 207.938383 | 16.483150 | 15.701546 | 234.037051 | 265.168457 | 12.655500 |
| prose-ja | 2.818223 | 173.665298 | — | — | — | — | — |
| agent-loop | 5.150314 | 302.544806 | — | — | — | — | — |

Timed repetitions (unrounded ratios retained in analysis.json):

| workload | tokens/verify per repeat | t/s per repeat |
|---|---|---|
| code-edit | 11.70732, 12.12121, 10.43478 | 648.53, 671.23, 580.78 |
| prose-en | 3.21716, 3.33333, 3.52941 | 193.93, 201.49, 228.40 |
| prose-ja | 3.25203, 2.54777, 2.65487 | 200.35, 157.12, 163.53 |
| agent-loop | 5.26316, 5.30973, 4.87805 | 309.91, 309.30, 288.42 |

### w16-fp8 measured results

Needle PASS; minimum monitored free VRAM **8044 MiB**.

| workload | tokens/verify | t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | draft period us | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 12.268789 | 680.850857 | 16.606335 | 15.645689 | 223.659738 | 261.216797 | 7.328000 |
| prose-en | 3.212879 | 195.282555 | 16.440787 | 15.469713 | 231.742551 | 258.848145 | 7.264000 |
| prose-ja | 2.569399 | 160.115939 | — | — | — | — | — |
| agent-loop | 5.384748 | 318.007568 | — | — | — | — | — |

Timed repetitions (unrounded ratios retained in analysis.json):

| workload | tokens/verify per repeat | t/s per repeat |
|---|---|---|
| code-edit | 12.00000, 12.43523, 12.37113 | 667.53, 687.64, 687.39 |
| prose-en | 3.19149, 3.30579, 3.14136 | 195.95, 199.41, 190.49 |
| prose-ja | 2.60304, 2.47934, 2.62582 | 160.76, 155.41, 164.17 |
| agent-loop | 6.62983, 4.72441, 4.80000 | 387.95, 281.57, 284.50 |

### w16-control2 measured results

Needle PASS; minimum monitored free VRAM **7865 MiB**.

| workload | tokens/verify | t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | draft period us | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 9.679538 | 543.138775 | 17.057429 | 16.391458 | 228.550950 | 267.664062 | 12.656500 |
| prose-en | 3.196098 | 192.436694 | 15.247127 | 14.501265 | 233.524047 | 265.408691 | 12.640000 |
| prose-ja | 2.587071 | 158.924672 | — | — | — | — | — |
| agent-loop | 5.157919 | 303.127346 | — | — | — | — | — |

Timed repetitions (unrounded ratios retained in analysis.json):

| workload | tokens/verify per repeat | t/s per repeat |
|---|---|---|
| code-edit | 11.59420, 7.76699, 9.67742 | 648.14, 445.54, 535.73 |
| prose-en | 2.81030, 3.21716, 3.56083 | 168.76, 194.49, 214.06 |
| prose-ja | 2.67261, 2.47423, 2.61438 | 162.86, 152.53, 161.38 |
| agent-loop | 4.66926, 4.97925, 5.82524 | 276.33, 293.39, 339.66 |

### W16 gate available; W4 pending

W16 acceptance changes vs both-control mean: code-edit +16.2883%, prose-en -1.9876%, prose-ja -4.9303%, agent-loop +4.4747%. All four fall outside the symmetric ±0.5% gate (an increase also fails an unchanged-acceptance gate).

Whole-step code-edit +1.823241% faster, prose-en -3.628386% faster (regression). Even if the W16 timing gate were interpreted as the mean of the two workloads instead of requiring each, it fails: control = 16.389934714 ms, FP8 = 16.523560748 ms; (16.389934714 - 16.523560748) / 16.389934714 * 100 = **-0.815293%**. W4 is still being measured as requested.

### w4-control1 measured results

Needle PASS; minimum monitored free VRAM **8760 MiB**.

| workload | tokens/verify | t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | draft period us | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 3.908805 | 390.281383 | 9.611515 | 9.462440 | 259.665425 | 267.744385 | 12.736000 |
| prose-en | 2.673083 | 284.068732 | 9.541572 | 9.268707 | 257.025481 | 266.768066 | 12.775750 |
| prose-ja | 2.391045 | 256.638325 | — | — | — | — | — |
| agent-loop | 3.235076 | 339.978558 | — | — | — | — | — |

Timed repetitions (unrounded ratios retained in analysis.json):

| workload | tokens/verify per repeat | t/s per repeat |
|---|---|---|
| code-edit | 3.90244, 3.91517 | 388.87, 391.69 |
| prose-en | 2.70880, 2.63736 | 287.90, 280.24 |
| prose-ja | 2.42915, 2.35294 | 261.04, 252.24 |
| agent-loop | 3.36134, 3.10881 | 352.14, 327.81 |

### w4-fp8 measured results

Needle PASS; minimum monitored free VRAM **8900 MiB**.

| workload | tokens/verify | t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | draft period us | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 3.767086 | 382.008102 | 9.598929 | 9.419314 | 247.219804 | 261.728516 | 7.408250 |
| prose-en | 2.626625 | 281.196488 | 9.433796 | 9.154554 | 248.026200 | 260.168457 | 7.384000 |
| prose-ja | 2.369203 | 256.992069 | — | — | — | — | — |
| agent-loop | 3.174959 | 336.506064 | — | — | — | — | — |

Timed repetitions (unrounded ratios retained in analysis.json):

| workload | tokens/verify per repeat | t/s per repeat |
|---|---|---|
| code-edit | 3.90879, 3.62538 | 391.21, 372.81 |
| prose-en | 2.58065, 2.67261 | 276.67, 285.72 |
| prose-ja | 2.37154, 2.36686 | 258.16, 255.82 |
| agent-loop | 3.20856, 3.14136 | 339.12, 333.89 |

### Kernel mechanism and limited step budget

The server confirms one FP8 `[80,5,1]` entry kernel per forward (280 calls in each W16 20-step trace; 40 at W4), replacing the BF16 WMMA plus separate splitKreduce pair. W16 fc_hidden pair/FP8 medians versus the mean of both controls are 12.66425 → 7.328 us (code-edit) and 12.64775 → 7.264 us (prose-en), savings 42.14% / 42.57%. Multiplying only this measured component by 14 gives **74.7075 / 75.3725 us per step**, or **0.44167% / 0.47508%** of the respective control whole-step time. This arithmetic is component attribution, not an additive proof of wall speedup under overlapping kernels.

This is not simply a launch-bound no-op: the BF16 matrix occupies 13.1 MB and its GEMM takes ~11 us, consistent with substantial weight-streaming cost (~1.19 TB/s if read once), plus ~1.6 us reduction launch. FP8 halves weight bytes and removes that separate reduction launch; its measured ~7.3 us is clearly faster. No hardware memory-transaction counters were captured, so bandwidth attribution is an inference from sizes, timing, and the changed kernel sequence, not a direct roofline measurement. The key limitations are that fc_embedding is already absent from steady forwards under R3, and the remaining fc_hidden saving is only about 0.45% of a W16 step.

### W16 prose timing confound (CPU trace audit while waiting for final W4 control)

Target verify MoE GEMM1 medians across three traces: control1 **64.4645 us**, FP8 **64.8640 us**, control2 **48.9445 us**; GEMM2 **37.072 / 36.784 / 28.400 us**. These target kernels are outside the fc code change. The final control's much lower whole-step prose time therefore includes substantial change elsewhere in the target workload, not a regression of the new fc_hidden kernel (which is measured faster). This bounds the conclusion: the specified short-run gate fails, but these data do not establish that fc FP8 alone causes the observed whole-step regression. Acceptance/trajectory variation, expert routing and session conditions are possible contributors; none is isolated by this experiment. Raw diagnostic: `specs/e0/w16-prose-target-diagnostic.json`.

### w4-control2 measured results

Needle PASS; minimum monitored free VRAM **8760 MiB**.

| workload | tokens/verify | t/s | trimmed whole step ms | legacy trimmed ms | draft exclusive us/forward | draft period us | fc_hidden us |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 3.698032 | 375.905038 | 9.673426 | 9.521367 | 254.416799 | 268.096924 | 12.864000 |
| prose-en | 2.601563 | 277.216345 | 9.514675 | 9.271579 | 261.118601 | 266.184326 | 12.768000 |
| prose-ja | 2.480864 | 265.958083 | — | — | — | — | — |
| agent-loop | 3.200091 | 339.239952 | — | — | — | — | — |

Timed repetitions (unrounded ratios retained in analysis.json):

| workload | tokens/verify per repeat | t/s per repeat |
|---|---|---|
| code-edit | 3.68664, 3.70943 | 374.62, 377.19 |
| prose-en | 2.66075, 2.54237 | 283.94, 270.49 |
| prose-ja | 2.54237, 2.41935 | 272.69, 259.23 |
| agent-loop | 3.21716, 3.18302 | 341.11, 337.37 |

## Final E0 gate

**FAIL — do not enable in common exports**.

| width | workload | mean control acceptance | FP8 acceptance | change % | acceptance ±0.5% | step saving % | t/s change % |
|---|---|---:|---:|---:|---|---:|---:|
| w16 | code-edit | 10.550321 | 12.268789 | +16.288304 | FAIL | +1.823241 | +15.726411 |
| w16 | prose-en | 3.278033 | 3.212879 | -1.987601 | FAIL | -3.628386 | -2.450194 |
| w16 | prose-ja | 2.702647 | 2.569399 | -4.930287 | FAIL | — | -3.715714 |
| w16 | agent-loop | 5.154116 | 5.384748 | +4.474707 | FAIL | — | +5.009803 |
| w4 | code-edit | 3.803419 | 3.767086 | -0.955256 | FAIL | +0.451558 | -0.283249 |
| w4 | prose-en | 2.637323 | 2.626625 | -0.405630 | PASS | +0.989993 | +0.197386 |
| w4 | prose-ja | 2.435955 | 2.369203 | -2.740280 | FAIL | — | -1.647977 |
| w4 | agent-loop | 3.217584 | 3.174959 | -1.324752 | FAIL | — | -0.913753 |

W16 step arithmetic (each traced workload must reach 0.5%):

- code-edit: (16.914730900 − 16.606334559) / 16.914730900 × 100 = **+1.823241%**, PASS.
- prose-en: (15.865138528 − 16.440786937) / 15.865138528 × 100 = **-3.628386%**, FAIL.

Control-to-control drift (final / initial − 1):

| width | workload | acceptance drift % | whole-step drift % |
|---|---|---:|---:|
| w16 | code-edit | -15.248670 | +1.701616 |
| w16 | prose-en | -4.877128 | -7.498707 |
| w16 | prose-ja | -8.202080 | — |
| w16 | agent-loop | +0.147663 | — |
| w4 | code-edit | -5.392265 | +0.644136 |
| w4 | prose-en | -2.675553 | -0.281888 |
| w4 | prose-ja | +3.756448 | — |
| w4 | agent-loop | -1.081439 | — |

Acceptance variation includes changed draft numerics and ordinary greedy-trajectory/server-run variability; these short runs do not identify a unique causal component. Gates are applied exactly rather than excusing a failing cell as noise. Source, raw outputs, traces, and invalid initial control are retained; no cherry-picking of repetitions.

Launcher change: **none**. Keep SGLANG_MTP_FC_FP8 unset (default off), or use `export SGLANG_MTP_FC_FP8=0` to explicitly disable in an experimental overlay. Do not add a default-on common export.

## Final validation and resource ledger

CPU **6/6 PASS**; real-weight GPU **8 shape cases PASS** (M=1,4,16,64 for each fc layer), finite BF16 outputs, row scales, large-M fallback, CUDA graph replay and table destination equality checked. BF16-vs-FP8 numerical differences are listed above rather than called bit-exact. All 30 server traces contain 20 draft annotations / 19 complete draft-to-next-draft intervals and the expected 14 or 2 forwards per phase. All 60 timed workload requests and 6 needles completed.

| arm | start (JST, includes load) | end | min free MiB | needle |
|---|---|---|---:|---|
| w16-control1 | 05:00:16 | 05:03:56 | 7904 | PASS |
| w16-fp8 | 05:06:08 | 05:09:48 | 8044 | PASS |
| w16-control2 | 05:11:13 | 05:14:56 | 7865 | PASS |
| w4-control1 | 05:17:13 | 05:20:09 | 8760 | PASS |
| w4-fp8 | 05:24:48 | 05:27:45 | 8900 | PASS |
| w4-control2 | 05:38:11 | 05:41:07 | 8760 | PASS |

Every valid arm uses mem fraction 0.920, MAX_TOTAL_TOKENS=262144, SERVE_DISPLAY_HZ empty, a private launcher/cache snapshot and the E0 PYTHONPATH overlay. Min free memory across all valid arms: **7865 MiB** (continuous 2-second samples plus stage checks), comfortably above 4096 MiB. Needle input is **17307 actual prompt tokens** for the existing script's nominal 18500-token construction, depth 0.4; all six outputs exactly include `AURORA-CEDAR-7319`. The initial 2956-MiB partial arm is preserved as invalid and excluded for the documented VRAM violation; it is not replaced selectively based on speed or acceptance.

There were other GPU evaluations between arms because lock ownership was released per arm, as requested. No clock/display settings were changed. Reported run-to-run drift remains a limitation. No further GPU runs were made to chase a passing gate.

Final source checks: `git diff --check` PASS; branch remains `codex/e0-mtp-fc-fp8`, HEAD remains `7b4d539f9bd896265f498fabccc6466e45ffe818`; no commit or push. Production launcher pair and inspected MTP/FP8/GEMV files match all five initial SHA-256 fingerprints (`e0/production-unchanged.json`). No production venv edits or installations. Read-only host teardown audit found **no remaining E0 sglang processes**; other owners' processes were not terminated.

Implementation: `sglang-e0/python/sglang/srt/layers/mtp_fc.py` and `python/sglang/srt/models/qwen4_exp_mtp.py`. Reproduction helpers: `sglang-e0/bench/e0/test_cpu.py`, `gpu_test.sh`, `test_gpu.py`, `run_arm.py`, `run_all.sh`, `analyze.py`, `write_report.py`; traces, raw requests, GPU errors, metadata and logs are under `flash-next-bench/specs/e0/`. Existing output directories are deliberately not overwritten by the arm runner.

**Launcher decision: no change.** Leave the flag unset (the code defaults to off). An explicit experimental rollback is `export SGLANG_MTP_FC_FP8=0`. A default-on common export is not justified by this gate. The fc_hidden kernel is faster, but its small remaining step budget and the measured acceptance deviations do not meet the requested adoption criteria.
