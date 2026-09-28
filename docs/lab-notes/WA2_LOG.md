# WA2: three-width confidence adaptive speculation

Status: FINAL — Iteration 3 COMPLETE, both test-c arms FAIL the adoption gate. All 36 measured requests and 3 needle checks completed. No fourth config/code iteration was started or will be started. Production remains unchanged; retain w16_conf.json and defaults 9.74/0.70. Experimental worktree code and CPU tests are uncommitted.

## Scope and execution

- Production commit: `14d4c4c985a87f329f27af9342970b28b35a8ed4` (`codex/perf-v1`); harness: `9a26ebc060fbd4101c8506b6373c31ae0bfeff15` (`codex/harness-v1`).
- Read first: WIDTH_SWEEP_0907.md, C1_LOG.md, README-confidence.txt, w16_conf.json, production adaptive_confidence.py, C1 ab_run.sh and prof/validate.sh.
- No code changes needed; serve production code without a PYTHONPATH overlay. No WA2 worktree/branch created.
- Production launcher explicitly passes initial_steps=15. Both arms retain initial 15; explicit initial 3 also remains 3. The fallback for an unsupported initial value is 7 with three candidates; retain it as a neutral fallback because it is unused by this launcher. There is no production initial-3 behavior to preserve.
- Config w16_3_7_15.json changes only slot 1 candidate_steps to [3,7,15]. Slot 2 and all hysteresis settings remain byte-value equivalent.
- best_steps checks every candidate. From current S=7, down requires >1.12x incumbent value; up requires >1.12*1.30=1.456x. Reversal grace doubles; same-direction continuation resets to 40; warmup/interval/grace CPU checks passed.
- S=7 -> S=15 extrapolates eight tail positions using r=min(.999,P(a>=7)/P(a>=6)), times tail_bias=.75. S=3 counterfactual uses min(a,3). A synthetic sample [0,3,6,7,7] yields E15=5.176588935; stale S15 data is ignored. Finite/zero/saturated-tail cases checked.
- Known hypothesis: retained conservative upward margin may prevent W4->W8 transitions. CPU checks establish correctness, not workload optimality; do not retune during the registered A/B.

## Step-time fit

Equal-weight least squares over the 10 trimmed code-edit/prose-en measurements, S in {3,5,7,11,15}. W4b is a drift repeat and excluded. `step_ms = 7.942995689655 + 0.555366379310 * S`; RMSE 0.325801 ms. Test env rounded to A=7.942995690, B=0.555366379. Controls remain A=9.74, B=0.70.

| S (W=S+1) | fitted ms | code-edit observed | residual obs-fit | prose-en observed | residual obs-fit |
|---|---:|---:|---:|---:|---:|
| 3 (W4) | 9.609095 | 9.47 | -0.139095 | 9.16 | -0.449095 |
| 5 (W6) | 10.719828 | 11.18 | +0.460172 | 10.82 | +0.100172 |
| 7 (W8) | 11.830560 | 12.01 | +0.179440 | 11.49 | -0.340560 |
| 11 (W12) | 14.052026 | 14.63 | +0.577974 | 14.16 | +0.107974 |
| 15 (W16) | 16.273491 | 16.05 | -0.223491 | 16.00 | -0.273491 |

## Registered server protocol

- One `flock -w 28800 $HOME/.gpu.lock bash calib/wa2_ab_run.sh <timestamped-prefix>` holds the lock throughout control -> test -> control. C1 driver adapted only into new calib scripts: same launcher, fnbench and needle scripts, three repeats instead of two, robust owned-process-group cleanup and incremental reporting.
- fnbench workload order code-edit, prose-en, prose-ja, agent-loop, one warmup then 3 measured requests each; greedy; needle approx 18.5k/depth .4 once per arm. Raw outputs use unique `specs/wa2/<label>/runs.jsonl` paths instead of top-level runs/ to obey the authorized write scope.
- Every server uses unchanged production serve-fast.sh wa, v5 head, allocator default expandable_segments:True, mem fraction .925. No memory fractions raised. The original 4096 MiB startup abort was incorrect and is superseded by the supervisor clarification below: no low-memory startup abort; after readiness, <1024 MiB sustained for 10 s triggers cleanup.
- Sandbox nvidia-smi could not reach the driver; approved outside-sandbox read-only preflight succeeded: RTX PRO 6000, total 97887 MiB, 91265 MiB free, no compute processes. xrandr read succeeds; display is currently 3840x2160 at 160 Hz. SERVE_DISPLAY_HZ is explicitly empty for every arm to hold the existing desktop rate; no xrandr modification is requested.
- To enforce production immutability despite serve-local.sh hardcoding cache paths, bwrap mounts production (including venv/launchers) read-only and overlays only its .cache with a private copy under specs/wa2/runtime-cache. PYTHONDONTWRITEBYTECODE=1. Actual production files and caches are not written.
- SGLANG_ADAPTIVE_TRACE and SGLANG_ADAPTIVE_DEBUG=1 on every arm. Histogram counts verified decode steps, W=S+1, measured requests only. Exact request start/end wall clocks come from a thin wrapper around the unchanged fnbench client; warmups and needle excluded. fnbench record timestamp is REQUEST START, contrary to the old WS1 report parser assumption.
- Acceptance is mean over repeats of trace mean(1+accepted drafts), with SSE completion_tokens/chunks as a comparable cross-check. Trace emission totals must agree with completion usage within 32 tokens per request (prefill/tail/overlap); failures stop reporting rather than silently mislabeling boundaries.
- Gate against equal-weight pooled controls (n=6): agent-loop >=+5%, prose-en/prose-ja >=-2%, code-edit >=0%. End control drift reported separately. Recommendation only, never edit the launcher.

### Starting wa220260907-141943-ctl at 2026-09-07T14:20:03+09:00

ABORT wa220260907-141943-ctl: desktop reserve violated (2141 MiB free).

Run exit 1 at 2026-09-07T14:22:03+09:00; owned server process group cleaned up.

## Startup interruption (14:21:52 JST)

The first control was stopped by the VRAM monitor before readiness: free memory fell from 4543 to 2141 MiB during the target-to-draft loading transition. No fnbench requests or needle request were sent, so this is a failed startup, not a performance result. The first observed reserve breach was 2141 MiB; the watchdog reacted by terminating the owned process group. This was the original monitor treating a normal startup transient as a reserve breach; the supervisor subsequently clarified that the reserve is a steady-state observation. This aborted attempt contributes no benchmark results.

At 14:22:30, outside-sandbox nvidia-smi confirmed no compute processes / no sglang processes and 5955 MiB total used (desktop graphics). Firefox accounted for 2874 MiB, Xorg 880 MiB, and the other desktop applications for the remainder. The same-day WS1 control log also shows only 3.92 GiB available at target load completion, followed by memory reclamation during draft loading; serving-state free memory alone does not prove the startup reserve.

User was asked to free roughly 3 GiB or more of desktop GPU usage before a retry. No unrelated desktop process was killed. No memory fraction, allocator, model head, or reserve requirement was relaxed. A retry will use a fresh timestamped prefix and repeat the complete control -> test -> control sequence.

| arm | code-edit t/s | prose-en t/s | prose-ja t/s | agent-loop t/s | width histogram | needle |
|---|---|---|---|---|---|---|
| first control | Not measured | Not measured | Not measured | Not measured | No measured requests | Not run |
| test | Not started | Not started | Not started | Not started | Not available | Not run |
| drift control | Not started | Not started | Not started | Not started | Not available | Not run |

Current gate status: NOT EVALUATED. No serve-fast.sh change recommended without a completed passing A/B. Production launcher and policy SHA-256 match the pre-run snapshot.

## Historical handoff (superseded by supervisor clarification)

Blocked pending a reduction in desktop GPU memory use; the user-input request remains pending. CPU implementation/configuration/fit work is complete; GPU A/B is not complete. No automatic retry while the same reserve problem remains.

Resume with a fresh label using the existing driver under `flock -w 28800 $HOME/.gpu.lock`; retain all three arms and the 4096 MiB monitor. The initial attempt prefix is `wa220260907-141943`.

## Supervisor clarification and retry

The supervisor clarified that >=4 GB free is a STEADY-STATE observation after `ready to roll`, not a startup constraint. The production launcher's transient 2-4 GB free at the end of weight loading is expected. The first attempt was aborted by an incorrectly strict WA2 monitor, not evidence of an unhealthy production start. The previous request to close desktop applications and the blocked handoff are superseded; no user action is needed.

Retry policy: record VRAM during startup without low-memory abort; after readiness, abort the owned server group only when free VRAM remains below 1024 MiB for at least 10 consecutive seconds. Record ready, median and minimum steady free VRAM per arm, with startup and cleanup separated. All other experiment settings remain unchanged, including empty SERVE_DISPLAY_HZ, production mem fractions and allocator. Retry all three arms with a fresh timestamped prefix under the GPU lock.

### Starting wa220260907-142815-ctl at 2026-09-07T14:28:35+09:00

wa220260907-142815-ctl ready: steady-state free VRAM 3688 MiB at 2026-09-07T14:31:08+09:00.

## Arm wa220260907-142815-ctl

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 588.68 (499.91-652.51) | 11.280 / 11.172 | 0 (0.00%) | 0 (0.00%) | 652 (100.00%) |
| prose-en | 252.25 (248.04-256.75) | 2.535 / 2.530 | 1422 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 237.56 (231.59-241.71) | 2.325 / 2.340 | 1549 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 314.60 (303.68-320.74) | 3.466 / 3.452 | 937 (89.32%) | 0 (0.00%) | 112 (10.68%) |

Needle: `[wa220260907-142815-ctl] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 3688 MiB; minimum 2264 MiB (2.211 GiB); median 2264 MiB. Startup minimum 2139 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [17, 28, 16]; prose-en: [4, -1, -1]; prose-ja: [1, 1, -1]; agent-loop: [1, 2, 1].

Artifacts: `specs/wa2/wa220260907-142815-ctl/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa220260907-142815-test at 2026-09-07T14:32:57+09:00

wa220260907-142815-test ready: steady-state free VRAM 2450 MiB at 2026-09-07T14:35:13+09:00.

## Arm wa220260907-142815-test

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 463.61 (447.93-473.61) | 7.145 / 7.105 | 0 (0.00%) | 1011 (100.00%) | 0 (0.00%) |
| prose-en | 263.69 (259.99-270.72) | 2.622 / 2.615 | 1375 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 240.79 (229.20-253.76) | 2.395 / 2.398 | 1506 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 311.35 (307.88-316.69) | 3.119 / 3.104 | 1156 (100.00%) | 0 (0.00%) | 0 (0.00%) |

Needle: `[wa220260907-142815-test] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 2450 MiB; minimum 1024 MiB (1.000 GiB); median 1026 MiB. Startup minimum 2141 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [10, 6, 4]; prose-en: [4, -1, 1]; prose-ja: [2, -1, 0]; agent-loop: [2, 1, 2].

Artifacts: `specs/wa2/wa220260907-142815-test/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa220260907-142815-ctl2 at 2026-09-07T14:37:07+09:00

wa220260907-142815-ctl2 ready: steady-state free VRAM 3688 MiB at 2026-09-07T14:39:11+09:00.

## Arm wa220260907-142815-ctl2

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 535.61 (486.76-600.59) | 10.197 / 10.072 | 0 (0.00%) | 0 (0.00%) | 719 (100.00%) |
| prose-en | 264.55 (260.82-268.94) | 2.633 / 2.628 | 1368 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 237.52 (234.26-240.26) | 2.352 / 2.365 | 1530 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 299.21 (295.56-301.09) | 4.135 / 4.108 | 537 (59.27%) | 0 (0.00%) | 369 (40.73%) |

Needle: `[wa220260907-142815-ctl2] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 3688 MiB; minimum 1764 MiB (1.723 GiB); median 2264 MiB. Startup minimum 2141 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [24, 28, 19]; prose-en: [3, -1, -1]; prose-ja: [1, -1, -1]; agent-loop: [6, 1, 5].

Artifacts: `specs/wa2/wa220260907-142815-ctl2/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

## Final gate and drift

| workload | control pooled mean (n=6) | test mean (n=3) | test vs control | control2 vs control1 | gate | result |
|---|---:|---:|---:|---:|---:|---|
| code-edit | 562.14 | 463.61 | -17.53% | -9.01% | +0% | FAIL |
| prose-en | 258.40 | 263.69 | +2.04% | +4.88% | -2% | PASS |
| prose-ja | 237.54 | 240.79 | +1.37% | -0.02% | -2% | PASS |
| agent-loop | 306.91 | 311.35 | +1.45% | -4.89% | +5% | FAIL |

### Steady-state VRAM

| arm | ready free MiB | steady median MiB | steady minimum MiB | startup minimum MiB |
|---|---:|---:|---:|---:|
| wa220260907-142815-ctl | 3688 | 2264 | 2264 | 2139 |
| wa220260907-142815-test | 2450 | 1026 | 1024 | 2141 |
| wa220260907-142815-ctl2 | 3688 | 2264 | 1764 | 2141 |

The supervisor-requested sustained <1024 MiB / 10-second abort did not trigger in the completed sequence. The measured steady 4 GiB target was not met; report this separately from the throughput gates. SERVE_DISPLAY_HZ remained empty, and mem-fraction-static remained 0.925 for every arm.

Verdict: FAIL. Do not change serve-fast.sh; retain the production w16_conf.json and STEP_A/B defaults 9.74/0.70.

Both controls contribute equally (3 repeats each). These are descriptive small-sample gates, not a confidence-interval claim. All arms enable the same chain tracing/debug instrumentation; tracing synchronizes the chain staging event and can affect throughput, so absolute numbers describe the instrumented server. No launcher edits, commits, pushes, new branches, production source or venv changes were made.

Run exit 0 at 2026-09-07T14:41:02+09:00; owned server process group cleaned up.

## Interpretation and exact recommendation

The three-candidate implementation works mechanically; the retained two-width hysteresis does not deliver the intended three-width behavior. No CPU policy code fix was required. The test started at S=15 as configured, then switched 15 -> 7 during the code-edit warmup at 14:35:15 (batch 85): predicted throughput 397 -> 457 tok/s cleared the 12% down margin. It stayed at S=7 for all three measured code-edit requests. It switched 7 -> 3 during the prose-en warmup at 14:35:46 and stayed there through both prose workloads and agent-loop.

The code recovery failure is visible in the decision log: at 14:35:16, batch 125, predicted values were S7=596 and S15=731 tok/s (+22.7%), but the upward challenger must clear 1.12*1.30=1.456x, so it remained at S7. At 14:36:34 on agent-loop, the model predicted S3=360, S7=440 and S15=466 tok/s. W8's +22.2% predicted improvement still did not clear the same 45.6% threshold; the measured agent-loop histogram was 100% W4. These are policy-model predictions, not measured throughput.

The saturated, consistent same-state reference illustrates the asymmetry: with tail_bias=.75 and the fitted costs, S3 -> S7 predicts only 1.4199x and S7 -> S15 only 1.2698x even when the live chain fully accepts. Both are below 1.456x. This is an explanatory reference, not a new universal bound over mixtures with stale cross-state EMA history. The upward margin was deliberately left unchanged for this A/B, so no post-hoc policy tuning was mixed into the test arm.

Control2 vs control1 changes include run-to-run acceptance and policy-state variation, not just hardware drift: both controls stayed 100% W16 on measured code-edit, but trace acceptance fell from 11.280 to 10.197; agent-loop W16 occupancy rose from 10.68% to 40.73%. The -9.01% code control drift is material, and the +1.45% agent result is smaller than the -4.89% control drift. The gate nevertheless fails against either control separately, as quantified below.

- code-edit: test vs first control -21.25%; vs final control -13.44%.

- agent-loop: test vs first control -1.03%; vs final control +4.06%.

**Exact serve-fast.sh recommendation: no edit (zero-line diff).** Keep the wa arm's fallback config `$HOME/tools/flash-next-bench/adaptive/w16_conf.json`; do not add STEP_A/B exports. Retain the production module defaults A=9.74, B=0.70. The new config and fit remain experimental artifacts, not a production recommendation.

The additional test runtime state reduced median steady free VRAM by 1238 MiB versus either control (2264 -> 1026 MiB). Ready-state free VRAM was also below 4 GiB on every arm. This is an observed memory limitation of this instrumented configuration on the current desktop, independent of the performance-gate failure. No supervisor abort condition triggered in the successful sequence. Needle used the unchanged C1 `needle_test.py 18500 0.4 <label>` invocation once per arm; actual reported prompt length was 17307 tokens on each arm, with the exact passphrase returned.

Final cleanup verification: no compute processes and no sglang processes remained; `flock -n ~/.gpu.lock true` succeeded. nvidia-smi reported 5955 MiB used and 91261 MiB free. serve-fast.sh, serve-local.sh and adaptive_confidence.py SHA-256 matched the pre-run snapshot; production git status matched the starting snapshot. Production was mounted read-only, with only a private runtime-cache mount under specs/wa2; no production source, launcher or venv write was made. No commits, pushes or new branches.

## Iteration 2

Status: configuration selected and CPU decision-snapshot checks passed; GPU sequence pending. Configuration-only policy iteration; no production code change. New config `adaptive/w16_3_7_15_b.json` changes slot 1 switch_margin 0.12 -> 0.08 and up_margin 0.30 -> 0.05 relative to WA2's three-width config. Candidate steps stay [3,7,15]. Slot 2 is unchanged from production. Tail bias, all EMA settings, warmup, update interval and grace/backoff remain unchanged.

<!-- WA2b configuration selection rationale, registered before the first GPU start:
Use switch_margin=0.08 and up_margin=0.05: effective upward threshold
(1.08 * 1.05) = 1.134, or +13.4%, compared with WA2's +45.6%.
The WA2 test log at 14:35:16 batch125 predicts code S7=596, S15=731,
a +22.65% gain; this now promotes to S15. At 14:36:22 batch5205,
agent-loop predicts S3=324, S7=375, S15=361; S7/S3=1.1574,
so the first eligible upward candidate becomes S7 rather than S15.
Prose upward W8/W4 maxima are 1.0963 (en) and 1.1103 (ja), below 1.134.
The logged snapshot screen holds every included prose state at S3.
Prefer an 8% down margin to 5% to retain a productive incumbent against
small exact-value fluctuations; with up_margin=5% it gives the same
13.4% total up threshold as switch_margin=5%, up_margin=8%.
Grace remains 40 batches, backoff 2: WA2 had only two switches, no flapping.
This screen evaluates logged state snapshots, not a counterfactual trajectory.
New transitions will change statistics and grace; the GPU sequence decides.
-->

Selection evidence from `calib/wa2b_select.py` and `specs/wa2/wa2b-selection.txt`: use the existing production best_steps implementation with new margins on rounded prediction snapshots. Log timestamps have 1-second precision; use second midpoints and exclude half a second at each request boundary. Include warmups and measured requests for this decision screen; GPU result histograms still exclude warmups.

| workload/current | snapshots | relevant predicted-value ratio | choices under new margins |
|---|---:|---|---|
| code-edit / S7 | 106 | S15/S7 median 1.2003; first promotion 731/596=1.2265 | S15 71; S7 35 |
| prose-en / S3 | 121 | max S7/S3 1.0963; max S15/S3 1.0633 | S3 121 |
| prose-ja / S3 | 161 | max S7/S3 1.1103; max S15/S3 1.0554 | S3 161 |
| agent-loop / S3 | 119 | S7/S3 p90 1.1480, max 1.2222; first promotion 375/324=1.1574 | S3 97; S7 18; S15 4 |

The four agent snapshots that prefer S15 are an explicit residual risk: a single up_margin cannot rank two upward candidates differently. The first eligible promotion does prefer S7, after which S15 must beat the S7 incumbent by 13.4%. No other tuning is applied before this registered A/B.

Protocol: `calib/wa2b_ab_run.sh` reuses the validated WA2/C1 serving flow, with test-b1 -> production control -> test-b2. Each uses fnbench code-edit, prose-en, prose-ja, agent-loop, three greedy repeats, one warmup per workload, and one unchanged needle_test.py 18500 0.4 invocation. Test arms use A=7.942995690, B=0.555366379; control uses production A=9.74, B=0.70 and w16_conf.json. A single `flock -w 28800 $HOME/.gpu.lock` covers the entire sequence. All labels and artifact directories are fresh. SERVE_DISPLAY_HZ remains empty, mem fraction .925, v5 head and allocator defaults unchanged. Production mounted read-only, same private runtime cache. Startup is monitored without low-VRAM abort; after ready, abort only below 1024 MiB for at least 10 seconds. Record steady VRAM separately.

Gate: each test-b arm must independently satisfy agent-loop >=+5%, prose-en/prose-ja >=-2%, code-edit >=0% against the intervening control mean. Report test-b2 vs test-b1 repeat drift; do not pool away a failing arm. No commits or launcher changes.

#### Starting wa2b20260907-144813-b1 at 2026-09-07T14:48:33+09:00

wa2b20260907-144813-b1 ready: steady-state free VRAM 2434 MiB at 2026-09-07T14:50:41+09:00.

ABORT wa2b20260907-144813-b1: steady VRAM below 1024 MiB for >=10 seconds (1014 MiB free).

Run exit 2 at 2026-09-07T14:51:00+09:00; owned server process group cleaned up.

### Iteration 2 operational retry: common memory fraction 0.920

The first b1 attempt `wa2b20260907-144813-b1` reached readiness at 14:50:41, then the measured free VRAM remained at 1014 MiB for 10 seconds. The specified steady-state monitor terminated the owned group at 14:50:52. fnbench was still in code-edit warmup; there are no measured benchmark results to include. This is a memory-triggered aborted attempt, not a throughput failure. The trace and logs are retained.

The previous complete WA2 three-width arm reached a minimum of exactly 1024 MiB. A small desktop-use difference thus removes its entire safety margin. To complete the authorized GPU experiment without changing the monitor or closing applications, reduce WA_MEM_FRACTION from 0.925 to 0.920 identically for test-b1, control, and test-b2, and restart the entire sequence under a fresh prefix. This reduces rather than raises the memory fraction, as required by the original constraint. It changes common memory-pool sizing, not the policy or kernels. All policy parameters, STEP_A/B per-arm values, v5 head, allocator and display settings remain as registered.

The control still uses production code, w16_conf.json and A/B defaults 9.74/0.70, but now at the same 0.920 memory fraction as both tests. This operational deviation must accompany the final results and any recommendation; this sequence does not validate deployment of the three-width config at the original 0.925 memory fraction. No launcher edit is made. VRAM abort remains <1024 MiB for 10 consecutive seconds after readiness; no startup abort.

#### Starting wa2b20260907-145212-b1 at 2026-09-07T14:52:32+09:00

wa2b20260907-145212-b1 ready: steady-state free VRAM 2834 MiB at 2026-09-07T14:54:44+09:00.

### Arm wa2b20260907-145212-b1

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 540.69 (498.07-624.97) | 9.924 / 9.846 | 0 (0.00%) | 160 (21.42%) | 587 (78.58%) |
| prose-en | 234.05 (215.22-247.68) | 2.721 / 2.714 | 883 (66.24%) | 370 (27.76%) | 80 (6.00%) |
| prose-ja | 236.94 (230.52-242.01) | 2.339 / 2.358 | 1542 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 342.20 (328.67-350.85) | 5.004 / 4.984 | 0 (0.00%) | 721 (100.00%) | 0 (0.00%) |

Needle: `[wa2b20260907-145212-b1] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 2834 MiB; minimum 688 MiB (0.672 GiB); median 1410 MiB. Startup minimum 2123 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [19, 20, 18]; prose-en: [4, 0, 1]; prose-ja: [3, -1, 2]; agent-loop: [2, 2, 0].

Artifacts: `specs/wa2/wa2b20260907-145212-b1/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

#### Starting wa2b20260907-145212-ctl at 2026-09-07T14:56:35+09:00

wa2b20260907-145212-ctl ready: steady-state free VRAM 4072 MiB at 2026-09-07T14:58:43+09:00.

### Arm wa2b20260907-145212-ctl

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 550.83 (490.23-654.62) | 10.404 / 10.316 | 0 (0.00%) | 0 (0.00%) | 711 (100.00%) |
| prose-en | 261.71 (257.75-264.42) | 2.609 / 2.600 | 1382 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 234.58 (228.21-244.51) | 2.306 / 2.313 | 1562 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 325.65 (289.73-362.57) | 4.744 / 4.715 | 437 (53.55%) | 0 (0.00%) | 379 (46.45%) |

Needle: `[wa2b20260907-145212-ctl] prompt_tokens=17307 completion=12 time=1.5s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 4072 MiB; minimum 1988 MiB (1.941 GiB); median 2648 MiB. Startup minimum 2106 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [20, 17, 16]; prose-en: [5, -1, 1]; prose-ja: [-1, 1, -1]; agent-loop: [7, 0, 9].

Artifacts: `specs/wa2/wa2b20260907-145212-ctl/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

#### Starting wa2b20260907-145212-b2 at 2026-09-07T15:00:34+09:00

wa2b20260907-145212-b2 ready: steady-state free VRAM 2834 MiB at 2026-09-07T15:02:42+09:00.

### Arm wa2b20260907-145212-b2

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 463.62 (428.05-498.41) | 8.139 / 8.062 | 0 (0.00%) | 395 (44.04%) | 502 (55.96%) |
| prose-en | 234.94 (222.47-255.56) | 2.920 / 2.910 | 556 (44.59%) | 691 (55.41%) | 0 (0.00%) |
| prose-ja | 239.52 (230.22-246.74) | 2.366 / 2.374 | 1523 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 337.75 (317.02-372.33) | 5.014 / 4.966 | 0 (0.00%) | 688 (94.51%) | 40 (5.49%) |

Needle: `[wa2b20260907-145212-b2] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 2834 MiB; minimum 688 MiB (0.672 GiB); median 1410 MiB. Startup minimum 2125 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [21, 15, 17]; prose-en: [7, 0, -1]; prose-ja: [-1, 1, 1]; agent-loop: [4, 7, 18].

Artifacts: `specs/wa2/wa2b20260907-145212-b2/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Iteration 2 gate and repeat drift

| workload | control mean t/s (n=3) | test-b1 t/s | b1 vs control | b1 gate | test-b2 t/s | b2 vs control | b2 gate | b2 vs b1 | required |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| code-edit | 550.83 | 540.69 | -1.84% | FAIL | 463.62 | -15.83% | FAIL | -14.25% | +0% |
| prose-en | 261.71 | 234.05 | -10.57% | FAIL | 234.94 | -10.23% | FAIL | +0.38% | -2% |
| prose-ja | 234.58 | 236.94 | +1.01% | PASS | 239.52 | +2.11% | PASS | +1.09% | -2% |
| agent-loop | 325.65 | 342.20 | +5.08% | PASS | 337.75 | +3.71% | FAIL | -1.30% | +5% |

### Iteration 2 steady-state VRAM

| arm | ready free MiB | steady median MiB | steady minimum MiB | startup minimum MiB |
|---|---:|---:|---:|---:|
| wa2b20260907-145212-b1 | 2834 | 1410 | 688 | 2123 |
| wa2b20260907-145212-ctl | 4072 | 2648 | 1988 | 2106 |
| wa2b20260907-145212-b2 | 2834 | 1410 | 688 | 2125 |

Gate verdict: test-b1 FAIL; test-b2 FAIL; combined FAIL. Each test arm must independently pass all four gates versus the intervening production control; test means are not pooled to hide a failing arm.

Exact launcher recommendation: no edit. Keep production w16_conf.json and default STEP_A/B=9.74/0.70. Decision-log interpretation follows after analysis.

All three needle invocations passed. All arms used the common reduced mem-fraction-static=0.920 after the initial 0.925 attempt hit the sustained low-VRAM abort. Empty SERVE_DISPLAY_HZ, v5 head, expandable_segments:True, and trace/debug instrumentation were unchanged. Startup low-VRAM values are recorded without abort; after ready the abort condition is <1024 MiB continuously for at least 10 seconds. Tables report observed steady VRAM separately from throughput gates. Comparison is against a production-policy control at the same 0.920 memory fraction; it does not validate serving the new config with the launcher-default 0.925 fraction.

Run exit 0 at 2026-09-07T15:04:37+09:00; owned server process group cleaned up.

### Iteration 2 decision-log diagnosis

The upward retune did what the CPU screen predicted: both tests now used W16 again on code-edit and W8 predominantly on agent-loop. It did not preserve prose-en or code performance. Both arms fail the full gate; agent-loop only passes in b1 (+5.08%), narrowly above the +5% threshold, and misses in b2 (+3.71%). The code difference between tests is large (-14.25% b2 vs b1), corresponding to W8 occupancy rising from 21.42% to 44.04%; it is not safely attributable to hardware drift alone. The single control's code range is 490.23-654.62 t/s and agent range 289.73-362.57 t/s, so these n=3 comparisons are descriptive gates, not statistical significance claims.

`calib/wa2b_diagnose.py` reconstructs grace from production's direction/backoff rules and each logged switch, and records the next eligible decision in each arm's diagnostics.json. Each test made 9 switches over warmups and measured workloads. Workload attribution of a second-resolution switch timestamp is approximate; the measured width histograms use exact request intervals and the millisecond trace timestamps.

| evidence | test-b1 | test-b2 |
|---|---|---|
| code reversals | 15->7->15->7->15 | 15->7->15->7->15->7 |
| reversal grace growth | 40,80,160,320; next downshift gets 640 | 40,80,160,320,640 |
| long S7 hold | batch1045 -> next decision1685 (14:55:10 -> 14:55:19), spanning prose warmup and measured prose | batch995 -> next decision1635 (15:03:06 -> 15:03:16), spanning late code and early prose |
| measured prose-en wide occupancy | W8 27.76%, W16 6.00% | W8 55.41%, W16 0% |
| later false upward excursion | prose-en batch1745: S3->S15; S15->S3 at1825 | agent-loop batch5655: S7->S15; S15->S7 at5695 |

The grace ceiling is not the entire explanation. In b2, after the 640-batch grace expired at batch1635, the policy still predicted S7=343 versus S3=316 tok/s and retained S7. At batch1705 it predicted S3=224 versus S7=215 (+4.19%), but the 8% incumbent margin blocked the downshift. At batch2285 it predicted S3=245 versus S7=229 (+6.99%) and still held S7. It finally moved down at batch2295 when 216/196=1.102 cleared the 8% margin. A grace-only change removes forced waiting but cannot guarantee the correct subsequent argmax.

The lower up_margin also exposes the known optimistic tail hazard. In b1 prose-en at batch1745, predicted S15=375 versus S3=301 (+24.58%) triggered a false long-chain excursion. In b2 agent-loop, S15=607 versus S7=533 (+13.88%) just cleared the 13.4% effective threshold; after 40 batches the measured long-state prediction collapsed to S15=401 versus S7=459, reversing immediately. This last excursion occupies 5.49% of measured b2 agent verify steps and is consistent with part of the lost agent benefit, but its exact causal throughput cost cannot be isolated from this run.

### Is a third configuration-only iteration plausible?

**Yes, one bounded config-only iteration is still justified, specifically to cap reversal grace; the evidence does not yet require a production code change.** A concrete first candidate is `max_grace_batches=160` while retaining the 40-batch base grace and the current margins, so a short code reversal cannot impose a 640-2000-batch hold on the following workload. Both b arms supply direct evidence for this change, unlike the no-flapping WA2 trace available before iteration 2. This candidate has NOT been created, applied, or GPU-tested; no passing result is claimed.

However, another margin-only adjustment is not a demonstrated solution. Increasing the global switch_margin to preserve code's W16 also makes prose's S7->S3 exit harder; reducing it helps prose but increases the code reversals that created the long grace. Raising the global up_margin suppresses the agent S7->S15 error but can delay legitimate code recovery: b2's code S7->S15 at batch285 had only 666/584=1.1404 predicted gain, very close to the false agent gain 607/533=1.1388. Rounded snapshot ratios cannot robustly separate those cases with one scalar.

If a capped-grace run still retains W8 on prose or makes these false W16 excursions, the next substantive change should be in policy code: transition-specific hysteresis/grace and/or better calibration/freshness handling of the upward estimate and state cost model. Such changes must operate on observed policy statistics, not hardcoded workload names. Current logs support testing the grace cap first; they do not establish that config tuning can satisfy all four gates reliably.

### Iteration 2 recommendation and completion checks

**Exact serve-fast.sh recommendation: no edit (zero-line diff).** Neither b arm passes all four gates. Keep the production w16_conf.json fallback and module defaults STEP_A=9.74, STEP_B=0.70. Do not add the b config, refitted model coefficients, or the experiment's lower memory fraction to production. The b configuration remains an experiment.

The complete sequence used the same reduced WA_MEM_FRACTION=0.920 in all three arms; the aborted initial 0.925 b1 warmup is excluded. Two-second VRAM monitoring recorded a single below-1024 sample in each completed b arm (688 MiB at 14:56:06 and 15:04:08, during the needle window), with no sustained 10-second violation. Steady medians were 1410 / 2648 / 1410 MiB for b1/control/b2; the original 4 GiB steady target was not achieved. This operational constraint is separate from the failed performance gate. The original user-authorized prohibition on raising memory fractions was respected, and the monitor was not weakened.

All 36 measured records are present and unique by workload/repeat; each arm has 3 repetitions of all four workloads. All three unchanged needle_test.py 18500 0.4 invocations returned PASS, actual prompt length 17307. Trace emission totals matched completion usage within the same per-request tolerance used in iteration 1. Exact request timing, trace, debug/server log, memory samples, needle output, JSONL and summary/diagnostics JSON are retained under specs/wa2/wa2b20260907-145212-{b1,ctl,b2}/.

Final cleanup: no compute or sglang processes remained, and nonblocking acquisition confirmed the GPU lock was released. nvidia-smi reported 5971 MiB used / 91246 MiB free. SHA-256 for production serve-fast.sh, serve-local.sh and adaptive_confidence.py matched the original snapshot; production git status remained unchanged. Slot 2 was unchanged; slot 1 differs from WA2's three-width config only in switch_margin and up_margin. Production code, launcher and venv were not modified. No worktree, branch, commit or push was created.

## Iteration 3

Status: COMPLETE. Final sequence wa320260907-152122 ran on 2026-09-07, approximately 15:21-15:34 JST; both test-c arms failed the gate. Worktree `$HOME/tools/sglang-wa2`, branch `codex/wa2-three-widths`, created exactly from `codex/perf-v1` at `14d4c4c985a87f329f27af9342970b28b35a8ed4`. Production source, launcher and venv are unchanged; no commits. This is the LAST config/code iteration, including if the performance gate fails.

### Opt-in code and CPU validation

Only adaptive_confidence.py is changed in the worktree, with a new CPU unit test alongside the existing adaptive tests. `max_grace_batches` already exists in production and caps reversal backoff; reuse it rather than duplicate it or alter its default 2000. New tests cover its cap and the decision hold/interval, plus continuation resetting to base grace.

`down_margin` is optional: a scalar applies to every downshift; a mapping by destination S allows W16->W8 and W8->W4 to be tuned independently (e.g. {"7":0.15,"3":0.03}). Unspecified destinations retain legacy switch_margin. This mapping form is used because a single scalar cannot independently control those two downward transitions. Downward challenger score is multiplied by (1+switch_margin)/(1+down_margin), keeping the incumbent's existing score and making the effective downward threshold precisely 1+down_margin. Upward threshold remains (1+switch_margin)*(1+up_margin). When the key is absent, the old score arithmetic is executed unchanged.

`adjacent_only_promotion` is an optional boolean, default false. When true, consider only the next larger configured S for promotion (S3->S7 first, then S7->S15); any downward candidate remains eligible. Justification: b1's prose S3->S15 jump at batch1745 bypassed the middle candidate even though S7's +12.29% forecast was weak. No new environment flag and no hardcoded workload names are added.

CPU: 54 adaptive unit tests PASS (10 new confidence tests plus existing sizing/switch/EMA tests). Exact legacy score comparisons cover two/three/single candidates, and a deterministic trajectory guard covers update/EMA/grace state. Separately, calib/wa3_select.py imports the actual production and worktree modules: with production w16_conf.json, each slot passed 3000 updates with exact equality of decisions, scores and all legacy state fields. The new implementation's overlay import path was verified. CPU-only import emitted the existing unavailable-NVML/AWQ warnings in the sandbox; no GPU execution was required for these tests. Evidence: specs/wa2/wa3-cpu-tests.txt and wa3-selection.txt.

### Registered final configuration and prediction evidence

New config: adaptive/w16_3_7_15_c.json. Slot 1 candidates [3,7,15]; switch_margin=0.10, up_margin=0.04 (effective up=14.4%); down_margin={"3":0.03,"7":0.15}; max_grace_batches=80; adjacent_only_promotion=true. Keep base grace=40, grace_backoff=2, tail_bias=.75, all EMA/warmup/update settings unchanged. Slot 2 remains byte-value equivalent to production.

<!-- WA3 selection rationale registered before GPU start:
Use a 15% threshold for demotion to S7 to block logged code losses that
triggered at +9.1..12.9%; use 3% for demotion to S3 to pass prose gains
of +4.19% and +6.99% that WA2b blocked. Keep up at 14.4%, above the
false agent S7->S15 gain +13.88%, but below strong code recovery +18.23%
and agent S3->S7 gain +16.11%. The marginal code recovery +14.04% is
intentionally delayed; no claim that every code upward snapshot passes.
Adjacent promotion blocks the prose S3->S15 jump (+24.58%): its S7
alternative was only +12.29%, so the new policy stays at S3.
Cap backed-off grace at 80: retains the tested 40-batch base/cold hold
and one backoff, without letting 640-batch holds cross workloads.
-->

| iteration-2 logged case | predicted ratio | WA3 decision |
|---|---:|---|
| b1 code 15->7, 14:54:46 | 431/392 = 1.0995 | block under 1.15 down-to-7 threshold |
| b1 code 15->7, 14:54:56 | 489/437 = 1.1190 | block |
| b2 code 15->7, 15:02:44 | 368/326 = 1.1288 | block |
| b2 code 7->15, 15:02:47 | 666/584 = 1.1404 | block/delay under 1.144 up threshold |
| b2 code 7->15, 15:02:58 | 746/631 = 1.1823 | allow |
| b2 prose 7->3, batch1705 | 224/215 = 1.0419 | allow under 1.03 down-to-3 threshold |
| b2 prose 7->3, batch2285 | 245/229 = 1.0699 | allow |
| b1 prose false 3->15, batch1745 | S15/S3=375/301=1.2458; S7/S3=338/301=1.1229 | exclude S15; S7 fails 1.144, hold S3 |
| b2 agent 3->7, 15:03:52 | 382/329 = 1.1611 | allow |
| b2 agent false 7->15, 15:04:03 | 607/533 = 1.1388 | block |

These are rounded logged prediction snapshots checked through the real worktree best_steps method, not a simulated future trajectory. Pairwise cases use zero for irrelevant unreported candidate scores and validate only the named transition threshold. Grace changes alter the future state distribution; the GPU result remains decisive.

### Final server protocol

calib/wa3_ab_run.sh reuses the WA2b/C1 flow: test-c1 -> control -> test-c2, each with 4 workloads x 3 greedy repeats, the same warmups and needle_test.py 18500 0.4 once. One flock -w 28800 ~/.gpu.lock covers the whole sequence. Every arm uses PYTHONPATH=$HOME/tools/sglang-wa2/python with the unchanged production serve-fast.sh wa and production venv, exactly the C1 overlay pattern. Verify code-origin.txt in each arm; mount both production and the overlay worktree read-only for server execution, with the same private runtime cache.

All arms use WA_MEM_FRACTION=0.920, empty SERVE_DISPLAY_HZ, unchanged v5 head/allocator and identical chain trace/debug. Tests use config c and STEP_A/B=7.942995690/0.555366379; control uses production w16_conf.json and STEP_A/B=9.74/0.70. The new keys are absent from control; default parity was verified on CPU. Startup VRAM is recorded without abort; after ready, abort only for <1024 MiB continuously for 10 seconds. Record ready/steady/startup values separately. Always terminate the owned server process group on completion or failure.

Gate each test-c arm independently against the intervening control: agent-loop >=+5%, prose-en/prose-ja >=-2%, code-edit >=0%. Both test-c arms must pass all gates. Append results as each lands. If the gate fails, write the mechanism analysis and stop; no fourth iteration.

#### Starting wa320260907-152122-c1 at 2026-09-07T15:21:42+09:00

wa320260907-152122-c1 ready: steady-state free VRAM 2834 MiB at 2026-09-07T15:23:58+09:00.

### Arm wa320260907-152122-c1

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 623.66 (610.23-648.98) | 11.936 / 11.816 | 0 (0.00%) | 0 (0.00%) | 610 (100.00%) |
| prose-en | 264.13 (245.77-278.02) | 2.695 / 2.688 | 1258 (94.02%) | 80 (5.98%) | 0 (0.00%) |
| prose-ja | 233.98 (230.94-235.60) | 2.311 / 2.320 | 1560 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 307.77 (299.00-324.02) | 4.457 / 4.419 | 0 (0.00%) | 812 (100.00%) | 0 (0.00%) |

Needle: `[wa320260907-152122-c1] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 2834 MiB; minimum 908 MiB (0.887 GiB); median 1410 MiB. Startup minimum 2125 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [21, 22, 30]; prose-en: [4, 0, 0]; prose-ja: [4, -1, 1]; agent-loop: [10, 1, 3].

Artifacts: `specs/wa2/wa320260907-152122-c1/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

#### Starting wa320260907-152122-ctl at 2026-09-07T15:25:47+09:00

wa320260907-152122-ctl ready: steady-state free VRAM 4072 MiB at 2026-09-07T15:27:56+09:00.

### Arm wa320260907-152122-ctl

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 616.93 (610.84-621.33) | 11.712 / 11.596 | 0 (0.00%) | 0 (0.00%) | 621 (100.00%) |
| prose-en | 256.85 (250.85-262.39) | 2.553 / 2.549 | 1411 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 241.06 (231.70-246.65) | 2.375 / 2.371 | 1520 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 324.96 (302.03-340.72) | 4.209 / 4.186 | 720 (76.60%) | 0 (0.00%) | 220 (23.40%) |

Needle: `[wa320260907-152122-ctl] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 4072 MiB; minimum 1928 MiB (1.883 GiB); median 2648 MiB. Startup minimum 2125 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [21, 26, 25]; prose-en: [1, -1, 1]; prose-ja: [6, 1, 0]; agent-loop: [2, 4, 1].

Artifacts: `specs/wa2/wa320260907-152122-ctl/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

#### Starting wa320260907-152122-c2 at 2026-09-07T15:29:44+09:00

wa320260907-152122-c2 ready: steady-state free VRAM 2834 MiB at 2026-09-07T15:32:01+09:00.

### Arm wa320260907-152122-c2

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 522.63 (488.17-587.93) | 9.815 / 9.714 | 0 (0.00%) | 0 (0.00%) | 746 (100.00%) |
| prose-en | 259.03 (228.54-279.13) | 2.789 / 2.781 | 1063 (82.21%) | 230 (17.79%) | 0 (0.00%) |
| prose-ja | 247.53 (242.64-250.54) | 2.435 / 2.443 | 1480 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 322.86 (305.38-341.22) | 4.712 / 4.694 | 0 (0.00%) | 767 (100.00%) | 0 (0.00%) |

Needle: `[wa320260907-152122-c2] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 2834 MiB; minimum 848 MiB (0.828 GiB); median 1410 MiB. Startup minimum 2125 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: BELOW 4 GiB (reported; supervisor abort threshold is <1024 MiB for 10 s).

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [21, 16, 28]; prose-en: [3, 1, 1]; prose-ja: [2, 1, 0]; agent-loop: [5, -1, 0].

Artifacts: `specs/wa2/wa320260907-152122-c2/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Iteration 3 gate and repeat drift

| workload | control mean t/s (n=3) | test-c1 t/s | c1 vs control | c1 gate | test-c2 t/s | c2 vs control | c2 gate | c2 vs c1 | required |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| code-edit | 616.93 | 623.66 | +1.09% | PASS | 522.63 | -15.28% | FAIL | -16.20% | +0% |
| prose-en | 256.85 | 264.13 | +2.84% | PASS | 259.03 | +0.85% | PASS | -1.93% | -2% |
| prose-ja | 241.06 | 233.98 | -2.94% | FAIL | 247.53 | +2.68% | PASS | +5.79% | -2% |
| agent-loop | 324.96 | 307.77 | -5.29% | FAIL | 322.86 | -0.65% | FAIL | +4.90% | +5% |

### Iteration 3 steady-state VRAM

| arm | ready free MiB | steady median MiB | steady minimum MiB | startup minimum MiB |
|---|---:|---:|---:|---:|
| wa320260907-152122-c1 | 2834 | 1410 | 908 | 2125 |
| wa320260907-152122-ctl | 4072 | 2648 | 1928 | 2125 |
| wa320260907-152122-c2 | 2834 | 1410 | 848 | 2125 |

Gate verdict: test-c1 FAIL; test-c2 FAIL; combined FAIL. Each test arm must independently pass all four gates versus the intervening production control; test means are not pooled to hide a failing arm.

Exact launcher recommendation: no edit. Keep production w16_conf.json and default STEP_A/B=9.74/0.70. Decision-log interpretation follows after analysis. This is the last iteration; no fourth experiment will be started.

All three needle invocations passed. All arms used the same sglang-wa2 PYTHONPATH overlay; the control config omits the new keys and its default CPU trajectory was verified exactly against production. All arms used the common reduced mem-fraction-static=0.920 as authorized for iteration 3. Empty SERVE_DISPLAY_HZ, v5 head, expandable_segments:True, and trace/debug instrumentation were unchanged. Startup low-VRAM values are recorded without abort; after ready the abort condition is <1024 MiB continuously for at least 10 seconds. Tables report observed steady VRAM separately from throughput gates. Comparison is against a production-policy control at the same 0.920 memory fraction; it does not validate serving the new config with the launcher-default 0.925 fraction.

Run exit 0 at 2026-09-07T15:33:50+09:00; owned server process group cleaned up.

### Iteration 3 final mechanism analysis

The new controls corrected the main state-selection failures from iteration 2. Both test-c arms spent 100% of measured code-edit at W16 and 100% of measured agent-loop at W8; both switched directly from S15 to S3 during prose-en warmup. The backed-off grace never exceeded 80 in either test, and no 640-batch hold crossed workloads. Each test had six total switches across warmups and measured requests, versus nine in each WA2b test. Adjacent promotion prevented direct S3->S15 excursions; neither test had any measured W16 prose or agent steps.

This did not satisfy the performance gate. The remaining findings are distinct:

1. **W8 selection did not deliver the fixed-frontier agent gain in this instrumented adaptive server.** Agent means were 307.77 and 322.86 t/s, versus control 324.96 (-5.29% and -0.65%). Both tests were 100% W8, so this failure is no longer explained by refusing to promote or by time spent at W16. Acceptance at W8 was 4.457 / 4.712 tokens per verify, versus the older WS1 W8 agent figure 4.77. The first arm in particular did not reproduce that older acceptance level.

2. **The fitted trimmed-time model understates the current effective cost of W8 relative to W4.** Below, effective step ms = mean over repeats of 1000 * traced acceptance / client decode t/s. This is an end-to-end estimate, not a CUDA profiler measurement; small first/tail token discrepancies remain. For the control's pure W4 agent request, the estimate is 10.00 ms; for test agent W8 it is 14.48 / 14.59 ms. The fit predicts only 11.83/9.61=1.231x W8/W4 cost; the observed same-workload estimates imply about 1.448x/1.459x. Thus the value model overstates the throughput benefit of W8. These measurements do not isolate contributions from chain tracing/synchronization, adaptive runtime state or fixed-profile differences; no additional untraced/profiling run was made.

3. **Prose excursions were bounded but not eliminated.** c1 spent 80 measured prose-en steps at S7 (5.98%); c2 spent 230 (17.79%). Logs show false S3->S7 promotions and subsequent S7->S3 reversals rather than the former direct S3->S15 jumps. c2's first S7 hold lasted 150 batches even with cap80: the cap bounds forced waiting, not the number of batches until the estimator later prefers a downshift. Both prose-en means nevertheless passed (-2%) against this control. Prose-ja was 100% W4 in all arms; c1 failed at -2.94%, c2 passed at +2.68%. This difference cannot be attributed to width occupancy; trace acceptance changed 2.311 / 2.435 versus control 2.375.

4. **The code result is not repeatably non-regressing despite correct width occupancy.** c1 passed at +1.09%, c2 failed at -15.28%. All three arms were 100% W16 on measured code, and c2's effective step estimate was slightly faster than control, not slower. Its trace acceptance fell to 9.815 versus control 11.712 (c1 11.936), accounting arithmetically for the throughput difference. The origin of this acceptance variation is not isolated here; it should not be relabeled as a demonstrated W8 dwell or grace regression. The large c2-vs-c1 code change (-16.20%) makes the small-sample repeatability limit explicit.

These are observed descriptive gates, not significance tests. With either test failing one required workload, adoption fails; averaging the two tests does not rescue the result.

#### Effective step-time cross-check

| arm / workload / measured width | effective step ms | model ms used by test | interpretation |
|---|---:|---:|---|
| control / agent repeat1 / W4 | 9.999 | 9.609 | pure W4 same-workload reference, one request |
| test-c1 / agent-loop / W8 | 14.482 | 11.831 | all three measured requests at this width |
| test-c1 / code-edit / W16 | 19.136 | 16.273 | all three measured requests at this width |
| test-c2 / agent-loop / W8 | 14.590 | 11.831 | all three measured requests at this width |
| test-c2 / code-edit / W16 | 18.774 | 16.273 | all three measured requests at this width |
| control / code-edit / W16 | 18.983 | 16.273 | same-width comparison for acceptance drift |

The old control actually uses its own default model 9.74+0.70*S; the model column above is the test fit shown as a common reference, not a claim that the control was configured with it. For each test's W8 agent trace, applying the policy's own min(a,3) downward estimator gives mean S3-prefix token counts of 3.096 / 3.158. The fitted cost ratio would therefore imply a sizable W8 benefit from those same observations, while the larger measured W8/W4 cost ratio leaves little benefit. This is a model-diagnostic counterfactual, not a measured alternate-W4 execution of the same token stream.

### Iteration 3 exact recommendation and stop

**FAIL — no serve-fast.sh edit (zero-line diff), no production integration of the worktree patch, no new default config or STEP_A/B exports.** Keep production w16_conf.json and defaults A=9.74, B=0.70. The optional implementation and w16_3_7_15_c.json remain experimental review artifacts; they are not a passing production policy. The memory fraction used for this experiment is not being applied to the production launcher.

This was the last authorized config/code iteration. No fourth configuration, model refit, code change or GPU experiment was started after the gate. The mechanism analysis above is the final outcome, not a plan for another tuning run.

### Iteration 3 completion and artifact checks

- 54 CPU adaptive tests passed, including all 10 new confidence tests. Production-default parity was additionally verified against the actual production module for 3000 updates per slot, with exact score/decision/legacy-state equality. The logged-ratio examples were checked through the real worktree implementation. Test output and selection evidence are preserved.
- All three arms used the requested PYTHONPATH overlay; code-origin.txt confirms the worktree module path per arm. All 36 measured JSONL records are present and unique by workload/repeat. All three needle_test.py 18500 0.4 calls passed with actual prompt_tokens=17307. Histogram traces were matched to exact request boundaries and emission totals cross-checked as in previous iterations.
- All arms retained memory fraction 0.920 and empty SERVE_DISPLAY_HZ. The specified sustained post-ready abort did not trigger. Steady minima 908 / 1928 / 848 MiB and medians 1410 / 2648 / 1410 MiB are recorded; the 4 GiB steady target was not met. Startup minima are separate from this observation.
- Owned HTTP server PIDs 3216849, 3220363 and 3223655 no longer existed at final verification; the driver completed successfully after process-group cleanup. A different server group (leader3226668, started15:33:49) was present during the later read-only GPU check and the lock was busy; it was not killed or modified. Do not interpret that other job's GPU usage as WA3 residual allocation.
- Production serve-fast.sh, serve-local.sh and adaptive_confidence.py SHA-256 match the original snapshot; production tracked/untracked worktree status is unchanged. The requested new worktree is on codex/wa2-three-widths with only adaptive_confidence.py modified and the new adjacent CPU test untracked. No commit, push or additional branch was created. No production source, launcher or venv edit was made.
