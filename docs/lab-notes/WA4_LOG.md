# WA4: three-width confidence policy after X3 target autotune fix

Status: COMPLETE — 2026-09-07. A1 -> B -> C -> A2 complete; B PASS, C FAIL versus pooled A. D not run because C shows no mistuned margin. All 48 measured requests and four needle checks passed validation; all arms retain >=4 GiB steady free VRAM. Production unchanged; no commit or push.

## Registered setup and protocol

- Base: codex/x3-adaptive-cost, commit 446c801189. New worktree $HOME/tools/sglang-wa4, branch codex/wa4-three-widths. Applied WA2 commit 2671ec1b00 with cherry-pick --no-commit, without conflicts. HEAD remains X3; WA2 changes are staged, no new commit or push.
- A1 -> B -> C -> A2, releasing the GPU flock between every arm. Only if C fails its gate and its decision log identifies a mistuned margin: D -> A3. Each server lifetime runs under flock -w 28800 $HOME/.gpu.lock.
- A: actual production source at 14d4c4c985, production w16_conf.json, default step model 9.74/0.70, autotune fix absent. B: WA4 overlay, same production config/defaults, SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1. WA2 optional policy controls are absent/inert in B. C: WA4 overlay, config c, fix=1, STEP_A/B=7.943/0.5554. D remains conditional and unconfigured.
- Unmodified production launcher and venv, PYTHONDONTWRITEBYTECODE=1. bwrap mounts production and overlay source read-only, with only the hardcoded production .cache path mapped to a WA4-private copy of the X3 cache. No production source/cache/launcher/venv writes.
- MEM_FRACTION=W16_MEM_FRACTION=WA_MEM_FRACTION=0.920, SERVE_DISPLAY_HZ empty. MAX_TOTAL_TOKENS=131072 in ALL arms, as in X3, to retain >=4096 MiB free at steady state while preserving the requested fraction. This common KV cap is an experimental control, not a production change. Full default KV capacity is untested.
- Startup VRAM recorded without abort. Post-ready watchdog terminates only the owned process group for free VRAM <1024 MiB continuously for >=10 s. Steady >=4096 MiB requirement is checked separately in the results; 2-second VRAM samples, startup/steady/cleanup separated. Unrelated processes are never terminated.
- Same WA2 fnbench wrapper: code-edit, prose-en, prose-ja, agent-loop; one warmup and n=3 greedy measured requests each; needle 18500/depth .4 once per arm. Fresh timestamped labels; output directory and events use exclusive creation. Existing labels are never reused.
- Identical adaptive trace/debug instrumentation in every arm. Width histograms count measured verified decode steps (W=S+1); warmups and needle excluded via exact client request timestamps. Acceptance is mean tokens per verify from trace, with SSE acceptance and <=32-token request emission cross-check. Tracing synchronizes the staging event and is not a production-default cost; absolute results describe the instrumented server.
- Gate versus equal-weight mean of all completed A arms: C/D require code-edit >=0%, prose-en/prose-ja >=-2%, agent-loop >=+5%. B requires every workload >=-2% and at least one >+2%. All A arms have n=3; final A mean pools 6 (or 9 if D runs) requests. Initial/final drift and small sample limitations are explicit.

## CPU integration and step model

**59 adaptive CPU tests PASS** (0.798 s). Combined adaptive CPU test discovery includes both test_adaptive_target_autotune.py and test_adaptive_confidence.py plus existing adaptive sizing/switch/spec tests. Evidence: specs/wa4/cpu-tests.txt. git diff --cached --check passed.

The user-rounded STEP_A/B=7.943/0.5554 predicts W4=9.6092, W8=11.8308, W16=16.2740 ms. Recomputed against the 10 FIXED-width trimmed observations in WIDTH_SWEEP_0907, RMSE=0.325801 ms; this remains the original fixed-width fit. X3's more recent fixed W4/W8 and post-fix pinned-W8 traces are independently rechecked below. These are legacy-trimmed times, not wall time or client acceptance/tps effective cost. No new fixed-width server arm is added to the requested order.

| width / evidence | model ms | code trimmed ms | prose trimmed ms | agent trimmed ms |
|---|---:|---:|---:|---:|
| fixed W4, X3 d | 9.6092 | 9.448 | 9.273 | 9.304 |
| fixed W8, X3 a | 11.8308 | 11.768 | 11.600 | 11.839 |
| pinned adaptive W8 + fix, X3 b-fix | 11.8308 | 11.957 | 11.792 | 11.662 |

Post-fix pinned W8 differs from the fixed-width model by +1.07%/-0.33%/-1.43% and from same-session fixed W8 by +1.61%/+1.66%/-1.50% for code/prose/agent. Thus the FIXED-width trimmed model still fits at its intended scale after the fix restores W8 cost. This does not establish equal end-to-end adaptive costs or a same-session fresh W16 trace; WA4 effective costs and decisions will be analyzed separately. Inputs are existing X3 artifacts, not new WA4 GPU measurements.

### Starting wa4-20260907-225858-A

Arm A, 2026-09-07T22:59:30.050623+09:00; per-arm flock held.

wa4-20260907-225858-A: ready; 9725 MiB free VRAM.

wa4-20260907-225858-A: owned process groups cleaned up; lock released on command exit.

## Arm wa4-20260907-225858-A

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 483.21 (436.52-527.18) | 10.843 / 10.710 | 0 (0.00%) | 0 (0.00%) | 674 (100.00%) |
| prose-en | 219.74 (216.80-225.13) | 2.557 / 2.552 | 1409 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 208.45 (206.37-211.31) | 2.375 / 2.376 | 1517 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 257.76 (237.34-269.19) | 4.162 / 4.115 | 588 (63.29%) | 0 (0.00%) | 341 (36.71%) |

Needle: `[wa4-20260907-225858-A] prompt_tokens=17307 completion=12 time=1.7s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 9725 MiB; minimum 7778 MiB (7.596 GiB); median 8257 MiB. Startup minimum 5401 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [23, 28, 18]; prose-en: [3, 0, -1]; prose-ja: [3, -1, 0]; agent-loop: [14, 9, 1].

Artifacts: `specs/wa4/wa4-20260907-225858-A/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

A1 mechanism checkpoint: measured code is 100% W16; both prose workloads 100% W4; agent-loop 63.29% W4 / 36.71% W16. Effective client-derived step costs (mean 1000*trace acceptance/decode tps) are code 22.445 ms, prose-en 11.638 ms, prose-ja 11.392 ms; mixed-width agent 16.098 ms. This is not a profiler measurement. In agent warmup, A1 promotes S3->S15 at batch5025, then demotes at batch5415 during measured repeat2; subsequent grace is 160 batches. No causal attribution or gate is made from A1 alone. Source hashes still match the pre-run production snapshot. B is queued behind other shared-lock users.

### Starting wa4-20260907-230303-B

Arm B, 2026-09-07T23:16:02.090403+09:00; per-arm flock held.

wa4-20260907-230303-B: ready; 9737 MiB free VRAM.

wa4-20260907-230303-B: owned process groups cleaned up; lock released on command exit.

## Arm wa4-20260907-230303-B

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 578.59 (570.59-589.14) | 11.743 / 11.653 | 0 (0.00%) | 0 (0.00%) | 618 (100.00%) |
| prose-en | 242.60 (232.54-254.90) | 2.569 / 2.562 | 1405 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 220.01 (216.20-224.42) | 2.363 / 2.367 | 1524 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 295.05 (289.08-304.55) | 3.161 / 3.153 | 1140 (100.00%) | 0 (0.00%) | 0 (0.00%) |

Needle: `[wa4-20260907-230303-B] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 9737 MiB; minimum 7982 MiB (7.795 GiB); median 8324 MiB. Startup minimum 5406 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [15, 12, 29]; prose-en: [3, 1, 0]; prose-ja: [1, 0, 0]; agent-loop: [3, -1, 0].

Artifacts: `specs/wa4/wa4-20260907-230303-B/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa4-20260907-231935-C

Arm C, 2026-09-07T23:19:53.040313+09:00; per-arm flock held.

B interim comparison to A1 only (not the registered final gate):
| workload | B vs A1 t/s | B effective step ms |
|---|---:|---:|
| code-edit | +19.74% | 20.297 |
| prose-en | +10.40% | 10.589 |
| prose-ja | +5.55% | 10.743 |
| agent-loop | +14.47% | 10.714 |

B code remains 100% W16, both prose and agent 100% W4. Code acceptance rises from A1 10.843 to B 11.743; the initial W16 graph is not the non-initial state targeted by X3, so the entire code gain must not be assigned to the fix. Contemporaneous drift and acceptance variation will be assessed against A2. Corrected W4 effective costs are observational end-to-end estimates, not newly profiled kernel costs.

wa4-20260907-231935-C: ready; 7231 MiB free VRAM.

wa4-20260907-231935-C: owned process groups cleaned up; lock released on command exit.

## Arm wa4-20260907-231935-C

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 493.69 (467.46-519.93) | 12.101 / 11.966 | 0 (0.00%) | 0 (0.00%) | 602 (100.00%) |
| prose-en | 216.18 (210.89-220.24) | 2.561 / 2.554 | 1407 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 220.13 (217.33-224.02) | 2.382 / 2.394 | 1512 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 333.72 (312.44-360.33) | 4.691 / 4.656 | 0 (0.00%) | 774 (100.00%) | 0 (0.00%) |

Needle: `[wa4-20260907-231935-C] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 7231 MiB; minimum 5255 MiB (5.132 GiB); median 5791 MiB. Startup minimum 5006 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [29, 25, 27]; prose-en: [2, 0, 0]; prose-ja: [1, 0, 0]; agent-loop: [8, 6, -1].

Artifacts: `specs/wa4/wa4-20260907-231935-C/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa4-20260907-232335-A

Arm A, 2026-09-07T23:23:35.540271+09:00; per-arm flock held.

wa4-20260907-232335-A: ready; 8411 MiB free VRAM.

wa4-20260907-232335-A: owned process groups cleaned up; lock released on command exit.

## Arm wa4-20260907-232335-A

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 592.83 (583.67-599.66) | 11.894 / 11.805 | 0 (0.00%) | 0 (0.00%) | 610 (100.00%) |
| prose-en | 247.73 (238.71-258.01) | 2.600 / 2.595 | 1386 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 224.21 (222.39-226.98) | 2.342 / 2.347 | 1539 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 302.50 (293.79-315.34) | 3.268 / 3.248 | 1102 (99.46%) | 0 (0.00%) | 6 (0.54%) |

Needle: `[wa4-20260907-232335-A] prompt_tokens=17307 completion=12 time=1.6s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 8411 MiB; minimum 6286 MiB (6.139 GiB); median 6991 MiB. Startup minimum 4066 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [26, 6, 22]; prose-en: [1, -1, 1]; prose-ja: [3, -1, 2]; agent-loop: [4, 0, 11].

Artifacts: `specs/wa4/wa4-20260907-232335-A/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

## Gate versus all A controls (n=6)

| workload | A pooled mean | B t/s | B delta | B floor | C t/s | C delta | C gate | required C/D |
|---|---:|---:|---:|---|---:|---:|---|---:|
| code-edit | 538.02 | 578.59 | +7.54% | PASS | 493.69 | -8.24% | FAIL | +0% |
| prose-en | 233.73 | 242.60 | +3.79% | PASS | 216.18 | -7.51% | FAIL | -2% |
| prose-ja | 216.33 | 220.01 | +1.70% | PASS | 220.13 | +1.76% | PASS | -2% |
| agent-loop | 280.13 | 295.05 | +5.33% | PASS | 333.72 | +19.13% | PASS | +5% |

B gate: all deltas >=-2%, at least one >+2%. C/D gate: all four workload thresholds must pass.

Verdicts: B PASS, C FAIL.

### Control drift

| workload | A1 mean | A2 mean | A2 vs A1 |
|---|---:|---:|---:|
| code-edit | 483.21 | 592.83 | +22.69% |
| prose-en | 219.74 | 247.73 | +12.74% |
| prose-ja | 208.45 | 224.21 | +7.56% |
| agent-loop | 257.76 | 302.50 | +17.36% |

### VRAM and needle

| arm | ready MiB | steady median MiB | steady min MiB | >=4096 MiB | needle |
|---|---:|---:|---:|---|---|
| wa4-20260907-225858-A | 9725 | 8257 | 7778 | PASS | PASS |
| wa4-20260907-230303-B | 9737 | 8324 | 7982 | PASS | PASS |
| wa4-20260907-231935-C | 7231 | 5791 | 5255 | PASS | PASS |
| wa4-20260907-232335-A | 8411 | 6991 | 6286 | PASS | PASS |

## Final mechanism analysis and conditional D decision

**D NOT RUN; no config d created.** C fails the gate, but the additional requested condition for D is not met: its decision log does not show a mistuned margin. Every measured code request is W16, every measured prose request is W4, and every measured agent request is W8. A hysteresis-only variant cannot repair the observed same-width code/prose cost regressions by selecting the same widths more often. No speculative retuning or extra A3 arm was started.

C has exactly two policy switches across warmups and measured requests, both in warmup. At batch885 on prose-en, predicted S3=283, S7=291, S15=251 t/s selects S3: 283/251=1.1275 clears down-to-3 threshold 1.03, whereas 291/251=1.1594 barely clears down-to-7 threshold 1.15 but loses the adjusted score comparison. At agent batch4725, S7/S3=393/347=1.1326 remains below the 1.144 promotion threshold; at batch4735, 378/328=1.1524 clears it and selects S7. The 80-batch reversal hold finishes within warmup. There are no measured false promotions, lost W8 opportunities, or code W8 dwell to motivate config d.

### Client-derived effective step costs

Effective ms = mean across repeats of 1000 * trace tokens-per-verify / client decode t/s. These are not CUDA profiler times and must not be fitted interchangeably with the fixed-width legacy-trimmed measurements.

| arm | code W16 ms | prose-en W4 ms | prose-ja W4 ms | agent ms | agent measured widths |
|---|---:|---:|---:|---:|---|
| A1 | 22.445 | 11.638 | 11.392 | 16.098 | W4 63.29%, W16 36.71% |
| B | 20.297 | 10.589 | 10.743 | 10.714 | W4 100.00% |
| C | 24.540 | 11.846 | 10.821 | 14.046 | W8 100.00% |
| A2 | 20.062 | 10.496 | 10.445 | 10.800 | W4 99.46%, W16 0.54% |

C agent acceptance is 4.691 versus B 3.161, while effective step cost is 14.046 versus 10.714 ms (1.311x). This yields +13.11% measured C versus B and +19.13% versus pooled A: the corrected three-width policy now recovers the intended agent advantage. C's own S3-prefix acceptance is 3.151 tokens/verify, consistent with B's measured W4 acceptance. The fit predicts W8/W4=1.2312x; C/B effective cost is still larger, so the fixed-width trimmed fit does not fully describe the instrumented end-to-end adaptive runtime.

C code acceptance (12.101) is higher than B (11.743) and pooled A (11.368), yet C code effective step cost is 24.540 ms versus B 20.297 and A1/A2 22.445/20.062. English prose acceptance is close (C 2.561, B 2.569, pooled A 2.578), while C effective cost 11.846 ms exceeds B 10.589 and A2 10.496. These failures therefore cannot be blamed on an observed margin-induced width error; their extra effective time is not isolated by this A/B. No new profiler trace, tracing ablation, or alternate runtime implementation was run.

### Drift and limits of attribution

A2 versus A1 drift is +22.69% code, +12.74% prose-en, +7.56% prose-ja, +17.36% agent. The B lock queue introduced about 13 minutes between A1 cleanup and B start; B/C/A2 ran in succession afterward. The registered mean-A gate is reported exactly, without replacing it with a favorable control. B versus A2 alone is -2.40%/-2.07%/-1.87%/-2.46% (code/en/ja/agent), so B does not independently pass against that later control. The pooled +7.54%/+3.79%/+1.70%/+5.33% figures are descriptive observations, not a precise causal production gain or a repeated significance result. X3 separately establishes the target-kernel mechanism and post-fix fixed-width restoration.

Full nvidia-smi before/after snapshots retain power/temperature context. Power limit remains 325 W. Whole-server SW power-capping counter deltas are A1 1.630 s, B 1.609 s, C 1.697 s, A2 1.853 s; after snapshots follow the needle request and all report SW Power Cap active. No thermal slowdown counter increase is recorded. These snapshots are not aligned to measured decode windows and cannot explain C's code cost or exclude transient clock/desktop effects. No power, clock, or display setting was changed. This WA4 run does not extend X3's no-observed-decode-cap finding to new fnbench requests.

## Exact production recommendation (not applied)

**(i) Recommend shipping the X3 fix for the wa profile, as a profile-scoped opt-in.** B passes the specified pooled-control gate and its needle passes; X3 provides independent target-tactic and kernel-count evidence. Keep the implementation default off globally. Integrate commit `446c801189` from `codex/x3-adaptive-cost` into the production SGLang fork `codex/perf-v1` (currently `14d4c4c985`). It contains `adaptive_runtime_state.py`, the `eagle_worker_v2.py` construction wrapper, and `test_adaptive_target_autotune.py`. The export alone cannot enable a fix absent from the production source. The WA2 commit is unnecessary for this recommendation.

After that fork integration, the exact wa-profile launcher addition is the following environment assignment alongside its existing assignments:

```sh
SGLANG_ADAPTIVE_TARGET_AUTOTUNE="${SGLANG_ADAPTIVE_TARGET_AUTOTUNE:-1}"
```

Retain `adaptive/w16_conf.json`, `[3,15]`, default `SGLANG_ADAPTIVE_STEP_A=9.74` and `SGLANG_ADAPTIVE_STEP_B=0.70`, and the existing confidence-policy parameters. Do not enable the flag globally for other profiles. This is a code/profile recommendation under the tested single-stream conditions, with the large drift caveat above; it does not promise the pooled percentages as a stable production uplift.

**(ii) Do not ship the three-width config or the WA2 policy commit.** Config c FAILS the full gate despite its agent gain. Keep `w16_3_7_15_c.json`, the refit 7.943/0.5554 and the WA2 options experimental. No config d was justified or created. Do not change the production config fallback or STEP defaults.

The tested common memory envelope is fraction 0.920 plus MAX_TOTAL_TOKENS=131072, with all steady minima above 4096 MiB. These experimental overrides were not applied to production or recommended as new permanent capacity defaults. Full default KV capacity, long-context saturation, bs>1, multi-GPU, and untraced fnbench throughput remain untested. Needle PASS establishes this one 17,307-token retrieval test, not general generation equivalence.

## Final verification

All four arms completed in A1 -> B -> C -> A2 order, each under its own flock. All 48 measured requests, 16 warmups, four needle calls and all request-to-trace checks completed. 59 adaptive CPU tests passed, including both requested test files. Each server used the recorded source origin and production read-only mount; no venv install, source edit, launcher edit, commit, push or production application was performed. Only branch codex/wa4-three-widths was created. Its HEAD remains X3 commit 446c801189; the WA2 commit is applied as two staged file changes without a new commit. Final manifests and owned-process checks follow.

Final manifest verification: production HEAD/status and the six tracked/source/launcher/venv-config hashes match `production-before.json`; all 19 harness/workload hashes match. Actual server argument logs confirm mem_fraction_static=0.92 and max_total_tokens=131072 in every arm. Owned process groups 3420285, 3434605, 3438021, 3442030 have no remaining processes; final nvidia-smi lists no compute applications. Evidence: `specs/wa4/completeness.json`, `production-final-check.json`, `process-cleanup-check.json`, `integration.json`, `harness-manifest.json`, `step-model-check.json`, `mechanism-summary.json` and per-arm artifacts. The final verification parser was corrected to read the server argument log's dictionary representation; no benchmark rerun or result change was involved.

## Consolidated throughput table

| workload | A1 mean (min-max) | B mean (min-max) | C mean (min-max) | A2 mean (min-max) |
|---|---:|---:|---:|---:|
| code-edit | 483.21 (436.52-527.18) | 578.59 (570.59-589.14) | 493.69 (467.46-519.93) | 592.83 (583.67-599.66) |
| prose-en | 219.74 (216.80-225.13) | 242.60 (232.54-254.90) | 216.18 (210.89-220.24) | 247.73 (238.71-258.01) |
| prose-ja | 208.45 (206.37-211.31) | 220.01 (216.20-224.42) | 220.13 (217.33-224.02) | 224.21 (222.39-226.98) |
| agent-loop | 257.76 (237.34-269.19) | 295.05 (289.08-304.55) | 333.72 (312.44-360.33) | 302.50 (293.79-315.34) |

All cells are decode t/s, n=3 per arm/workload. Width counts, percentages, trace/SSE acceptance and request-boundary cross-checks are in the per-arm tables above.
