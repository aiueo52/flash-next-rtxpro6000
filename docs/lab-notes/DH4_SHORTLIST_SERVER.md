# DH4 shortlist server integration

Status: COMPLETE — NO-GO. All five requested arms, 52 measured workload requests, 13 needles, and 10 CUPTI traces are complete. No production launcher change is recommended or applied.

## Frozen scope and protocol

Worktree `$HOME/tools/sglang-dh4`, branch `codex/dh4-shortlist-server`, based on `codex/perf-v1` at `14d4c4c985`. DH3 commit `7625982427` applied with `git cherry-pick --no-commit` to satisfy the explicit no-commit instruction. Production tree/launcher/venv are read-only inputs. No push or additional branch.

Read DH3, DH2, DH1 budget, C1 and X3 graph construction evidence, MTP forward, hot-head installation, W8A16 GEMV, draft loop and graph runner. The C1 position-0 controller input remains the original full-head probability. Recursive chain diagnostics receive DH2's frozen calibrated full-distribution proxy; raw shortlist softmax is never published to C1. Fixed-width arms do not exercise adaptive width decisions or claim joint controller calibration.

DH2 `by_step.csv` (learned_r128/K1024/no fallback) was measured offline on the DH2 held-out split of the author's private MTP data; its results are not published. Scope: position0 (draft-extend) keeps the full head; the 14 recursive forwards at W16 (2 at W4) use the shortlist. Step0 was excluded from DH2 frozen gate and DH1 budget.

Policy: rank128, K1024, selector margin threshold -infinity (no fallback), selector checkpoint and 20-bin calibration supplied by explicit env paths. Default flag is off. Singleton greedy topk1/direct-chain only; larger batch shapes retain full head. Reject incompatible head/model/configuration rather than silently changing semantics.

Numerical correction relative to DH3: production reduces 10 sequential 256-wide tiles, scales after the dot, rounds to BF16, then argmax ties select the lowest hot-row index. Server gather/rescore reproduces this contract; DH3 FP32-dequantized reference remains unchanged under bench/dh3. Candidate membership uses the unchanged two BF16 selector kernels and 64-partition selection. GPU comparisons on DH2 states were run; see the numerical test notes below.

Fallback: attempt CUDA IF insertion into active Torch capture, capturing the body on a separate stream and updating the parent's dependencies. Also provide masked full GEMV/reducer with device-flag early return for comparison/unsupported IF capture. Both have no host flag read. Buffers and library owned by worker; parent graph owns IF/body lifetime. Publication writes mapped token, chain column, position increment, calibrated confidence, and cumulative diagnostic counters in one kernel.

A/B order: W16 control-before -> W16 shortlist -> W16 control-after (3 fnbench repeats/workload), W4 control -> W4 shortlist (2 repeats). Every arm has code-edit, prose-en, prose-ja, agent-loop, and needle; CUPTI 20-step traces code-edit/prose-en. Fresh timestamped labels. One `flock -w 28800 $HOME/.gpu.lock` per server lifetime. All memory fractions 0.920; SERVE_DISPLAY_HZ empty; common MAX_TOTAL_TOKENS=131072 (same X3 reserve strategy). No loading-transient VRAM abort; >=4096 MiB at steady state. Only owned process groups cleaned up.

Gate: W16 throughput gain >=3% versus two-control mean, acceptance loss <=1% for every workload; W4 non-negative; needle PASS. Report requested legacy trimmed CUPTI metric alongside its limitation (kernel-name clipping is not physical elapsed time), wall/busy evidence, fallback rate and full shortlist graph-region time including scheduling gaps. No launcher changes will be applied.

## Incremental evidence

- Worktree created; no commit. Production before-manifest at `specs/dh4/production-before.json`; private runtime cache copied from X3; production will be mounted read-only with only `.cache` redirected to this private copy.
- CPU parsing passed. Conditional bridge initially failed with conda GCC headers; rebuild explicitly selects `/usr/bin/g++`, matching the production launcher toolchain. No installation or environment mutation.

### Starting dh4-20260907-233642-w16-control-before

w16/control-before, repeats=3; per-arm flock acquired at 2026-09-07T23:39:02.500424+09:00.

dh4-20260907-233642-w16-control-before: ERROR Command '['$HOME/tools/sglang-rtxpro6000/.venv/bin/python', '-B', '-c', 'import sglang.srt.speculative.eagle_worker_v2 as m; print(m.__file__)']' returned non-zero exit status 1.

dh4-20260907-233642-w16-control-before: owned process groups cleaned; flock releases on exit.

### GPU integration tests

Both conditional and masked modes were tested with real Torch CUDAGraph capture, false/true/false fallback replay and a 14-forward graph, driven by DH2 states (private data); those results are not published. On the same DH2 states, per mode, candidate logits versus the production BF16-output full GEMV, selector IDs versus DH3, included-winner recovery, hot map, chain column, position increments, confidence interpolation, counter reset and forced-fallback full confidence were checked; the results of these checks on private states are not published. A synthetic all-zero tie input selected deterministically. Results and CUPTI traces: `sglang-dh4/bench/dh4/gpu-tests.json`, `trace-conditional.json`, `trace-masked.json`.

Conditional mode is the configured default; in that mode a skipped full-head body launches no kernels. Both modes remain available for tests. The initial default-sandbox invocation could not access CUDA and performed no GPU computation; later tests used host execution under the same flock. Two test harness corrections (BF16 input cast, CUDA enum namespace) were made before the final run.

The first control label failed before server launch: the origin-check imported deep_ep before serve-local established CUDA_HOME. Replaced that diagnostic with importlib.find_spec (no module execution). No startup/memory abort and no benchmark samples from this label. A fresh label will be used.

### Starting dh4-20260907-233947-w16-control-before

w16/control-before, repeats=3; per-arm flock acquired at 2026-09-07T23:44:11.096962+09:00.

dh4-20260907-233947-w16-control-before: ready, 9185 MiB free; requested memory fraction 0.920.

dh4-20260907-233947-w16-control-before: all benchmarks/traces completed; needle 3/3 PASS.

dh4-20260907-233947-w16-control-before: owned process groups cleaned; flock releases on exit.

Acceptance measurement definition: this server's chat SSE does not return `accept_length`; fnbench preserves Prometheus before/after snapshots for every repeat. Use `delta(sglang:generation_tokens_total) / delta(sglang:spec_verify_calls_total)` per completed request (delivered tokens per verify). Do not difference the windowed acceptance gauge or use `/get_server_info` lifetime averages. This includes the initial prefill output token and clips the final request to its output limit; the same definition applies to all arms. Four-workload prompt lengths/output limits remain the unmodified fnbench manifest (`bench/workloads`), with an untimed warmup for each workload.

### First W16 control complete

| workload | t/s mean | tokens/verify mean | repeat t/s |
|---|---:|---:|---|
| code-edit | 609.049 | 11.1435 | 550.95, 639.81, 636.39 |
| prose-en | 182.817 | 3.1122 | 182.42, 186.55, 179.48 |
| prose-ja | 150.413 | 2.4902 | 147.79, 154.67, 148.78 |
| agent-loop | 310.655 | 5.5246 | 290.13, 287.68, 354.15 |

Needle 3/3 PASS; minimum steady free VRAM 7012 MiB. CUPTI legacy trimmed code-edit/prose-en: 16.0309/15.7328 ms. The first code-edit repeat is slower and has lower acceptance than the next two; retain it as required, and use the second control to assess session/trajectory variability.

Control CUPTI validates the original head-cost premise: `_w8a16_gemv_kernel`, grid `[1536,1,1]`, 280 calls = 14 forwards x 20 traced W16 steps. Median 81.441 us code-edit and 80.6245 us prose-en. The remaining server integration result therefore compares against a head that actually costs ~81 us in this session.

### Recovery after desktop restart (2026-09-08)

User reports the task process was killed by a system-wide desktop freeze at 2026-09-07 23:51 JST, followed by a desktop restart. All implementation/checkpoint hashes still match the pre-interruption manifest; three protected production file hashes also match; the production launcher and eagle_worker changed externally (see audit below). The completed W16 control contains both traces, 12 measured requests, three passing needles, and a complete/cleanup event. It is retained. The queued shortlist arm never acquired the lock: no arm directory/start event exists. Restart it from scratch with a fresh timestamped label. No report section was partially written; the README tool call was aborted before writing and will be recreated.

The desktop restart lies between the first W16 control and the remaining arms. Report this temporal discontinuity with the final A/B results; do not erase or substitute the completed control. No cause attribution for the system freeze is inferred from this task's evidence.

Recovery audit correction: production `serve-fast.sh` and `eagle_worker_v2.py` changed externally while DH4 was interrupted. Production HEAD now contains the previously described X3 commit `446c801189`. Removing exactly the two X3/WA4 comments and the wa-only `SGLANG_ADAPTIVE_TARGET_AUTOTUNE` default from a private launcher copy reproduces the initial SHA256 `2f44a387f894d1653b50f5860058181460339a0bcfd3e9711e6c1ead0de0c9fe`. Thus W16/W4 launcher behavior is unchanged; the changed production worker is bypassed by the unchanged DH4 PYTHONPATH overlay in every arm. Production files are preserved in their externally updated state, not reverted. See `restart-check.json` and private launcher snapshots.

Remaining arms additionally bind-mount private launcher snapshots over the read-only production launcher paths, inside each child mount namespace only. Both snapshots reproduce the original before-manifest hashes exactly; production files on disk are untouched. This freezes W16/W4 launch behavior against further external edits. `serve-local` and the production venv entrypoint were unchanged at restart.

### Starting dh4-20260908-000135-w16-shortlist

w16/shortlist, repeats=3; per-arm flock acquired at 2026-09-08T00:05:45.999162+09:00.

dh4-20260908-000135-w16-shortlist: ready, 9326 MiB free; requested memory fraction 0.920.

dh4-20260908-000135-w16-shortlist: all benchmarks/traces completed; needle 3/3 PASS.

dh4-20260908-000135-w16-shortlist: owned process groups cleaned; flock releases on exit.

### Integration preflight correction: batch context lost by dataclass copying

`dh4-20260908-000135-w16-shortlist` completed requests but is **INVALID for A/B**: CUPTI contains zero selector kernels, the original 280 full-head calls, and diagnostic shortlist counters remain zero. The eager runner (`EagerRunner.load_batch` / buffer registry extraction) uses `dataclasses.replace`, which drops the dynamically attached head context. This is an integration connection bug, not a performance/quality result. Added `dh4_shortlist` as a formal optional `ForwardBatch` dataclass field so both runner copy paths preserve it; add a CPU regression and require nonzero selector kernels in every shortlist trace before accepting any arm. Preserve the invalid artifacts under their original label. The queued control-after was canceled before acquiring the lock so the valid order remains control-before -> corrected shortlist -> control-after.

CPU regression passed against the actual ForwardBatch class with CUDA hidden: the marker survives dataclasses.replace both unchanged and with batch overrides, and resetting to None survives the next copy. Kernel code is unchanged from the GPU equality/IF tests.

### Starting dh4-20260908-001138-w16-shortlist

w16/shortlist, repeats=3; per-arm flock acquired at 2026-09-08T00:14:16.827979+09:00.

dh4-20260908-001138-w16-shortlist: ready, 9297 MiB free; requested memory fraction 0.920.

dh4-20260908-001138-w16-shortlist: all benchmarks/traces completed; needle 3/3 PASS.

dh4-20260908-001138-w16-shortlist: owned process groups cleaned; flock releases on exit.

### Corrected W16 shortlist complete

| workload | t/s mean | tokens/verify mean | shortlist calls including warmup | fallbacks |
|---|---:|---:|---:|---:|
| code-edit | 633.150 | 10.5800 | 12432 | 0 |
| prose-en | 205.221 | 3.1749 | 21350 | 0 |
| prose-ja | 168.547 | 2.5668 | 25970 | 0 |
| agent-loop | 307.573 | 4.8964 | 13846 | 0 |

Each 20-step trace has 280 selector/publish pairs, no full-head fallback body executions. Total per-forward span (selector_down start through dh4_publish end): 35.184 us code-edit, 34.528 us prose-en. Rescoring kernel medians 12.672/12.512 us; they include the sequential production BF16-equivalent reduction, and chain/position publication is a separate kernel. IF scheduling gaps remain in the span. Step legacy trimmed 15.8632/15.1549 ms; wall 16.0817/15.4948 ms. Final A/B gate waits for the second control. Needle 3/3 PASS; minimum steady free VRAM 6787 MiB.

### Starting dh4-20260908-001525-w16-control-after

w16/control-after, repeats=3; per-arm flock acquired at 2026-09-08T00:28:14.059939+09:00.

dh4-20260908-001525-w16-control-after: ready, 8914 MiB free; requested memory fraction 0.920.

dh4-20260908-001525-w16-control-after: all benchmarks/traces completed; needle 3/3 PASS.

dh4-20260908-001525-w16-control-after: owned process groups cleaned; flock releases on exit.

### W16 gate determined: FAIL

| workload | control mean t/s | shortlist t/s | throughput change | acceptance change |
|---|---:|---:|---:|---:|
| code-edit | 611.838 | 633.150 | +3.483% | -5.052% |
| prose-en | 181.001 | 205.221 | +13.381% | +3.759% |
| prose-ja | 154.974 | 168.547 | +8.758% | +0.818% |
| agent-loop | 304.928 | 307.573 | +0.868% | -8.812% |

The W16 primary gate fails: code-edit exceeds the 1% acceptance-loss limit; agent-loop exceeds that limit and does not reach +3% throughput. Needle passes for all three W16 arms. No selector/K/threshold retuning follows this failure. Finish the two explicitly requested W4 arms for the secondary table, then stop with NO-GO. The comparison is small-n and spans the documented desktop restart; both W16 controls show materially higher mean code-edit and agent-loop acceptance than the shortlist arm. This is a measured serving gate failure, not a claim that every trajectory difference is causally identified.

### Starting dh4-20260908-002250-w4-control

w4/control, repeats=2; per-arm flock acquired at 2026-09-08T00:34:53.164641+09:00.

dh4-20260908-002250-w4-control: ready, 12454 MiB free; requested memory fraction 0.920.

dh4-20260908-002250-w4-control: all benchmarks/traces completed; needle 2/2 PASS.

dh4-20260908-002250-w4-control: owned process groups cleaned; flock releases on exit.

### Starting dh4-20260908-002735-w4-shortlist

w4/shortlist, repeats=2; per-arm flock acquired at 2026-09-08T00:38:02.338866+09:00.

dh4-20260908-002735-w4-shortlist: ready, 12472 MiB free; requested memory fraction 0.920.

dh4-20260908-002735-w4-shortlist: all benchmarks/traces completed; needle 2/2 PASS.

dh4-20260908-002735-w4-shortlist: owned process groups cleaned; flock releases on exit.

## Final measured tables

Gate: **NO-GO**.

Per-cell acceptance = delivered tokens/verify; throughput = arithmetic mean t/s across repeats. W16 n=3, W4 n=2. Control reference is the equal mean of the before/after arm means. No invalid or partial arm contributes.

| arm | code-edit acc / t/s | prose-en acc / t/s | prose-ja acc / t/s | agent-loop acc / t/s |
|---|---:|---:|---:|---:|
| w16 control-before | 11.1435 / 609.05 | 3.1122 / 182.82 | 2.4902 / 150.41 | 5.5246 / 310.65 |
| w16 shortlist | 10.5800 / 633.15 | 3.1749 / 205.22 | 2.5668 / 168.55 | 4.8964 / 307.57 |
| w16 control-after | 11.1423 / 614.63 | 3.0075 / 179.19 | 2.6017 / 159.54 | 5.2144 / 299.20 |
| w4 control | 3.8592 / 384.40 | 2.4688 / 251.41 | 2.3318 / 242.35 | 3.2582 / 329.42 |
| w4 shortlist | 3.6980 / 379.41 | 2.5143 / 273.37 | 2.3544 / 256.99 | 3.3442 / 357.00 |

| arm | code-edit trimmed / wall ms | prose-en trimmed / wall ms | shortlist span code / prose us, median (mean) | steady min free MiB | needle |
|---|---:|---:|---|---:|---|
| w16 control-before | 16.0309 / 17.1207 | 15.7328 / 16.7986 | N/A | 7012 | 3/3 PASS |
| w16 shortlist | 15.8632 / 16.0817 | 15.1549 / 15.4948 | 35.184 (36.419) / 34.528 (35.872) | 6787 | 3/3 PASS |
| w16 control-after | 16.0025 / 16.8661 | 15.7594 / 16.7290 | N/A | 6795 | 3/3 PASS |
| w4 control | 9.5201 / 9.7337 | 9.2387 / 9.5438 | N/A | 10316 | 2/2 PASS |
| w4 shortlist | 9.4145 / 9.5500 | 9.1172 / 9.4128 | 36.192 (35.888) / 36.768 (44.186) | 10334 | 2/2 PASS |

Legacy trimmed time clips individual kernel durations by kernel-name population; it is not physical elapsed time. Replacing the head changes that population. Wall and shortlist spans are reported separately, and throughput is measured outside profiling.

| shortlist arm | code-edit fallback/calls | prose-en fallback/calls | prose-ja fallback/calls | agent-loop fallback/calls | needle fallback/calls |
|---|---:|---:|---:|---:|---:|
| w16 shortlist | 0/12432 | 0/21350 | 0/25970 | 0/13846 | 0/84 |
| w4 shortlist | 0/3912 | 0/2746 | 0/3070 | 0/2216 | 0/16 |

Fallback denominators include each workload's untimed fnbench warmup, exclude boot and profiling through counter differencing. Control arms do not use a shortlist, so fallback rate is N/A.

| workload | W16 throughput change | W16 acceptance change | W4 throughput change | W4 acceptance change (diagnostic) | gate |
|---|---:|---:|---:|---:|---|
| code-edit | +3.483% | -5.052% | -1.297% | -4.176% | FAIL |
| prose-en | +13.381% | +3.759% | +8.734% | +1.844% | PASS |
| prose-ja | +8.758% | +0.818% | +6.040% | +0.969% | PASS |
| agent-loop | +0.868% | -8.812% | +8.371% | +2.639% | FAIL |

W16 requires throughput >=+3% and acceptance >=-1% on every workload; W4 requires throughput >=0%; all needles must pass. No statistical-confidence claim is attached to these small repeat counts.

| arm | artifact label |
|---|---|
| w16 control-before | `dh4-20260907-233947-w16-control-before` |
| w16 shortlist | `dh4-20260908-001138-w16-shortlist` |
| w16 control-after | `dh4-20260908-001525-w16-control-after` |
| w4 control | `dh4-20260908-002250-w4-control` |
| w4 shortlist | `dh4-20260908-002735-w4-shortlist` |

## Failure mechanism and decision

The head optimization is active and removes the recursive full-head work. W16 traces contain 280 selector/publish pairs each, W4 40 each: exactly 14 and 2 recursive forwards per 20 captured steps. There are zero fallback reducer executions in all four shortlist traces. The frozen threshold is -infinity; zero fallback is the intended DH2/DH3 policy, not evidence that missed full argmaxes were detected. The forced-fallback GPU tests separately exercise the branch.

In the server the whole selector-to-publication span has a W16 median of 35.184 us (code-edit) / 34.528 us (prose-en). The server rescore preserves production arithmetic: ten sequential 256-wide FP32 dot chunks, post-dot scales, then BF16 rounding. Its medians are 12.672/12.512 us. Chain/position/confidence publication adds 1.920/1.760 us, with IF scheduling gaps included in the measured whole-region span. Per-kernel medians are descriptive and do not sum exactly to the median span; selector/cache state also differs. The original W16 control full-head GEMV really costs 81.441/80.625 us in these traces, so the optimization does save head time.

W4 shortlist spans have medians 36.192/36.768 us. Prose-en's mean is 44.186 us because of long span outliers (maximum 229.440 us; p90 39.764 us), which remain in the artifact rather than being discarded. W16 also has rare spans around 322–324 us. This task does not identify the cause of those outliers or the desktop freeze. Kernel names, counts and per-trace distributions are retained in each analysis.json.

Conditional IF works inside the actual captured server draft runner and is the selected implementation. In the GPU comparison's false-flag replay, IF launches no full-head body kernels; masked mode launches early-exiting GEMV (median 1.248 us) and reducer (0.736 us). Both pass forced true/false replay. An additional full server A/B of masked mode was unnecessary once actual IF capture succeeded; no claim is made about masked-mode server throughput.

The serving gate fails despite lower step times. W16 code-edit acceptance drops 5.052%, and agent-loop drops 8.812%; agent-loop throughput improves only 0.868%, below +3%. W4 code-edit throughput drops 1.297% and diagnostic acceptance drops 4.176%. Missed candidates and subsequent draft trajectories remain approximate. No online full-head oracle was inserted into timed runs, so the exact causal split among candidate misses and trajectory variability is not measured.

The desktop restart separates the first control from the later arms. Both W16 controls independently have higher code-edit and agent-loop acceptance than the shortlist arm, but n=3/n=2 remains a small descriptive experiment. This report applies the requested point-estimate gate and makes no statistical-significance claim. Fixed W16/W4 arms do not validate adaptive width decisions: C1 retains full-head position-0 confidence and receives the frozen calibrated proxy in recursive chain diagnostics.

**Decision: NO-GO; stop here, without selector, K, threshold, or launcher retuning.**

## Final verification and deliverables

- CPU: DH3 reference 5 tests, DH4 config/map 4 tests, actual ForwardBatch dataclass-copy regression 1 test passed (10 total). The copy regression prevents recurrence of the preserved invalid no-op shortlist arm.
- GPU under flock: both fallback modes were compared on DH2 states (private data; candidate-logit, included-winner, hot-token mapping, confidence interpolation and publication results not published). Ties, false/true/false fallback and 14-step graph capture were also tested on those states; results not published.
- Valid A/B: exactly 52 measured workload requests, 13/13 needle PASS, 10 CUPTI traces with 200 draft-step annotations, and 640 traced shortlist forwards. Minimum steady free VRAM across all arms: 6787 MiB. All arms use fraction 0.920, empty SERVE_DISPLAY_HZ and common MAX_TOTAL_TOKENS=131072. No startup memory abort was used.
- Final source/checkpoint/calibration/bridge hashes match the frozen v2 manifest; `git diff --check` passed for DH4 tracked edits. A separate staged check flags inherited trailing whitespace in 294 lines of 14 DH3 raw trace JSON files; all are byte-identical to commit `7625982427` and are preserved as reference evidence (`dh3-staged-whitespace.txt`). Host process audit finds no remaining process in any DH4-owned server group. Each arm exited and released its flock; no other workload was killed.
- Worktree `$HOME/tools/sglang-dh4`, branch `codex/dh4-shortlist-server`, HEAD remains `14d4c4c985a87f329f27af9342970b28b35a8ed4`. DH3 cherry-pick is staged without a commit; DH4 changes and tests remain in the working tree. No commit, push or additional branch was made.
- Protected production inputs were mounted read-only. The externally changed X3 launcher remains identical to the restart snapshot; the other unchanged protected hashes still match the initial audit. The changed production worker is bypassed by the DH4 overlay. No production file was restored, overwritten or installed into the venv.

Evidence: `specs/dh4/results.json`, `gate.json`, `final-audit.json`, `final-process-audit.json`, each labeled arm's JSONL/Prometheus snapshots/CUPTI traces, and `sglang-dh4/bench/dh4/gpu-tests.json`. CPU analysis entrypoints: `analyze.py`, `gate.py`, `final_tables.py`, `final_audit.py`. Build, tests and experimental launch variables are documented in `sglang-dh4/bench/dh4/README.md`.

## Exact launcher recommendation (not applied)

Recommended production launcher diff: **empty (0 lines)**. Keep `SGLANG_DRAFT_SHORTLIST` unset, or retain `SGLANG_DRAFT_SHORTLIST=0` where an explicit default is desired. Do not add the DH4 PYTHONPATH overlay or selector/calibration variables to production. The experimental opt-in block remains available in the worktree README for reproducibility only; the measured policy failed the adoption gate.

## DH5

Status: COMPLETE — W16 GO for K2048 / selector margin <0.25; W4 NO-GO. Launcher recommendation is W16-only and has not been applied. User-authorized configuration/threshold follow-up in the same worktree. Start HEAD is now `3d7caf070d` (the previously delivered DH4 changes were committed externally); this task makes no commit. DH4 evidence above remains immutable.

DH4 saved aggregate counters and CUPTI timing, not per-position candidates, calibrated confidence or a full-head oracle. Exact historical loss positions cannot be reconstructed from those artifacts. DH5 adds separate untimed diagnostic control/shortlist runs with C1 position-level acceptance and a production-arithmetic full-head oracle on each exact recursive input. These diagnostics do not enter timing/gates and do not claim to recover the historical DH4 trajectories.

Preserve DH2's existing fallback signal: selector top1-top2 raw score margin, fallback iff margin < threshold. The frozen calibrated full-distribution proxy is logged separately and still feeds recursive C1 diagnostics. Derive initial thresholds from DH2 calibration split only; use untimed serving margin histograms to target about 5/10/20 percent before freezing the four A/B arms. K=2048 uses its existing DH2 calibration map; K support is extended without changing selector/rescore arithmetic.

All server lifetimes and GPU tests hold the shared flock. All server arms use memory fraction 0.920, empty SERVE_DISPLAY_HZ, common MAX_TOTAL_TOKENS=131072, read-only production mounts and frozen DH4 launchers. No startup VRAM abort, >=4096 MiB steady free, cleanup only owned process groups. W16 n=3 for four workloads plus needle; timing traces cover all four workloads with 20 steps each. Gate compares against the saved DH4 two-control mean. W4 n=2 only if a W16 configuration passes every workload.

### Starting dh5-20260908-005331-diagnostic-control

w16/diagnostic-control, repeats=1; per-arm flock acquired at 2026-09-08T00:59:47.161278+09:00.

### DH5 GPU preflight

K=1024 and K=2048 were each checked on DH2 states (private data; results not published) for production BF16-logit equality, finite margin threshold capture/replay (0.25,0.5,1,2 plus +/-infinity), mapped winner, frozen confidence, ties and 14-step conditional graph capture. K1024 is compared to the unchanged DH3 reference membership. K2048 uses the generalized partition kernel and full production GEMV equality; its `winner_differences_from_dh3_fp32` JSON key is inherited test wording and compares the serving reference, not an independent DH3 K2048 implementation.

Candidate margin thresholds were first located on the DH2 calibration split (private data) by counting strict-threshold (margin < threshold) outcomes; an offline check on the DH2 test split was also run. No result from either step is published.

dh5-20260908-005331-diagnostic-control: ready, 11538 MiB free; requested memory fraction 0.920.

dh5-20260908-005331-diagnostic-control: all benchmarks/traces completed; needle 0/0 PASS.

dh5-20260908-005331-diagnostic-control: owned process groups cleaned; flock releases on exit.

### Starting dh5-20260908-010010-diagnostic-shortlist

w16/diagnostic-shortlist, repeats=1; per-arm flock acquired at 2026-09-08T01:12:19.879069+09:00.

dh5-20260908-010010-diagnostic-shortlist: ready, 11536 MiB free; requested memory fraction 0.920.

dh5-20260908-010010-diagnostic-shortlist: all benchmarks/traces completed; needle 0/0 PASS.

dh5-20260908-010010-diagnostic-shortlist: owned process groups cleaned; flock releases on exit.

### DH5 diagnostic complete; thresholds frozen before A/B

Both untimed diagnostic arms completed all four workloads (one fnbench repeat plus warmup each; no diagnostic needle run). Oracle and C1 acceptance arrays and per-position calibrated confidence agree. Every included full-head winner matches the shortlist winner.

Code-edit has **zero full-argmax misses in 5558 recursive forwards** and its diagnostic tokens/verify is 12.224 versus control 12.135: the DH4 code-edit acceptance loss was not reproduced. Do not retrospectively attribute it to rare-token misses without historical oracle data. Agent-loop has 20/6776 misses, eight at the first rejection; its diagnostic tokens/verify is 4.988 versus 5.507. Prose-en has 53/10724 misses (six at first rejection), prose-ja 64/12614 (six at first rejection). These identify local full-head disagreements at rejection boundaries, not proven counterfactual target acceptance: no alternate chain was verified.

Freeze margin thresholds 0.125/0.25/0.4375 for K1024, yielding pooled diagnostic fallback 3.5658%/9.6350%/19.7466%. The nominal 5% point is the closest attainable strict threshold due to BF16 margin ties. K2048 uses threshold0.25, whose margin is K-independent on the same input. Timed rates may shift with the generated trajectories. All four workloads receive CUPTI profiles; diagnostics are disabled in these timing runs. See `specs/dh5/policies.json`, `positions.csv`, `*-miss-positions.json`, and `diagnostic-summary.json`.

### DH5 per-position diagnosis

Fresh untimed diagnostics only; these are not the historical DH4 request trajectories. Position0 is the target-fed full-head step. Each cell below is **shortlist minus control accepted-survival percentage points / full-winner-outside-K count / first-rejection-and-miss count**. Survival means accepted prefix length > position; each arm uses its own chain count.

| position | code-edit | prose-en | prose-ja | agent-loop |
|---:|---:|---:|---:|---:|
| 0 | +1.98 / N/A / N/A | -2.91 / N/A / N/A | +3.74 / N/A / N/A | -0.52 / N/A / N/A |
| 1 | +1.96 / 0 / 0 | -0.44 / 7 / 3 | +2.17 / 8 / 3 | -5.04 / 3 / 3 |
| 2 | +1.18 / 0 / 0 | +0.89 / 2 / 0 | +2.95 / 5 / 0 | -3.57 / 1 / 0 |
| 3 | -0.85 / 0 / 0 | +3.68 / 7 / 2 | +3.05 / 4 / 2 | -0.33 / 1 / 1 |
| 4 | +0.13 / 0 / 0 | +2.29 / 4 / 1 | +1.95 / 5 / 0 | -2.31 / 3 / 3 |
| 5 | +1.11 / 0 / 0 | +2.23 / 7 / 0 | +0.38 / 8 / 1 | -3.64 / 1 / 0 |
| 6 | +0.09 / 0 / 0 | +0.88 / 2 / 0 | +0.05 / 1 / 0 | -3.49 / 1 / 0 |
| 7 | +0.58 / 0 / 0 | +0.19 / 5 / 0 | +0.33 / 6 / 0 | -5.21 / 1 / 0 |
| 8 | +0.55 / 0 / 0 | -0.60 / 2 / 0 | +0.19 / 7 / 0 | -4.71 / 1 / 1 |
| 9 | +1.26 / 0 / 0 | +0.15 / 1 / 0 | -0.16 / 3 / 0 | -4.15 / 0 / 0 |
| 10 | +0.23 / 0 / 0 | +0.01 / 4 / 0 | -0.06 / 5 / 0 | -3.68 / 1 / 0 |
| 11 | +0.21 / 0 / 0 | +0.14 / 3 / 0 | +0.03 / 2 / 0 | -3.97 / 2 / 0 |
| 12 | -0.07 / 0 / 0 | -0.25 / 2 / 0 | +0.02 / 1 / 0 | -4.26 / 1 / 0 |
| 13 | -0.10 / 0 / 0 | +0.01 / 4 / 0 | +0.01 / 4 / 0 | -3.47 / 3 / 0 |
| 14 | +0.65 / 0 / 0 | +0.13 / 3 / 0 | +0.01 / 5 / 0 | -3.57 / 1 / 0 |

Every observed first-rejection miss is listed below. Confidence is the pre-fallback frozen full-distribution proxy of the shortlist winner; the full-head winner was outside K in every row. A full-head disagreement at rejection is a potential repair, not proof the alternate token would be accepted by the target.

| workload | chain ordinal within workload | position | calibrated confidence | selector margin | protected at thresholds .125 / .25 / .4375 |
|---|---:|---:|---:|---:|---|
| prose-en | 332 | 1 | 0.594816 | 0.7500 | no / no / no |
| prose-en | 485 | 1 | 0.331740 | 3.5000 | no / no / no |
| prose-en | 505 | 4 | 0.257198 | 2.1250 | no / no / no |
| prose-en | 590 | 3 | 0.434756 | 3.6250 | no / no / no |
| prose-en | 617 | 1 | 0.374931 | 0.3750 | no / no / yes |
| prose-en | 730 | 3 | 0.177303 | 0.8750 | no / no / no |
| prose-ja | 27 | 3 | 0.926421 | 0.5625 | no / no / no |
| prose-ja | 189 | 1 | 0.410883 | 3.0000 | no / no / no |
| prose-ja | 235 | 1 | 0.275186 | 0.1250 | no / yes / yes |
| prose-ja | 462 | 3 | 0.921221 | 0.4375 | no / no / no |
| prose-ja | 603 | 5 | 0.135198 | 0.5625 | no / no / no |
| prose-ja | 820 | 1 | 0.173003 | 1.0000 | no / no / no |
| agent-loop | 13 | 1 | 0.279719 | 0.7500 | no / no / no |
| agent-loop | 26 | 4 | 0.514876 | 4.0000 | no / no / no |
| agent-loop | 248 | 1 | 0.406293 | 0.1250 | no / yes / yes |
| agent-loop | 375 | 4 | 0.457866 | 0.1250 | no / yes / yes |
| agent-loop | 391 | 3 | 0.755554 | 0.4375 | no / no / no |
| agent-loop | 396 | 1 | 0.128969 | 0.3125 | no / no / yes |
| agent-loop | 397 | 8 | 0.315415 | 0.8750 | no / no / no |
| agent-loop | 398 | 4 | 0.225880 | 0.5000 | no / no / no |

All 137 recursive misses, including post-rejection and accepted positions, retain individual calibrated/raw probabilities, full winner, shortlist winner, full top1 probability and margin in the per-workload `*-miss-positions.json` files. `positions.csv` contains all 60 workload/position rows; `diagnostic-summary.json` includes confidence quantiles. The oracle result and C1 probability agree within C1 JSON rounding at every recursive position.

Code-edit loss was not reproduced and cannot be localized retrospectively. Agent-loop shows lower acceptance survival mainly from position1 onward, with first-rejection misses at positions1,3,4,8. Its eight local first-rejection misses have calibrated confidence0.129–0.756; thresholds .125/.25/.4375 protect0/2/3 of them on these fixed diagnostic inputs. Prose-en first-rejection misses are at positions1,3,4; prose-ja at1,3,5. Two Japanese misses have calibrated confidence above0.92, illustrating that the frozen map is not a per-position guarantee. Thresholding changes future inputs, so these fixed-input protection counts are not asserted to be the timed-arm counts.

### Starting dh5-20260908-011544-w16-k1024-f05

w16/k1024-f05, repeats=3; per-arm flock acquired at 2026-09-08T01:24:29.897775+09:00.

dh5-20260908-011544-w16-k1024-f05: ready, 11536 MiB free; requested memory fraction 0.920.

dh5-20260908-011544-w16-k1024-f05: all benchmarks/traces completed; needle 3/3 PASS.

dh5-20260908-011544-w16-k1024-f05: owned process groups cleaned; flock releases on exit.

### Starting dh5-20260908-011544-w16-k1024-f10

w16/k1024-f10, repeats=3; per-arm flock acquired at 2026-09-08T01:28:12.700279+09:00.

### DH5 first timing arm complete

K1024/margin<0.125: all throughput point estimates exceed +3%, but agent-loop acceptance is -4.048% against DH4 control mean, so this configuration fails. Observed workload fallback rates code-edit/prose-en/prose-ja/agent-loop are 1.367%/3.833%/4.251%/2.811%. Needle3/3 PASS, minimum steady free9376 MiB. Keep thresholds frozen and complete the other three configurations before selecting a trade-off or a W4 candidate.

dh5-20260908-011544-w16-k1024-f10: ready, 11536 MiB free; requested memory fraction 0.920.

### Frozen confidence map failure examples

In the oracle diagnostic, two prose-ja position3 first-rejection misses have calibrated shortlist confidence0.926421/0.921221 while the different, omitted full-head winner has probability0.966192/0.973774. Therefore the actually selected candidate's full-head probability is **at most0.033808/0.026226**, since all nonwinning tokens share the remaining mass. Agent-loop position3 similarly has calibrated0.755554 versus omitted full winner0.967742, so selected true probability is at most0.032258. These are direct bounds from the saved full-head distribution, not an inferred target acceptance probability. The frozen scalar calibration map cannot detect a dominant missing candidate from the shortlist's renormalized probability alone. See `first-rejection-probability-bounds.json`.

dh5-20260908-011544-w16-k1024-f10: all benchmarks/traces completed; needle 3/3 PASS.

dh5-20260908-011544-w16-k1024-f10: owned process groups cleaned; flock releases on exit.

### Starting dh5-20260908-011544-w16-k1024-f20

w16/k1024-f20, repeats=3; per-arm flock acquired at 2026-09-08T01:31:52.158412+09:00.

dh5-20260908-011544-w16-k1024-f20: ready, 11536 MiB free; requested memory fraction 0.920.

### DH5 second timing arm complete

K1024/margin<0.25: code-edit/prose-en/prose-ja pass the per-workload point gate, but agent-loop acceptance is -12.270% and throughput -1.980%, so this configuration fails. Needle3/3 PASS, minimum steady free9376 MiB. Observed fallback rates are 3.742%/11.103%/10.970%/7.005%. More fallback did not produce monotone serving acceptance across these separate trajectories; do not retune after this result.

dh5-20260908-011544-w16-k1024-f20: all benchmarks/traces completed; needle 3/3 PASS.

dh5-20260908-011544-w16-k1024-f20: owned process groups cleaned; flock releases on exit.

### Starting dh5-20260908-011544-w16-k2048-f10

w16/k2048-f10, repeats=3; per-arm flock acquired at 2026-09-08T01:35:34.830929+09:00.

### DH5 third timing arm complete

K1024/margin<0.4375: every workload exceeds +3% throughput, but code-edit acceptance -1.990% and agent-loop -1.360% exceed the1% loss limit. Needle3/3 PASS, minimum steady free9378 MiB. Observed fallback rates are8.961%/24.075%/22.945%/17.081%. No threshold adjustment is made; proceed to the already frozen K2048/margin<0.25 arm.

dh5-20260908-011544-w16-k2048-f10: ready, 11536 MiB free; requested memory fraction 0.920.

Quantization detail: the diagnostic fallback rate jumps from3.5658% at threshold0.125 to9.1108% at0.1875. No intermediate strict scalar margin threshold achieves5%; the nearest attainable rate was frozen rather than adding random tie-breaking or a new fallback rule.

dh5-20260908-011544-w16-k2048-f10: all benchmarks/traces completed; needle 3/3 PASS.

dh5-20260908-011544-w16-k2048-f10: owned process groups cleaned; flock releases on exit.

### DH5 W16 gate PASS for K2048 / margin<0.25

All four requested W16 configurations are complete. Only K2048/margin<0.25 passes every workload: throughput changes code-edit/prose-en/prose-ja/agent-loop are +11.820%/+8.578%/+14.842%/+10.842%, acceptance changes +3.798%/+0.551%/+6.136%/+0.482%. Needle3/3 PASS. The three K1024 configurations fail as recorded above. Start the explicitly conditional W4 n=2 follow-up for the passing configuration, using a fresh label and a separate flock lifetime. No production change has been applied.

### DH5 measured configurations

All rates below are observed, not the requested nominal fallback target. Acceptance is mean delivered tokens/verify from per-request Prometheus deltas, identical to DH4. W16 n=3; CUPTI covers all four workloads with 20 steps each. The oracle and C1 debug trace are disabled in timed arms.

| configuration | workload | tokens/verify | t/s | acceptance change | t/s change | fallback/calls | fallback % | trimmed ms | shortlist median / mean us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k1024-f05 (margin<0.125) | code-edit | 12.2955 | 726.76 | +10.344% | +18.783% | 151/11046 | 1.367 | 16.0633 | 35.808 / 39.900 |
| k1024-f05 (margin<0.125) | prose-en | 3.1491 | 203.73 | +2.917% | +12.555% | 814/21238 | 3.833 | 15.2877 | 35.552 / 41.156 |
| k1024-f05 (margin<0.125) | prose-ja | 2.5700 | 171.28 | +0.943% | +10.519% | 1117/26278 | 4.251 | 15.7040 | 35.664 / 42.815 |
| k1024-f05 (margin<0.125) | agent-loop | 5.1522 | 323.11 | -4.048% | +5.964% | 377/13412 | 2.811 | 16.0176 | 35.712 / 39.915 |
| k1024-f10 (margin<0.25) | code-edit | 11.1844 | 663.70 | +0.372% | +8.477% | 450/12026 | 3.742 | 15.5627 | 36.032 / 41.088 |
| k1024-f10 (margin<0.25) | prose-en | 3.3806 | 223.10 | +10.483% | +23.260% | 2240/20174 | 11.103 | 15.3426 | 35.952 / 50.795 |
| k1024-f10 (margin<0.25) | prose-ja | 2.6812 | 177.59 | +5.310% | +14.591% | 2789/25424 | 10.970 | 15.7366 | 35.937 / 49.028 |
| k1024-f10 (margin<0.25) | agent-loop | 4.7107 | 298.89 | -12.270% | -1.980% | 969/13832 | 7.005 | 16.0805 | 35.618 / 43.720 |
| k1024-f20 (margin<0.4375) | code-edit | 10.9212 | 647.12 | -1.990% | +5.766% | 1094/12208 | 8.961 | 15.5291 | 35.968 / 46.630 |
| k1024-f20 (margin<0.4375) | prose-en | 3.2892 | 207.89 | +7.494% | +14.857% | 5103/21196 | 24.075 | 15.3663 | 36.448 / 59.800 |
| k1024-f20 (margin<0.4375) | prose-ja | 2.5571 | 165.60 | +0.438% | +6.858% | 6068/26446 | 22.945 | 15.9239 | 36.720 / 61.859 |
| k1024-f20 (margin<0.4375) | agent-loop | 5.2965 | 329.74 | -1.360% | +8.136% | 2200/12880 | 17.081 | 15.6831 | 36.033 / 52.684 |
| k2048-f10 (margin<0.25) | code-edit | 11.5662 | 684.15 | +3.798% | +11.820% | 384/11396 | 3.370 | 15.4250 | 37.375 / 42.164 |
| k2048-f10 (margin<0.25) | prose-en | 3.0767 | 196.53 | +0.551% | +8.578% | 2564/21826 | 11.747 | 15.4542 | 37.200 / 55.845 |
| k2048-f10 (margin<0.25) | prose-ja | 2.7022 | 177.97 | +6.136% | +14.842% | 2899/25284 | 11.466 | 15.8919 | 37.168 / 49.057 |
| k2048-f10 (margin<0.25) | agent-loop | 5.3954 | 337.99 | +0.482% | +10.842% | 908/12292 | 7.387 | 15.5786 | 37.103 / 48.706 |

Fallback denominators include untimed fnbench warmup and exclude boot/profiling. Shortlist span includes executed full fallback plus IF scheduling and publication. CUPTI trimmed is the inherited kernel-name-clipped metric; physical wall and distributions are retained in analysis.json.

| configuration | gate | needle | minimum steady free MiB | label |
|---|---|---|---:|---|
| k1024-f05 | FAIL | 3/3 | 9376 | `dh5-20260908-011544-w16-k1024-f05` |
| k1024-f10 | FAIL | 3/3 | 9376 | `dh5-20260908-011544-w16-k1024-f10` |
| k1024-f20 | FAIL | 3/3 | 9378 | `dh5-20260908-011544-w16-k1024-f20` |
| k2048-f10 | PASS | 3/3 | 9378 | `dh5-20260908-011544-w16-k2048-f10` |

### Starting dh5-20260908-014025-w4-k2048-f10

w4/k2048-f10, repeats=2; per-arm flock acquired at 2026-09-08T01:48:59.331552+09:00.

dh5-20260908-014025-w4-k2048-f10: ready, 12472 MiB free; requested memory fraction 0.920.

dh5-20260908-014025-w4-k2048-f10: all benchmarks/traces completed; needle 2/2 PASS.

dh5-20260908-014025-w4-k2048-f10: owned process groups cleaned; flock releases on exit.

### DH5 selected configuration at W4

K2048, selector margin <0.25, n=2. Reference is the saved DH4 W4 control. W4 acceptance change is diagnostic; the original W4 safeguard is non-negative throughput on every workload.

| workload | tokens/verify | t/s | acceptance change | t/s change | fallback/calls | fallback % | trimmed ms | shortlist median / mean us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 3.7189 | 380.36 | -3.636% | -1.051% | 102/3818 | 2.672 | 9.4137 | 39.280 / 46.295 |
| prose-en | 2.5641 | 276.88 | +3.861% | +10.130% | 301/2830 | 10.636 | 9.1383 | 38.369 / 50.335 |
| prose-ja | 2.1412 | 232.46 | -8.175% | -4.079% | 376/3316 | 11.339 | 9.1976 | 37.744 / 43.519 |
| agent-loop | 3.2138 | 344.87 | -1.361% | +4.688% | 117/2222 | 5.266 | 9.3033 | 38.144 / 39.405 |

W4 safeguard: **FAIL**. Needle 2/2 PASS; minimum steady free 10332 MiB. Label `dh5-20260908-014025-w4-k2048-f10`.

### DH5 final interpretation and scope of GO

**W16 gate PASS: K2048 / selector margin <0.25. W4 non-negative-throughput safeguard FAIL. Recommend enabling only the fixed W16 profile.** Do not enable the policy globally or in W4/WA based on these results. No further K/threshold sweep or retuning follows this round.

The selected W16 point has throughput +11.820%/+8.578%/+14.842%/+10.842% and acceptance +3.798%/+0.551%/+6.136%/+0.482% for code-edit/prose-en/prose-ja/agent-loop. All four exceed the requested +3% speed and -1% acceptance limits. W4 code-edit and prose-ja regress in throughput (-1.051%/-4.079%), with diagnostic acceptance losses -3.636%/-8.175%; W4 is excluded from the proposal despite its passing needle.

Actual pooled fallback rates (four workloads, including their untimed warmups) are3.4165%/9.0237%/19.8886% for the three K1024 points and9.5412% for K2048. The three K1024 policies do not satisfy the quality gate: f05 agent-loop -4.048%; f10 agent-loop -12.270% and speed -1.980%; f20 code-edit -1.990% and agent-loop -1.360%. Among K1024 choices, f20 is closest to the acceptance gate, while f05 retains more speed; neither is recommended over the passing K2048 point.

K2048's W16 all-forward median spans are37.103–37.375 us, about1–1.5 us above the K1024 f10 medians. The executed-fallback median spans are121.440–121.759 us. Means include both fallback frequency and long scheduling outliers; conditional IF actually runs the full head only on its device flag. The trace's skip/fallback classification and full-reducer counts agree. K expansion plus a modest fallback budget passes this serving comparison; it does not make the scalar probability map correct on every missed-candidate position.

Historical attribution remains limited: DH4 did not retain position-level confidence, candidates, full-head oracles, or reasoning token sequences. DH5's separate oracle run localizes current misses and rejection boundaries, not the exact historical DH4 failures. Code-edit's prior loss was not reproduced in that diagnostic; agent-loop's lower survival and first-rejection misses were observed. All137 misses were recorded,20 at the first rejection and117 after rejection; none occurred in an accepted prefix. Even a miss at first rejection is a potential repair, since an alternate full-head chain was not target-verified. Output-content hashes in DH4 often hash an empty final-content string and cannot establish equality of the unretained reasoning trajectories.

Gates use the explicitly requested saved DH4 control means, not a newly substituted control. Diagnostic control numbers are excluded. W16 n=3 and W4 n=2 are descriptive point estimates; no statistical non-inferiority claim is made, and the historical controls span the previously documented desktop restart. The recommendation is restricted to the tested model, hot map, fixed W16 profile and launch settings. The first target-fed head remains full; K2048 uses DH2's frozen K2048 map for recursive C1 confidence. Adaptive width/hysteresis joint behavior is unverified and receives no GO.

### DH5 final verification

- Five timed arms:56 measured workload requests,14/14 needle PASS,20 CUPTI traces,400 draft-step annotations,4640 shortlist forwards in those traces. Additional untimed control/oracle diagnostics cover four workloads each and are excluded from gates.
- Oracle diagnostic:35672 recursive forward observations,137 full-winner misses. Every included full winner matches the shortlist winner; every recursive calibrated probability pairs with C1 within its JSON rounding. Per-position results and individual miss confidences are retained.
- CPU configuration tests4/4 PASS, including K2048 acceptance and invalid-policy rejection. GPU preflight: K1024 and K2048 checked on DH2 states (private data; results not published). K1024 uses independent unchanged DH3 reference membership; K2048's reference-membership limitation is documented above.
- Every GPU/server run held the shared flock; each server arm released it after owned-group cleanup. All fractions0.920, SERVE_DISPLAY_HZ empty, MAX_TOTAL_TOKENS131072, no startup memory abort. Minimum observed steady free VRAM across timed arms:9376 MiB. Host process audit finds no remaining process in any of the seven diagnostic/timed server groups or among their14 recorded CUDA process IDs.
- Frozen runtime/selector/calibration/bridge/arm-driver hashes match. `git diff --check` and proposed-launcher `bash -n` pass. Worktree remains `$HOME/tools/sglang-dh4`, branch `codex/dh4-shortlist-server`, HEAD `3d7caf070d1cb1269bad79a64eb7ac7b605f1257` (the external pre-DH5 commit). All DH5 changes remain uncommitted; no branch creation, commit or push.
- Production was read-only in every server mount namespace. The launcher proposal is a private text copy and diff; its generation verifies that the production launcher text is unchanged. No production source, launcher or venv file was written by DH5.

Evidence under `specs/dh5/`: `results.json`, `gate.json`, `audit.json`, `process-audit.json`, `frozen-inputs.json`, `diagnostic-summary.json`, `positions.csv`, `confidence-histograms.json`, per-workload miss records, and each labeled arm's traces/JSONL/counter snapshots. Reproduction and diagnostic flags are documented in `sglang-dh4/bench/dh4/README.md`.

### DH5 exact launcher recommendation — W16 only, NOT APPLIED

Insert the following block immediately after `PROFILE="${1:-w4}"; shift || true` in `serve-fast.sh`. The reviewable exact diff is `specs/dh5/launcher-proposal.patch`; the syntax-checked private result is `serve-fast.proposed.sh`. The runtime default remains off; this proposed W16 launcher block opts in with a `SGLANG_DRAFT_SHORTLIST=0` opt-out. W4 and adaptive profiles receive no new default.

```bash
# DH5: validated fixed-width shortlist policy; SGLANG_DRAFT_SHORTLIST=0 opts out.
case "$PROFILE" in
  w16)
    export SGLANG_DRAFT_SHORTLIST="${SGLANG_DRAFT_SHORTLIST:-1}"
    if [ "$SGLANG_DRAFT_SHORTLIST" = "1" ]; then
      export PYTHONPATH=$HOME/tools/sglang-dh4/python
      export SGLANG_DRAFT_SHORTLIST_K=2048
      export SGLANG_DRAFT_SHORTLIST_THRESHOLD=0.25
      export SGLANG_DRAFT_SHORTLIST_FALLBACK=conditional
      export SGLANG_DRAFT_SHORTLIST_WEIGHTS=$HOME/tools/mtp-train/results/dh2/checkpoints/learned_r128.safetensors
      export SGLANG_DRAFT_SHORTLIST_CALIBRATION=$HOME/tools/mtp-train/results/dh2/confidence_calibration.json
      export SGLANG_DRAFT_SHORTLIST_CONDITIONAL_LIB=$HOME/tools/sglang-dh4/bench/dh4/conditional.so
      export MEM_FRACTION=0.920 W16_MEM_FRACTION=0.920
      export MAX_TOTAL_TOKENS=131072 SERVE_DISPLAY_HZ=
    fi
    ;;
esac
```

Keep the diagnostic oracle and C1 debug trace variables unset for normal serving. Continue acquiring `$HOME/.gpu.lock` around the entire server lifetime. This block pins the memory reserve and empty display setting used in the measurements. It uses the existing production venv through a PYTHONPATH overlay; no package installation is part of the proposal. The patch has not been applied and no server is left running by this task.

## DH6 — same-session production W16 ABAB adoption confirmation

Preregistered: A1 → D1 → A2 → D2, 3 measured repeats per workload and 3 needles per arm. Four workloads: code-edit, prose-en, prose-ja, agent-loop. A explicitly imports production code from codex/perf-v1 (446c801189); D imports the unchanged DH5 working tree at codex/dh4-shortlist-server (3d7caf070d plus uncommitted DH5 changes). K=2048, strict selector margin <0.25, conditional fallback; diagnostic oracle and adaptive trace disabled. Both arms use byte-identical snapshots of the current production launcher and serve-local, memory fractions 0.920, SERVE_DISPLAY_HZ empty, MAX_TOTAL_TOKENS 131072. Production is read-only in the server mount namespace; only D uses the worktree overlay.

Gate compares the mean of D1/D2 against the mean of A1/A2, all n=3: prose-en and prose-ja throughput each >=+3%; no workload throughput below -1%; acceptance loss <=1% on every workload; all needles PASS. D1/A1 and D2/A2 are reported separately. No historical DH4/DH5 numbers enter this gate. Every arm holds the GPU lock through cleanup, no startup transient-memory abort, and >=4096 MiB free in steady state. CUPTI profiling precedes unprofiled throughput requests, using identical ordering in each arm. Arm directories, source hashes, schedule and driver are under specs/dh6/.

### Starting dh6-20260908-020435-w16-A1

w16/A1, repeats=3; per-arm flock acquired at 2026-09-08T02:06:48.628848+09:00.

dh6-20260908-020435-w16-A1: ERROR Server startup failed rc=1

dh6-20260908-020435-w16-A1: owned process groups cleaned; flock releases on exit.

### Starting dh6-20260908-020732-w16-A1

w16/A1, repeats=3; per-arm flock acquired at 2026-09-08T02:10:26.061080+09:00.

dh6-20260908-020732-w16-A1: ready, 11538 MiB free; requested memory fraction 0.920.

dh6-20260908-020732-w16-A1: all benchmarks/traces completed; needle 3/3 PASS.

dh6-20260908-020732-w16-A1: owned process groups cleaned; flock releases on exit.

DH6 source-freeze restart: the first private launcher copy lacked executable permission and failed before server launch (label020435). After that correction, A1 label020732 completed correctly, but external production changes appeared while D1 waited: WA5 commit7b4d539f9b (adaptive_confidence.py and tests), plus wa-only launcher/config defaults. The frozen-input assertion stopped D1 before launch. Although these changes affect the adaptive path, the completed A1 is excluded and the entire ABAB is restarted with a refreshed source/launcher freeze, avoiding mixed source revisions. No production edits were made by DH6. The current production baseline is 7b4d539f9bd896265f498fabccc6466e45ffe818; D remains unchanged. Prior attempts, source hashes and exclusion reasons are retained.

### Starting dh6-20260908-022304-w16-A1

w16/A1, repeats=3; per-arm flock acquired at 2026-09-08T03:00:45.074910+09:00.

dh6-20260908-022304-w16-A1: ready, 11538 MiB free; requested memory fraction 0.920.

dh6-20260908-022304-w16-A1: all benchmarks/traces completed; needle 3/3 PASS.

dh6-20260908-022304-w16-A1: owned process groups cleaned; flock releases on exit.

### Starting dh6-20260908-022304-w16-D1

w16/D1, repeats=3; per-arm flock acquired at 2026-09-08T03:38:39.450624+09:00.

dh6-20260908-022304-w16-D1: ready, 11536 MiB free; requested memory fraction 0.920.

dh6-20260908-022304-w16-D1: all benchmarks/traces completed; needle 3/3 PASS.

dh6-20260908-022304-w16-D1: owned process groups cleaned; flock releases on exit.

### DH6 interim first pair (D1/A1; final ABAB decision pending)

| workload | A1 tokens/verify | D1 tokens/verify | A1 t/s | D1 t/s | acceptance change | t/s change | D1 fallback % |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 11.9602 | 11.8505 | 663.13 | 697.68 | -0.917% | +5.211% | 4.405 |
| prose-en | 3.0645 | 3.1695 | 185.05 | 202.84 | +3.425% | +9.613% | 11.336 |
| prose-ja | 3.0295 | 2.5840 | 185.66 | 171.03 | -14.708% | -7.884% | 11.902 |
| agent-loop | 5.4096 | 5.0647 | 317.43 | 318.21 | -6.375% | +0.245% | 7.428 |

A1 and D1 each completed 12 measured requests, 4 traces, and 3/3 needle PASS. The first pair misses the quality limits on prose-ja and agent-loop; prose-ja also misses throughput. The required A2 and D2 still run before the pooled adoption gate is evaluated. D1 observed minimum steady free VRAM 9378MiB; shortlist median spans 37.136–37.344us.

### Starting dh6-20260908-022304-w16-A2

w16/A2, repeats=3; per-arm flock acquired at 2026-09-08T04:18:03.141447+09:00.

dh6-20260908-022304-w16-A2: ready, 11538 MiB free; requested memory fraction 0.920.

dh6-20260908-022304-w16-A2: all benchmarks/traces completed; needle 3/3 PASS.

dh6-20260908-022304-w16-A2: owned process groups cleaned; flock releases on exit.

### Starting dh6-20260908-022304-w16-D2

w16/D2, repeats=3; per-arm flock acquired at 2026-09-08T04:29:05.329695+09:00.

dh6-20260908-022304-w16-D2: ready, 11536 MiB free; requested memory fraction 0.920.

dh6-20260908-022304-w16-D2: all benchmarks/traces completed; needle 3/3 PASS.

dh6-20260908-022304-w16-D2: owned process groups cleaned; flock releases on exit.

### DH6 measured arms

Acceptance is the mean of each request’s completed-token / verify counter delta. Throughput is mean unprofiled client decode t/s, n=3 per workload per arm. All four workloads have a separate 20-step CUPTI trace.

| arm | workload | tokens/verify | t/s | fallback % | trimmed ms | physical wall ms | shortlist median / mean us |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | code-edit | 11.9602 | 663.13 | — | 15.9782 | 16.8912 | — |
| A1 | prose-en | 3.0645 | 185.05 | — | 15.6853 | 16.4417 | — |
| A1 | prose-ja | 3.0295 | 185.66 | — | 16.2420 | 16.9385 | — |
| A1 | agent-loop | 5.4096 | 317.43 | — | 15.9555 | 16.8418 | — |
| D1 | code-edit | 11.8505 | 697.68 | 4.405 | 15.7666 | 15.8032 | 37.344 / 42.560 |
| D1 | prose-en | 3.1695 | 202.84 | 11.336 | 15.4875 | 15.7081 | 37.296 / 51.939 |
| D1 | prose-ja | 2.5840 | 171.03 | 11.902 | 15.7977 | 16.1641 | 37.136 / 54.149 |
| D1 | agent-loop | 5.0647 | 318.21 | 7.428 | 15.4520 | 15.7034 | 37.153 / 50.578 |
| A2 | code-edit | 10.0633 | 562.88 | — | 15.8357 | 16.5819 | — |
| A2 | prose-en | 3.3303 | 200.51 | — | 15.7682 | 16.4194 | — |
| A2 | prose-ja | 2.7021 | 167.34 | — | 16.3728 | 17.1018 | — |
| A2 | agent-loop | 5.4452 | 318.97 | — | 16.0452 | 16.7261 | — |
| D2 | code-edit | 11.8167 | 694.20 | 3.601 | 15.5611 | 15.6454 | 37.184 / 45.111 |
| D2 | prose-en | 3.1986 | 203.59 | 11.381 | 15.4853 | 15.6563 | 36.929 / 53.426 |
| D2 | prose-ja | 2.5432 | 166.44 | 11.591 | 16.0644 | 16.3167 | 37.088 / 53.928 |
| D2 | agent-loop | 5.8366 | 360.70 | 6.822 | 15.6281 | 15.9642 | 36.993 / 47.027 |

Fallback denominators include the untimed workload warmup, exclude startup/profiling, and are retained with counts in results.json. Shortlist spans run from selector_down through dh4_publish and include IF scheduling and any executed full head. Trimmed time is the inherited kernel-name-clipped metric, not the physical wall time; both are reported.

### DH6 same-session gate and paired comparisons

| comparison | workload | A t/s | D t/s | t/s change | A tokens/verify | D tokens/verify | acceptance change | speed / acceptance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| D/A | code-edit | 613.00 | 695.94 | +13.530% | 11.0118 | 11.8336 | +7.464% | PASS / PASS |
| D/A | prose-en | 192.78 | 203.22 | +5.413% | 3.1974 | 3.1840 | -0.419% | PASS / PASS |
| D/A | prose-ja | 176.50 | 168.73 | -4.403% | 2.8658 | 2.5636 | -10.545% | FAIL / FAIL |
| D/A | agent-loop | 318.20 | 339.46 | +6.680% | 5.4274 | 5.4506 | +0.429% | PASS / PASS |
| D1/A1 | code-edit | 663.13 | 697.68 | +5.211% | 11.9602 | 11.8505 | -0.917% | PASS / PASS |
| D1/A1 | prose-en | 185.05 | 202.84 | +9.613% | 3.0645 | 3.1695 | +3.425% | PASS / PASS |
| D1/A1 | prose-ja | 185.66 | 171.03 | -7.884% | 3.0295 | 2.5840 | -14.708% | FAIL / FAIL |
| D1/A1 | agent-loop | 317.43 | 318.21 | +0.245% | 5.4096 | 5.0647 | -6.375% | PASS / FAIL |
| D2/A2 | code-edit | 562.88 | 694.20 | +23.330% | 10.0633 | 11.8167 | +17.424% | PASS / PASS |
| D2/A2 | prose-en | 200.51 | 203.59 | +1.536% | 3.3303 | 3.1986 | -3.957% | FAIL / FAIL |
| D2/A2 | prose-ja | 167.34 | 166.44 | -0.541% | 2.7021 | 2.5432 | -5.878% | FAIL / FAIL |
| D2/A2 | agent-loop | 318.97 | 360.70 | +13.084% | 5.4452 | 5.8366 | +7.188% | PASS / PASS |

**DH6 adoption gate: FAIL.** The decision uses only the pooled D/A comparison plus needle/sample/reserve checks. Pair results are reported individually, not substituted for the requested pooled gate. These are descriptive n=3 arm means, not a statistical non-inferiority test.

| arm | needle | minimum steady free MiB | server start–complete JST | fresh label |
|---|---|---:|---|---|
| A1 | 3/3 PASS | 9382 | 03:00:48–03:04:18 | `dh6-20260908-022304-w16-A1` |
| D1 | 3/3 PASS | 9378 | 03:38:43–03:42:15 | `dh6-20260908-022304-w16-D1` |
| A2 | 3/3 PASS | 9382 | 04:18:06–04:21:40 | `dh6-20260908-022304-w16-A2` |
| D2 | 3/3 PASS | 9378 | 04:29:08–04:32:36 | `dh6-20260908-022304-w16-D2` |

### DH6 decision and observed mechanism

**NO-GO for production adoption of configuration D. No launcher change is recommended.** Pooled prose-ja throughput is -4.403% (fails both the prose +3% target and the all-workload -1% floor), and acceptance is -10.545% (fails the -1% limit). The other three workloads pass the pooled criteria. Prose-ja acceptance fails independently in both pairs: -14.708% in D1/A1 and -5.878% in D2/A2. D1/A1 also loses agent-loop acceptance; D2/A2 also misses prose-en speed and acceptance. Those paired results are reported, while the decision uses the requested pooled gate.

The integration executes and saves head/step time. A1/A2 full-head GEMV kernel medians are 80.864–81.633us, while D's entire selector-to-publication median spans are 36.929–37.344us. D's all-forward means are 42.560–54.149us, including executed fallback and scheduling outliers; executed-fallback median spans are 121.139–122.400us. These timing windows are intentionally identified: the A number is the GEMV kernel only, and D includes selection, rescoring/confidence, IF handling, fallback when taken, and publication.

Conditional fallback is active: D1 executes 119 full-head/reducer pairs and D2 executes 109 across 2240 profiled shortlist forwards; inactive IF bodies contribute no full-head kernels. Timed-workload counter brackets (including each full-length untimed warmup) observe 7057/73150=9.6473% fallback for D1 and 6718/71358=9.4145% for D2. This is not DH4's zero-fallback integration condition. In prose-ja, mean trimmed step time still falls 16.3074→15.9311ms (-2.308%) and physical profiled wall time 17.0201→16.2404ms (-4.581%), yet delivered tokens/verify falls 2.8658→2.5636. The acceptance loss exceeds the step-time benefit and throughput falls. DH6 timed traces do not contain full-head oracles at skipped positions, so they cannot identify the exact omitted argmax tokens; no new position-level causal claim is made.

The controls themselves vary: A2/A1 t/s changes are -15.118%,+8.352%,-9.866%,+0.483% for code-edit/en/ja/agent, alongside acceptance changes. The full-head GEMV medians remain within 0.1% between corresponding control traces. Client inspection confirms fixed prompts reused for warmup and all repeats, temperature 0.0, and no label insertion into request bodies. All measured records agree on model, max_tokens, prompt length, engine and endpoint. This is a descriptive n=3-per-arm experiment with observed control variation; the saved DH4/DH5 controls are not used in its gate.

The shared lock was released after every arm and reacquired for the next arm, as requested. Other holders caused gaps between A1, D1 and A2; actual server windows are recorded above. The accepted ABAB sequence ran in one uninterrupted orchestrator session with the same frozen production revision, launchers, D source and assets. The accepted orchestrator sequence completed without an execution restart. Earlier permission-failure/source-freeze-restart attempts are retained but excluded.

### DH6 final verification and delivery

- Accepted dataset: 48 measured requests, 16 full-length untimed fnbench workload warmups, 12/12 needle PASS, 16 CUPTI traces, 320 draft-step annotations, 2240 shortlist forwards and 2240 control recursive full-head calls in traces. All four arm processes exited 0 after cleanup.
- Every server/GPU operation ran inside its per-arm `flock -w 28800 $HOME/.gpu.lock`. All memory fractions 0.920, SERVE_DISPLAY_HZ empty, MAX_TOTAL_TOKENS 131072. Server API confirms fixed 15 steps/16 draft tokens, adaptive disabled, the same model and hot map. No startup memory-abort rule was added. Minimum observed steady free VRAM: 9378MiB (>4096MiB).
- Source-origin assertions verify A uses `$HOME/tools/sglang-rtxpro6000/python` and D uses `$HOME/tools/sglang-dh4/python`. The source/launcher/asset freeze matches at final audit. Conditional head/reducer execution counts agree with fallback classification. `git diff --check` passes. DH6 introduced no runtime/kernel changes, so no new kernel-equality GPU run was substituted for the existing DH5 preflight; the in-server graph path was verified by the new traces.
- Host process audit finds no remaining process in any DH6-owned server group or among the recorded CUDA PIDs, including excluded earlier attempts. Other users' processes were not terminated. No production source, launcher or venv file was written by DH6; server namespaces mounted production read-only.
- Final baseline: production `codex/perf-v1` at `7b4d539f9bd896265f498fabccc6466e45ffe818`; D worktree `$HOME/tools/sglang-dh4`, branch `codex/dh4-shortlist-server`, HEAD `3d7caf070d1cb1269bad79a64eb7ac7b605f1257`, with the same uncommitted DH5 changes. No new branch, commit, push, cherry-pick or production merge was performed in DH6.
- The PASS-only adoption condition is unmet. No production launcher diff was generated or applied, no assets were installed, and no merge is recommended by DH6. Preparatory `merge-review.json`, `dh5-uncommitted-code.patch` and `asset-shipping-plan.json` are retained as review evidence only, explicitly not an adoption recommendation. The earlier DH5 launcher proposal remains unapplied and is not approved for adoption by this same-session test.

Evidence: `specs/dh6/results.json`, `gate.json`, `audit.json`, `process-audit.json`, `frozen-inputs.json`, `schedule.json`, `measurement-source-audit.json`, `decision.json`, the analyzer/driver scripts, and each fresh arm directory's request records, counter snapshots, traces, memory samples, server logs and event timestamps. **Stop after this failed DH6 gate; leave the production launcher unchanged.**
