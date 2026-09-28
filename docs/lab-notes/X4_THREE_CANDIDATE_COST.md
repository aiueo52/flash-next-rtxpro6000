# X4: remaining three-candidate runtime cost

Status: **Measurements COMPLETE; historical same-width penalty NOT REPRODUCED and its causal attribution remains unresolved.** All requested arms and supplementary tracing/fixed-width controls completed. No production changes, runtime fix, commits or pushes.

The post-restart three-candidate server does not reproduce WA4's slower same-width behavior: code W16 wall is 17.043 versus 17.417 ms (trimmed 16.061 versus 15.951), and prose W4 wall is 9.624 versus 10.118 ms (trimmed 9.297 versus 9.408). Full kernel symbols, grids and counts match at both widths, including with WA4 chain tracing enabled. There is no measured third-candidate multi-ms kernel, launch or graph-gap tax to fix. Extra S7 allocation is approximately 1.20 GiB, while KV/Mamba capacity remains unchanged. Short traces do not establish a causal speedup or exclude every memory-placement effect.

Historical WA4 B/C figures are client decode measurements and inferred effective step costs, not the CUPTI trimmed metric used below:

| workload | B two-state t/s | C three-state t/s | effective B / C ms | selected width |
|---|---:|---:|---:|---|
| code-edit | 578.59 | 493.69 | 20.297 / 24.540 | W16 / W16 |
| prose-en | 242.60 | 216.18 | 10.589 / 11.846 | W4 / W4 |
| agent-loop | 295.05 | 333.72 | 10.714 / 14.046 | W4 / W8 |

Those +4.243/+1.257 ms same-width historical gaps are not attributable to a named third-candidate mechanism from the X4 evidence.

Protocol: one server per `flock -w 28800 $HOME/.gpu.lock`; memory fractions 0.920, MAX_TOTAL_TOKENS=131072, SERVE_DISPLAY_HZ empty. Production tree/venv/launcher mounted read-only with the launcher's hardcoded .cache mapped to an X4-private copy from X3. FlashInfer's SGLANG_CACHE_DIR autotune files retain the existing global $HOME/.cache/sglang path (recorded in server logs); these are not a frozen per-arm snapshot. Full kernel symbols, grids and counts are compared directly rather than assuming cache equality. Steady reserve >=4096 MiB; startup transients recorded separately. Only owned process groups terminated. CUPTI 20 steps per code-edit, prose-en, agent-loop, preceded by a 1200-token warmup to settle the policy. Final selected widths must be verified from trace and switch logs, never inferred solely from config. No fnbench.

Arms: 1 production wa [3,15] with merged X3 fix; 2 WA4 overlay config c [3,7,15] with fix and STEP 7.943/0.5554; 3 same three-state allocations with slot1 warmup_batches=1000000000 (pins initial S15 without changing graph pruning or candidate union); 4 production fixed w16. Chain tracing/debug disabled for the primary X3-style comparison; WA4 tracing is a separate potential confound. Arm3 intentionally differs from a singleton candidate config, which would eliminate the unused graphs under investigation.

Evidence directory: `specs/x4/`. Per-arm traces, exact launch environments, allocation logs, analysis JSON and scripts are retained there.

Protocol correction after initial traces: the first arm2 prose trace was W8, then switched to W4 after the trace. A 1200-token warmup does not prevent a new request's early prefix from transiently promoting the policy. Initial arms 1a/2a are retained as exploratory and excluded from the final settled primary pair. `profile_settled.py` copies profile_decode2.py's CUPTI procedure, with max_tokens=1200 and workload-specific trigger at 30/200/120 streamed chunks (code/prose/agent). All subsequent arms and replacement 1/2 use this same trigger. Width histograms remain mandatory; trace only 20 steps.

## Source and CPU findings

Production and WA4 HEAD are both 446c801189165380465a73782c93658576283372. The runtime, worker and both draft graph runner files are byte-identical; only WA4 confidence policy differs. Source hashes: source-manifest.json. The requested WA4 policy is staged from WA2 commit 2671ec1b00; no new worktree is warranted until a fixable mechanism is demonstrated.

- AdaptiveController.init_states builds the sorted candidate union once. activate_step_by_batch only invokes apply_runtime_state if S changes. Settled updates do not traverse or replay unused states. apply_runtime_state drains outgoing recovery, swaps the active graph/backend references and rebuilds chain buffers on a switch.
- Initial target/draft KV, Mamba state and request reservations precede candidate graph construction. Initial S15 is the maximum in both adaptive arms. The arm1 allocation log records 131072 target tokens (K/V 0.75/0.75 GiB), draft K/V 0.06/0.06 GiB, 10 Mamba slots / 5 slots per request / max_running_requests=2, conv 0.02 GiB, SSM 0.58 GiB, intermediate conv 0.04 GiB, and QSA pending ring 3 request slots x 20 rows. These are not multiplied by candidate count.
- Every additional candidate has new attention workspaces, separate draft-extend metadata, private input buffers and graph runners. FullCudaGraphBackend uses the process-global graph memory pool, while share_input_buffer is bypassed during extra candidate capture. The private metadata prevents prior candidate graph pointers being replaced. Additional allocation can change overall VRAM availability, but the source does not relocate already allocated weights/KV/Mamba to accommodate a candidate. Physical page placement/cache effects cannot be proved absent from source or allocation totals alone.
- A second autotune guard exists in maybe_flashinfer_autotune_speculative_draft: per runner class on the shared draft model. S7 candidate draft capture can therefore see the target tactic table left by S3 target capture. This is a plausible construction-order issue, **not an established slowdown**: initial exploratory S7 draft has the same fast GEMM medians and no extra finalizer; its draft-extend is actually faster than the historical X3 b-fix trace. Initial S15 graphs were captured earlier and their full kernel symbols/counts are identical in arm1a and arm2a.
- CPU-only pinned15 check: 10000 zero-acceptance updates stay S15; candidate union [3,7,15], graph BS S3=[1], S7=[1], S15=[1,2] exactly preserved. Evidence: pin-check.txt.
- CPU policy microbenchmark (5 repetitions, 20000 observe+update calls per repetition, decisions timed separately): same-W16 three-vs-two adds ~0.43 us/update plus ~0.10 us amortized decision cost. Same-W4 adds ~0.02 us/update plus ~0.40 us amortized decision cost. This synthetic isolated timing is not a server latency estimate, but the scale cannot explain WA4's +4.243 ms W16 / +1.257 ms prose effective gaps. Evidence: cpu-policy-cost.txt and reproducible script.
- WA4 chain tracing records a per-step D2H and ConfidenceChannel.pop_chain().synchronize(); production latest_position0() only queries events. Tracing can change CPU run-ahead and confidence freshness. It must be measured separately rather than attributed to candidate allocation.

## Desktop interruption and resume

User reports system-wide desktop freeze at 23:51 JST on September 7, terminating the assistant and queued processes; desktop was restarted. On September 8 resume: no compute/server/X4-waiter processes, GPU free 93646 MiB, lock free; production hashes unchanged; WA4 still only its two staged policy files; no sglang-x4 created. Arm1a/2a/3a each have complete, cleanup and three valid gzip traces. Arm3 analysis was interrupted before writing analysis.json; report paragraphs are intact. All pending server arms were restarted with fresh September 8 labels. A fresh 1/2/3/4 set avoids treating the desktop restart as an unchanged experimental environment. Pre-freeze traces remain separate diagnostic evidence.

## Recovered pre-freeze measurements (diagnostic only)

| arm | workload | W, all 20 steps | wall ms | trimmed ms |
|---|---|---:|---:|---:|
| x4-arm1-20260907-a | code-edit | 16 | 19.105 | 16.230 |
| x4-arm1-20260907-a | prose-en | 4 | 9.486 | 9.263 |
| x4-arm1-20260907-a | agent-loop | 4 | 11.922 | 9.527 |
| x4-arm2-20260907-a | code-edit | 16 | 17.373 | 16.006 |
| x4-arm2-20260907-a | prose-en | 8 | 12.051 | 11.529 |
| x4-arm2-20260907-a | agent-loop | 8 | 12.545 | 11.811 |
| x4-arm3-20260907-a | code-edit | 16 | 18.073 | 16.142 |
| x4-arm3-20260907-a | prose-en | 16 | 17.314 | 15.827 |
| x4-arm3-20260907-a | agent-loop | 16 | 18.822 | 16.899 |

Arm2a prose is W8 and cannot be used as a same-width W4 comparison. For W16 code, all 219 full phase/kernel-symbol/grid families and their counts match arm1a exactly; wall changes 19.105 -> 17.373 ms, opposite the WA4 penalty. Fixed-shape dense medians remain near-identical while duration tails change: verify WMMA [8,3,20], 240 calls in each, median 10.592/10.560 us but maximum 1069.347/12.576 us. Verify W8A16 GEMV [256,1,1], 720 calls in each, median 30.144/30.048 us but maximum 1064.419/566.626 us. These are measured long-duration tails, not proof identifying a specific external preemptor. CUPTI kernel spans can include stalls and cannot by themselves establish sustained extra arithmetic. Evidence: kernel-signatures.json, prefreeze-duration-tails.json, exploratory-diff.txt.

Auxiliary fixed-W4/W8 controls are configured before execution with MAMBA_SLOTS=10 and SGLANG_SHARED_GATEUP_FUSED=0, matching the adaptive and fixed-W16 envelopes. Otherwise serve-fast.sh w4 uses 24 Mamba slots and both w4/w8 enable shared gate/up fusion; retaining those defaults would confound a width-only ratio. These are environment overrides on the unmodified launcher, not production edits. Historical X3 fixed-profile ratios retain their original differing profile defaults and are labeled separately.

## Post-restart primary pair (complete)

Both code-edit W16 and prose-en W4 have exactly 20 selected-width steps in both arms. Each has 219 distinct full phase/kernel-symbol/grid families, zero added/removed families and zero count changes. Code GPU events/phase: draft 675, verify 1365, draft-extend 102. Prose: 115/1365/102. The two-vs-three negative runtime difference is not an added-launch problem.

| workload | two candidates wall / trimmed ms | three candidates wall / trimmed ms | wall delta | trimmed delta |
|---|---:|---:|---:|---:|
| code-edit, W16 | 17.417 / 15.951 | 17.043 / 16.061 | -2.15% | +0.69% |
| prose-en, W4 | 10.118 / 9.408 | 9.624 / 9.297 | -4.88% | -1.18% |
| agent-loop, W4 -> W8 | 10.084 / 9.401 | 12.236 / 11.515 | ratio 1.2134x | ratio 1.2248x |

These are short-window observations, not significance tests or a causal speedup claim. The observed invariant penalty from the WA4 client measurements does not reproduce in this pair.

Supplementary physical-GPU boundary check: exclusive_gpu.py preserves the existing CPU launch-to-phase attribution but bounds the measurement from each draft graph's first GPU kernel to the next. This eliminates profiler/run-ahead leading edges from the interval selection; raw traces are unchanged. All 20 draft replays are found, yielding 19 physical intervals. GPU-aligned median W16: 17.679 -> 17.029 ms; W4 prose: 10.174 -> 9.524 ms; agent W8/W4=1.2204x. The conclusion is unchanged by this alternative boundary. Files gpu-aligned.json and primary-host-gaps.json retain details. Mean partitions sum exactly; independent medians and legacy per-name clipping must not be added as a physical partition.

Primary W8/W4 stage accounting (agent-loop, untraced): legacy trimmed draft 0.518 -> 1.528 ms, verify 8.446 -> 9.519 ms, draft-extend 0.438 -> 0.467 ms. Four additional draft forwards (2 -> 6) and the wider target verify explain the approximately 2.113 ms / 1.2248x normal width cost. This does not estimate the historical client-derived effective ratio, whose workload trajectory, tracing and measurement definition differ. The auxiliary fixed and trace-on comparisons below test those distinctions.

## Fixed W16 floor (complete)

The three-state W16 pin differs from fixed W16 by +0.224/+0.164/-0.167 ms wall for code/prose/agent (+1.34/+0.96/-0.98%). Fixed and adaptive draft/verify phase counts are identical (675/1365). Adaptive draft-extend has 102 events versus fixed 89: 13 additional GPU events (12 kernels and one D2H copy) for the confidence probability, including two max reductions, a sum reduction, cast/copy, exp/log and small pointwise operations. The 12 kernels' sum of per-kernel median durations is 32.128 us/step in the pinned-code trace; this raw sum is not an exclusive-time saving estimate. They occur in BOTH two- and three-candidate servers and do not explain the WA4 third-candidate penalty. Full symbols/counts are in fixed-floor-extra-kernels.json. No consistent multims floor penalty is observed across the three W16 workloads.

## Named kernels and graph/host attribution

The following medians are per full kernel symbol and launch grid, using all 20 phase occurrences. Exclusive deltas use the 19 CPU-boundary intervals of `exclusive_time.py`; overlapping GPU execution is split into exclusive and shared time. Consequently median-duration deltas and exclusive-time deltas need not have the same sign. No rows here represent added kernels. `kernel-signatures.json` retains the full, unshortened symbols and counts.

| width / workload | phase / kernel | grid | median two us | median three us | delta exclusive us/step |
|---|---|---|---:|---:|---:|
| code-edit | verify / cutlass_moe_grouped_gemm1 | [1,188,1] | 61.184 | 65.216 | +100.115 |
| code-edit | verify / cutlass_moe_grouped_gemm2 | [1,188,1] | 34.272 | 36.576 | -14.518 |
| code-edit | verify / _w8a16_gemv_kernel | [1940,1,1] | 405.344 | 406.273 | -74.380 |
| code-edit | draft / _w8a16_gemv_kernel | [1536,1,1] | 81.633 | 81.824 | -194.264 |
| code-edit | verify / flashinfer gdn_decode_bf16_wy_output_only | [1,48,1] | 4.512 | 4.480 | -3.835 |
| code-edit | verify / _causal_conv1d_update_chain_kernel | [1,160,4] | 3.264 | 3.216 | -47.696 |
| prose-en | verify / cutlass_moe_grouped_gemm1 | [1,188,1] | 27.072 | 26.992 | +20.643 |
| prose-en | verify / cutlass_moe_grouped_gemm2 | [1,188,1] | 16.928 | 16.960 | -26.992 |
| prose-en | verify / _w8a16_gemv_kernel | [1940,1,1] | 399.649 | 400.241 | -39.118 |
| prose-en | draft / _w8a16_gemv_kernel | [1536,1,1] | 80.800 | 81.041 | -13.305 |
| prose-en | verify / flashinfer gdn_decode_bf16_wy_output_only | [1,48,1] | 4.224 | 4.224 | +20.631 |
| prose-en | verify / _causal_conv1d_update_chain_kernel | [1,160,1] | 2.432 | 2.432 | +39.064 |

Both adaptive arms issue nominally four graph replays per step, including recovery; unused candidate graphs do not replay. Fractional API counts in the raw table arise from the 19-interval boundary and recovery overlap. Code median total GPU-idle time is 106.8 -> 88.6 us/step, prose 80.4 -> 83.8 us. The CPU launch-to-first-kernel latency is **queued work**, not automatically idle time: draft code 15.979 -> 15.583 ms, verify 19.241 -> 18.693 ms and extend 30.258 -> 29.235 ms. A CPU launch can precede the completion of earlier GPU graphs by multiple phases. Actual no-kernel gaps and their API overlaps were inspected, rather than assigning all launch latency to host overhead.

CPU-boundary prose has a leading 7.6/8.2 ms gap in the two arms due to run-ahead/trace boundaries. The supplemental physical-GPU intervals remove this leading boundary sensitivity. Their mean three-minus-two partitions are:

| workload | wall delta ms | exclusive delta ms | shared delta ms | no-kernel gap delta ms |
|---|---:|---:|---:|---:|
| code-edit W16 | -0.618 | -0.466 | -0.070 | -0.081 |
| prose-en W4 | -0.364 | -0.115 | -0.277 | +0.029 |

These mean partitions account for the difference, with rounding, and show no multi-ms additional host hole. The changing durations of existing GEMMs, overlap and duration tails explain the **observed X4 differences**; their microscopic source (routing, cache or external scheduling) is not isolated. In particular, the code MoE GEMM1 median is 4.032 us slower in the three-state arm, but this is offset elsewhere and does not produce the historical +4.243 ms effective cost. There is no return of the X3 extra-finalizer kernel pattern.

## WA4 chain-trace control

WA4 enabled chain tracing; the primary pair uses the X3 profiling settings without it. To test this distinction, both candidate sets were rerun with `SGLANG_ADAPTIVE_TRACE` and debug enabled, at the same final widths.

| workload | two-state traced wall / trimmed ms | three-state traced wall / trimmed ms |
|---|---:|---:|
| code-edit W16 | 16.949 / 15.910 | 16.819 / 16.047 |
| prose-en W4 | 9.550 / 9.354 | 9.564 / 9.327 |
| agent-loop W4 -> W8 | 9.791 / 9.559 | 12.472 / 12.070 |

The same-width phase/symbol/grid/count maps again match exactly (219 families). Chain tracing adds two GPU events per step and changes `cudaEventSynchronize` from one to two calls. The second call is already satisfied: medians are 0.661/0.561/0.581 us in the two-state arm and 0.801/0.601/0.641 us in the three-state arm (code/prose/agent). It does not expose a hidden multi-ms wait. Evidence: `trace-sync-pairs.json`, `traced-candidates-diff.txt`.

The traced three-state agent sample is 0.236 ms slower in wall and 0.556 ms slower in trimmed time than the untraced three-state sample. Existing target `cutlass_moe_grouped_gemm1` median rises 40.42 -> 46.29 us, with +0.274 ms/step exclusive time; GEMM2 rises 23.98 -> 26.94 us, +0.041 ms exclusive. Launch counts are unchanged. Other traced workloads are flat or faster. This is evidence of variable existing-kernel cost, not proof that tracing itself causes that GPU-duration change: the runs were separated by shared-lock users, and confidence timing can change request trajectories. It also does not reproduce the same-width two-versus-three penalty.

## Fix decision

No runtime fix was implemented. The conditional requirement to create `sglang-x4` / `codex/x4-three-candidate-cost`, cherry-pick the WA4 policy, CPU-unit-test a change and re-trace that change was not activated: no reproducible same-width third-candidate regression or small demonstrated fix emerged. No new worktree/branch, commit or push was made. Fix effect: **N/A, not measured**; the faster three-state samples are not presented as a fix result.

The requested same-width trace comparisons, graph/host-gap inspection, allocation/slot review and CPU pin/policy checks were performed. The original WA4 causal attribution remains **unresolved / not reproduced**, rather than assigned to an unverified memory-placement theory. Establishing that historical cause would require reproducing its full client request trajectories with contemporaneous CUPTI evidence; WA4 throughput and effective-step estimates alone cannot identify a kernel or host stall. This X4 task intentionally does not rerun fnbench or claim a new production acceptance gate.

## Reproduction and interpretation

- Prior basis: [WA4_LOG.md](WA4_LOG.md), [X3_ADAPTIVE_COST.md](X3_ADAPTIVE_COST.md), [WA2_LOG.md](WA2_LOG.md) Iteration 3, [C1_LOG.md](C1_LOG.md). The overlay was checked by actual Python module origin, not only by the environment string.
- Exact per-arm command/environment: `x4/<label>/command.json`. Each directory was newly created, and `run_arm.py` refuses to reuse an existing label. `resume_sequence.py` wraps each complete server lifetime in the shared flock, releasing between arms. It is a record of the executed labels; use fresh labels for any reproduction.
- CUPTI capture: `prof/profile_decode2.py` procedure through `x4/profile_settled.py`; 1200-token warmup, followed by a separate profiled request with the late trigger described above. All prompts come from the existing profiling workload definitions. Timing uses GPU events, not streamed client chunks as token counts.
- `prof/trimmed_step.py`: legacy trimmed cost is the sum of per-kernel-name medians times median counts per phase, including GPU memcpy/memset events. It is useful for X3 comparison but is not a physically exclusive latency partition. `wall`, GPU-busy and idle columns are separate medians and do not necessarily sum. Every 20-step trace yields 19 inter-step intervals.
- `prof/exclusive_time.py` via `x4/analyze.py` supplies per-kernel exclusive contributions, shared overlap, no-kernel gaps, runtime API overlap and graph launch timing. `x4/exclusive_gpu.py` adds the alternate GPU-boundary check without modifying traces. `x4/kernel_signatures.py` checks all 20 full phase occurrences, avoiding clipped first/last CPU-window counts.
- CPU microbenchmark/pin evidence is in `cpu-policy-cost.txt` and `pin-check.txt`; no new runtime implementation exists to unit-test. Host microbenchmark numbers are an order-of-magnitude estimate for isolated policy calculation, not an end-to-end server benchmark.
- The allocation review includes `adaptive_runtime_state.py` (`AdaptiveController`, `init_states`, `activate_step_by_batch`), `eagle_worker_v2.py` (`apply_runtime_state`), both EAGLE CUDA graph runners, and the graph pool helper. `SGLANG_ENABLE_GRAPH_POOL_BORROW` remains disabled; candidate buffers/graphs retain distinct inputs while using the global CUDA graph pool. Detailed source hashes are in `source-manifest.json`.

Limitations: three workloads and 20 selected steps per trace are not confidence intervals over repeated randomized server pairs. Lock contention separated arms in wall-clock time; the desktop environment was restarted between exploratory and primary runs. Global FlashInfer caches were not frozen, and physical GPU page placement was not instrumented. Power/thermal flags are sampled at 100 ms, not continuously; CUPTI kernel duration cannot uniquely distinguish arithmetic from stalls. There are no contemporaneous CUPTI traces for the original WA4 B/C windows in this investigation. Therefore the negative reproduction result neither identifies a historical external preemptor nor proves an invariant absence of a candidate-count penalty under every request trajectory.

## W8/W4 effective ratio

| agent-loop comparison | W4 wall ms | W8 wall ms | wall ratio | trimmed ratio | GPU-boundary wall ratio |
|---|---:|---:|---:|---:|---:|
| adaptive untraced | 10.084 | 12.236 | 1.2134x | 1.2248x | 1.2204x |
| adaptive chain trace | 9.791 | 12.472 | 1.2738x | 1.2627x | 1.2816x |
| fixed profiles | 9.621 | 12.289 | 1.2772x | 1.2755x | 1.2833x |

The new fixed-profile ratio is **1.2755x trimmed / 1.2772x wall**, also above the historical 1.2312x fit. Fixed W8 target MoE GEMM1 is 44.72 us versus 40.42 us in untraced adaptive W8, accounting for +0.256 ms/step exclusive duration; GEMM2 is 26.08 versus 23.98 us, +0.048 ms. Target phase counts are unchanged. Fixed target trimmed time is 9.850 ms versus adaptive 9.519 ms, while draft is 1.531 versus 1.528 ms. Thus an above-fit width ratio occurs without adaptive candidate allocation or tracing, and is not diagnostic of a third-candidate tax. These separated samples still cannot establish what caused the original WA4 1.311x result. Evidence: `adaptive-fixed-w8-diff.txt`.

Historical WA4 C/B effective ratio is **1.311x**, versus the refit fixed-width model's **1.2312x**. X4's untraced adaptive trimmed ratio is **1.2248x**. Four additional draft forwards and wider target verification explain its normal width scaling; no common demonstrated third-candidate mechanism explains a historical residual above that scaling. The chain-traced sample reaches 1.2627x trimmed / 1.2738x wall through slower existing target GEMM durations, with no additional target kernel counts and a sub-microsecond extra synchronization. This does not reproduce or causally explain the full 1.311x client ratio.

The fixed W4/W8 controls use ten Mamba slots and shared gate/up fusion disabled, matching the adaptive envelope. They are separate server runs; they are not the original launcher-default profile calibration. Fixed W4/W8 have four/twelve fewer `memcpy32_post` draft kernels per step than adaptive W4/W8 (0.992 us median each), in addition to the common confidence overhead discussed above. These small copies are not a third-candidate ms-scale mechanism. The full fixed/adaptive symbol differences and stage medians are retained with the evidence; any ratio difference must not silently be treated as identical graph layouts.

## Full post-restart measurement tables

| arm | workload | selected W (count) | wall ms | busy ms | idle ms | trimmed ms | draft/verify/extend GPU events |
|---|---|---|---:|---:|---:|---:|---|
| 1 | code-edit | {'16': 20} | 17.417 | 17.274 | 0.107 | 15.951 | 675/1365/102 |
| 1 | prose-en | {'4': 20} | 10.118 | 10.031 | 0.080 | 9.408 | 115/1365/102 |
| 1 | agent-loop | {'4': 20} | 10.084 | 10.007 | 0.082 | 9.401 | 115/1365/102 |
| 2 | code-edit | {'16': 20} | 17.043 | 16.963 | 0.089 | 16.061 | 675/1365/102 |
| 2 | prose-en | {'4': 20} | 9.624 | 9.472 | 0.084 | 9.297 | 115/1365/102 |
| 2 | agent-loop | {'8': 20} | 12.236 | 12.145 | 0.092 | 11.515 | 311/1365/102 |
| 3 | code-edit | {'16': 20} | 16.980 | 16.865 | 0.108 | 16.108 | 675/1365/102 |
| 3 | prose-en | {'16': 20} | 17.173 | 17.033 | 0.134 | 16.193 | 675/1365/102 |
| 3 | agent-loop | {'16': 20} | 16.885 | 16.759 | 0.109 | 15.932 | 675/1365/102 |
| 4 | code-edit | {'16': 20} | 16.757 | 16.557 | 0.099 | 15.846 | 675/1365/89 |
| 4 | prose-en | {'16': 20} | 17.009 | 16.909 | 0.119 | 16.189 | 675/1365/89 |
| 4 | agent-loop | {'16': 20} | 17.051 | 16.924 | 0.125 | 16.018 | 675/1365/89 |
| 1trace | code-edit | {'16': 20} | 16.949 | 16.699 | 0.114 | 15.910 | 676/1365/103 |
| 1trace | prose-en | {'4': 20} | 9.550 | 9.458 | 0.088 | 9.354 | 116/1365/103 |
| 1trace | agent-loop | {'4': 20} | 9.791 | 9.664 | 0.088 | 9.559 | 116/1365/103 |
| 2trace | code-edit | {'16': 20} | 16.819 | 16.683 | 0.108 | 16.047 | 676/1365/103 |
| 2trace | prose-en | {'4': 20} | 9.564 | 9.468 | 0.085 | 9.327 | 116/1365/103 |
| 2trace | agent-loop | {'8': 20} | 12.472 | 12.378 | 0.094 | 12.070 | 312/1365/103 |
| w4 | code-edit | {'4': 20} | 9.705 | 9.562 | 0.079 | 9.498 | 111/1365/89 |
| w4 | prose-en | {'4': 20} | 9.590 | 9.438 | 0.079 | 9.307 | 111/1365/89 |
| w4 | agent-loop | {'4': 20} | 9.621 | 9.441 | 0.080 | 9.290 | 111/1365/89 |
| w8 | code-edit | {'8': 20} | 12.381 | 12.271 | 0.077 | 11.868 | 299/1365/89 |
| w8 | prose-en | {'8': 20} | 12.025 | 11.934 | 0.090 | 11.640 | 299/1365/89 |
| w8 | agent-loop | {'8': 20} | 12.289 | 12.210 | 0.085 | 11.850 | 299/1365/89 |

| arm | ready MiB | steady min MiB (2 s) | steady median MiB (2 s) | occupied seconds incl startup |
|---|---:|---:|---:|---:|
| 1 | 8617 | 7181 | 7185.5 | 157.2 |
| 2 | 7051 | 5491 | 5598.0 | 150.5 |
| 3 | 9596 | 8160 | 8162.0 | 158.0 |
| 4 | 11538 | 10102 | 10102.0 | 140.3 |
| 1trace | 10856 | 9420 | 9420.0 | 138.5 |
| 2trace | 9624 | 8186 | 8190.0 | 146.6 |
| w4 | 13342 | 11906 | 11910.0 | 133.0 |
| w8 | 12768 | 11352 | 11354.0 | 133.8 |

| comparison | workload | wall delta ms (%) | trimmed delta ms (%) | mean exclusive delta ms | mean shared delta ms | mean no-kernel gap delta ms |
|---|---|---:|---:|---:|---:|---:|
| 2 minus 1 | code-edit | -0.374 (-2.15%) | +0.110 (+0.69%) | -0.422 | -0.065 | -0.078 |
| 2 minus 1 | prose-en | -0.494 (-4.88%) | -0.111 (-1.18%) | -0.073 | -0.273 | +0.062 |
| 3 minus 4 | code-edit | +0.224 (+1.34%) | +0.261 (+1.65%) | +0.213 | -0.003 | -0.014 |
| 3 minus 4 | prose-en | +0.164 (+0.96%) | +0.003 (+0.02%) | +0.256 | -0.058 | +0.073 |
| 3 minus 4 | agent-loop | -0.167 (-0.98%) | -0.085 (-0.53%) | -0.533 | +0.055 | -0.039 |
| 1trace minus 1 | code-edit | -0.468 (-2.69%) | -0.041 (-0.26%) | -0.575 | -0.072 | -0.040 |
| 1trace minus 1 | prose-en | -0.569 (-5.62%) | -0.053 (-0.57%) | -0.251 | -0.295 | +0.026 |
| 1trace minus 1 | agent-loop | -0.292 (-2.90%) | +0.158 (+1.68%) | -0.268 | -0.137 | +0.031 |
| 2trace minus 2 | code-edit | -0.223 (-1.31%) | -0.014 (-0.09%) | -0.051 | -0.058 | +0.011 |
| 2trace minus 2 | prose-en | -0.059 (-0.62%) | +0.030 (+0.33%) | -0.276 | +0.080 | -0.046 |
| 2trace minus 2 | agent-loop | +0.236 (+1.93%) | +0.556 (+4.82%) | +0.214 | +0.026 | -0.083 |
| 2trace minus 1trace | code-edit | -0.130 (-0.77%) | +0.137 (+0.86%) | +0.102 | -0.051 | -0.027 |
| 2trace minus 1trace | prose-en | +0.015 (+0.16%) | -0.027 (-0.29%) | -0.098 | +0.101 | -0.010 |

| arm | workload | graph launches/step | event sync/step | graph-launch CPU us/step | sync CPU us/step |
|---|---|---:|---:|---:|---:|
| 1 | code-edit | 4.053 | 1.000 | 1511.5 | 11596.7 |
| 1 | prose-en | 4.000 | 1.000 | 895.6 | 6467.0 |
| 1 | agent-loop | 4.000 | 1.000 | 916.4 | 6426.7 |
| 2 | code-edit | 4.105 | 1.000 | 1243.5 | 11527.0 |
| 2 | prose-en | 4.000 | 1.000 | 864.3 | 6081.6 |
| 2 | agent-loop | 4.053 | 1.000 | 1044.7 | 8202.4 |
| 3 | code-edit | 4.158 | 1.000 | 1190.6 | 11722.3 |
| 3 | prose-en | 4.000 | 1.000 | 1027.1 | 13262.4 |
| 3 | agent-loop | 4.474 | 1.000 | 1047.1 | 12848.9 |
| 4 | code-edit | 4.105 | 1.000 | 1039.5 | 12157.1 |
| 4 | prose-en | 4.474 | 1.000 | 1074.5 | 12963.7 |
| 4 | agent-loop | 4.263 | 1.000 | 1050.7 | 13538.8 |
| 1trace | code-edit | 4.158 | 2.000 | 1087.1 | 12081.2 |
| 1trace | prose-en | 4.000 | 2.000 | 731.3 | 6287.4 |
| 1trace | agent-loop | 4.000 | 2.000 | 721.4 | 6478.7 |
| 2trace | code-edit | 4.105 | 2.000 | 1391.7 | 11081.9 |
| 2trace | prose-en | 4.263 | 2.000 | 717.9 | 6185.9 |
| 2trace | agent-loop | 4.368 | 2.000 | 807.6 | 8695.7 |
| w4 | code-edit | 4.000 | 1.000 | 736.1 | 6272.2 |
| w4 | prose-en | 4.000 | 1.000 | 703.3 | 6530.1 |
| w4 | agent-loop | 4.000 | 1.000 | 690.7 | 6561.9 |
| w8 | code-edit | 4.105 | 1.000 | 853.1 | 8518.3 |
| w8 | prose-en | 4.000 | 1.000 | 803.2 | 8849.7 |
| w8 | agent-loop | 4.211 | 1.000 | 802.2 | 9005.0 |

## Completeness and resource accounting

33 valid CUPTI traces: 24 fresh post-restart traces (eight server runs, three workloads each), plus nine retained pre-freeze diagnostic traces. Every trace contains 20 draft phase annotations and 19 inter-step intervals. All fresh primary/control selected widths pass the expected histograms. The exploratory pre-freeze arm2 prose W8 trace is explicitly excluded from same-width W4 conclusions. All 11 runs have complete markers and successful owned-group cleanup. GPU run lifetimes including startup and cleanup total **0.469 hours**; post-restart runs total 0.340 hours. Shared-lock queue time is excluded.

| server label | traces | ready free MiB | steady min free MiB (2 s + 100 ms samples) | run seconds incl cleanup |
|---|---:|---:|---:|---:|
| x4-arm1-20260907-a | 3 | 8527 | 6609 | 148.8 |
| x4-arm2-20260907-a | 3 | 7317 | 5840 | 152.7 |
| x4-arm3-20260907-a | 3 | 7309 | 5839 | 164.5 |
| x4-20260908-1 | 3 | 8617 | 7181 | 165.2 |
| x4-20260908-2 | 3 | 7051 | 5491 | 158.6 |
| x4-20260908-3 | 3 | 9596 | 8160 | 166.0 |
| x4-20260908-4 | 3 | 11538 | 9998 | 148.3 |
| x4-20260908-1trace | 3 | 10856 | 9420 | 146.6 |
| x4-20260908-2trace | 3 | 9624 | 8186 | 154.8 |
| x4-20260908-w4 | 3 | 13342 | 11906 | 141.0 |
| x4-20260908-w8 | 3 | 12768 | 11352 | 141.9 |

Minimum recorded steady reserve: **5491 MiB**, above the 4096 MiB requirement. Startup samples are separate. Trace-aligned 100 ms telemetry records power/thermal cap active in 0 of the 11 runs' decode windows; this sampling does not exclude shorter events. All run environments verify fraction 0.920, MAX_TOTAL_TOKENS=131072, empty SERVE_DISPLAY_HZ and the X3 flag enabled.

Production launcher/runtime hashes, HEAD and status match the initial snapshot. WA4 still has only its original two staged policy files on the existing branch; no X4 branch/worktree exists. The venv was mounted read-only for every server and no installation was performed. No fnbench, production gate, long-context saturation or multi-request throughput test was run. Evidence audit: [completeness.json](x4/completeness.json); complete command/environment, memory, power, allocation, trace and cleanup artifacts are under each listed label. The remaining unresolved item is historical causality, not an interrupted or omitted required trace arm.

Final process inspection: all 11 recorded X4 server process groups have zero remaining processes. Another shared-GPU job was present after X4 released the lock and was left untouched. Evidence: [process-cleanup-check.json](x4/process-cleanup-check.json).
