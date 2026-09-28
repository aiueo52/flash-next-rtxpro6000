# X3 adaptive runtime step cost and power diagnostic

Status: COMPLETE — 2026-09-07. All four requested arms plus the flag-gated fix re-trace completed; 15 CUPTI traces, 49 CPU tests passed. Production unchanged; no commits or pushes. Created before GPU execution and updated as results landed.

**Result:** candidate target graph capture skipped autotune after the draft cache replaced FlashInfer's loaded tactic table. Opt-in target warmup removes 48 finalize kernels/step and reduces pinned-W8 wall by 9.14–17.08% across workloads. No limiting reason was observed in 39 decode-window power samples; transient power capping before the first trace and the 100 ms resolution limit are reported separately.

## Scope and protocol

Compare fixed W8 (S=7,T=8), confidence-adaptive pinned S=7, production wa ([3,15]), and fixed W4, using `prof/profile_decode2.py` with 20 steps on code-edit, prose-en, and agent-loop. No fnbench. Each server run independently holds `flock -w 28800 $HOME/.gpu.lock`; release between runs. All comparison arms use MEM_FRACTION=W16_MEM_FRACTION=WA_MEM_FRACTION=0.920 and empty SERVE_DISPLAY_HZ. Require >=4096 MiB free VRAM at steady state; loading transients are not an abort condition. Only owned process groups are terminated.

Production source, launchers, and venv must remain unchanged. Any fix belongs in `$HOME/tools/sglang-x3`, branch `codex/x3-adaptive-cost` from `codex/perf-v1`, served through a PYTHONPATH overlay. No commits or pushes.

## Prior evidence and interpretation

WA2_LOG Iteration 3 reports agent effective W8/W4 costs 1.448x/1.459x from acceptance divided by client throughput, not CUPTI measurements. WIDTH_SWEEP_0907 reports fixed-width trimmed times. These differ in measurement definition, instrumentation, run time, and runtime configuration; X3 will isolate the pinned-runtime comparison and report that limitation explicitly.

## Results

The entries below record results as they landed. Final comparison tables, power verdict, and completeness are at the end.

## Registered implementation details before measurement

- Slot `1` alone is changed to `[7]` in `specs/x3/pinned7.json`; slot `2` remains `[15]`. Candidate union is `[7,15]`. `AdaptiveController.validate_candidates` rejects launch S=7 because slot 2 exceeds it. Therefore arm (b) retains launcher initial S=15,T=16, then activates S=7,T=8 for bs=1. No `--speculative-num-steps 7` override on (b).
- `wa` disables `SGLANG_SHARED_GATEUP_FUSED` through `W16_GATEUP=0`, while fixed `w4` enables it. The requested arms preserve this difference, which must not be misattributed to adaptive bookkeeping.
- All arms additionally set `MAX_TOTAL_TOKENS=131072` identically. Prior WA3 at fraction 0.920 fell below the current strict 4 GiB reserve; reducing the automatically oversized KV pool reserves headroom without changing the requested memory fractions. These short prompts plus 600 output tokens fit well below the common cap. Full default KV capacity and long-context performance are not tested.
- Production is bind-mounted read-only during server execution; only its hardcoded `.cache` mount maps to an X3-private copy. Python bytecode writes disabled. Original launchers and production venv are used.
- No adaptive chain tracing/debug logging is enabled in these production-style arms. WA3 enabled both, including `ConfidenceChannel.pop_chain().synchronize()`; that difference is tracked separately from runtime state overhead.
- Power sampler is the requested nvidia-smi query at `-lms 100`, with timestamp and memory.free added for alignment/reserve verification. GPU runs require execution outside the tool sandbox because its driver is unreachable there.

### Starting x3-20260907-221750-a

Arm `a` at 2026-09-07T22:17:52.588011+09:00; per-run flock held.

## CPU/code-path findings before trace attribution

`specs/x3/pinned-config-check.txt` verifies against the actual production modules: initial S=15, union [7,15], bs1 current S=7, bs2 current S=15; 100 confidence/acceptance updates keep bs1 at S=7. Launch S=7 is explicitly rejected.

The decode worker calls `activate_step_by_batch` before drafting. `AdaptiveController` applies a runtime state only if the requested S differs from the active S. On a switch, `apply_runtime_state` drains pending GDN recovery, replaces draft/target/draft-extend attention backends and graph runners, updates worker S/T, and updates the runtime context. Thus switch-only work cannot explain a per-step penalty in a settled singleton slot without contrary trace evidence. Candidate graph capture temporarily overrides both S and T, and draft-extend replay explicitly requires its batch width to equal captured width; it is not intentionally replaying T=16 for a settled T=8 batch.

Confidence `top1_prob` is eager (`float`, amax, logsumexp, exp) after draft-extend, followed by an asynchronous pinned-host copy. Its consumer polls CUDA events without synchronization. Policy updates consume already-host acceptance results in the scheduler result processor. Chain tracing is different: `pop_chain` explicitly synchronizes its pending event, and was enabled in WA3 but is disabled by production defaults and X3.

Power semantics follow [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html): SW Power Cap means power scaling reduced requested clocks; SW Thermal Slowdown means thermal capping reduced clocks. Clock-event counters accumulate microseconds under each reason. A 100 ms sample cannot temporally resolve every 10–20 ms step; report trace-aligned sample counts and avoid asserting that absent samples prove absence of every short event. Full `nvidia-smi -q` snapshots are retained for supported counter checks.

x3-20260907-221750-a: ready free VRAM 10710 MiB. Initial S=7.

x3-20260907-221750-a / code-edit:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-221750-a/code-edit/1788787257.948267-TP-0.trace.json.gz
  step wall= 13.72ms  gpu_busy= 13.65ms  gpu_idle_in_step= 0.15ms
  draft                        occ= 20 kernels/step=    293 busy_ms=  1.87 raw_ms=  2.11 legacy_trimmed_ms=  1.56
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1317 busy_ms= 11.15 raw_ms= 13.05 legacy_trimmed_ms=  9.75
  draft_extend                 occ= 20 kernels/step=     88 busy_ms=  0.46 raw_ms=  0.53 legacy_trimmed_ms=  0.46
  TOTAL                        kernels/step=   1698 busy_ms= 13.48 wall_ms= 13.72 idle_ms= 0.15 legacy_trimmed_ms= 11.77
  (check)                      sum(busy)+idle= 13.63ms vs wall= 13.72ms  residual= 0.09ms (86 unattributed kernels/step)
```

x3-20260907-221750-a / prose-en:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-221750-a/prose-en/1788787266.1893673-TP-0.trace.json.gz
  step wall= 12.92ms  gpu_busy= 12.68ms  gpu_idle_in_step= 0.09ms
  draft                        occ= 20 kernels/step=    293 busy_ms=  1.87 raw_ms=  2.12 legacy_trimmed_ms=  1.55
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1317 busy_ms= 10.18 raw_ms= 12.17 legacy_trimmed_ms=  9.60
  draft_extend                 occ= 20 kernels/step=     88 busy_ms=  0.44 raw_ms=  0.52 legacy_trimmed_ms=  0.45
  TOTAL                        kernels/step=   1698 busy_ms= 12.50 wall_ms= 12.92 idle_ms= 0.09 legacy_trimmed_ms= 11.60
  (check)                      sum(busy)+idle= 12.59ms vs wall= 12.92ms  residual= 0.33ms (86 unattributed kernels/step)
```

x3-20260907-221750-a / agent-loop:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-221750-a/agent-loop/1788787275.0097716-TP-0.trace.json.gz
  step wall= 13.70ms  gpu_busy= 13.23ms  gpu_idle_in_step= 0.09ms
  draft                        occ= 20 kernels/step=    293 busy_ms=  1.84 raw_ms=  2.07 legacy_trimmed_ms=  1.55
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1317 busy_ms= 10.89 raw_ms= 12.85 legacy_trimmed_ms=  9.84
  draft_extend                 occ= 20 kernels/step=     88 busy_ms=  0.44 raw_ms=  0.52 legacy_trimmed_ms=  0.45
  TOTAL                        kernels/step=   1698 busy_ms= 13.17 wall_ms= 13.70 idle_ms= 0.09 legacy_trimmed_ms= 11.84
  (check)                      sum(busy)+idle= 13.26ms vs wall= 13.70ms  residual= 0.44ms (88 unattributed kernels/step)
```

x3-20260907-221750-a: all three trace requests complete.

x3-20260907-221750-a: owned server group cleanup finished; flock released when this command exits.

Fixed W8 first arm is complete: all 3 traces contain 20 draft annotations (19 complete draft-start intervals), six draft forwards per step. Trace-aligned power samples: 6 total, SM 2197–2295 MHz, median 2242 MHz; no power/thermal/HW-slowdown reason in those six samples. Entire ready-to-cleanup population includes five power-cap samples; their timestamps precede the first CUPTI window, so these populations must not be conflated. The next arm is queued on the shared lock.

### Starting x3-20260907-222131-b

Arm `b` at 2026-09-07T22:26:40.986111+09:00; per-run flock held.

### Fixed W8 (a), initial detailed summary

| workload | complete intervals | draft forwards | wall ms | busy ms | idle ms | legacy trimmed ms | trace samples | SM median MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code-edit | 19 | 6 | 13.721 | 13.645 | 0.153 | 11.768 | 2 | 2197 |
| prose-en | 19 | 6 | 12.915 | 12.675 | 0.090 | 11.600 | 2 | 2295 |
| agent-loop | 19 | 6 | 13.700 | 13.230 | 0.090 | 11.839 | 2 | 2242 |

Artifact: `specs/x3/x3-20260907-221750-a/analysis.json`. Busy/idle/wall are independent medians, so their sum need not equal exactly. `legacy_trimmed_ms` is retained only for comparison with WIDTH_SWEEP_0907; it clips durations by kernel name and is not a physical step-time partition.

x3-20260907-222131-b: ready free VRAM 9170 MiB. Initial S=15.

x3-20260907-222131-b / code-edit:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-222131-b/code-edit/1788787812.543283-TP-0.trace.json.gz
  step wall= 15.47ms  gpu_busy= 15.24ms  gpu_idle_in_step= 0.09ms
  draft                        occ= 20 kernels/step=    311 busy_ms=  1.88 raw_ms=  2.12 legacy_trimmed_ms=  1.53
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1413 busy_ms= 12.55 raw_ms= 14.73 legacy_trimmed_ms= 11.35
  draft_extend                 occ= 20 kernels/step=    103 busy_ms=  0.52 raw_ms=  0.60 legacy_trimmed_ms=  0.52
  TOTAL                        kernels/step=   1827 busy_ms= 14.95 wall_ms= 15.47 idle_ms= 0.09 legacy_trimmed_ms= 13.39
  (check)                      sum(busy)+idle= 15.05ms vs wall= 15.47ms  residual= 0.42ms (80 unattributed kernels/step)
```

x3-20260907-222131-b / prose-en:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-222131-b/prose-en/1788787820.6670752-TP-0.trace.json.gz
  step wall= 14.75ms  gpu_busy= 14.41ms  gpu_idle_in_step= 0.10ms
  draft                        occ= 20 kernels/step=    311 busy_ms=  1.94 raw_ms=  2.18 legacy_trimmed_ms=  1.52
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1413 busy_ms= 12.03 raw_ms= 13.99 legacy_trimmed_ms= 11.07
  draft_extend                 occ= 20 kernels/step=    103 busy_ms=  0.48 raw_ms=  0.57 legacy_trimmed_ms=  0.49
  TOTAL                        kernels/step=   1827 busy_ms= 14.45 wall_ms= 14.75 idle_ms= 0.10 legacy_trimmed_ms= 13.08
  (check)                      sum(busy)+idle= 14.55ms vs wall= 14.75ms  residual= 0.21ms (80 unattributed kernels/step)
```

x3-20260907-222131-b / agent-loop:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-222131-b/agent-loop/1788787830.2005076-TP-0.trace.json.gz
  step wall= 16.02ms  gpu_busy= 15.86ms  gpu_idle_in_step= 0.10ms
  draft                        occ= 20 kernels/step=    311 busy_ms=  1.88 raw_ms=  2.12 legacy_trimmed_ms=  1.52
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1413 busy_ms= 12.91 raw_ms= 14.76 legacy_trimmed_ms= 11.17
  draft_extend                 occ= 20 kernels/step=    103 busy_ms=  0.62 raw_ms=  0.70 legacy_trimmed_ms=  0.51
  TOTAL                        kernels/step=   1827 busy_ms= 15.41 wall_ms= 16.02 idle_ms= 0.10 legacy_trimmed_ms= 13.21
  (check)                      sum(busy)+idle= 15.50ms vs wall= 16.02ms  residual= 0.52ms (80 unattributed kernels/step)
```

x3-20260907-222131-b: all three trace requests complete.

x3-20260907-222131-b: owned server group cleanup finished; flock released when this command exits.

## A/B attribution and candidate fix (before re-trace)

The pinned adaptive arm reproduces a substantial same-width penalty: wall (code/prose/agent) 15.466/14.755/16.023 ms versus fixed W8 13.721/12.915/13.700 ms. Legacy trimmed 13.394/13.084/13.209 versus 11.768/11.600/11.839 ms. GPU idle medians remain ~0.09–0.10 ms, and both arms have one `cudaEventSynchronize` and four graph launches per step (boundary rounding applies). More time in the existing synchronization is waiting for more GPU work, not an added synchronization count.

The large change is target MoE tactics: `cutlass_moe_grouped_gemm1/2` use a different CUTLASS specialization in (b), and a separate `trtllm::finalizeMoeRoutingKernel` appears once per target layer. The fixed target GEMM2 uses a `ScaledAccPerRowBiasPerColScaleScatter` epilogue; the pinned adaptive target uses a generic `LinearCombination` epilogue and separate finalization. The actual verify token shapes remain T=8; draft remains six forwards, so this is not a padded T=16 replay.

- Prose target GEMM1 median 43.25 -> 48.00 us; GEMM2 25.31 -> 36.18 us; extra finalize 9.06 us/call. Across the exact exclusive-time window those three families add ~1.551 ms/step exclusive GPU time. This accounts for most of the 1.839 ms wall increase, with a noisy residual and other small kernel changes.
- Target kernels per phase occurrence rise 1317 -> 1413: 48 separate finalizations plus 48 shared-expert activation kernels from the launcher gate/up-fusion difference. Draft rises 293 -> 311 from the shared-expert unfused path; draft-extend rises 88 -> 103, including confidence reductions/pointwise kernels. The shared-expert difference must remain separate from the target tactic bug.
- `BaseRunner.warmup` returns early when the shared model's `_kernel_warmed_up` is true. Every additional target graph runner reuses that already-warmed model, so its candidate-specific target autotune is skipped. The ordinary first target runner runs autotune; speculative draft autotune follows before additional state construction. Candidate shapes/tactic state are not guaranteed to be covered by that first pass. This mechanism predicts an improvement from explicitly running the existing target warmup before candidate graph capture.

Worktree `$HOME/tools/sglang-x3`, branch `codex/x3-adaptive-cost`, base `14d4c4c985`, created without commit. Flag `SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1` temporarily resets the target warmup guard and installs the candidate attention backend while its target graph runner is constructed. Existing autotune-disable and deterministic gates still apply. Both original attributes are restored, including on exception. No decode-loop work or policy math changes. Candidate backend installation is necessary because the warmup dummy forward reads `model_runner.attn_backend`; using the base backend would risk disturbing its metadata. CPU tests and measured effect follow.

CPU validation: **49 tests PASS**, including five new candidate-target-warmup tests and existing adaptive sizing/switch/policy tests. The new tests exercise the real `BaseRunner.warmup` with mocked GPU work: flag-off skip, flag-on autotune once, existing should-autotune gate, candidate-backend use, attribute restoration, absent attribute, and capture exception. `git diff --check` passes. Evidence: `specs/x3/cpu-tests.txt`; overlay origin: `specs/x3/fix-code-origin.txt`.

### Starting x3-20260907-223046-c

Arm `c` at 2026-09-07T22:35:56.571937+09:00; per-run flock held.

Further source confirmation: the installed FlashInfer source (`sglang-rtxpro6000/.venv/lib/python3.12/site-packages/flashinfer/autotuner/autotuner.py`, `autotune` context entry; confirmed by `specs/x3/flashinfer-origin.txt`) explicitly clears `_file_configs` when loading each cache file. SGLang uses distinct target and draft cache keys (`152015503b605a2e` and `37205f5386c1d677` in the logs). Target initial graphs are captured after the target cache is loaded; additional adaptive target graphs are built after the draft cache has replaced that file-config table, and skip target warmup due to the shared model guard. In-memory profiling hits may mask this in other cache states, so this is a cache-order-sensitive construction bug, not an inherent requirement that adaptive W8 be slower. The opt-in warmup reloads/warms the target tactics immediately before the additional target graph capture.

x3-20260907-223046-c: ready free VRAM 9483 MiB. Initial S=15.

x3-20260907-223046-c / code-edit:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223046-c/code-edit/1788788336.0699255-TP-0.trace.json.gz
  step wall= 21.04ms  gpu_busy= 20.93ms  gpu_idle_in_step= 0.11ms
  draft                        occ= 20 kernels/step=    675 busy_ms=  5.42 raw_ms=  6.07 legacy_trimmed_ms=  3.66
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1365 busy_ms= 14.63 raw_ms= 16.97 legacy_trimmed_ms= 11.89
  draft_extend                 occ= 20 kernels/step=    102 busy_ms=  0.54 raw_ms=  0.62 legacy_trimmed_ms=  0.55
  TOTAL                        kernels/step=   2142 busy_ms= 20.59 wall_ms= 21.04 idle_ms= 0.11 legacy_trimmed_ms= 16.09
  (check)                      sum(busy)+idle= 20.70ms vs wall= 21.04ms  residual= 0.34ms (91 unattributed kernels/step)
```

x3-20260907-223046-c / prose-en:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223046-c/prose-en/1788788344.2510695-TP-0.trace.json.gz
  step wall= 20.10ms  gpu_busy= 19.54ms  gpu_idle_in_step= 0.13ms
  draft                        occ= 20 kernels/step=    675 busy_ms=  4.78 raw_ms=  5.46 legacy_trimmed_ms=  3.63
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1365 busy_ms= 14.05 raw_ms= 16.67 legacy_trimmed_ms= 11.94
  draft_extend                 occ= 20 kernels/step=    102 busy_ms=  0.48 raw_ms=  0.56 legacy_trimmed_ms=  0.49
  TOTAL                        kernels/step=   2142 busy_ms= 19.31 wall_ms= 20.10 idle_ms= 0.13 legacy_trimmed_ms= 16.05
  (check)                      sum(busy)+idle= 19.44ms vs wall= 20.10ms  residual= 0.66ms (85 unattributed kernels/step)
```

x3-20260907-223046-c / agent-loop:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223046-c/agent-loop/1788788353.8080235-TP-0.trace.json.gz
  step wall= 12.15ms  gpu_busy= 11.99ms  gpu_idle_in_step= 0.08ms
  draft                        occ= 20 kernels/step=    115 busy_ms=  0.57 raw_ms=  0.65 legacy_trimmed_ms=  0.53
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1365 busy_ms= 10.60 raw_ms= 12.28 legacy_trimmed_ms=  8.58
  draft_extend                 occ= 20 kernels/step=    102 busy_ms=  0.44 raw_ms=  0.52 legacy_trimmed_ms=  0.45
  TOTAL                        kernels/step=   1582 busy_ms= 11.61 wall_ms= 12.15 idle_ms= 0.08 legacy_trimmed_ms=  9.55
  (check)                      sum(busy)+idle= 11.69ms vs wall= 12.15ms  residual= 0.46ms (96 unattributed kernels/step)
```

x3-20260907-223046-c: all three trace requests complete.

x3-20260907-223046-c: owned server group cleanup finished; flock released when this command exits.

### Production wa (c) reference completed

All three CUPTI traces contain 20 draft annotations / 19 complete draft-start intervals. Measured S histograms: code-edit S15=20; prose-en S15=19 and S3=1; agent-loop S3=20. The prose trace catches a policy transition after code-edit and is not a steady W4 prose comparison. Workload order is code-edit -> prose-en -> agent-loop, without fnbench warmup blocks, exactly as the profiling harness. Wall / legacy trimmed ms: code 21.042 / 16.092; prose 20.104 / 16.055; agent 12.148 / 9.551.

Trace-aligned power population for (c): n=10, SM median 2287 MHz (range 2265–2287), memory 13365 MHz throughout, 0 power-cap / 0 SW-thermal / 0 HW-slowdown samples. Width transitions and phase medians are retained in `specs/x3/x3-20260907-223046-c/analysis.json`.

### Starting x3-20260907-223544-b-fix

Arm `b-fix` at 2026-09-07T22:43:19.090181+09:00; per-run flock held.

x3-20260907-223544-b-fix: ready free VRAM 8984 MiB. Initial S=15.

x3-20260907-223544-b-fix / code-edit:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223544-b-fix/code-edit/1788788702.2172105-TP-0.trace.json.gz
  step wall= 13.39ms  gpu_busy= 13.30ms  gpu_idle_in_step= 0.09ms
  draft                        occ= 20 kernels/step=    311 busy_ms=  1.74 raw_ms=  1.98 legacy_trimmed_ms=  1.56
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1365 busy_ms= 10.92 raw_ms= 12.74 legacy_trimmed_ms=  9.88
  draft_extend                 occ= 20 kernels/step=    103 busy_ms=  0.50 raw_ms=  0.59 legacy_trimmed_ms=  0.51
  TOTAL                        kernels/step=   1779 busy_ms= 13.17 wall_ms= 13.39 idle_ms= 0.09 legacy_trimmed_ms= 11.96
  (check)                      sum(busy)+idle= 13.26ms vs wall= 13.39ms  residual= 0.13ms (80 unattributed kernels/step)
```

x3-20260907-223544-b-fix / prose-en:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223544-b-fix/prose-en/1788788709.1346695-TP-0.trace.json.gz
  step wall= 13.41ms  gpu_busy= 13.13ms  gpu_idle_in_step= 0.10ms
  draft                        occ= 20 kernels/step=    311 busy_ms=  1.87 raw_ms=  2.11 legacy_trimmed_ms=  1.55
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1365 busy_ms= 10.66 raw_ms= 12.54 legacy_trimmed_ms=  9.74
  draft_extend                 occ= 20 kernels/step=    103 busy_ms=  0.49 raw_ms=  0.58 legacy_trimmed_ms=  0.50
  TOTAL                        kernels/step=   1779 busy_ms= 13.03 wall_ms= 13.41 idle_ms= 0.10 legacy_trimmed_ms= 11.79
  (check)                      sum(busy)+idle= 13.13ms vs wall= 13.41ms  residual= 0.28ms (80 unattributed kernels/step)
```

x3-20260907-223544-b-fix / agent-loop:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223544-b-fix/agent-loop/1788788717.5658214-TP-0.trace.json.gz
  step wall= 13.29ms  gpu_busy= 13.16ms  gpu_idle_in_step= 0.10ms
  draft                        occ= 20 kernels/step=    311 busy_ms=  1.97 raw_ms=  2.22 legacy_trimmed_ms=  1.52
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1365 busy_ms= 10.48 raw_ms= 12.44 legacy_trimmed_ms=  9.64
  draft_extend                 occ= 20 kernels/step=    103 busy_ms=  0.49 raw_ms=  0.57 legacy_trimmed_ms=  0.50
  TOTAL                        kernels/step=   1779 busy_ms= 12.95 wall_ms= 13.29 idle_ms= 0.10 legacy_trimmed_ms= 11.66
  (check)                      sum(busy)+idle= 13.04ms vs wall= 13.29ms  residual= 0.24ms (80 unattributed kernels/step)
```

x3-20260907-223544-b-fix: all three trace requests complete.

x3-20260907-223544-b-fix: owned server group cleanup finished; flock released when this command exits.

### Starting x3-20260907-223929-d

Arm `d` at 2026-09-07T22:45:33.556187+09:00; per-run flock held.

## Fix re-trace: measured mechanism confirmed

Arm `x3-20260907-223544-b-fix` completed all three workload traces with the production launcher and venv, `$HOME/tools/sglang-x3/python` overlay, and `SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1`. Log order now explicitly includes a target-cache autotune pass at 22:44:51 immediately before completing the S=7 additional runtime state; baseline (b) had only initial-target and draft passes.

Target verify kernels/occurrence fall **1413 -> 1365**, exactly **48 fewer** (one separate target MoE finalize per layer). The shared-expert gate/up setting remains disabled as in baseline (b), so the remaining 48-kernel difference versus fixed W8 is retained rather than conflated with the fix. Draft stays 311 kernels/occurrence and draft-extend stays 103. No width/policy change.

Measured wall / legacy trimmed ms after fix: code-edit 13.395 / 11.957; prose-en 13.406 / 11.792; agent-loop 13.286 / 11.662. Baseline wall was 15.466 / 14.755 / 16.023 ms. The improvement is in target GPU work, not removal of the ordinary acceptance synchronization. Full kernel comparison and power populations follow.

The fixed-W8 versus adaptive comparison is not perfectly identical at the launcher level: shared gate/up fusion and mamba-slot defaults differ. Those differences remain unchanged in the decisive (b) -> (b-fix) test. The fix changes graph-construction-time autotune/cache order only.

### Named-kernel medians (median of the three workload medians, us/call)

| phase / kernel / grid | fixed W8 (a) | pinned (b) | pinned + fix |
|---|---:|---:|---:|
| verify / cutlass_moe_grouped_gemm1 / [1,188,1] | 43.25 | 48.74 | 43.06 |
| verify / cutlass_moe_grouped_gemm2 / [1,188,1] | 25.31 | 37.26 | 25.33 |
| verify / trtllm::finalizeMoeRoutingKernel / [8,1,1] | absent | 9.09 | absent |
| verify / _w8a16_gemv_kernel / [1940,1,1] | 403.42 | 402.19 | 401.01 |
| verify / _w8a16_gemv_kernel / [256,1,1] | 29.89 | 30.11 | 30.02 |

The generic target MoE specialization and separate finalize disappear with the flag; shared-expert activation and confidence-side kernels remain. Full phase/family/grid counts, medians, raw sums, and exclusive contributions are in each arm's kernel CSV/analysis, with `ab-diff.txt` and `b-fix-diff.txt` for all three workloads. Raw kernel duration sums overlap across streams and must not be added as independent wall savings.

| workload | b wall ms | fix wall ms | wall delta ms (%) | b trimmed ms | fix trimmed ms | trimmed delta % |
|---|---:|---:|---:|---:|---:|---:|
| code-edit | 15.466 | 13.395 | -2.071 (-13.39%) | 13.394 | 11.957 | -10.72% |
| prose-en | 14.755 | 13.406 | -1.349 (-9.14%) | 13.084 | 11.792 | -9.87% |
| agent-loop | 16.023 | 13.286 | -2.737 (-17.08%) | 13.209 | 11.662 | -11.71% |

x3-20260907-223929-d: ready free VRAM 11299 MiB. Initial S=3.

x3-20260907-223929-d / code-edit:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223929-d/code-edit/1788788832.6784737-TP-0.trace.json.gz
  step wall= 10.74ms  gpu_busy= 10.47ms  gpu_idle_in_step= 0.08ms
  draft                        occ= 20 kernels/step=    109 busy_ms=  0.57 raw_ms=  0.64 legacy_trimmed_ms=  0.53
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1317 busy_ms=  9.33 raw_ms= 11.05 legacy_trimmed_ms=  8.48
  draft_extend                 occ= 20 kernels/step=     88 busy_ms=  0.42 raw_ms=  0.49 legacy_trimmed_ms=  0.44
  TOTAL                        kernels/step=   1514 busy_ms= 10.32 wall_ms= 10.74 idle_ms= 0.08 legacy_trimmed_ms=  9.45
  (check)                      sum(busy)+idle= 10.39ms vs wall= 10.74ms  residual= 0.34ms (86 unattributed kernels/step)
```

x3-20260907-223929-d / prose-en:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223929-d/prose-en/1788788840.325849-TP-0.trace.json.gz
  step wall= 10.50ms  gpu_busy= 10.41ms  gpu_idle_in_step= 0.07ms
  draft                        occ= 20 kernels/step=    109 busy_ms=  0.56 raw_ms=  0.64 legacy_trimmed_ms=  0.53
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1317 busy_ms=  9.17 raw_ms= 10.98 legacy_trimmed_ms=  8.31
  draft_extend                 occ= 20 kernels/step=     88 busy_ms=  0.41 raw_ms=  0.49 legacy_trimmed_ms=  0.43
  TOTAL                        kernels/step=   1514 busy_ms= 10.15 wall_ms= 10.50 idle_ms= 0.07 legacy_trimmed_ms=  9.27
  (check)                      sum(busy)+idle= 10.22ms vs wall= 10.50ms  residual= 0.28ms (91 unattributed kernels/step)
```

x3-20260907-223929-d / agent-loop:

```text
$HOME/tools/flash-next-bench/specs/x3/x3-20260907-223929-d/agent-loop/1788788848.5659604-TP-0.trace.json.gz
  step wall= 10.58ms  gpu_busy= 10.47ms  gpu_idle_in_step= 0.09ms
  draft                        occ= 20 kernels/step=    109 busy_ms=  0.56 raw_ms=  0.64 legacy_trimmed_ms=  0.53
  step[TARGET_VERIFY bs=1]     occ= 20 kernels/step=   1317 busy_ms=  9.07 raw_ms= 10.80 legacy_trimmed_ms=  8.34
  draft_extend                 occ= 20 kernels/step=     88 busy_ms=  0.41 raw_ms=  0.49 legacy_trimmed_ms=  0.43
  TOTAL                        kernels/step=   1514 busy_ms= 10.04 wall_ms= 10.58 idle_ms= 0.09 legacy_trimmed_ms=  9.30
  (check)                      sum(busy)+idle= 10.13ms vs wall= 10.58ms  residual= 0.45ms (91 unattributed kernels/step)
```

x3-20260907-223929-d: all three trace requests complete.

x3-20260907-223929-d: owned server group cleanup finished; flock released when this command exits.

## Final comparison tables

All numbers are measured unless stated. One 20-step trace per workload per arm; 19 complete draft-start intervals. Values are not confidence intervals.

| arm | code wall / trimmed ms | prose wall / trimmed ms | agent wall / trimmed ms | measured S (code; prose; agent) |
|---|---:|---:|---:|---|
| a | 13.721 / 11.768 | 12.915 / 11.600 | 13.700 / 11.839 | S7×20; S7×20; S7×20 |
| b | 15.466 / 13.394 | 14.755 / 13.084 | 16.023 / 13.209 | S7×20; S7×20; S7×20 |
| c | 21.042 / 16.092 | 20.104 / 16.055 | 12.148 / 9.551 | S15×20; S15×19,S3×1; S3×20 |
| d | 10.737 / 9.448 | 10.498 / 9.273 | 10.582 / 9.304 | S3×20; S3×20; S3×20 |
| b-fix | 13.395 / 11.957 | 13.406 / 11.792 | 13.286 / 11.662 | S7×20; S7×20; S7×20 |

| arm | trace n | SM min / p10 / median / p90 / max MHz | memory MHz | power-cap fraction | SW thermal fraction | HW slowdown fraction | steady min free MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| a | 6 | 2197/2197/2242/2295/2295 | 13365 | 0.00% | 0.00% | 0.00% | 9258 |
| b | 9 | 2115/2115/2272/2287/2287 | 13365 | 0.00% | 0.00% | 0.00% | 7518 |
| c | 10 | 2265/2265/2287/2287/2287 | 13365 | 0.00% | 0.00% | 0.00% | 8007 |
| d | 6 | 2145/2145/2280/2295/2295 | 13365 | 0.00% | 0.00% | 0.00% | 9767 |
| b-fix | 8 | 2145/2145/2280/2295/2295 | 13365 | 0.00% | 0.00% | 0.00% | 7399 |

Trace power samples are selected by `baseTimeNanoseconds + draft_annotation.ts * 1000`, first through last draft start. 100 ms polling is not sub-step resolution; repeated NVML values are not independent observations.

| arm | ready-to-cleanup n (includes waits) | SM p10 / median / p90 MHz | power-cap fraction | SW thermal / HW slowdown | power-draw median / max W | whole-server power-cap counter delta us (includes loading) |
|---|---:|---|---:|---|---|---:|
| a | 265 | 2287/2302/2325 | 1.89% | 0.00% / 0.00% | 93.36 / 272.23 | 481550 |
| b | 267 | 2272/2302/2325 | 1.87% | 0.00% / 0.00% | 97.11 / 280.57 | 459078 |
| c | 269 | 2287/2302/2325 | 1.86% | 0.00% / 0.00% | 112.03 / 287.74 | 361390 |
| d | 237 | 2295/2302/2325 | 2.11% | 0.00% / 0.00% | 101.18 / 257.00 | 498873 |
| b-fix | 242 | 2295/2302/2332 | 2.07% | 0.00% / 0.00% | 92.89 / 272.92 | 477942 |

| workload | fixed W8/W4 wall | pinned W8/W4 wall | fixed W8/W4 trimmed | pinned W8/W4 trimmed | fix vs b wall | fix vs b trimmed |
|---|---:|---:|---:|---:|---:|---:|
| code-edit | 1.278x | 1.440x | 1.246x | 1.418x | -13.39% | -10.72% |
| prose-en | 1.230x | 1.405x | 1.251x | 1.411x | -9.14% | -9.87% |
| agent-loop | 1.295x | 1.514x | 1.272x | 1.420x | -17.08% | -11.71% |

Artifacts:

- a: `specs/x3/x3-20260907-221750-a/` (CUPTI, server log, requested command/environment, 100 ms power CSV, before/after nvidia-smi snapshots, event boundaries, kernel CSV, full analysis).
- b: `specs/x3/x3-20260907-222131-b/` (CUPTI, server log, requested command/environment, 100 ms power CSV, before/after nvidia-smi snapshots, event boundaries, kernel CSV, full analysis).
- c: `specs/x3/x3-20260907-223046-c/` (CUPTI, server log, requested command/environment, 100 ms power CSV, before/after nvidia-smi snapshots, event boundaries, kernel CSV, full analysis).
- d: `specs/x3/x3-20260907-223929-d/` (CUPTI, server log, requested command/environment, 100 ms power CSV, before/after nvidia-smi snapshots, event boundaries, kernel CSV, full analysis).
- b-fix: `specs/x3/x3-20260907-223544-b-fix/` (CUPTI, server log, requested command/environment, 100 ms power CSV, before/after nvidia-smi snapshots, event boundaries, kernel CSV, full analysis).

## Verdict

**The adaptive-runtime cost penalty is real and primarily a graph-construction/autotune-cache bug, not intrinsic step-policy bookkeeping.** For agent-loop, pinned adaptive W8/W4 wall ratio is 1.514x versus fixed W8/W4 1.295x in this session. The corresponding trimmed ratios are 1.420x versus 1.272x. These reproduce the direction and approximate size of WA3's penalty, while not conflating its acceptance/client-throughput effective step time with CUPTI wall or the historical legacy-trimmed fit.

The model-wide warmup guard skips autotuning for additional target runtime states. By then FlashInfer's draft-cache load has replaced its file-config lookup table. The candidate target graph therefore captures generic MoE GEMM tactics with separate finalization. The flag-gated candidate warmup restores tuned GEMM1/GEMM2 and removes 48 target-finalize launches per step. Measured wall improvements are **13.39% code-edit, 9.14% prose-en, 17.08% agent-loop**; legacy-trimmed improvements are **10.72%, 9.87%, 11.71%**. Post-fix trimmed time is within -1.50% to +1.66% of fixed W8 across the three workloads, despite retaining wa's unfused shared gate/up and confidence-side computation. Do not interpret the post-fix agent wall being slightly faster than fixed W8 as a separately established adaptive advantage; this is one trace per workload and session drift/preemption remain.

This does not retrospectively assign every microsecond of WA3's effective penalty to the cache bug. WA3 also enabled chain tracing/debug (including synchronous `pop_chain`), used different request lengths and acceptance distributions, and estimated end-to-end cost. X3 isolates and fixes one large reproducible production-profile overhead. No WA3 chain-tracing ablation or fnbench throughput/quality gate is claimed.

### Power/clock verdict (Finding 13)

**No power-cap, SW-thermal, or HW-slowdown reason was observed within the measured decode-step windows: 0/39 trace-aligned samples across all five arms.** Memory clock was 13365 MHz throughout those windows, and SM distributions are in the per-arm table. The 325 W limit was confirmed in the GPU snapshots. The large cost difference changes MoE tactics and finalization count, while memory-bound lm_head and GDN projection medians stay nearly constant; clock limiting is not supported as the main explanation for the adaptive penalty.

**The cap does bind transiently elsewhere in the server lifetime.** Every arm has five ready-to-cleanup samples with SW Power Cap active (1.86–2.11% of that population), all before the first CUPTI decode window. Whole-server SW-power-capping counters increase by 0.361–0.499 s; SW/HW thermal and HW-power-braking counter deltas are zero for every arm. The counter snapshots bracket loading, warmup, prefill, decode, export and idle time, so they cannot assign that accumulated power-cap time to a particular decode step. The initial-request samples cannot distinguish prefill from the earliest unprofiled decode with this harness. Slow `power.draw` values below 325 W do not negate an active limiting reason.

Therefore the justified conclusion is **no demonstrated binding cap during the sampled settled decode windows, and no supported power/clock remedy for this cost penalty; short unsampled binding within a step is not ruled out by 100 ms polling**. No power limit, clocks, display setting or administrative GPU control was changed. This closes the proposed clock-gain assumption for X3 without claiming that the cap is never active.

## Completeness and practical limits

| item | outcome / evidence |
|---|---|
| Required arms | a fixed W8, b pinned wa, c production wa, d fixed W4: complete |
| Fix re-trace | b-fix: complete, same pinned config and launch S=15/T=16, measured S=7/T=8 |
| CUPTI coverage | 15 traces; each has 20 draft annotations and 19 complete intervals; 300 annotations / 285 intervals total |
| Workloads | code-edit, prose-en, agent-loop in every arm; no fnbench invocation |
| CPU validation | 49 tests passed; includes real BaseRunner.warmup gating with mocked GPU work and state restoration on failure |
| Memory / display | all fractions 0.920, MAX_TOTAL_TOKENS=131072, SERVE_DISPLAY_HZ empty; minimum steady free VRAM 7399 MiB, above 4096 MiB in every arm |
| Loading transients | sampled, not used to abort; no VRAM abort or failed server run |
| Shared GPU lock | one flock per server lifetime, released between runs; other tasks queued independently |
| GPU time proxy | owned server-start to cleanup-start lifetimes sum 900.600 s = 0.250 GPU-hours; includes loading and idle, excludes lock queue and 8 s cleanup grace per run |
| Cleanup | all five owned process groups absent at final ps check; no unowned process termination |
| Production protection | production mounted read-only for each server, private runtime-cache mount; source/launcher/venv-entrypoint hashes and production git status unchanged |
| Worktree | $HOME/tools/sglang-x3, codex/x3-adaptive-cost, HEAD remains base 14d4c4c985a87f329f27af9342970b28b35a8ed4 |
| Changes | adaptive_runtime_state.py context helper, eagle_worker_v2.py construction wrapper, one CPU test file; no adaptive_confidence.py or production edit |
| Flag | SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1, default off; no production launcher export added |
| Reproduction | specs/x3/run_arm.py, analyze.py, compare.py, final_tables.py; per-arm command.json, pinned7.json, cache snapshot/hashes, fix.patch |
| Validation manifests | specs/x3/completeness.json, production-final-check.json, process-cleanup-check.json, cpu-tests.txt, fix-code-origin.txt, flashinfer-origin.txt |
| Commit / push | none; only the requested worktree branch created |
| Not tested / not claimed | full production-quality/needle or acceptance parity; fnbench throughput; long context/default full KV capacity; bs>1; three-candidate configs; multi-GPU; repeated-run significance |

The patch remains an **opt-in experimental fix with measured step-cost benefit**, ready for review in the named worktree. Autotuning may choose a different floating-point reduction order; the profiling requests completed without runtime errors, but that is not a quality-equivalence test. Multiple additional candidates can also interact with target/draft cache ordering, and were not exercised by the requested [7,15] runtime-state set. Existing disable-autotune/determinism gates and the default-off behavior are preserved. No deployment or integration is implied.
