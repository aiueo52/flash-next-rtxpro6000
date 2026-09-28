# WA6 — request-boundary priors with measured width costs

Status: COMPLETE. The policy arms were evaluated on private data; no result or conclusion from that evaluation is published. The WA6 prior is not part of the production configuration (opt-in, off by default). No production change, commit or push.

> Publication note: every timed WA6 serving stream (preparation, fixed-width references, the A1/P1/P2/A2 policy arms and the diagnostic replays) was built from the author's private MTP-training prompts ("v5 holdout parents"). All throughput, acceptance, latency, width-occupancy, swap, stream-size and confidence-interval numbers measured on those streams have been removed. Qualitative outcomes, mechanisms and pass/fail results from those streams have also been removed. What remains is the protocol, the implementation, the per-width cost table (derived from earlier public-workload X4 traces), CPU-side tests, and the resulting state of the software.

## Frozen scope and protocol

- Base codex/perf-v1 7b4d539f9b; worktree $HOME/tools/sglang-wa6, branch codex/wa6-request-prior. No commit/push. Current production [3,7,15], config c, STEP 7.943/0.5554, X3 autotune. DH7 has not been adopted at preparation time.
- Read ASTRA_REVIEW_2026-09-08 sections 3 and 5 Task 2, WA5, WA4, WA2 Iteration 3, X4, WIDTH_SWEEP_0907, C1 and production controller/runtime/worker.
- Candidates remain [3,7,15]. Measured per-width cost is opt-in; missing key retains exact linear scoring. Request prior independently opt-in; no target/draft weights changed.
- Every server lifetime, GPU diagnostic and GPU preflight under flock -w 28800 $HOME/.gpu.lock; release after each arm. Only owned process groups terminated. Production source/launchers/venv read-only with private cache mapping and PYTHONPATH overlay. No display changes.
- All arms fraction 0.920, SERVE_DISPLAY_HZ empty, MAX_TOTAL_TOKENS=131072, MAMBA_SLOTS=10, shared gate/up=0. Require >=4096 MiB steady free VRAM; startup separate. No training. New artifacts capped at 120 GB.
- Freeze predictor and cost table before final holdout; no final-holdout tuning. Explicit caller designation is task metadata, not an oracle width. Unknown leaves the ordinary controller in control.
- Predeclared weights: equal domain (code/agent/en/ja), equal budget (64/128/256/2048) within domain; report short (64/128/256) and long (2048) separately. Aggregate delivered throughput = total actually delivered decode tokens / total client decode seconds under this balanced stream. Also report mean normalized per-cell time so long requests do not hide short regressions.
- Domain non-regression provisional tolerance -2%, asked before execution; use user override if supplied. Primary >=3% delivered decode throughput OR >=3% decode-time reduction. Gain must be supported by avoided dwell / initial-width evidence; oracle diagnostic only.
- Warm server once with a neutral request; no per-domain controller warmup or between-request reset in current policy. Mixed paired blocks include code->prose, prose->code, en<->ja. Parent groups disjoint between preparation and held-out streams and among held-out requests; same parents reused across matched arms only.
- Plan: fixed W4/W8/W16 references and a current-policy diagnostic screen, then frozen current/prior paired arms in alternating order, multiple held-out parents, paired block bootstrap; diagnostics separate from uninstrumented timings. Maximum 4 occupied GPU-hours; queue time excluded. Do not add W12.

## Implementation and evidence

Implementation is in the uncommitted `codex/wa6-request-prior` worktree. Final CPU suite: 73 adaptive tests PASS; see `wa6/IMPLEMENTATION.md` and `wa6/cpu-tests.txt`. Final measured S3/S7/S15 costs are 9.648835/12.330200/17.112206 ms, including the confidence D2H correction documented below.

### CPU preflight and frozen inputs

70 adaptive CPU tests PASS (0.772 s) at preflight, including 11 new WA6 tests: exact 3000-update current-production parity, complete/nonfinite cost validation, caller/env/unknown precedence, token-only predictor path, request-owned slot reuse, delayed result ownership, stale-width rejection, and BS1->BS2->BS1 preservation. Initial test invocations with CUDA_VISIBLE_DEVICES empty/-1 failed in existing test_utils integer parsing before those two modules ran; rerun with nonexistent device 999 passes without GPU computation.

Measured cost table: equal mean of existing X4 code/en/agent physical fixed-width intervals (public-workload traces), plus a mean raw confidence-kernel charge measured from pinned adaptive traces. Physical boundaries include overlapped recovery and host/GPU gaps; recovery is not double-counted. The confidence charge is conservative raw execution time, not an exclusive-time saving claim. Japanese has no old fixed-width trace, so a shared BS1 table is used. Raw hashes and complete arithmetic in costs.json / calib/wa6_costs.py. No linear extrapolation and no new width.

Cost-component audit before prior execution: the extra-kernel name list contained 12 kernels but omitted the separate confidence D2H transfer. The raw-trace extractor now includes that transfer explicitly (13 events/step). Final confidence mean charge is 32.394083 us; final S3/S7/S15 table is {"3": 9.648835, "7": 12.3302, "15": 17.112206} ms. This corrects an omitted component from the same frozen historical traces; no new measurement or holdout tuning is used.

Streams: separate preparation and evaluation streams drawn from unique private prompts (sizes not published); maximum prompt 4096 tokens. Even blocks explicitly designate caller domain, odd blocks use the predictor. Original prompts and natural EOS retained; long controls mean 2048-token output budgets, not forced output length. Predictor confusion is recorded before GPU execution and not tuned on the final holdout.

The prior initializes decision statistics only. The existing worker runtime swap drains outgoing recovery and retains valid KV/SSM and request output arrays. Each request keeps its slot through later mixed batches. Requests first seen in mixed batches receive no later start prior. Verify completions are paired FIFO with the producing slot and width, so late results cannot initialize a different request. Unknown/invalid designation inherits live statistics; it does not force a width. Default config omits both keys and retains the existing policy arithmetic and result path.

Diagnostic refinement before GPU start: switch timing records three CUDA events (start, post-recovery-fence, end). The subsequent CPU-ready verify callback queries completion without synchronizing, then reports GPU-visible uncovered recovery wait and complete switch region, alongside CPU submission time. These events are absent from timed runs.

Sandbox-only NVIDIA preflight acquired/released its lock but could not reach the driver (exit 9, no GPU run). The authorized server arm uses host access with the same flock and read-only production mounts. This is a sandbox access limitation, not evidence of a driver fault; no driver or system change was attempted.

### Harness validity correction

The first preparation arm used a defective client and is excluded from all gates. Native SSE returns cumulative text, and `requests.iter_lines(chunk_size=1)` adds large CPU per-byte work, which produces client-side backlog (reproduced by the synthetic transport guard below) — a harness defect, not a width-policy result.

The client was corrected to `iter_lines(chunk_size=None)`, consuming each HTTP chunk as available without per-byte iteration. CPU chunked-SSE transport guard PASS (synthetic): 20 x 32 KiB records at a 20 ms cadence. The old per-byte reader delivered the last record at 2.870 s with maximum 2.288 s backlog; the corrected chunk reader delivered it at 0.384 s with maximum 0.0216 s delay. Evidence: sse-transport-test.json. The analyzer now rejects any request >=0.5 s whose client total exceeds server e2e by max(0.1 s, 5% of server e2e). The preparation arm was rerun with the corrected client.

### Runtime-safety fixes found during the run

- A delayed BS1 decision could otherwise be published while a BS2 slot was active. The prior path now compares the producing slot with the currently live batch slot. A dedicated CPU test checks that a late BS1 promotion cannot select an S7 graph for the active BS2 batch (71 CPU tests pass after this guard).
- Request metadata validation abstains for explicit null/invalid designations and malformed custom_params instead of raising inside the scheduler (covered by CPU tests).
- Metric clarification before any policy timing: natural EOS can change emitted token counts across greedy-width trajectories, so the alternative >=3% decode-time gate uses prior per-request seconds/token applied to matched current-request token counts; emitting fewer tokens is not counted as a speedup.
- A live BS1->BS2->BS1 overlap smoke was run on prompts from the private preparation stream; its outcome is not published. The OpenAI-compatible endpoint forwards `custom_params.adaptive_task` to the controller. Needle PASS.

## Closure and audit

The preregistered A1/P1/P2/A2 comparison was run on the private evaluation stream; no result or conclusion from it (gate results, diagnostics, fixed-width reference comparisons) is published. Artifact/environment audit True.

All four timed policy arms used identical implementation/input/client hashes. Production manifest unchanged; 419 model-file stat/symlink records unchanged. Branch codex/wa6-request-prior, base HEAD 7b4d539f9bd896265f498fabccc6466e45ffe818; no commit/push. Total occupied GPU-lock time stayed under one hour. Final process cleanup audit: all recorded owned server process groups absent; no unrelated process terminated.

Interpretation limits: fixed-reference greedy trajectories and natural EOS differ across widths; fixed-best labels are optimistic diagnostics, never policy inputs. Timed throughput is uninstrumented; occupancy/switch/recovery events come from separate matched diagnostic replays. The small block count bounds generalization. The live overlap smoke is not a test of bit-exact cross-width outputs or a general NI-quality battery. The -2% domain tolerance was predeclared as the provisional default in the absence of a user override.

Production change: **none**. Production remains candidates [3,7,15], adaptive/w16_3_7_15_c.json, STEP 7.943/0.5554 and the X3 autotune fix. The WA6 code remains opt-in and uncommitted in the named worktree. Per the frozen protocol, W12 is not added and the predictor is not tuned on the final holdout.
