# QL1 long-context QSA map

Status: QL1 measurement and analysis complete, 2026-09-08. The execution log below was written incrementally; the final tables supersede the W4 interim deletion ceilings.

## Decision

Fund **one bounded 96k joint index-score/top-k prototype**, not a general attention rewrite. The corrected whole-region deletion ceilings are **+4.82% W4 / +5.01% W16** at 96k, but achieving +3% requires removing **63.4% / 61.0%** of that region's measured raw time. This is a context-specific opportunity envelope, not a measured implementation gain or a promise of +3%.

At 8k and 32k, even deleting the complete scan/score/top-k region is below +3% for both widths. At 64k the ceilings rise to **+4.02% W4 / +3.84% W16**, but require **75.3% / 78.8%** removal: retain 64k as a follow-up gate only after a 96k prototype passes. Packing and shared-prefix packing fail the standalone +3% gate everywhere. W16/96k query tiling has a **+3.09%** complete-scan-deletion ceiling (per-family range **+2.98% to +3.25%**), requiring approximately **97.1% scan removal**; this near-zero replacement allowance does not justify a standalone tiling project.

Completed: **24 CUPTI traces, 20 GPU iterations each; 72 unprofiled 128-token measurements; 24/24 passphrase-only needle checks with exact output equality**. Actual allocated KV capacity was **131072 tokens for both widths**, under an explicitly recorded cap. Both served all 96k prompts successfully; the uncapped maximum was not measured. Minimum steady free VRAM: **10336 MiB W4 / 8931 MiB W16**. Total occupied GPU-lock time: **0.158995 hours (9.54 minutes)**, below the requested 2–4 GPU-hour allowance; no idle padding of the budget was necessary. Both owned process groups exited. Production source/launcher hashes match; production and venv were mounted read-only. No commit or push.

## Scope and frozen design

Review: `ASTRA_REVIEW_2026-09-08.md` item 4.7 / rank 6. Production source and launchers are read-only in a bwrap mount; a private copy of the runtime cache is bound over `.cache`. No production source, launcher, or venv edits; no commit or push. Frozen input hashes: `workloads/longctx/frozen.json`.

- Fixed production w4 and w16, BS=1; mem fraction 0.920 for both, `SERVE_DISPLAY_HZ=''`, expandable segments. Explicit KV upper cap 131072; actual profiled capacity must be verified at readiness. Model context limit remains production 262144.
- Each server arm holds `flock -w 28800 $HOME/.gpu.lock` for its lifetime, including startup and owned-process-group cleanup. No unowned processes are killed. Each arm stops at 7100 occupied seconds (combined maximum <4 GPU lock-hours). One-second VRAM monitor aborts the owned server if ready-state free VRAM falls below 4096 MiB.
- Exact rendered input lengths: 8192, 32768, 65536, 98304 tokens. `workloads/longctx/manifest.json` contains prompt hashes, token ID hashes, actual needle/directive positions, and tokenizer/template provenance. The native `/generate` API consumes the frozen chat-rendered IDs directly, with thinking disabled.
- Two long archive tasks (`doc-a`, operational handover; `doc-b`, directive reconciliation) and a needle-plus-audit task (`needle-perf`, depth 0.4) at every context. The needle family follows `prof/needle_test.py`'s seed, vocabulary and exact passphrase. Separate passphrase-only checks at depths 0.1/0.4/0.9 honor EOS, up to 48 tokens.
- Each performance prompt: flush cache, untimed 128-token warmup, three unprofiled requests of exactly 128 generated tokens, then a separate 600-token diagnostic request with a 20-step CPU/GPU CUPTI trace triggered after eight decode chunks. Forced `ignore_eos` is used only for the bounded performance samples. Prompt tasks request >=900 words to avoid forced continuation of a naturally short task. Cached prompt tokens are recorded by the server metadata. Profile traces are never used for wall-clock throughput.
- Wall decode t/s = `(last cumulative tokens - first cumulative tokens) / (last token-bearing SSE time - first token-bearing SSE time)`, excluding prefill and the first emission. Each sample still generates exactly 128 tokens; the first emitted chunk's token count is recorded and excluded from the timed numerator. `128 / decode_seconds` is retained as a separately labeled conventional value, not the primary estimator. Per-request speculative counters are saved instead of a windowed acceptance gauge.
- Every accepted trace must have 20 `draft` annotations and GPU kernels. `prof/exclusive_time.py` uses first-to-last draft starts, giving 19 complete intervals. Use mean wall as the denominator of the additive exclusive/shared/gap partition; report median wall separately.

## Source findings before GPU execution

The config selects 2048 tokens through 512 compressed blocks at ratio 4. All ladder contexts exceed that budget. `_compact_kv` moves selected K and V; its work must not be extrapolated linearly with full context after selection saturates. Full KV is FP8, with 2 KV heads and head dimension 256; for 2048 valid selected rows, one pack reads and writes about 4 MiB combined (excluding metadata and the short uncompressed tail).

The QSA decode indexer in `qsa/mqa.py` scans compressed keys and produces FP32 scores. Its launch grid is based on maximum page-table capacity, while the active-work branch depends on actual compressed context length. The scan kernel already computes scores; do not double-count it in a separate score bucket. `fast_topk` consumes those scores. Score initialization, index expansion, valid counts, packing and attention are distinct operations.

Draft index selection is already reused from the preceding draft-extend. Repeated packing remains. A shared-packing proposal can only remove redundant immutable prefix copies; it must retain new tail tokens, accepted-row selection, and invalidation at the next verification. Target verification rows need not have identical selections.

## Attribution and economic interpretation

Report scan/score generation, score initialization and top-k, packing, paged attention, and all remaining work. Exclusive time is the existing tool's time with exactly one kernel active, with overlapping time and idle gaps shown separately. This is an observed interval partition, **not a dependency-DAG critical-path proof**: PDL can launch a dependent kernel before producer completion. Exclusive deletion credit can understate dependency-chain savings. For each idea report both (a) observed exclusive credit, and (b) a deliberately optimistic raw-duration ceiling for the eligible kernels, which can reject a >=3% idea if even that ceiling is too small. Neither is a demonstrated throughput gain. No overlap budget is added twice, and the four ideas' ceilings are not additive.

At unchanged acceptance, removing fraction b gives max gain `1/(1-b)-1`; +3% needs b >= 0.0291262136. Required saved time is `T * (1-1/1.03)`. Fit component time versus actual context separately by width and prompt family; compare affine and constant fits, show slopes and residuals, and only infer saturation within this ladder. Tensor-valued selections are not exposed by CUPTI; valid-selection counts must be labeled inferred from source unless separately captured through an existing supported diagnostic.

## Execution log

### w4 / 20260908-ql1: lock acquired

UTC start: 2026-09-08T00:59:45.305954+00:00.

w4: ready with 12472 MiB free; mem_fraction_static=0.920, requested KV cap=131072. Runtime server info and allocation log recorded.

w4: observed KV capacity 131072 tokens; 96k trace needs 99160 including 600 output tokens and 256-token reserve; supports_96k=True.

w4 doc-a-8k: 3 x 128-token wall measurements complete, pooled decode=246.585 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-a-8k`.

w4 doc-b-8k: 3 x 128-token wall measurements complete, pooled decode=253.730 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-b-8k`.

w4 needle-perf-8k-d40: 3 x 128-token wall measurements complete, pooled decode=276.778 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-needle-perf-8k-d40`.

w4 needle-qa-8k-d10: needle PASS=True, completion=12.

w4 needle-qa-8k-d40: needle PASS=True, completion=12.

w4 needle-qa-8k-d90: needle PASS=True, completion=12.

w4 doc-a-32k: 3 x 128-token wall measurements complete, pooled decode=237.661 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-a-32k`.

w4 doc-b-32k: 3 x 128-token wall measurements complete, pooled decode=271.039 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-b-32k`.

w4 needle-perf-32k-d40: 3 x 128-token wall measurements complete, pooled decode=279.176 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-needle-perf-32k-d40`.

w4 needle-qa-32k-d10: needle PASS=True, completion=12.

w4 needle-qa-32k-d40: needle PASS=True, completion=12.

w4 needle-qa-32k-d90: needle PASS=True, completion=12.

w4 doc-a-64k: 3 x 128-token wall measurements complete, pooled decode=261.730 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-a-64k`.

w4 doc-b-64k: 3 x 128-token wall measurements complete, pooled decode=254.928 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-b-64k`.

w4 needle-perf-64k-d40: 3 x 128-token wall measurements complete, pooled decode=273.034 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-needle-perf-64k-d40`.

w4 needle-qa-64k-d10: needle PASS=True, completion=12.

w4 needle-qa-64k-d40: needle PASS=True, completion=12.

w4 needle-qa-64k-d90: needle PASS=True, completion=12.

w4 doc-a-96k: 3 x 128-token wall measurements complete, pooled decode=235.413 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-a-96k`.

w4 doc-b-96k: 3 x 128-token wall measurements complete, pooled decode=253.005 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-doc-b-96k`.

w4 needle-perf-96k-d40: 3 x 128-token wall measurements complete, pooled decode=267.222 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w4-needle-perf-96k-d40`.

w4 needle-qa-96k-d10: needle PASS=True, completion=12.

w4 needle-qa-96k-d40: needle PASS=True, completion=12.

w4 needle-qa-96k-d90: needle PASS=True, completion=12.

w4: owned server process group cleaned, occupied lock-hours=0.0750; flock releases on process exit.

## W4 interim decision

All 12 traces and all 36 unprofiled 128-token measurements completed. Strict needle: 12/12 PASS. Observed KV allocation: 131072 tokens under the explicit cap; steady free VRAM minimum 10336 MiB. Occupied GPU lock time: 0.0750375 h. This allocation supports the complete 96k ladder; it is not a measurement of the uncapped maximum capacity.

| Context | Mean step us | Pack raw ceiling us / gain | Scan raw ceiling us / gain | Scan + score/top-k raw ceiling us / gain | Reduction needed in combined region for +3% |
|---|---:|---:|---:|---:|---:|
| 8k | 9427.1 | 77.1 / +0.82% | 95.7 / +1.03% | 226.5 / +2.46% | 121.2% |
| 32k | 9586.2 | 76.7 / +0.81% | 85.9 / +0.90% | 235.2 / +2.51% | 118.7% |
| 64k | 9756.0 | 85.2 / +0.88% | 138.7 / +1.44% | 360.5 / +3.84% | 78.8% |
| 96k | 9772.8 | 72.5 / +0.75% | 160.2 / +1.67% | 430.4 / +4.61% | 66.1% |

Packing and index tiling individually fail the 3% gate at every W4 context even with zero replacement cost. Combined score/index integration remains an upper-bound-only candidate at 64k/96k and requires an aggressive reduction. W16 is pending under the shared lock. Per-family tables and fits are in `workloads/longctx/tables.md`; no implementation gain has been measured.

### w16 / 20260908-ql1: lock acquired

UTC start: 2026-09-08T01:09:36.087859+00:00.

w16: ready with 11219 MiB free; mem_fraction_static=0.920, requested KV cap=131072. Runtime server info and allocation log recorded.

w16: observed KV capacity 131072 tokens; 96k trace needs 99160 including 600 output tokens and 256-token reserve; supports_96k=True.

w16 doc-a-8k: 3 x 128-token wall measurements complete, pooled decode=152.159 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-a-8k`.

w16 doc-b-8k: 3 x 128-token wall measurements complete, pooled decode=162.002 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-b-8k`.

w16 needle-perf-8k-d40: 3 x 128-token wall measurements complete, pooled decode=194.031 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-needle-perf-8k-d40`.

w16 needle-qa-8k-d10: needle PASS=True, completion=12.

w16 needle-qa-8k-d40: needle PASS=True, completion=12.

w16 needle-qa-8k-d90: needle PASS=True, completion=12.

w16 doc-a-32k: 3 x 128-token wall measurements complete, pooled decode=141.475 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-a-32k`.

w16 doc-b-32k: 3 x 128-token wall measurements complete, pooled decode=164.038 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-b-32k`.

w16 needle-perf-32k-d40: 3 x 128-token wall measurements complete, pooled decode=188.121 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-needle-perf-32k-d40`.

w16 needle-qa-32k-d10: needle PASS=True, completion=12.

w16 needle-qa-32k-d40: needle PASS=True, completion=12.

w16 needle-qa-32k-d90: needle PASS=True, completion=12.

w16 doc-a-64k: 3 x 128-token wall measurements complete, pooled decode=152.836 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-a-64k`.

w16 doc-b-64k: 3 x 128-token wall measurements complete, pooled decode=158.158 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-b-64k`.

w16 needle-perf-64k-d40: 3 x 128-token wall measurements complete, pooled decode=165.510 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-needle-perf-64k-d40`.

w16 needle-qa-64k-d10: needle PASS=True, completion=12.

w16 needle-qa-64k-d40: needle PASS=True, completion=12.

w16 needle-qa-64k-d90: needle PASS=True, completion=12.

w16 doc-a-96k: 3 x 128-token wall measurements complete, pooled decode=138.279 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-a-96k`.

w16 doc-b-96k: 3 x 128-token wall measurements complete, pooled decode=150.927 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-doc-b-96k`.

w16 needle-perf-96k-d40: 3 x 128-token wall measurements complete, pooled decode=210.802 tokens/s; 20-step CUPTI trace validated at `prof/traces/ql1-20260908-ql1-w16-needle-perf-96k-d40`.

w16 needle-qa-96k-d10: needle PASS=True, completion=12.

w16 needle-qa-96k-d40: needle PASS=True, completion=12.

w16 needle-qa-96k-d90: needle PASS=True, completion=12.

w16: owned server process group cleaned, occupied lock-hours=0.0840; flock releases on process exit.

## Final attribution audit and trace-boundary correction

`prof/exclusive_time.py` was used unchanged, including its phase/correlation mapping, labels, 19-interval CPU draft-start window, and exclusive/shared/gap identity. Every trace has a saved `exclusive.txt`. The independent component sweep agrees with its total exclusive time exactly (within 0.01 microseconds before division). The per-context exclusive table intentionally retains this convention.

A material boundary effect appeared in the final audit: the first captured GPU work trails the CPU draft marker by approximately one iteration. For example, the first W16/96k index scan starts about 20 ms after the first CPU draft annotation. There are **260 scans in each full trace (13 x 20)**, but the official CPU window commonly contains **234 scans / 19 intervals = 12.316 scans/step**. Charging the startup gap as steady idle time and using this cropped raw sum would understate the removable cost by roughly 4–6%.

Therefore the **final raw deletion ceilings and aggregate raw scaling fits use all 20 complete GPU iterations**, while exclusive credits continue to use the requested tool's 19 CPU intervals. We verified **13 scans and 15 / 27 packing calls per full GPU step at W4 / W16**, respectively. Both the cropped raw value (`cpu_window_raw_us`) and corrected raw ceiling are preserved in `results.json`. This correction changes the W16/96k complete-scan ceiling from +2.92% to +3.09%; it does not make a practical row-tiling-only +3% project credible. The W4 interim table earlier in this document predates this correction.

The raw ceiling deliberately counts all eligible kernel durations, including overlapping portions and producer wait time inside PDL kernels. It is optimistic and not an achieved wall-time saving. This is a trace-local deletion model holding the remaining kernel costs fixed; unmeasured cache, contention, or clock effects are excluded. The group-exclusive credit includes intervals during which all active kernels belong to the candidate region, so overlap between scan and its top-k consumer is not dropped or counted twice. PDL dependencies prevent treating exclusive occupancy as a full dependency-DAG critical path. These qualifications matter most near the 3% boundary.

Scan is the TileLang `kernel_kernel`, which also computes scores; score generation is not counted twice. The score/top-k bucket contains the associated float score initialization, int row-start initialization, and `fast_topk_kernel`. `_compact_kv` is packing, and `kernel_mha` is paged attention. Q/K preparation/compression, index expansion, valid counts, attention projections, and all other model/scheduler work remain in other. Float/int fills are assigned by the adjacent same-stream scan/top-k call, not by globally grouping every fill of that shape.

## Scaling interpretation

Let x be input context in units of 1024 tokens and y be full-trace mean raw microseconds per GPU step. The four-point affine fits are descriptive within 8k–96k; per-family fits and residuals are included below.

- **Indexer scan:** W4 `y = 80.30 + 0.897 x` (R2=0.866); W16 `y = 140.93 + 4.073 x` (R2 approximately 1.000). W16 has a clear linear growth term and has not saturated at 96k. W4 has a substantial fixed launch/idle-CTA floor and an 8k/32k plateau before increasing.
- **Score initialization + top-k:** W4 `y = 113.98 + 1.755 x` (R2=0.981); W16 `y = 120.23 + 2.043 x` (R2=0.995). Neither has saturated. The score buffer has fixed maximum-context capacity, but top-k reads only the valid compressed length; distribution-dependent selection work also matters.
- **Packing:** W4 is effectively flat (`84.64 - 0.052 x`, R2=0.154). W16 rises mildly (`236.22 + 0.318 x`, R2=0.990), from about 240 to 268 us over a 12x context increase. The **selected-token work is already saturated**, but the W16 measured time is not perfectly constant. Locality, scheduling outliers, and prompt-family effects remain; do not label its entire runtime mathematically constant or extrapolate it proportional to full context.
- **Paged attention:** approximately 260 us W4 and 532–567 us W16 in raw duration, with essentially flat long-context behavior (96k/64k ratios 0.998 and 0.978). Its approximately 10–22 us exclusive occupancy is much smaller because of PDL overlap. Selected-context capacity, rather than the full document length, controls its main work.
- **Everything else:** no substantial context trend in exclusive time. Full raw slopes are only 0.704 us/k W4 and 1.232 us/k W16 with low R2 (0.324/0.131), against roughly 11/19 ms of overlapping raw work. Total CPU-start step fits are `9435.03 + 4.010 x` W4 and `16935.23 + 9.723 x` W16. The official gap includes the boundary effect above and is not a pure steady scheduler-overhead estimate.

The source-derived selection contract is saved in `workloads/longctx/selection-contract.json`. Runtime grids confirm 129 packing tiles for target rows and W4 draft, and 130 for W16 draft. Config budget=2048 selected tokens, compression ratio=4, with at most three uncompressed tail tokens in fresh selections; recursive draft appends a bounded additional tail. **CUPTI does not expose selected-index tensor values. Exact live valid-count distributions and selected-prefix identity were not tensor-captured**; bounds are source-derived, while kernel counts, grids and runtime saturation are measured. The generic indexer-topk capturer is not wired as a supported drop-in for this compressed QSA path. No target/draft tensor-capture modification was introduced for QL1.

## Single cheapest confirming prototype: tiled scoring with local top-512 and exact merge

Scope: **96k, fixed production W16 first; W4 as the second shape in the same microbenchmark**, in an isolated disposable checkout. Keep target weights, acceptance policy, selected-token budget, packing, and paged attention unchanged.

Build one two-stage candidate: score four query rows together while loading a compressed-key tile (initial tile length 4096), retain that tile's exact local top-512 scores/indices, then merge the local candidate lists into the global exact top-512. The union of each tile's top-512 contains the global top-512, subject to an explicit tie rule. This shares compressed-key loads and avoids the full maximum-context score fill/materialization; it is a single joint scan/top-k prototype, not an addition of independent forecast gains. Short final tiles, per-row causal lengths, and equal-score boundary ties must be handled explicitly. If a tie cannot reproduce the selected set, use and charge a baseline fallback rather than silently changing selection.

Capture representative Q, compressed K, page tables, row lengths, and baseline selected sets in a separate untimed diagnostic, then benchmark the **entire replacement region**, including local selection, merge, metadata and fallback costs. Verify selected-token sets and downstream attention outputs against the baseline on the captured cases. A faster isolated score kernel is not a pass.

Hard timing gate from this map, before any acceptance loss:

| Shape | Current scan + score/top-k raw us/step | Required saving us/step | Maximum replacement region us/step | Approximate replacement us per one of 13 index calls |
|---|---:|---:|---:|---:|
| W4 / 96k | 449.3 | 284.6 | **164.6** | **12.7** |
| W16 / 96k | 848.7 | 518.0 | **330.7** | **25.4** |

These are necessary optimistic microbenchmark gates, not sufficient serving guarantees. Continue only if at least one shape passes with margin. Then perform matched unprofiled serving measurements and CUPTI checks on the 96k prompt families, charging all fallback and launch costs and rechecking needle retrieval. Require `A_new/A_old * T_old/T_new >= 1.03`; any acceptance loss increases the required saving. Stop if the candidate misses the timing gate, changes selected tokens without a charged fallback, or fails the matched serving gain. Test 64k only after the 96k gate passes; its residual budgets are just 93.1 us W4 / 139.2 us W16 for the entire replacement region.

Budget: one bounded prototype, **at most 8 occupied GPU lock-hours** (capture/parity and one kernel design followed by a serving gate), within one GPU-day. This prototype is a proposal only; QL1 made no production implementation changes and measured no optimized path.

## Reproducibility and limits

Production commit: `7b4d539f9bd896265f498fabccc6466e45ffe818`; tracked diff was empty at preparation. The launchers are untracked production files, so their actual bytes, rather than only the commit, are separately hashed in `workloads/longctx/frozen.json`. Model: `$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5`, production `hot2_49152` token map and P2 defaults. Per-arm `command.json`, `server.log`, readiness/final server info, `kv-budget.json`, memory samples, response metadata, profile-trigger emission bounds, exact outputs, and lock-hours are retained under `prof/traces/ql1-20260908-ql1-{w4,w16}/`.

The 131072 KV cap was a measurement resource setting passed through the environment; it does not edit a launcher. The production model context limit and index launch capacity remained 262144. With the recorded 856-token trace/output reserve, the capped allocation has a conservative prompt allowance of 130216 tokens; only through 98304 input tokens was exercised. The uncapped maximum and contexts above 96k are not claimed as verified.

A performance cell pools three 128-token requests for each of three prompt families, not independent draws from a broad production workload population. Single-request t/s ranges are saved in `aggregate.json` and can be broad, particularly at W16. The archive-report tasks have low W16 acceptance, so these rates do not describe production W16 code-edit throughput. No statistical shipping claim or optimization gain is made. CPU-start trace cadence and wall throughput come from separate diagnostic and uninstrumented requests in the same server arm; acceptance from one is not used to manufacture a speedup from the other.

The original CUPTI files were never rewritten; `trace-manifest.json` records all 24 file sizes and SHA256 hashes. Analysis code evolved during the final boundary audit; final analysis-script hashes are in `validation.json`, while `frozen.json` retains the pre-run snapshot.

`validation.json` verifies all requested prompt/output lengths, all trace counts, 24 exact needle answers, KV capacity, the 4 GiB steady-state VRAM gate, and frozen source hashes. `process-cleanup.json` confirms no survivors in either owned server process group. `runtime-cache/` is ignored by Git. No unrelated existing work was edited, and no commit or push was performed.

## Final generated tables

## Aggregate context map

Three prompt-family traces are equally weighted for time. Decode t/s pools nine unprofiled 128-token requests. Component cells show exclusive microseconds (% of mean step). The remainder includes other kernels, all shared intervals, and idle gaps.

| Width | Context | Mean step ms | Decode t/s | Tokens/verify | Scan | Score/top-k | Packing | Paged attention | Remainder us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w4 | 8k | 9.427 | 258.4 | 2.477 | 68.2 (0.72%) | 103.5 (1.10%) | 70.4 (0.75%) | 10.4 (0.11%) | 9174.7 |
| w4 | 32k | 9.586 | 261.3 | 2.521 | 70.6 (0.74%) | 112.1 (1.17%) | 69.7 (0.73%) | 10.5 (0.11%) | 9323.3 |
| w4 | 64k | 9.756 | 263.0 | 2.577 | 122.4 (1.25%) | 181.3 (1.86%) | 79.1 (0.81%) | 10.6 (0.11%) | 9362.6 |
| w4 | 96k | 9.773 | 251.2 | 2.472 | 144.8 (1.48%) | 229.8 (2.35%) | 66.8 (0.68%) | 10.7 (0.11%) | 9320.7 |
| w16 | 8k | 16.999 | 167.6 | 2.873 | 161.8 (0.95%) | 106.4 (0.63%) | 214.7 (1.26%) | 18.3 (0.11%) | 16497.4 |
| w16 | 32k | 17.189 | 162.3 | 2.873 | 248.3 (1.44%) | 138.0 (0.80%) | 219.3 (1.28%) | 18.6 (0.11%) | 16565.0 |
| w16 | 64k | 17.712 | 158.7 | 2.824 | 376.6 (2.13%) | 205.0 (1.16%) | 229.9 (1.30%) | 18.5 (0.10%) | 16881.5 |
| w16 | 96k | 17.786 | 161.3 | 2.851 | 495.9 (2.79%) | 258.6 (1.45%) | 237.9 (1.34%) | 21.9 (0.12%) | 16771.9 |

## Aggregate removable budgets

Cells: full-20-GPU-step optimistic raw deletion ceiling in microseconds / share of mean CPU-start step / maximum throughput gain. The exclusive table retains the official 19-CPU-interval window. Zero replacement cost and unchanged acceptance; ceilings overlap and are not additive. Per-family raw and exclusive values are in the detailed table.

| Width | Context | Direct selected access | Shared recursive prefix | Query-row tiling | Score/top-k integration | Required saving us |
|---|---:|---:|---:|---:|---:|---:|
| w4 | 8k | 83.9 / 0.89% / +0.90% | 7.9 / 0.08% / +0.08% | 98.5 / 1.04% / +1.06% | 234.2 / 2.48% / +2.55% | 274.6 |
| w4 | 32k | 80.0 / 0.83% / +0.84% | 3.6 / 0.04% / +0.04% | 89.9 / 0.94% / +0.95% | 246.9 / 2.58% / +2.64% | 279.2 |
| w4 | 64k | 88.0 / 0.90% / +0.91% | 7.2 / 0.07% / +0.07% | 145.6 / 1.49% / +1.51% | 377.2 / 3.87% / +4.02% | 284.2 |
| w4 | 96k | 76.2 / 0.78% / +0.79% | 3.6 / 0.04% / +0.04% | 166.6 / 1.70% / +1.73% | 449.3 / 4.60% / +4.82% | 284.6 |
| w16 | 8k | 239.9 / 1.41% / +1.43% | 48.9 / 0.29% / +0.29% | 174.7 / 1.03% / +1.04% | 314.8 / 1.85% / +1.89% | 495.1 |
| w16 | 32k | 245.3 / 1.43% / +1.45% | 49.4 / 0.29% / +0.29% | 270.5 / 1.57% / +1.60% | 449.1 / 2.61% / +2.68% | 500.7 |
| w16 | 64k | 255.5 / 1.44% / +1.46% | 58.6 / 0.33% / +0.33% | 399.7 / 2.26% / +2.31% | 655.1 / 3.70% / +3.84% | 515.9 |
| w16 | 96k | 267.7 / 1.51% / +1.53% | 52.8 / 0.30% / +0.30% | 533.3 / 3.00% / +3.09% | 848.7 / 4.77% / +5.01% | 518.0 |

## Aggregate scaling fits

y(us) = intercept + slope * context_k, k=1024; four context means. Aggregate raw fits use all 20 complete GPU iterations; exclusive fits retain the official 19-CPU-interval window. Detailed per-family CPU-window fits are reported separately. No extrapolation beyond 96k.

| Width | Component | Metric | Intercept us | Slope us/k | R2 | RMSE | Constant RMSE | 96k/64k |
|---|---|---|---:|---:|---:|---:|---:|---:|
| w4 | scan | raw | 80.30 | 0.897 | 0.866 | 11.70 | 31.96 | 1.144 |
| w4 | scan | exclusive | 53.31 | 0.964 | 0.934 | 8.51 | 33.07 | 1.183 |
| w4 | score_topk | raw | 113.98 | 1.755 | 0.981 | 8.08 | 58.78 | 1.220 |
| w4 | score_topk | exclusive | 80.00 | 1.533 | 0.960 | 10.36 | 51.90 | 1.268 |
| w4 | packing | raw | 84.64 | -0.052 | 0.154 | 4.06 | 4.41 | 0.866 |
| w4 | packing | exclusive | 71.83 | -0.007 | 0.002 | 4.58 | 4.59 | 0.845 |
| w4 | paged_attention | raw | 261.12 | -0.014 | 0.816 | 0.22 | 0.52 | 0.998 |
| w4 | paged_attention | exclusive | 10.35 | 0.004 | 0.985 | 0.01 | 0.12 | 1.007 |
| w4 | other | raw | 10947.15 | 0.704 | 0.324 | 33.73 | 41.02 | 0.994 |
| w4 | other | exclusive | 6994.05 | 0.153 | 0.028 | 29.93 | 30.35 | 0.992 |
| w4 | step | wall | 9435.03 | 4.010 | 0.893 | 46.06 | 140.74 | 1.002 |
| w16 | scan | raw | 140.93 | 4.073 | 1.000 | 1.35 | 135.08 | 1.334 |
| w16 | scan | exclusive | 129.53 | 3.822 | 1.000 | 2.31 | 126.80 | 1.317 |
| w16 | score_topk | raw | 120.23 | 2.043 | 0.995 | 4.56 | 67.90 | 1.235 |
| w16 | score_topk | exclusive | 88.20 | 1.776 | 0.995 | 4.36 | 59.06 | 1.261 |
| w16 | packing | raw | 236.22 | 0.318 | 0.990 | 1.07 | 10.60 | 1.048 |
| w16 | packing | exclusive | 211.85 | 0.272 | 0.993 | 0.77 | 9.05 | 1.035 |
| w16 | paged_attention | raw | 529.70 | 0.276 | 0.552 | 8.25 | 12.33 | 0.978 |
| w16 | paged_attention | exclusive | 17.46 | 0.037 | 0.676 | 0.86 | 1.51 | 1.186 |
| w16 | other | raw | 19057.67 | 1.232 | 0.131 | 105.33 | 112.97 | 0.994 |
| w16 | other | exclusive | 13101.49 | -0.175 | 0.006 | 75.29 | 75.51 | 0.993 |
| w16 | step | wall | 16935.23 | 9.723 | 0.924 | 92.20 | 335.39 | 1.004 |

## Detailed per-family measurements and fits

## Measured per-context map (generated)

Times are microseconds per complete decode interval. Component cells are exclusive / raw; wall is mean / median. Other exclusive plus shared plus idle complete the wall partition. Throughput is the pooled rate from three unprofiled 128-token requests.

| Width | Context | Family | Step mean / median | Decode t/s | Tokens/verify | Scan | Score + top-k | Packing | Paged attention | Other excl | Shared | Gap |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w4 | 8k | doc-a | 9144.9 / 9489.6 | 246.6 | 2.341 | 68.3 / 97.1 | 93.6 / 121.3 | 72.7 / 78.7 | 10.4 / 233.7 | 6958.3 | 1773.0 | 168.6 |
| w4 | 8k | doc-b | 9526.3 / 9572.7 | 253.7 | 2.446 | 67.6 / 94.6 | 112.5 / 140.1 | 76.4 / 83.1 | 10.4 / 258.4 | 6936.0 | 1795.2 | 528.2 |
| w4 | 8k | needle-perf | 9610.0 / 9633.1 | 276.8 | 2.667 | 68.6 / 95.3 | 104.3 / 131.3 | 62.1 / 69.4 | 10.3 / 258.0 | 6995.8 | 1834.3 | 534.7 |
| w4 | 32k | doc-a | 9523.2 / 9575.1 | 237.7 | 2.272 | 73.9 / 89.4 | 111.3 / 143.7 | 80.2 / 87.0 | 10.6 / 259.8 | 6990.3 | 1741.7 | 515.2 |
| w4 | 32k | doc-b | 9534.6 / 9586.3 | 271.0 | 2.630 | 75.9 / 90.8 | 112.5 / 159.0 | 64.4 / 71.3 | 10.4 / 257.8 | 6928.3 | 1817.0 | 526.1 |
| w4 | 32k | needle-perf | 9701.0 / 9733.1 | 279.2 | 2.704 | 62.1 / 77.6 | 112.6 / 145.1 | 64.5 / 71.6 | 10.3 / 233.6 | 7162.1 | 1775.2 | 514.2 |
| w4 | 64k | doc-a | 9590.1 / 9712.1 | 261.7 | 2.543 | 126.6 / 142.5 | 176.4 / 212.3 | 66.1 / 72.4 | 10.7 / 268.9 | 6931.3 | 1789.4 | 489.7 |
| w4 | 64k | doc-b | 9881.4 / 9924.5 | 254.9 | 2.526 | 127.0 / 143.7 | 200.0 / 236.4 | 88.7 / 94.8 | 10.7 / 233.8 | 7106.8 | 1819.8 | 528.5 |
| w4 | 64k | needle-perf | 9796.4 / 9912.9 | 273.0 | 2.667 | 113.6 / 130.0 | 167.4 / 216.5 | 82.5 / 88.2 | 10.5 / 247.2 | 7068.7 | 1818.1 | 535.7 |
| w4 | 96k | doc-a | 9713.7 / 9721.8 | 235.4 | 2.299 | 150.3 / 165.5 | 241.6 / 281.6 | 67.2 / 72.3 | 10.8 / 244.8 | 6922.4 | 1769.9 | 551.5 |
| w4 | 96k | doc-b | 9738.1 / 9761.0 | 253.0 | 2.510 | 150.1 / 164.9 | 227.0 / 267.4 | 66.7 / 72.7 | 10.6 / 246.5 | 6988.1 | 1781.5 | 514.1 |
| w4 | 96k | needle-perf | 9866.6 / 9947.2 | 267.2 | 2.630 | 134.0 / 150.0 | 220.9 / 261.8 | 66.6 / 72.4 | 10.6 / 255.9 | 7032.1 | 1878.0 | 524.4 |
| w16 | 8k | doc-a | 16199.9 / 16842.8 | 152.2 | 2.595 | 151.8 / 156.5 | 98.0 / 125.0 | 215.8 / 230.8 | 18.4 / 498.9 | 12866.5 | 2638.3 | 211.1 |
| w16 | 8k | doc-b | 17530.8 / 17480.3 | 162.0 | 2.783 | 179.8 / 187.0 | 110.8 / 137.9 | 213.5 / 228.3 | 18.3 / 503.4 | 13428.9 | 2663.5 | 916.1 |
| w16 | 8k | needle-perf | 17265.1 / 17185.0 | 194.0 | 3.339 | 153.8 / 159.8 | 110.4 / 137.7 | 214.8 / 228.6 | 18.3 / 485.7 | 13179.6 | 2610.9 | 977.4 |
| w16 | 32k | doc-a | 17001.7 / 17064.3 | 141.5 | 2.430 | 254.4 / 275.0 | 142.7 / 173.5 | 214.7 / 229.8 | 18.9 / 563.2 | 12743.0 | 2695.1 | 932.9 |
| w16 | 32k | doc-b | 17207.7 / 17262.9 | 164.0 | 3.048 | 252.4 / 258.4 | 142.1 / 174.3 | 215.1 / 230.6 | 18.3 / 501.2 | 12999.4 | 2646.6 | 933.7 |
| w16 | 32k | needle-perf | 17358.1 / 17389.7 | 188.1 | 3.282 | 238.2 / 243.4 | 129.1 / 162.0 | 228.1 / 243.3 | 18.4 / 480.0 | 13194.8 | 2628.9 | 920.6 |
| w16 | 64k | doc-a | 17093.5 / 17124.7 | 152.8 | 2.667 | 393.4 / 398.2 | 194.8 / 230.9 | 212.4 / 228.1 | 18.4 / 578.0 | 12637.5 | 2690.9 | 946.1 |
| w16 | 64k | doc-b | 17646.1 / 17475.8 | 158.2 | 2.803 | 341.0 / 347.1 | 207.0 / 242.5 | 215.7 / 229.7 | 18.4 / 535.9 | 13150.9 | 2738.7 | 974.3 |
| w16 | 64k | needle-perf | 18395.0 / 18467.4 | 165.5 | 3.024 | 395.3 / 401.1 | 213.4 / 250.0 | 261.5 / 278.6 | 18.6 / 507.8 | 13703.3 | 2782.3 | 1020.6 |
| w16 | 96k | doc-a | 17494.7 / 17492.2 | 138.3 | 2.430 | 524.0 / 529.2 | 238.9 / 278.0 | 202.2 / 217.2 | 18.4 / 537.5 | 12834.3 | 2737.9 | 939.0 |
| w16 | 96k | doc-b | 18156.2 / 18175.6 | 150.9 | 2.685 | 494.8 / 512.4 | 264.9 / 303.0 | 210.3 / 225.8 | 29.1 / 540.8 | 13467.3 | 2695.3 | 994.6 |
| w16 | 96k | needle-perf | 17707.6 / 17749.6 | 210.8 | 3.728 | 468.9 / 474.6 | 272.0 / 325.2 | 301.2 / 315.8 | 18.3 / 501.8 | 12907.7 | 2764.7 | 974.8 |

## Removal ceilings (generated)

Each cell: 19-CPU-interval exclusive credit / full-20-GPU-step raw-duration ceiling (microseconds), then ideal throughput gain from the raw ceiling. Shared prefix retains one recursive copy. Integration includes the entire score-producing scan: this intentionally loose ceiling cannot separate arithmetic from materialization. Top-k/init-only is shown to expose that uncertainty. These ideas overlap and must not be added.

| Width | Context | Family | Direct access | Shared prefix | Query tiling | Score integration | Top-k/init only | 3% threshold us |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| w4 | 8k | doc-a | 72.7 / 93.2; +1.03% | 3.3 / 3.6; +0.04% | 68.3 / 99.8; +1.10% | 164.1 / 226.2; +2.54% | 95.8 / 126.4; +1.40% | 266.4 |
| w4 | 8k | doc-b | 76.4 / 85.8; +0.91% | 17.2 / 16.6; +0.17% | 67.6 / 97.4; +1.03% | 182.3 / 242.0; +2.61% | 114.8 / 144.5; +1.54% | 277.5 |
| w4 | 8k | needle-perf | 62.1 / 72.9; +0.76% | 3.5 / 3.5; +0.04% | 68.6 / 98.2; +1.03% | 175.1 / 234.3; +2.50% | 106.5 / 136.1; +1.44% | 279.9 |
| w4 | 32k | doc-a | 80.2 / 89.7; +0.95% | 3.5 / 3.6; +0.04% | 73.9 / 93.2; +0.99% | 187.4 / 245.0; +2.64% | 113.6 / 151.8; +1.62% | 277.4 |
| w4 | 32k | doc-b | 64.4 / 74.9; +0.79% | 3.5 / 3.6; +0.04% | 75.9 / 94.5; +1.00% | 190.6 / 260.7; +2.81% | 114.6 / 166.2; +1.77% | 277.7 |
| w4 | 32k | needle-perf | 64.5 / 75.4; +0.78% | 3.3 / 3.6; +0.04% | 62.1 / 82.0; +0.85% | 176.8 / 234.9; +2.48% | 114.7 / 152.9; +1.60% | 282.6 |
| w4 | 64k | doc-a | 66.1 / 76.0; +0.80% | 3.6 / 3.7; +0.04% | 126.6 / 149.1; +1.58% | 305.2 / 371.7; +4.03% | 178.5 / 222.6; +2.38% | 279.3 |
| w4 | 64k | doc-b | 88.7 / 97.1; +0.99% | 14.9 / 14.4; +0.15% | 127.0 / 150.3; +1.54% | 329.2 / 395.8; +4.17% | 202.2 / 245.5; +2.55% | 287.8 |
| w4 | 64k | needle-perf | 82.5 / 90.9; +0.94% | 3.6 / 3.6; +0.04% | 113.6 / 137.2; +1.42% | 283.2 / 364.1; +3.86% | 169.6 / 226.9; +2.37% | 285.3 |
| w4 | 96k | doc-a | 67.2 / 76.1; +0.79% | 3.6 / 3.6; +0.04% | 150.3 / 171.6; +1.80% | 394.1 / 465.2; +5.03% | 243.8 / 293.7; +3.12% | 282.9 |
| w4 | 96k | doc-b | 66.7 / 76.5; +0.79% | 3.3 / 3.6; +0.04% | 150.1 / 171.1; +1.79% | 379.2 / 451.3; +4.86% | 229.1 / 280.1; +2.96% | 283.6 |
| w4 | 96k | needle-perf | 66.6 / 75.9; +0.78% | 3.7 / 3.7; +0.04% | 134.0 / 157.0; +1.62% | 357.2 / 431.3; +4.57% | 223.2 / 274.3; +2.86% | 287.4 |
| w16 | 8k | doc-a | 215.8 / 241.5; +1.51% | 45.0 / 49.5; +0.31% | 151.8 / 164.1; +1.02% | 251.9 / 296.1; +1.86% | 100.2 / 131.9; +0.82% | 471.8 |
| w16 | 8k | doc-b | 213.5 / 239.1; +1.38% | 44.3 / 48.9; +0.28% | 179.8 / 192.9; +1.11% | 292.7 / 337.1; +1.96% | 112.9 / 144.2; +0.83% | 510.6 |
| w16 | 8k | needle-perf | 214.8 / 239.2; +1.40% | 43.7 / 48.2; +0.28% | 153.8 / 167.1; +0.98% | 266.3 / 311.4; +1.84% | 112.5 / 144.3; +0.84% | 502.9 |
| w16 | 32k | doc-a | 214.7 / 240.9; +1.44% | 44.6 / 49.4; +0.29% | 254.4 / 285.7; +1.71% | 399.3 / 467.7; +2.83% | 144.9 / 182.0; +1.08% | 495.2 |
| w16 | 32k | doc-b | 215.1 / 241.5; +1.42% | 44.6 / 49.2; +0.29% | 252.4 / 270.2; +1.60% | 396.6 / 452.7; +2.70% | 144.2 / 182.5; +1.07% | 501.2 |
| w16 | 32k | needle-perf | 228.1 / 253.6; +1.48% | 44.7 / 49.5; +0.29% | 238.2 / 255.7; +1.50% | 369.4 / 426.8; +2.52% | 131.2 / 171.0; +1.00% | 505.6 |
| w16 | 64k | doc-a | 212.4 / 239.2; +1.42% | 45.2 / 50.0; +0.29% | 393.4 / 414.8; +2.49% | 590.4 / 657.0; +4.00% | 196.9 / 242.2; +1.44% | 497.9 |
| w16 | 64k | doc-b | 215.7 / 240.4; +1.38% | 58.4 / 62.2; +0.35% | 341.0 / 366.5; +2.12% | 550.1 / 629.6; +3.70% | 209.1 / 263.1; +1.51% | 514.0 |
| w16 | 64k | needle-perf | 261.5 / 286.9; +1.58% | 59.8 / 63.5; +0.35% | 395.3 / 417.8; +2.32% | 610.8 / 678.6; +3.83% | 215.5 / 260.8; +1.44% | 535.8 |
| w16 | 96k | doc-a | 202.2 / 243.4; +1.41% | 45.4 / 50.3; +0.29% | 524.0 / 551.5; +3.25% | 765.1 / 843.8; +5.07% | 241.1 / 292.4; +1.70% | 509.6 |
| w16 | 96k | doc-b | 210.3 / 237.0; +1.32% | 54.5 / 58.9; +0.33% | 494.8 / 535.5; +3.04% | 761.8 / 851.5; +4.92% | 267.0 / 316.0; +1.77% | 528.8 |
| w16 | 96k | needle-perf | 301.2 / 322.8; +1.86% | 44.4 / 49.4; +0.28% | 468.9 / 512.9; +2.98% | 743.0 / 850.9; +5.05% | 274.1 / 338.0; +1.95% | 515.8 |

## Context fits (generated)

OLS y = intercept + slope * context_k, k=1024. Four points per width/family. R2 is descriptive, not an inferential test. The ratio compares 96k to 64k; a flat final ratio alone is insufficient to establish saturation.

| Width | Family | Metric | Component | Intercept us | Slope us/k | R2 | RMSE us | Constant RMSE | 96k / 64k |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| w4 | doc-a | raw_us | scan | 79.07 | 0.891 | 0.877 | 11.06 | 31.56 | 1.161 |
| w4 | doc-a | raw_us | score_topk | 96.00 | 1.874 | 0.981 | 8.71 | 62.76 | 1.327 |
| w4 | doc-a | raw_us | packing | 83.67 | -0.121 | 0.444 | 4.49 | 6.02 | 0.999 |
| w4 | doc-a | raw_us | paged_attention | 245.77 | 0.121 | 0.088 | 12.92 | 13.52 | 0.910 |
| w4 | doc-a | raw_us | other | 10328.86 | -0.472 | 0.499 | 15.69 | 22.17 | 0.997 |
| w4 | doc-a | exclusive_us | scan | 53.80 | 1.019 | 0.947 | 8.02 | 34.75 | 1.187 |
| w4 | doc-a | exclusive_us | score_topk | 68.80 | 1.738 | 0.975 | 9.20 | 58.38 | 1.370 |
| w4 | doc-a | exclusive_us | packing | 77.03 | -0.110 | 0.424 | 4.24 | 5.59 | 1.016 |
| w4 | doc-a | exclusive_us | paged_attention | 10.43 | 0.004 | 0.901 | 0.04 | 0.14 | 1.012 |
| w4 | doc-a | exclusive_us | other | 6980.18 | -0.592 | 0.551 | 17.74 | 26.46 | 0.999 |
| w4 | doc-b | raw_us | scan | 78.14 | 0.907 | 0.899 | 10.10 | 31.73 | 1.148 |
| w4 | doc-b | raw_us | score_topk | 122.70 | 1.561 | 0.962 | 10.34 | 52.78 | 1.131 |
| w4 | doc-b | raw_us | packing | 81.64 | -0.023 | 0.007 | 9.42 | 9.45 | 0.767 |
| w4 | doc-b | raw_us | paged_attention | 259.15 | -0.201 | 0.439 | 7.53 | 10.06 | 1.054 |
| w4 | doc-b | raw_us | other | 10375.77 | 0.953 | 0.097 | 96.60 | 101.63 | 0.979 |
| w4 | doc-b | exclusive_us | scan | 54.25 | 1.018 | 0.957 | 7.18 | 34.51 | 1.182 |
| w4 | doc-b | exclusive_us | score_topk | 89.25 | 1.475 | 0.906 | 15.73 | 51.38 | 1.135 |
| w4 | doc-b | exclusive_us | packing | 74.67 | -0.013 | 0.002 | 9.58 | 9.59 | 0.752 |
| w4 | doc-b | exclusive_us | paged_attention | 10.41 | 0.003 | 0.733 | 0.05 | 0.10 | 0.996 |
| w4 | doc-b | exclusive_us | other | 6933.81 | 1.120 | 0.271 | 60.92 | 71.34 | 0.983 |
| w4 | needle-perf | raw_us | scan | 75.46 | 0.755 | 0.777 | 13.42 | 28.42 | 1.154 |
| w4 | needle-perf | raw_us | score_topk | 109.69 | 1.580 | 0.970 | 9.27 | 53.20 | 1.209 |
| w4 | needle-perf | raw_us | packing | 71.34 | 0.082 | 0.131 | 6.98 | 7.48 | 0.820 |
| w4 | needle-perf | raw_us | paged_attention | 246.51 | 0.043 | 0.022 | 9.49 | 9.60 | 1.035 |
| w4 | needle-perf | raw_us | other | 10487.64 | 1.073 | 0.665 | 25.25 | 43.65 | 1.006 |
| w4 | needle-perf | exclusive_us | scan | 51.90 | 0.854 | 0.879 | 10.50 | 30.20 | 1.180 |
| w4 | needle-perf | exclusive_us | score_topk | 81.96 | 1.387 | 0.959 | 9.49 | 46.96 | 1.319 |
| w4 | needle-perf | exclusive_us | packing | 63.80 | 0.103 | 0.181 | 7.23 | 7.99 | 0.808 |
| w4 | needle-perf | exclusive_us | paged_attention | 10.21 | 0.004 | 0.982 | 0.02 | 0.15 | 1.014 |
| w4 | needle-perf | exclusive_us | other | 7068.16 | -0.070 | 0.001 | 61.85 | 61.89 | 0.995 |
| w16 | doc-a | raw_us | scan | 130.72 | 4.181 | 0.998 | 6.66 | 138.82 | 1.329 |
| w16 | doc-a | raw_us | score_topk | 114.93 | 1.738 | 0.995 | 3.88 | 57.79 | 1.204 |
| w16 | doc-a | raw_us | packing | 233.82 | -0.147 | 0.796 | 2.46 | 5.45 | 0.952 |
| w16 | doc-a | raw_us | paged_attention | 524.76 | 0.393 | 0.188 | 27.05 | 30.02 | 0.930 |
| w16 | doc-a | raw_us | other | 17742.69 | 0.922 | 0.064 | 117.06 | 120.99 | 1.019 |
| w16 | doc-a | exclusive_us | scan | 118.85 | 4.241 | 1.000 | 1.94 | 140.66 | 1.332 |
| w16 | doc-a | exclusive_us | score_topk | 88.71 | 1.598 | 0.996 | 3.34 | 53.10 | 1.227 |
| w16 | doc-a | exclusive_us | packing | 218.71 | -0.149 | 0.840 | 2.15 | 5.37 | 0.952 |
| w16 | doc-a | exclusive_us | paged_attention | 18.65 | -0.002 | 0.120 | 0.22 | 0.24 | 0.997 |
| w16 | doc-a | exclusive_us | other | 12798.29 | -0.559 | 0.043 | 87.10 | 89.06 | 1.016 |
| w16 | doc-b | raw_us | scan | 145.28 | 3.619 | 0.976 | 18.98 | 121.52 | 1.476 |
| w16 | doc-b | raw_us | score_topk | 118.91 | 1.910 | 0.997 | 3.48 | 63.44 | 1.249 |
| w16 | doc-b | raw_us | packing | 230.19 | -0.031 | 0.326 | 1.49 | 1.82 | 0.983 |
| w16 | doc-b | raw_us | paged_attention | 495.12 | 0.504 | 0.849 | 7.06 | 18.14 | 1.009 |
| w16 | doc-b | raw_us | other | 18194.69 | 2.234 | 0.125 | 196.24 | 209.76 | 1.014 |
| w16 | doc-b | exclusive_us | scan | 141.58 | 3.509 | 0.982 | 15.83 | 117.45 | 1.451 |
| w16 | doc-b | exclusive_us | score_topk | 91.72 | 1.789 | 0.995 | 4.26 | 59.49 | 1.280 |
| w16 | doc-b | exclusive_us | packing | 215.30 | -0.033 | 0.265 | 1.82 | 2.12 | 0.975 |
| w16 | doc-b | exclusive_us | paged_attention | 15.37 | 0.113 | 0.651 | 2.75 | 4.66 | 1.580 |
| w16 | doc-b | exclusive_us | other | 13197.93 | 1.274 | 0.047 | 189.85 | 194.50 | 1.024 |
| w16 | needle-perf | raw_us | scan | 133.84 | 3.717 | 0.980 | 17.49 | 124.52 | 1.183 |
| w16 | needle-perf | raw_us | score_topk | 107.85 | 2.218 | 0.980 | 10.59 | 74.32 | 1.301 |
| w16 | needle-perf | raw_us | packing | 216.07 | 1.010 | 0.987 | 3.82 | 33.73 | 1.133 |
| w16 | needle-perf | raw_us | paged_attention | 480.73 | 0.262 | 0.585 | 7.32 | 11.36 | 0.988 |
| w16 | needle-perf | raw_us | other | 18217.58 | 2.421 | 0.051 | 347.55 | 356.71 | 0.956 |
| w16 | needle-perf | exclusive_us | scan | 128.15 | 3.718 | 0.980 | 17.41 | 124.53 | 1.186 |
| w16 | needle-perf | exclusive_us | score_topk | 84.18 | 1.941 | 0.974 | 10.45 | 65.22 | 1.274 |
| w16 | needle-perf | exclusive_us | packing | 201.55 | 0.997 | 0.980 | 4.68 | 33.40 | 1.152 |
| w16 | needle-perf | exclusive_us | paged_attention | 18.35 | 0.001 | 0.076 | 0.14 | 0.14 | 0.983 |
| w16 | needle-perf | exclusive_us | other | 13308.27 | -1.239 | 0.020 | 284.54 | 287.49 | 0.942 |
