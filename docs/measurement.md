# Measurement methodology and noise

Most wrong turns in this project came from measurement, not from kernels.
This page describes how the numbers in the other documents were produced,
what each metric means, and which sources of noise were found and how large
they are.

## 1. Setup

* One RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB GDDR7, 188 SMs, 128 MiB L2),
  power cap 300-325 W. The same GPU drives the desktop: a 4K 160 Hz display
  under KWin, plus a browser and other desktop applications.
* Batch size 1, one request at a time. Speculative decoding uses the model's
  native MTP layer as a NEXTN/EAGLE linear chain (top-k = 1).
* Benchmark harness: a small client (`fnbench`) with fixed workloads
  (`code-edit`, `prose-en`, `prose-ja`, `agent-loop`; early on also
  `long-ctx`), a needle-in-a-haystack test at 18.5k tokens (thinking off,
  depths 0.4 / 0.85; the context is synthetic, built in `prof/needle_test.py`),
  a `sanity_gen` degeneration detector (one fixed story prompt written in
  `prof/sanity_gen.py`; flags acceptance above 12, repetition and low
  vocabulary diversity), and a validation script
  that starts a server, profiles about 20 decode steps, runs the benchmark
  and the needle test, then stops the server.

## 2. Greedy vs sampling

* The first GPU-day numbers (2026-09-02 morning) used the model's recommended
  sampling. Code-edit acceptance at temperature 0.2 ranged from 7.2 to 11.7 on
  the **same** configuration, because the seed changes every server start. One
  early "regression" (FP8 router gates) may have been inside that spread.
* From 2026-09-02 afternoon all comparisons are **greedy** (temperature 0).
* Greedy is still **not bit-reproducible** on this stack:
  * the MoE GEMM2 fused-finalize epilogue scatter-adds with `red.global.add`,
    so MoE outputs differ at about 1e-4 absolute between runs;
  * the HC mix K1 kernel accumulates its down-projection with device-scope
    atomics (the same unfused call on the same input differed in 5 of 5
    repeats at M = 16);
  * the P1/P2 pruning decision depends on the union of routes in the verify
    batch, so a different draft can change the target's arithmetic.
  
  As a result, a "greedy output identical" gate cannot be used as a
  correctness test. Acceptance length moves by about ±4-6 % (sometimes ±15 %
  on W16 code-edit) between repeats of the same build. Comparisons therefore
  use at least 4 repeats, or the held-out prompt sets described in §6.
* Bit-identity was tested at kernel level instead: `torch.equal` on real
  layer weights, or a deterministic mode
  (`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` plus `--autotune-cache none`,
  because the production autotune cache indexes tactics by position).

## 3. Acceptance definitions

* **Acceptance length** = tokens emitted per verify step, **including the bonus
  token** = `completion_tokens / Δspec_verify_calls_total` (Prometheus counter,
  differenced around a run). In the width sweep, `completion_tokens / SSE
  chunks` was used as well. At bs = 1 there is one token-bearing chunk per
  verify step, and the two agreed within 2-8 %.
* The adaptive controller's statistics use **accepted drafts** (acceptance − 1).
* Do **not** use the Prometheus gauge `sglang:spec_accept_length`. It is a
  windowed value sampled once after a run. On W16 code-edit it read 14.12
  while the run average was 11.0.
* Offline acceptance: during development the draft head was also evaluated
  offline on dumped hidden states from the author's private data, first with
  an `accept@k` metric and from v3 on with a "renewal" evaluator, which replays the server's verify-and-advance process on self-generated
  sequences and reports tokens per verify step. All offline results, including
  how well they matched the server, come from private data and are not
  published. A server A/B on the public workloads was always the final gate,
  and every acceptance number in this repository is a server measurement.

## 4. Step-time metrics

* **Client t/s** (`fnbench`): decode tokens per second from the SSE stream.
  **Reasoning tokens are counted**, because the model thinks by default and
  thinking tokens are decoded like any other. This is the number users see, and
  also the noisiest.
* **Per-step wall from SSE**: `decode_seconds / (chunks − 1)`. This is a sound
  client-side step time at bs = 1. The older estimate `accept_len / tps` was
  unsound, because it relied on the gauge above and on `completion_tokens`, which
  disagreed with the streamed count for thinking outputs.
* **CUPTI trace metrics** (torch profiler, about 20 decode steps per trace):
  * `kernels/step`, the number of kernels launched per step.
  * `busy_ms`, the union of kernel intervals per phase (draft / verify /
    draft_extend), plus idle. This became the primary metric on 2026-09-05.
  * `legacy_trimmed_ms`, the earlier metric: the sum of kernel durations with
    each kernel clipped at `2 × median(by name) + 5 µs` to remove compositor
    preemption spikes. It **under-reported by 0.8-3.6 ms**, and the error grew
    with kernel count, because one kernel name (`_w8a16_gemv_kernel`) covers
    shapes from 4 µs to 420 µs. For a while this looked like "host overhead that
    grows with draft steps". There is no such overhead: GPU idle inside a step
    is 0.15-0.21 ms at every width.
  * **Exclusive time** (sweep-line): time during which a kernel is the *only*
    thing running. This is the critical-path share. Kernels on alternate
    streams that are fully overlapped (the shared-expert/router chain, the GDN
    MTP state stream) have near-zero exclusive time even when their raw time
    is large. Caveat: exclusive time is not the saving from deleting a
    kernel. If the deleted kernel's work has to be redone elsewhere (J1), or it
    was producing a layout that the next kernel needs, little is saved. Under
    PDL, the "overlap" also includes the consumer spinning on its wait.
* Per-kernel medians from the same trace are the most stable comparison.
  `busy_ms` for a whole step drifted ±0.5 ms between sessions from the MoE
  grouped GEMM alone (for example 6.81 -> 7.30 ms exclusive with no code
  change), which is ten times the size of many of the effects measured.
* CUDA-event timing in microbenchmarks includes launch gaps and read about
  25 % higher than CUPTI kernel time. Compare CUPTI with CUPTI.

## 5. Why client t/s swings ±10-20 %

1. **Desktop compositor preemption.** The display runs on the same GPU. At
   4K 160 Hz, KWin preempts the CUDA context. Any kernel can grow by about
   400-700 µs, 2-7 times per step (a total of 2-4.5 ms on a 13-19 ms step in the
   first traces). A/B on 2026-09-05 with compositing off: step wall W4
   12.2 -> 11.3 ms (code-edit) and 12.0 -> 11.1 ms (prose), **-7.5 %**; W16
   -1 % / -4.6 %. Dropping the refresh rate to 60 Hz recovered most of it
   (-5.5 to -6.7 %). The launcher has an opt-in `SERVE_DISPLAY_HZ=60`.
   Streaming output into a desktop chat window is itself a redraw load.
2. **Other GPU contexts clear the L2 persisting set.** A draft-head
   `accessPolicyWindow` pin worked in isolation (draft forward 82 -> 29 µs at
   W4), but the compositor's context switches evicted the persisting lines
   within about 3.5 ms, so in the W16 traces the head read cold (median 81 µs).
3. **Session drift.** Between control arms 30-60 minutes apart, drift of +8
   to +23 % was seen while other GPU applications were running on the desktop. In the daytime
   with a browser open it was −4 to −13 %. Quiet-night measurements are
   consistently faster, so numbers from night and day must not be compared
   directly. The final production figures are quiet-night figures.
4. **Server-to-server variance.** The same configuration differs by about ±6 %
   between two server processes. When possible, A/B within one process, or
   use a fresh server per arm in an ABBA order (§6).
5. **Acceptance variance.** Covered in §2. W16 code-edit acceptance ranged
   8.3-11.6 between runs of the fixed profile.

The 325 W power cap was checked (2026-09-07, X3). None of the 39 nvidia-smi
samples taken at 100 ms intervals inside decode windows showed a power-cap,
thermal or hardware-slowdown reason, and the per-arm median SM clock was
2,242-2,287 MHz. Throttling shorter than the sampling interval cannot be ruled
out, and the cap did engage briefly in each server's lifetime before the first
traced decode window.

## 6. The BN1 protocol (from 2026-09-08)

By 2026-09-08 several candidate optimisations had been judged on "one prompt ×
3 repeats" A/Bs, in which two identical control arms differed by 10-16 %. The
BN1 study (`lab-notes/BN1_BENCH_NOISE.md`) measured production noise and replaced
that practice with the protocol below.

**Prompt sets.**
* 8 held-out prompts per domain (code-edit, prose-en, prose-ja, agent-loop),
  32 in total.
* The prompts were written deterministically and audited against the
  author's private training and capture prompt files: NFKC-normalised exact and
  containment checks plus 50-character shingle overlap, with a 20 % review
  threshold. The audit's result is not published (evaluation on private data).
  These prompts are reserved for evaluation.
* Token caps: code-edit 12,288, prose 6,144, agent-loop 8,192. Early EOS is
  kept.
* Run protocol: greedy, one request at a time, one 1,024-token warmup per
  server start.

**Metrics.**
* Throughput = `(completion_tokens − 1) / (last_token_time −
  first_token_time)`.
* Acceptance = generated tokens per verify call, read per request from
  counters bracketed around the request. Missing, changed or negative counters
  invalidate the arm; they are never read as zero.

**Design.**
* A fresh server per arm, in ABBA order.
* Paired analysis: per-prompt log ratios averaged over the two blocks, then a
  prompt-level bootstrap (20,000 draws, fixed seed) for the CI and the
  one-sided lower bound.
* An A2/A1 drift diagnostic flags a cycle when |drift| > 3 %.
* With several cycles, use a crossed cycle + prompt interval.

**Measured residual CV across three restarts** (t/s; acceptance similar):

| Profile | code-edit | prose-en | prose-ja | agent-loop |
|---|--:|--:|--:|--:|
| W4 | 1.4 % | 5.9 % | 5.9 % | 2.7 % |
| wa | 7.6 % | 3.2 % | 19.6 % | 3.9 % |

The large wa prose-ja value comes mostly from one prompt. In one restart it
ran at 629 t/s (7.6 tokens per verify), against 252-262 t/s in the other two,
because the adaptive policy happened to go wide on it. Production leaves the
server seed unset, and deterministic inference is off, so every restart is a
new draw even under greedy decoding.

**Minimum detectable effect** for one ABBA cycle with 8 prompts, one-sided α = 0.05,
80 % power, including restart noise:

* W4: code-edit about 2 %, agent-loop about 4 %, prose about 10 %.
* wa: 6-31 %.

Detecting a 3 % throughput effect in every domain would take about **11 ABBA
cycles for W4 (about 11 GPU-hours) and about 86 for wa (about 81 GPU-hours)**.

**Conclusion.** A uniform "≥3 % in every domain" gate cannot be met in 1-2
hours of GPU time. Before running, declare the primary domain and metric and a
cycle budget. If the budget cannot fund the cycles, the verdict is
"inconclusive", not "fail". Some results easily clear the MDE and remain valid,
for example WA5's agent-loop +14.4 %, reproduced in both pairs. Many earlier
wa-level "FAIL" verdicts were really "not resolvable".

## 7. Quality gates for changes that alter the target's arithmetic

Most optimisations here are bit-exact, or within 1-2 bf16 ulp from a
different split-K partition. For those, the gates are acceptance, needle and
sanity. Changes that alter what the target model computes (singleton pruning,
NVFP4 lm_head, sparse FP4, expert-width or HC-rank truncation) must also pass a
quality battery. The battery is GSM8K (1,319), MMLU (14 subjects × 100),
HumanEval (164), JCommonsenseQA (1,119), needle at 3 depths, and long-form
generation, run greedy with thinking off. Each arm is compared by McNemar on
paired items, with Wilson intervals, and later by a one-sided non-inferiority
test (margin 0.5 pp). Two harness defects were found and fixed:

* the regression flag's sign was inverted, so it fired when production was
  *better*;
* the first audit ran two requests at a time, not bs = 1, which weakens
  pruning (it depends on how many rows share an expert). The harness now
  defaults to bs = 1.

On HumanEval's 164 problems a 0.5 pp margin is underpowered when the
difference is small: one discordant problem is 0.61 pp, so a PASS needs a
clearly favourable split (Q1 prod2 passed; noprune and legacy did not), and
non-inferiority was not established in the other comparisons. A later gate
(T10) came back inconclusive.

## 8. Operational lessons

* **Microbench vs in-server.** Several microbenchmarks were 1.2-3× off the
  server. The cause was usually a wrong setup, not "clock noise". The main
  case: the W8A16 GEMV tuner benchmarked `[K, N]`-contiguous weights while the
  server passes `[N, K]`-contiguous ones, so the whole v2 tuning table was
  never used (the server ran the fallback tile). The rule since then: before
  tuning, a benchmark must reproduce the in-server per-shape CUPTI medians
  within about ±20 %, and adoption is decided by the in-server trace. Small
  launch-bound shapes (3-4 CTAs) cannot be resolved by microbenchmarks at all.
* **Check that the flag took effect.** Silent fallbacks happened several
  times. A FP8 category applied to 0 modules. A `QKVParallelLinear` without a
  prefix quantised 36 of 48 layers. An OOM fallback ran the old path and
  produced a "passing" table. Deleting `lm_head.weight` without re-registering
  it fell back to a dense matmul. Log the module counts and inspect the trace.
* **Unique labels.** The benchmark refuses to overwrite an existing result
  file, so re-using a label silently reported an older run. Every arm now gets
  a unique label.
* **Orphan servers.** A validation script that timed out while waiting for a
  server exited without killing it. The orphan kept holding VRAM outside the
  GPU lock, and two quality-audit arms died with KV-pool allocation errors.
  Timeouts now kill the process group, and the next server does not start until
  `nvidia-smi` shows the memory released. Starting a server right after killing
  one also varied the KV pool from 5k to 180k tokens, so wait a few seconds.
* **Shared checkouts.** Switching branches in a checkout that another job was
  using changed that job's code partway through. Use one worktree per task.
  Do not edit the serving tree while a measurement is running.
* **VRAM headroom.** The desktop needs several GB. Memory fractions were tuned
  to leave at least 4 GB free in steady state (W4 0.935, W16 0.93, wa 0.925),
  after a compositor crash at 3.4 GB free.
* **Thinking mode.** The model reasons by default. All t/s figures include
  reasoning tokens. The needle test and the quality audit (GSM8K, MMLU,
  HumanEval, JCommonsenseQA) run with thinking off, because a thinking response
  can hit the output limit and come back empty.

## 9. The 2026-10 protocol

The 2026-10 round (`optimizations.md` §H, `rejected.md` §7) changed three things. Result files:
`../results/runs-1002/`.

**Both sampling modes.** Until September every A/B was greedy, but the server is used through LM Studio, which
samples with temperature 0.8, top_p 0.95, top_k 40 and min_p 0.05. `fnbench --sampling lmstudio` sends exactly
those parameters, and every arm runs the prompts in both modes. The modes can disagree: SV1/SV2 only act on
sampling steps, rejection sampling only changes sampling acceptance, and the extra eager work of a sampling step
was 250-300 µs on the old build (`lab-notes/FC_fc-map_2026-10-01.md` §5b). A change is reported per mode, never
as one blended number.

**Eight server starts, ABBA then BAAB.** One server start moves ms/step by 1-3 % on its own (FG1's two B arms, with
the same code, differed by 6.7 %), and in wa it also changes the generated texts, so a 4-start ABBA cannot resolve
a 1 % effect however many requests each arm has. From the stack ABBA on, A/Bs use 8 starts in the order
A1 B1 B2 A2 B3 A3 A4 B4, with 4 or 8 BN1 held-out prompts per workload (§6) in both modes.

**ANCOVA instead of paired t/s** (`bench/bench/stats/ancova_ab.py`). Per mode, three least-squares fits on log
values, each with one intercept per prompt, the arm position (linear drift) and a B indicator:

* `ms/token`: the user-facing cost (1 / t/s), including any acceptance change;
* `tok/step`: the acceptance;
* `ms/step|a`: ms per verify step with per-workload slopes on log tok/step, i.e. the step cost at equal
  acceptance. In wa a run that accepted more also picked wider steps, which are slower for reasons unrelated to
  the change; the slope removes that.

Each fit gives the B effect with two CIs:

* **request level**: the usual regression CI. It treats every request as independent and ignores that requests in
  one server start share that start's offset, so it is too narrow. (Sanity check: SV1 cannot change greedy steps,
  yet its greedy rows gave −1.22 % [−2.02, −0.40] at request level.)
* **arm level**: the arm means regressed on drift + B, one server start as the unit (5 degrees of freedom with 8
  starts). This is the honest interval. It is often 2-3× wider.

The docs quote both where they disagree. A result whose arm-level CI crosses 0 is called unresolved, not a
win or a loss. `t/s = 1 / (1 + ms/token) − 1` converts the first fit to a throughput change.

**Display time-slicing.** The desktop shares the GPU. On 2026-10-01 morning the display contexts paused the
kernels for 0.3-1.4 ms at a time, 20-28 % of every step span (2-8 % on a quiet night in September). This moves
unprofiled ms/step between arms minutes apart without changing tok/step. Two counters were used:

* **clipped R.eager**: the eager (non-graph) sample/accept/commit block of a step from in-server CUPTI traces,
  each kernel clipped to its median + 20 µs (`bench/prof/fc_eager_clip.py`). This is the number behind "the sampling
  eager tail fell from 376-391 to 119-139 µs/step" in the stack ABBA.
* **pause share**: each kernel's excess over median + 20 µs plus idle gaps over 100 µs
  (`bench/prof/fc_pause_share.py`). Without the pauses, the 10-01 steps matched 09-07 within about 3 %.

**Which number to trust for what.**

| question | number | why |
|---|---|---|
| did a kernel get faster | CUPTI median per kernel, in-server trace | the most stable number (§4); medians ignore display pauses |
| did the step get cheaper | `ms/step|a`, arm level | removes acceptance and per-start offsets |
| does the user see it | `ms/token` / client t/s, arm level | includes acceptance; noisiest |
| did acceptance change | `tok/step` per workload and mode | sampling and greedy can move in opposite directions |

**Greedy exactness is not testable on this stack.** Even with MoE autotune off, two W4 servers with the same
code produced 0/8 identical greedy texts, and two repeats inside one server 0/4 (DT1 note, `results/runs-1002/
stack8/exact/`, hashes only). Bit-exact changes are therefore checked at kernel level (`torch.equal` against the
reference on real shapes), as in §2, and rejection-sampling changes by statistics on the GPU: accept@0 against
Σ min(p, q), and chi-square / Fisher tests on the output distribution per position.

**Pre-registered rules.** Each A/B's adoption rule (which CI must exclude what, per mode and workload) was
written before the run. Where a candidate failed its rule and shipped anyway (the RS package, code-edit
unresolved), the docs say so.
