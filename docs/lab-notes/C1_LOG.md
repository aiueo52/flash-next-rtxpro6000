# C1 — confidence-driven speculative step choice (Qwen3.8-Flash-Next)

Worktree `$HOME/tools/sglang-c1`, branch `opus/spec-confidence` off
`codex/perf-v1` @ ef31f26346.  Never touches the main worktree, serve-fast.sh
or the venv (PYTHONPATH overlay only).

## 0. Starting point (read 2026-09-06)

Shipped controller (`adaptive_spec_params.AdaptiveStepSlot`) decides 3-vs-15
from ONE statistic: an EMA (alpha .07, 20-batch interval, 40-batch grace) of
`accept_lens - 1` over all requests in the batch.  Measured result, one
window 2026-09-05 21:31-21:40:

    workload     W4 acc/tps    W16 acc/tps    adaptive acc/tps
    code-edit    3.88 / 318    11.55 / 528     8.98 / 463   (-12%)
    prose-en     2.57 / 220     3.10 / 156     2.54 / 229   (+4%)
    prose-ja     2.29 / 208     3.31 / 170     2.25 / 205   (-1%)
    agent-loop   3.19 / 282     4.87 / 235     4.05 / 306   (+9%)

code-edit's loss is the bimodal accept distribution: full 15-chains
interleaved with 0/1, so the 20-batch EMA dips under the 5.0 step-down line.

## 1. Architectural constraints found by reading the code

These bound which of the three proposed options are implementable at all.

a. **The step-count decision happens BEFORE the draft, not during it.**
   `EAGLEWorkerV2.forward_batch_generation` (eagle_worker_v2.py:1342) calls
   `activate_step_by_batch(bs)` first; that swaps the whole `SpecRuntimeState`
   (draft graph runner + target verify graph + draft-extend runner) atomically.
   The 15-step draft loop then runs *inside one captured CUDA graph*
   (`self.cuda_graph_runner.execute`, eagle_worker_v2.py:772).  There is no
   host control flow between draft step 3 and step 4, so option 1 as literally
   written ("after the first 3 draft tokens, decide 3-vs-15 for this step")
   is not reachable without breaking the draft graph into two captures.

   What IS available at decision time: the top-1 probability of chain
   position 0 — the token `_draft_extend_for_decode` selected at the END of
   the previous iteration and stashed on `next_draft_input.topk_p`.  That is
   genuinely "this request's current draft confidence", one position deep,
   and it is on the host-side critical path already.

b. **That probability is not currently computed.**  On the topk=1 fast path
   both producers return a constant:
   - `draft_topk1_postprocess` (kernels/ops/speculative/topk1.py:74)
     `tl.store(topk_p + row, 1.0)` — "chain probabilities are unused
     downstream";
   - `_draft_extend_for_decode` (eagle_worker_v2.py:1152) uses
     `torch.argmax` + `torch.ones_like`.
   So step 1 of this task is to actually produce the number.

c. **Verify cost cannot shrink by shortening the chain inside the T=16
   graph.**  See section 4.

## 2. What was built

`opus/spec-confidence` (worktree `$HOME/tools/sglang-c1`, PYTHONPATH
overlay only — no venv, no serve-fast.sh, no main-worktree change).

**a. The probability itself** (`kernels/ops/speculative/topk1.py`).  The split
argmax already streams every logit of the row, so each split now optionally
stores `sum(exp(v - m_split))` next to its max, and the finalize kernel folds
them into `p_top1 = 1 / sum_j s_j exp(m_j - M)`.  Guarded by a `tl.constexpr`,
so passing `chain_probs=None` specialises the kernel back to the original
code.  `topk_p` is deliberately left at its constant 1.0 so every existing
consumer is bit-identical; the probability goes to a separate buffer.
`_draft_extend_for_decode` computes the position-0 probability eagerly
(`logsumexp` + `amax` over the hot vocabulary, two reductions).

**b. Getting it to the host without breaking overlap**
(`speculative/adaptive_confidence.py::ConfidenceChannel`).  First cut
synchronised on the staging event; that is wrong — the scheduler runs the CPU
ahead of the GPU on purpose (which is why the existing controller is fed from
`batch_result_processor` after `accept_lens` is already on the host, not from
the worker hot path).  The reader now walks back to the newest slot whose
event has already completed and caches it, and never waits.  A value one or
two iterations old is well inside the lag the controller already tolerates.

**c. The policy** (`ConfidenceStepSlot`).  Two changes against the shipped EMA
slot, both about asking a better question rather than smoothing the old answer
harder.

  *Rate instead of mean.*  Modelling assumption (not measured here):
  per-position acceptance is constant after position 2, so a length-S chain behaves like S Bernoulli(r) trials in series and
  `E[accepted | S] = r(1-r^S)/(1-r)`.  A single `r` therefore explains the mean
  at whichever S is live AND predicts the mean at the S that is not.  The EMA
  has no such transfer — it tracks a quantity whose scale changes with every
  switch, which is why the shipped config needs the asymmetric "re-seed on
  step-down only" hack.  Checked against the 2026-09-05 fixed-profile numbers
  with one multiplicative correction (`position_bias = 1.15`, absorbing the
  fact that positions 0-1 do accept better than the tail):

      workload    measured E@15 -> predicted E@3     measured E@3 -> predicted E@15
      code-edit        9.87          2.97                 2.86         8.75
      agent-loop       3.82          2.10                 2.29         4.66
      prose-en         1.83          1.41                 1.51         2.05
      prose-ja         1.55          1.26                 1.31         1.65
      (measured E@3:   2.86 / 2.29 / 1.51 / 1.31; measured E@15: 9.87/3.82/1.83/1.55)

  *Mixture instead of mean.*  This is what addresses the code-edit gap.  Its
  accept distribution at S=15 is bimodal, so the mean is a summary of a
  distribution with no mass near it and a threshold test on that mean asks the
  wrong question.  `r` is tracked per position-0 confidence bucket, and the
  decision integrates over the recent bucket occupancy `w_b`:

      argmax_S  sum_b w_b (1 + E[accepted | r_b, S]) / step_time(S)

  A first cut instead let the *latest* confidence pick the step count.  That
  was wrong for a structural reason worth recording: a switch leaves the draft
  state cold, so a decision is committed for a grace window, and steering a
  20-batch commitment with a one-step signal only thrashes (27 switches, -10%
  on prose-en offline).  What the per-step confidence buys is not a per-step
  decision — it is the ability to decompose the window into buckets that are
  internally homogeneous and to weight them by how often they occur.

`SGLANG_ADAPTIVE_POLICY=confidence` selects it; default `ema` is unchanged.
`SGLANG_ADAPTIVE_TRACE=<path>` dumps `(steps, bs, per-position chain probs,
accepted)` per verified step for the offline simulator; it is the only setting
that turns on the in-graph kernel writes.

## 3. Option 2 (draft early exit) — measured against the code, and dropped

The brief's question was whether padded rows in the captured T=16 verify still
route to experts (so early exit saves only the ~0.23 ms per skipped draft
forward), or whether they can be masked out of routing (so D shrinks toward
the T=4 cost).  Answer: they route, and there is no hook to stop them.
Verified in-tree, all paths shared with the running server:

* Every one of the `padded_bs * 16` verify rows is a genuine token id
  (`speculative/eagle_utils.py:515` sets `batch.input_ids =
  verify_input.draft_token`).  There is no valid-length concept in the target
  forward.
* SGLang's only row-validity primitive is `num_token_non_padded`, and it is
  dead here three times over: it is `None` unless expert parallelism is on
  (`model_executor/forward_batch_info.py:1730` — `return
  get_parallel().moe_ep_size > 1`; this server is `--tp 1`, no EP); the
  non-DeepEP MoE block never passes it to the router anyway
  (`models/qwen2_moe.py:773`); and it is a *prefix* mask
  (`layers/moe/topk.py:1467`), so it cannot express "request 0's rows 8..15
  are dead but request 1's are live", which is the shape a per-request early
  exit produces.
* On the NVFP4 target the router top-k is not even computed in Python — the
  backend resolves to `FLASHINFER_TRTLLM` and `TopK.forward_cuda` returns
  `BypassedTopKOutput`; selection happens inside
  `trtllm_fp4_block_scale_moe`, whose kwargs
  (`layers/moe/moe_runner/flashinfer_trtllm.py:1117-1155`) contain no
  row-validity argument at all (`grep -c num_token_non_padded` on that file:
  0).  Masking would mean giving up the fused routing for
  `TopKOutputFormat.STANDARD`, or a new kernel argument.
* Ragged verify exists and is fully built (`speculative/ragged_verify.py`,
  `SGLANG_RAGGED_VERIFY_MODE`, `pad_verify_lens_to_bucket`) but is gated to
  DSpark (`speculative/spec_info.py:139` — `return self.is_dspark()`), and its
  token buckets are `{bs * captured_req_width}`
  (`decode_cuda_graph_runner.py:517`), i.e. multiples of 16 across the whole
  batch.  The smallest tier is 16, so at bs=1 — this server's operating point
  — shortening 16 -> 8 -> 4 rounds straight back up to 16 and buys nothing.
  It could only pay by packing several shortened requests into a lower
  total-token tier.

Also worth recording, though not on this critical path: the CUDA-graph
bs-padding rows are deliberately not zeroed (`decode_cuda_graph_runner.py:1343`,
"Padded tokens aren't read"), which is true of the logits gather but false of
the router — at bs=3 replaying a bs=4 graph, rows 48..63 carry the previous
replay's ids and each activate 10 experts.  So D is inflated by bs padding as
well.  Not exploited here; flagged for whoever owns MoE cost (P1).

Consequence: early exit reduces to "skip draft forwards", and the draft loop is
one captured CUDA graph, so the only way to skip them is to replay a *shorter
captured graph* — which is exactly what choosing the steps=3 runtime state
already does.  Option 2 is therefore not a separate mechanism from option 1 on
this stack; it collapses into it.  C1 pursues option 1 (+ option 3 as the
no-confidence baseline) and does not implement early exit.

## 4. Trace collection, and two bugs in my own instrumentation

`SGLANG_ADAPTIVE_TRACE` on a fixed-W16 server and again on a fixed-W4 server,
one fnbench workload at a time so the trace can be split
(`adaptive/c1/trace_run.sh`).  Two things had to be fixed before the data was
usable, both worth recording because both produce *plausible* numbers:

1. **`ChainTracer` was block-buffered** (64 KB, ~300 rows at steps=15) while the
   per-workload split was driven by `wc -l` between fnbench invocations.  Every
   boundary was shifted and each segment was a blend of two workloads.  Caught
   only by checking each segment's mean acceptance against fnbench's own `acc`
   for the same run: 8.94 / 4.01 / 1.81 / 2.68 traced against 7.30 / 2.16 /
   1.45 / 4.19 measured.  Now line-buffered, and every row carries a wall
   clock.  **Anyone re-collecting these traces should repeat that check.**
2. A server of mine was killed mid-load at 13:07 by another agent's teardown
   (it force-killed every GPU compute process, which catches a process that is
   still loading weights; that script is not published).  `trace_run.sh`/`ab_run.sh` now wait for the GPU to be
   genuinely empty and then settle before launching.

After the fix each segment's traced mean accepted matches fnbench: W16
9.63/2.10/1.41/3.81 against acc-1 of 8.68/2.05/1.43/3.55, W4
2.76/1.47/1.35/2.10 against 2.65/1.40/1.41/2.02 (the traced value is slightly
higher because it also covers fnbench's warm-up pass).

## 5. Does the draft's confidence predict acceptance?  Yes.

W16 traces, position-0 top-1 probability against accepted drafts:

    workload    corr(p0, a)   mean accepted by confidence bucket
                              <0.8   .8-.95  .95-.99  .99-.999  >0.999
    code-edit      0.51       2.93    4.68     8.83     9.95     11.70
    prose-en       0.53       1.29    2.15     3.58     3.20      3.45
    prose-ja       0.37       0.82    1.57     2.06     2.20      3.09
    agent-loop     0.41       2.10    3.73     4.56     4.74      5.18

Mostly increasing (prose-en dips from 3.58 to 3.20 in the .99-.999 bucket), and a 4x
spread on code-edit.  The product of the
first three chain positions correlates better still (0.55-0.72), which is the
statistic the brief proposed -- but it is only knowable after the draft has
run, and the draft is one captured CUDA graph, so it cannot be acted on for
the step it describes.

The distribution is strongly skewed toward 1.0 (code-edit at W4: 10th
percentile 0.966, median 1.000), so the bucket edges are packed against 1.0.

## 6. The estimator: exact one way, censored the other

This is the substantive change against the shipped controller, and the
asymmetry is the point.

**Downward is exact and needs no model.**  A topk=1 draft is a greedy chain,
the target accepts a prefix of it, and the first S' tokens of a length-S chain
are the tokens a length-S' chain would have drafted.  So a step observed at S
says exactly what S' < S would have accepted: `min(accepted, S')`.  Checked by
predicting the W4 runs from the W16 traces:

    workload    mean min(a_15, 3)   measured mean accepted at W4
    code-edit        2.66                     2.76   (+4%)
    prose-en         1.49                     1.47   (-1%)
    prose-ja         1.14                     1.35   (+18%)
    agent-loop       2.09                     2.10   (+0%)

**Upward is censored, and this is where the confidence pays.**  At S=3 an
accepted count of 3 could mean "3" or "would have been 14"; the mean saturates
and says nothing about how much chain is being left on the table.  What
survives censoring is the boundary hazard `r = P(a>=S)/P(a>=S-1)`, and under
the modelling assumption that per-position acceptance is constant after
position 2, that one rate extends the chain:

    E[accepted | T] = E[min(a,S)] + P(a>=S) * r (1 - r^(T-S)) / (1 - r)

Predicting the W16 runs from the W4 traces: +14%, -8%, +17%, +9% (pooled), and
+20%, -2%, +20%, +10% bucket-by-bucket.  Systematically optimistic, so
`tail_bias = 0.9` discounts it and `switch_margin` covers the rest.  The
upward estimate is always recomputed from the *live* state rather than read
from a remembered EMA of the other state -- a remembered value goes stale
exactly when it matters, which is when the workload has changed.

With this estimator the policy picks correctly at both fixed points for all
four workloads, in both directions, which the shipped EMA does not:

    from S=3  (up decision)      predicted tok/s: S=3 vs S=15   -> choice
      code-edit                        314        592             15  correct
      prose-en                         209        145              3  correct
      prose-ja                         199        131              3  correct
      agent-loop                       262        253              3  correct
    from S=15 (down decision, exact)
      code-edit                        309        524             15  correct
      prose-en                         210        153              3  correct
      prose-ja                         181        119              3  correct
      agent-loop                       261        238              3  correct

## 7. Offline simulation

`adaptive/c1/sim.py`.  The shipped controller's simulator drew accepted counts
i.i.d. from a shaped pool; that destroys the temporal clustering of hard
passages, which is precisely the thing the EMA trips over, and it is why that
simulator did not predict the measured -12% on code-edit either.  The C1
simulator replays the recorded W16 trace IN ORDER and uses `min(a_t, S)` as
the counterfactual -- exact per step, as section 6 shows -- so a policy is
scored against the real sequence of easy and hard stretches.  (The i.i.d.
"empirical" mode is kept as a second opinion and reported alongside.)

3000-batch replay, tokens/s per workload, against the better fixed profile:

    policy              code-edit    prose-en    prose-ja   agent-loop   worst  mean
    fixed W4               309/526     211        181         261
    fixed W16              526         154        119         237
    EMA (shipped cfg)      521/sw18    211/sw1    180/sw1     250/sw23   0.957 0.986
    confidence (tuned)     526/sw0     207/sw1    180/sw1     252/sw3    0.965 0.985
    ORACLE (per step)      594         235        191         321

The offline gap is small -- both controllers land within a few percent of the
better fixed profile -- but the *shape* differs: 5 switches against 43.  The
simulator's switch penalty (a few cold batches) is milder than the server's
measured cost, so a controller that reaches the same throughput with an eighth
of the switching should be strictly better on hardware, and the shipped
controller's 18 switches on code-edit are exactly the mechanism behind its
measured -12%.  That is the claim the server A/B has to settle.

The ORACLE row is the per-step upper bound: agent-loop leaves 23% on the table
for anyone who can make switching cheap.  Nothing here can reach it, because a
switch costs several batches of cold draft state.

## 8. Server A/B

`adaptive/c1/ab_run.sh` — one `serve-fast.sh wa` per arm, PYTHONPATH overlay
only, fnbench 4 workloads x 2 repeats greedy, needle test, both arms inside the
same hour.  The EMA arm runs the *same C1 build* with
`SGLANG_ADAPTIVE_POLICY=ema`, which is byte-identical to the shipped path: no
confidence channel is constructed, the draft kernel is called with
`chain_probs=None` and specialises back, and `AdaptiveStepSlot` is used.  So
the arms differ only in the decision rule.

**Round 1 — 2026-09-06 16:07 / 16:10** (config as tuned before the round):

    workload     EMA (shipped)      confidence       delta
    code-edit    acc  8.87 / 508    10.84 / 544      +7.1%
    prose-en          2.58 / 246     2.59 / 246       0.0%
    prose-ja          2.37 / 228     2.29 / 221      -3.1%
    agent-loop        3.43 / 292     3.75 / 267      -8.6%
    switches            13              3
    needle             PASS           PASS

The agent-loop loss had one identifiable cause in the decision log:

    16:11:35  steps 3 -> 15 (E@3=2.16 -> 267 tok/s, E@15=5.53 -> 323 tok/s)
    16:11:40  steps 15 -> 3 (E@15=3.21 -> 208 tok/s, E@3=1.91 -> 246 tok/s)

E@15 was predicted at 5.53 against a true ~3.5.  That is the upward
estimator's known optimism, and it is *not* symmetric noise: going down uses
`min(accepted, S)`, which is exact, while going up uses the hazard
extrapolation.  So the hysteresis should not be symmetric either.  `up_margin`
(default 0.30) makes a candidate whose estimate is extrapolated clear a wider
margin.  The two cases separate cleanly, which is why one constant suffices:
at S=3, code-edit predicts +88% for the long chain while agent-loop and both
prose workloads predict within +/-5%.

**Round 2 — 2026-09-06 16:47 / 16:51**, with `up_margin`:

    workload     EMA (shipped)      confidence       delta
    code-edit    acc  8.92 / 480    11.80 / 586     +22.1%
    prose-en          2.56 / 240     2.50 / 236      -1.7%
    prose-ja          2.25 / 216     2.30 / 219      +1.4%
    agent-loop        3.24 / 287     3.26 / 290      +1.0%
    switches            13              4
    needle             PASS           PASS

code-edit's acceptance under the controller (11.80) is at the level the *fixed*
W16 profile reaches (11.55 on 2026-09-05), i.e. the shipped controller's
documented -12% on code-edit is closed, and nothing else regresses.  The
controller now spends essentially the whole of each workload in the right
state: 4 switches over the benchmark against the EMA's 13, and the EMA's log
shows it entering and leaving steps=15 six times inside 50 seconds.

**Same-window fixed-profile references** (17:22 / 17:26, one hour block with
round 2), so the controller can be read against what a user would get by
hand-picking a profile:

    workload     fixed W4     fixed W16    EMA (shipped)   confidence
    code-edit    3.88 / 330   10.48 / 493   8.92 / 480    11.80 / 586
    prose-en     2.55 / 227    3.10 / 156   2.56 / 240     2.50 / 236
    prose-ja     2.37 / 214    2.56 / 133   2.25 / 216     2.30 / 219
    agent-loop   3.21 / 284    5.70 / 281   3.24 / 287     3.26 / 290

    vs the BETTER fixed profile:  +18.9%  +4.0%  +2.3%  +2.1%
    vs the shipped EMA config:    +22.1%  -1.7%  +1.4%  +1.0%

Free VRAM while serving at `WA_MEM_FRACTION=0.925`: 4.59 GB (confidence) vs
4.56 GB (EMA) — the confidence path costs nothing in VRAM, because the staging
buffers are pinned *host* memory and the trace-only device buffer is not
allocated in production.  Both above the 4 GB floor.

## 9. Recommended configuration

`~/tools/flash-next-bench/adaptive/w16_conf.json`, documented in
`adaptive/README-confidence.txt`:

    cd ~/tools/sglang-rtxpro6000 && \
      PYTHONPATH=~/tools/sglang-c1/python SGLANG_ADAPTIVE_POLICY=confidence \
      ./serve-fast.sh wa --speculative-adaptive-config \
        ~/tools/flash-next-bench/adaptive/w16_conf.json

    switch_margin 0.12   up_margin 0.30   tail_bias 0.75
    rate_alpha 0.10      weight_alpha 0.05
    update_interval 10   switch_grace_batches 40   grace_backoff 2.0
    buckets [0.8, 0.95, 0.99, 0.999]   min_bucket_samples 20

Tuned by joint selection over the recorded traces under two criteria at once —
3000-batch throughput replay, and convergence to the better fixed profile from
BOTH starting states for all four workloads (8/8).  Regression guard:
`adaptive/c1/test_policy.py` (CPU only, seconds).

## 10. Caveats

* **code-edit acceptance is noisy run to run.**  Fixed W16 measured acc 8.30
  (13:47), 9.68 (15:00), 10.48 (17:22) and 11.55 (2026-09-05) on the same
  greedy workload.  The +18.9% against the same-hour fixed W16 should be read
  with that spread in mind; the +22.1% against the EMA arm four minutes later
  is the more robust number, and it reproduced in both rounds (+7.1% before
  `up_margin`, +22.1% after).
* **The upward estimate is a model, the downward one is not.**  The hazard
  extrapolation measured 8-17% optimistic and is what caused round 1's
  agent-loop regression.  `up_margin` + `tail_bias` contain it, but a workload
  whose per-position acceptance is *not* constant past position 2 would
  break the assumption in a way no margin fixes.
* **Everything here is bs=1.**  fnbench is single-stream, so `w_b` is a
  batch-average confidence and the traces contain no bs>1 data.  Config slot
  "2" pins bs>=2 to steps=15, same as the shipped config, so the untested path
  is not reachable.
* **The offline simulator's counterfactual is exact per step but not per
  trajectory.**  `min(a, S)` is exactly what a shorter chain would have
  accepted at that step, but a run that accepts fewer tokens per step arrives
  at different text later on.  Over thousands of steps the difficulty
  distribution is the same; over a short window it is not.
* **`SGLANG_ADAPTIVE_TRACE` is debug only.**  It enables writes inside the
  captured draft graph and a per-step D2H copy.  It was not benchmarked for
  overhead and must not be set in production.
* **Untested combinations**: topk>1 (the prefix identity that makes the
  downward estimate exact relies on a chain, not a tree), steps=0 candidates
  (the estimator has no branch for a disabled draft), and multi-rank (the
  tracer writes per-rank files but the controller is per-rank and
  unsynchronised, same as the shipped one).
* **Option 2 (draft early exit) was not implemented** — see section 3; on this
  stack it collapses into option 1.
* **The per-step oracle is still far above both controllers** (agent-loop 321
  vs 261 for the better fixed profile).  That headroom is only reachable by
  someone who makes a state switch cheap; today it costs several batches of
  cold draft state.
