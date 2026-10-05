# RS2 (2026-10-01): sparse rejection sampling

Status (2026-10-02 06:40): the package ABBA (RS2 + RS2d knobs + G1 + RS3 BV against target-only) is in the RS2d
spec's 06:40 status: LM Studio pooled t/s +6.5% [+4.6..+8.5], code-edit +1.6% [-2.5..+5.9], so not adopted by this
section-6 rule, read as "code-edit unresolved". Shipping is the user's call.
Status (23:40): **8-start ABBA done (`runs/rs2/abba-rs2.log`): RS2 alone is not adopted. LM Studio is faster
overall, but code-edit is slower and greedy misses its +1% bound.** It stays as the base for RS2d + G1 + RS3 BV,
which the next ABBA tests against target-only (`bench/rs2d/abba_rs2d.sh`).
- Section-6 rule, t/s = 1/(1 + ms/token) - 1, request-level CI (arm-level in brackets where it matters):
  - pooled lmstudio t/s +3.7% [+1.6..+5.8] (arm level [-0.3..+7.8]); tok/step +4.4% [+1.8..+7.2]; ms/step|a -0.3%
    [-1.3..+0.7]: PASS (request level).
  - code-edit t/s -5.0% [-9.1..-0.8]: FAIL (lower bound must be > -2%). tok/step -3.8% [-9.4..+2.2], ms/step|a
    +2.8% [+0.3..+5.4]: RS's acceptance (sum of min(p, q)) loses to target-only's p(draft argmax) on code.
  - agent-loop t/s +4.8% [+1.2..+8.5]: PASS. prose-en +8.0% [+5.6..+10.4], prose-ja +7.5% [+2.4..+13.0].
  - greedy ms/step|a +1.9% [+0.8..+3.2]: FAIL. The A1/B1 pair carries it (arm means -6.1% and +6.4%; B1 was
    display stalls, 20:30 status). Without that pair (post hoc) -1.7% [-2.4..-0.9]. G1 removes the proposal's
    +1.07% (RS2d spec 5).
  - errors: none (0 error lines in all 8 server logs; RS2 on in the B arms only).
- Display load (`bench/rs2/pmon_phase.py`, from 20:25): graphics SM 6-13% mean per arm phase in B2-B4, no A/B
  pattern; A1 and B1 are not covered.
- Per workload: scratchpad `ancova_per_workload.sh` (split by workload, then `bench/stats/ancova_ab.py`).
Status (20:30): **B1's greedy t/s (11-18% below A1) fell to display interference, not to RS2's code.**
- The end-of-arm greedy traces (`runs/rs2/traces/{A1,B1}-g-code-edit`, 20 verifies each) run the same 1365 kernels
  per verify with the same median duration in every kernel family. B1's verify is 2.0 ms longer (11.2 vs 9.2 ms
  GPU busy) because of 109 stalls of 0.2-1.3 ms inside single kernels; A1 has 23, none above 0.6 ms. Stalls like
  these are other GPU contexts taking time slices. At 20:25 the display (Xorg, kwin_x11) used 9-21% SM.
- RS2's own greedy cost: an all-greedy batch still drafts through the RS2 proposal (9.0 + 12.4 us per draft step)
  instead of topk1 (1.8 + 1.9 us), 7 steps per verify in B1: about 0.12 ms per verify (1.1% of 11 ms). Plus 4
  more 4-byte D2D copies per draft (sampling parameters into the graph inputs) and an RNG draw per draft-extend.
  RS2b would cut the kernel part to a third.
- From 20:25 `runs/rs2/pmon-1001-eve.log` (`nvidia-smi pmon`, 5 s) records per-process GPU use;
  `bench/rs2/pmon_phase.py` turns it into each arm phase's graphics SM%, a covariate for the ABBA.
Status (20:00): **RS2's verify costs about what SV1's does; the proposal is the gap, and tuning cannot close it.
A probe of FlashInfer's radix top-k (as SV2 calls it) for a faster proposal (RS2b) runs in the gap after arm B1.
The ABBA ends about 23:15 (arms take 30 min, not the 18 min planned).**
- Verify, GPU us per verify (`runs/rs2/e-verify.log`, `bench/rs2/bench_e_verify.py`, CUDA graph):

  | bs | slots | dense RS (RS1) | RS2 | SV1 target-only |
  |---|---|---|---|---|
  | 1 | 4 | 330.1 | 82.6 | 76.3 |
  | 1 | 8 | 392.4 | 78.6 | 77.1 |
  | 1 | 16 | 482.5 | 90.0 | 83.2 |
  | 4 | 4 | 473.9 | 84.3 | 83.6 |

  RS2 is 1-8% (0.7-6.7 us) above SV1 and 4-5x below RS1's dense verify.
- Proposal sweep (`runs/rs2/sweep-proposal.log`, `bench/rs2/sweep_proposal.py`): split blocks 512-4096 and
  num_warps 2-16 for each kernel; all 192 configs give bit-identical outputs. Default (2048, 4, 4): fp32 22.2 us
  (partial 9.7 + finalize 12.5), bf16 19.4 us (6.5 + 12.8). Best: fp32 19.9 us (2048, 16, 16), bf16 18.5 us
  (2048, 4, 16). Tuning saves 1-2.7 us of the 16-19 us over greedy: the finalize's int64 `tl.topk` over 1536
  candidates costs 12.5 us by itself, and a smaller block only moves the cost into the partial kernel. Not
  adopted. The S1 traces show 9.8 (code-edit) and 15 (prose-en) proposal steps per verify, so the proposal adds
  about 0.16-0.35 ms per verify (1.5-3% of a 10-13 ms step).
- RS2b idea: FlashInfer `top_k(sorted=True, tie_break=1)` (smaller index first = RS2's key order) on the 49152
  row, then a finalize over the 64 sorted entries only. `bench/rs2/probe_fi_topk.py` checks the order against
  RS2's keys, a CUDA graph replay on new logits, and the time.
Status (19:20): **CUDA tests A-D PASS; E misses its 15 us proposal budget (19-23 us, a quarter of RS1's cost);
smoke S1 clean; the 8-start ABBA runs from 19:10.**
- 19:06 run (`runs/rs2/test-cuda.log`): A, B PASS; C PASS (48 cases/120 rows, 0 rounding-edge mismatch rows);
  D PASS (Fisher p=0.837; accept@0 0.73396 against sum(min(p, q)) 0.73481, SE 0.00156).
- E proposal, GPU us per call (CUDA graph of 14 calls, Vd 49152):

  | logits | n | RS2 triton | torch | RS1+scatter | greedy topk1 |
  |---|---|---|---|---|---|
  | fp32 | 1 / 2 / 4 | 22.97 / 27.25 / 23.00 | 93.5 / 94.6 / 99.8 | 86.4 / 88.0 / 97.2 | 3.75 / 3.60 / 3.59 |
  | bf16 | 1 / 2 / 4 | 19.32 / 19.62 / 19.30 | 92.9 / 99.4 / 104.1 | 86.3 / 100.3 / 98.1 | 3.09 / 3.51 / 3.49 |

  - So RS2 adds 16-19 us per draft step over greedy. wa drafts 3, 7 or 15 steps per verify (candidates [3, 7, 15]),
    so +0.05..0.29 ms per verify, 0.5-2.5% of a 10-12 ms step, before any acceptance gain.
  - The budget assert stopped test E before its verify timings; `bench/rs2/bench_e_verify.py` runs them alone, and
    `bench/rs2/sweep_proposal.py` sweeps the split block and num_warps (bit-identical outputs), between arms.
- S1 smoke (`runs/rs2/chain-rs2b.log`): 0 error lines, the RS2 log line, and in both sampling traces 20
  `_chain_sampling_sparse_kernel` + 20 `_sparse_target_probs_kernel` per 20 verifies; `_draft_partial_topk_kernel`
  and `_draft_finalize_kernel` 196 each (code-edit) and 300 each (prose-en, wa at 15 steps): 2 launches per draft
  step (F1 profile). No `mbtopk` kernel.
Status (18:55): **CUDA tests A and B now PASS; C stopped on the test's FlashInfer setup. Fixed; the chain re-runs
after XA1 arm B2 (about 19:06).**
- 18:47 run (`runs/rs2/test-cuda-1847-failC.log`): A PASS for both top-k paths (868 rows, 38 of them flipped at a
  top-p tie and were checked on the kernel's own support; max delta 5.96e-08); B PASS (M=40000/row, chi2 p 0.086
  and 0.324).
- C needs FlashInfer's sampling module (`sgl_kernel.top_k_renorm_prob`). The chain gave the test a fresh FlashInfer
  workspace, so FlashInfer tried a JIT build and found no nvcc. Now the test uses the server arms' FlashInfer
  (flashinfer-p3 with the prebuilt rq2u2h cache, `FLASHINFER_P2_NO_NINJA=1`), so it can never build.
Status (18:45): **CUDA test A failed on a rounding tie in the test, not in the kernel. Test fixed and passes on
CPU; the GPU re-run waits for the GPU (another job had it).**
- 18:00 `chain_rs2.sh`, CUDA test A (`runs/rs2/test-cuda-1800-fail.log`): in the bf16-tie case, rows 8/20/32
  (top_k 40, top_p 0.5) kept 21 tokens in the reference and 20 in the kernel. The chain stopped there; no server ran.
- Parent's GPU check (`rs2_tie_diag.py`, same rows, both top-k paths): in fp64 the exclusive sum at rank 20 is
  0.5000000075, 7.5e-9 above top_p, so dropping rank 20 is right; the reference (batched fp32 `torch.cumsum` on
  CUDA) keeps it. This is a rounding-level tie: either support is a valid fp32 top-p, and losslessness does not
  depend on it (the same q both samples X and verifies).
- Fix, test only (implementer, parent-reviewed): the support may differ only at ranks within 1e-6 of the top-p
  cut (or of the min-p threshold); q is then compared, by token id, with the reference renormalized on the
  kernel's own support. Test A prints how many rows flipped. Parent's CPU run 18:44: A-D PASS, flips 0, D p values
  identical to 16:49 (Fisher p=0.244059).
Status (16:55): **implemented and committed; CPU tests PASS; GPU tests and server checks queued.**
- Worktree `~/tools/sglang-rs2`, branch opus/rs2-sparse, commit cf1e515722 (on c868f2ee86 = stack + DT1).
- The implementer (an AI coding assistant) wrote it in one run (`bench/rs2/PROGRESS.md`). The parent's review found 3 issues; one rework
  fixed all 3:
  - F1: RS2 had widened DT1's chain tree build (`_draft_tail_chain`) to RS2. Reverted to DT1 only. Not needed:
    the topk=1 chain path returns the preallocated parent/score indices, which the normal tree build accepts (as
    for RS1).
  - F2: the relay read the new support fields directly from any draft input, so DFlash's draft input would raise
    AttributeError. Fixed with class defaults on `SpecInput`; a real `DFlashDraftInputV2` check was added.
  - F3 (minor): the idle input read the env var on every call; now read once at import.
- Parent re-runs: `check_integration_cpu.py` all PASS (relay, constructors, storage, dispatch guards, flag-off
  byte equality with c868f2ee86, real draft_forward); 10 changed files compile; `git diff --check` clean.
  `test_sparse_rs.py` A-D PASS on CPU (16:49; D: the same p values as the implementer's run, same seed).
- GPU: `bench/rs2/chain_rs2.sh` runs tests A-E on CUDA (stops if A-D fail; E is the 15 us speed budget and only
  warns), then smoke S1 and the 8-start ABBA (to vs rs). The w4 greedy-text gate was dropped: w4 text differs
  even between two runs of one config (DT1 spec, 16:55).
Status (15:45): **design**. Implementation and CPU tests next; the GPU parts wait for the current GPU queue
(stack ABBA, C1, w4 exactness chain; until about 17:20).

## 0. Summary

- **Why.** RS1 (rejection sampling) raises acceptance under LM Studio sampling. LM Studio sends T 0.8, top_p 0.95,
  top_k 40, min_p 0.05 (`fnbench/models.py` LMSTUDIO_SAMPLING).
  - With min_p honored, tok/step rose by +5% (code), +6% (agent) and +7% (prose-en); prose-ja about 0 (RS1 §4c).
  - Its per-step cost ate the gain. ms/step rose by +7.5 / +10.3% on code and +5.4 / +4.5% on agent, against
    +0.8 / +1.8% on prose-en. A cost that grows with the chain length fits this: code and agent run the longest
    wa chains.
- **Where RS1's cost sits** (code reading, opus/stack-dt):
  1. **Every draft step** (S − 1 of them per verify, inside the draft CUDA graph):
     - `sample_draft_proposal_truncated`: `torch.topk(k=64)` over the 49152-entry hot vocab, which is torch's
       multi-kernel radix select plus a sort, and about 25 small elementwise launches (temperature, masks,
       cumsum, renorm, `exponential_`, clamp, argmax, gathers);
     - the hot-id gather, `scatter_` into a zero-filled (bs, V) fp32 row, `positions.add_`;
     - the generic tree bookkeeping (`select_top_k_tokens`: score product, `fast_topk`, gathers, lists). RS turns
       off the topk1 chain fast path (`draft_tokens_topk1 = None` under RS), so it pays this too.
     - About 50 launches per draft step, against 2 on the greedy topk1 path (`draft_topk1_postprocess`).
  2. **Every verify:**
     - a zero-filled (bs, S, V) fp32 q buffer (15 MB at S = 15);
     - (bs, V) q rows copied through the FutureMap relay, the draft graph's input buffer and row 0;
     - the dense target chain (softmax over V, top-k renorm, top-p renorm, min-p), because `_sparse_verify_kp`
       returns 0 under RS;
     - the dense chain kernel's two passes over V for the final draw.
- **RS2 keeps RS1's distributions:** the same truncated q and the same target p. It stores q on its K = 64 support
  only (values plus target-vocab tokens). It drafts with a two-launch fused proposal and writes the chain in place
  like the topk1 path. The verify uses SV1/SV2's sparse target probs and a new sparse chain RS kernel.
- **Expected.** The draft side costs about what the greedy topk1 path costs; the verify costs about what SV1's
  sparse verify costs. If the per-step cost goes, LM Studio t/s should follow RS1's tok/step: code +5%,
  agent +6%, prose-en +7%, prose-ja about 0. Greedy is unchanged by construction (§2.4).

## 1. Switch and scope

- `SGLANG_OPT_SPEC_SPARSE_RS` (EnvBool, default False), next to SV1's flags in `srt/environ.py`. Follow the
  env-var-conventions skill.
- Acts only with `--speculative-use-rejection-sampling`. It requires `SGLANG_RS_DRAFT_TOPK > 0` (the truncated
  proposal) and topk == 1 (RS already asserts it). Raise at worker init if RS2 is on and either fails.
- Single-layer EAGLE/MTP path only: `eagle_worker_v2.py`, `eagle_draft_cuda_graph_runner.py`, `eagle_info.py`,
  `eagle_utils.py`, `eagle_worker_common.py`, `managers/overlap_utils.py`. The multi-layer EAGLE, DFlash, ngram and
  dspark paths are untouched. Raise at init if RS2 is on with one of them.
- **Off means RS1's code paths exactly:** no new launches and no changed tensors.

## 2. Data layout and flow

K = `SGLANG_RS_DRAFT_TOPK` (64). For chain position j = 0..S−1:

- `draft_support_probs[b, j, :K]`, fp32: the truncated, renormalised proposal q_j. It is sorted by descending
  draft logit and is exactly 0 past the kept set.
- `draft_support_tokens[b, j, :K]`, int64: the same entries' tokens in the **target** vocab (hot ids mapped).
- Row j is the q that X_j = candidates[:, j + 1] was drawn from.

The names follow the speculative-naming skill: plural tensors, `_tokens` and not `_token_ids`. They may be changed
if the skill says otherwise, as long as they are used consistently.

### 2.1 Fields

- `EagleDraftInput`:
  - new `draft_support_probs` (b, K) and `draft_support_tokens` (b, K), which hold row 0 of the next chain;
  - `draft_probs` stays None under RS2;
  - `create_idle_input`, `filter_batch` and `merge_batch` handle both fields, in the same places as `draft_probs`.
- `EagleVerifyInput`: new `draft_support_probs` (bs, S, K) and `draft_support_tokens` (bs, S, K).
- `RelayPayload` and the FutureMap (`managers/overlap_utils.py`) relay both, the same way as `draft_probs`, with
  (req_pool_size, K) buffers. No new `getattr`: the existing `getattr(draft_input, "draft_probs", None)` is
  grandfathered, and new code reads fields directly.
- `eagle_draft_cuda_graph_runner`: under RS2, (max_bs, K) fp32 and (max_bs, K) int64 buffers replace the
  (max_bs, V) `draft_probs` buffer.
  - The replay copy and the padding zero-fill cover both.
  - The captured temperatures, top_ks, top_ps and min_ps buffers stay as RS1 has them.

### 2.2 Draft forward (RS2)

- Allocate `draft_support_probs` (bs, S, K) and `draft_support_tokens` (bs, S, K). No V-wide buffer. Row 0 comes
  from the draft input.
- Draw all uniforms for the chain before the loop: `u = torch.rand((bs, S - 1))`, one launch per draft forward,
  inside the captured graph like RS1's `exponential_`.
- Use the topk1 chain fast path: allocate `draft_tokens_topk1` under RS2 too. Each draft step makes one call of the
  fused proposal (§3.1), which also:
  - writes X (target vocab) into `draft_tokens_topk1[:, i + 1]`;
  - writes `topk_index` (the next step's input ids, target vocab, as the topk1 path does);
  - advances `positions`;
  - fills row i + 1 of both support tensors.
- With that, no `select_top_k_tokens`, no per-step hot map, no separate `positions.add_` and no `scatter_`.
- Return `parent_list, top_scores_index, draft_tokens_topk1` from the topk1 preallocations, plus the support pair.
  `build_eagle_verify_input` passes the pair into `EagleVerifyInput`.

### 2.3 Draft-extend (prefill and decode)

Same fused proposal, eager, for row 0. It returns `topk_index` in the draft vocab, as `_rs_draft_proposal` does
today, so the existing hot map at the top of `draft_forward` stays valid. `_record_position0_confidence` is
unchanged. DT1's `_draft_tail_select` is already off under RS.

### 2.4 Greedy rows and batches

- A request with top_k == 1 gets q one-hot on the draft argmax. Ties go to the smallest id, the same as
  `draft_topk1_postprocess`, where `tl.argmax` and the split order both prefer the first index. So its drafts are
  the greedy drafts.
- All-greedy batches take `eagle_sample`'s greedy branch (target argmax). So under RS2 a greedy request must emit
  exactly what the same server without RS emits (tested in §5 F2).

## 3. Kernels (`python/sglang/kernels/ops/speculative/sparse_rs.py`, new)

### 3.1 Fused draft proposal

`rs_draft_proposal_sparse(...)`:

- **Inputs:**
  - draft logits (n, Vd), bf16 or fp32, with a row stride;
  - temperatures, top_ks (int32; ≤ 0 means no top-k, which is the graph's capture-time filler), top_ps;
  - min_ps, or None;
  - uniforms u (n,);
  - `hot_token_id` (Vd,) int64, or None;
  - K (constexpr);
  - optional in-place outputs: positions (n,) to advance, and the chain buffer plus its column.
- **Outputs:** q (n, K) fp32, tokens (n, K) int64 in the target vocab, `topk_p` (n, 1) = q(X), `topk_index`
  (n, 1) = X (target vocab when the chain buffer is given, draft vocab otherwise; see §2.2 and §2.3).

**Semantics.** These must equal `sample_draft_proposal_truncated` up to fp32 rounding; only the draw mechanism
differs.

1. Take the top K of the row's logits (as fp32), sorted descending. Ties go to the smaller draft id; torch.topk's
   order among ties is unspecified, so tests compare sets and values, not tie order.
2. w_r = exp((l_r − l_0) / T).
3. Top-k: keep r < top_k, or everything when top_k ≤ 0.
4. Normalise.
5. Top-p: keep r while the exclusive cumsum < top_p; rank 0 always survives.
6. Min-p, if given: keep p_r ≥ p_0 · min_p.
7. Renormalise.
8. Draw X by inverse CDF in rank order: the first r whose inclusive cumsum > u · Σq. If rounding leaves none, take
   the last r with q_r > 0. RS1 drew X with an exponential race instead. Both draw X ~ q exactly; only the RNG
   stream differs.
9. Write q, which is 0 past the kept set, plus `tokens = hot_token_id[ids]`, `topk_p = q[X]`, X, the chain column
   and the positions.

**Robustness.** Padded graph rows may carry filler parameters (top_k 0, any T, including 0). The kernel must not
fault or write out of range on them; their outputs are ignored. NaN logits are treated as −inf, as in
`_draft_topk1_partial_argmax_kernel`.

**Launch structure.** The target is 2 launches per call, like `draft_topk1_postprocess`:

- Partial kernel, grid (n, num_splits): each split computes its local top K (value, id). For example, pack
  (order-preserving uint32 of the fp32 value) << 32 | (complemented id) into an int64 key and use `tl.topk` or
  `tl.sort`. Triton 3.7.1 has `tl.topk`, `tl.sort` and `tl.rand`.
- Finalize kernel, grid (n,): merge the num_splits × K candidates (padded to a power of 2), take the top K, then
  steps 2-9.
- Alternatives are allowed if the §5 E microbenchmark shows them faster at bs 1 and 4 inside a CUDA graph:
  - `torch.topk` plus one finalize kernel;
  - a threshold search per split.
  - FlashInfer's top_k only if it is CUDA-graph safe (it allocates a workspace) and only from the private
    rq2u2h cache, never the shared one.
- **Budget:** ≤ 15 µs GPU time per call at n = 1, Vd = 49152 inside a graph. Report it next to the greedy topk1
  path's time.

### 3.2 Sparse chain verify

`chain_speculative_sampling_sparse(...)`, grid (bs,).

**Inputs:**
- `predicts`, `accept_index`, `accept_token_num` (mutable);
- `candidates` (bs, S+1), `retrive_index` (bs, S+1);
- coins (bs, S+1) and final coins (bs,), from `_verify_coins` as today;
- target `P`/`PI` (bs, S+1, KP) from `sparse_target_probs` (SV1, with SV2's top-k flag);
- draft `Q`/`QI` (bs, S, K);
- the vocab size.

**Walk.** This is the classic kernel's walk, with the same comparison and the same coin layout. For step = 1..S:
- X = candidates[step], row = step − 1;
- p = P[row] at PI == X, or 0;
- q = Q[row] at QI == X, or 0; NaN q counts as 0;
- coin = coins[step − 1];
- accept iff coin · q < p;
- on accept, write predicts / accept_index exactly as `speculative_sampling_classic_kernel` does;
- otherwise stop.

**Final draw** at row r:
- **All accepted** (r == S): w = P[S].
- **Otherwise:** w(t) = max(0, P[r](t) − q_r(t)) over the target support, where q_r(t) is the Q[r] entry with
  QI[r] == t, or 0. This loses nothing: the dense residual is 0 wherever p = 0.
- Then u = coin_final · Σw. Take the cumsum in **token-id order** (as SV1 does, so equal coins give the dense
  kernel's token except at rounding edges) and pick the first t with cum > u and w > 0.
- **Fallbacks:**
  - Σw == 0 means p == q on the support, which is only reachable by rounding. Draw from P[r] instead. The dense
    kernel returns V − 1 here; this difference is deliberate and must be written down in the kernel.
  - If no t hits, take the largest id with w > 0.

**Known difference** (inherited from SV1 §2.3): ties at the top-k boundary that extend past KP are cut.

### 3.3 Where the verify branches

In `eagle_sample`, before the dense `else`:

- **RS2 on and the batch qualifies** (`sampling_info.need_top_k_sampling` and `sparse_verify_width(max top_k) > 0`):
  - run `sparse_target_probs` (`apply_min_p` = `SPEC_MIN_P and need_min_p_sampling`, `use_flashinfer_topk` =
    `SPEC_SPARSE_TOPK`), then `chain_speculative_sampling_sparse`;
  - TP broadcast as SV1.
  - RS2 computes its own width; it does not depend on `SGLANG_OPT_SPEC_SPARSE_VERIFY`.
- **RS2 on, but the batch does not qualify** (some request has top_k > 248, or none uses top-k): densify. Scatter
  the support pair into a zero (bs, S, V) buffer and take the existing dense RS path. LM Studio always sends
  top_k 40, so this is rare.
- **All-greedy batches** keep the greedy branch.

## 4. Code rules (worktree)

- Comments ASCII, at most 2 lines. New containers use `msgspec.Struct`; the existing dataclasses only gain fields.
- No defensive `getattr`/`hasattr`. Calls with 2+ args use keywords.
- Follow the speculative-naming and env-var-conventions skills.
- Keep RS1's functions (`sample_draft_proposal_truncated`, `_rs_draft_proposal`, the dense kernel). They remain the
  RS2-off path and the test reference.
- Commit in the worktree only. Never touch the production checkout `~/tools/sglang-rtxpro6000`, the shared venv or
  the shared FlashInfer caches.

## 5. Tests (`bench/rs2/test_sparse_rs.py`; DEV=cpu with TRITON_INTERPRET=1, DEV=cuda later)

Model the tests on `bench/sv/test_sparse_verify.py` (SV1) and `bench/rs1/test_rs_hot.py` (RS1: worktree functions
extracted by AST, no sglang import). Use fixed seeds and print one PASS/FAIL line per part.

**A. Proposal vs `sample_draft_proposal_truncated`.** Cases:
- Vd 49152 and a small Vd;
- bf16 ties at the K boundary;
- T ∈ {0.3, 0.8, 1.5};
- top_k ∈ {1, 5, 40, 0};
- top_p ∈ {1.0, 0.95, 0.5};
- min_p ∈ {None, 0.05, 0.2};
- per-row mixed parameters;
- hot map on and off.

Pass:
- kept token sets equal;
- |Δq| per token < 2e-6;
- Σq = 1 ± 1e-5;
- no mass outside the kept set;
- padded filler rows (T 0, top_k 0) do not fault.

**B. Draw.**
- Over M uniforms per row, X ~ q (chi², p > 1e-3); X is never outside the kept set; `topk_p == q[X]`.
- The chain column holds `hot_token_id[draft id]`; positions advance by 1.
- top_k = 1 gives the argmax with smallest-id ties, equal to `draft_topk1_postprocess`.

**C. Verify vs the dense `speculative_sampling_classic_kernel`, equal coins.** Use the real kernel from
`reject_sampling.py`, with dense q from scattering the support pair and dense p from the dense chain math.
- S ∈ {1, 4, 7, 15}.
- Cases:
  - all accepted;
  - first draft rejected;
  - greedy rows;
  - X outside the target support (p(X) = 0, so rejected);
  - q mass outside the target support;
  - min_p on and off;
  - bs 1 and 4.
- Pass: same accept counts and accept_index, and same emitted tokens except rounding-edge cases. Count those; they
  must be ≤ 1 in 48, as in SV1.

**D. End-to-end losslessness.** Use RS2's proposal plus RS2's verify on synthetic chains, as RS1 §2 did.
- The emitted token at each chain position follows the target distribution: chi² per position, Fisher-combined.
- Position-0 acceptance matches Σ min(p, q).

**E. (cuda, parent runs it) Time.**
- Per draft step at n ∈ {1, 2, 4}, Vd 49152, captured in a CUDA graph of 14 steps: RS1 helper vs RS2 vs
  `draft_topk1_postprocess`.
- Per verify at bs × S ∈ {1×4, 1×8, 1×16, 4×4}: dense RS vs RS2 vs SV1's target-only sparse verify.
- GPU time and back-to-back wall.

**F. (server, parent runs it)**
- **F1 smoke:** the RS2 server (`serve-fast.sh wa --speculative-use-rejection-sampling`, stack flags, min_p on)
  starts, logs the RS2 line, the probes return text, and there are 0 error lines.
- **F1 profile:** no `mbtopk`, no V-wide memset or scatter in the draft graph; 2 proposal launches per draft step.
- **F2 greedy exactness:** w4 with `--disable-flashinfer-autotune` (the DT1 X/Y method), greedy prompts:
  - stack without RS;
  - stack + RS + RS2;
  - stack without RS again.
  - Identical text expected (§2.4).

## 6. Measurement and decision (after F)

- **8-start ABBA BAAB** (`bench/stack` style), with min_p on in both arms (`SGLANG_SPEC_MIN_P=1`):
  - A = stack (target-only);
  - B = stack + RS + RS2;
  - modes: LM Studio sampling and greedy;
  - analysis: `ancova_ab.py` per domain (t/s, tok/step, ms/step|a) plus the clipped R.eager.
- **Adopt as an option for LM Studio use** if:
  - pooled lmstudio t/s has a CI above 0;
  - the code-edit and agent-loop lower bounds are > −2%;
  - greedy ms/step is flat (≤ +1%);
  - there are no errors.
- **Out of scope, noted:** wa's C1 thresholds are calibrated on greedy acceptance. Under RS the acceptance per
  position is Σ min(p, q), so wa's widths may be off for sampling. Re-tune only after RS2 is adopted.
- Shipping RS2 (or min_p) to production is the user's decision.
