# SV1: sparse target-only sampling verify (fc-map F2 / S1)

Status (2026-10-01 12:40): ADOPT (§5). The server ABBA cut the sampling eager tail by 205 µs/step with tok/step unchanged (§4). Commit 4f9cf50619 on `opus/spec-sparse-verify`; in the stack as 77dabf210a.

## 0. Summary

- **What:** replace the dense sampling verify with a top-KP path when every request in the batch has `top_k <= 248`.
  - Today's dense verify runs five full-vocabulary passes per draft row:
    1. softmax over V = 248,320;
    2. FlashInfer radix top-k renorm;
    3. AIR top-p renorm;
    4. an optional min-p pass;
    5. a `zeros_like` draft buffer.
  - After those, the tree kernel reads each row once more for its final draw.
- **Why:** on sampling requests the verify costs +250..+300 µs per step more than on greedy ones (FC_fc-map §5b(a), clipped R.eager). The estimated saving is **−0.20..−0.26 ms/step**, about 2–3% of a step.
  - This only helps LM-Studio-style requests (temperature, top_k 40, top_p 0.95, min_p 0.05).
  - Greedy requests take their own branch and do not change.
- **Where:**
  - Tree: `~/tools/sglang-sv`, branch `opus/spec-sparse-verify`, based on 567f1f8167 (the sglang-rs-dev min_p fix).
  - Switch: `SGLANG_OPT_SPEC_SPARSE_VERIFY=1`, default off.
- **Output difference:** rounding-level, with one known exception (§2.3).

## 1. When the sparse path runs

In `eagle_utils.eagle_sample`, the new `elif` sits between the greedy branch and the dense branch. `_sparse_verify_kp` returns KP > 0 only if all of these hold:

- the env switch is on;
- `sampling_info.need_top_k_sampling` is set (at least one request has a top_k);
- rejection sampling (RS1) is off, since RS needs the draft residual over the full vocabulary;
- `max(r.sampling_params.top_k for r in batch.reqs) <= 248`.
  - A request without a top_k has `TOP_K_ALL = 2^30`, so a mixed batch goes dense.
  - Greedy requests inside a sampling batch have top_k = 1 after normalization.

KP = `max(16, next_pow2(max_top_k + 8))`, so top_k 40 gives KP = 64. The max is read from the requests on the CPU, which costs no GPU sync.

Penalties, logit bias and the grammar mask are applied to `next_token_logits` before the branch, as in the dense path.

## 2. Kernels (`python/sglang/kernels/ops/speculative/sparse_verify.py`)

1. **`torch.topk(logits, KP, sorted=True)`** on the raw logits. Division by a temperature > 0 keeps the order, so only the KP values are divided later.
2. **`_sparse_target_probs_kernel`**, one program per draft row:
   - temperature, then softmax over the KP values;
   - **top-k, keeping ties:** p >= the k-th largest, as FlashInfer's radix top-k renorm does;
   - **top-p, keeping ties:** p >= the value at which the sorted cumulative mass reaches top_p, as AIR top-p does;
   - optional min-p (`SGLANG_SPEC_MIN_P`): p >= max(p) · min_p;
   - one final renormalization. Renormalizing once over the kept set is the same distribution as the dense chain of renorms.
3. **`_tree_sampling_target_only_sparse_kernel`**, one program per request. It is the tree walk of `TreeSpeculativeSamplingTargetOnly`:
   - `prob_acc` accumulates over siblings; the thresholds work the same way.
   - A rejected sibling's mass is removed from the final draw. The CUDA kernel skips the removal when every draft was accepted (`num_accepted != num_spec - 1`). Here it is always done, with the same result: when every draft is accepted, the walk stops before testing any child of the final row, so nothing was rejected there.
   - The final draw takes the cumulative sum **in token-id order**, like the dense kernel's full-row scan. So equal coins give equal tokens, except where a coin lands within rounding of a bucket edge.
   - Fallback when nothing is hit: the last valid id, else V − 1 (same as the dense kernel).

### 2.1 Launch count

| Path | Per verify |
|---|---|
| Dense | softmax, division, repeat_interleave ×2–3, radix top-k renorm (multi-CTA), AIR top-p (3 kernels), `zeros_like`, tree kernel, coins ×2 |
| Sparse | `torch.topk` (a few radix passes), probs kernel, tree kernel, coins ×2 |

### 2.2 Memory

The sparse path allocates N × KP floats and N × KP int64 (N = bs · draft tokens). That is small, so it does not use `borrow_graph_pool`.

### 2.3 Known difference: ties past KP

If more than KP − top_k tokens tie with the k-th value, the tie group is cut at KP. The dense path keeps the whole group.

- KP leaves at least 8 slots of margin; with top_k 40 it leaves 24.
- Only tokens at the k-th value are affected, and those carry the least mass of the kept set. top-p 0.95 and min-p usually drop them anyway.
- Test A's bf16 case measures whether FlashInfer really keeps ties.

## 3. Validation (`bench/sv/test_sparse_verify.py`)

- `DEV=cpu` runs the Triton kernels in the interpreter against Python ports of the dense chain and of `TreeSpeculativeSamplingTargetOnly`.
- `DEV=cuda` runs them against the real dense kernels (softmax, sgl_kernel `top_k_renorm_prob` / `top_p_renorm_prob`, `tree_speculative_sampling_target_only`).
  - It needs FlashInfer's JIT sampling module.
  - Use the private workspace from `scratchpad/fienv.sh`, never the server's shared cache.

| Part | Check | Pass |
|---|---|---|
| A. probabilities | peaked, bf16-tied and grammar-masked rows; with and without mixed per-request parameters and min-p | mass outside the kept set < 1e-6, no kept-set differences, max \|Δp\| < 2e-6 |
| B. tree walk and final draw | equal coins: chains of S = 4 and 16, a tree with siblings, top_k = 1 (full acceptance) | same accept path and same emitted tokens (up to bs/48 edge cases) |
| C. distribution | first emitted token over M coins vs the target row | nothing outside the support, chi² p > 1e-3 |
| D. time (cuda) | dense vs sparse, (bs, S) ∈ {(1,4), (1,8), (1,16), (4,4)} | GPU time and back-to-back wall per call |

**CPU result (08:09, after the renames):** all PASS.
- A: kept sets identical in all 9 cases, max |Δp| ≤ 1.2e-7.
- B: 10/10 cases identical, including full acceptance and siblings.
- C: chi² p = 0.25–0.49.

**GPU result (08:31, `runs/sv/test-gpu-1001.log`):** all PASS against the real dense kernels.
- A: kept sets identical in all 9 cases, including the bf16 ties (up to 44 kept at top_k 40, so FlashInfer keeps ties as assumed); max |Δp| ≤ 4.2e-7.
- B: 480/480 equal-coin trees give the same accept path and tokens (10 cases × 48), including full acceptance and siblings.
- C: chi² p = 0.13–0.82, nothing outside the support.
- D, µs per verify (median; GPU = kernels queued behind a sleep, b2b = back-to-back calls):

| bs, draft tokens | dense GPU | sparse GPU | saved | dense b2b | sparse b2b | saved | of the sparse GPU time, `torch.topk` |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1, 4 | 340.0 | 73.7 | **266** | 375.5 | 121.0 | 255 | 67.6 |
| 1, 8 | 325.6 | 73.7 | **252** | 359.9 | 106.8 | 253 | 68.6 |
| 1, 16 | 396.3 | 81.9 | **314** | 438.7 | 136.0 | 303 | 76.3 |
| 4, 4 | 386.0 | 81.9 | **304** | 429.9 | 128.9 | 301 | 75.8 |

- The dense numbers match the server trace (+250..+300 µs/step on sampling requests, FC_fc-map §5b a), so the whole sampling surcharge is gone except for the top-k.
- `torch.topk` is ~90% of what remains (65–76 µs; unsorted saves only 3–5 µs). Its cost is mostly fixed: the multi-pass radix select launches about ten kernels even for 4 rows. The probs and tree kernels together take ~6 µs.
- **Next lever, if the ABBA confirms the gain:** a one- or two-launch top-k for few long rows would remove another ~50 µs/step.

## 4. Server A/B (`bench/sv/abba_sv.sh`)

```
bash bench/sv/abba_sv.sh > runs/sv/abba-sv.log 2>&1
```

- The arms run A1 dense, B1 sparse, B2 sparse, A2 dense.
- Each arm takes one GPU-lock hold (`bench/sv/arm_sv.sh`): a fresh `serve-fast.sh wa` from `~/tools/sglang-sv`, then the BN1 grid (8 prompts × 4 domains) for `lmstudio` and `greedy`, then one profiled 20-step LM-Studio probe per workload.
- Afterwards: `paired_ab.py --schedule ABBA` per sampling mode, and `fc_eager_clip.py` over the traces.

What to read:

- **lmstudio:** the paired tok/s ratio (the target is about +2–3%), and tok/step, which must not move: equal distribution means equal acceptance.
- **greedy:** the control. It should be flat, since greedy does not take the sparse path.
- **clipped R.eager, sampling:** it should drop from ~360–380 µs/step (wa) toward the greedy ~100 µs plus the sparse cost.
- The server logs must have 0 Traceback / Error lines.

### 4.1 Result (09:49-12:15, `runs/sv/abba-sv.log`)

- All four arms: server up, 0 Traceback/Error lines. The B traces have 20 `_sparse_target_probs_kernel` and 20
  `_tree_sampling_target_only_sparse_kernel` per 20-step probe; the A traces have none.
- **Clipped R.eager, µs/step (code-edit / prose-en): A1 376 / 387, A2 367 / 376; B1 182 / 165, B2 159 / 157,
  so −205 µs/step.** `torch.topk` (`at::mbtopk`, 52-59 µs) is now the largest eager kernel.
- Paired lmstudio (`paired_ab.py`):

| workload | tok/s | tok/step | A2/A1 tok/step |
|---|---|---|---|
| agent-loop | +5.2% [+3.3, +7.0] | +5.1% [+2.0, +8.4] | −2.3% |
| code-edit | +2.8% [−1.0, +6.7] | +4.0% [−3.5, +11.5] | −8.0% (drift) |
| prose-en | +1.7% [+1.0, +2.7] | −0.4% [−1.1, +0.4] | +0.2% |
| prose-ja | +7.6% [+1.5, +16.7] | +5.5% [−1.3, +15.8] | +5.1% (drift) |

- Greedy (control; greedy batches never reach the sparse branch): tok/step pooled +0.6% [−1.7, +3.1], but
  ms/step pooled −1.3% [−2.4, −0.2] (`bench/sv/msstep.py`). A2 ran slower than A1 in every greedy workload
  (ms/step +2.5 / −0.3 / +3.4 / +2.4%), which favours B by about half that. Background CPU work started near the
  end of A2 (new jobs around 12:00).

### 4.2 The tok/step check (§5)

- Without a seed, lmstudio tok/step varies by about ±15% per prompt between identical arms, so 8 prompts × 2
  blocks cannot resolve a few percent.
- `bench/sv/pool_acc.py` compares the two sparse arms against all six dense, min_p-off lmstudio runs of the day
  (rs1 A1/A2, rs1-minp A1/A2, sv A1/A2), with an exact permutation test over which 2 of the 8 runs are sparse:

| workload | sparse vs dense tok/step | p |
|---|---|---|
| agent-loop | −0.2% | 1.00 |
| code-edit | +3.1% | 0.18 |
| prose-en | −0.9% | 0.18 |
| prose-ja | +2.1% | 0.68 |
| pooled | +1.0% | 0.64 |

- The ABBA's tok/step shift came from its two dense arms, which were low. For example, agent-loop gave 3.45 and
  3.37, against 3.61-3.72 in the morning's dense runs; B1 gave 3.73.
- So the server agrees with §3: equal distributions. Ties past KP (§2.3) do not show.
- Time at equal tok/step: lmstudio ms/step for prose-en is −2.1% [−2.6, −1.6], and for prose-ja −1.9%
  [−3.0, −0.6]. The greedy bias above is part of this. agent-loop and code-edit are confounded: `wa` drafts more
  when tok/step is higher, which lengthens the step.

## 5. Decision

- **Adopt (default on in the fork)** if all of these hold:
  - lmstudio tok/s improves with a CI that excludes 0, or clipped R.eager drops by at least 150 µs/step;
  - lmstudio tok/step is unchanged within noise;
  - greedy is flat;
  - there are no errors.
- If the saving is real but tok/step moves, check §2.3 ties and the top-p tie rule first.

**Verdict (2026-10-01 12:40): ADOPT. SV1 goes into the stack's on-set.**

- R.eager falls by 205 µs/step, which clears the 150 bar.
- tok/step is unchanged against the day's dense runs (§4.2).
- Greedy tok/step is flat. The greedy ms/step shift is the A2 drift: greedy does not run the new code.
- 0 errors.
- Expected gain: about −0.2 ms per sampling step, roughly 1-2% of a step. Greedy requests are unchanged.
- Shipping as the fork default still needs the user's go-ahead, together with the stack.
- Next lever: `torch.topk` costs about 55 µs/step of the remaining ~160 µs. A one- or two-launch top-k for a
  few long rows would remove most of it.
