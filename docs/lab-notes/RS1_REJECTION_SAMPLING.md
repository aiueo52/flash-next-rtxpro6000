# RS1 (2026-10-01): lossless rejection sampling with the hot draft vocab

Status (2026-10-01 09:50): **measured on the GPU; not adopted as the default** (§4b without the min_p fix, §4c with it).

## 0. Summary

- **Why this matters.** Production tuning and BN1 were all greedy. LM Studio, however, samples with T=0.8, top_p=0.95, top_k=40, min_p=0.05.
  - Under sampling, production verifies with *target-only* acceptance: the draft argmax X is accepted with probability p(X).
  - Rejection sampling (Leviathan) draws X ~ q and accepts with min(1, p(X)/q(X)). This accepts more often whenever q is close to p.
  - The gain compounds over the long chains of wa, which go up to 15 steps.
- **What was missing.** The fork's `--speculative-use-rejection-sampling` did not work with the reduced draft vocab (FIXME in eagle_worker_v2). RS1 adds that support, plus a proposal truncated to the request's own top-k/top-p support.
- **Exactness, checked offline.** The test drives the real chain Triton kernel on the CPU with the proposal code extracted from the worktree.
  - The emitted tokens follow the target distribution at every chain position (38 chi2 tests, Fisher combined p = 0.76; §2).
  - The acceptance at position 0 matches theory.
  - Greedy requests reproduce the target argmax.
- **min_p gap, found along the way.** The speculative verify ignores min_p, in both target-only and RS mode. With LM Studio's min_p=0.05, speculative outputs therefore come from a wider distribution than the non-speculative sampler would use.
  - This gap is in production today.
  - A fix exists behind `SGLANG_SPEC_MIN_P=1` (dev branch). fc-map found the same gap independently (FC_fc-map §3.2).
  - The fix is for fidelity. On synthetic logits it does not speed RS up (mean accepted −1.8%).
  - ABBA C checks that RS still wins with the fix on.
- **Result:** RS gains acceptance under LM Studio sampling, but its per-step cost eats the gain on code (§4b, §4c).
  - With the min_p fix on both arms, acceptance rises in code, agent and prose-en (+5..+8%).
  - t/s: prose-en +5.3%, code-edit -3.6%, agent and prose-ja within noise.
  - Next step if pursued: remove RS's per-step cost (sparse q).

## 1. Implementation

Branch `opus/rs-hotvocab` in `~/tools/sglang-rs`, commit 8b3834d7ac (on top of production `codex/perf-v1`).

- `spec_utils.sample_draft_proposal_truncated`:
  - takes the top-K_cap (64) draft logits over the hot vocab and applies the temperature;
  - applies the request's top-k and top-p masks, with the same rule `top_p_renorm_prob` applies to p, and renormalises;
  - draws X with an exponential race.
  - It returns q on the K support, the ids, q(X) and X.
- `EagleDraftWorker._rs_draft_proposal` maps hot ids to vocab ids (`hot_token_id`) and writes q into `draft_probs[:, i+1]`. That buffer is one preallocated (bs, S, V) fp32 tensor.
- **Why truncate q.** The verify renormalises p to the request's top-k/top-p support, so any q mass outside that support is always rejected. Truncating q to the same support moves that mass onto tokens that can be accepted.
- Greedy requests (top_k=1) get a one-hot q at the draft argmax, which reproduces target-only acceptance.
- `SGLANG_RS_DRAFT_TOPK` sets K_cap (default 64). 0 means the untruncated proposal; that is for A/B only, because it also samples the drafts of greedy requests.
- The draft CUDA graph captures the per-request top_k/top_p buffers.
- The C1 position-0 confidence is still recorded for wa.

**min_p** is on branch `opus/rs-minp` in `~/tools/sglang-rs-dev`, commit 567f1f8167. It is behind `SGLANG_SPEC_MIN_P`, default off.
- `eagle_sample` drops p < min_p · max(p) after the top-k/top-p renorm, the same rule the non-speculative sampler uses.
- The RS proposal applies the same rule to q.
- The draft graph captures a min_p buffer.

## 2. Offline exactness

Test: `bench/rs1/test_rs_hot.py`.
- Real `chain_speculative_sampling_triton` under `TRITON_INTERPRET=1`.
- Setup: vocab 600, hot set 256, 3 draft steps, N=20000 rows per case.
- The hot set covers most, but not all, of each row's head.
- p-values are chi2 (Wilson-Hilferty) per chain position.

Logs (in `runs/rs1/`):
- `offline-n20000.log`: the default cases and `CASES=minp`;
- `offline-seeds.log`: `CASES=seeds`;
- `offline-minp-off.log`: `CASES=minp_off`.

Trees:
- rs = `~/tools/sglang-rs` (8b3834d7ac);
- dev = `~/tools/sglang-rs-dev` (567f1f8167).

| Case | Tree | Setup | accept@0 emp / theory | Mean accepted | chi2 p, pos 0 / 1 / 2 / 3 |
|---|---|---|---|--:|---|
| trunc | rs | T 0.8, top_k 40, top_p 0.95, K_cap 64 (seed 0) | 0.8182 / 0.8136 | 1.841 | 0.585 / 0.423 / 0.234 / 0.999 |
| untrunc | rs | same logits, K_cap 0 | 0.7995 / 0.7998 | 1.761 | 0.745 / 0.957 / 0.590 / 0.166 |
| no top-k | rs | T 1.0, top_p 0.9 (seed 3) | 0.5782 / 0.5802 | 1.187 | **0.011** / 0.479 / 0.646 / 0.954 |
| no top-k, fresh seeds | rs | seeds 11 / 12 / 13 | 0.5401 / 0.5402; 0.5353 / 0.5386; 0.5428 / 0.5422 | 0.997 / 1.052 / 1.024 | pos 0: 0.679 / 0.572 / 0.140; all others 0.14–0.98 |
| good draft, K_cap 16 < top_k 40 | rs | draft noise 0.3 (seed 5) | 0.6223 / 0.6173 | 1.317 | 0.864 / 0.298 / 0.325 / 0.895 |
| greedy | rs | top_k 1, N 5000 | 1.0000 / 1.0000 | 1.000 | TV 0 at pos 0 and 1 (always the target argmax) |
| min_p on | dev | `SPEC_MIN_P` on; min_p 0.05 on p and q; T 0.8, top_k 40, top_p 0.95 (seed 21) | 0.6171 / 0.6183 | 1.174 | 0.664 / 0.903 / 0.755 / 0.275 |
| min_p on, untruncated q | dev | same logits, K_cap 0 | 0.5925 / 0.5922 | 1.086 | 0.638 / 0.988 / 0.813 / 0.250 |
| min_p ignored (today's verify) | dev | same logits, `SPEC_MIN_P` off, min_p not applied | 0.6171 / 0.6183 | 1.196 | same draws as "min_p on" / same / 0.344 / 0.033 |

Reading:
- **The output distribution is exact.**
  - There are 38 independent chi2 tests. Fisher's combined p is 0.76, and 2 tests fall below 0.05 where 1.9 are expected.
  - The one low value (0.011, "no top-k" at pos 0) did not repeat on three fresh seeds.
- **accept@0 matches theory**, Σ min(p, q), within 1.6 standard errors in every case (SE ≈ 0.003).
- **Truncating q to the request's support helps.** On the same logits, accept@0 goes from 0.7995 to 0.8182, and the mean accepted length from 1.761 to 1.841 (+4.5%).
- **The min_p fix is about fidelity, not speed.**
  - On these logits it lowers the RS mean accepted length slightly: 1.196 → 1.174 (−1.8%).
  - Why it can go down: min_p cuts the tails of p and of q, each at its own maximum, and that can shrink their overlap.
  - At positions 0 and 1 no token fell below 0.05 · max, so the draws there are identical in both runs.
  - In target-only mode, the same fix can only raise p(X) for a draft token X that survives the cut.
  - ABBA C (§4) runs both arms with the fix, so it answers whether RS still wins with the fix on.
  - The fix's own speed effect is not interleaved anywhere. Comparing ABBA C's A arms with ABBA B's A arms from the same boot gives a rough number.
- With min_p on, the fully truncated q accepts 8% more than an untruncated q (1.086 → 1.174). That is the same truncation effect as above; it is not the effect of the fix.
- Not covered offline: the verify-side min_p mask in `eagle_sample`. It is a mask plus a renorm after top-p, the same rule as the non-speculative flashinfer sampler (top-k renorm → top-p renorm → min_p). It is reviewed, and it first runs on the GPU in ABBA C.

## 3. Expected effect and cost (not measured yet)

- **Gain.** Only sampling requests gain. The proposal matches p better, so per-token acceptance rises, and the effect compounds over wa's 7- and 15-step chains.
- **Cost on every step, greedy included.** The draft graph always runs the top-64 proposal and writes the full-vocab q (bs × S × 248320 fp32; 15 MB at S=15).
  - Estimates: fc-map puts the buffer at +10..25 µs/step (FC_fc-map §3.2), and my loose upper bound was ≤ 0.5 ms/step at S=15.
  - The ABBA measures it directly (greedy grid in every arm).
- **The sampling verify itself is a fixed cost in both arms** (fc-map §3.1): it runs eagerly, outside the CUDA graphs, and makes five full-vocab passes per row, which is +0.08..0.22 ms/step.
  - fc-map's sparse top-k verify (F2) removes most of that.
  - For RS the residual max(0, p − q) needs only the union of the two supports: at most 40 + 64 entries with the truncated proposal.
  - So F2 also fits RS, and it is the natural next step after the ABBA.
- **C1/wa tuning.** The C1/wa thresholds were calibrated on greedy acceptance. If RS wins, re-tune them for sampling next.

## 4. Pending GPU measurements (after the reboot)

Record the conditions next to every number:
- power limit: 300 W unless the user re-applies 325 W;
- whether the display runs on the NVIDIA GPU again;
- compare only within one boot.

Every server start waits for `MemAvailable ≥ 110G` (`bench/rs1/memcheck.sh`) and runs under `MemoryMax=110G`, `MemorySwapMax=0`.

```
cd ~/tools/flash-next-bench
# A. smoke: both servers start, RS graphs capture, probes clean (~15 min)
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
  bash -c 'bash bench/rs1/smoke.sh smoke-rs rs; bash bench/rs1/smoke.sh smoke-to to' > runs/rs1/smoke.log 2>&1
# B. ABBA: A1 to / B1 rs / B2 rs / A2 to; each arm = fresh wa server, BN1 grid (4 domains x 8 prompts)
#    with LM Studio sampling, then greedy; paired analysis per mode (~70 min)
bash bench/rs1/abba.sh > runs/rs1/abba.log 2>&1
# C. (optional) the same with the min_p fix on both arms
WT=~/tools/sglang-rs-dev SERVER_ENV="SGLANG_SPEC_MIN_P=1" OUTDIR=runs/rs1-minp bash bench/rs1/abba.sh -minp > runs/rs1/abba-minp.log 2>&1
```

After the RS runs, the investigators' GPU measurements (FC_fc-glue §4, FC_fc-moe §4, FC_fc-map §5).
- Cheapest first: fc-glue `bench_gdn_front.py` + `bw_probe.py` (≈3 min).
- Then fc-moe `gpu1.sh` (30–40 min).
- Then fc-map's w4/w16/wa holds (≈20 min each). These also give the measured sampling fixed cost and a 300 W refit of the step model.

Decision:
- **Adopt RS** if the LM Studio paired t/s gain has its lower bound above 0, and the greedy change stays within noise.
  - BN1: wa's ABBA MDE is 6–31%.
  - Read per domain, and trust code/agent first.
- **Adopt the min_p fix** on fidelity grounds: it matches what the user asked the sampler for. The speed number is secondary.

## 4b. Results (2026-10-01, first boot after the reboot)

Conditions (runs/rs1/conditions.txt): 325 W, driver 595.91.07, display on the NVIDIA GPU (Xorg/KDE, ~1.4 GB), another CPU-heavy job on the host
(7 shards x 4 threads). Production `fused_moe_120` rebuilt at 04:42 (standard commands), weights freshly rewritten.
Arms: A1 to 04:53, B1 rs 05:27, B2 rs 06:00, A2 to 06:30; ~28 min each; 32 prompts x 2 samplings per arm, 0 error lines.
(`[abba] arm failed rc=1` is the final `grep -c` finding 0 error lines; not a failure.)

**Smoke (A):** pass. Both servers up in ~4 min (weight load 131-152 s), RS enabled per log, 12/12 probes returned text.

**ABBA (B), rs / to, paired over 8 prompts per domain** (`runs/rs1/paired-{lmstudio,greedy}.json`):

| LM Studio sampling | t/s | 95% CI | acceptance (tok/step) | 95% CI |
|---|---|---|---|---|
| agent-loop | +1.0% | -2.8 .. +4.9 | +4.5% | -0.8 .. +10.0 |
| code-edit | **-7.7%** | -12.0 .. -3.2 | -2.6% | -9.8 .. +5.3 |
| prose-en | **+6.2%** | +3.9 .. +8.8 | **+7.7%** | +7.1 .. +8.3 |
| prose-ja | +0.1% | -4.8 .. +4.4 | +0.5% | -5.8 .. +5.4 |

Greedy: -1.3 / -2.7 / -4.2 / -3.6% (agent / code / prose-en / prose-ja), all CIs include 0; drift flags on three domains.

Split by arm (vs the mean of A1 and A2; t/s, tok/step, ms/step):
- tok/step agrees between the two RS arms: prose-en +8.0 / +7.5%, agent +2.7 / +6.3%, prose-ja +0.4 / +0.6%, code-edit -0.1 / -5.0%.
- ms/step does not: B1 was slower per step in every domain (+3.6 .. +11.2%), B2 was not (-2.6 .. +2.0%).
  Restart-to-restart step time moves by ~5-10% in this setup (greedy A2 vs A1 drift -6 .. -8%), so one ABBA cycle
  cannot pin the RS step cost below ~5%. Most of code-edit's -7.7% comes from B1's slow steps (B2 alone: -5.1%).
- Greedy tok/step moves by up to ±8% between restarts of the *same* arm type (long thinking outputs diverge), so the
  greedy rows are noise, not an RS effect on greedy.
- **Caveat found later (FC_fc-map §5b c):** at 07:1x the display contexts time-sliced the GPU for 20-28% of every
  step span (kernels paused 0.3-1.4 ms at a time). Without the pauses, the kernels ran as fast as on 09-07.
  - The share moves with desktop activity (2-8% on 09-07 02:07, 19% on 09-07 23:37). So it adds noise between arms
    to ms/step, not to tok/step.
  - B1's slower steps fit a busier desktop during B1, but the ABBA did not record the desktop state, so this is not
    proven.
  - The acceptance results stand. The RS step-cost read is weaker than the CIs suggest.
  - The decision stands: code-edit's tok/step is not better under RS either (-0.1 / -5.0%).
  - Next time, run the arms with the desktop idle, or read step cost from pause-clipped traces
    (`prof/fc_eager_clip.py`, `prof/fc_pause_share.py`).

**Decision: not adopted as the default.** The rule reads code/agent first: code-edit's lower bound is -12%, agent-loop's -2.8%.
- RS pays where the target is spread out (prose-en: +7.7% acceptance, robust in both RS arms).
- On code the target is peaked, target-only acceptance is already near RS's, and RS only adds step cost.
- Next, if RS is pursued: remove the per-step cost first (sparse q instead of the full-vocab (bs, S, V) buffer, plus
  fc-map's sparse verify F2), then re-measure. With zero overhead the expected result is prose-en +7-8%, agent +4%,
  code-edit and prose-ja about 0.
- The min_p ABBA (C) runs after the fc queue and is a second, independent rs/to read under min_p.

## 4c. min_p ABBA (C): rs / to with the min_p fix on both arms

Same boot and conditions as §4b. Worktree `~/tools/sglang-rs-dev` (567f1f8167), `SGLANG_SPEC_MIN_P=1` in every arm.
Arms: A1 to 07:24, B1 rs 07:58, B2 rs 08:32, A2 to 09:07; ~33 min each; 0 error lines (the `arm failed rc=1` lines are the
same cosmetic `grep -c` exit as in §4b). The clock log covers A1 only: SM 2310 MHz median, SW power cap in 16% of samples.

**Paired over 8 prompts per domain** (`runs/rs1-minp/paired-minp-{lmstudio,greedy}.json`):

| LM Studio sampling | t/s | 95% CI | acceptance (tok/step) | 95% CI |
|---|---|---|---|---|
| agent-loop | +1.4% | -1.7 .. +5.3 | **+6.4%** | +1.0 .. +13.4 |
| code-edit | **-3.6%** | -6.8 .. -0.0 | +5.1% | -1.5 .. +11.9 |
| prose-en | **+5.3%** | +3.8 .. +6.6 | **+6.6%** | +5.8 .. +7.4 |
| prose-ja | -1.4% | -13.8 .. +7.3 | -2.3% | -17.1 .. +8.3 |

Greedy: +1.5 / -3.6 / +2.6 / +3.1% (agent / code / prose-en / prose-ja); drift flags on all four, so noise.

Split by arm (vs the mean of A1 and A2; tok/step, ms/step, t/s):

| LM Studio sampling | B1 | B2 | A2 vs A1 |
|---|---|---|---|
| agent-loop | +8.3 / +5.4 / +2.8 | +4.6 / +4.5 / +0.1 | -2.9 / -4.7 / +1.9 |
| code-edit | +5.2 / +7.5 / -2.1 | +4.9 / +10.3 / -5.0 | -2.9 / -5.2 / +2.4 |
| prose-en | +6.6 / +0.8 / +5.8 | +6.7 / +1.8 / +4.8 | +0.7 / -2.8 / +3.7 |
| prose-ja | -0.3 / -1.0 / +0.8 | -4.3 / -0.8 / -3.5 | -9.8 / -4.7 / -5.3 |

- **With min_p honored, RS raises acceptance on code too:** +5.2 / +4.9% in both RS arms, against -0.1 / -5.0% without the
  fix (§4b). Agent +8.3 / +4.6%, prose-en +6.6 / +6.7%.
- **The per-step cost is what loses.** Both RS arms are slower per step on code-edit (+7.5 / +10.3%) and agent (+5.4 / +4.5%),
  but barely on prose-en (+0.8 / +1.8%).
  - This fits a cost that grows with the chain length: code and agent run the longest wa chains (8-9 tok/step).
  - Restart noise is ~5% (A2 vs A1), so one arm alone proves little. Both B arms agree here, though.
- **The fix itself costs nothing measurable.** Compared with the §4b arms of the same mode (same boot, earlier):
  to ms/step -1.6 .. +0.6%, rs -0.7 .. +1.8%; tok/step moves -3.4 .. +4.2%. That is within restart noise.

**Decision: RS stays off by default** (code-edit's lower bound is -6.8%, agent's -1.7%).
- **The min_p fix:** keep it as a fidelity fix. It makes the verify sample from the distribution LM Studio asked for, at no
  measurable speed cost. Moving it into production is a separate step (prod is not touched by this work).
- **What RS would be worth:** with zero per-step cost, B's tok/step says code +5%, agent +6%, prose-en +7%,
  prose-ja about 0. That is the case for the sparse-q rework (§4b, Next).

## 5. Incident notes (2026-10-01)

- **02:54 freeze and 03:00 reset: memory exhaustion.**
  - A server pins about 67 GB of host RAM for the PLE table (cudaHostAlloc, never reclaimable), and swap is only 8 GB.
  - An uncapped MAX_JOBS=16 FlashInfer rebuild ran next to a server and another CPU-heavy job.
  - New rules: `MemAvailable ≥ 110G` before a server; servers under `MemoryMax=110G`/`MemorySwapMax=0`; compiles in their own `MemoryMax=40G` scope with `MAX_JOBS=4`.
- **03:10 GPU hang.** The machine had booted a kernel without the nvidia module. The module was then installed and hot-loaded (595.91.07) and failed DMA init:
  - NVRM `GPPut < WATCHDOG_GPFIFO_ENTRIES`;
  - nvidia-modeset `Failed to initialize DMA`.
  - Every CUDA/UVM client, the RS smoke server included, is stuck in the kernel and cannot be killed. The stuck server holds `~/.gpu.lock`.
  - Only a reboot clears it.
- **Lost files.** The reboot cleared /tmp, which held the first copy of the offline test. The test was rebuilt from the session log and committed as `bench/rs1/test_rs_hot.py`.
