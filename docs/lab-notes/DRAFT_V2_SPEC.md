# Draft v2: can a stronger draft raise acceptance 10-20 %, and what would it take?

Design study, 2026-09-05. **READ-ONLY: no GPU used, no server started, no repo file changed.**
Every number is tagged **[M]** measured (with source), **[P]** from a paper (cited), or
**[E]** my estimate (with the arithmetic shown).

---

## 0. Verdict

**Yes at W16 (15 steps), where +10-40 % is on the table. No — or only ~+3-5 % — at W4.**
The two profiles are limited by different things, and the cheapest intervention differs.

Ranked, with break-even (the acceptance gain that merely pays for the extra draft cost):

| # | option | break-even W4 / W16 | est. acceptance gain | verdict |
|---|---|---|---|---|
| **1** | **EAGLE-3-style multi-layer fusion bolted onto the existing MTP head** (2 extra taps, zero-init) | **+0.3 % / +1.0 %** [E] | +3-8 % W4, +5-15 % W16 [E] | **GO, phase-gated** |
| **2** | **PosS-style per-position entry adapters** (per-draft-step LoRA on the entry), K=15 rollout training | **~+0.5 % / ~+0.5 %** [E] | +1-3 % W4, +8-20 % W16 [E] | **GO first** (needs no new data, no target change) |
| 3 | deeper head (+1 decoder layer, copy-init) | +3.4 % / +9.5 % [E] | unknown, +5-15 % [E] | hold — worst cost/benefit of the three |
| 4 | tree / topk>1 at W4 | +11.3 % [E] | ≤ +5 % on code (capped), ~+18 % prose [E] | **NO** (see §3.3) |
| 5 | n-gram chain hybrid (`codex/ngram-chain-v0`) | n/a | measured: code **−22 %**, agent +8 %, prose +9 % [M] | **NO** — already merged, already measured, loses on the flagship workload |
| 6 | replace the head with a from-scratch EAGLE-3 draft | −25 % (cheaper) [E] | large loss [E] | **NO** — throws away the vendor's pretrained 2.6 B head |

Options 1 and 2 compose and share a training run.

---

## 1. Where the acceptance actually goes (the model that drives everything below)

Chain acceptance with topk=1 over S steps is well described by a per-step acceptance
probability `p`: `E(p,S) = (1 - p^(S+1)) / (1 - p)`. Fitting `p` separately to each profile
from today's server measurements on the public workloads [M, task brief]:

| workload | W4 accept [M] | p fitted from W4 (`p_early`) [E] | W16 accept [M] | p fitted from W16 (`p_avg`) [E] | gap |
|---|---:|---:|---:|---:|---:|
| code-edit | 3.77 | **0.961** | 8.59 | **0.910** | −0.051 |
| agent-loop | 3.21 | **0.855** | 4.77 | **0.796** | −0.059 |
| prose-en | 2.66 | **0.731** | 2.99 | **0.666** | −0.065 |

`p_avg < p_early` in every workload: **the draft's per-step acceptance decays along the chain.**
That decay is exactly the failure EAGLE-3 and PosS were built to fix — EAGLE-3 reports that its
acceptance rate "stays almost flat across positions, whereas EAGLE's drops significantly" [P, EAGLE-3].

**The prize, quantified.** If a stronger draft flattened the decay so that `p_avg = p_early`
(the optimistic ceiling of this whole line of work), with step time unchanged at 19.7 ms [M]:

| workload | W16 accept now [M] | flat-p ceiling [E] | gain | t/s now [M] → ceiling [E] |
|---|---:|---:|---:|---|
| code-edit | 8.59 | **12.07** | **+40 %** | ~470 → ~660 |
| agent-loop | 4.77 | **6.33** | **+33 %** | 238 → 316 |
| prose-en | 2.99 | **3.69** | **+24 %** | 152 → 188 |

At W4 that lever yields **nothing** by construction (`p_early` is already the fit). W4 can only be
moved by raising the one-step accuracy itself, and it is inelastic there [E]:

| Δ`p_early` | code W4 | agent W4 | prose-en W4 |
|---|---|---|---|
| +0.02 | 3.77 → 3.89 (+3.1 %) | 3.21 → 3.31 (+3.1 %) | 2.66 → 2.74 (+3.0 %) |
| +0.05 | 3.77 → 3.95 (+4.7 %) | 3.21 → 3.47 (+7.9 %) | 2.66 → 2.87 (+7.8 %) |

**A W4 gain of +10 % needs Δ`p_early` ≈ +0.06 absolute — i.e. cutting the head's one-step error
rate by a fifth to a quarter.** v3's distillation bought roughly Δp = +0.013 (W4 prose-en
2.39→2.54, +6 % t/s) [M, v3 server A/B on the public workloads]. So a further +0.06 is 4-5 v3-sized wins.
This is the honest reason to expect the answer "yes at W16, no at W4".

---

## 2. Break-even arithmetic

Corrected per-phase step times (`prof/OVERHEAD_REPORT.md` §2, interval-union of kernel intervals,
19-step medians) [M]:

| profile | draft phase | draft_extend | verify | wall |
|---|---:|---:|---:|---:|
| W4 (S=3, T=4), code / prose | 0.58 / 0.58 | 0.45 / 0.44 | 10.56 / 10.31 | **11.95 / 11.68** |
| W8 (S=7, T=8) | 2.10 / 2.12 | 0.54 / 0.51 | 11.80 / 11.55 | 14.90 / 14.62 |
| W16 (S=15, T=16) | 4.70 / 4.66 | 0.58 / 0.55 | 14.89 / 14.19 | **20.60 / 19.77** |

`step_ms ≈ 9.74 + 0.700·S`, split **0.322 ms per extra draft forward + 0.378 ms per extra verify
token** [M, same report §TL;DR].

Draft-side cost that scales with a heavier draft = draft phase + draft_extend =
**1.06 ms at W4, 5.05 ms at W16** [M]. If the draft's per-forward cost is multiplied by `m`:

```
break-even acceptance multiplier  =  1 + (draft_side / wall) · (m − 1)
   W4  :  1 + 0.089·(m − 1)          W16 :  1 + 0.249·(m − 1)
```

| `m` (draft slowdown) | W4 needs | W16 needs |
|---|---|---|
| 1.05 | +0.4 % | +1.2 % |
| 1.4 | +3.6 % | +10.0 % |
| **2.0** | **+8.9 %** | **+24.9 %** |
| 3.0 | +17.8 % | +49.8 % |

Confirms the brief's "≈ +25 % at W16 for a 2× draft"; the W4 figure is **+9 %**, not +25 %,
because at W4 the draft is only 8.9 % of the step.

**Draft bytes model** [E], used to convert an architecture change into `m`. At bs=1 the draft is
purely bandwidth-bound; per forward, with serving quantisation
(`--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense`, NVFP4 routed experts):

| component | bytes |
|---|---:|
| lm_head, hot2_49152 (49 152 × 2560, FP8) | 126 MB |
| attention (q 12288×2560, k/v 512×2560, o 2560×6144, FP8) | 50 MB |
| routed MoE, top-10 of 512 × 4.915 M params, NVFP4 + scales | 26 MB |
| shared expert + router | 6 MB |
| 2 × gated-residual hyper-connections (lowrank 320 over 10240) | 13 MB |
| entry `fc_embedding` + `fc_hidden` (2×2560², BF16) | 26 MB |
| **total** | **~247 MB** |

247 MB / 0.322 ms = 767 GB/s effective, i.e. 43 % of the card's 1792 GB/s — consistent with
the 1240-1300 GB/s marginal bandwidth measured for the *large* MoE GEMMs
(`specs/MOE_SMALLM_SPEC.md` §8.3 [M]) once the draft's many tiny kernels are accounted for.
**Half the draft's cost is the lm_head, not the transformer body** — which is why adding a layer
costs less than 2×, and why a "smaller" EAGLE-3 draft would not be much cheaper.

---

## 3. The options, with numbers

### 3.1 Option 1 — EAGLE-3-style multi-layer fusion on the *existing* head (recommended)

Keep the vendor's 2.607 B MTP head. Widen only its **entry**: instead of one 10240-wide
hyper-connection state from layer 47, feed it three taps (two new + the existing one), each
projected by its own 2560→2560 matrix, summed as today.

* Paper evidence [P, EAGLE-3 Table 2, LLaMA-Instruct-3.1-8B]: the ablation separates the two
  ideas — removing the feature-regression constraint took τ 4.05 → 5.37 (+33 %), and *then*
  multi-layer feature fusion took 5.37 → 6.13 (**+14 %**). Main table: EAGLE-2 → EAGLE-3
  τ 4.05→6.13 (MT-bench), 4.71→6.74 (HumanEval), 4.24→6.23 (GSM8K).
* **Only the +14 % fusion term is available to us**, and it must be discounted:
  - the Qwen4-Exp MTP head already predicts tokens directly through its own lm_head — there is no
    feature-regression constraint to remove [M, `qwen4_exp_mtp.py:160-178`];
  - the rollout ("training-time test") and top-8 distillation terms are already spent (mtp-train
    v2 / v3; see `optimizations.md` A10);
  - the draft's input is *already* 4 hyper-connection residual streams (10240 = 4×2560), not a
    single final hidden — some multi-scale information is present [M, `qwen4_exp.py:1714`].
  - **Estimate after discount: +3-8 % acceptance at W4, +5-15 % at W16.** [E]
* Cost. The two new taps are consumed **only on the chain's first forward** (steps 1..S−1 already
  recurse on the head's own `own_hc`), so the extra work is 2 × 2560² BF16 = 26 MB once per step:
  ~0.02 ms of GEMV [E] plus target-side packing of 2 extra 10240-wide vectors for T tokens
  (T=16 → 655 kB, <0.01 ms) [E]. Allow +0.2 ms/step for kernel-launch and buffer bookkeeping [E]:
  **m ≈ 1.03 → break-even +0.3 % (W4) / +1.0 % (W16).**
* Resulting t/s at the mid estimate (+5 % W4, +10 % W16) [E]:
  W4 prose-en 230→241, agent 289→303, code 332→348;
  W16 prose-en 152→167, agent 238→261, code ~470→~516.
* Extra VRAM: 13.1 M BF16 params = **26 MB**. Irrelevant.

### 3.2 Option 2 — PosS-style per-position entry adapters (do this one first)

The §1 model says the W16 prize is the positional decay. PosS attacks exactly that with
position-specialised draft layers and reports **+9.2 % average acceptance over EAGLE-3**
(4.69 → 5.12 across 6 datasets, speed-up 2.96× → 3.27×) [P, PosS/arXiv 2506.03566].

Full per-position draft layers are impossible here (15 × 2.6 B). The cheap analogue: a **rank-64
LoRA on `fc_hidden` and a per-position `pre_fc_norm_hidden` scale**, indexed by draft-step number.
Size: 15 × (2×64×2560 + 10240) ≈ **5.1 M params, 10 MB** [E]. Runtime cost ≈ +1.3 MB traffic per
draft forward → **m ≈ 1.005, break-even ≈ +0.5 %** [E]. Graph-safe: the draft loop is captured as
**one** CUDA graph with the S steps unrolled (exactly 4 graph replays per step for S=3..15
[M, `prof/OVERHEAD_REPORT.md`]), so a per-step weight is a statically-baked pointer, not a branch.
*(To verify before committing: that the unrolled loop really instantiates S distinct kernel sets
rather than a looping graph node.)*

Why it goes first: it needs **no new extraction, no target-model change, and no new dump**, only
extending `mtptrain`'s existing rollout (which already rolls K steps) from K=3 to K=15.

**Risk to weigh honestly:** for modified heads (the v2 rollout, v3), offline k=15 / k=7 estimates
were compared with served acceptance; the offline side was measured on the author's private
held-out data, and the comparison is not published [unpublished project report]. Whether
offline gains transfer to the server is the single biggest risk in the whole study (§6, R1).

### 3.3 Option 4 — trees / topk>1 at W4: **negative, with the arithmetic**

* Verify cost is steep here because it is MoE-bound. Distinct experts per layer per verify,
  measured over 197 470 / 135 534 calls: **T=4 → 28.4, T=16 → 69.4** (73 % and 50 % of the
  independent-routing prediction) [M, `specs/MOE_SMALLM_SPEC.md` §8.2], with
  `GEMM = 13.0 + 2.23·D µs` per layer × 48 layers [M, §8.3].
* Going W4 → a depth-3, 8-node tree costs the measured T=4→T=8 verify delta **+1.24 ms**, plus a
  draft loop that now proposes 1+2+4 = 7 rows instead of 3 (~+0.15 ms) [E]: step 11.95 → 13.34,
  **break-even +11.3 %**.
* But a depth-3 tree still caps acceptance at 4. **code-edit is already at 3.77/4 = 94 % of that
  cap, so its maximum possible gain is +6 % < 11.3 % break-even — strictly negative.** [E]
* prose-en has headroom: with top-2 coverage ≈ p+0.08 (0.731 → ~0.81 [E]) a full binary depth-3
  tree gives 1+0.81+0.656+0.531 = **3.00 (+13 %)** — barely above break-even, and at T=8 nodes
  (not the 15 a full binary tree needs; T must stay a power of two, since T=6/10 hit a verify-graph
  fallback costing +0.9/+2.8 ms [M]). Net ≈ a wash, for a large change to the draft loop
  (the fused topk1-chain kernel, `eagle_worker_v2.py:806-812`, is bypassed).
* **Conclusion: spend the token budget on depth, not width.** That is already what W16 does.

### 3.4 Option 5 — n-gram chain: already measured, already lost

`codex/ngram-chain-v0` is a **strict ancestor of `codex/perf-v1`** (empty diff) — it is merged and
live-capable via `serve-ngram-chain.sh`. It was measured on 2026-09-02 [M, `runs/ngram16-v3.jsonl`
vs `runs/nextn15-v3.jsonl`]: decode t/s **code-edit 314/347 vs 429/434 (−22 %)**, agent 214/200 vs
189/192 (+8 %), prose-en 110/111 vs 104/99 (+9 %). It loses on the workload it was designed for,
for reasons its own doc flags (all-or-nothing batch policy, `_flush_pending_commits` D2H stall).
No acceptance-length histogram was ever recorded. Not a path to higher acceptance.

---

## 4. What must change in SGLang

**Key finding: do NOT switch to `--speculative-algorithm EAGLE3`.** The fork's EAGLE3 runtime is
complete and model-agnostic, but it is the wrong door for this model:

* EAGLE3 **explicitly discards `--speculative-token-map`** and takes the hot vocabulary from a
  `d2t` buffer in the draft checkpoint (`eagle_worker_v2.py:284-295`, `llama_eagle3.py:339-345`).
  The served hot2_49152 map would be lost.
* EAGLE3 expects a `LlamaForCausalLMEagle3`-shaped draft (`fc: 3·h → h`, `midlayer.*`, own
  `lm_head` at `draft_vocab_size`). The MTP head shares the target's embed and lm_head via
  `set_embed_and_head` and has no `d2t` at all.
* `qwen4_exp.py` is **not** wired for aux capture: it has no `set_eagle3_layers_to_capture` and no
  `capture_aux_hidden_states`, and its `aux_hidden_states` accumulator at `:1691-1721` is dead code
  — the non-idle early return at `:1716-1717` shadows it.

**Recommended path: widen the existing MTP channel instead.** The target already returns
`(hidden, hc_hidden_states)` and publishes the second element at `qwen4_exp.py:1790`
(`output.hidden_states = hc_hidden_states`). Concatenate the taps into that same slot:

1. `qwen4_exp.py:1692-1709` — the `captured_last_layer_outputs=... if layer._is_layer_to_capture`
   plumbing **already exists in the loop**; it just never gets enabled. Set the flag on the two
   chosen layers at init, then at `:1714` emit `cat([aux_lo, aux_mid, hc_hidden_47])` = **30720**.
   No tuple-arity change, so `eager_runner.py:379-390` and `layers/cp/bcg.py:287-298` are untouched.
2. `configs/model_config.py:1096-1102` — `spec_hidden_size` derives from `hc_mult`; it must report
   3 × `hc_count` × `hidden` so every downstream buffer (incl. CUDA-graph capture) is sized right.
3. `qwen4_exp_mtp.py:227-301` `_fuse_residual_linear_shared` — branch on input width, exactly the
   trick `llama_eagle3.py:229-238` uses: **30720 ⇒ three `fc_hidden_{lo,mid,hi}` projections summed;
   10240 ⇒ the existing single path.** Step 0 of the chain gets 30720, steps 1..S−1 keep 10240
   (the head's own `own_hc`). `fc_hidden` becomes `fc_hidden_hi`; the two new ones are **zero-init**,
   so day-0 output is bit-identical to today.
4. `qwen4_exp_mtp.py:537-560` `_set_hc_logits_hidden_states` asserts width == `hc_count*hidden` —
   keep that assertion (the *draft's own* output stays 10240); relax only on the target side.
5. **Main integration risk:** `EagleDraftInput.hidden_states` is one buffer used both for the
   target→draft handoff (now 30720) and for the draft's own recursion (10240). Add a separate
   field for the wide handoff rather than padding; `eagle_utils.py:445-485`
   (`get_draft_input_from_target_hidden_dim`) already has an aux-aware branch to model this on.

**Which layers to tap.** The generic default `[2, n//2, n−3] = [2, 24, 45]` lands on **three
gated-delta-net layers** here; the 12 sparse-attention (QSA) layers are `[3, 7, ..., 47]`.
Recommend **layers 3 and 23 (both QSA) plus the existing layer-47 state** — QSA outputs carry the
long-range information the GDN layers do not, which is the plausible source of new signal.
Tap ids must be recorded in `config.json` so training and serving cannot disagree.

**Carry-over of existing draft optimisations:**

| knob | effect |
|---|---|
| `SGLANG_MTP_EMBED_TABLE` (+`_CHUNK`/`_CHECK`/`_RESERVE_GB`) | unaffected — token-side half only |
| `SGLANG_MTP_FC_GEMV` | needs two more `bf16_gemv` calls at step 0 |
| `SGLANG_MTP_ENTRY_FUSED` | its guard asserts `hidden.shape == (T, 4·h)` (`qwen4_exp_mtp.py:248-258`) — falls back at step 0 (fine, 1 of S forwards), unchanged for steps 1..S−1 |
| `SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS` | unaffected |
| `--speculative-token-map hot2_49152.pt` | unaffected — stays on the non-EAGLE3 branch (`eagle_worker_v2.py:332-353`) |
| `writeback/write_mtp.py` | 31 → 35 `mtp.*` tensors; loader name list must accept the new keys |

---

## 5. Data and training plan

**What is reusable, and what is not.** `~/mtp-dump` now holds only the self-generated half
(the corpus half was deleted); `~/mtp-dump-topk` holds the trainable rows with top-8 target
logits + full-vocab `lse` [M]. (Both are private; their sizes are not published.)
Both dump **exactly one tensor per token from one tap**: `hc_hidden` bf16 [T, 10240], the
post-layer-47 pre-mixer residual (`sglang-mtp-dump/.../qwen4_exp.py:1714-1725`) [M].
There is no low/mid-layer state anywhere on disk, and `MTPDumper.flush()` writes whole safetensors
files with `os.replace` — **no incremental augmentation is possible; a new prefill pass is required.**

* **Option 2 (PosS-lite) needs no new data at all.** `~/mtp-dump-topk` is sufficient.
* **Option 1 needs re-extraction.** Cost per row with three 10240-wide taps + top-8 + ids:
  `3 × 10240 × 2 + 50 + 12 = 61 502 B/row` [E]. A small phase-1 pilot extraction and a full
  re-extraction were planned (row counts and on-disk sizes describe the private dataset and are not
  published); both fit comfortably in the free disk space, and
  self-generation does **not** need re-running: it was greedy (T=0) and every dumped sequence's
`input_ids` are recoverable from the existing shards, so the identical token sequences can be
replayed as pure prefill, keyed by the same `doc_hash`/`split_key` so the held-out split carries over.

**Training recipe (v4 = option 1 + option 2, one run).** Start from `mtpft3`; keep v3's objective —
`0.5·hard CE + 0.5·top-8 distillation (temperature 1, true top-8 mass via lse) + 0.5·self-rollout CE`
[M, v3 recipe] — and change three things: (i) the entry consumes 3 taps, the two new
projections zero-init; (ii) rollout K raised 3 → 15 with `--rollout-mask accepted`; (iii) per-position
entry adapters. LR 1e-5, 1000 steps × 24 576 tokens, checkpoint selection by the **renewal evaluator**
on held-out docs with `--token-map hot2_49152`.

**Cost.** The target stays frozen (the NVFP4 target is used for extraction only). The v2
reference run's time and memory were measured on private data and are not published; the cost
of a v4 iteration (K=15 rollout, 16 passes/step instead of 4) has to be measured on its first
run.

**Offline evaluation before touching the server.** `mtptrain/renewal.py` / `scripts/eval_renewal.py`,
`--split eval --token-map hot2_49152.pt`, on freshly generated or held-out docs only. Its
agreement with the server was compared on private data; no result from that comparison is
published — see R1.

---

## 6. Risks, and the phased plan

**R1 (highest). Offline k=15 estimates may not transfer to the server for *modified* heads.** The
offline-versus-served comparison for v2/v3 was made on private data; no result or conclusion from it is published. *Mitigation:* re-calibrate the evaluator against the server at **k=7** for every new
head before any k=15 claim is believed; require a
4-repeat server A/B with the desktop GPU freed (per-repeat spread on W16 agent is 4.4-5.9 [M]).

**R2. Hyper-connections may already carry the multi-scale signal**, making the fusion gain much
smaller than EAGLE-3's +14 %. This is what phase 1 tests, and the reason phase 1 exists.

**R3. CUDA-graph handoff-buffer change** (§4 item 5) is the one place a mistake produces silent
garbage rather than a crash. *Mitigation:* the `prof/needle_test.py` gate, plus a bit-exact day-0
parity run — with the new projections zero-init the served output must be **identical** to mtpft3.

**R4. VRAM.** Today the draft load needs ~4.7 GB free after the target's 84.9 GB, and the desktop has
already blocked an A/B once [M]. Option 1 adds 26 MB (fine); option 3 (+1 layer) adds ~1.35 GB [E]
and is the reason it is ranked below the others.

**R5. Disk.** The full extraction plus the trainer's 31.3 GB `latest.pt` and 5.21 GB snapshots still
leaves ample free space. Do not run the all-rows variant.

### Phased plan

**Phase 0 — free, no GPU.** Decompose the positional decay: run the renewal evaluator on the
existing `~/mtp-dump-topk` held-out set to produce per-position acceptance `p_j` for j=1..15 (rather
than the aggregate), and cross it with the target's top-1 probability (already dumped as top-8 +
`lse`) at each rejection site. **Deliverable:** the real `p_j` curve, replacing the geometric fit of
§1, and the fraction of rejections that occur where the target itself is uncertain (`p1 < 0.5`) —
positions no draft can win. *Kill signal:* if `p_j` is already flat, options 1 and 2 both lose their
premise and the whole line is dead at the cost of a CPU run.

**Phase 0b — the cheapest confirming experiment.** PosS-lite pilot: no extraction, no
SGLang change. Extend `mtptrain` rollout to K=15, add per-position entry adapters, train 400 steps,
score with the renewal evaluator at k=3 **and k=7**.
*Gate:* **≥ +6 % at k=7 held-out under hot2_49152.** Proceed only if met.

**Phase 1 — the EAGLE-3 probe.** Add the two taps to the dump hook in the
`opus/mtp-dump` worktree; run the small pilot extraction; train **only** the two zero-init
projections and their norms (13.1 M params, whole body frozen — trainable on the pilot);
renewal-eval at k=3 and k=7.
*Gate:* **≥ +3 % at k=3 AND ≥ +6 % at k=7.** Below +2 % at k=3 ⇒ the extra layers carry no usable
signal for this architecture ⇒ **kill option 1** and keep only option 2.

**Phase 2 — full v4.** Full re-extraction, joint training (fusion + K=15
rollout + per-position adapters + v3 distillation), checkpoint selection on the renewal evaluator,
write-back to `...-NVFP4-mtpft4`.

**Phase 3 — serving.** The §4 changes; day-0 bit-parity check
(zero-init ⇒ identical output); `prof/needle_test.py`; fnbench 4-repeat A/B on both profiles with the
desktop GPU freed. *Adopt only on a server-measured win.*

**Go/no-go: GO on phases 0 and 0b now** (no code in the serving path, no data cost; GPU time to be
measured on the 0b run). **Phase 1 conditional** on 0b's gate or on 0's `p_j` curve showing a real decay.
Phases 2-3 only justify themselves if a phase-1/0b gate is cleared.
Expected end state if both gates pass: **W16 +10-20 % t/s (prose-en ~152→170-180, agent ~238→265-285,
code ~470→520-560), W4 +3-6 %** [E].

---

## Sources

Measured, in-repo: `~/tools/flash-next-bench/prof/OVERHEAD_REPORT.md`;
`~/tools/flash-next-bench/specs/MOE_SMALLM_SPEC.md` §8.2-8.3;
`~/tools/flash-next-bench/runs/{ngram16-v3,nextn15-v3}.jsonl`;
`~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/config.json`;
`sglang-rtxpro6000@codex/perf-v1` (`qwen4_exp.py`, `qwen4_exp_mtp.py`, `speculative/*`,
`models/llama_eagle3.py`); `sglang-mtp-dump@opus/mtp-dump` (`models/mtp_dump.py`).

Papers: EAGLE-3, *Scaling up Inference Acceleration of LLMs via Training-Time Test*,
arXiv:2503.01840 (τ tables and Table 2 ablation). HASS, *Learning Harmonized Representations for
Speculative Sampling*, arXiv:2408.15766 (ICLR 2025) — +8-20 % over EAGLE-2; both of its components
(objective distillation, context alignment) are already implemented here as v3 and v2.
PosS, *Position Specialist Generates Better Draft for Speculative Decoding*, arXiv:2506.03566 —
+9.2 % acceptance over EAGLE-3.
