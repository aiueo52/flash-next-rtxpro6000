# X2 — Exclusive (critical-path) time map, 2026-09-07 re-take

Date: 2026-09-07 02:08 (W4) / 02:35 (W16). Method identical to `EXCLUSIVE_TIME_MAP.md` (09-06 map):
`prof/exclusive_time.py` sweep line over kernel+memcpy+memset intervals, window = first…last `draft`
annotation ÷ 19 steps, family = (phase, label, grid), `%` against the **median** step wall, numbers are
the mean of `code-edit` / `prose-en` unless a cell shows both.

Traces (gitignored, on disk): `prof/traces/x20152-{w4,w16}-{code-edit,prose-en}/*.trace.json.gz`,
server logs `~/tools/sglang-rtxpro6000/logs/serve-x20152-{w4,w16}.log`. Build: fork `codex/perf-v1`
`14d4c4c985` + production venv (FlashInfer csrc a0/a3/g2-1/g2-2/g1-pack/p2), `serve-fast.sh` defaults of
2026-09-07 00:25: **P2 singleton prune τ=0.08 in-prologue, PDL on all profiles (W16 included), H1/H2, HC
fusions, GDN layouts, `hot2_49152` token map, `mtpft5` (v5 MTP head)**. `wa` was not profiled (budget).

Two measurement caveats, both stated up front:

* **Monitor powered off** during these traces (the 09-06 traces had the desktop compositor drawing).
  Absolute step walls are therefore a few % lower than a daytime run for reasons unrelated to any
  patch; compare kernel *shares* and per-kernel medians. The big weight-streaming kernels agree with
  09-06 to ≤ 1 % (target lm_head 402.5/398.0 W4, 404.9/403.1 W16 vs 397.8/403.3; GDN qkvz 29.0/29.8 vs
  29.2/29.3), so on this session the monitor effect on kernel time is below noise; the gap/idle rows
  are where it would show, and they are ~unchanged (76 µs W4 / 100 µs W16 vs 77 / 160).
* **Allocator.** The server ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (passed via
  `SERVER_ENV`, launcher untouched). Reason: since ~00:20 today every bare `serve-fast.sh` start fails
  at the KV-pool check on ~80 % of launches (6/6 of mine, 4/4 of Q1's first tries, the other agent's
  `q1diag`): the draft-load phase releases only 4.21 GB instead of 10.76 GB (bimodal; no launch in between was seen)
  because the native caching allocator leaves ~9 GB of inactive split blocks after the target load
  (diagnosed by the Q1 owner at 00:40, `bench/quality/run_arm.sh` comment: reserved 86.9 vs allocated
  77.7 GB). The setting is allocator-only; CUDA-graph kernels are unaffected. **Ops action for the
  owner: put this in `serve-fast.sh`/`serve-local.sh`, otherwise production starts are a coin flip.**
  Q1's arms run under the same setting, so these traces match what Q1 measured.

Everything below is **measured** unless a line says *estimated*.

---

## 0. TL;DR

* **Median step: W4 8 919 µs (09-06: 9 966, −10.5 %), W16 16 083 µs (09-06: 17 421, −7.7 %).**
  Σ exclusive fell −1 022 µs (W4) / −1 778 µs (W16); the step is still a serial chain (one kernel
  resident 75–81 % of the time, idle 0.08–0.10 ms).
* **P2 is the whole W4 story and 83 % of the W16 one.** GEMM1+GEMM2 exclusive −836 µs (W4) /
  −1 484 µs (W16); implied distinct experts per call from G1's fit: **W4 29.2 → 18.8, W16 75.9 → 52.3**.
  The prologue paid +1.2 µs/call for the in-kernel mask (11.28 → 12.46 µs) and its exclusive time rose
  +27 µs at W16, as `P2_LOG` §4 said it would.
* **PDL on W16 did what §5.3 of the 09-06 map predicted, and it landed on attention.** Both
  `kernel_mha` families went from 189 + 217 µs exclusive to **15.8 + 9.9 µs** (now 93–96 % hidden;
  medians unchanged at 16.8–16.9 µs). W16's ≥ 2-kernel share rose 11.4 → 16.3 %. The 09-06 lever
  "draft `kernel_mha` 18-CTA grid, −145 µs" is therefore **gone** — do not pursue it.
* **v5 head: no per-kernel signature.** 14 draft forwards/step, 220/228 µs per forward (09-06: 234),
  every draft family within 1 % of its 09-06 median. v5 changes acceptance (t/s), not the step.
* **The MoE GEMMs are now 30 % (W16) / 23.5 % (W4) of the step and the only ≥ 3 % lever left is more
  of the same: raising τ.** Offline on the recorded route census, τ 0.08 → 0.10 buys another
  **−0.85 ms (−5.3 %) at W16 / −0.51 ms (−5.7 %) at W4** — but it drops 10.6–16 % of routing mass at
  W16 and up to 38 % at W4 (code-edit), against 6.5–14 % at the τ=0.08 that passed the gates. It is a
  quality experiment, not an engineering one (§4, lever 1).
* Everything else is ≤ 1.7 % per item (§4). The dense-GEMV block (lm_heads, GDN projections) is at
  88–98 % of the read roof, unchanged; launch-bound glue over 20 µs exclusive is 4 families / 367 µs
  (W4) and 14 families / 697 µs (W16), of which the HC pair (`_hc_branch_stats`, `hc_combine_gate`)
  is 267–312 µs and the rest is the W16 draft loop's per-forward tail.

---

## 1. The partition of the step

| | W4 (S=3, T=4) | W16 (S=15, T=16) |
|---|--:|--:|
| median step wall (code-edit / prose-en) | **8 994 / 8 843** | **16 371 / 15 796** |
| Σ exclusive | 7 136 / 6 995 (79.2 %) | 13 072 / 12 983 (81.0 %) |
| shared (≥ 2 kernels, charged to nobody) | 1 785 / 1 764 (19.9 %) | 2 724 / 2 551 (16.4 %) |
| gap — no kernel (median step) | **75 / 76 (0.85 %)** | **91 / 109 (0.62 %)** |
| tiny (< 2 µs) kernel alone | 412 / 431 (4.7 %) | 599 / 668 (3.9 %) |
| kernels/step | 1 516 / 1 525 | 2 109 / 2 104 |
| distinct families | 208 / 214 | 210 / 202 |

Concurrency (code-edit): W4 0: 2.2 % · **1: 78.4 %** · 2: 17.9 % · 3: 1.5 %; W16 0: 1.4 % ·
**1: 80.7 %** · 2: 16.3 % · 3: 1.6 % (09-06: W16 86.1 / 11.4 — the PDL-on-W16 change).

### 1.1 By phase (exclusive µs/step, mean of workloads)

| phase | W4 | % | W16 | % | 09-06 W4 / W16 |
|---|--:|--:|--:|--:|--:|
| `step[TARGET_VERIFY bs=1]` | 6 039 | 67.7 | 9 264 | 57.6 | 7 019 / 10 850 |
| `draft` (2 / 14 forwards) | 480 | 5.4 | 3 134 | 19.5 | 498 / 3 280 |
| `draft_extend` | 171 | 1.9 | 122 | 0.8 | 195 / 173 |
| `[run_batch]` (sampling graph + async GDN state) | 74 | 0.8 | 106 | 0.7 | 74 / 100 |

### 1.2 By functional block (exclusive µs/step)

| block | W4 | % | W16 | % | 09-06 W4 / W16 |
|---|--:|--:|--:|--:|--:|
| MoE grouped GEMM (verify GEMM1+GEMM2) | **2 093** | **23.5** | **4 827** | **30.0** | 3 085 / 6 858 |
| dense FP8 gemv (`_w8a16_gemv*`, all phases) | 2 385 | 26.7 | 3 656 | 22.7 | 2 374 / 3 547 |
| HC chain (`_hc_up/_down/_branch_stats`, `hc_combine_gate`) | 1 082 | 12.1 | 1 649 | 10.3 | 1 141 / 1 622 |
| MoE routing prologue + doActivation + glue (verify) | 314 | 3.5 | 413 | 2.6 | 360 / 542 |
| attention + QSA indexer (`kernel_mha`, `kernel_kernel`, `_compact_kv`, `fast_topk`, `_expand_qsa`) | 213 | 2.4 | 470 | 2.9 | 169 / 698 |
| GDN / mamba (`wy_output_only`, `conv1d_update_chain`, `bf16state_mtp`) | 237 | 2.7 | 305 | 1.9 | 253 / 308 |
| draft MoE (GEMM1+GEMM2+prologue+act, draft phase) | 93 | 1.0 | 550 | 3.4 | 96 / 574 |

---

## 2. Per-family exclusive tables

Columns: calls/step · median µs · **exclusive µs/step** · % of median wall · excl/raw. Mean of both
workloads; a cell "a / b" is code-edit / prose-en where they differ materially.

### 2.1 W4 — verify (6 039 µs)

| family | grid | n | med µs | **EXCL** | % | e/r |
|---|---|--:|--:|--:|--:|--:|
| `cutlass_moe_grouped_gemm1` | [1,188,1] | 45.5 | 30.7 / 27.0 | **1 327** | 14.9 | 99 % |
| `_w8a16_gemv` GDN in_proj | [512,1,1] | 34.1 | 29.0 | **893** | 10.0 | 88 % |
| `cutlass_moe_grouped_gemm2` | [1,188,1] | 45.5 | 17.6 / 16.8 | **766** | 8.6 | 95 % |
| `_w8a16_gemv` GDN out_proj | [40,4,1] | 34.1 | 14.1 | 470 | 5.3 | 97 % |
| `_hc_up_kernel` | [160,1,1] | 91.9 | 4.92 | 422 | 4.7 | 91 % |
| `_w8a16_gemv` **target lm_head** | [1940,1,1] | 0.9 | 400.2 | 387 | 4.3 | 100 % |
| `_hc_down_kernel` | [20,5,1] | 91.9 | 4.16 | 361 | 4.0 | 89 % |
| `trtllm::fusedBuildExpertMapsSort…` (prologue) | [56,1,1] | 45.5 | 11.9 / 10.5 | 186 | 2.1 | **38 %** |
| `_hc_branch_stats_kernel` | [16,1,1] | 91.9 | 1.97 | 153 | 1.7 | 80 % |
| `_w8a16_gemv` attn qkv | [80,6,1] | 11.4 | 13.0 | 145 | 1.6 | 98 % |
| `gdn_decode_bf16_wy_output_only` | [1,48,1] | 34.1 | 4.24 | 139 | 1.6 | 96 % |
| `trtllm::doActivationKernel` | [40,1,1] | 45.5 | 3.50 | 114 | 1.3 | 72 % |
| `hc_combine_gate_kernel` | [4,32,1] | 90.9 | 1.60 | 114 | 1.3 | 75 % |
| `_w8a16_gemv` (in_proj_ba) | [3,1,1] | 34.1 | 5.55 | 88 | 1.0 | **43 %** |
| `_causal_conv1d_update_chain_kernel` | [1,160,1] | 34.1 | 2.32 | 79 | 0.9 | 100 % |
| `kernel_kernel` (QSA indexer) | [4,1024,1] | 11.4 | 5.4 | 60 | 0.7 | 86 % |
| `fast_topk_kernel` (QSA top-k) | [4,1,1] | 11.4 | **6.18 / 1.31** | **80 / 13** | 0.9 / 0.1 | 97 % |
| `_compact_kv` | [4,2,129] | 11.4 | 4.86 / 2.72 | 53 / 29 | 0.5 | 95 % |
| `kernel_mha (trtllm-gen)` | [9,2,4] | 11.4 | 16.5 | 8 | 0.1 | **4 %** |
| draft: `_w8a16_gemv` draft lm_head | [1536,1,1] | 1.9 | 81.2 | 163 | 1.8 | 100 % |
| draft_extend: draft lm_head / GEMM1 / GEMM2 | | 0.9 | 83.9 / 34.7 / 30.2 | 74 / 28 / 25 | 1.4 | 86–99 % |

### 2.2 W16 — verify (9 264 µs) and draft (3 134 µs = 14 forwards)

| family (phase) | grid | n | med µs | **EXCL** | % | e/r |
|---|---|--:|--:|--:|--:|--:|
| `cutlass_moe_grouped_gemm1` (verify) | [1,188,1] | 45.5 | 61.6 / 65.5 | **3 093** | 19.2 | 99 % |
| `cutlass_moe_grouped_gemm2` (verify) | [1,188,1] | 45.5 | 34.6 / 37.4 | **1 733** | 10.8 | 98 % |
| `_w8a16_gemv` **draft lm_head** (draft) | [1536,1,1] | 13.4 | 81.2 | **1 124** | 7.0 | 98 % |
| `_w8a16_gemv` GDN in_proj (verify) | [256,1,1] | 34.1 | 29.8 | 893 | 5.6 | 82 % |
| `_w8a16_gemv` GDN out_proj (verify) | [40,4,1] | 34.1 | 14.4 | 503 | 3.1 | 98 % |
| `_hc_up_kernel` (verify) | [160,1,1] | 91.9 | 5.09 | 454 | 2.8 | 92 % |
| `_w8a16_gemv` **target lm_head** (verify) | [1940,1,1] | 0.9 | 404.0 | 428 | 2.7 | 100 % |
| `_hc_down_kernel` (verify) | [20,5,1] | 91.9 | 4.27 | 355 | 2.2 | 88 % |
| prologue `fusedBuildExpertMapsSort…` (verify) | [176,1,1] | 45.5 | 12.46 | 276 | 1.7 | **47 %** |
| `_w8a16_gemv` MTP in_proj (draft) | [416,1,1] | 13.4 | 23.2 | 247 | 1.5 | 75 % |
| `cutlass_moe_grouped_gemm1` (draft) | [1,188,1] | 13.4 | 17.7 | 238 | 1.5 | 97 % |
| `cutlass_moe_grouped_gemm2` (draft) | [1,188,1] | 13.4 | 16.9 | 209 | 1.3 | 92 % |
| `_hc_up_kernel` (draft) | [160,1,1] | 40.2 | 4.85 | 184 | 1.1 | 90 % |
| `_w8a16_gemv` MTP out_proj (draft) | [160,6,1] | 13.4 | 12.3 | 173 | 1.1 | 96 % |
| `gdn_decode_bf16_wy_output_only` (verify) | [1,48,1] | 34.1 | 4.48 | 167 | 1.0 | 96 % |
| `_hc_branch_stats_kernel` (verify) | [64,1,1] | 91.9 | 2.03 | 164 | 1.0 | 71 % |
| `_hc_down_kernel` (draft) | [20,5,1] | 40.2 | 4.19 | 157 | 1.0 | 88 % |
| `hc_combine_gate_kernel` (verify) | [16,32,1] | 90.9 | 1.73 | 148 | 0.9 | 90 % |
| `cutlass80_wmma…32x32_64x1` QSA q/k (draft) | [8,10,20] | 13.4 | 11.0 | 143 | 0.9 | 97 % |
| `_w8a16_gemv` attn qkv (verify) | [80,6,1] | 11.4 | 12.8 | 143 | 0.9 | 99 % |
| `kernel_kernel` QSA indexer (verify) | [16,1024,1] | 11.4 | 10.3 | 132 | 0.8 | 100 % |
| `trtllm::doActivationKernel` (verify) | [160,1,1] | 45.5 | 3.82 | 126 | 0.8 | 71 % |
| `_causal_conv1d_update_chain_kernel` (verify) | [1,160,4] | 34.1 | 3.26 | 119 | 0.7 | 100 % |
| `_compact_kv` (verify) | [16,2,129] | 11.4 | **13.2 / 4.7** | **164 / 51** | 1.0 / 0.3 | 97 % |
| `_w8a16_gemv` in_proj_ba (verify) | [3,1,1] | 34.1 | 7.2 | 93 | 0.6 | **32 %** |
| `_hc_branch_stats_kernel` (draft) | [4,1,1] | 40.2 | 1.71 | 71 | 0.4 | 79 % |
| prologue (draft) | [26,1,1] | 13.4 | 12.4 | 69 | 0.4 | 40 % |
| `gdn_decode_bf16state_mtp` (run_batch, async) | [384,1,1] | 37 | 9–14 | 50 | 0.3 | **9 %** |
| `fast_topk_kernel` (verify) | [16,1,1] | 11.4 | 6.21 / 1.15 | 71 / 11 | 0.4 / 0.1 | 97 % |
| `kernel_mha` (verify) / (draft) | [5,2,16] / [9,2,1] | 11.4 / 13.4 | 16.9 / 16.8 | **16 / 10** | 0.2 | **8 % / 4 %** |
| draft_extend: draft lm_head / GEMM1 / GEMM2 | | 0.9 | 91.8 / 48 / 40 | 49 / 39 / 21 | 0.7 | 52–72 % |

W16 draft loop per forward: **220 / 228 µs** (09-06: 234), of which lm_head 81.2 (37 %), MoE GEMMs
34.6, in_proj 23.2, out_proj 12.3, QSA q/k 11.0, HC 30, prologue+act 8, and a ~13-launch tail of
26 µs (`_fused_qk_rmsnorm_rope_gate` 2.6, `memcpy32_post` ×3, `direct_copy` 2.7, `hc_combine_gate` ×2,
`RMSNorm` 2.4, `indexSelect` 1.9, topk1 partial+finalize 3.9, `add` 2.2, `splitKreduce` 1.6).

---

## 3. What moved vs the 09-06 map (exclusive µs/step, mean of workloads)

| family | W4 09-06 → now | Δ | W16 09-06 → now | Δ | attributed to |
|---|--:|--:|--:|--:|---|
| verify GEMM1 | 1 860 → 1 327 | **−533** | 4 132 → 3 093 | **−1 039** | **P2** (D 29.2→18.8 / 75.9→52.3 implied) |
| verify GEMM2 | 1 069 → 766 | **−303** | 2 178 → 1 733 | **−445** | **P2** |
| draft `kernel_mha` [9,2,1] | – | – | 217 → 10 | **−207** | **PDL on W16** (median unchanged; now 96 % hidden) |
| verify `kernel_mha` | ~40 → 8 | ≈ −30 | 189 → 16 | **−174** | **PDL on W16** (W4 already had it) |
| `doActivationKernel` | 115 → 114 | 0 | 169 → 126 | −43 | PDL (more of it under GEMM2) |
| prologue `fusedBuild…` | 207 → 186 | −21 | 249 → 276 | **+27** | P2 in-kernel mask +1.2 µs/call (11.28→12.46), 09-06 hidden fraction kept (47 %) |
| draft_extend GEMM1/GEMM2 | 70 → 53 | −17 | 95 → 60 | −35 | P2 (draft_extend batch is T rows too) |
| GDN out_proj [40,4,1] | 520 → 470 | −50 | 494 → 503 | +9 | session (median 14.1 both) |
| target lm_head | 377 → 387 | +10 | 393 → 428 | +36 | less overlap with the sampling tail (median 400/404, unchanged) |
| draft lm_head | 158 → 163 | +5 | 1 068 → 1 124 | +57 | same (median 81.2 vs 80.9) |
| `_hc_branch_stats` | 147 → 153 | +6 | 185 → 164 | −21 | PDL |
| **Σ exclusive** | 7 786 → 6 764 | **−1 022** | 14 404 → 12 626 | **−1 778** | P2 −836 / −1 484; PDL-W16 ≈ −0 / −420; rest ±session |
| **median wall** | 9 966 → 8 919 | **−1 047 (−10.5 %)** | 17 421 → 16 083 | **−1 338 (−7.7 %)** | |

Nothing is attributable to the v5 head: every draft-phase family is within 1 % of its 09-06 median and
count (14 forwards, 1.0 call/forward). Not moved and still at roof: GDN in_proj (29.0–29.8 µs, 88 %),
both lm_heads (96–98 %), the HC pair (`_hc_up` 4.9–5.1, `_hc_down` 4.2–4.3 µs).

---

## 4. Ranked lever list

Costs: 1 distinct expert per MoE call = 1.036 µs (GEMM1, `G1_LOG` §3) + 0.500 µs (GEMM2, fitted here
from the 09-06 W4/W16 medians: `GEMM2 = 9.46 + 0.500·D`) = **1.536 µs × 45.5 calls = 69.9 µs/step per
unit of D**. `%` against 8 919 (W4) / 16 083 (W16). "census" = offline replay of the P1 rule on the
recorded pre-prune routes (`runs/census-{w4,w16}-{code-edit,prose-en}.npz`, 12.7–15.7 k T=16/T=4 calls
each; the sweep script was a throwaway and is not published, but it only applies `prof/p1_moe.py`'s rule).

| # | lever | mechanism | targets (measured) | W16 | W4 | effort | quality risk | ≤ 1 GPU-h confirm / kill |
|--:|---|---|---|--:|--:|---|---|---|
| **1** | **τ sweep 0.08 → 0.10** (singleton prune, config only) | more low-weight singleton routes dropped → fewer distinct experts per call. Routing weights are normalised over k=10 (mean by within-row rank 0.201, 0.142, 0.115, **0.100, 0.089, 0.081**, 0.075…): 0.08 cuts the bottom ~4 ranks, 0.10 the bottom ~7, hence the cliff | GEMM1+GEMM2 3 093+1 733 (W16), 1 327+766 (W4). census ΔD at 0.10 vs 0.08: W16 **+17.3 / +7.0** (code / prose), W4 **+10.8 / +3.8** | mean ΔD 12.1 × 69.9 = **−848 µs (−5.3 %)**, range −489…−1 209 | ΔD 7.3 × 69.9 = **−510 µs (−5.7 %)**, range −266…−755 | 0 h (`PRUNE_TAU=0.10`) | **high**: dropped routing mass W16 6.5→16.3 % (code) / 6.8→10.6 % (prose); W4 14→**38 %** (code) / 14→22 % (prose). τ=0.08 passed the gates at ≤ 14 %; W4 at 0.10 is well past that, W16 is inside it | `runs/p2/quality.sh` at τ=0.10, **W16 first** (needle + acceptance, ~15 min) plus one `MODE=prof` pass: GEMM1 median must fall ≥ 8 µs/call. Kill on a needle failure or an acceptance drop; then try 0.09 (not simulated here — bisect between the two). |
| 2 | 2-row experts (τ₂ on both routes of a count-2 expert, in the same prologue) | P2's histogram already has the count; add `count==2 && both w<τ₂ && both rank≥1` | census at (τ=0.08, τ₂=0.08): W16 ΔD **+2.9 / +3.9**, W4 **+0.4 / +1.1**; at τ₂=0.05 ΔD ≤ 0.3 (nothing) | ΔD 3.4 × 69.9 = **−238 µs (−1.5 %)** | ΔD 0.75 × 69.9 = **−52 µs (−0.6 %)** | 1 day csrc (P2 pattern) + gates | medium: mass +2.3 / +3.0 pts on top of τ=0.08 | offline only until lever 1 is settled: it is dominated (same quality budget buys 3× more ΔD via τ). Build only if τ=0.10 fails the gate and this passes it. |
| 3 | fold `_hc_branch_stats` into `_hc_up`'s epilogue (09-06 item 4, unchanged) | independent stage → launch deletion, not a barrier fusion | 1.97–2.03 µs × 91.9, 71–80 % exclusive | **−164 µs (−1.0 %)** | **−153 µs (−1.7 %)** | 1–2 days | low (bit-exact achievable, `HC_LAYER_APPLY_FUSED` precedent) | unit `torch.equal` + one `MODE=prof` pass; kill if `_hc_up` median grows > 1 µs |
| 4 | fold `hc_combine_gate` (1.60–1.73 µs × 91) into its producer | only if it is an independent stage; if it needs the full reduction of `_hc_up` it is a barrier fusion → NO-GO (`MEGAKERNEL` §3.2) | 114 (W4) / 148 (W16), 75–90 % exclusive | **−148 µs (−0.9 %)** | **−114 µs (−1.3 %)** | 0.5 day to read the dependency, 1–2 days if foldable | low | read `hc_combine_gate_kernel` inputs first (CPU); confirm with one `MODE=prof` pass |
| 5 | A6 MXFP4 expert weights (09-06 item 1) | 4.25 vs 4.5 bits, flat 5.6 % of pure weight traffic | GEMMs 4 827 (W16) / 2 093 (W4) | 0.056 × 4 827 = **−270 µs (−1.7 %)** (was −372 before P2) | 0.056 × 2 093 = **−117 µs (−1.3 %)** | high (requantise + sm120 path) | medium (coarser scales) | none under 1 h; the prize shrank by a third with P2 and shrinks again with lever 1 |
| 6 | `_compact_kv` long-context anomaly (**new**) | QSA KV compaction is 13.2 µs/call on code-edit vs 4.7 on prose-en (W4: 4.9 vs 2.7); grid [T,2,129], 97 % exclusive. Un-audited kernel body (`ASTRA_REVIEW` flag) | 164 / 51 (W16), 53 / 29 (W4) | if code-edit reached prose-en's rate: **−113 µs (−0.7 %) on code-edit** | −24 µs (−0.3 %) | 0.5–1 day to characterise | none | CPU: read what it moves per page; GPU: 20 min microbench of the kernel at 2 context lengths |
| 7 | GDN out_proj re-tune (09-06 item 5, unchanged) | 40-tile shape at 1 198 GB/s (74 % of roof) via split-K 5–10 | 14.1–14.4 µs × 34.1, 97 % exclusive | **−123 µs (−0.8 %)** | **−130 µs (−1.5 %)** | medium (sweep + table entry) | low; watch `NORM_INTO_GEMV` | 30-min Triton config sweep on the isolated shape |
| 8 | `TOKEN_MAP` 49 152 → 32 768 (09-06 item 3) | draft lm_head at 96 % of roof scales with vocab: 81.2 → ~54 µs × 14.3 calls | 1 124 + 49 (W16), 163 + 74 (W4) | **−390 µs (−2.4 %)** step | −76 µs (−0.9 %) | config only | **not measured on public data**: production has served 49 152 since 09-04, and the only 32 768-vs-49 152 comparison was an offline evaluation on private data with no published result; untested with v5 → only an end-to-end t/s A/B decides | 20-min fnbench A/B on t/s, not step time; kill unless t/s rises |
| 9 | W16 draft-loop glue tail | 11 launch-bound families ≥ 20 µs in `draft` = 365 µs (2.3 %); the topk1 chain (`partial_argmax` + `finalize` + `indexSelect` + `add` + `RMSNorm`, 5 launches, 133 µs) is the only contiguous run | 26 µs/forward of tail | one fused topk/epilogue kernel: *estimated* **−80 µs (−0.5 %)** | 0 | 1–2 days | low | one `MODE=prof` pass |
| 10 | `fast_topk` (QSA top-k, verify) on long context | 6.2 µs on a 4–16-CTA grid on code-edit vs 1.2–1.3 µs on prose-en | 71–80 µs code-edit only | ≤ −40 µs (−0.25 %) | ≤ −40 µs (−0.4 %) | 0.5 day | none | include in lever 6's microbench |
| — | verify lm_head | 400–405 µs, flat in T (W4 = W16), 1 577 GB/s = 98 % of roof; hot-vocab target head changes the output distribution (N1 class, rejected) | 387 / 428 | **0** | **0** | — | — | none — done |
| — | sampling / accept glue | `[run_batch]` 74 / 106 µs total; `VerifyTreeGreedy` 4 µs, `build_tree_efficient` 3.6 / 21.9 µs (one launch/step), draft topk1 chain 7 / 50 µs; total ≈ 0.7 % of the step | ≈ 110–130 µs | ≤ −30 µs | ≤ −10 µs | — | — | not a lever |
| — | routing prologue | 276 / 186 µs, 47 / 38 % hidden; the +1.2 µs/call P2 mask runs in every block by the A3 design (`P2_LOG` §1) and G2's structural floor is 10.34 µs | ≤ 2.1 µs × 45.5 | ≤ −96 µs (−0.6 %) | ≤ −70 µs | unknown mechanism | — | none |

**Is ≥ 3 % still on the table?** Yes, but only through lever 1, which is a quality decision rather than
engineering: τ=0.10 is worth −5.3 % (W16) / −5.7 % (W4) *if* the quality/needle/acceptance gates pass
at 2–3× the routing mass that τ=0.08 drops. Nothing else clears 2 %. The safe kernel-side items
(levers 3, 4, 6, 7 and A6) sum to *estimated* −818 µs (−5.1 %) at W16 and −538 µs (−6.0 %) at W4 for
about 2–3 weeks of work, in ≤ 1.7 % pieces. The launch-floor tax is unchanged in nature (0.42 / 0.63 ms
of sub-2 µs kernels alone) and its harvesting rules are the ones the 09-06 map gave.

**Not examined (completeness).** The `wa` profile (skipped on budget); `agent-loop`/`prose-ja`
workloads; bs > 1 and prefill; the shared (≥ 2 concurrent) 1.8–2.7 ms was not decomposed by pair; the
QSA kernel bodies (`kernel_kernel`, `_compact_kv`, `fast_topk`) were not bandwidth-characterised; the τ
and 2-row figures are offline replays on the P1-era census whose pre-prune D (80 / 69 W16) is higher
than this session's implied D — the ΔD are relative and the only real test is the gate; one trace pair
per profile, monitor off, `expandable_segments` allocator (see the caveats at the top).
