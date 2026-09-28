# N1 — Dense NVFP4 for the Qwen3.8-Flash-Next serving path

Worktrees: `$HOME/tools/sglang-n1` (`opus/dense-nvfp4` from `codex/perf-v1` @ ef31f26346),
`$HOME/tools/mtp-train-n1` (`opus/n1-eval` from `opus/v4-fullhead` @ 66a42b0).
The main worktree, `serve-fast.sh`, the venv and the model directories are untouched.

---

## 0. The brief's premise is wrong, and it moves the bar

The task states the dense FP8 GEMVs stream weights "at 78–85 % of the 1.6 TB/s read roof"
and concludes the 4-bit kernel needs **≥ 40 % of roofline to break even, ≥ 60 % to ship**.
`EXCLUSIVE_TIME_MAP.md` §0 and §5 measure something different for the two shapes N1
actually targets:

| shape | bytes moved | measured | % of the 1615 GB/s read roof |
|---|--:|--:|--:|
| draft lm_head 49152×2560 (`_w8a16_gemv[1536,1,1]`) | 125.8 MB | 80.9 µs | **96 %** |
| target lm_head 248320×2560 (`_w8a16_gemv[1940,1,1]`) | 635.7 MB | 403 µs | **98 %** |
| GDN in_proj qkvz 16384×2560 | 40 MB | 29.3 µs | 88 % |
| GDN out_proj 2560×6144 | 15 MB | 13.1–13.7 µs | 74 % |

78–85 % is roughly the *family average* including the out_proj; it is not the lm_head,
and the lm_head is where the bytes are.

Recomputing the bar. NVFP4 moves **0.5 B/element of codes + 1/16 B of E4M3 block scale
= 0.5625 B**, against 1.0 B for FP8 — a 1.778× byte advantage, not 2×. Time is
bytes ÷ achieved bandwidth, so against a 96–98 % FP8 baseline:

* **break-even** needs 0.5625 × 0.97 = **55 % of roof** (not 40 %)
* a **20 % kernel win** needs **69 % of roof**
* the full 1.778× needs the 4-bit kernel to match FP8's 96 % — i.e. to be *as*
  bandwidth-perfect while doing ~7 extra ALU ops per weight element

Worked example with an **assumed 60 % of roofline** (a round number between the 55 % break-even
and the 69 % ship bar, chosen only to illustrate the arithmetic; it is not a measurement):
0.5625/0.60 × 0.97 = **0.91**, a 9 % kernel win.

What 9 % is worth, from the same map (W16 / W4 exclusive µs of a 17.4 / 9.94 ms step):

| | W16 µs | at −9 % | W16 step | W4 µs | W4 step |
|---|--:|--:|--:|--:|--:|
| draft lm_head | 1068 | −97 | −0.56 % | 158 | −0.14 % |
| target lm_head | 393 | −36 | −0.20 % | 377 | −0.34 % |
| **both** | | | **−0.76 %** | | **−0.49 %** |

So the brief's "up to ~8–10 % of the step" is the *ceiling at 96 % roof efficiency
across all four families*, not the expected value. At the assumed 60 %
the whole N1 line is worth under 1 %. **The isolated kernel bandwidth is
therefore the decision, and it is measured before any server work.**

---

## 1. Kernel — `w4a16_nvfp4_gemv` (Stage A step 1)

`python/sglang/srt/layers/quantization/w4a16_nvfp4_gemv.py` (new, ~330 lines).

Structure is deliberately `w8a16_gemv`'s: one CTA per (n block, k split), fp32
accumulator, split-K partials reduced **in the same launch** by whoever observes the
last counter increment (so it is CUDA-graph-replay safe with no memset), split-store
epilogue, optional PDL, `USE_DOT=False` broadcast path at M == 1. It **shares
w8a16_gemv's split-K scratch** on purpose: the invariant that matters is "no two
split-K launches overlap", which the decode path already guarantees by issuing every
linear on one stream; a second buffer would not make an overlap safe, only cost memory.

What is new is the weight A-load:

* **Unpack** — a `[BLOCK_N, BLOCK_K/2]` uint8 tile is split into low/high nibbles and
  `tl.join`-ed back to `[BLOCK_N, BLOCK_K]`. A pure register shuffle; the join's
  trailing axis order is exactly (even k, odd k), which is the checkpoint's nibble
  order (`modelopt_quant.py:810-828`), so the reshape is the identity permutation.
* **Dequant by bit trick** (the in-house engine `nvfp4_linear.py:126-159`) — the code's mantissa
  bit and 2 exponent bits are shifted into fp16 bits [11:9] and the sign into [15]; a
  bitcast then yields exactly `e2m1_value × 2^-14`, with fp16's *subnormal* encoding
  making the `e == 0` codes land on the same 2^-14 factor. Exact for all 16 codes, no
  table, no branch. The 2^14 is folded into the **global** scale, applied once per
  output outside the k loop.
* **Exactness** — e2m1 carries 2 significand bits and e4m3 4, so their product needs 6;
  bf16 has 8. The dequantised weight is therefore *exact in bf16*, and `tl.dot` sees the
  same class of operand the FP8 path feeds it. Cost: ~7 ALU ops per weight element
  against FP8's single hardware convert — this is the thing that decides whether the
  bandwidth bar above is cleared.

Triton 3.7.1 has no `tl.float4e2m1`, so the bit trick is not avoidable by a native cast.

## 2. Quantiser, and a bug the consistency check caught

`quantize_nvfp4()` / `dequantize_nvfp4()` in the same file: ModelOpt convention,
`gscale = amax/(448·6)`, per-16-block `amax_blk/6` in E4M3, codes by round-to-nearest-**even**.
Deterministic and chunked over rows.

Checking it against the offline emulator that the acceptance gate uses
(`parity/nvfp4_emulate.py`, a development tool not published in `mtp-train/`) found **two independent divergences**, both fixed, on
`/tmp/.../qcheck.py` over a random [512, 2560] bf16 weight:

| | mismatched elements |
|---|--:|
| first attempt | 14 715 / 1 310 720 (**1.12 %**) |
| after matching the global-scale *expression* | 767 (0.06 %) |
| after ties-to-even on the E2M1 rounding | **0** |

1. **Global-scale arithmetic.** `amax_blk/6/gscale` and `amax_blk/6·(1/gscale)` differ by
   one fp32 ulp. E4M3's 4-bit significand is coarse enough that ~1 % of block scales land
   *exactly* on a tie point (e.g. 168.0, midway between 160 and 176), where that ulp flips
   the rounding and the whole 16-element block is then scaled 10 % differently. Both sides
   now use the fork's own form (`nvfp4_online.py:296` hands flashinfer `1.0/weight_scale_2`).
2. **E2M1 tie-breaking.** `parity/nvfp4_emulate._round_e2m1` used a plain
   `torch.bucketize`, which sends every exact tie *down*; hardware and the packer round
   half to even. 0.06 % of elements, but each one a full code step — a 1.33–1.5× error on
   that weight. Fixed in the emulator (this also very slightly changes `--quantize-experts`
   results; that flag is a diagnostic, not a published baseline).

Neither would have raised an error anywhere. Both would have quietly decoupled the
offline acceptance number from the served one, which is the entire premise of the gate.

## 3. Offline acceptance harness (Stage A step 3)

`scripts/eval_renewal.py --quantize-lm-head {fp8,nvfp4,fp8+nvfp4}`, applied **after**
the token map is loaded, because with a token map the server materialises only the hot
rows and an NVFP4 per-tensor global scale is an amax over *those* rows.

**The brief's baseline is not the right one.** `serve-fast.sh` passes
`--qwen4-exp-dense-fp8 ...,lm_head,...`, and `eagle_worker_v2.init_lm_head:334-361`
builds the draft head as a **row slice of the target's already-FP8 head**, sharing the
target's `Fp8LinearMethod`. So the shipped draft head is FP8, not BF16 — while the
evaluator has never modelled head quantization at all (a real gap in its calibration
against the server). The existing v3 offline baselines are therefore **BF16-head**
numbers, and the stage-A gate is `fp8 → nvfp4`, not `bf16 → nvfp4`.
Hence three arms, and `fp8+nvfp4` because the cheap implementation quantises the
*already-FP8* rows, which compounds the error.

That last point is a hard constraint, not a preference. `eagle_worker_v2.py:334-352`
reaches the draft head only *after* `Fp8LinearMethod.process_weights_after_loading` has
replaced the target's weight in place — the branch literally tests
`head.dtype == torch.float8_e4m3fn` and slices `head.t()[hot_token_id]`. So the BF16
rows are gone by then, and **the cheap implementation is forced into the worse of the
two quality arms.** Reaching the `nvfp4` arm needs either a BF16 copy of the head held
through startup (1.27 GB) or a second read of the checkpoint — a real cost to weigh
against the ~2 points of random-probe argmax agreement it buys (§4).

## 4. Weight- and logit-error on the real head (CPU, no GPU)

Real `lm_head.weight` from `Qwen3.8-Flash-Next-NVFP4-mtpft3`, the 49152 `hot2_49152`
rows, probed with 256 random hidden states scaled to the model's activation norm:

| arm | weight RMS err (rel) | argmax agreement vs BF16 | logit RMS err (÷σ) | top-5 set match |
|---|--:|--:|--:|--:|
| **fp8** (ships today) | 0.0264 | **0.9492** | 0.0264 | 0.9437 |
| nvfp4 (from BF16) | 0.0945 | 0.8164 | 0.0945 | 0.8187 |
| fp8+nvfp4 (cheap impl) | 0.0982 | 0.7969 | 0.0981 | 0.8148 |

**On this probe, NVFP4 puts 3.6× the logit noise on the draft head that FP8 does.** The
argmax-flip rates are properties of random hidden states; how they compare with the head's
real inputs was not established on public inputs, so they are not a prediction of served
acceptance. The 3.6× noise ratio is the signal, and the stage-A gate allows only 1 %
acceptance loss at k=7. Quantising from BF16 rather than
from the FP8 rows is worth ~2 points of argmax agreement, so if stage A proceeds the
server should keep a BF16 copy of the hot rows to quantise from.

## 5. What that noise costs acceptance — a margin-based projection (CPU)

The §4 probe drives the head with *random* hidden states, so its argmax-flip rates say
nothing definite about real inputs. A second CPU projection used the target's top-k logits stored in the author's private MTP training
dumps to estimate how much noise it actually takes to flip an argmax, then pushed the
resulting flip rate through a simple p + p² + p³ acceptance model. (The margin
statistics and the projected numbers were measured on private data and are not
published.)

The projection's outcome is not published. It was a chain of estimates, so both the
kernel bandwidth and the real acceptance were measured rather than assumed.

**The one way such an estimate can be too harsh.** It treats every flipped draft
proposal as a lost accept, but flips need not be uniformly distributed: they may
concentrate at *low-margin* positions, which are also the positions where the target
is most likely to disagree with the draft anyway. Where the draft was going to be
rejected regardless, the flip costs nothing. How much that correlation eats back cannot
be reasoned out from the margin distribution alone — it needs the renewal evaluator,
which is why the measurement was run rather than the projection reported.

---

## Status

* Kernel written; numerics check (`bench/n1/check_nvfp4_gemv.py`) queued on the GPU lock.
* Bench written (`bench/n1/bench_nvfp4_gemv.py`): FP8 vs NVFP4 head-to-head at the five
  dense shapes plus a tile sweep, on `bench/w8a16v2/harness.py`'s CUPTI-median metric.
* Offline evaluator patched and consistency-verified bit-exact against the packer.
* **Blocked on GPU.** `~/.gpu.lock` is held by another job (MTP training-data generation
  on private prompts), and the P1 pruning agent is also queued ahead. Both N1 GPU jobs are
  queued behind `flock -w 28800` and will run unattended when it frees:

  | job | what it writes |
  |---|---|
  | kernel numerics + FP8-vs-NVFP4 head-to-head + M=1 tile sweep | three logs (throwaway driver script, not published) |
  | acceptance A/B, arms none/fp8/nvfp4 at k=3,7 (private held-out data) | an evals file and a delta table; no result from it is published |

  The numerics check gates the bench: a decode mismatch aborts before the timing runs.

## 6. Offline acceptance gate

`mtp-train-n1/calib/n1_lmhead_eval.sh` ran the three arms (`none` = BF16 head, `fp8` =
ships today, `nvfp4`) at k=3 and k=7 on identical rows of the private held-out split
(2026-09-06). Gate: ≤ 1 % acceptance loss at k=7. Because the rows are private, no result
or conclusion from this evaluation is published — not the numbers, not the gate outcome,
and nothing about §5's projection or the `fp8+nvfp4` compounding.

## 7. MEASURED: kernel bandwidth — the bar is cleared

`bench/n1/bench_nvfp4_gemv.py`, CUPTI kernel-duration medians over CUDA-graph replay,
weight working set several times the 128 MB L2, GPU otherwise idle (2026-09-06 16:44).
Bytes counted honestly: FP8 `N*K`, NVFP4 `N*K/2 + N*K/16`.

| shape | M | FP8 µs | %roof | NVFP4 µs | %roof | speedup |
|---|--:|--:|--:|--:|--:|--:|
| **draft lm_head** 49152×2560 | 1 | 78.26 | 99.6 % | 56.19 | 78.0 % | **1.393×** |
| | 4 | 79.81 | 97.6 % | 57.79 | 75.8 % | 1.381× |
| | 16 | 89.22 | 87.3 % | 61.57 | 71.2 % | 1.449× |
| **target lm_head** 248320×2560 | 1 | 392.96 | 100.2 % | 250.24 | **88.5 %** | **1.570×** |
| | 4 | 394.53 | 99.8 % | 263.14 | 84.1 % | 1.499× |
| | 16 | 399.30 | 98.6 % | 283.97 | 78.0 % | 1.406× |
| GDN in_proj_qkvz 16384×2560 | 1 | 27.36 | 94.9 % | 18.88 | 77.4 % | 1.449× |
| | 16 | 28.35 | 91.6 % | 22.34 | 65.4 % | 1.269× |
| GDN out_proj 2560×6144 | 1 | 11.90 | 81.8 % | 10.56 | 51.9 % | 1.127× |
| | 16 | 12.58 | 77.4 % | 11.62 | 47.2 % | 1.083× |
| attn qkv 13312×2560 | 1 | 22.67 | 93.1 % | 17.18 | 69.1 % | 1.319× |
| | 16 | 23.94 | 88.2 % | 20.35 | 58.3 % | 1.176× |

The §0 measurement of the FP8 kernel is reproduced here in a separate run — 99.6 / 100.2 % of
roof at the two lm_heads, i.e. the brief's 78–85 % premise does not hold at these two shapes.

**The 4-bit kernel reaches 78 % of roof at the draft head and 88.5 % at the target
head**, against a 55 % break-even and a 69 % ship bar. The
~7 extra ALU ops per weight element are not the binding constraint at these widths.

**Tile sweep** (M=1, 49152×2560, 26 candidates) improves it further: the planner's
default `(32, 256, 1)` gives 56.19 µs, but **`(32, 512, 1, no-dot, 4 warps, 2 stages)`
gives 52.64 µs = 1344.6 GB/s = 83.3 % of roof → 1.487×**. `BLOCK_K` 512 wins because
one row's per-16 block scales then occupy exactly one 32 B sector; at 256 the kernel
fetches a sector to use half of it. Wide `BLOCK_N` is bad here (128 → 69.8 % roof):
fewer, fatter CTAs lose more memory parallelism than the unpack amortisation buys.
Committed to `_TUNED`.

Numerics, same run: **0/5632 one-hot decode mismatches** against the reference dequant
(the kernel's bit-trick decode is bit-exact), all 16 reachable codes round-trip at
0.000e+00, and GEMV rms error **1.66e-03 vs the FP8 kernel's 1.65e-03** — the same
accumulation quality, as the bf16-exactness argument in §1 predicted.

---

## 8. Stage A verdict: **GO, but only at W16**

t/s ∝ acceptance ÷ step time. Draft lm_head exclusive time is 1068 µs of a 17.4 ms W16
step and 158 µs of a 9.94 ms W4 step; the tuned 1.487× removes 32.8 % of it.

| | step-time saving |
|---|--:|
| **W16** | −350 µs = **−2.01 %** |
| **W4** | −52 µs = **−0.52 %** |

On step time alone, stage A is a modest W16 win and a wash at W4. The bandwidth gate
passes (83 % ≫ 55 %); the offline acceptance gate (§6) ran on private data and no
result from it is published. Stage A went on to a server A/B — but
a gain of a percent or so on one width is a thin return for a new kernel plus a
load-time quantisation path, and that judgement should be made before the integration
is written.

## 9. Stages B and C: GO to evaluate, on these numbers

| | shape | speedup | W16 step | W4 step |
|---|---|--:|--:|--:|
| **B** target lm_head | 248320×2560, M=1 | **1.570×** | −0.82 % | **−1.38 %** |
| **C** GDN in_proj_qkvz | 16384×2560 | 1.269–1.449× | −1.06 % | −2.22 % |
| **C** GDN out_proj | 2560×6144 | 1.083–1.127× | −0.22 % | −0.55 % |
| | **all three** | | **−2.1 %** | **−4.2 %** |

Two things this changes about the plan:

* **Stage B is the better target than stage A at W4**, where it is worth 2.7× more step
  time, and it is the single largest speedup measured (1.570×, 88.5 % of roof) because
  248320 rows give the widest grid. Its cost is that it changes model outputs, so it
  needs its own gates — greedy agreement, needle, a quality check — and **stage A's
  acceptance evaluation does not transfer to it.** The draft head only proposes and is
  checked by the target; the target head decides. What draft-head quantisation costs in
  *acceptance* says little about what target-head quantisation costs in output quality.
* **GDN out_proj is not worth doing.** 1.08–1.13× for 0.22 % of the W16 step, and it is
  the one shape where the 4-bit kernel falls to 47–52 % of roof — below break-even
  against a *hypothetical* well-tuned FP8, and it is already the shape `EXCLUSIVE_TIME_MAP`
  §5 item 5 flags as having 3.4 µs/call of FP8-side headroom left. Re-tuning its FP8
  GEMV is the better use of that call. Drop it from stage C; keep qkvz and attn qkv.

## The number that decided stage A (resolved in §7)

Everything above is either measured off-GPU or arithmetic on the existing trace. Two
GPU measurements settle it, and they would have to surprise **in the same direction**
for stage A to ship:

*(Written before §7; the measured answer was 83.3 %.)* What was left was
**`w4a16_nvfp4_gemv` bandwidth at 49152x2560, M=1**, as a fraction of the 1615 GB/s roof:

* **< 55 %** — the kernel is *slower* than the FP8 one it replaces. Stage A dies here.
* **~60 %** (the assumed figure from §0) — 9 % kernel win, −0.56 % of the
  W16 step: not worth shipping the complexity.
* **~80 %** — 30 % kernel win, −1.8 % W16 step, a net gain that is marginal but real.
* **~96 %** (matching FP8's efficiency) — the full 1.778x, −2.7 % W16 step, and stage
  B/C become worth their gates.

The argument in this section rests on the kernel alone; the offline acceptance gate (§6)
ran on private data and no result or conclusion from it is published.


---

# Part 2 — server integration and the in-server A/B

## 10. Stage A shipped behind `SGLANG_MTP_LMHEAD_NVFP4=1`

`eagle_worker_v2._install_nvfp4_draft_head`, plus a new
`layers/quantization/nvfp4_dense.py` (shard reader, disk cache, `NVFP4DenseLinearMethod`).

* **Quantised from BF16, not from the FP8 head.** `init_lm_head` only ever sees the
  target's already-FP8 weight, so the rows are re-read from the checkpoint shard
  instead. `safetensors` slices lazily: 49152 rows in **0.93 s**, no 1.27 GB transient.
* **Cached** next to the token map, keyed by shard identity + tensor name + exact row
  set. Cold **2.2 s**, warm **0.02 s**; the file is **70.8 MB** against the FP8 slice's
  125.8 MB. Verified the packed weight is **0 / 125 829 120 elements different** from
  what the offline acceptance run used, so the server serves exactly those weights.

Two bugs the server A/B caught that nothing else would have:

1. **The wide-M fallback was not CUDA-graph safe.** Draft-extend *is* captured, and
   `dequantize_nvfp4` built its e2m1 level table with `torch.tensor(list, device=cuda)`
   per call (a host→device copy) and read the global scale with `.item()` (a sync).
   Capture died with "Cannot copy between CPU and CUDA tensors during CUDA graph
   capture". The table is now cached per device, the scale stays a tensor, and the
   fallback loops the GEMV up to M=128 (capture-safe) before falling back to
   dequant+GEMM for the never-captured prefill regime.
2. **`lm_head.weight` is a registered `nn.Parameter`**, so assigning the packed uint8
   tensor raised until the installer deletes it first — and it must be reinstated,
   because `should_apply_lm_head_quant_method` rejects a head with no `.weight` and
   would have silently fallen back to a dense matmul against the packed bytes.

## 11. MEASURED in-server: the kernel is *better* in the server than on the bench

Three interleaved off/on rounds per width, two workloads each (2026-09-06 17:52–18:15).

**Whole-step wall clock is useless here** — it moved ±5 % round to round, with one W16
round showing the arm *slower* than its own control. A 2 % effect cannot be read off it.
The per-kernel CUPTI medians from `prof/exclusive_time.py` reproduce to ~1 %, so the A/B
is read off those:

| profile | workload | FP8 med | NVFP4 med | speedup | exclusive Δ | as % of step |
|---|---|--:|--:|--:|--:|--:|
| W16 | code-edit | 81.41 µs | 50.40 µs | **1.615×** | −384.2 µs | **−2.07 %** |
| W16 | prose-en | 80.77 | 50.11 | 1.612× | −403.4 | −2.24 % |
| W4 | code-edit | 81.44 | 50.26 | 1.620× | −60.4 | −0.57 % |
| W4 | prose-en | 80.67 | 50.58 | 1.595× | −60.7 | −0.58 % |

Dead consistent across 3 rounds and both workloads, and **better than the isolated
bench's 1.487×**: in the server the FP8 kernel is slower than on the bench (81.4 vs
78.3 µs) while the NVFP4 kernel is faster (50.4 vs 52.6). The 4-bit kernel's smaller
footprint appears to suffer less from the real workload's L2 contention — worth
remembering, since it means the bench *understates* this class of change.

On step time, that is a modest positive at W16 and roughly break-even at W4. (§6's
offline acceptance gate ran on private data; no result from it is published.)

## 12. Stages B and C implemented

* **Stage B** (`SGLANG_LMHEAD_NVFP4=1`) packs all 248320 target rows from the
  checkpoint BF16 and installs *after* the draft head has been sliced, so the draft
  still takes its rows from the target's FP8 head exactly as in production and the
  control arm is the shipped build.
* **Stage C** (`SGLANG_LINEAR_ATTN_NVFP4=1`, `SGLANG_ATTN_NVFP4=1`) hooks the top of
  `Fp8LinearMethod.process_weights_after_loading`, before it quantises. That is not a
  stylistic choice: `qkv_proj` is fused from q/k/v_proj and `in_proj_qkvz` from
  in_proj_qkv + in_proj_z **by the loader**, so those tensors do not exist in the
  checkpoint under their module names and a post-load re-read would have to replicate
  that mapping. At this hook the true BF16 fused tensor is simply there.
  GDN `out_proj` is excluded per §9.


## 13. MEASURED: stages B and C in the server (W4, `n1/qual.sh`)

Both flags were verified to have actually fired before any number below was read — the
first attempt at this produced a perfect-looking Stage B result that was worthless
because the arm had silently fallen back (see §14).

| | kernel (M=1/4) | exclusive Δ | quality |
|---|--:|--:|---|
| **B** target lm_head | 400.5 → **235.2 µs** = **1.703×** | −151.8 µs = **−1.37 %** of the W4 step | synthetic needle 18.5k PASS with identical output, greedy agreement **1.000** (code-edit, first 256 tokens, 2 repeats; other workloads not comparable because two runs of the unchanged build already disagreed) |
| **C** GDN qkvz | 29.28 → **19.14 µs** = **1.530×** | −459 µs on that family, **but ≈ 0 on the step** | needle PASS, agreement 1.000 (code-edit, first 256 tokens) |

### Stage B: **GO**

Whole-step medians moved with it: wall 11070 → 10767 µs, exclusive 8897 → 8643 µs, and
the family diff sums to −325 µs. The lm_head's own −151.8 µs is the part that is
certainly real; the rest is other families becoming *less* exclusive (MoE GEMM2 −117,
`_hc_down` −55, conv1d −47), which is what happens when a single 400 µs kernel that was
blocking the pipeline is halved. Call it **−1.4 % of the W4 step guaranteed, up to
−2.7 % with the overlap effects.** It also frees **278 MB of VRAM** (357.6 vs 635.7 MB).

### Stage C: **NO-GO — a 1.53× kernel win that buys nothing**

This is the most interesting result in N1. The qkvz GEMV family lost 459 µs of exclusive
time and **the step did not move**: median wall 10497 → 10550 µs, total exclusive
8410.6 → 8382.0 µs. The family diff says where it went:

| family | Δ exclusive |
|---|--:|
| GDN qkvz FP8 → NVFP4 | **−459** |
| `cutlass_moe_grouped_gemm1` | **+348** |
| `cutlass_moe_grouped_gemm2` | **+171** |
| `fused_qkvzba_split_reshape_cat_...` | **+115** |
| others | −175 |

Two distinct causes, and only one is a bug:

1. **The qkvz GEMV was acting as cover for the MoE GEMMs.** It went from 88.5 % exclusive
   to 75.4 % — the NVFP4 kernel overlaps *more* — so the MoE grouped GEMMs, which
   `EXCLUSIVE_TIME_MAP` §5 already puts at the expert-weight roofline and calls the real
   bottleneck, simply became exposed. Making a kernel faster does not help when what it
   was really doing was hiding something irreducible behind itself. Nothing about the
   4-bit kernel fixes that.
2. **`NVFP4DenseLinearMethod` has no `apply_into_split`,** so the GDN in_proj two-destination
   fusion (H1-B) falls back to a separate split/reshape/cat kernel: **+115 µs** appearing
   from nothing. This *is* fixable — the NVFP4 GEMV already supports `out2`/`split_n` —
   but recovering 115 µs against a +519 µs MoE exposure does not change the verdict.

Also worth recording: the run logged **99 layers with no `.prefix`** (`RowParallelLinear`,
`ParallelLMHead`). Those are the o_proj/out_proj/down_proj family N1 deliberately skips,
so nothing was missed here — but the same gap is what silently cost the attention qkv
its first run, and any future prefix-keyed selection over `RowParallelLinear` will hit it.

## 14. Four bugs the server found that no offline test would have

Every one of these produced a *plausible-looking* result rather than an error:

1. **CUDA-graph-unsafe dequant** — draft-extend capture died on a host→device copy in
   the wide-M fallback (§10).
2. **`nn.Parameter` assignment** — packed uint8 into `lm_head.weight` raised until the
   installer deletes it first (§10).
3. **Stage B OOM'd and was caught** — `quantize_nvfp4` cast the whole 248320-row head to
   fp32 (2.5 GB) with ~2 GB free. The exception was logged and the arm silently ran as
   the FP8 build, producing needle/agreement results that looked like a clean
   pass and meant nothing. Fixed by chunking the amax and packing on the CPU.
4. **Stage C packed 36 of 48 target layers, silently** — `QKVParallelLinear` never set
   `self.prefix` (its sibling `MergedColumnParallelLinear` does), so every attention qkv
   presented an empty prefix to the matcher and was skipped without an error. Fixed in
   `linear.py`, and the matcher now warns instead of quietly declining.

The lesson for the next stage of this line: **check that the flag actually fired before
reading any A/B number.** Two of the four bugs above produced complete, passing,
entirely meaningless result tables.

---

# Final recommendation

| stage | ship? | step (W16) | step (W4) | acceptance | quality |
|---|---|--:|--:|--:|---|
| **A** draft lm_head | **YES** | **−2.15 %** | −0.58 % | offline gate (§6) ran on private data; no result published | needle PASS |
| **B** target lm_head | **YES** | (not run) | **−1.4 %**, up to −2.7 % | not isolated | needle PASS, agreement 1.000 (code-edit, first 256 tokens) |
| **C** GDN qkvz + attn qkv | **NO** | (not run) | **≈ 0** | — | needle PASS, agreement 1.000 (code-edit, first 256 tokens), but there is no gain to bank |

* **Ship A and B together** — they are independent (A takes its rows before B replaces
  the target head) and together are worth roughly −2 % of the W16 step and −2 % of the
  W4 step, plus 278 MB of VRAM, with no measurable change on the synthetic needle
  and the code-edit agreement check. Stage A's offline acceptance gate ran on private data; no result
  or conclusion from it is published.
* **Do not ship C.** Restore `apply_into_split` on the NVFP4 method first if anyone
  revisits it, but the MoE-exposure effect is the real obstacle and it is not a kernel
  problem.
* **Caveats.** Stage B was measured at W4 only, in a single off/on pair; its −1.4 % core
  is solid but the additional overlap gains need a repeat run to confirm. Stage B's
  effect on *acceptance* was not isolated (the target head decides accepts, so it can
  move acceptance even though the draft is untouched) — the offline evaluator cannot
  model this and it needs an in-server acceptance A/B. End-to-end t/s from fnbench is
  **±20 % at 2 repeats** and cannot confirm any of these; every figure above comes from
  per-kernel CUPTI medians, which reproduce to ~1 %.
