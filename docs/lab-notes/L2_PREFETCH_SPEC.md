# L2 — hiding the dense-weight DRAM stream in the 128 MiB L2: feasibility and verdict

Date: 2026-09-06. GPU: RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs, 128 MiB L2, measured
1.6 TB/s read roof), shared with V5/N1/P1/P2 the whole time (every GPU command under
`flock ~/.gpu.lock`; each bench holds the lock for 1-3 min). CUDA 13.0 runtime (torch 2.13 cu130),
nvcc 13.3 for the helper library, Triton 3.7.1, cuda-python 13.3.1 (already in the venv).
Bench code: `bench/l2_prefetch/` (branch `codex/harness-v1`, commit d0b0dd8). Log: `L2_LOG.md`.
Worktree for Phase B: `$HOME/tools/sglang-l2` (branch `fable/l2-prefetch`).

Everything below is **measured** unless marked *estimated*. All GEMV numbers are CUPTI kernel
medians inside CUDA-graph replays with the production `w8a16_gemv` (production tile, production
`[N, K]`-contiguous fp8 layout, per-channel fp32 scale), after a 320 MB default-policy L2 flush
unless a condition says otherwise. Session note: the same session measured the qkvz GEMV cold at
28.6 us (trace: 29.2), out_proj 12.95 (13.1-14.1), draft head 78.5 (80.9-83.8), i.e. this is a
representative fast session.

---

## 0. TL;DR

1. **The device can do it.** Persisting-L2 set-aside max = **80 MiB** (`cudaDevAttrMaxPersistingL2CacheSize`),
   access-policy window max = **128 MiB**, stream priorities 0..-5. A persisting window keeps
   **98-99 % of up to 80 MiB resident across any amount of normal streaming** (190 MB tested);
   beyond that it degrades gracefully (96 MiB -> 86 %, 112 MiB -> 74 %). Default policy alone keeps
   a working set only while `resident + streamed <~ 120 MB`: the 40 MB qkvz weight survives the W4
   expert stream (77 MB) but not the W16 one (190 MB). `evict_last` cache hints protect nothing
   unless the *streaming* side uses evict-first loads, which the CUTLASS expert GEMM does not.
2. **The GEMV gains less than hoped from an L2 hit.** With the production tiles the dense GEMVs
   run only **1.4-1.8x faster from L2** (qkvz 28.6 -> 16.9 us = 2.4 TB/s; out_proj 12.95 -> 7.1;
   attn qkv 23.9 -> 14.0; at M=16 attn qkv only 24.7 -> 18.0). The draft lm_head (M=1, 120 MiB)
   is the exception: **78.5 -> 21.3 us (5.9 TB/s)** fully hot.
3. **A side-stream prefetch is free only while the GPU runs latency-bound glue; under any
   DRAM-bound kernel it is paid 1:1.** Alongside a chain of tiny kernels a 32 MB prefetch at grid
   32 (1.85 TB/s) leaves the chain's kernels at 1.00x and costs 3.5-5 us per fork/join round;
   alongside 4 GEMVs or a 77 MB stream it adds its whole DRAM time (GEMVs x1.07-1.18). Node
   priorities are captured into the graph and change nothing measurable.
4. **End-to-end emulation of the per-layer scheme (8 rotating emulated layers, next-layer weights
   prefetched on a side stream, one join per graph):** the consumer GEMVs do hit
   (qkvz 28.6 -> 16.0, out_proj 13.2 -> 7.0, i.e. -19 us/layer of kernel time) but the layer only
   shortens by **-10 us (M=4) / -8..-12 us (M=16)** with 62 us of idle glue per layer, and by
   **-8..-10 us** with ~30 us of idle glue (the realistic figure, §4.3) — the part of the prefetch
   that spills past the glue overlaps the consumer's *own* read of the same lines, which is
   nearly free. Per-layer joins eat the prize (-1..-6 us with them); one join per forward is required.
5. **Per-layer dense prefetch: NO-GO, measured in the server (§7).** The emulation projected
   -3.8..-4.7 % at W4 / -2.2..-2.7 % at W16 (47 layers x -8..-10 us) and it was implemented as an
   experiment (`SGLANG_L2_PREFETCH=1`, §6.2). In the server the consumer GEMVs do hit (W4: qkvz
   29.3 -> 18.3 us, out_proj 14.2 -> 10.3; W16: 29.5 -> 17.9, 14.4 -> 10.8) but the "latency-bound"
   glue is not DRAM-idle: the HC mix kernels (2 x 3 MB per layer at 0.8 TB/s) stretch 2.3x while the
   prefetch runs (`_hc_up` raw 503 -> 1147 us/step, `_hc_down` 413 -> 833 at W4; 480 -> 1061 / 411 ->
   827 at W16), the routing prologue +14 %, the grouped GEMMs +1-2 %. Net **+0.86/+0.93 ms/step at W16
   (+4.9/+5.3 %)** and **+1.3 ms at W4 (+13 %, of which ~0.6 ms is the session's own drift)**.
6. **Draft-head pin (`SGLANG_L2_PIN_DRAFT_HEAD=1`, §6.1): NO-GO on this box, for an environmental
   reason.** The mechanism is right and measured: a touch under a persisting window keeps the 120 MiB
   head at 28.3 us/call (vs 78.5) through 10 GB of streaming and dozens of graph launches (§2.1), and
   in the server the two W4 draft forwards right after the per-step touch do hit (draft lm_head median
   81.7 -> **29.2 us**). But **any other CUDA context running on the GPU wipes the persisting lines**
   (`test_pin_api7.py`: a second process launching a 1-us kernel every ms turns 48/48 hits into
   20/48, always from some point on until the next touch), and the desktop compositor is such a
   context: at W16 the 14 draft forwards spread over ~3.5 ms after the touch stay cold (median 81 us),
   and the per-step touch itself reads the head cold (89 us). Net **+0.24/+0.61 ms at W16, +0.83 ms
   at W4** (touch 34-90 us/step + the 80 MiB smaller normal L2: GEMM1 +2..+6 %, GEMM2 +3..+7 %
   exclusive; the rest is drift). It would need a headless GPU (or `SERVE_DISPLAY_HZ`-style compositor
   suppression) plus a touch per draft forward to be re-tried; the code stays in the worktree, default off.
7. **Verdict: NO-GO for Phase B at both widths. Nothing merges.** Both flags are in
   `$HOME/tools/sglang-l2` (branch `fable/l2-prefetch`, default OFF, flag-off path unchanged:
   the worktree's `l2-off` runs match the main tree's `final0906b` step within 0.1 %).

---

## 1. Device limits (measured, `l2ctl.query()`)

| attribute | value |
|---|--:|
| `cudaDevAttrL2CacheSize` | 134 217 728 (128 MiB) |
| `cudaDevAttrMaxPersistingL2CacheSize` | **83 886 080 (80 MiB)** |
| `cudaDevAttrMaxAccessPolicyWindowSize` | 134 217 728 (128 MiB) — a window over 128 MiB is `cudaErrorInvalidValue` |
| `cudaDeviceGetLimit(PersistingL2CacheSize)` before any call | 24 MiB (driver default in this process) |
| stream priority range | 0 (least) .. -5 (greatest) |
| `cudaGraphInstantiateFlagUseNodePriority` | accepted; captured nodes carry the capture stream's priority |

## 2. Item 1 — L2-hit speed of the real GEMV and retention across the expert stream

`bench_hit.py` (results `results/hit_v2.json`, log `logs/hit_v2.log`). Graph per condition:
`flush 320 MB -> [precondition] -> [stream Y MB of other data] -> GEMV`; the GEMV's CUPTI median.

| shape (fp8) | M | MB | cold | L2-hot | hot/cold | after 77 MB, default | after 190 MB, default | after 190 MB, persisting window on the **prefetcher** | after 190 MB, window on the **consumer** (stream attr) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GDN qkvz 16384x2560 | 4 | 40 | 28.64 | **16.93** | 0.59 | 17.15 | 28.64 | 17.44 | 17.38 |
| GDN qkvz | 16 | 40 | 28.94 | **16.26** | 0.56 | 16.61 | 28.90 | 16.75 | 16.77 |
| GDN out_proj 2560x6144 | 4 | 15 | 12.95 | **7.10** | 0.55 | 7.15 | 12.96 | 7.36 | 7.36 |
| GDN out_proj | 16 | 15 | 13.58 | **8.54** | 0.63 | 8.64 | 13.54 | 8.80 | 8.83 |
| attn qkv 13312x2560 | 4 | 32.5 | 23.90 | **14.00** | 0.59 | 14.21 | 23.98 | 14.40 | 14.42 |
| attn qkv | 16 | 32.5 | 24.66 | **18.00** | 0.73 | 18.34 | 24.51 | 18.56 | 18.56 |
| draft lm_head 49152x2560 | 1 | 120 | 78.48 | **21.34** | 0.27 | 69.62 | 78.50 | **28.35** (hitRatio 1.0, window 120 MiB > 80 MiB set-aside) | 34.98 (hitRatio 0.67) |

* **L2-hot GEMV bandwidth is 2.2-2.6 TB/s for the dense shapes** (vs 1.45 TB/s cold): the tile
  is DRAM-tuned. `bench_hotcfg.py` re-tuned the tile for the hot case: qkvz M=4 (64,128,2,4w)
  hot **13.8** us / cold 29.8 (production 16.9 / 28.6); attn qkv M=16 (64,128,4,8w) hot 13.3 /
  cold 28.9 (production 18.0 / 25.2); out_proj (64,256,4,8w) hot 6.7 / cold 13.0. So a
  hot-tuned second tile table would add ~15-25 % to the hit gain, at a cold-side cost for some
  shapes — only worth it if the prefetch itself is worth it.
* **Retention (touch X MB, stream Y MB, re-touch; `hit_frac` from the re-touch time):**

  | policy | X \ Y | 40 | 77 | 120 | 190 |
  |---|--:|--:|--:|--:|--:|
  | default | 48 | 0.99 | 0.96 | 0.00 | 0.00 |
  | default | 64 | 0.99 | 0.05 | 0.00 | 0.00 |
  | default | 96 | 0.64 | 0.00 | 0.00 | 0.00 |
  | persisting window | 64 | 0.99 | 0.98 | 0.98 | 0.98 |
  | persisting window | 80 | 0.99 | 0.99 | 0.99 | 0.99 |
  | persisting window | 96 | 0.97 | 0.86 | 0.86 | 0.86 |
  | persisting window | 112 | 0.74 | 0.74 | 0.74 | 0.74 |
  | evict_last hint, stream evict-first (.cs) | 112 | 0.99 | 0.99 | 0.98 | 0.98 |
  | evict_last hint, stream default | (GEMV rows above) | | 40 MB -> 0.36 | | |

  The default policy behaves like an LRU of ~120 MB. The persisting set-aside is worth its
  nominal 80 MiB (effective ~82-83 MiB from the 96/112 rows). Persisting lines survive across
  graph replays and across the 320 MB flush (steady state), and a normal access to a persisting
  line does not demote it (the consumer needs no attribute).
* **Streaming reads under a streaming window (`hitProp=Streaming`) or `.cs` loads make no
  difference once the target is persisting** (persist+stream 17.44 vs persist+cs 17.14 vs
  persist+swin 17.39 for qkvz) — the set-aside is the mechanism, not the streaming hint.
* **The persisting carve-out costs the pure stream nothing** (77 MB touch: 50.5 us at limit 0 and
  at limit 80 MiB) — but that kernel has no reuse; the real grouped GEMM has ~13 % L2 hits (G1),
  which only the in-server A/B can price.
### 2.1 What marks a line persisting, and how long it stays (`test_pin_api2-5.py`)

| test | result |
|---|--:|
| consumer GEMV under a stream-attribute window (captured node shows the window), no touch | 78.4 us — **a Triton GEMV's plain loads create no persisting lines** |
| touch under the window inside the graph, GEMV also under the window (hitRatio 1.0, 120 MiB) | 46.4 us — the consumer's own persisting accesses **thrash** the 80 MiB set-aside |
| touch under the window (CUDA launch attribute *or* stream attribute set/cleared around a Triton touch), GEMV without window | **28.3 us** |
| one touch, then 3 x [190 MB stream][GEMV] and a 320 MB flush + GEMV in the same graph | 28.1 / 28.1 / 28.1 / 28.4 — normal reads do not demote |
| one touch graph, then 16 GEMVs + 3.5 GB stream per "step", 3 steps (48 consumers, 10.5 GB streamed) | all 28 us — survives graph boundaries and the verify-sized stream |
| touch once at init only (out of graph), replay [flush][stream][GEMV] | 78 us — does not survive |
| touch inside a draft-extend-like graph once per emulated step | step 0 cold, every later consumer 28 us |
| the same with a **second process** launching a 1 us kernel every ~1 ms (`test_pin_api7.py`) | 20/48 hot: from some consumer on, cold until the next touch — **a context switch wipes the persisting set** |
| the exact server path (`l2_pin.pin_tensor` + `touch_now` inside a capture, cuda-python stream attribute) | 28.3 us (`test_pin_api6.py`) |

=> mark with a dedicated touch under the window, never on the consumer; and it only holds while no
other context touches the GPU — which the desktop compositor does every frame.

* **Partial pinning of a big weight is better than linear**: the draft head with only 2/3 of its
  bytes resident runs at 28.4 us, not 1/3 x 78.5 + 2/3 x 21.3 = 40 us: the DRAM third and the L2
  two-thirds stream in parallel through different CTAs, so the call costs about the DRAM time of
  the non-resident part (40 MB / 1.6 TB/s = 25 us). hitRatio 1.0 on the whole 120 MiB window beat
  both the 80 MiB prefix window (29.0) and hitRatio 0.67 (30.8).

## 3. Item 2 — is a side-stream prefetch hidden, and what does it move?

`bench_overlap.py` (`results/overlap.json`). Per round: fork; side stream: touch X = 32 MB of
DRAM-cold data (rotating over 256 MB); main stream: chain; join. 4 rounds per graph.

Prefetcher alone (32 MB, 256 threads/CTA, 4 sector loads in flight per thread):

| grid | 8 | 16 | 32 | 64 | 128 | 188 | 376 |
|---|--:|--:|--:|--:|--:|--:|--:|
| us | 64.2 | 32.8 | 18.1 | 10.9 | 8.0 | 8.2 | 7.6 |
| GB/s | 522 | 1023 | **1852** | 3069 | 4212 | 4105 | 4438 (L2-hot on later rounds) |

| main chain (per round) | prefetch grid | main alone | both | extra | side alone | hidden | chain kernels |
|---|--:|--:|--:|--:|--:|--:|--:|
| 40 tiny Triton kernels (0.8 us each, grid 20) = 34.1 us | 16 | 34.1 | 39.2 | +5.1 | 32.8 | 84 % | x1.04 |
| | **32** | 34.1 | 37.6 | **+3.5** | 18.1 | 81 % | **x1.00** |
| | 64 | 34.1 | 37.6 | +3.6 | 10.9 | 67 % | x1.00 |
| | 188 | 34.1 | 39.0 | +5.0 | 8.2 | 40 % | x1.00 |
| same, grid 160 tiny kernels | 32 | 34.2 | 38.2 | +4.0 | 18.1 | 78 % | x1.00 |
| 4 x qkvz GEMV (DRAM-bound), 122.5 us | 16 | 122.5 | 140.8 | +18.3 | 32.8 | 44 % | **x1.18** |
| | 32 | 122.5 | ~145 | +22 | 18.1 | <0 | x1.07-1.18 |
| 77 MB stream (GEMM proxy), 53.7 us | 32 | 53.7 | 75.4 | +21.7 | 18.1 | <0 | x1.26 |
| | 188 | 53.7 | 75.5 | +21.8 | 8.2 | <0 | x1.38 |

* **During latency-bound glue the prefetch runs at full speed (17 us for 32 MB = 1.85 TB/s at grid
  32) and the glue kernels are untouched (x1.00).** The residual +3.5-5 us per round is the
  fork/join dependency cost inside the graph, independent of grid — it is the price of a
  cross-stream edge per layer, and the pipeline bench (§4) shows a single join per graph removes it.
* **Over a DRAM-bound kernel the prefetch is paid in full**: +18-22 us per 32 MB = 32 MB / 1.5-1.75
  TB/s. DRAM bandwidth is simply shared; stream/node priority does not ration it.
* **Grid 32 is the right prefetcher shape**: enough to reach ~1.85 TB/s, small enough (32 CTAs
  of 8 warps) to leave the SMs to the main chain. Grid >= 64 hides less (its own CTAs contend).
* Node priorities: captured (main -3 / side 0 in the node dump), and instantiating with
  `cudaGraphInstantiateFlagUseNodePriority` changes nothing measurable (+-1 us). The launch-attribute
  priority path (`cudaLaunchAttributePriority`) behaves the same.

## 4. End-to-end emulation of the per-layer scheme

`bench_pipeline.py` (`results/pipeline_v2*.json`). 8 emulated decoder layers rotate (8 x 55 MB of
dense weights >= 3.4x L2, so without prefetch every GEMV is DRAM-cold as in the server). Each
layer: `tiny x6 | qkvz GEMV | tiny x5 | out_proj GEMV | tiny x12 | 77 or 190 MB stream (expert GEMM
proxy) | tiny x8`, tiny counts x `glue_scale`. Each layer's two weights live in one contiguous
arena (the layout Phase B would need for a single window). Prefetch of layer L+1's arena on a side
stream, grid PG; variants: at layer start (`start`), right after the expert stream (`postgemm`),
qkvz(L+1) after the stream + out_proj(L+1) after qkvz(L+1) (`jit`); with/without a persisting
window; join per layer or once per graph.

### 4.1 glue_scale 2.5 (= 62 us of DRAM-idle launch-bound kernels per layer), one join per graph

| M | expert stream | none (control) | postgemm g32 | jit g32 | jit g64 | GEMV medians with prefetch |
|--:|--:|--:|--:|--:|--:|---|
| 4 | 77 MB | 165.5 | **155.0 (-10.5)** | 155.9 (-9.6) | 168.3 (+2.8) | qkvz 16.0-17.0, out 7.0 |
| 4 | 190 MB | 244.1 | 234.1 (-10.0) | **234.1 (-10.0)** | 241.9 (-2.2) | qkvz 15.9-16.6, out 6.9-7.1 |
| 16 | 77 MB | 167.2 | 159.9 (-7.3) | **155.0 (-12.2)** | 165.8 (-1.4) | qkvz 16.1-16.6, out 8.2-8.6 |
| 16 | 190 MB | 248.2 | 243.6 (-4.5) | **240.2 (-8.0)** | 255.0 (+6.8) | qkvz 15.9-16.1, out 8.2-8.6 |

(us per layer; control GEMVs qkvz 28.3-29.3, out 12.9-13.7.) The persisting window makes no
difference here (`postgemm+persist` = `postgemm` +-2 us) because the prefetch lands *after* the
expert stream and the consumer follows within one layer — the default LRU already keeps it.
`start` variants (prefetch fired before the layer's own GEMVs) are **worse than the control** by
+12..+35 us: the prefetch overlaps the layer's own DRAM-bound GEMVs and the qkvz GEMV of the
current layer slows from 28.6 to 37-41 us.

### 4.2 Same, but one join per layer (as a naive implementation would do)

| M | stream | none | best variant | delta |
|--:|--:|--:|--:|--:|
| 4 | 77 | 168.2 | jit g32 166.9 | -1.3 |
| 4 | 190 | 246.6 | jit g32 242.1 | -4.5 |
| 16 | 77 | 167.2 | jit g64 164.1 | -3.1 |
| 16 | 190 | 251.8 | jit g64 245.8 | -6.0 |

The per-layer cross-stream join costs 4-9 us per layer — most of the prize.

### 4.3 glue_scale 1.2 (= ~30 us of idle glue per layer, the realistic figure), one join per graph

| M | expert stream | none (control) | postgemm g32 | jit g32 | GEMV medians with prefetch |
|--:|--:|--:|--:|--:|---|
| 4 | 77 MB | 127.7 | **119.4 (-8.3)** | 119.6 (-8.1) | qkvz 18.2-18.6 (partial hit), out 7.2-9.1 |
| 4 | 190 MB | 205.5 | **195.8 (-9.7)** | 196.3 (-9.2) | qkvz 17.9-18.1, out 7.0-8.8 |
| 16 | 77 MB | 129.8 | 121.2 (-8.6) | **120.9 (-8.9)** | qkvz 18.6-19.1, out 8.3-10.2 |
| 16 | 190 MB | 206.1 | **197.0 (-9.1)** | 198.2 (-7.9) | qkvz 17.9-18.7, out 8.3-9.9 |

With half the glue the saving is the same -8..-10 us/layer: the prefetch (41 us concurrent for
55 MB) runs past the glue into the next qkvz GEMV, but that GEMV reads the very lines the
prefetcher is fetching (in-flight misses merge), so the spill costs little; the GEMV lands at
18 us instead of 16 (partially hot). The fork+single-join overhead is inside these numbers.

Why 30 us is the realistic figure: in the W4 verify profile (EXCLUSIVE_TIME_MAP §2.3) the
non-GEMM, non-GEMV exclusive time is ~2.6 ms/step = 54 us/layer, of which the HC chain (24 us/layer)
moves 6 MB/layer at 0.8 TB/s and the shared-expert GEMVs run in the same window as the routing
prologue; the DRAM-idle part is ~30-35 us/layer, and it is split across 5-6 windows of 3-13 us, not
one contiguous block.

## 5. Projection for the per-layer dense prefetch (*estimated* from the measured pieces)

Upper bound with every dense GEMV hitting and a zero-cost prefetch (per-call hit deltas from §2 x
calls/step from EXCLUSIVE_TIME_MAP §3; shared-expert GEMVs excluded because they are already
97-100 % hidden, HC mix weights counted at -1.5 us/call):

| | W4 | W16 |
|---|--:|--:|
| GDN qkvz 36 x (-11.7 / -12.7) | -421 | -457 |
| GDN out_proj 36 x (-5.9 / -5.0) | -212 | -180 |
| attn qkv 12 x (-9.9 / -6.7), o_proj 12 x (-5.9 / -5.0) | -190 | -140 |
| HC mix 96 x -1.5 | -144 | -144 |
| **upper bound** | **-967 us (-9.7 %)** | **-921 us (-5.3 %)** |

What survives, from §4: with 62 us of idle glue per layer 50-60 % of the kernel gain survives
(-10 of -19 us at M=4); with ~30 us of idle glue still 45-50 % (-8..-10 of -19). The emulated
layer prefetches only the two big projections (55 MB); scaling by layer count (47 prefetched
layers, the first layer of a forward cannot be prefetched) rather than by the upper bound:

| | W4 | W16 |
|---|--:|--:|
| 47 layers x (-8..-10 us) | **-0.38..-0.47 ms (-3.8..-4.7 %)** | **-0.38..-0.47 ms (-2.2..-2.7 %)** |

Not in the emulation, all of them costs: the HC chain's own 6 MB/layer of DRAM traffic inside the
"idle" windows, the shared-expert GEMVs (hidden under the routing prologue today; a prefetch
overlapping them slows the prologue window), PDL chaining across the fork (the prefetcher is not
a PDL kernel; `griddepcontrol.wait` in the next main-stream kernel still only waits for its
main-stream predecessor, so no correctness issue, but the overlap PDL buys may shrink), and the
Triton touch kernel reaching 1.5 TB/s (grid 64) instead of the CUDA one's 1.85 TB/s at grid 32.

Why no arenas and no persisting window: with the post-MoE placement nothing large streams between
the prefetch and its consumer, so the default LRU (~120 MB) keeps the 47-58 MB regardless of W4/W16
(§4.1: `+persist` variants = plain variants +-2 us), and a table kernel over the separate weight
tensors needs no launch attribute at all — a plain Triton kernel does it, which also means it
captures into the fork's graphs like any other kernel. The persisting set-aside stays free for the
draft-head pin, so the two flags compose.

## 6. What was built (worktree `sglang-l2`, branch `fable/l2-prefetch`, both flags default OFF)

### 6.1 `SGLANG_L2_PIN_DRAFT_HEAD=1` — `python/sglang/srt/layers/l2_pin.py`

* At draft-head install (`eagle_worker_v2.init_lm_head`): `cudaDeviceSetLimit(PersistingL2CacheSize,
  device max | SGLANG_L2_PIN_MB)` and register the head's storage span (120 MiB, one contiguous fp8
  tensor) as the window (cuda-python runtime API, already in the venv).
* At the end of every draft-extend forward (`Qwen4ExpForCausalLMMTP.forward`, once per decode
  step): set the persisting access-policy window (hitRatio `SGLANG_L2_PIN_HIT_RATIO`, default 1.0)
  as a stream attribute on the current stream, launch one Triton touch over the head
  (`l2_prefetch.touch`, grid 64), clear the attribute. Only that captured node carries the window
  (§2.1). `SGLANG_L2_PIN_TOUCH=every` touches in every draft forward instead (fallback if the
  per-step marking ever decays in the server).
* Cost: one touch per step, ~20 us when the lines are resident, ~50 us cold. With the flag off no
  CUDA call is made.
* Measured per call (bench, production tile, M=1, after 190 MB streaming): **78.5 -> 28.4 us**.
  Server calls: draft lm_head 80.9 us x 14 + draft_extend 83.8 us x 1 at W16; 81.1 x 2 + 83.8 at W4.
* *Estimated*: **W16 -0.75 + 0.03 = -0.72 ms/step (-4.1 %), W4 -0.15 + 0.03 = -0.12 ms (-1.2 %)**,
  before any cost from the 48 MiB normal L2 left to the verify step. Risk: the MoE grouped GEMM's
  ~13 % L2 hits (G1 §3); `SGLANG_L2_PIN_MB=48` is the fallback.

### 6.2 `SGLANG_L2_PREFETCH=1` — `python/sglang/srt/layers/l2_prefetch.py` + hooks in `qwen4_exp.py`

* `LayerPrefetcher` (one per `Qwen4ExpModel`, not for the NextN draft): a side stream, a sink, and
  per-layer int64 tables of (sector-aligned base, sectors) over the next layer's
  `linear_attn.in_proj_qkvz/out_proj` or `self_attention.qkv_proj/o_proj` weights (resolved lazily
  on first use, after weight post-processing). `SGLANG_L2_PREFETCH_WHICH=proj,hc,shared` extends
  the set; `_GRID` (64), `_BLOCK` (1024), `_UNROLL` (4), `_WARPS` (8) tune the kernel
  (55 MB in 38 us = 1.5 TB/s alone; `test_prefetch_tune.py`).
* Hook 1, `_run_qwen4_exp_mlp` right after `self.mlp(...)` (decode / target-verify batches only):
  `side.wait_stream(main)` + one `_touch_table_kernel` launch on the side stream for layer L+1.
* Hook 2, `Qwen4ExpModel.forward` after the layer loop: one `main.wait_stream(side)`.
* Capture-safe (same pattern as the PLE prefetch stream), no host sync, no new allocations after
  the first forward; flag off = no code path change.

## 7. In-server validation (MODE=prof, `prof/validate_l2.sh` = validate.sh with a 15-min start
window; same worktree PYTHONPATH for every run, flags differ; traces `prof/traces/l2*-{w16,w4}-*`)

**Session caveat first**: the box drifted during the runs (W16 `off` = 17.30/17.49 ms at 23:44,
18.84/19.11 ms at 23:56, same binary and flags; W4 `off` 10.38/10.13 at 00:03). Wall deltas are
therefore only meaningful against the `off` run of the same pair, and the per-kernel medians
(`prof/l2_compare.py`, `prof/exclusive_time.py`) are the evidence.

| profile / pair | wall code-edit | wall prose-en | draft lm_head med | qkvz med | out_proj med | GEMM1 excl | `_hc_up` raw/step | touch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| W16 `l2-off` (23:44) | 17 300 | 17 494 | 81.8 | 29.5 | 14.4 | 3885 / 3893 | 480 | — |
| W16 `l2-pf` (23:48) | **18 156 (+4.9 %)** | **18 419 (+5.3 %)** | 81.5 | **17.9** | **10.8** | 4002 / 4111 | **1061** | 33/step, 45 us |
| W16 `l2b-off` (23:56) | 18 840 | 19 115 | 82.0 | | | 4342 / 4436 | 474 | — |
| W16 `l2b-pin` (00:00) | 19 075 (+1.2 %) | 19 722 (+3.2 %) | 81.2 / 80.5 (**cold**) | | | 4311 / 4746 | 523 | 0.9/step, **89-92 us (cold)** |
| W4 `l2b-off` (00:03) | 10 381 | 10 135 | 81.7 | 29.3 | 14.2 | 1964 / 1789 | 503 | — |
| W4 `l2b-pin` (00:10) | 11 240 (+8.3 %) | 10 964 (+8.2 %) | **29.2 / 29.0 (hot)** | | | 2189 / 2014 | 502 | 0.9/step, 34 us |
| W4 `l2b-pf` (00:13) | 11 773 (+13.4 %) | 11 452 (+13.0 %) | 81.4 | **18.3** | **10.3** | 2134 / 1973 | **1147** | 33/step, 45 us |

(us; "excl" = exclusive us/step for code-edit / prose-en; `_hc_up` raw us/step code-edit.)

* pf: the GEMV families lose 400-780 us/step of exclusive time as intended, and gain nothing back:
  `_hc_up`/`_hc_down` raw time doubles (their median stays 5.0-5.4 us because only the calls that
  overlap the prefetch stretch, to 10-20 us), the routing prologue 540 -> 613 us raw, GEMM1 +1-2 %.
  The touch kernel itself is 96-99 % hidden (5 us exclusive) — hiding it was never the problem;
  its bytes are. 33 (not 47) prefetches/step because the attention layers' projections were not
  resolved (`self_attention` is a method on the layer, not the module attribute) — fixing that would
  only add traffic.
* pin: at W4 the mechanism visibly works (the two draft forwards after the touch run at 29 us) yet
  the step is slower: the touch (34 us) + GEMM1/GEMM2 exclusive +6-12 % (the 48 MiB normal L2) +
  drift. In the W16 traces the head showed no hits after the first forwards (median 81) and the touch is cold
  (89-92 us): the persisting set is wiped between the touch and the drafts — §2.1 / `test_pin_api7.py`
  reproduce the wipe with a second context on the GPU.
* Not run: MODE=full acceptance / needle (both flags are NO-GO on prof; neither changes numerics —
  the prefetch and the touch have no outputs — so acceptance would be unchanged).

## 8. Reproduce

```
. bench/l2_prefetch/env.sh
flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_hit.py --sweep      # item 1 (+ retention)
flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_overlap.py           # item 2
flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_pipeline.py --glue-scale 2.5 --prios 0 --grids 32,64 --no-join-each
flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_hotcfg.py
PYTHONPATH=~/tools/sglang-l2/python:$PYTHONPATH SGLANG_L2_PIN_DRAFT_HEAD=1 flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/test_pin_api.py
```
