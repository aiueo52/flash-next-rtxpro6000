# L2 — log: L2 prefetch of the next layer's dense weights (task L2)

Worktree: `$HOME/tools/sglang-l2` (branch `fable/l2-prefetch` off `codex/perf-v1` @ 5cf83aa321).
Bench: `bench/l2_prefetch/` (branch `codex/harness-v1`). Spec/verdict: `specs/L2_PREFETCH_SPEC.md`.

## 2026-09-06

* 18:30 read EXCLUSIVE_TIME_MAP / MEGAKERNEL_SPEC §5 / w8a16_gemv.py / qwen4_exp.py glue. Created the worktree.
* 18:55 wrote `bench/l2_prefetch/{l2ctl.cu,l2ctl.py,common.py,kernels.py,bench_hit.py,bench_overlap.py,bench_pipeline.py,run_all.sh}`.
  `l2ctl.cu` (nvcc, sm_120) = cudaLaunchKernelEx touch kernel with access-policy-window / priority
  launch attributes, persisting-L2 limit control, graph instantiate with node priorities, node attribute get/set.
* 19:00-… GPU lock held by N1's `n1/all.sh` (validate series) with V5E and P1 queued; waiting in 10-min flock calls.
  First smoke run at 19:28 failed on an import clash (`l2ctl.so` shadowed `l2ctl.py`) -> renamed to `libl2ctl.so`.
* 21:05 first bench_hit run (smoke, qkvz M=4): L2 128 MiB, **max persisting 80 MiB**, max window 128 MiB, stream prio 0..-5.
  qkvz GEMV cold 28.5 us (1470 GB/s) -> L2-hot 19.3 us (2174 GB/s) = only **1.48x**; the 40 MB weight survives 77 MB of
  default-policy streaming (W4 regime) on its own but is fully evicted by 190 MB (W16); a persisting window on the
  *prefetcher* keeps it (19.5 us after 190 MB), consumer GEMV needs nothing. Stream-attribute window IS inherited by
  captured Triton nodes (node dump shows the window on `_w8a16_gemv_kernel`). Crash at Y=190 streaming-window (window >
  128 MiB max) -> clamped, requeued with --sweep.
* 21:13 bench_overlap: side prefetch of 32 MB alongside 40 tiny kernels (0.8 us each): grid 32 -> 17 us concurrent (1.85 TB/s),
  chain kernels x1.00, but +3.5-5 us per round of fork/join cost regardless of grid. Alongside DRAM-bound work (4 GEMVs or a
  77 MB stream) the prefetch is NOT hidden: extra = its full DRAM time (hidden -0.1..-2), GEMVs x1.07-1.18.
  Node priorities are captured (main -3, side 0) and instantiating with UseNodePriority changes nothing measurable.
* 21:14 bench_pipeline v1: GEMVs 1.8x slower than in bench_hit (qkvz 50 us) -> random *bytes* as fp8 (NaN/large values)
  suspected (power-capped clocks); fixed to randn/8, added clock sampling, --glue-scale (tiny kernel is 0.8 us, server
  glue launches ~2 us) and --no-join-each; requeued as pipeline2/3.
* 21:17 bench_hotcfg: a tile re-tuned for the L2-hot case: qkvz M=4 (128,128,4,4w,3st) hot 17.2 vs production-tile hot 22.8
  (cold 31.2 vs 32.5, same session); out_proj (64,256,4,8w) hot 9.7 vs 11.9; attn_qkv (64,128,4) hot 15.8 vs 21.3.
  Session drift: production qkvz cold 28.5 (21:05) vs 32.5 (21:17); out_proj cold 19.1 here vs 13.1 in MEGAKERNEL_SPEC §5.
* 21:38 bench_hit (all shapes + retention sweep): persisting window retains 98-99 % of up to 80 MB across any streaming
  (effective set-aside ~82 MB; 96 MB -> 86 %, 112 MB -> 74 %). Default policy: X+Y <~ 120 MB survives (48 MB survives
  77 MB, 64 MB does not; nothing survives 190 MB). evict_last hint retains only if the *stream* uses evict-first (.cs),
  i.e. useless against the CUTLASS expert GEMM. Draft head (49152x2560, M=1): cold 95 us -> hot 51; with an 80 MB
  persisting prefix window after 190 MB streaming 61 us. **Then found the weight-layout bug**: make_weight built a
  [K,N]-contiguous weight (W_KN path); production is [N,K]-contiguous. out_proj measured 50 us (vs 13-14 in the
  server), qkvz 28.5 (vs 29.2: happened to match). All 21:05-21:41 GEMV numbers superseded by *_v2.
* 21:41 pipeline2/3 (wrong layout, still informative for the *mechanism*): every prefetch variant costs about what it
  saves; the out-of-glue part of the prefetch is paid 1:1 in DRAM time. Per-layer join costs ~3-5 us; a single join at
  the end recovers it (jit grid 32 no-join: -11 us/layer at M=4/77 MB vs +1 with per-layer joins).
* 21:45 a pattern-matched stop of the waiter matched its own shell (exit 144) once; stopped the waiter by its PID instead. Queued the v2 chain
  (hit, pipeline, pipeline no-join, hotcfg) under one lock hold.
* 21:55-22:01 v2 (production layout): qkvz 28.6->16.9 (M=4) / 28.9->16.3 (M=16); out_proj 12.95->7.1 / 13.6->8.5;
  attn qkv 23.9->14.0 / 24.7->18.0; draft head (M=1) 78.5->21.3 hot, 28.4 with an 80 MiB persisting set-aside after
  190 MB streaming. hotcfg: hot-tuned tiles gain another 15-25 % (qkvz 13.8, attn qkv 13.3) at some cold cost.
  pipeline_v2 (per-layer joins) -1..-6 us/layer; no-join -8..-12 us/layer; glue 1.2x still -8..-10.
* 22:23 test_pin_api: the stream-attribute-on-capture design (commit 1f9c256) has NO effect (GEMV 78.4 with the
  window on its node). v2/v3: a GEMV's loads create no persisting lines; a consumer under the window thrashes (46 us);
  a touch under the window (launch attr, or stream attr set/cleared around a Triton touch) + plain consumer = 28.3.
  v4/v5: persisting lines survive normal reads, 3.5 GB streams and graph boundaries (48 consumers / 10.5 GB); a
  single init-time touch does not stick; touch-in-draft-extend once per step works.
* 22:47 SGLANG_L2_PREFETCH implemented (2372411da4): Triton touch-table kernel, side stream after each layer's MoE,
  one join per forward. Kernel tune: 55 MB in 38 us (1.5 TB/s) at grid 64, unroll makes no difference.
* 23:19 pin rewritten to the touch design (ba8c528f2e); queued MODE=prof validation off/pin/pf/both x w16/w4.
## 2026-09-07
* 23:44-00:15 in-server MODE=prof A/B (validate_l2.sh, 15-min start window after an 8-min cold start orphaned a
  server — killed by pgid): W16 pf +4.9/+5.3 % (GEMVs hit, HC kernels stretch 2x); pin: first attempt crashed the
  draft-extend capture (host->device table copy inside capture, fixed 3a766b6707); pin W16 +1.2/+3.2 % with the head
  cold (81 us) and the per-step touch cold (89 us); W4 pin: drafts hot (29 us) but +8 % wall; W4 pf +13 %.
* 00:12 test_pin_api6: the exact server touch path works in isolation (28.3). test_pin_api7: a second CUDA context
  (1 us kernel every ms) wipes the persisting set -> the desktop compositor explains the server result.
* Verdict: NO-GO for both; code stays in the worktree, flags default off. Spec finalised.
