# Kernels

Read-only snapshots of the Triton kernels written for this project, copied out of the patched SGLang
tree so they can be read without applying the patch series, plus the micro-benchmarks used to tune
them. The authoritative versions are in `patches/sglang/` (they import `sglang.*` and are wired in
behind environment flags). All run on SM120 (RTX PRO 6000 Blackwell) with Triton 3.7.1.

| file (`sglang/…`) | flag | what it does | status |
|---|---|---|---|
| `srt/layers/quantization/w8a16_gemv.py` | `SGLANG_FP8_W8A16_GEMV=1` | weight-only FP8 (e4m3, per-channel scale) × BF16 activation GEMV for M ≤ 16 rows: per-shape tile table tuned on the server's real `[N, K]` layout, in-launch "fixup" split-K (last CTA reduces, no extra launch, per-stream scratch slots), optional fused `silu(gate)*up` epilogue, optional fused gated RMSNorm on the A-load, two-destination column split | adopted (dense FP8 weights, draft/target lm_head) |
| `srt/layers/quantization/w4a16_nvfp4_gemv.py` | `SGLANG_MTP_LMHEAD_NVFP4`, `SGLANG_LMHEAD_NVFP4` | NVFP4 (e2m1 + e4m3 block scales) weight × BF16 GEMV, 83–88 % of DRAM roofline | measured, off (acceptance loss) |
| `srt/layers/hc_mix2_triton.py` | `SGLANG_HC_MIX2=1`, `SGLANG_HC_MIX2_FP8=1` | hyper-connection norm + low-rank mix as three kernels (stats/normalise, split-K down-projection with atomics, up-projection + gate + mean), optional FP8 weights; 19.8 → 14.3 µs/call, 9.9 µs with FP8 | adopted |
| `srt/layers/moe/prune_singleton.py` | `SGLANG_MOE_PRUNE_SINGLETON_TAU` | P1: drop routes to experts used by exactly one verify row whose weight < τ (O(N) histogram, k×k rank). Production uses the in-prologue version (FlashInfer patch 06) | P1 superseded by P2 |
| `srt/layers/moe/router_gemv.py` | `SGLANG_ROUTER_GEMV=1` | router gate through the skinny BF16 GEMV | adopted |
| `srt/mem_cache/fp8_kv_store.py` | `SGLANG_KV_FP8_FUSED_STORE=1` | fused BF16→FP8 KV scale + cast + paged store (bit-exact with the unfused path) | adopted |
| `srt/layers/mtp_entry.py` | `SGLANG_MTP_ENTRY_FUSED=1` | fused MTP-entry norm + projection | measured, off |
| `srt/layers/hc_fused_triton.py`, `hc_combine_fused_triton.py` | `SGLANG_HC_FUSED`, `SGLANG_HC_COMBINE_FUSED` | persistent single-kernel HC chain / one-launch combine | measured, off (no gain) |
| `kernels/triton_pdl.py` | `SGLANG_TRITON_PDL=1` | `pdl_wait` / `pdl_trigger` helpers (`griddepcontrol.wait` / `launch_dependents`) so the fork's Triton kernels take part in programmatic dependent launch: the next kernel schedules its blocks and runs its independent prologue while the previous one drains | adopted |

`microbench/` holds the stand-alone benchmarks and numerics checks from the fork's `bench/` directory
(`w8a16/`, `w8a16v2/` incl. the tuners, `hc_mix2/`, `n1/`, `glue_d/`, `glue_e/`, `h1/`,
`hc_layer_boundary/`, `hc_fused/`). They expect the patched SGLang on `PYTHONPATH` and, for some,
the model checkpoint path in a variable at the top of the script.

`prototype/w8a16_gemv.py` is the first stand-alone version of the W8A16 GEMV with its own
micro-benchmark (2026-09-02: 10240×2560 FP8 in 18.9 µs = 1.38 TB/s; lm_head 248320×2560 in 420 µs
vs 860 µs for the BF16 cuBLAS GEMV).

Lesson that cost us two tuning rounds: micro-benchmark the layout the server actually passes. The
first tuned tables were measured on `[K, N]`-contiguous weights while the server hands the kernel an
`[N, K]`-contiguous transposed view, so the whole table was dead at runtime. Reproduce the in-server
per-grid kernel times (CUPTI) within ±20 % before trusting a micro-benchmark.
