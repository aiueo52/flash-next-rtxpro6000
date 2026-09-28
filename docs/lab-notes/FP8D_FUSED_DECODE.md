# FP8D — fused byte-Huffman dense GEMV feasibility

Status: **KILL — first cold-L2 head gate failed, 2026-09-08.** Benchmarks only.

The implemented restart256 lane-serial decoder is bit-exact but makes the target
head **34.92x slower at M=4 and 38.64x slower at M=16**. No further GPU runs,
family sweep, server integration, commit, or push followed the failed gate.
The final tables and arithmetic are below; earlier sections retain the incremental
implementation and preflight record. A mirror is also saved in
`$HOME/tools/flash-next-bench/specs/FP8D_FUSED_DECODE.md`.

Worktree: `$HOME/tools/sglang-fp8d`, branch `codex/fp8d-fused-decode`,
base `codex/perf-v1` at `7b4d539f9b`. No commits, pushes, server integration,
launcher changes, environment installation, or production source modifications.

## Contract and gate

Read FP8C survey, its CPU encoder/reference scripts, ASTRA review section 3,
EXCLUSIVE_TIME_MAP_0907, and DH3 cold-L2 protocol before implementation.
Real checkpoint weights are reconstructed by FP8C's `survey.quantize`; byte and
scale hashes must match its saved proofs. Canonical byte Huffman uses independent,
byte-aligned 256-weight segments, u32 offsets, unchanged FP32 row scales.
The GPU decoder consumes the stream inside the GEMV K loop. No decoded weight
tensor is written to DRAM. Original reduction, scaling, and BF16 rounding remain.

First gate: target lm_head 248320 x 2560, M=4 and M=16, cold L2.
If coded GEMV is not faster than the production comparator, stop GPU experiments.
Other families will be listed as not measured when this gate rejects the candidate.
The 2–4 GPU-hour allocation is a ceiling for feasibility, not a reason to continue
after rejection. Each GPU process holds the specified flock, is timeout bounded,
and has a <6 GB memory guard. Evict 256 MiB before every timing region.

## Implementation choice

Triton retains the production arithmetic and supports tuple-valued
`inline_asm_elementwise`. PTX handles serial bit cursor updates, funnel shifts,
word refill, and a two-level canonical lookup. One logical decoder per independent
segment returns 64 packed u32 registers, then Triton joins/unpacks the exact 256
FP8 bytes into the original tile. The codebook is cached read-only device memory;
it is not a 2^18 shared-memory table. No full CUDA rewrite is needed to express
this first candidate. See the [Triton API](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html).

The segment's sequential dependency, irregular compressed reads, register pressure,
and conversion to the GEMV's thread layout are candidate risks to measure.
128-wide K tiles decode the owning 256-byte segment and select its half; this
retains arithmetic order but duplicates decoding. The head gate uses K tiles of
256 and incurs no half-segment duplication.

## Economic accounting

Use FP8C's published-map weights: W4/W16 dense 2385/3656 us and full steps
8919/16083 us. Preserve its attention-grid label correction and zero saving for
BF16/unattributed residuals. Restart256 survey ideal: 315.461/537.068 us.
Required savings: user budget 260/468 us; remaining overhead 55.461/69.068 us.
For a measured family, net saving = exclusive_us * (1 - coded_us/plain_us).
Decode/implementation overhead relative to the traffic-only model is
coded_us - plain_us * coded_bytes/plain_bytes; this includes layout, occupancy,
lookup, and instruction costs, not an independently timed standalone decoder.

Results will be appended as each stage completes.

## CPU completion and offline compilation

All six selected real tensors were reconstructed through FP8C, with identical
FP8/scales proofs. Independent bit-tree decoding verified 869,007,360 emitted
weight bytes with zero differences, plus one-symbol/all-symbol/partial-segment
edge cases. This is CPU correctness evidence, not a GPU timing claim.

| Representative family | N x K | Device-format byte saving |
|---|---|---:|
| Target head | 248320 x 2560 | 16.274% |
| Draft head | 49152 x 2560 | 16.245% |
| GDN input, layer 0 | 16384 x 2560 | 14.883% |
| GDN output, layer 0 | 2560 x 6144 | 13.413% |
| Attention qkv, layer 3 | 13312 x 2560 | 14.492% |
| Attention output, layer 3 | 2560 x 6144 | 14.356% |

These are actual emitted device-format sizes for representative tensors, including
expanded lookup tables and scales. They are not the FP8C all-layer weighted mean.
The serving-shaped target head's table is 4 KiB primary plus a small secondary;
the full per-family table sizes and source hashes are in each archive manifest.

Offline SM120 compilation passed for the diagnostic decoder and target GEMV.
Shared memory: 4096 / 74752 bytes. Generated PTX has no `.local` operations;
ptxas register spilling still needs the GPU-loaded kernel resource check.
Two CPU-only API typing issues were fixed before execution: cast mixed pointer
types to integer addresses at the inline-assembly boundary and use `dim` for
Triton's concatenation argument. No package or environment changes were needed.
The source from the arithmetic `if USE_DOT` through the epilogue is byte-identical
to the production source. GPU runs remain pending the existing lock queue.

## GPU smoke — PASS

The initial sandboxed attempt acquired the lock but failed in `nvidia-smi`
(exit status 9) before creating a CUDA context or launching a kernel. It was
not a kernel failure. The successful host execution retained the same flock,
240-second timeout and memory guard; no environment installation or source
permission changes were made.

RTX PRO 6000 Blackwell Max-Q, driver 595.84, Torch 2.13.0+cu130, Triton 3.7.1.
The first 128 real target rows (327,680 FP8 bytes) decoded with zero differences;
the 4 x 128 GEMV output was bit-identical to production. Loaded-kernel resources:
255 registers/thread, **0 spills**, 82,944 bytes shared/CTA. The runtime-specialized
kernel's resources supersede the generic offline compilation figures above.
Maximum observed process memory was 1,910,505,472 bytes (1.911 GB); guarded phase
6.388 seconds. These are correctness/resource observations, not performance data.
The full target M=4/M=16 cold-L2 gate is queued under the same lock policy.

## GPU gate result

KILL; see bench/fp8d/results/gate_target.json.

## Cold-L2 measurements and final verdict

| Family | M | Plain us | Coded us | Coded/plain | Plain effective GB/s | Coded effective GB/s | Coded stored GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| target lm_head | 4 | 396.960 | 13860.544 | 34.917 | 1603.9 | 45.9 | 38.5 |
| target lm_head | 16 | 405.504 | 15667.200 | 38.636 | 1570.1 | 40.6 | 34.0 |
| draft head (49152 x 2560) | — | Not measured after kill | — | — | — | — | — |
| GDN in_proj (16384 x 2560) | — | Not measured after kill | — | — | — | — | — |
| GDN out_proj (2560 x 6144) | — | Not measured after kill | — | — | — | — | — |
| attention qkv (13312 x 2560) | — | Not measured after kill | — | — | — | — | — |
| attention out (2560 x 6144) | — | Not measured after kill | — | — | — | — | — |

**Verdict: KILL.** Stop this candidate; no family sweep or server integration.

| Width | Ideal saving us | Allowed overhead us | Target overhead alone us | Optimistic net step saving us | Required us |
|---|---:|---:|---:|---:|---:|
| w4 | 315.461 | 55.461 | 13188.673 | -12873.212 | 260 |
| w16 | 537.068 | 69.068 | 16177.928 | -15640.860 | 468 |

Historical-exclusive-time projection, not measured integrated/server performance. Untested families receive their entire FP8C traffic-only saving and zero decode overhead; this intentionally favors the candidate. No draft or GDN timing is inferred.

This rejects the implemented lane-serial, restart256 decoder. It is not a proof that every
possible fused entropy-decoding architecture is slower. No positive speedup is budgeted.
A future candidate would need a materially different decoder/parallel mapping and a new gate.

## Exactness, variation, resources, and limits

- GPU byte diagnostic: **635,699,200 / 635,699,200 target FP8 bytes identical**.
  This diagnostic alone writes decoded bytes to a temporary DRAM buffer; it is
  outside every timing region and absent from the fused GEMV path.
- Full output checks: four deterministic BF16 activation cases per M (scales
  1, 0.01, 100, and zero), totaling **19,865,600 BF16 outputs**, with zero bit
  differences to production. Captured graph outputs also match. These are
  activation test vectors, not a server/acceptance experiment.
- Both loaded fused kernels: **255 registers/thread, zero spills, 82,944 bytes
  shared/CTA**. Thus the candidate's timing is not a decode-to-DRAM or register-
  spill benchmark. Original FP8 encodings and FP32 scales are retained.
- Fifteen cold samples per comparator and width, eviction excluded. M=4 plain
  p10/p90: 394.976/406.733 us; coded: 13,369.773/14,739.475 us. M=16 plain:
  400.083/806.605 us; coded: 13,490.490/16,336.166 us. There are M=16 plain
  outliers up to 900.832 us; their cause was not separately profiled. Even the
  fastest coded M=16 sample (13,381.664 us) exceeds the slowest plain sample by
  14.86x. Variation cannot rescue this candidate, so the kill rule precludes
  additional GPU diagnosis or tuning.
- The byte-saving-only target time for this emitted device format would be
  332.359/339.512 us at M=4/16. Actual implementation overhead above that model
  is **13,528.185/15,327.688 us per call**. This is the total incremental cost
  after compression, including decoding, data redistribution, instruction and
  occupancy effects; it is not a separately timed standalone decoder.
- The integrated table uses the FP8C target ratio (533,210,109 / 636,692,480)
  when subtracting from FP8C's whole-model ideal, so its historical framing is
  not mixed with this archive's slightly smaller framing. The projection gives
  all other families their full survey saving and **zero** decode overhead.
  It still misses the +3% budget by **13,133.212 us W4 / 16,108.860 us W16**.
- Maximum observed process VRAM: **2,306,867,200 bytes (2.307 GB)**; peak PyTorch
  allocated/reserved: 1,514,293,248 / 1,528,823,808 bytes. Successful guarded phase
  intervals were 6.388 s smoke + 8.252 s gate = **14.640 s** (0.00407 hours).
  These intervals include loading/checks and exclude initial imports and shared
  lock queue time; they are not a precise active-kernel or complete lock-hold
  accounting. Both processes used the prescribed flock and 240-second timeout.
  The 2–4 GPU-hour allowance was not consumed after a decisive early rejection.
- CPU offline compilation covered nine configurations including the byte
  diagnostic, production split-K/multiply paths, and GDN gated RMSNorm plans.
  Other tensor families have real encoded archives and compilable candidates,
  but **GPU exactness and timing remain Not measured / Not verified** after kill.
- CPU reconstruction exactly matches FP8C's saved FP8 and scale proofs. Parity
  of that CPU quantization with a freshly executed production CUDA quantizer
  remains FP8C's existing unverified boundary; both GEMV comparators consume the
  same verified bytes/scales. No serving-quality or integrated throughput claim.

Reproduction and artifacts: `bench/fp8d/README.md`, `encode.py`, `test_cpu.py`,
`generate_kernel.py`, `kernels.py`, `compile_cpu.py`, `microbench.py`, and
`run_gpu.sh` in `$HOME/tools/sglang-fp8d`. Raw medians, all samples, byte
proofs, compiler PTX, memory records, and economic arithmetic are under
`bench/fp8d/results/`. `KILL.json` blocks further benchmark phases for this
candidate. The decoder's serial segment mapping and very high register/instruction
footprint warrant a different design, not a positive production forecast.
