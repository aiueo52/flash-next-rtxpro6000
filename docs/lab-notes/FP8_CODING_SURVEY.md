# FP8C — CPU lossless coding survey of serving dense FP8 weights

Date: 2026-09-07. Scope: CPU only; no GPU work, no server, no installation, no commit.

## Verdict

**Storage gate: PASS. In-GEMV performance gate: NOT ESTABLISHED; no production GO.** The dense streams contain useful redundancy. Several measured codecs and exact Huffman size models exceed the >11% W4 / >13% W16 effective dense-block storage thresholds. A specialized restartable Huffman decoder is arithmetically plausible at ~1.6 TB/s, so entropy alone does not kill this direction. However, this CPU survey does not establish a decoder that preserves the existing GEMV roof while delivering the required net saving. Published general ANS/Huffman results do not demonstrate that specific fused FP8 case.

Ignoring decode cost, the best storage format in the tested set is `huffman_byte` at 64 KiB: W4 14.779%, W16 16.381%. Among actual exhaustive round-trip codecs, `zlib9` wins by storage. For the decoder assessment, byte Huffman with 256-weight restarts is the strongest entropy-only candidate; exponent Huffman plus raw sign/mantissa is simpler but saves less.

| 64 KiB format | W4 effective dense saving | W16 effective dense saving | Evidence |
| --- | --- | --- | --- |
| huffman_byte | 14.779% | 16.381% | exact size model, no lane restarts |
| zlib9 | 14.473% | 15.957% | all bytes round-tripped |
| zstd19 | 14.379% | 15.886% | all bytes round-tripped |
| huffman_byte_restart256 | 13.227% | 14.690% | all bytes round-tripped |
| huffman_exp_raw_sm_restart256 | 12.636% | 13.991% | all bytes round-tripped |

## 1. Coverage, serving conversion, and exclusions

**All 198 actual serving FP8 tensors measured**, across all 48 target layers (36 linear-attention, 12 full-attention), the MTP dense attention/shared-expert projections, the full target head (248,320 × 2,560), and the hot2_49152 draft head (49,152 × 2,560). FP8 payload: **3,726,049,280 bytes**; unchanged FP32 row scales: **5,495,808 bytes**. No layer or block sampling.

The measured tensor is the **serving matrix**, in contiguous `[N,K]` order. Checkpoint `q/k/v`, shared `gate/up`, and GDN `qkv/z` are concatenated in loader order into `qkv_proj`, `gate_up_proj`, and `in_proj_qkvz`. The transposed parameter in `fp8.py` is a view, and the GEMV transposes it back; no physical column-major byte permutation is assumed. The manifest lists every checkpoint source for every serving matrix. TP=1 matches `serve-local.sh`.

**Two scope corrections from the actual code:** (1) `mtp.fc_embedding` and `mtp.fc_hidden` are ordinary BF16 `nn.Linear` instances in `_init_linear_projections`; the category selector mentions these names but does not quantize those instances. We measured their counterfactual FP8 streams as two supplemental tensors, excluded from serving totals. (2) GDN `in_proj_ba` and router GEMVs are BF16 despite sharing the `_w8a16_gemv` kernel name; their time remains in the dense-block denominator with zero FP8 saving. Indexer, HC, expert FP4, embeddings, norms, convolution, gates, and vision weights are outside the requested five-category FP8 weight set. HC has a separate opt-in FP8 path and is not silently included.

| Category | Serving tensors | FP8 bytes before | Raw scale bytes | FP8 + scales before |
| --- | --- | --- | --- | --- |
| shared_expert | 96 | 235,929,600 | 737,280 | 236,666,880 |
| attn | 24 | 597,688,320 | 761,856 | 598,450,176 |
| linear_attn | 72 | 2,076,180,480 | 2,727,936 | 2,078,908,416 |
| lm_head.target | 1 | 635,699,200 | 993,280 | 636,692,480 |
| lm_head.draft | 1 | 125,829,120 | 196,608 | 126,025,728 |
| mtp_dense | 4 | 54,722,560 | 78,848 | 54,801,408 |
| TOTAL | 198 | 3,726,049,280 | 5,495,808 | 3,731,545,088 |

Conversion follows the DH1 CPU pattern, with the current production rounding path made explicit:

```python
w = bf16_weight.float()
scale = w.abs().amax(dim=1, keepdim=True) / 448.0
inv_scale = torch.where(scale == 0, 0.0, 1.0 / scale)
q = (w * inv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
```

`Fp8LinearMethod.process_weights_after_loading` passes `group_size = weight.shape[-1]`. On CUDA this reaches the dedicated whole-row JIT in `fp8_kernel.py`, which computes an FP32 reciprocal and multiplies before E4M3 conversion. SM120 uses `precise_math` (the wrapper enables fast math only on SM90). The dedicated whole-row path does not use the group kernel’s 1e-10 floor. **20 observed rows (14 in layer 1 GDN input, 6 in layer 2 GDN input) fall below that floor**, and no row is all zero. We follow the dedicated serving path, without adding the group-kernel floor, and preserve the resulting scales raw. [conversion_audit.json](../bench/fp8coding/results/conversion_audit.json) records the separate below-floor comparison and independent FP32 division checks.

The literal DH1 floor-plus-`w/scale` expression differs from the serving reconstruction on **1,871,292 / 3,726,049,280 bytes** (0.050222%). This comparison includes reciprocal rounding and the 20 below-floor rows. Both hashes are recorded per tensor. This is why blindly reusing the DH1 byte stream would be incorrect. **CPU reconstruction is exhaustive and source-aligned; direct CPU-vs-CUDA bit parity is Not verified**, as required by the no-GPU scope. The independent E4M3 nearest-even reference checks all positive finite codes, their adjacent midpoints, both signs, and signed zero. The lossless-codec claim applies exactly to the reconstructed bytes, not to an unperformed GPU comparison.

Source evidence: [DH1](../../mtp-train/scripts/dh1/screen.py), [category selector](../../sglang-rtxpro6000/python/sglang/srt/qwen4_exp_dense_fp8.py), [FP8 loader](../../sglang-rtxpro6000/python/sglang/srt/layers/quantization/fp8.py), [quantization dispatcher](../../sglang-rtxpro6000/python/sglang/kernels/ops/quantization/fp8_kernel.py), [whole-row CUDA body](../../sglang-rtxpro6000/python/sglang/kernels/jit/csrc/gemm/per_token_quant_fp8.cuh), [MTP constructor](../../sglang-rtxpro6000/python/sglang/srt/models/qwen4_exp_mtp.py), [GEMV layout and shape plans](../../sglang-rtxpro6000/python/sglang/srt/layers/quantization/w8a16_gemv.py).

## 2. Zero-order and scale-conditional entropy

E4M3FN fields: sign `b>>7`, exponent `(b>>3)&15`, mantissa `b&7`. Split coding uses exponent plus **sign-and-mantissa** `((b>>7)<<3)|(b&7)`; it never drops the sign or signed zero. Entropies are empirical bits per original FP8 byte. Category values below average per-tensor entropies by byte count, rather than pooling distinct tensor distributions. `H(B|Q)` uses `Q=floor(4*log2(scale))` (quarter-octave buckets); octave `floor(log2(scale))` results and bucket populations are also in each JSON. Scales are already available to the decoder, but conditional model tables and access costs are not free, so conditional entropy is an information bound, not a codec size.

| Category | H(byte) | H(exponent) | H(mantissa) | H(sign) | H(sign,mantissa) | H(byte \| quarter-octave scale) |
| --- | --- | --- | --- | --- | --- | --- |
| shared_expert | 6.65662 | 2.70944 | 2.98125 | 1.00000 | 3.98124 | 6.56940 |
| attn | 6.67122 | 2.72356 | 2.98117 | 0.99999 | 3.98116 | 6.57472 |
| linear_attn | 6.69762 | 2.74829 | 2.98118 | 0.99998 | 3.98117 | 6.59034 |
| lm_head.target | 6.52839 | 2.59106 | 2.98127 | 0.99999 | 3.98126 | 6.51701 |
| lm_head.draft | 6.52914 | 2.59201 | 2.98129 | 1.00000 | 3.98129 | 6.51329 |
| mtp_dense | 6.72660 | 2.77403 | 2.98122 | 1.00000 | 3.98122 | 6.66132 |

Across actual serving tensors, scale conditioning removes only 0.08422 additional bits/weight on average. Most redundancy is in the exponent. Full-byte coding can also exploit exponent/mantissa dependence, explaining its advantage over two independent fields.

## 3. Block formats and accounting

All 16/32/64 KiB boundaries are in the uncompressed contiguous byte stream, including partial final blocks. zlib-9 and zstandard-19 use independent blocks, no dictionary, and every compressed block is decompressed and compared byte-for-byte. zstd uses the already installed `zstandard` module, `threads=0`, checksum enabled; six CPU compression workers operate on bounded batches. Raw-fallback sizes are also recorded and never count expanded blocks as savings.

A common random-access container adds 32 bytes/tensor, `(blocks+1)` 64-bit offsets, and 4 bytes/block for mode/uncompressed length. Unchanged FP32 row scales are included in **both** before and after totals. Framing and codebooks are not hidden. No external alignment padding or sector-fetch amplification is assumed; these are discussed as decoder risks.

* `huffman_byte`: tensor-static canonical Huffman, 256 one-byte code lengths; exact `ceil(sum(count*length)/8)` per block.
* `huffman_split`: separate 16-symbol exponent and sign/mantissa codebooks, independently byte-aligned planes, 32 code-length bytes. `huffman_exp_raw_sm` leaves the low four sign/mantissa bits raw and needs only 16 code-length bytes.
* `ans12_*_model`: tensor-static frequencies normalized to 4,096, all observed symbols assigned nonzero frequency; cross entropy, not Shannon entropy alone. Tables use 2 bytes/symbol. The model adds 32 independent states/plane/block, 4-byte state + 4-byte offset + 1-byte rounding allowance each. It is an estimate, **not an encoded ANS bitstream or a proven finite-state upper bound**.
* `huffman_*_restart256`: actual reference encoder/decoder, each 256-weight segment independently byte-aligned, one u32 start per segment plus a final u32 offset per block. Exact padding is measured, both variants round-trip every tensor byte. These restart costs are needed for parallel consumption; the no-restart Huffman models are storage comparisons only. Huffman segment payloads are actually encoded and decoded; the surrounding container/index has an explicit exact size specification but is not emitted as a persistent archive.

### Whole-corpus rates at all block sizes

| Format | 16 KiB after bytes | saving | 32 KiB after bytes | saving | 64 KiB after bytes | saving |
| --- | --- | --- | --- | --- | --- | --- |
| zlib9 | 3,131,797,524 | 16.072% | 3,130,628,779 | 16.104% | 3,130,335,507 | 16.112% |
| zstd19 | 3,140,576,719 | 15.837% | 3,134,973,876 | 15.987% | 3,133,999,417 | 16.013% |
| huffman_byte | 3,122,412,976 | 16.324% | 3,120,998,783 | 16.362% | 3,120,291,688 | 16.381% |
| huffman_split | 3,146,710,846 | 15.673% | 3,145,296,636 | 15.711% | 3,144,589,483 | 15.730% |
| ans12_byte_model | 3,185,683,546 | 14.628% | 3,151,513,754 | 15.544% | 3,134,428,637 | 16.002% |
| ans12_split_model | 3,256,355,367 | 12.734% | 3,189,323,148 | 14.531% | 3,155,807,143 | 15.429% |
| huffman_exp_raw_sm | 3,146,707,678 | 15.673% | 3,145,293,468 | 15.711% | 3,144,586,315 | 15.730% |
| ans12_exp_raw_sm_model | 3,199,383,855 | 14.261% | 3,165,214,257 | 15.177% | 3,148,129,182 | 15.635% |
| huffman_byte_restart256 | 3,187,809,486 | 14.571% | 3,185,990,126 | 14.620% | 3,185,080,446 | 14.644% |
| huffman_exp_raw_sm_restart256 | 3,212,105,099 | 13.920% | 3,210,285,739 | 13.969% | 3,209,376,059 | 13.993% |

### Explicit category before/after bytes, 64 KiB

| Category | Before: FP8+scales | zlib-9 | zstd-19 | Huffman byte (size model) | Huffman byte + restart256 | Exponent Huffman + raw sign/mantissa + restart256 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_expert | 236,666,880 | 199,261,427 | 199,427,650 | 197,977,725 | 202,080,009 | 203,626,727 |
| attn | 598,450,176 | 501,174,436 | 501,972,760 | 501,546,012 | 511,938,210 | 515,746,414 |
| linear_attn | 2,078,908,416 | 1,752,386,250 | 1,755,425,366 | 1,748,978,929 | 1,785,080,041 | 1,797,501,403 |
| lm_head.target | 636,692,480 | 526,828,239 | 526,464,625 | 522,156,682 | 533,210,109 | 538,276,253 |
| lm_head.draft | 126,025,728 | 104,283,963 | 104,217,188 | 103,373,653 | 105,561,764 | 106,551,656 |
| mtp_dense | 54,801,408 | 46,401,192 | 46,491,828 | 46,258,687 | 47,210,313 | 47,673,606 |
| TOTAL | 3,731,545,088 | 3,130,335,507 | 3,133,999,417 | 3,120,291,688 | 3,185,080,446 | 3,209,376,059 |

All per-category × format × block-size byte totals are in [aggregate.json](../bench/fp8coding/results/aggregate.json); all per-tensor counts, entropies, per-bucket populations, per-block-size codec totals, and proofs are in the [results directory](../bench/fp8coding/results/). The appendix below is an index, not a sampling subset.

## 4. W4 / W16 exclusive-time weighting

The [09-07 exclusive-time map](EXCLUSIVE_TIME_MAP_0907.md) is the authority for the dense block (2,385 / 3,656 µs, rounded) and median wall (8,919 / 16,083 µs). We reran its existing CPU `prof/exclusive_time.py` sweep on the same four stored X2 traces to recover all small families omitted from the printed top tables; no new profiling. [time_weights.json](../bench/fp8coding/results/time_weights.json) contains both the published-map weights used below and every recovered trace family.

**The printed map does not fully reconcile with the trace sweep:** the matching `_w8a16_gemv*` families sum to 2307.012 µs W4 / 3679.348 µs W16, versus the printed 2,385 / 3,656. This exceeds rounding. We do not silently replace the requested denominator. The primary calculation uses exactly the named dense-family exclusive times printed in the map, with **165 µs W4 / 3 µs W16 of unlisted residual assigned zero saving**. This is conservative: 93.08% / 99.92% of the published dense time has an explicit named attribution; the residual is not claimed as compressible. Recovered trace attribution is retained as a sensitivity calculation, not a purported exact decomposition of the published map.

**Label correction:** the map calls verify grid `[80,6,1]` “attn qkv”, but the source shape plan maps that grid to the 2,560×6,144 `o_proj`; 13,312×2,560 QKV uses `[416,1,1]`. Their combined attention time is unchanged, and the calculation uses the matching tensor distribution. Similarly `[32,10,1]` is the BF16 router, not a shared FP8 projection.

| Subgroup | W4 exclusive µs | W4 dense share | W16 exclusive µs | W16 dense share |
| --- | --- | --- | --- | --- |
| BF16_ba | 88.000000 | 3.690% | 93.000000 | 2.544% |
| attn.out | 145.000000 | 6.080% | 143.000000 | 3.911% |
| linear_attn.in | 893.000000 | 37.442% | 893.000000 | 24.426% |
| linear_attn.out | 470.000000 | 19.706% | 503.000000 | 13.758% |
| lm_head.draft | 237.000000 | 9.937% | 1173.000000 | 32.084% |
| lm_head.target | 387.000000 | 16.226% | 428.000000 | 11.707% |
| mtp_dense.in | 0.000000 | 0.000% | 247.000000 | 6.756% |
| mtp_dense.out | 0.000000 | 0.000% | 173.000000 | 4.732% |
| unattributed_residual | 165.000000 | 6.918% | 3.000000 | 0.082% |
| TOTAL | 2385.000000 | 100% | 3656.000000 | 100% |

For subgroup g, `s_g = 1 - sum(after_bytes_g)/sum(before_bytes_g)`. Effective dense saving is `S_W = sum(t_W,g * s_g) / sum(t_W,g)`, assigning zero to BF16 families and the unlisted residual. Potential step-wall reduction before decode overhead is `sum(t_W,g*s_g)/median_wall_W`. This is a traffic-proportional estimate on **exclusive** time, not an observed speedup. Time traces do not identify individual layer durations within a repeated shape, so within each shape group the all-layer compression ratio is byte-weighted. Equal-shaped layer matrices therefore receive equal weight; min/max sensitivity below avoids overstating that assumption.

| Format | Block KiB | W4 dense saving | W16 dense saving | W4 ideal saved µs / step % | W16 ideal saved µs / step % |
| --- | --- | --- | --- | --- | --- |
| zlib9 | 16 | 14.438% | 15.921% | 344.358 / 3.861% | 582.064 / 3.619% |
| zlib9 | 32 | 14.466% | 15.950% | 345.015 / 3.868% | 583.137 / 3.626% |
| zlib9 | 64 | 14.473% | 15.957% | 345.169 / 3.870% | 583.388 / 3.627% |
| zstd19 | 16 | 14.228% | 15.693% | 339.347 / 3.805% | 573.745 / 3.567% |
| zstd19 | 32 | 14.358% | 15.853% | 342.428 / 3.839% | 579.597 / 3.604% |
| zstd19 | 64 | 14.379% | 15.886% | 342.950 / 3.845% | 580.803 / 3.611% |
| huffman_byte | 16 | 14.728% | 16.326% | 351.272 / 3.938% | 596.865 / 3.711% |
| huffman_byte | 32 | 14.762% | 16.363% | 352.080 / 3.948% | 598.214 / 3.720% |
| huffman_byte | 64 | 14.779% | 16.381% | 352.484 / 3.952% | 598.889 / 3.724% |
| huffman_split | 16 | 14.138% | 15.626% | 337.182 / 3.780% | 571.291 / 3.552% |
| huffman_split | 32 | 14.172% | 15.663% | 337.990 / 3.790% | 572.641 / 3.561% |
| huffman_split | 64 | 14.188% | 15.681% | 338.394 / 3.794% | 573.315 / 3.565% |
| ans12_byte_model | 16 | 13.210% | 14.653% | 315.055 / 3.532% | 535.720 / 3.331% |
| ans12_byte_model | 32 | 14.029% | 15.545% | 334.581 / 3.751% | 568.323 / 3.534% |
| ans12_byte_model | 64 | 14.438% | 15.991% | 344.344 / 3.861% | 584.625 / 3.635% |
| ans12_split_model | 16 | 11.506% | 12.788% | 274.423 / 3.077% | 467.522 / 2.907% |
| ans12_split_model | 32 | 13.112% | 14.537% | 312.728 / 3.506% | 531.481 / 3.305% |
| ans12_split_model | 64 | 13.915% | 15.412% | 331.881 / 3.721% | 563.460 / 3.503% |
| huffman_exp_raw_sm | 16 | 14.138% | 15.626% | 337.183 / 3.781% | 571.293 / 3.552% |
| huffman_exp_raw_sm | 32 | 14.172% | 15.663% | 337.991 / 3.790% | 572.642 / 3.561% |
| huffman_exp_raw_sm | 64 | 14.188% | 15.682% | 338.395 / 3.794% | 573.317 / 3.565% |
| ans12_exp_raw_sm_model | 16 | 12.871% | 14.275% | 306.978 / 3.442% | 521.891 / 3.245% |
| ans12_exp_raw_sm_model | 32 | 13.690% | 15.167% | 326.504 / 3.661% | 554.494 / 3.448% |
| ans12_exp_raw_sm_model | 64 | 14.099% | 15.613% | 336.268 / 3.770% | 570.796 / 3.549% |
| huffman_byte_restart256 | 16 | 13.161% | 14.619% | 313.901 / 3.519% | 534.464 / 3.323% |
| huffman_byte_restart256 | 32 | 13.205% | 14.666% | 314.941 / 3.531% | 536.200 / 3.334% |
| huffman_byte_restart256 | 64 | 13.227% | 14.690% | 315.461 / 3.537% | 537.068 / 3.339% |
| huffman_exp_raw_sm_restart256 | 16 | 12.571% | 13.919% | 299.812 / 3.362% | 508.896 / 3.164% |
| huffman_exp_raw_sm_restart256 | 32 | 12.614% | 13.967% | 300.852 / 3.373% | 510.632 / 3.175% |
| huffman_exp_raw_sm_restart256 | 64 | 12.636% | 13.991% | 301.372 / 3.379% | 511.500 / 3.180% |

| Restartable byte Huffman, 64 KiB | Min layer ratio per shape | Mean estimate | Max layer ratio per shape |
| --- | --- | --- | --- |
| w4 | 12.410% | 13.227% | 13.734% |
| w16 | 14.148% | 14.690% | 15.028% |

| Trace-only sensitivity, restart byte Huffman 64 KiB | Trace dense µs | Weighted saving |
| --- | --- | --- |
| w4 | 2307.012 | 14.091% |
| w16 | 3679.348 | 14.611% |

## 5. Decode-side arithmetic and feasibility

Use 1.6×10^12 **uncompressed weight bytes/s**, 188 SMs, and the local [G1 clock observations](G1_LOG.md) / [megakernel measurements](MEGAKERNEL_SPEC.md) of about 2.3 GHz. This is an explicit clock assumption from prior local measurements, not a live GPU query. The required per-SM output is `1.6e12/188 = 8.511 GB/s = 3.700 bytes/SM/cycle`. At 1.5 GHz it rises to 5.674 bytes/SM/cycle.

At 2.3 GHz, raw streaming permits only 10.24 / 20.48 / 40.96 ns per 16/32/64 KiB across the whole GPU, or 1.925 / 3.850 / 7.700 µs per SM at even load. A single dependent symbol decoder per SM would need 3.7 symbols/cycle and cannot suffice. Even one 32-lane decoding warp/SM has only `32/3.7 = 8.65` cycles per lane-symbol, before GEMV work. With four warps it becomes 34.59 cycles per lane-symbol, but issue bandwidth is still shared; more warps hide latency rather than create ALU throughput.

A practical canonical byte-Huffman decoder needs: refill/amortized loads, bit-window alignment, primary LUT lookup, code/length extraction, bit-pointer advance, and occasional secondary lookup for long codes. A reasonable **instruction estimate, not SASS measurement**, is 8–16 scalar integer/bit instructions plus ~1 table read per weight, excluding the existing FP8-to-BF16 conversion and GEMV math. Exponent-only coding adds raw nibble extraction and reassembly (~3–4 simple operations), but shrinks the alphabet and table. For the target head the exponent Huffman maximum is 13 bits and 0.3694% of exponents exceed 8 bits; an 8-bit primary table with a fallback is small, but the byte-Huffman maximum is 18 bits. A flat 2^18 table is inappropriate for per-CTA shared memory.

rANS per symbol needs `slot=state&4095`, symbol/frequency/cumulative lookup, `state=freq*(state>>12)+(slot-cum)`, a threshold test, and conditional byte refill, plus indexing/state maintenance. Budget roughly 10–18 scalar instructions and one table lookup/decoded symbol, not an integer divide (division is encoder-side). Full split coding runs two dependent decoders/weight; exponent-only leaves the sign/mantissa nibble raw. These are algorithmic estimates; real issue cost depends on packing, compiler lowering, bank conflicts, register spills, and latency hiding.

The [NVIDIA RTX Blackwell architecture whitepaper, pp. 9–11 and Appendix A](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) describes 128 unified FP32/INT32 CUDA lanes/SM and shared use of those lanes. Using the optimistic one simple lane instruction/cycle roof, `188*128*2.3e9 = 55.35e12` lane-instructions/s. At 1.6 TB/s, 8 / 12 / 16 / 20 added instructions/byte consume 12.8 / 19.2 / 25.6 / 32.0 Tinst/s, or **23.1 / 34.7 / 46.3 / 57.8%** of that ideal issue capacity. If the relevant instruction mix sustains only 64 lanes/cycle, these fractions double. Thus the arithmetic does **not** prove impossibility, but also leaves no basis to treat decode as free, especially in M=1 GEMV, which uses the same scalar execution resources.

**Random access/coalescing is the harder constraint.** The 256-weight reference segments are a concrete format with measured padding/offsets; each can be decoded independently. They are not yet an optimal mapping to the production 128/256-wide K tiles and split-K grids. For K=2,560 or 640, flat 16/32/64 KiB blocks are not always row/tile aligned. A CTA must either fetch only the segment bytes it owns, or the offline packer must reorder segments into its tile without changing reconstructed weights. Fetching a whole 64 KiB block for each small tile would amplify traffic and invalidate these savings. Different lane streams also produce scattered compressed reads; staging a compressed tile into shared memory restores coalescing but adds shared writes/reads, synchronization, and occupancy pressure. 64 KiB of staging can collide with GEMV multistage shared-memory use. Table expansion, tile-local offset layout, bank conflicts, and sector-rounded fetch costs were not measured.

### How little decode overhead fits

| Candidate (64 KiB) | Profile | Storage saving S | Required net bar b | Remaining dense-time overhead S-b | Equivalent serial decode floor f*1.6/(S-b) TB/s |
| --- | --- | --- | --- | --- | --- |
| huffman_byte_restart256 | w4 | 13.227% | 11.000% | 2.227% | 64.23 |
| huffman_byte_restart256 | w16 | 14.690% | 13.000% | 1.690% | 92.19 |
| huffman_exp_raw_sm_restart256 | w4 | 12.636% | 11.000% | 1.636% | 87.42 |
| huffman_exp_raw_sm_restart256 | w16 | 13.991% | 13.000% | 0.991% | 157.26 |

Here `time_new/time_old ≈ (1-S) + f*D/B_decode`, with `D=1.6 TB/s` and coded time fraction `f=0.893920` W4 / `0.973742` W16, treats decoder time as additive. The table shows why a separate or poorly overlapped decoder cannot work. A perfectly overlapped design instead needs the slowest pipeline stage to deliver at least `1.6/(1-b/f)` = **1.825 TB/s W4 / 1.847 TB/s W16** to preserve the requested net dense-time reduction; merely matching 1.6 TB/s would remove all speedup if decode becomes the bottleneck. This is a uniform-reference-bandwidth screening model: individual GEMVs have different actual bandwidths, and the equations neglect non-weight traffic and unequal CTA load. The usable overhead margin must also pay for any sector padding or reduced occupancy.

### Published evidence and its limits

* [DietGPU, author README](https://github.com/facebookresearch/dietgpu) reports 250–410 GB/s for generalized byte rANS and 250–600 GB/s for its floating-point codec on A100. It describes 4 KiB warp segments and advises sufficiently large arrays. These are broad codec ranges, not a promised FP8 decode-only result on this RTX GPU. 1,600/410 = 3.90× and 1,600/600 = 2.67×; even the favorable endpoints do not establish this target.
* [ZipServ, ASPLOS 2026, §3.2](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf) reports L40S decode throughput at 43.7% of peak bandwidth for DietGPU and 76.5% for Huffman-based DFloat11, and demonstrates why a fused format can beat a materialized decompression pipeline. Those results concern BF16 weights and different kernels; they are evidence for co-design, not a transferable FP8 throughput guarantee.
* [CUHD / Weißenberger GPU Huffman decoder](https://github.com/weissenberger/gpuhd) is a primary published implementation of parallel unpartitioned Huffman decoding. Its README gives no directly comparable TB/s result; no uncertain remembered number is used as evidence here.

A decode-to-DRAM pass would read compressed bytes, write the full bytes, and then make GEMV read them again: roughly `(1-S)+1+1 ≈ 2.85` weight-byte traffic units for S≈15%, versus 1 originally. It is excluded from a viable design. The only surviving direction is a small-table, independently restartable, tile-aware decoder fused into consumption, with nearly complete overlap and byte-identical reconstructed weights. **Plausible candidate: yes. A format shown to clear both the storage and net performance bars: no.** No GPU experiment is performed or authorized by this report.

## 6. Completeness and reproduction

| Check | Status |
| --- | --- |
| 48 target layers; both heads; MTP attention/shared expert | Complete: 198 actual FP8 serving matrices |
| MTP fc_embedding/fc_hidden | 2 supplemental counterfactual FP8 matrices; excluded from actual totals |
| 16 / 32 / 64 KiB zlib-9 and zstd-19 | Every block encoded, decoded, exact byte comparison |
| Byte and exponent/raw-sign-mantissa Huffman restart256 | Every byte encoded and decoded; hashes match independent reconstruction |
| Byte / exponent / mantissa / sign entropy; scale-conditional byte entropy | All tensors, all bytes; octave and quarter-octave buckets |
| Static no-restart Huffman | Exact sizes from actual symbol counts, code lengths, and block padding |
| ANS | 12-bit normalized cross-entropy model only; no ANS bitstream, timing, or GPU kernel |
| Serving CUDA parity / GPU decode / integrated GEMV numerics / throughput | Not verified: CPU-only scope |
| Within-family per-layer time shares | Not available; all-layer byte weighting plus min/max sensitivity |
| Printed dense-total vs recovered trace-family sum | Mismatch disclosed; published-map primary weights, unlisted residual receives zero credit |
| Workspace | Only bench/fp8coding and this report changed; no commit or server |

Run with the requested production interpreter and low priority:

```bash
cd $HOME/tools/flash-next-bench
bash bench/fp8coding/run_all.sh
```

Environment recorded in [manifest.json](../bench/fp8coding/results/manifest.json): Torch 2.13.0+cu130, NumPy 2.3.5, zlib 1.3.2, zstandard 0.25.0, OMP threads 12, nice 19, empty CUDA_VISIBLE_DEVICES; `torch.cuda.is_initialized()` remained false. Tensor-at-a-time conversion uses 1,024-row FP32 chunks; no whole BF16 model load, quantized-model artifact, cache of full FP8 weights, or serving import. Only the pure category selector is loaded by filename. The report and JSON are resumable from existing results, with source-hash drift checks.

### Supplemental BF16 MTP linears, hypothetical FP8 conversion only

| Name | Hypothetical FP8 bytes | H(byte) | zstd-19 64 KiB bytes incl. scale/framing |
| --- | --- | --- | --- |
| mtp.fc_embedding | 6,553,600 | 6.52719 | 5,427,773 |
| mtp.fc_hidden | 6,553,600 | 6.81081 | 5,622,483 |

## Appendix — every serving tensor

Target names abbreviate `model.language_model.layers.` to `L`; MTP names retain their prefix. `Hcond` is quarter-octave scale-conditional byte entropy. Savings include scales and framing. Every row links to its complete per-tensor 16/32/64 KiB results; the matching `.huffman.json` contains actual restartable Huffman round-trip proof.

| Serving tensor | FP8 bytes | H(byte) | H(exp) | H(mant) | Hcond | zstd19 64K saving | restart Huffman byte 64K saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [lm_head](../bench/fp8coding/results/lm_head.json) | 635,699,200 | 6.52839 | 2.59106 | 2.98127 | 6.51701 | 17.313% | 16.253% |
| [L0.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.0.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.63326 | 2.68723 | 2.98120 | 6.55879 | 16.226% | 14.912% |
| [L0.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.0.linear_attn.out_proj.json) | 15,728,640 | 6.75582 | 2.80254 | 2.98109 | 6.64247 | 14.655% | 13.530% |
| [L0.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.0.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.68108 | 2.73233 | 2.98181 | 6.58918 | 15.418% | 14.312% |
| [L0.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.0.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.81342 | 2.86076 | 2.98109 | 6.55441 | 13.882% | 12.709% |
| [L1.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.1.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.75215 | 2.79894 | 2.98120 | 6.63193 | 15.012% | 13.533% |
| [L1.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.1.linear_attn.out_proj.json) | 15,728,640 | 6.69587 | 2.74732 | 2.98104 | 6.60228 | 15.306% | 14.156% |
| [L1.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.1.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59169 | 2.64879 | 2.98150 | 6.53798 | 16.479% | 15.420% |
| [L1.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.1.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.81364 | 2.85768 | 2.98114 | 6.56260 | 13.920% | 12.741% |
| [L10.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.10.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68695 | 2.73785 | 2.98121 | 6.55848 | 15.720% | 14.242% |
| [L10.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.10.linear_attn.out_proj.json) | 15,728,640 | 6.65106 | 2.70553 | 2.98120 | 6.54825 | 15.918% | 14.709% |
| [L10.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.10.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.57796 | 2.63667 | 2.98119 | 6.52575 | 16.630% | 15.575% |
| [L10.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.10.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.63506 | 2.68932 | 2.98103 | 6.53148 | 16.047% | 14.892% |
| [L11.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.11.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.54797 | 2.60888 | 2.98151 | 6.51513 | 16.933% | 15.896% |
| [L11.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.11.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.64364 | 2.69734 | 2.98120 | 6.53285 | 15.962% | 14.787% |
| [L11.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.11.self_attn.qkv_proj.json) | 34,078,720 | 6.64384 | 2.69696 | 2.98123 | 6.54555 | 16.589% | 14.771% |
| [L11.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.11.self_attn.o_proj.json) | 15,728,640 | 6.63113 | 2.68767 | 2.98112 | 6.53725 | 16.184% | 14.999% |
| [L12.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.12.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.65199 | 2.70481 | 2.98117 | 6.55065 | 16.082% | 14.650% |
| [L12.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.12.linear_attn.out_proj.json) | 15,728,640 | 6.62437 | 2.68041 | 2.98098 | 6.53907 | 16.236% | 15.050% |
| [L12.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.12.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59574 | 2.65381 | 2.98121 | 6.53615 | 16.427% | 15.369% |
| [L12.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.12.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.66512 | 2.71708 | 2.98128 | 6.55760 | 15.655% | 14.489% |
| [L13.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.13.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.63958 | 2.69371 | 2.98113 | 6.54597 | 16.234% | 14.849% |
| [L13.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.13.linear_attn.out_proj.json) | 15,728,640 | 6.62220 | 2.67914 | 2.98123 | 6.53357 | 16.303% | 15.106% |
| [L13.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.13.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.56375 | 2.62358 | 2.98125 | 6.52560 | 16.786% | 15.730% |
| [L13.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.13.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.61231 | 2.66787 | 2.98102 | 6.52671 | 16.355% | 15.229% |
| [L14.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.14.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68069 | 2.73267 | 2.98123 | 6.56500 | 15.842% | 14.321% |
| [L14.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.14.linear_attn.out_proj.json) | 15,728,640 | 6.70036 | 2.75232 | 2.98114 | 6.57663 | 15.313% | 14.118% |
| [L14.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.14.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.58460 | 2.64129 | 2.98143 | 6.53873 | 16.557% | 15.502% |
| [L14.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.14.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.65006 | 2.70324 | 2.98125 | 6.54507 | 15.847% | 14.700% |
| [L15.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.15.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.55642 | 2.61593 | 2.98136 | 6.53052 | 16.865% | 15.820% |
| [L15.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.15.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.63634 | 2.69058 | 2.98103 | 6.56398 | 16.041% | 14.894% |
| [L15.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.15.self_attn.qkv_proj.json) | 34,078,720 | 6.63386 | 2.68846 | 2.98117 | 6.54722 | 16.623% | 14.937% |
| [L15.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.15.self_attn.o_proj.json) | 15,728,640 | 6.66154 | 2.71543 | 2.98119 | 6.57581 | 15.788% | 14.582% |
| [L16.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.16.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.67310 | 2.72463 | 2.98119 | 6.57634 | 15.993% | 14.388% |
| [L16.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.16.linear_attn.out_proj.json) | 15,728,640 | 6.67606 | 2.72880 | 2.98111 | 6.57576 | 15.600% | 14.394% |
| [L16.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.16.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.54582 | 2.60706 | 2.98135 | 6.52758 | 16.950% | 15.931% |
| [L16.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.16.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.61425 | 2.66926 | 2.98116 | 6.56687 | 16.303% | 15.203% |
| [L17.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.17.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68848 | 2.73917 | 2.98122 | 6.58619 | 15.779% | 14.211% |
| [L17.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.17.linear_attn.out_proj.json) | 15,728,640 | 6.64606 | 2.70147 | 2.98100 | 6.55550 | 15.994% | 14.791% |
| [L17.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.17.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.55170 | 2.61185 | 2.98153 | 6.52861 | 16.920% | 15.869% |
| [L17.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.17.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.62131 | 2.67557 | 2.98123 | 6.55565 | 16.223% | 15.079% |
| [L18.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.18.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.69094 | 2.74164 | 2.98133 | 6.58109 | 15.682% | 14.191% |
| [L18.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.18.linear_attn.out_proj.json) | 15,728,640 | 6.64717 | 2.70231 | 2.98111 | 6.56153 | 15.976% | 14.778% |
| [L18.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.18.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.57170 | 2.63031 | 2.98112 | 6.53798 | 16.681% | 15.638% |
| [L18.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.18.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.66443 | 2.71615 | 2.98122 | 6.57535 | 15.604% | 14.482% |
| [L19.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.19.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.55483 | 2.61453 | 2.98115 | 6.53282 | 16.886% | 15.840% |
| [L19.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.19.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.66196 | 2.71438 | 2.98094 | 6.56092 | 15.699% | 14.520% |
| [L19.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.19.self_attn.qkv_proj.json) | 34,078,720 | 6.63875 | 2.69288 | 2.98118 | 6.55221 | 16.543% | 14.861% |
| [L19.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.19.self_attn.o_proj.json) | 15,728,640 | 6.62026 | 2.67700 | 2.98128 | 6.53775 | 16.311% | 15.127% |
| [L2.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.2.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.67546 | 2.72664 | 2.98112 | 6.56709 | 15.763% | 14.368% |
| [L2.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.2.linear_attn.out_proj.json) | 15,728,640 | 6.70123 | 2.75258 | 2.98122 | 6.58005 | 15.302% | 14.097% |
| [L2.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.2.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.67414 | 2.72475 | 2.98138 | 6.59242 | 15.487% | 14.383% |
| [L2.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.2.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.83978 | 2.88158 | 2.98127 | 6.62540 | 13.634% | 12.305% |
| [L20.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.20.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.67848 | 2.73000 | 2.98117 | 6.58266 | 15.962% | 14.332% |
| [L20.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.20.linear_attn.out_proj.json) | 15,728,640 | 6.70975 | 2.76105 | 2.98114 | 6.58401 | 15.215% | 14.005% |
| [L20.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.20.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.56216 | 2.62062 | 2.98132 | 6.53755 | 16.805% | 15.753% |
| [L20.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.20.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.65547 | 2.70789 | 2.98120 | 6.57630 | 15.762% | 14.604% |
| [L21.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.21.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70010 | 2.74999 | 2.98113 | 6.60450 | 15.471% | 14.090% |
| [L21.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.21.linear_attn.out_proj.json) | 15,728,640 | 6.65690 | 2.71059 | 2.98112 | 6.55443 | 15.833% | 14.615% |
| [L21.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.21.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.55655 | 2.61636 | 2.98142 | 6.52967 | 16.858% | 15.822% |
| [L21.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.21.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.63632 | 2.69062 | 2.98102 | 6.55937 | 16.000% | 14.865% |
| [L22.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.22.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.74499 | 2.79228 | 2.98122 | 6.60929 | 15.049% | 13.588% |
| [L22.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.22.linear_attn.out_proj.json) | 15,728,640 | 6.71464 | 2.76477 | 2.98119 | 6.62518 | 15.141% | 13.941% |
| [L22.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.22.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.57960 | 2.63676 | 2.98168 | 6.54257 | 16.606% | 15.559% |
| [L22.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.22.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68608 | 2.73616 | 2.98142 | 6.58560 | 15.379% | 14.238% |
| [L23.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.23.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59216 | 2.64742 | 2.98147 | 6.55572 | 16.504% | 15.448% |
| [L23.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.23.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.65093 | 2.70351 | 2.98136 | 6.57305 | 15.812% | 14.654% |
| [L23.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.23.self_attn.qkv_proj.json) | 34,078,720 | 6.64936 | 2.70208 | 2.98114 | 6.57599 | 16.319% | 14.675% |
| [L23.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.23.self_attn.o_proj.json) | 15,728,640 | 6.66463 | 2.71754 | 2.98115 | 6.55931 | 15.735% | 14.521% |
| [L24.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.24.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.71774 | 2.76641 | 2.98116 | 6.62024 | 15.394% | 13.895% |
| [L24.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.24.linear_attn.out_proj.json) | 15,728,640 | 6.65968 | 2.71327 | 2.98113 | 6.57192 | 15.806% | 14.584% |
| [L24.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.24.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59358 | 2.65117 | 2.98120 | 6.54355 | 16.445% | 15.393% |
| [L24.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.24.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.69895 | 2.74872 | 2.98117 | 6.57767 | 15.257% | 14.096% |
| [L25.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.25.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68879 | 2.73971 | 2.98119 | 6.57872 | 15.777% | 14.218% |
| [L25.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.25.linear_attn.out_proj.json) | 15,728,640 | 6.67443 | 2.72709 | 2.98118 | 6.57305 | 15.605% | 14.412% |
| [L25.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.25.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59427 | 2.65154 | 2.98135 | 6.54683 | 16.470% | 15.394% |
| [L25.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.25.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68816 | 2.73909 | 2.98107 | 6.58373 | 15.345% | 14.208% |
| [L26.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.26.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.72145 | 2.77055 | 2.98117 | 6.59336 | 15.329% | 13.849% |
| [L26.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.26.linear_attn.out_proj.json) | 15,728,640 | 6.66202 | 2.71633 | 2.98117 | 6.56370 | 15.800% | 14.588% |
| [L26.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.26.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.61226 | 2.66864 | 2.98155 | 6.54928 | 16.247% | 15.179% |
| [L26.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.26.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.69052 | 2.74095 | 2.98128 | 6.57606 | 15.319% | 14.190% |
| [L27.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.27.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.56500 | 2.62492 | 2.98118 | 6.52889 | 16.769% | 15.713% |
| [L27.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.27.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.69596 | 2.74668 | 2.98094 | 6.56770 | 15.286% | 14.124% |
| [L27.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.27.self_attn.qkv_proj.json) | 34,078,720 | 6.67628 | 2.72797 | 2.98119 | 6.57026 | 16.324% | 14.357% |
| [L27.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.27.self_attn.o_proj.json) | 15,728,640 | 6.63003 | 2.68676 | 2.98109 | 6.53708 | 16.201% | 15.019% |
| [L28.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.28.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68226 | 2.73335 | 2.98116 | 6.57555 | 15.782% | 14.284% |
| [L28.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.28.linear_attn.out_proj.json) | 15,728,640 | 6.63257 | 2.68864 | 2.98109 | 6.54708 | 16.152% | 14.966% |
| [L28.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.28.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59895 | 2.65668 | 2.98134 | 6.54498 | 16.384% | 15.329% |
| [L28.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.28.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.69287 | 2.74344 | 2.98100 | 6.59709 | 15.278% | 14.161% |
| [L29.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.29.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.66222 | 2.71510 | 2.98119 | 6.56464 | 16.025% | 14.546% |
| [L29.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.29.linear_attn.out_proj.json) | 15,728,640 | 6.62970 | 2.68685 | 2.98107 | 6.54015 | 16.213% | 15.016% |
| [L29.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.29.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.58013 | 2.63886 | 2.98137 | 6.53702 | 16.595% | 15.537% |
| [L29.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.29.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.64975 | 2.70300 | 2.98133 | 6.55521 | 15.869% | 14.691% |
| [L3.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.3.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59880 | 2.65491 | 2.98117 | 6.54252 | 16.419% | 15.346% |
| [L3.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.3.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.70293 | 2.75209 | 2.98163 | 6.56377 | 15.160% | 14.050% |
| [L3.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.3.self_attn.qkv_proj.json) | 34,078,720 | 6.66129 | 2.71348 | 2.98121 | 6.55437 | 16.409% | 14.532% |
| [L3.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.3.self_attn.o_proj.json) | 15,728,640 | 6.66771 | 2.71968 | 2.98124 | 6.57178 | 15.653% | 14.473% |
| [L30.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.30.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70461 | 2.75554 | 2.98124 | 6.58750 | 15.587% | 14.045% |
| [L30.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.30.linear_attn.out_proj.json) | 15,728,640 | 6.71242 | 2.76453 | 2.98112 | 6.58220 | 15.199% | 13.978% |
| [L30.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.30.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.61650 | 2.67249 | 2.98127 | 6.55880 | 16.187% | 15.127% |
| [L30.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.30.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.70135 | 2.75155 | 2.98121 | 6.58891 | 15.182% | 14.057% |
| [L31.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.31.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.56922 | 2.62849 | 2.98133 | 6.54381 | 16.711% | 15.674% |
| [L31.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.31.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68858 | 2.73959 | 2.98100 | 6.59560 | 15.340% | 14.202% |
| [L31.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.31.self_attn.qkv_proj.json) | 34,078,720 | 6.65469 | 2.70842 | 2.98115 | 6.55798 | 16.520% | 14.670% |
| [L31.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.31.self_attn.o_proj.json) | 15,728,640 | 6.66977 | 2.72368 | 2.98124 | 6.57826 | 15.702% | 14.490% |
| [L32.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.32.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68988 | 2.74068 | 2.98118 | 6.59299 | 15.823% | 14.195% |
| [L32.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.32.linear_attn.out_proj.json) | 15,728,640 | 6.68661 | 2.73936 | 2.98123 | 6.58288 | 15.485% | 14.281% |
| [L32.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.32.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.54823 | 2.60993 | 2.98130 | 6.53160 | 16.927% | 15.889% |
| [L32.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.32.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.64602 | 2.69874 | 2.98133 | 6.59333 | 15.847% | 14.710% |
| [L33.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.33.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70664 | 2.75622 | 2.98124 | 6.60523 | 15.588% | 14.010% |
| [L33.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.33.linear_attn.out_proj.json) | 15,728,640 | 6.65248 | 2.70804 | 2.98110 | 6.55645 | 15.940% | 14.727% |
| [L33.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.33.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.55784 | 2.61791 | 2.98146 | 6.53337 | 16.850% | 15.789% |
| [L33.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.33.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.64989 | 2.70223 | 2.98129 | 6.58015 | 15.829% | 14.651% |
| [L34.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.34.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70823 | 2.75842 | 2.98129 | 6.59766 | 15.484% | 14.001% |
| [L34.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.34.linear_attn.out_proj.json) | 15,728,640 | 6.65886 | 2.71392 | 2.98108 | 6.56250 | 15.856% | 14.644% |
| [L34.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.34.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.56919 | 2.62839 | 2.98153 | 6.53799 | 16.717% | 15.671% |
| [L34.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.34.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68254 | 2.73281 | 2.98113 | 6.59694 | 15.400% | 14.278% |
| [L35.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.35.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.55307 | 2.61297 | 2.98134 | 6.52876 | 16.895% | 15.851% |
| [L35.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.35.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68887 | 2.73912 | 2.98126 | 6.58222 | 15.335% | 14.208% |
| [L35.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.35.self_attn.qkv_proj.json) | 34,078,720 | 6.65652 | 2.70983 | 2.98119 | 6.56465 | 16.443% | 14.625% |
| [L35.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.35.self_attn.o_proj.json) | 15,728,640 | 6.61661 | 2.67369 | 2.98116 | 6.54337 | 16.345% | 15.177% |
| [L36.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.36.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70735 | 2.75701 | 2.98124 | 6.61358 | 15.616% | 14.005% |
| [L36.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.36.linear_attn.out_proj.json) | 15,728,640 | 6.78434 | 2.83285 | 2.98111 | 6.64241 | 14.318% | 13.102% |
| [L36.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.36.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.59958 | 2.65448 | 2.98166 | 6.56163 | 16.392% | 15.334% |
| [L36.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.36.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.66688 | 2.71828 | 2.98151 | 6.59131 | 15.594% | 14.459% |
| [L37.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.37.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70825 | 2.75761 | 2.98122 | 6.61868 | 15.381% | 13.996% |
| [L37.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.37.linear_attn.out_proj.json) | 15,728,640 | 6.68439 | 2.73690 | 2.98116 | 6.57165 | 15.498% | 14.298% |
| [L37.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.37.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.58002 | 2.63734 | 2.98127 | 6.54532 | 16.627% | 15.549% |
| [L37.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.37.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.64797 | 2.70124 | 2.98094 | 6.57775 | 15.814% | 14.686% |
| [L38.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.38.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.77319 | 2.81902 | 2.98125 | 6.63777 | 14.754% | 13.280% |
| [L38.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.38.linear_attn.out_proj.json) | 15,728,640 | 6.73934 | 2.78771 | 2.98122 | 6.66595 | 14.844% | 13.691% |
| [L38.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.38.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.60088 | 2.65644 | 2.98110 | 6.55935 | 16.396% | 15.345% |
| [L38.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.38.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.71621 | 2.76439 | 2.98129 | 6.61746 | 15.041% | 13.935% |
| [L39.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.39.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.61949 | 2.67334 | 2.98114 | 6.56681 | 16.174% | 15.113% |
| [L39.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.39.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.72974 | 2.77782 | 2.98129 | 6.60429 | 14.882% | 13.742% |
| [L39.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.39.self_attn.qkv_proj.json) | 34,078,720 | 6.68520 | 2.73586 | 2.98118 | 6.61396 | 15.994% | 14.256% |
| [L39.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.39.self_attn.o_proj.json) | 15,728,640 | 6.72226 | 2.77300 | 2.98106 | 6.61782 | 15.031% | 13.855% |
| [L4.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.4.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.67541 | 2.72648 | 2.98114 | 6.57069 | 15.759% | 14.368% |
| [L4.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.4.linear_attn.out_proj.json) | 15,728,640 | 6.71373 | 2.76343 | 2.98107 | 6.59560 | 15.158% | 13.953% |
| [L4.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.4.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.61157 | 2.66652 | 2.98120 | 6.55309 | 16.267% | 15.188% |
| [L4.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.4.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68991 | 2.74074 | 2.98103 | 6.57077 | 15.355% | 14.194% |
| [L40.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.40.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.71196 | 2.76129 | 2.98121 | 6.59954 | 15.524% | 13.955% |
| [L40.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.40.linear_attn.out_proj.json) | 15,728,640 | 6.64930 | 2.70394 | 2.98120 | 6.55505 | 15.926% | 14.729% |
| [L40.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.40.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.61176 | 2.66621 | 2.98172 | 6.56180 | 16.273% | 15.213% |
| [L40.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.40.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.66077 | 2.71287 | 2.98104 | 6.58919 | 15.670% | 14.513% |
| [L41.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.41.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.67003 | 2.72184 | 2.98117 | 6.57292 | 15.831% | 14.433% |
| [L41.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.41.linear_attn.out_proj.json) | 15,728,640 | 6.66154 | 2.71553 | 2.98113 | 6.56409 | 15.798% | 14.576% |
| [L41.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.41.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.60543 | 2.65977 | 2.98137 | 6.56294 | 16.341% | 15.291% |
| [L41.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.41.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.68624 | 2.73694 | 2.98123 | 6.59245 | 15.368% | 14.240% |
| [L42.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.42.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68398 | 2.73500 | 2.98112 | 6.59286 | 15.706% | 14.271% |
| [L42.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.42.linear_attn.out_proj.json) | 15,728,640 | 6.70448 | 2.75673 | 2.98113 | 6.57065 | 15.297% | 14.074% |
| [L42.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.42.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.63397 | 2.68724 | 2.98125 | 6.57439 | 15.995% | 14.932% |
| [L42.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.42.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.70516 | 2.75482 | 2.98107 | 6.59283 | 15.168% | 14.022% |
| [L43.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.43.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.63415 | 2.68650 | 2.98167 | 6.60163 | 16.012% | 14.929% |
| [L43.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.43.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.72000 | 2.76930 | 2.98081 | 6.61948 | 14.999% | 13.842% |
| [L43.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.43.self_attn.qkv_proj.json) | 34,078,720 | 6.76667 | 2.81299 | 2.98120 | 6.64109 | 15.539% | 13.304% |
| [L43.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.43.self_attn.o_proj.json) | 15,728,640 | 6.64620 | 2.70106 | 2.98109 | 6.55386 | 15.987% | 14.785% |
| [L44.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.44.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.82211 | 2.86601 | 2.98116 | 6.67557 | 15.075% | 12.647% |
| [L44.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.44.linear_attn.out_proj.json) | 15,728,640 | 6.75566 | 2.80321 | 2.98118 | 6.64870 | 14.643% | 13.463% |
| [L44.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.44.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.64141 | 2.69403 | 2.98167 | 6.58419 | 15.914% | 14.843% |
| [L44.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.44.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.70359 | 2.75370 | 2.98117 | 6.61161 | 15.232% | 14.030% |
| [L45.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.45.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.68232 | 2.73305 | 2.98121 | 6.58462 | 15.726% | 14.288% |
| [L45.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.45.linear_attn.out_proj.json) | 15,728,640 | 6.78020 | 2.82623 | 2.98112 | 6.65085 | 14.386% | 13.206% |
| [L45.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.45.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.67840 | 2.72843 | 2.98132 | 6.59852 | 15.424% | 14.305% |
| [L45.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.45.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.69634 | 2.74727 | 2.98141 | 6.60992 | 15.319% | 14.123% |
| [L46.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.46.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.79646 | 2.84070 | 2.98123 | 6.67472 | 14.448% | 12.986% |
| [L46.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.46.linear_attn.out_proj.json) | 15,728,640 | 6.77685 | 2.82279 | 2.98115 | 6.68145 | 14.414% | 13.244% |
| [L46.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.46.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.67888 | 2.72857 | 2.98176 | 6.59447 | 15.406% | 14.290% |
| [L46.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.46.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.71254 | 2.76290 | 2.98124 | 6.61827 | 15.071% | 13.934% |
| [L47.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.47.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.74034 | 2.78567 | 2.98123 | 6.68489 | 14.606% | 13.524% |
| [L47.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.47.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.76572 | 2.81215 | 2.98135 | 6.65062 | 14.440% | 13.293% |
| [L47.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.47.self_attn.qkv_proj.json) | 34,078,720 | 6.79122 | 2.83616 | 2.98116 | 6.66681 | 15.467% | 13.054% |
| [L47.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.47.self_attn.o_proj.json) | 15,728,640 | 6.67502 | 2.72686 | 2.98106 | 6.58352 | 15.584% | 14.400% |
| [L5.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.5.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.71131 | 2.76034 | 2.98123 | 6.59370 | 15.450% | 13.963% |
| [L5.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.5.linear_attn.out_proj.json) | 15,728,640 | 6.76926 | 2.81670 | 2.98114 | 6.60319 | 14.540% | 13.308% |
| [L5.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.5.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.62662 | 2.68126 | 2.98108 | 6.55782 | 16.073% | 15.009% |
| [L5.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.5.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.74698 | 2.79586 | 2.98143 | 6.58063 | 14.691% | 13.538% |
| [L6.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.6.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.70573 | 2.75510 | 2.98124 | 6.58846 | 15.470% | 14.029% |
| [L6.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.6.linear_attn.out_proj.json) | 15,728,640 | 6.72985 | 2.77848 | 2.98118 | 6.59768 | 14.950% | 13.775% |
| [L6.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.6.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.63528 | 2.68886 | 2.98147 | 6.57004 | 15.980% | 14.918% |
| [L6.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.6.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.71778 | 2.76633 | 2.98106 | 6.57846 | 15.020% | 13.883% |
| [L7.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.7.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.60237 | 2.65843 | 2.98145 | 6.55029 | 16.350% | 15.294% |
| [L7.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.7.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.70004 | 2.74967 | 2.98097 | 6.56128 | 15.222% | 14.081% |
| [L7.self_attn.qkv_proj](../bench/fp8coding/results/model.language_model.layers.7.self_attn.qkv_proj.json) | 34,078,720 | 6.68896 | 2.73900 | 2.98118 | 6.57046 | 16.080% | 14.206% |
| [L7.self_attn.o_proj](../bench/fp8coding/results/model.language_model.layers.7.self_attn.o_proj.json) | 15,728,640 | 6.65001 | 2.70414 | 2.98109 | 6.56221 | 15.904% | 14.701% |
| [L8.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.8.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.72404 | 2.77280 | 2.98119 | 6.62399 | 15.365% | 13.817% |
| [L8.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.8.linear_attn.out_proj.json) | 15,728,640 | 6.66984 | 2.72213 | 2.98122 | 6.57268 | 15.651% | 14.454% |
| [L8.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.8.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.57261 | 2.63143 | 2.98159 | 6.52152 | 16.690% | 15.634% |
| [L8.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.8.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.65449 | 2.70769 | 2.98101 | 6.53677 | 15.809% | 14.630% |
| [L9.linear_attn.in_proj_qkvz](../bench/fp8coding/results/model.language_model.layers.9.linear_attn.in_proj_qkvz.json) | 41,943,040 | 6.66063 | 2.71316 | 2.98124 | 6.55185 | 16.037% | 14.549% |
| [L9.linear_attn.out_proj](../bench/fp8coding/results/model.language_model.layers.9.linear_attn.out_proj.json) | 15,728,640 | 6.66819 | 2.72106 | 2.98118 | 6.55835 | 15.689% | 14.482% |
| [L9.mlp.shared_expert.down_proj](../bench/fp8coding/results/model.language_model.layers.9.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.57757 | 2.63574 | 2.98138 | 6.52757 | 16.646% | 15.582% |
| [L9.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/model.language_model.layers.9.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.63214 | 2.68692 | 2.98113 | 6.53797 | 16.083% | 14.948% |
| [mtp.layers.0.mlp.shared_expert.down_proj](../bench/fp8coding/results/mtp.layers.0.mlp.shared_expert.down_proj.json) | 1,638,400 | 6.70988 | 2.75718 | 2.98141 | 6.67833 | 14.920% | 13.840% |
| [mtp.layers.0.mlp.shared_expert.gate_up_proj](../bench/fp8coding/results/mtp.layers.0.mlp.shared_expert.gate_up_proj.json) | 3,276,800 | 6.82452 | 2.86755 | 2.98119 | 6.69817 | 13.774% | 12.531% |
| [mtp.layers.0.self_attn.qkv_proj](../bench/fp8coding/results/mtp.layers.0.self_attn.qkv_proj.json) | 34,078,720 | 6.72237 | 2.76946 | 2.98125 | 6.67382 | 15.340% | 13.936% |
| [mtp.layers.0.self_attn.o_proj](../bench/fp8coding/results/mtp.layers.0.self_attn.o_proj.json) | 15,728,640 | 6.71710 | 2.76619 | 2.98114 | 6.62478 | 15.095% | 13.946% |
| [draft.lm_head.hot2_49152](../bench/fp8coding/results/draft.lm_head.hot2_49152.json) | 125,829,120 | 6.52914 | 2.59201 | 2.98129 | 6.51329 | 17.305% | 16.238% |

