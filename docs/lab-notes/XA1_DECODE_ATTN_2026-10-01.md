# XA1 — QSA decode attention as one split-KV Triton kernel (2026-10-01)

Worktree `~/tools/sglang-xa`, branch `opus/decode-attn`, commit **dba52c1994** on production 7b4d539f9b.
Flag `SGLANG_OPT_TRITON_DECODE_ATTN` (EnvBool, default off). Harness: `bench/xa1/`.

The GPU work was microbenchmarks only: 7 GPU-lock holds of 9-32 s each. No server was started. The server
A/B scripts are ready but have NOT been run (§7).

**Status (parent, 2026-10-02 01:15): §7b server A/B done, 8 starts (A1 B1 B2 18:01-19:05, A2-B4 23:44-01:12;
`runs/xa1-st1/ancova8.log`, `ancova8_per_workload.log`). Recommend turning the flag on together with ST1; the
switch is the user's call.**
- Rule 1 PASS: every B arm traces `triton_decode` only (12.2 us median), every A arm the production `xqa_mha`
  (16.2-16.4 us); fallback warnings 0, error lines 0, ST1 lines 3 in every arm.
- Rule 2: lmstudio ms/step|a -2.52% [-3.54 .. -1.49], arm level -2.63% [-5.52 .. +0.35], upper bound below +0.5%:
  PASS. Greedy ms/step|a -0.44%, arm level [-6.67 .. +6.18]: unresolved.
- Rule 3 PASS: no workload x mode tok/step CI excludes 0 (lowest: lmstudio agent-loop -3.2% [-10.2 .. +4.3]).
- Rule 4 PASS: pooled t/s lmstudio +2.82% [+0.01 .. +5.70] (arm level [-0.88 .. +6.66]), greedy +1.17%
  [-1.33 .. +3.75].
- Rule 6: |A2/A1 - 1| on ms/step|a is 2.9% (lmstudio) and 6.3% (greedy). A2 ran almost six hours after A1, and every
  arm after 23:40 sits 2-5% below the evening arms, so the drift is the GPU's evening-to-night change, not XA1;
  the position term absorbs a linear part of it.
- Call per §7 ("if the speed effect stays unresolved, the call rests on rules 1, 3 and 5 plus the
  microbenchmark"; rule 5 dropped in §7b): rules 1 and 3 hold, and the microbenchmark gives 12.2 against 16.3 us
  per call.

**Status (parent, 17:15).** XA1 now also sits on top of ST1, which fixes the §1.3 bug for both arms. It has a
seq_len bound for index-shared rows; the GPU check passed. The server A/B on top of ST1 is queued (§7b).

Labels:
- **measured**: from a GPU microbenchmark, a CPU check or a production trace in this note;
- **est.**: a measured per-call cost × an assumed count.

## 0. Result

**What production does.** Every QSA paged attention call (target verify, draft extend, draft decode) runs three
kernels:
1. a valid-count pass;
2. `_compact_kv`, which gathers the selected fp8 K/V rows into a page-aligned scratch;
3. trtllm-gen XQA decode over that scratch.

**What the flag does.** It runs one Triton kernel instead (`qsa/decode_attn.py`):
- it reads the fp8 KV pool directly, top-k index → `req_to_token` → K/V, so nothing is packed;
- the columns are split over CTAs, and the last-arriving CTA combines the partials (deterministic);
- PDL aware, and CUDA graph safe.

**Per call: production vs flag on** (measured). Setup:
- bs = 1, rows at the 2051-column cap (ctx 8192);
- CUDA graphs of 12 chained calls over 12 layers' KV;
- the real `QwenSparseAttnBackend._forward_paged_attention` built with the flag off and on;
- medians in µs; cold = after an L2 flush;
- source `bench/xa1/results/xa1-integ-gpu.json`.

| call (rows) | off warm | on warm | saving | off cold | on cold | saving |
|---|---|---|---|---|---|---|
| draft decode R1 (index-shared row, 2067 columns) | 14.65 | 8.43 | **−6.2** | 17.74 | 10.24 | **−7.5** |
| verify / draft extend W4 (R4) | 16.54 | 9.03 | **−7.5** | 19.28 | 11.78 | **−7.5** |
| verify / draft extend W8 (R8) | 19.36 | 10.22 | **−9.1** | 22.19 | 13.31 | **−8.9** |
| verify / draft extend W16 (R16) | 29.97 | 16.01 | **−14.0** | 32.09 | 18.25 | **−13.8** |

**The bar is met.** The bar was ≥ 6 µs saved per call, warm and cold, at the verify shapes.
- At R4/R8/R16 the saving is 7.2-14.2 µs at every context measured (512-8192, §4.1). One noisy job4 value
  is the exception; it was re-measured in §4.2.
- R1 also saves 6.1-7.5 µs at the cap.
- The exception is R1 draft decode at short context: it saves only 5.1-5.3 µs warm (6.7 cold). The kernel scans
  all 2067 columns of an index-shared row, while production attends only the ~1000 packed ones.

**XQA's own cost** (measured, chained graph, hot scratch):
- 11.1-11.5 µs per call at R1-R8 and 15.9-16.1 at R16, at ctx ≥ 2048 (10.5-10.6 at R16 below that);
- the rest of production's cost is `_compact_kv` (3.7 / 4.9 / 7.4 / 12.7 µs at R1/R4/R8/R16, isolated) and the
  count pass (1.4-1.5 µs).

**A 5-6 µs call is not reachable at the cap with this design.**
- Best R1 is 8.4 µs warm and 10.2 cold. Rows of ≤ 512 columns reach 6.2-6.6.
- An empty-loop kernel with the same split and combine already costs 3.9 µs.
- Each 64-column tile costs a CTA about 2 µs, so a split of 2 tiles already spends 4.6 µs in its loop (§4.4).

**Production bug found (§1.3).** On index-shared draft decode rows, the packed path drops the newest drafted
positions, including the query's own token, and attends unwritten scratch slots instead.
- At short context it drops all drafted positions: anchor ≤ ~2048 tokens, which covers every prose/agent-loop
  request in the bench workloads.
- At long context it drops the newest h = 3 − (L_a mod 4).
- The flag fixes this as a side effect, so a server A/B measures the speed change and an acceptance change together.

**Estimated ms/step at wa** (est., §8):

| profile | saving per step |
|---|---|
| W4 | −0.11 ms |
| W8 | −0.15..−0.17 ms |
| W16 | −0.26..−0.29 ms |
| default wa mix (0.67/0.21/0.12) | −0.13..−0.15 ms |
| code-edit, mix of the rq1 trace (12 × W8 + 8 × W16 in 20 steps) | −0.20..−0.22 ms |

At the default mix that is about 0.7-1.4% of a wa step, which takes 10-19 ms across the bench workloads; at the
code-edit trace mix it is 1.0-1.6% of a code-edit step. On top of that comes whatever the hole fix does to
acceptance (unknown sign).

**Integration.** Commit dba52c1994 in `~/tools/sglang-xa`, 3 files, +456 lines:
- `environ.py`;
- `qwen_sparse_attn_backend.py` (+67);
- new `qsa/decode_attn.py` (386 lines).

With the flag off, the code path is production's. The A/B scripts are `bench/xa1/arm_xa.sh` and `abba_xa.sh`
(port 8027), ready but not run.

## 1. What the server calls (production 7b4d539f9b, read only)

File:line references are in `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` (“backend”) unless
named otherwise.

### 1.1 Shapes and arguments

| item | value | source |
|---|---|---|
| attention layers | 12 QSA full-attention layers in the target (48 layers, interval 4); MTP draft: 1 full-attention layer | model config.json `text_config` (bench/xa1/cases.py) |
| q | bf16 [rows, 24 q heads, 256] | same |
| KV pool | fp8 e4m3 [slots, 2 kv heads, 256] per layer (GQA 12), page_size 64, `--kv-cache-dtype fp8_e4m3`. Stored as a plain cast with no scales: `set_kv_buffer` gets no k/v scale | backend:1458, :1696; server_args in `runs/rq1/serve-A1.log` |
| top-k rows, verify + draft extend | int32 [rows, 2051]. 2051 = indexer budget 2048 tokens (512 blocks × ratio 4), plus ratio − 1 uncompressed-tail columns. Order: blocks in score order, then the L mod 4 tail, then −1 | `eagle_worker_v2.py:570` (`expanded_width = qsa_token_topk + qsa_compress_ratio - 1`); `qsa/kernel.py` torch_expand_qsa_block_indices |
| top-k rows, draft decode | int32 [rows, 2051 + tail_width], tail_width = num_steps + 1: 2055 / 2059 / 2067 at W4 / W8 / W16. Index sharing is ON in production (see the log line below) | `eagle_worker_v2.py:575`; `:584` |
| which calls | target verify and draft_extend_v2 take the `forward_extend` speculative branch (backend:1473, `_is_speculative_paged_mode` :322-325). Draft decode goes through `forward_decode` (:1682-1700). Both go to `_forward_paged_attention` (:1702), then `_forward_trtllm_sparse` (:1596), on sm100/sm120 via `_resolve_trtllm_sparse_decode` (:85-101) | backend |
| step 1 | `_fa2_valid_counts`: count of columns with 0 ≤ idx < seq_len per row | `qsa/sparse_attn.py:312`, test at :329 |
| step 2 | `_compact_kv`: column c of row r goes to packed slot r × stride + c, stride = 33 pages × 64 = 2112. `valid = (cols < valid_count) & (positions >= 0) & (positions < length)`, where valid_count is the strided width (2112), so every valid column lands at its column number and −1 columns leave their slot unwritten | `sparse_attn.py:370`, :396, :403; backend:1586 (cu_strided) |
| step 3 | `trtllm_batch_decode_with_kv_cache(query=q, kv_cache=(kc, vc), workspace=128 MB, block_tables, seq_lens=valid_counts, max_seq_len=stride, bmm1_scale=layer.scaling, bmm2_scale=1.0)`. kc/vc are HND views of the scratch; the 128 MB workspace is lazy, one per backend instance. block_tables is a static arange [rows, 33] | backend:1667, :1670-1678, :1587-1591 |
| defaults of that call | window_left −1 (no window), sinks None, skip-softmax off, out_dtype None → bf16, q_len_per_req 1 (no spec-dec mask: each verify token is its own row with its own valid count). enable_pdl None → `device_support_pdl` → on | flashinfer `decode.py:3008` (signature), :3217 |
| XQA module | `xqa_input_bf16_kv_cache_e4m3_output_bf16_page_size_64_head_dim_256_head_group_ratio_12_use_sliding_window_False_use_spec_dec_False_spec_q_seq_len_1`, sha256 6c52691b2691977c… | shared FlashInfer cache (read only) |
| seq_lens | `metadata.sequence_lengths` is the row's visible length (position + 1); validity is tested against it. XQA's `seq_lens` is the per-row valid count, i.e. “attend packed slots [0, count)”: valid columns must form a prefix | backend:1616-1625 |
| CUDA graphs / PDL | decode graphs use the `full` backend. 2 eager warm-ups run before capture, so lazy buffers are allocated eagerly. Triton PDL is on in production | `runner_backend/full_cuda_graph_backend.py:107`, `breakable_cuda_graph_backend.py:117`; `serve-fast.sh:65` (`SGLANG_TRITON_PDL=1`) |

Index sharing is on in production. Server log, `runs/rq1/serve-A1.log` 13:07:20: “QSA MTP index sharing enabled:
draft decode steps reuse the draft-extend selection for layers [0]”.

### 1.2 Calls per step

Measured from production traces: 20 profiled greedy code-edit steps, `bench/xa1/xa_trace.py`, per kernel grid.

| trace | rows of verify + draft extend calls | draft decode calls (R1) | per step |
|---|---|---|---|
| `runs/sv/traces/A1-code-edit` (all W8) | R8: 260 = 13/step | 120 = 6/step | 19 |
| `runs/fg1/traces/A1-code-edit` (W4) | R4: 260 = 13/step | 40 = 2/step | 15 |
| `runs/rq1/traces/A1-code-edit` (12 W8 + 8 W16 steps) | R8 156 + R16 104 = 13/step | 184 = 12 × 6 + 8 × 14 | 22.2 |

So each step has 12 verify calls and 1 draft extend call at R = width, plus (draft steps − 1) draft decode calls at R1.

In-server durations (trace medians, µs):

| kernel | R1 | R4 | R8 | R16 |
|---|---|---|---|---|
| XQA | 16.9-17.3 | 16.3 | 16.1 | 18.7 |
| compact | 3.4-3.7 | 5.0 | 7.8 | 13.3 |

The count pass takes 1.2-1.25 µs at every row count.

These are 2.5-4.5 µs above the isolated XQA times in the microbenchmark (12.4-12.8 µs, 16.2 at R16, §4.3). Under PDL a
kernel's trace span includes its wait for the predecessor, so the trace durations are not additive. The wall-time
basis in this note is the chained-graph microbenchmark.

### 1.3 Index-shared draft rows break the valid-prefix contract (production bug)

**Row layout.** `QSAMTPSharedSparseIndices` (backend:155; the lookup docstring at :216-218 says “-1 (dropped
downstream)”) builds each draft decode row from three parts:
1. the 2051 frozen columns of the draft-extend (anchor) row;
2. tail_width columns [captured_len .. position];
3. −1 beyond.

The anchor part has −1 columns inside its first 2051:
- short context: everything after the anchor's L_a positions;
- long context: h = 3 − (L_a mod 4) unused uncompressed-tail columns.

**What the packed path then does.**
- valid_count = (valid anchor columns) + j at draft step j.
- XQA attends packed slots [0, valid_count). Those slots include hole columns that were never written in this call,
  and they exclude tail columns beyond the count.
- Long context: the newest min(j, h) visible positions are not attended, including the current token (whenever
  h > 0, i.e. 3 of 4 anchors). The same number of stale slots are attended instead.
- Short context: no tail column (c ≥ 2051) is packed at all. All j drafted positions, including the current token,
  are dropped, and slots [L_a, L_a + j) are stale.

**Evidence.**

1. `bench/xa1/hole_demo.py`, CPU only. It runs the production code (`QSAMTPSharedSparseIndices` CPU lookup, then
   the production valid-count and `_compact_kv` kernels in the Triton interpreter) on W8 rows. K holds position + 1,
   so an unwritten slot reads as −1. Results are in `results/xa1-hole-demo.json`:

   | anchor L_a | draft step j = 1..6 | dropped visible positions | stale slots attended |
   |---|---|---|---|
   | 1000 | position 1000..1005 | all j drafted positions [1000 .. 1000+j−1] | j |
   | 8184 (mod 4 = 0) | 8184..8189 | newest min(j, 3) | min(j, 3) |
   | 8190 (mod 4 = 2) | 8190..8195 | the current token | 1 |
   | 8191 (mod 4 = 3) | 8191..8196 | none | 0 |

2. GPU, through the real backend (`xa1-integ-gpu.json`). Max relative error vs the fp32 reference on shared rows
   (max over rows of max |out − ref| / max |ref|):

   | shared row | flag off | flag on |
   |---|---|---|
   | ctx 8192, W16 tail | 7.3e-2 | 2.2e-3 |
   | ctx 1024 | 1.1e2 (stale scratch: garbage) | 3.1e-3 |

   In the microbenchmark, the production path gave NaN on the shared rows at ctx 1024/4096 (stale scratch), and
   1.3e-2 / 9.8e-3 at ctx 8192 (`xa1-job4.log`).

3. `selftest_cpu.py` part 3 shows the same with other anchors (`results/xa1-selftest-cpu.log`).

**What the stale slots hold in the server (not measured).** Each draft step has its own backend instance
(`QwenSparseMultiStepDraftBackend`, backend:1797-1814), and each instance has its own `torch.empty` scratch
(`_get_fa2_scratch`, :1562-1579). A stale slot therefore holds one of:
- what an earlier call of the same instance packed there: MTP-layer K/V of other positions or another request;
- never-written memory.

**Which workloads are hit.** prose-en (0.5 KB), prose-ja (0.6 KB) and agent-loop (1.4 KB) prompts stay below
2048 tokens, so every draft decode step there takes the short-context pattern. code-edit (15 KB) and
longctx/doc-a-8k (35 KB) take the long-context pattern.

**Fix options.**
- (a) This flag. Validity is checked per column, with `prefix_valid=False` for index-shared rows.
- (b) Keep XQA and make shared rows a valid prefix in the lookup:
  - store the anchor's valid count n_a at capture;
  - write the tail at columns n_a + t instead of 2051 + t (the anchor's own −1 columns are already at its end).

  This is one small change to the lookup kernel, and it keeps the packed path as it is.
- (c) Pack by rank in `_compact_kv`: a prefix sum of validity per row.

Neither (b) nor (c) is implemented. (b) is the cheapest fix if the flag is not adopted.

## 2. Harness (`bench/xa1/`)

| file | role |
|---|---|
| `cases.py` | Synthetic served shapes. Top-k rows follow torch_expand_qsa_block_indices; consecutive rows/layers share a fraction `overlap` = 0.75 of their blocks. Index-shared rows follow `QSAMTPSharedSparseIndices.lookup`. Also the fp32 reference and the real server call path |
| `fi.py` | XQA through flashinfer's API with the server's arguments. It copies the production-compiled `.so` (sha256 6c52691b2691977c) into a private cache (`~/.cache/xa1-sglang`) and never builds |
| `kernels.py` | Candidates. A = Triton over the packed scratch, replacing XQA only. B = the gather kernel, replacing all three; this is the integrated kernel. Versions v1-v4 |
| `grid.py`, `bench.py` | Phases: validate, sweep, main, profile (see the `bench.py` docstring) |
| `ablate.py` | Ablated and `%globaltimer`-stamped copies of B (§4.4) |
| `precompile.py`, `selftest_cpu.py` | CPU only: precompile into the private Triton cache (`~/.cache/xa1-triton`); interpreter checks against the fp32 reference |
| `integ_check.py` | The integrated kernel. CPU: vs the reference. GPU: through the real backend, flag on vs off, graph-replay vs eager, and timing |
| `edge_cpu.py`, `hole_demo.py` | CPU only: edge rows of the integrated kernel; production packing on shared rows |
| `gpu_job.sh`, `env.sh` | Job body run under the GPU lock (the caller takes it), private caches, 540 s timeout, PDL on as in production |
| `arm_xa.sh`, `abba_xa.sh`, `xa_table.py`, `xa_trace.py`, `greedy_dump.py`, `memcheck.sh` | Server A/B (§7) |

**Timing method** (all GPU numbers):
- CUDA graphs of chained calls over 12 layers' KV (different KV per layer, like the 12 QSA layers of a step);
- warm = 24 calls (2 passes) replayed back to back;
- cold = 12 calls after zeroing a 256 MB buffer and reading another 256 MB (L2 is 128 MB);
- 25 rounds per shape, interleaved with a rotating order across paths;
- median µs per call. Some rounds are much slower on every path, XQA alone included (its p90 is 23-39 µs in 11
  of the 24 job4 shapes, against medians of 10.5-16.1), so only medians are reported;
- PDL on: `SGLANG_TRITON_PDL=1`, and XQA's default.

**Paths compared:**

| path | what it runs |
|---|---|
| `xqa` | XQA alone over a hot packed scratch, i.e. right after `_compact_kv` |
| `srv` | the production `_forward_trtllm_sparse`, all three kernels |
| `A` | candidate A alone over a hot packed scratch |
| `srvA` | srv with A in place of XQA |
| `B` | the gather kernel |

GPU jobs: 7 GPU-lock holds of 9-32 s each (start and end stamps in the local `runs/xa1/*.log`; `gpu_job.sh` caps
a hold at 540 s):
- job1-job4: one per kernel version;
- abl1 and abl3: ablations (job4 and abl3 shared one hold);
- abl2: stopped at compile with Triton OutOfResources (`nogather` at 3 × st − 2 stages needs 138 KB of shared
  memory, the limit is 99 KB); no numbers. abl3 runs `nogather` at st stages;
- integ-gpu.

## 3. Kernel (`python/sglang/srt/layers/attention/qsa/decode_attn.py` = candidate B v4)

**Layout.** Grid (splits, kv heads, rows). One program covers one contiguous column chunk for one kv head and its
12 query heads, padded to 16 MMA rows.

**Per tile of BLOCK_N columns:**
- top-k index → `req_to_token` slot → K/V rows;
- the K/V loads are unmasked 16-byte loads: invalid columns read the pool's padding slot 0 and are masked in the
  scores;
- loads are pipelined with `num_stages = 3 × kv_stages − 2`, because the pipeliner spreads stages over the
  three-level dependent chain;
- K is converted fp8 → f16 (exact);
- V is converted by `cvt.rn.f16x2.e4m3x2` inline asm at R1/R4. The asm is `is_pure=False` so it is not hoisted;
  V then reaches the MMA through smem as f16 (ldmatrix.trans) instead of byte loads;
- f16 MMAs, fp32 softmax state and accumulator; q is converted bf16 → f16.

**Splits and combine:**
- Each split stores a partial: acc/l in f16, plus lse = m + log2(l) in fp32.
- Each program then does one acq_rel atomic. The last arriver combines all splits in split order, with no smem
  and no barrier, then resets its counter to 0. The result is deterministic and safe under graph replay.
- Empty splits carry lse −1e30. A row with no valid column outputs zeros (`edge_cpu.py`).

**Validity:**
- `prefix_valid=True` (fresh indexer rows): columns at or past min(seq_len, C) are skipped.
- `prefix_valid=False` (index-shared draft rows): every column is checked.

**PDL:** `griddepcontrol.wait` before the first load, launch_dependents at the end (`sglang.kernels.triton_pdl`).

**Launch table** (job4 sweep at ctx 8192, scored on warm + cold; for the shared R1 row the sweep picked 17 × 64
with plain V, within 0.03 µs of 22 × 32):

| rows | splits × columns per tile | warps | KV stages | V conversion |
|---|---|---|---|---|
| R1, R4 | 22 × 32 | 4 | 3 | inline asm |
| R8 | 11 × 64 | 4 | 3 | plain |
| R16 | 11 × 32 | 4 | 3 | plain |
| rows > 16 (at most 32: 2 requests at W16) | max(1, 176 // rows) × 32 (5 × 32 at 32 rows) | 4 | 3 | plain (not measured on GPU) |

**Workspace:** a fixed 2.9 MB per backend instance (176 row-splits × 2 kv heads × 16 × 256 f16, plus lse and
counters), allocated lazily.

**Version history** (B at ctx 8192, warm / cold µs per call; measured, job1-job4):

| version | change | R1 | R4 | R8 | R16 |
|---|---|---|---|---|---|
| production (srv) | counts + compact + XQA | 14.58 / 17.58 | 16.53 / 19.12 | 19.69 / 22.14 | 29.77 / 31.99 |
| v1 (job1) | bf16 MMA, scalar fp8 conversion, serial combine | 14.67 / 16.56 | 15.17 / 17.76 | 17.31 / 19.13 | 28.49 / 31.06 |
| v2 (job2) | f16 MMA via packed `cvt.f16x2.e4m3x2`, 3D combine tiles | 12.66 / 15.05 | 12.86 / 16.04 | 15.19 / 18.43 | 20.63 / 23.21 |
| v3 (job3) | pipelined 3-level chain, unmasked gather, V by inline asm, in-thread combine | 10.73 / 12.99 | 11.25 / 13.83 | 12.44 / 14.84 | 18.58 / 20.65 |
| v4 (job4, integrated) | lse partials (no smem/barrier in the combine), more splits | **8.52 / 10.41** | **9.03 / 11.94** | **10.31 / 12.98** | **15.60 / 18.08** |

The production row is stable across the four jobs (14.5-14.6 / 16.5 / 19.6-19.7 / 29.7-29.9 warm).

## 4. Numbers

### 4.1 Per call, microbenchmark (job4, µs, medians; saving = B − srv)

| shape | xqa warm | srv warm / cold | B warm / cold | saving warm / cold |
|---|---|---|---|---|
| R1 ctx 512 | 11.33 | 14.06 / 16.22 | 6.38 / 7.68 | −7.7 / −8.5 |
| R1 ctx 1024 | 11.43 | 14.15 / 16.56 | 7.41 / 9.05 | −6.7 / −7.5 |
| R1 ctx 2048 | 11.14 | 14.57 / 17.59 | 8.52 / 10.28 | −6.1 / −7.3 |
| R1 ctx 4096 | 11.16 | 14.57 / 17.59 | 8.52 / 10.40 | −6.1 / −7.2 |
| R1 ctx 8192 | 11.16 | 14.58 / 17.58 | 8.52 / 10.41 | −6.1 / −7.2 |
| R4 ctx 512 | 11.25 | 14.57 / 16.42 | 6.55 / 7.85 | −8.0 / −8.6 |
| R4 ctx 1024 | 11.24 | 15.17 / 16.90 | 7.83 / 9.40 | −7.3 / −7.5 |
| R4 ctx 2048 | 11.25 | 16.54 / 18.25 | 9.03 / 10.93 | −7.5 / −7.3 |
| R4 ctx 4096 | 11.16 | 16.54 / 18.79 | 9.03 / 11.60 | −7.5 / −7.2 |
| R4 ctx 8192 | 11.18 | 16.53 / 19.12 | 9.03 / 11.94 | −7.5 / −7.2 |
| R8 ctx 512 | 11.25 | 15.34 / 16.87 | 6.21 / 7.34 | −9.1 / −9.5 |
| R8 ctx 1024 | 11.32 | 16.78 / 18.43 | 8.09 / 9.73 | −8.7 / −8.7 |
| R8 ctx 2048 | 11.50 | 19.34 / 21.34 | 10.13 / 12.11 | −9.2 / −9.2 |
| R8 ctx 4096 | 11.53 | 19.44 / 21.67 | 10.14 / 12.60 | −9.3 / −9.1 |
| R8 ctx 8192 | 11.50 | 19.69 / 22.14 | 10.31 / 12.98 | −9.4 / −9.2 |
| R16 ctx 512 | 10.46 | 16.71 / 18.46 | 8.52 / 9.88 | −8.2 / −8.6 |
| R16 ctx 1024 | 10.57 | 18.68 / 20.67 | 10.38 / 11.78 | −8.3 / −8.9 |
| R16 ctx 2048 | 15.94 | 29.06 / 30.88 | 27.28¹ / 17.22 | (−1.8¹) / −13.7 |
| R16 ctx 4096 | 16.03 | 29.25 / 31.72 | 15.60 / 17.58 | −13.7 / −14.1 |
| R16 ctx 8192 | 16.10 | 29.77 / 31.99 | 15.60 / 18.08 | −14.2 / −13.9 |
| R1 shared 2067, ctx 1024 | 11.25 | 14.15 / 16.56 | 9.03 / 9.91 | −5.1 / −6.7 |
| R1 shared 2067, ctx 4096 | 11.16 | 14.58 / 17.59 | 8.43 / 10.41 | −6.1 / −7.2 |
| R1 shared 2067, ctx 8192 | 11.16 | 14.57 / 17.58 | 8.51 / 10.41 | −6.1 / −7.2 |
| R1 shared 2055, ctx 8192 | 11.25 | 14.57 / 17.68 | 8.46 / 10.56 | −6.1 / −7.1 |

¹ An outlier: the same job's cold value is 17.22. B's 25 warm rounds at this shape were bimodal (p10 15.17,
p90 35.19) and the median fell into the slow mode. The integration check re-measured this shape at 15.94 warm vs
29.17 off (−13.2). The job4 `srvA` R16-ctx8192 warm median of 41.11 (p10 28.23) is also an outlier.

The shared rows ran B at the sweep's pick for them (17 × 64, plain V). The flag runs 22 × 32 on every R1 row;
§4.2 measures that through the backend.

**Candidate A** (Triton over the packed scratch, replacing XQA only), hot, ctx 8192:

| | R1 | R4 | R8 | R16 |
|---|---|---|---|---|
| A | 7.66 | 8.26 | 9.87 | 14.32 |
| XQA | 11.16 | 11.18 | 11.50 | 16.10 |

A is 1.6-3.5 µs faster than XQA itself. But srvA (the production path with A in place of XQA) saves only
1.5-3.8 µs warm and 2.0-4.3 cold per call at ctx ≥ 2048, against 6.1-14.2 for B. The bulk of B's saving comes
from dropping the count and compaction kernels, so A was not pursued.

### 4.2 Integrated kernel through the real backend (measured, `xa1-integ-gpu.json`)

| shape | off warm | on warm | off cold | on cold | on vs ref | off vs ref |
|---|---|---|---|---|---|---|
| R1 shared 2067, ctx 8192 | 14.65 | 8.43 | 17.74 | 10.24 | 2.2e-3 | 7.3e-2 (bug) |
| R1 shared 2067, ctx 1024 | 14.24 | 8.95 | 16.71 | 10.05 | 3.1e-3 | 1.1e2 (bug) |
| R1 ctx 8192 | 14.74 | 8.43 | 17.76 | 10.25 | 2.6e-3 | 3.3e-3 |
| R4 ctx 8192 | 16.54 | 9.03 | 19.28 | 11.78 | 2.9e-3 | 4.7e-3 |
| R8 ctx 8192 | 19.36 | 10.22 | 22.19 | 13.31 | 4.0e-3 | 4.1e-3 |
| R16 ctx 8192 | 29.97 | 16.01 | 32.09 | 18.25 | 3.3e-3 | 5.3e-3 |
| R4 ctx 1024 | 15.26 | 7.75 | 17.06 | 9.37 | 2.6e-3 | 4.8e-3 |
| R16 ctx 2048 | 29.17 | 15.94 | 31.05 | 17.39 | 3.0e-3 | 4.9e-3 |

The check's other conditions held for every shape:
- the backend is built through `__init__`, with the flag read there;
- a CUDA graph of 12 chained layer calls, replayed twice, equals eager bitwise;
- arrival counters are zero after eager and after graph runs.

CPU (interpreter, `xa1-integ-cpu.json`): all OK, max rel 3.9e-3 to 6.9e-3. This includes rows past the measured
buckets: R32 (5 splits, 6.9e-3) and R40 (4 splits, 6.9e-3). The interpreter runs another instruction path (fp32
dots, no inline-asm V conversion), so it checks indexing, masking and the combine, not GPU rounding. Why its
errors sit above the GPU's was not investigated.

### 4.3 Isolated kernel durations (job4 profile, sync around every call, ctx 8192, µs)

| kernel | R1 | R4 | R8 | R16 |
|---|---|---|---|---|
| `_fa2_valid_counts` | 1.38 | 1.39 | 1.47 | 1.47 |
| `_compact_kv` | 3.73 | 4.88 | 7.44 | 12.74 |
| XQA `kernel_mha` | 12.74 | 12.67 | 12.38 | 16.18 |

The XQA grid is [9, 2, R] for R ≤ 8 and [5, 2, 16] at R16 (production trace), so at R1 only 18 CTAs run. XQA is
latency-bound as well: its time barely moves from R1 to R8.

### 4.4 Why ≤ 5-6 µs per call is out of reach at the cap

Ablation `abl3`: R1, 2051 columns, ctx 8192, µs, `results/xa1-abl3.log`.

| B variant (v4 unless noted: 17 splits × 64 columns, 4 warps, 2 KV stages, V by inline asm) | warm | cold |
|---|---|---|
| full kernel | 8.52 | 10.89 |
| `nored`: partials + atomic kept, no combine | 5.96 | 8.35 |
| `nosplit`: no partials, atomic or combine (racy store) | 5.43 | 7.84 |
| `noloop`: no main loop, partials + atomic + combine kept | 3.91 | 4.26 |
| `noloop` at 11 splits | 3.99 | 4.26 |
| empty kernel: `noloop` + `nosplit` (v3, 11 splits) | 0.66 | 0.85 |

The flag's R1 configuration (22 × 32, 3 KV stages) gave 8.51 / 10.45 in the job4 sweep, and this 17 × 64 one
8.59 / 10.91 there.

**Timeline** (`%globaltimer` stamps, median call, v4 at 17 splits):

| part | µs |
|---|---|
| main loop | 4.61 (max over CTAs 4.80) |
| arrival atomic | 0.83 |
| last-arriver combine | 2.27 |
| kernel total | 8.03 |
| end to end | 8.42 |

**Serial tile cost** (`nosplit`; v3 for 1-11 splits, v4 for 17). Each CTA walks ceil(33 / splits) tiles of 64
columns. With 1 split (one CTA per kv head over all 33 tiles) the kernel takes 69.6 µs:

| splits | 1 | 2 | 4 | 8 | 11 | 17 |
|---|---|---|---|---|---|---|
| tiles per CTA | 33 | 17 | 9 | 5 | 3 | 2 |
| µs | 69.6 | 31.5 | 17.7 | 10.6 | 7.1 | 5.4 |
| µs per tile, after the 0.66 µs empty launch | 2.1 | 1.8 | 1.9 | 2.0 | 2.1 | 2.4 |

**The gather is cheap.** Taking slots from the column number instead of top-k → `req_to_token` (`nogather`, v3)
saves 0.3 µs at R1 (10.39 vs 10.65 warm, 11 splits) and is 1.1-1.8 µs slower at R4, R8 and R16. Serially, with
1 split, the two take the same time (70.2 vs 69.6 µs). The 3-level indirection is not where the time goes.

**Reading.**
- The split and combine machinery alone costs about 3.3 µs over an empty launch (`noloop` 3.9-4.0 vs 0.66):
  partials to L2, the acq_rel atomic, and the reload in the last arriver.
- Each tile costs a CTA about 2 µs, and a longer walk barely amortizes it (2.1 µs per tile over 33 tiles).
- More splits shorten the loop but lengthen the combine. 33 × 64 at 8 warps gives 8.43 / 10.95 in the job4
  sweep, against 8.51 / 10.45 for the flag's 22 × 32: no gain. Fewer splits lengthen the loop.
- Short rows bottom out at 6.2-6.6 µs (ctx 512, R1-R8).

Getting below about 6 µs at the cap needs a different structure: a cluster/DSMEM combine, a persistent kernel that
overlaps consecutive calls, or fusing the attention into its neighbours. None was tried.

## 5. Numerics

**Against the fp32 reference** (same fp8 K/V, fp32 math). The metric is max over rows of
max |out − ref| / max |ref|.

| path | GPU, all job4 shapes | through the backend |
|---|---|---|
| integrated kernel | 1.9e-3 .. 3.4e-3 | ≤ 4.0e-3 |
| production (shared rows excluded) | 2.7e-3 .. 5.2e-3 | ≤ 5.3e-3 |

The kernel is at least as close to the reference as production at every shape.

**Against production's bf16 output** (fresh rows; job4 and the integration check):

| measure | value |
|---|---|
| elements that differ | 43-45% |
| differ by > 1 bf16 ulp | 11-14% |
| differ by > 4 ulp | 3.5-4.8% |
| rows with a difference | every row (6144 elements per row) |
| max abs difference | 0.002-0.016 |

This is two different kernels rounding differently. Greedy outputs will therefore diverge from production at some
token in most prompts, as with any kernel change. §7 measures by how much.

**Determinism.**
- Outputs are bitwise reproducible: graph replay equals eager, and the combine order is fixed.
- Per-row results depend on the row bucket, i.e. the split layout: R1/R4 differ from R8, which differs from R16. The
  same token verified at different widths can round differently. XQA has the same property between R ≤ 8 (9 blocks)
  and R16 (5 blocks).

**Edge rows** (`edge_cpu.py`, `results/xa1-edge-cpu.log`), all with zero counters afterwards:

| row | result |
|---|---|
| seq_len 0 | zeros |
| all −1 | zeros |
| one valid column in the last split | that V row exactly |
| holes mid-row plus columns past seq_len | 4.2e-3 abs vs reference |

## 6. Integration (worktree `~/tools/sglang-xa`, branch `opus/decode-attn`)

Commit **dba52c1994** (not pushed), on 7b4d539f9b:

- **`python/sglang/srt/environ.py:323-325`** adds `SGLANG_OPT_TRITON_DECODE_ATTN = EnvBool(False)`, with a 2-line
  comment.
- **`.../layers/attention/qwen_sparse_attn_backend.py`** (+67):
  - `__init__` reads the flag once (`self._triton_decode_attn`) and sets `self._decode_attn_workspace = None`.
  - In `_forward_paged_attention`, right after `topk_indices.to(torch.int32).contiguous()`: if the flag is on and
    `qsa_decode_attention_supported(...)` holds, it returns `_forward_triton_decode(...)`. Otherwise it prints a
    one-time warning (“SGLANG_OPT_TRITON_DECODE_ATTN: unsupported QSA attention shape, falling back to the packed
    path”) and continues on the production path.
  - `_forward_triton_decode` allocates the workspace lazily. It raises if the first call happens during a capture;
    the eager warm-ups run first (§1.1).
  - It passes the same `row_req_pool_indices`, `metadata.sequence_lengths` and `layer.scaling` as production, with
    `prefix_valid = not self.should_reuse_mtp_sparse_indices(forward_batch)`. Shared draft decode rows therefore
    get the per-column check.
- **`.../layers/attention/qsa/decode_attn.py`** (new, 386 lines): the kernel, the launch table, the workspace,
  `qsa_decode_attention_supported` and `qsa_decode_attention`.

**Supported shapes:** bf16 q with head_dim 256; contiguous fp8 e4m3 K/V; GQA ≤ 16; one top-k row per q row.

**Flag off.** The only differences are two attribute assignments in `__init__` and one boolean test per call.

**Flag on.**
- The production buffers are never allocated: the 128 MB XQA workspace and the packed scratch per backend instance.
- The Triton workspace (2.9 MB per instance) is allocated instead.

**Checks done:**
- CPU import of the worktree (`integ_check.py --mode cpu`);
- the GPU check of §4.2;
- `git merge-tree` with `opus/stack` and `opus/stack-dt`: clean.

## 7. Server A/B (scripts ready; NOT run)

**Setup (done).** Copies only; the production files and the shared FlashInfer caches were not touched.

```bash
WT=~/tools/sglang-xa; P=~/tools/sglang-rtxpro6000
git -C $P worktree add -b opus/decode-attn $WT 7b4d539f9b
cp $P/serve-fast.sh $WT/
sed "s|\$REPO/.venv/bin|$P/.venv/bin|" $P/serve-local.sh > $WT/serve-local.sh && chmod +x $WT/serve-local.sh
mkdir -p $WT/.cache && cp -a $P/.cache/{jit,triton,sglang,xdg} $WT/.cache/   # 2.4 GB
bash ~/tools/flash-next-bench/bench/jitcache/jitprobe.sh $WT                 # hit=15 miss=3
```

- `serve-fast.sh` and `serve-local.sh` are untracked in the worktree on purpose.
- The 3 jitprobe misses are stale hc_combine builds whose sources are identical to production's. The FG1, DG1, DT1
  and stack worktrees report the same 15 / 3; FG1 notes the same three in sv and rs-dev, whose servers compiled
  nothing.
- Before the first start, also check that `ninja -n` reports no work on the shared FlashInfer cache. The new kernel
  is Triton only; the first B start compiles 7 variants into `$WT/.cache/triton`, in the eager runs before each
  graph capture (§9, risk 3).

| item | choice |
|---|---|
| arms | off = production QSA attention; on = `SGLANG_OPT_TRITON_DECODE_ATTN=1`. Same worktree and commit, `serve-fast.sh wa`, port 8027 |
| order | Default `ORDER="A1:off B1:on B2:on A2:off"`. Recommended 8 starts: `ORDER="A1:off B1:on B2:on A2:off B3:on A3:off A4:off B4:on"`, because one start moves ms/step by a few percent (STACK spec, opening paragraph) and the expected effect is about 1% |
| arm | Each arm is a fresh server in its own GPU-lock hold (`arm_xa.sh`, `MemoryMax=110G`, memcheck ≥ 110 GB), with 30 s between arms for other lock waiters |
| grid | code-edit, prose-en, prose-ja, agent-loop, longctx/doc-a-8k × greedy and sampling × N=4 requests × 600 tokens (`prof/fc_sampling_probe.py`). doc-a-8k puts every QSA row at the cap; the prose and agent prompts exercise the short-context hole pattern |
| primary metric | ms per output token per request; and ms/step at equal acceptance (`bench/stats/ancova_ab.py`, the arm-level line with ≥ 5 arms) |
| acceptance | tok/step (ANCOVA). The hole fix changes every draft decode step, so tok/step may move |
| deterministic evidence | Per arm, from the 20-step greedy code-edit trace (`xa_trace.py`). A arms: `triton_decode` 0, and `xqa_mha` = `compact_kv` = `valid_counts` = 13 + (draft steps − 1) per step (19/step at W8 in the sv trace). B arms: `triton_decode` = that call count, the other three 0, with grids [22, 2, 1] for draft decode, [22, 2, 4] / [11, 2, 8] / [11, 2, 16] for verify |
| server log | 0 Traceback/Error lines; 0 “SGLANG_OPT_TRITON_DECODE_ATTN: unsupported” warnings (`arm_xa.sh` greps both) |
| greedy | `greedy_dump.py`: 4 workloads × 2 repeats × 320 tokens, temperature 0, thinking off, `/flush_cache` before each. Then `prof/agree_cmp.py` over the first 256 tokens for A1-A2, B1-B2 (between-server floors), A1-B1 and A2-B2 |

Commands, from `~/tools/flash-next-bench`:

```bash
# optional smoke: one B server, no grid, no greedy dump (about 6-7 min)
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
    env GREEDY=0 bash bench/xa1/arm_xa.sh smoke-xa on 0
# the A/B: per-arm logs and probe rows in runs/xa1/, then ANCOVA, (4-arm) paired table, greedy match rates
mkdir -p runs/xa1
ORDER="A1:off B1:on B2:on A2:off B3:on A3:off A4:off B4:on" bash bench/xa1/abba_xa.sh > runs/xa1/abba.log 2>&1
```

**Decision rule.** Turn the flag on in `serve-fast.sh` only if all of these hold:
1. The deterministic evidence holds in every B arm, every A arm shows the production kernels, and there are no
   fallback warnings or errors.
2. ms/step at equal acceptance (arm-level ANCOVA): B is not slower. The upper 95% bound must be below +0.5%; the
   expected change is about −1%.
3. tok/step: no workload × mode loses more than 2% with a CI that excludes 0. A loss would point at a bug in the
   new path, since the fix only restores tokens the draft should have seen.
4. Pooled ms/token is not worse.
5. Greedy: every B dump is coherent (no repetition loops or garbage). The A–B first-divergence indices are not
   systematically earlier than the A1–A2 / B1–B2 floors. Divergence itself is expected (§5).
6. Drift |A2/A1 − 1| on ms/step stays within the 1-3% that one start moves it (DT1 spec, 14:25 status); a
   larger drift points at something else (another GPU job, clocks), so rerun.

If 1-5 hold but the speed effect stays unresolved (likely at about 1%), the call rests on rules 1, 3 and 5 plus the
microbenchmark. The hole fix is a correctness argument on its own.

**Short greedy match-rate plan.**
- `agree_cmp.py` prints the within-arm repeat floor (repeat 0 vs 1) and, per prompt, the identical-token fraction
  over 256 tokens plus the first-divergence index.
- Compare the A1–B1 and A2–B2 distributions with A1–A2 and B1–B2.
- The floors will be low. wa's greedy text is not reproducible: between two production starts 0/16 prompts kept
  the same completion, and repeats within one start differ too (STACK spec §4.0; the wa controller's width choice
  depends on CPU/GPU timing). In wa, the comparison only catches gross breakage.
- Sharper, optional: w4 with `--disable-flashinfer-autotune`, the setup of DT1's exactness check
  (`bench/stack/exact_arm.sh`: one width, no controller, untuned MoE tactics with a separate finalize kernel).
  Whether two production starts repeat there (X1 == X2) was still pending at 15:33 (`runs/stack/chain-exact.log`).
  If they do, off/on/off greedy-only arms give an exact floor, and off vs on shows where the flag first flips a
  near-tie. The flag is not bit-exact, so divergence is expected; what counts is coherent text and no
  systematically early divergence. About 6-7 min per arm:

```bash
for arm in X1:off Y1:on X2:off; do
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env PRESET=w4 SERVE_ARGS=--disable-flashinfer-autotune OUTDIR=runs/xa1/w4 \
      bash bench/xa1/arm_xa.sh ${arm%%:*} ${arm##*:} 0
  sleep 30
done
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
$PY prof/agree_cmp.py runs/xa1/w4/X1-greedy.json runs/xa1/w4/X2-greedy.json 256   # floor
$PY prof/agree_cmp.py runs/xa1/w4/X1-greedy.json runs/xa1/w4/Y1-greedy.json 256   # the flag
```

**Effort (est.).** About 7-8 minutes of GPU lock per arm. Today's rq1 ABBA arms (the same `serve-fast.sh wa` start
and grid, without doc-a-8k and the greedy dump) took 4.1-5.4 min to start and 5.7-7.0 min in all
(`runs/rq1/abba-rq2u2h.log`). doc-a-8k and the greedy dump add about a minute, and the first B start also
compiles 7 Triton variants. With the 30 s gaps that is about 35 min for 4 arms and 70 min for 8, if nothing
else waits for the lock.

## 7b. XA1 on top of ST1 (parent, 17:15)

**Why.** ST1 (`SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX`, `specs/ST1_SHARED_TAIL_2026-10-01.md`) fixes the §1.3 bug on
the packed path: it writes the drafted tail right after the anchor row's valid prefix. On top of ST1:
- both arms attend the drafted positions, so an A/B measures XA1's speed without the acceptance change of the hole
  fix (§7 has the two together);
- index-shared rows keep every valid column in front and below seq_len, so XA1 may stop at min(seq_len, columns)
  on them as well. That closes risk 2 (§9): before, these rows scanned all 2055-2067 columns.

**Worktree** `~/tools/sglang-xa-st1`, branch `opus/xa-st1`:

| commit | what |
|---|---|
| 1df3112044 | ST1 at 8 warps, on stack + DT1 (c868f2ee86) |
| 8801f24173 | XA1 (dba52c1994) cherry-picked, clean |
| 47d1b4e8ab | `prefix_valid = not should_reuse or shared_tail_prefix` in `_forward_triton_decode`, and the matching kernel comments (+11/−7, written by the parent) |

- Caches as in §7: Triton hard-linked, `jit`/`sglang`/`xdg` real copies; jitprobe hit=15 miss=3.
- `serve-fast.sh` and `serve-local.sh` are untracked.
- The comment edit inside the kernel changes its Triton cache key, so the first B start compiles the variants, as
  in §7.

**Why the bound is safe.** The kernel still masks every column by 0 ≤ pos < seq_len; `PREFIX_VALID` only moves the
end of the scan to min(seq_len, NCOLS). Under ST1 a shared row holds its k valid columns in front: the anchor's
valid prefix plus the drafted tail, with k ≤ seq_len. So no valid column is skipped, and the −1 columns between k
and the end of the scan stay masked.

**GPU check** (`bench/xa1/st1_prefix_check.py`, 17:07, 18 s under the lock; `results/xa1-st1-prefix.log`).
- Rows: `cases.Case(mode="shared")` rows (the production hole layout), compacted to the ST1 layout. ctx 300, 700,
  1500, 2100, 4000 and 8192 × (rows, tail width, drafted) = (1, 4, 2), (2, 4, 3), (1, 8, 6).
- Error vs the fp32 reference: hole layout + full scan, ST1 + full scan and ST1 + bound all agree, at 1.0e-3 to
  4.2e-3 max |err|, within rounding of each other. The bound and the full scan differ by ≤ 7.8e-3 at ctx 300/700
  (a different split, bf16 rounding) and by 0 from ctx 1500.
- Time per call (CUDA graph of 20 calls on one layer's KV, warm, median of 7):

| ctx | full scan (µs) | bound (µs) | saving (µs) |
|---|---|---|---|
| 300 | 10.6-15.1 | 6.7-7.3 | −3.7..−7.8 |
| 700 | 10.7-13.6 | 7.2-7.6 | −3.5..−6.3 |
| 1500 | 9.6-10.7 | 9.5-10.0 | −0.1..−0.6 |
| 2100-8192 | 9.5-10.1 | 9.5-10.2 | −0.4..+0.5 (noise) |

It only pays below about ctx 1000, where a row has far fewer valid columns than the cap. The full scan is slower
there than at the cap: every −1 column still loads the pool's padding slot 0.

**Per step (est.).** R1 calls per step are the draft steps − 1 (§8): 2, 6 and 14 at W4, W8 and W16, and 4.3 at the
default wa mix. At ctx ≤ 700 the bound saves a further 15-33 µs per step at the mix, about 0.1-0.3% of a prose step;
little at ctx 1500 and nothing beyond. The main XA1 saving (§8: −0.13..−0.15 ms at the mix) is unchanged.

**A/B** (`bench/xa1/abba_xa_st1.sh`, port 8033; queued after the ST1 A/B and the RS2 chain):
- A = the stack (RT1, SV1, SV2, FG1, DG1; RQ2 u2h) + ST1. B = A + `SGLANG_OPT_TRITON_DECODE_ATTN=1`.
- Both run from `~/tools/sglang-xa-st1` through `bench/stack/arm_stack.sh` (mode on, lmstudio and greedy samplings,
  4 prompts).
- 8 starts, `A1 B1 B2 A2 B3 A3 A4 B4`, each its own GPU-lock hold; about 80 min.
- Per arm:
  - ST1's log line count, > 0 in every arm since both arms have ST1;
  - XA1 fallback warnings: 0;
  - `xa_trace.py` on the greedy code-edit trace: B shows `triton_decode` only, A the production kernels.
- Then the ANCOVA per sampling mode and the clipped R.eager.

**Decision rule.** §7 rules 1, 2, 4 and 6 apply as written.
- Rule 3 tightens: both arms are correct now, so tok/step should not move beyond the numerics (§5). A loss with a
  CI that excludes 0 points at a bug.
- Rule 5 (greedy text) is dropped: greedy text is not reproducible even within one config (DT1 spec, 16:55).

## 8. Estimated ms/step at wa (est.)

Basis:
- per-call savings at the cap (ctx 4096-8192) from §4.1/§4.2: R4 7.2-7.5, R8 8.9-9.4, R16 13.7-14.2, R1 shared
  6.1-7.5 (envelope over warm, cold and both measurements);
- at short context (ctx 512-1024): R4 7.3-8.6, R8 8.7-9.5, R16 8.2-8.9, R1 shared 5.1-6.7;
- calls per step from §1.2.

| width | calls per step | saving at the cap | saving at short context |
|---|---|---|---|
| W4 (3 draft steps) | 13 × R4 + 2 × R1 | −0.106..−0.113 ms | −0.106..−0.125 ms |
| W8 (7) | 13 × R8 + 6 × R1 | −0.152..−0.167 ms | −0.144..−0.164 ms |
| W16 (15) | 13 × R16 + 14 × R1 | −0.262..−0.289 ms | −0.178..−0.209 ms |
| wa, default mix 0.67 / 0.21 / 0.12 | — | −0.134..−0.145 ms | −0.122..−0.143 ms |
| code-edit, rq1 trace mix (12 × W8 + 8 × W16 in 20 steps) | — | −0.196..−0.216 ms | — |

**Relative to a step.**
- Production wa steps in the rq1 A arms (`runs/rq1/A*-probe.jsonl`, per request, greedy and sampling): prose
  10.1-13.1 ms, agent-loop 12.2-13.7 ms, code-edit 13.3-19.0 ms. DT1 measured a 13.64 ms trace step for code-edit.
- The default mix saves 0.134-0.145 ms, about 0.7-1.4% of those steps.
- code-edit at its trace mix saves 0.196-0.216 ms, 1.0-1.6% of a code-edit step. A step at W8 throughout (the sv
  code-edit trace) saves 0.15-0.17 ms, 0.8-1.3%.
- The A/B also carries any acceptance change from the hole fix. Its size and sign are unknown, and it could exceed
  the speed effect on the prose and agent workloads.

## 9. Verified vs assumed; open risks

**Verified (measured or checked):**
- Server call shapes and arguments from the code (file:line above), the compiled XQA module name, the index-sharing
  log line, and per-step call counts and grids in production traces.
- Per-call times in CUDA graphs, warm and after an L2 flush, for all three production kernels and the candidates,
  with the production-compiled XQA module and the production Triton kernels.
- The integrated kernel through the real backend: flag on vs off, graph replay = eager, counters zero.
- Numerics vs the fp32 reference: CPU interpreter and GPU, with edge rows on CPU.
- The production hole behaviour: on CPU with the production code, and on GPU through the real backend.
- CPU import of the worktree, and a clean merge with `opus/stack` and `opus/stack-dt`.

**Assumed (not measured):**
- That the in-server saving equals the chained-graph saving. The production kernels look slower in the server trace
  (§1.2), but PDL makes trace spans non-additive, so this can go either way.
- The per-step counts × per-call savings, and the wa width mix (§8).
- The synthetic inputs: random fp8 K/V and q, with 0.75 block overlap between layers. Timing should not depend on
  the values; error statistics on real activations (peakier softmax) may differ.
- What the stale scratch slots hold in the server, and so how much the bug costs today in acceptance.

**Open risks:**
1. **bs > 1.** The server runs at most 2 requests (`max_running_requests=2` in every serve log: 10 Mamba slots,
   5 per request, X4 spec), so rows reach 32 (W16, 2 requests). Past 16 rows the launch rule is unmeasured:
   5 splits × 32 columns at 32 rows. Correct on CPU (R32, R40); speed untested. The A/B runs one request at a time.
2. **Short-context draft decode.** It scans all 2067 columns of a shared row, so it saves only 5.1-5.3 µs warm. A
   per-row bound (anchor valid count + tail) would close most of the gap. On top of ST1 the seq_len bound does
   this (§7b).
3. **Compilation.** NCOLS, NUM_SPLITS and PREFIX_VALID are constexprs, so each width and row bucket is its own
   Triton variant. The captured graphs need 7 (`runs/rq1/serve-A1.log`, 13:07:23-13:07:51): verify and draft
   extend at 16 and 32 rows (W16, bs 1-2) and at 4 and 8 rows (W4, W8, bs 1), all 2051 wide; draft decode 2067
   wide (bs 1-2), 2055 and 2059 wide (bs 1). They compile at the first B start, in the eager runs before each
   capture, into the worktree's Triton cache. With at most 2 requests (risk 1), the graph row counts (1-2 draft
   decode rows; 4, 8, 16, 32 verify and extend rows) all map to these 7. An eager call with another row count
   past 16, if the server makes one, would compile its split count on first use. Not checked in a server.
4. **Hardware.** The flag needs sm89+ (fp8 conversions). It is only measured on sm120. With the flag on, the Triton
   path is taken before the sm100/sm120 resolve.
5. **Numerics.** Outputs change, and per-row results depend on the row bucket (§5). Greedy outputs will diverge
   from production; acceptance can move either way.
6. **Speed below 6 µs.** Not reachable with this structure (§4.4). The next steps would be a cluster/DSMEM combine,
   a persistent kernel, or fusion with the neighbouring kernels; none was tried.
7. **Production bug without the flag.** If the flag is not adopted, the bug (§1.3) stays in production. Fix
   option (b) is small and independent of this kernel.
