# SV2: FlashInfer radix top-k in the sparse sampling verify

Status (13:42): worktree commit done; GPU micro-benchmark (200/200 correct, −32 to −38 µs per top-k call), the
verify-level GPU test (PASS; sparse verify 73-82 -> 37-39 µs on the GPU, 137-176 -> 45-60 µs back to back) and the
in-server smoke (CLEAN; 2 top-k launches per sampling step instead of 28) done. The stack ABBA judges it with the
rest (`bench/stack/chain_stack8.sh`). Shipping needs the user's go-ahead.

## 0. Summary

- **What:** SV1's sparse verify starts with the KP largest logits per draft row (fp32, V = 248320, KP = 64 for
  top_k 40). That `torch.topk` is the largest piece of the sparse verify (about 60-70 µs of its 74-82 µs GPU time,
  `runs/sv/test-gpu-1001.log` §D). SV2 swaps it for FlashInfer's radix top-k with `sorted=True,
  deterministic=True`, which sorts on the device (k ≤ 2048) instead of `torch.sort` + `gather`.
- **Switch:** `SGLANG_OPT_SPEC_SPARSE_TOPK=1` (EnvBool, default off); only acts when
  `SGLANG_OPT_SPEC_SPARSE_VERIFY=1` takes the sparse path.
- **Output:** the same values as `torch.topk` bit for bit. Tied ids may come in a different order, which the
  verify does not see: `_sparse_target_probs_kernel` reads values only (pivot, cumulative top-p), and the tree
  kernel matches drafts by id and draws in token-id order.
- **Cost:** −32 to −38 µs per call on the GPU (CUDA graph), −38 to −47 µs queued behind other work, at 4-16 rows.
  At about 13 ms per sampling step that is about −0.3%; greedy is untouched.
- **Constraint:** FlashInfer JIT-builds its `topk` module on first use. Use SV2 only with a private FlashInfer
  cache that already holds it (the rq2u2h cache does; load it with `FLASHINFER_P2_NO_NINJA=1`). The shared cache's
  `topk` build has pending ninja work, so the stock environment would compile into it.

## 1. Change (`~/tools/sglang-sv2`, branch `opus/sparse-topk`, b92809af70; stack 73e379d725)

| file | change |
|---|---|
| `kernels/ops/speculative/sparse_verify.py` | `sparse_target_probs(..., use_flashinfer_topk)`: `flashinfer.top_k(input, k=kp, sorted=True, deterministic=True)` or `torch.topk` |
| `srt/environ.py` | `SGLANG_OPT_SPEC_SPARSE_TOPK = EnvBool(False)` next to SV1's flag |
| `srt/speculative/spec_utils.py` | `SPEC_SPARSE_TOPK` read once at import |
| `srt/speculative/eagle_utils.py` | the one caller passes it (keyword arguments throughout) |

## 2. Candidates (`bench/sv2/topk_bench.py`, `runs/sv2/topk1.jsonl`, 12:58-13:02)

fp32 rows of V = 248320, k 64 and 128, 1-32 rows, randn / peaked / ties / masked (300 finite entries).
Correctness against `torch.topk`: values equal (sorted), `gather(idx) == values`, ids unique, no id above the
k-th value missing. **200/200 cases pass**, all five variants.

GPU µs per call at k = 64 (median of 7 rounds of 20 calls; graph = CUDA-graph replay, queued = eager calls
queued behind a sleep; randn rows, peaked within 1 µs):

| rows | torch.topk | the implementer's Triton | FI sorted | **FI sorted+det** | FI unsorted |
|--:|--:|--:|--:|--:|--:|
| 1 | 55.5 / 71.0 | 47.0 / 60.5 | 30.2 / 44.3 | **27.5 / 41.1** | 22.5 / 36.0 |
| 4 | 63.6 / 89.6 | 57.2 / 71.2 | 31.8 / 45.2 | **28.0 / 42.2** | 23.1 / 37.2 |
| 8 | 60.9 / 79.2 | 70.0 / 83.0 | 33.5 / 49.5 | **28.9 / 40.9** | 23.8 / 38.5 |
| 16 | 68.6 / 84.8 | 130.2 / 145.4 | 36.8 / 49.9 | **30.4 / 47.4** | 25.2 / 42.0 |
| 32 | 90.0 / 109.4 | 201.0 / 219.9 | 57.4 / 74.9 | **51.8 / 69.0** | 46.0 / 63.0 |

- The server's rows are bs × draft tokens = 4, 8 or 16 at bs 1 (wa widths 3/7/15).
- **the implementer's Triton `sparse_topk` (rejected):** correct, but slower than `torch.topk` from 8 rows on (2× at 16
  rows; one program per row scans the whole row). Kept as `bench/sv2/alt_triton_topk.patch` (on 4f9cf50619)
  with its tests (`check_topk_cpu.py`, `alt_gpu_check.py`).
- **FI unsorted** is 5 µs faster still but needs a sort before `_sparse_target_probs_kernel` (pivot and cumulative
  top-p read sorted values). `torch.sort` + `gather` cost 8-9 µs (FI sorted, which is unsorted plus those two, minus
  FI unsorted), more than the 5 µs. Sorting inside the Triton kernel is possible (KP ≤ 256) but not done.
- **sorted+det beats sorted:** with `deterministic=True` and k ≤ 2048, FlashInfer sorts on the device; without it,
  the wrapper calls `torch.sort` + `gather` (`flashinfer/topk.py`). Deterministic mode also makes tie order
  reproducible run to run.
- k = 128 (top_k 57-120): within 1 µs of k = 64 for FI; the implementer's kernel grows to 103-283 µs.
- The sync column (a host sync after every call, `runs/sv2/topk1.jsonl`) is dominated by launch cost and the
  machine's load (load average 32 from another job) and is not used.

## 3. Verify-level and in-server checks

- CPU (Triton interpreter), `bench/sv/test_sparse_verify.py` on the sv2 worktree, torch path (the refactor):
  PASS (13:13).
- GPU (13:33-13:35), the same test with `TOPK=fi` and `TOPK=torch`, the real dense kernels as the reference
  (`runs/sv2/test-cuda-{fi,torch}.log`): **PASS both** (kept sets and values, tree walk and final draw with equal
  coins, first-token distribution). §D, µs per verify (sparse path; dense 326-397 for reference):

| bs x S | GPU, queued: torch -> FI | back to back (launches exposed): torch -> FI |
|---|--:|--:|
| 1 x 4 | 72.7 -> **36.9** | 148.7 -> **45.3** |
| 1 x 8 | 73.7 -> **36.9** | 175.8 -> **59.5** |
| 1 x 16 | 81.9 -> **38.9** | 137.3 -> **48.2** |
| 4 x 4 | 81.9 -> **38.9** | 153.5 -> **40.4** |

  The sparse verify halves on the GPU (−36..−43 µs) and drops by 89-116 µs back to back, so the eager sampling
  block should also save host time when it is launch-bound. The smoke below shows why: 28 launches become 2.
- In-server (13:35-13:41, `runs/stack/smoke-stack2.log`): the stacked smoke with SV2, RQ2 u2h and min_p on:
  **CLEAN**, 0 error lines. Top-k kernels per profiled sampling request (20 steps, code-edit and prose-en alike):

| smoke | top-k kernels per step |
|---|---|
| SV1 only (`smoke-stack`, 12:29) | **28**: 27 `at::native::mbtopk` (12 BlockIdxToKey, 4 each of computeBlockDigitCounts, computeBlockwiseWithinKCounts and computeDigitCumSum, 1 each of computeBlockwiseKthCounts, fill and gatherTopK) and the sort `at::native::warpMergeSortKVInPlace` |
| SV1 + SV2 (`smoke-stack2`) | **2**: `flashinfer::sampling::RadixTopKKernel`, `StableSortTopKByValueKernel`; no torch top-k kernels left |

  SV2 removes 26 launches from every sampling step's eager block.
- `bench/stack/arm_stack.sh` refuses `SGLANG_OPT_SPEC_SPARSE_TOPK=1` without `STACK_FI` (the cache constraint).

## 4. Open

- The R.eager share: SV2 saves GPU time inside the eager sampling block; whether the step shortens by the same
  amount depends on whether the GPU or the host bounds that block. The stack ABBA's clipped R.eager shows it.
