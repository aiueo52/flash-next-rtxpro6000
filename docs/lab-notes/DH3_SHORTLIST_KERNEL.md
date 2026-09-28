# DH3 — learned draft-head shortlist kernel

Status: implemented and benchmarked as a standalone CUDA-graph kernel path; the candidate then went
on to in-server integration (DH4). Frozen policy: r=128, K=1024, selector-margin threshold
-infinity (0 % fallback).

> Publication note: every DH3 GPU measurement (kernel and graph-path timings, the cold-L2 and warm
> microbenchmarks, the fallback-branch cost, the numerical and graph-branch checks) ran with states
> and a selector derived from the author's private MTP data (DH2) as inputs. Those timings and
> results are not published. The shortlist's cost measured in the server on the public workloads
> is in `DH4_SHORTLIST_SERVER.md` (selector-to-publication span, W16 and W4, CUPTI traces of
> `bench/workloads` runs). This note keeps the contract, the design and which checks were run.

CUDA conditional-node API reference: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes

## Contract and baseline

Worktree `$HOME/tools/sglang-dh3`, branch `codex/dh3-shortlist-kernel`,
from `codex/perf-v1` at `14d4c4c985`. Production tree, launcher and venv are read-only inputs.
All GPU invocations use `flock -w 28800 $HOME/.gpu.lock`; each phase has a short external timeout,
a 6 GB process VRAM ceiling, and releases the lock at process exit.

Read: DH1/DH2 screen/evaluation/GPU scripts, DH2 frozen checkpoints and confidence
calibration, `specs/C1_LOG.md`, production `qwen4_exp_mtp.py`, `eagle_worker_v2.py`,
`eagle_draft_cuda_graph_runner.py`, `fp8.py`, `w8a16_gemv.py`, `triton_pdl.py`, and profiler driver.

The head input is the MTP final HC mixer output, width 2560. The hot head has 49152 contiguous
FP8 e4m3fn rows and per-row FP32 scales; hot row IDs map through the stored token map. The
production W8A16 GEMV rounds logits to BF16 even when its destination is FP32. DH2's teacher
instead accumulates dequantized FP8 rows in FP32. Both numerical contracts must be distinguished.
`draft_forward` executes 14 recursive forwards inside one CUDA graph at W16. C1 confidence is
separate from `topk_p` (which remains 1 on the greedy path).

DH1 planning baseline: full head 81 us, W16 step 16083 us, 14 forwards; 3.9% saving requires
`81 - 0.039*16083/14 = 36.19736 us` per optimized forward including expected fallback cost.
The input policy is DH2's frozen r128/K1024 (DH2 was an offline study on private data; no result or conclusion from it is published). The confidence fallback rule is selector top1-top2
raw BF16 score margin; DH2's frozen learned policy has threshold -infinity. The confidence
output uses DH2's calibration-only monotone 20-bin interpolation to full-head probability.
This diagnostic map is not a validated replacement for the C1 serving controller.

## Initial implementation decisions

Keep the two BF16 selector GEMVs and their intermediate BF16 rounding. A folded 49152x2560
BF16 matrix costs 251.7 MB per forward versus 13.24 MB for the factors, removes intermediate
rounding and is unlikely to meet a memory-bound budget. Fusing both stages redundantly across
vocabulary tiles similarly repeats the down projection; measure the two-stage path first.
Use deterministic approximate per-partition top-K selection (64 interleaved vocabulary partitions, top K/64
per partition) as the approximate shortlist, with no torch.topk in the runtime. Validate its
membership and winner quality on DH2's held-out recursive states before any GO verdict.
Fuse FP8 row addressing, dequantization, FP32 dot products, global winner, mapped token,
shortlist normalization, calibrated confidence and margin fallback flag using the fork's
last-CTA counter pattern. Counters reset in-kernel; storage is preallocated.
Use PDL wait/trigger and launch_pdl from the fork. A CUDA graph IF node will consume the
device condition; both skipped and executed full-head branches will be measured.

## CPU checkpoint

Five CPU contract tests passed: bounded unique IDs and deterministic ties, global top-two
survival, K512/K1024 nestedness, prefix truncation/bonus semantics, and exact budget arithmetic.
All Python files parse. The CUDA condition bridge compiled with the existing CUDA toolkit,
without modifying the environment. First GPU smoke phase is queued on the shared lock.
The microbenchmark separately measures warm graphs and 256 MiB L2-evicted graphs; eviction is
outside event timing regions. CUPTI traces preserve individual kernels and conditional branches.
Resource accounting identifies the one new NVML context after CUDA initialization (the shell
sandbox uses a PID namespace); ambiguity aborts the phase rather than guessing a process.

## GPU smoke checkpoint

K1024 CUDA graph IF false/true/false replay, hot-token mapping, in-kernel counter reset,
forced fallback versus the production BF16 GEMV, and candidate-logit/selector-score parity
were checked on a few DH2 states (private data; results not published). Deterministic
all-zero tie selection passed. The first compile attempt's FP8 masked-load integer literal was corrected to 0.0.
The sandbox could not access the GPU driver, so GPU phases use the approved host execution
path with the same lock, timeout and VRAM guard.

## Measurements (not published)

Warm and 256 MiB L2-evicted (cold) graph timings were taken for K=512 and K=1024, for the
no-fallback path and a forced full fallback, with a same-session full-head GEMV as comparator,
using CUPTI per-kernel medians and whole-graph event timing. Held-out shortlist checks and the
production fallback numerical contract (strict margin<threshold tests, comparison with DH2's FP32
teacher) were run on DH2's calibration and test splits. All of these used private inputs; none of
their numbers or conclusions is published.

Reproduction scripts (not in this repository): `bench/dh3/` in the DH3 worktree — kernels
`kernels.py`, graph IF builder `graphs.py` and `conditional.cu`, driver `microbench.py` /
`run_gpu.sh`, cold-cache protocol `coldbench.py`, replay checks `quality.py`,
`fallback_quality.py`, CPU tests `test_cpu.py`.
