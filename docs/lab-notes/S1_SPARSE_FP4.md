# S1 — sparse block-scaled FP4 expert feasibility

Date: 2026-09-08. Status: COMPLETE — NO-GO for a grouped-kernel project on the untrained format.

Worktree: `$HOME/tools/sglang-s1`, branch `codex/s1-sparse-fp4`,
base `codex/perf-v1` at `446c801189`. No commits or pushes.
Production checkout, launcher, venv and checkpoint are read-only.
Build checkout: `$HOME/tools/cutlass-s1`, detached at
`b46b16d003484063bca4ed365e44095c4c6ed633` (exact FlashInfer 0.6.17 submodule,
resolved from FlashInfer commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`).
Toolkit: launcher's `$HOME/tools/mamba/envs/cuda13/bin/nvcc`, CUDA 13.3.73.
The production venv has no standalone `bin/nvcc`; its captured launcher sets
`CUDA_HOME` to that cuda13 environment, `CUDACXX=$CUDA_HOME/bin/nvcc`, and places
that toolkit immediately after the venv on PATH. The builds follow that
venv-launcher compiler selection without installing or changing the venv.

## Execution environment

The default command sandbox hides `/dev/nvidia*` and uses a separate PID
namespace. An authorized host read-only preflight resolved the initial apparent
GPU failure: RTX PRO 6000 Blackwell Max-Q, 97,887 MiB total, 8,204 MiB free while
another task's server was active. No driver reset or other-owner process kill.
The obsolete S1 sandbox preflight waiter alone was terminated by its recorded
process group. All measured host server and primitive jobs used per-run
`flock -w 28800`; the final primitive queue completed with exit status 0.

After 26 primitive rows, two attempts at dense down M8/D52 stopped at the
compute-owner preflight after acquiring the lock. No GEMM ran in either
attempt and no foreign process was killed. Both failed preflights are preserved
in `cutlass-s1/results-preflight/`; they are excluded from timing tables.
On the second abort, the recorded PIDs had already exited by the follow-up
read-only check, and a different server was then visible. A lock handoff versus
GPU cleanup delay is plausible, but the exact external cause is not established.
The primitive wrapper now allows up to 60 seconds of post-lock owner-exit grace,
and starts a benchmark only after compute-apps is empty; otherwise it aborts.
The six remaining conditions completed through `cutlass-s1/run_remaining.sh`,
each under a separate 28800-second flock wait. Benchmark binaries and timing
scope are unchanged; preflight grace is outside the timed region.

## Part A design

Flag `SGLANG_S1_SPARSE_FP4=off|all|down|scale-aware` in the worktree only.
Mask raw packed NVFP4 weights after w13 deinterleaving, before backend swizzling.
Keep two adjacent K pairs in every eight elements, using pair squared magnitude
(sum of the two squared E2M1 magnitudes), with deterministic earlier-pair ties.
This is the minimum removed L2 weight energy among the six legal pair choices.
Retained nibbles and all original scale tensors remain bit-identical; dropped
pairs become positive zero. Dense FlashInfer CUTLASS still executes. All routed
experts, including draft experts once online quantized, are in scope; shared
experts and other linear layers are excluded by the MoE-only hook.

The real checkpoint has gate/up `[640,1280]` uint8 + `[640,160]` E4M3 scales;
down `[2560,320]` uint8 + `[2560,40]` scales. Each scale covers 16 consecutive
K elements. Each 8-element mask group lies entirely inside one scale block.
Therefore `|value*scale|` pair ranking is identical to magnitude ranking for
finite positive scales. Global scales are also common within each group.
The scale-aware arm is an algebraically equivalent control, not a distinct
recovery mechanism. Zero-scale groups use magnitude as a deterministic tie-break.

Server protocol: fixed W4, `BS=1`, `CONC=1`, max running requests 1,
mem fraction 0.920, `MAX_TOTAL_TOKENS=131072`, `MAMBA_SLOTS=10`, `SERVE_DISPLAY_HZ=`; require >=4096 MiB VRAM free once ready.
Each server/benchmark run holds its own `flock -w 28800 $HOME/.gpu.lock`.
No other owner's process is killed; owned server groups are terminated on exit.
Quality follows `bench/quality/run_all.sh` semantics: pre/post needles at
18.5k and depths 0.1/0.5/0.9, smoke, complete fixed GSM8K/MMLU/HumanEval/JCQA
subsets and five sanity generations (prompts in `bench/bench/quality/run_bench.py`). Original Q1 results are not a BS=1 baseline.

Measured results are maintained in the live measurement tables below.

Acceptance is completion tokens divided by verify count, pooled across three
measured repeats after one warmup per workload. Decode t/s is the median of
the three server `meta_info.decode_throughput` values; bytewise cumulative-SSE
client timing is retained in raw logs but is not the reported throughput.
Dense kernels run in every Part A arm. Differences in generated text, stopping
length and speculative agreement can still change t/s. Higher acceptance on
a repetitive output is not a quality gain or evidence of a sparse-kernel win.
The three measured repetitions are not bit-reproducible in this serving
configuration: even prod has three different texts for each prose and agent
workload. Sparse-all and scale-aware also differ in observed generations,
despite the algebraically identical mask rule. This is recorded in
`bench_s1/results/s1-20260908-b/acceptance-repeat-variation.json`; their metric
difference must not be attributed to an improved scale-aware ranking. The
3% screen applies to pooled tokens/verify per workload, not to a claim of a
confidence-bounded acceptance guarantee from three repetitions.

Non-inferiority must test **candidate minus prod**. The current corrected Q1
`analyze.py` tests BASE minus other, so run it with each candidate as BASE and
prod as comparator; blindly using prod as BASE would reverse this experiment's
question. Margin 0.5 pp, paired one-sided 95% bounds: PASS if lower > -0.5;
FAIL if upper < -0.5; otherwise INCONCLUSIVE. The bounds are Tango score
bounds (a Wald interval collapses with few or zero discordances).
A >3% acceptance loss or any NI FAIL kills the untrained format. INCONCLUSIVE
is not a quality pass. N1's actual server A/B, not an offline proxy, is the judge.

## Part B design and arithmetic

Use weights as sparse operand A: CUTLASS `(m,n,k)=(N,M,K)`, computing `W X^T`.
Logical production shapes: `M in {1,4,8,16}`, gate/up `(N,K)=(1280,2560)`,
down `(2560,640)`. Rotate D=19/52 calls through enough disjoint expert banks
that the **compressed** values + metadata + weight scales exceed 2 GiB.
19/52 experts alone fit in or near L2 and are not a DRAM streaming test.

Upstream 80b uses NVFP4 output and its `verify()` compares reference D to itself.
A valid feasibility harness must fix verification and match dense/sparse
operand orientation, output dtype, scale layouts, timing scope and padding.
No grouped kernel implementation is authorized by a primitive PASS alone.

Ideal, unpadded expert bytes:

| projection | elements | dense values | dense scales | sparse values | sparse metadata | sparse scales | total dense | total sparse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate+up | 3,276,800 | 1,638,400 | 204,800 | 819,200 | 204,800 | 204,800 | 1,843,200 | 1,228,800 |
| down | 1,638,400 | 819,200 | 102,400 | 409,600 | 102,400 | 102,400 | 921,600 | 614,400 |
| total | 4,915,200 | 2,457,600 | 307,200 | 1,228,800 | 307,200 | 307,200 | 2,764,800 | 1,843,200 |

Dense = 4.5 bits/weight; sparse = 2 values + 0.5 metadata + 0.5 scales = 3 bits.
Saving = 921,600 bytes/expert (33.33%), excluding the unchanged 24 scalar bytes.
At 48 target layers: D19 saves 840,499,200 bytes/verify; D52 saves 2,300,313,600.
Down-only saves one third of those bytes (11.11% of all expert bytes).
The byte/time projection counts the 48 target layers only. The additional MTP
draft MoE is masked in Part A, but no extra draft-layer byte or time saving is
credited to this historical target-layer projection.
Finding 7's historical variable-term projection is 438 us W4 / 1,219 us W16,
about +5.2% / +8.2% ideal throughput; these are **not S1 measurements**.
Down-only analog: 146 us / 406.3 us, about +1.7% / +2.6% with unchanged acceptance.
Using the review's historical step times 8,919 / 16,083 us, a 3% throughput
increase requires saving at least `T*0.03/1.03` = 259.78 / 468.44 us. Therefore
the full format has **178.22 / 750.56 us** of scheduler/packing/other overhead
budget at unchanged acceptance. Down-only's 146 / 406.33 us cannot clear 3%
even at zero added overhead. These are planning budgets, not newly measured
server savings. With acceptance ratio r, throughput ratio is
`r*T/(T-saving+overhead)`; numerical loss directly spends the speedup budget.

Real layout padding, launch, metadata and scheduler overhead must be counted.
Primitive saving estimate: `48*D*((t_dense_gate+t_dense_down)-(t_sparse_gate+t_sparse_down))`.
This is an ungrouped projection, not proof of grouped performance.

## Verdict

**NO-GO without recovery training for both all-projection and down-only
sparsification.** Completed BS=1 server A/B fails non-inferiority at 0.5 pp on
all four benchmarks for sparse-all, and on GSM8K/MMLU for sparse-down.
Sparse-down HumanEval/JCQA are INCONCLUSIVE, not passes. The scale-aware server
control is complete, with no pooled acceptance kill and 6/6 needle passes.
The full 32-condition primitive sweep also completed: native sparse is slower
in every matched comparison, by 11.80–12.45% for gate/up and 17.87–18.47% for
down. This is evidence against a speed benefit from these upstream ungrouped
primitives at the tested shapes, not a measurement of a future grouped kernel.
Recovery training could change the numerical result; none was run. The
numerical gate blocks proceeding to a grouped-kernel project on this untrained
format, independently of the primitive result. A future recovery attempt would
also need to address the native block32 weight and activation scale conversion.

Final evidence audit PASS: four server arms, three full BS=1 quality batteries
(12,006 scored questions, zero transport errors), 24/24 needles, 32 primitive
measurements and 96 GPU reference checks. Minimum sampled steady free VRAM
was 11,075 MiB across all arms, above the 4 GiB requirement. Every primitive
ring contained 2,151,444,480–2,204,467,200 distinct stored weight bytes including
metadata/scales, exceeding 2 GiB. The audit also verifies identical launcher
snapshots/configuration except the transform flag, batch size 1, primitive
binary hashes, ring arithmetic, and unchanged quality scripts/data.

Initial label `s1-20260908-a` was a failed client attempt: the server reached
readiness with 4,418 MiB free, but Transformers returned a BatchEncoding that
could not be serialized before the first generation request. The owned server
group was cleaned up and lock released. `return_dict=False` plus flat-int-list
validation was CPU-checked on all four prompts; measured runs use label
`s1-20260908-b`. The smaller fixed cache/state pools above apply equally to all
measured arms and provide additional VRAM reserve.

## CPU validation and build results

- Four invariant tests PASS: independent NumPy reference, all byte codes/signs/
  deterministic ties, off/down scope and repeated-hook rejection, scale
  invariance, invalid shapes/scales rejection.
- 27 real tensors sampled: layers 0/23/47, experts 0/127/511, gate/up/down;
  44,236,800 weights. Legal paired masks, unchanged retained bytes and scales,
  and bit-identical scale-aware masks verified. Relative removed-weight RMS
  ranges 0.429909–0.451187. This is **not** a measured quality or acceptance loss.
- Original HumanEval sandbox preflight PASS for a known-correct program.
- Built original upstream 80b sparse NVFP4-output and 79a dense BF16-output
  examples, plus S1 sparse/dense BF16-output probes. CUDA 13.3.73, system g++ 13.3,
  explicit `-gencode=arch=compute_120a,code=sm_120a`. Using nvcc's default conda
  host compiler produced incompatible type-trait errors; plain `-arch=sm_120a`
  also emitted generic sm_120 PTX rejected for sparse block-scaled MMA. The
  explicit host compiler and architecture-specific code target resolve both.
- 16 CPU-only layout probes execute successfully without a CUDA device.
- Sparse reference corrected to compare actual D to reference D and to use
  the **dense** logical layout when reading decompressed reference A.
  Each measured primitive row requires three GPU reference checks (initial,
  first rotated slot and last rotated slot) to pass before emitting a result.

## Additional feasibility constraint: native sparse scaling differs

The exact upstream builder chooses `SfVectorSize=32` for sparse `nv_float4_t`,
versus 16 for dense. Its MMA wrapper asserts VS==32 for E2M1/UE4M3 sparse MMA.
Consequently the sparse primitive consumes one weight scale per **32 logical K
values**, not one per 16 as in the existing checkpoint and the Part A mask.
This also changes B/activation scale granularity. It cannot directly preserve
all the original scale boundaries by simply compressing the masked checkpoint.

Source: `include/cutlass/gemm/collective/builders/sm1xx_common.inl:468`,
`include/cute/arch/mma_sm120_sparse.hpp:3403`, at the pinned upstream revision.
[Upstream builder](https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/gemm/collective/builders/sm1xx_common.inl#L468).

CPU checkpoint confirmation: 1,238,355 of 1,382,400 adjacent scale pairs differ
(89.5801%) across the same 27 tensor sample. This is not a hypothetical mismatch
that can be removed by aliasing identical scales. Raw counts are in
`bench_s1/results/scale32_compatibility.json`.

The CPU-instantiated native layouts report:

| projection | format | value B | metadata B | scale B | total B/expert |
|---|---|---:|---:|---:|---:|
| gate/up | dense, block16 | 1,638,400 | 0 | 204,800 | 1,843,200 |
| gate/up | sparse, block32 | 819,200 | 204,800 | 102,400 | 1,126,400 |
| down | dense, block16 | 819,200 | 0 | 102,400 | 921,600 |
| down | sparse, block32 | 409,600 | 122,880 | 51,200 | 583,680 |
| both | dense | 2,457,600 | 0 | 307,200 | 2,764,800 |
| both | native sparse | 1,228,800 | 327,680 | 153,600 | 1,710,080 |

The down metadata pads logical K640 to K768 (128 metadata-alignment rows on N,
256 logical K); thus 122,880 B rather than the ideal 102,400 B. Native sparse
saves 1,054,720 B/expert = 38.15%, but that **includes a different scaling
format**. It is not the byte count of Part A's scale-preserving format.
The original ~33% calculation above remains a hypothetical scale-preserving
format calculation, not the demonstrated upstream implementation.

At 48 target layers, the corresponding streamed expert-byte arithmetic is:

| D | dense B | hypothetical block16 sparse B | native block32 sparse B | native bytes saved |
|---|---:|---:|---:|---:|
| 19 | 2,521,497,600 | 1,680,998,400 | 1,559,592,960 | 961,904,640 |
| 52 | 6,900,940,800 | 4,600,627,200 | 4,268,359,680 | 2,632,581,120 |

These byte savings require the different native scaling format; they do not
establish either numerical equivalence or a positive latency saving.

Any future attempt that recovers Part A quality and improves primitive speed
would still need an additional numerical screen of conversion to native sparse
block32 weights **and activations** before a grouped project could claim the
same numerical behavior.
Do not silently requantize during Part A; that would conflate the requested
sparsification-only gate with a second source of error.

## Small-M tactic check

The main primitive table uses **dense tile 128x32x128** and the supported
**sparse tile 128x64x256**, both with weights on A and BF16 output. The dense
small-N tile matches the serving-side orientation/tile width much better than
upstream's default 128x128x128. Merely comparing the two default examples could
repeat the tactic mistake documented in MOE_SMALLM_SPEC section 8.1.

A CPU compilation probe of sparse tile 128x32x256 fails in the pinned upstream
mainloop at `sm120_blockscaled_sparse_mma_tma.hpp:1098`:
`static_assert(decltype(size<1>(tCrSFB) == size<2>(accum))::value)`.
Dense N32 and sparse N64 build successfully. The sparse N64 instantiation
reduces the oversized default tile without changing the mainloop. The main
sweep uses N64; the original two sparse N128 M1/D19 rows are preserved in
`results-default/sparse-n128-{gate,down}-M1-D19.log`, with N64 backfills in the
main results. This is a limitation of the tested sparse N32 upstream
instantiation, not a proof of a hardware minimum tile size. No sparse mainloop
or grouped kernel is modified to repair it in S1.

The first default-tile diagnostic is preserved separately in `results-default`:
M1/D19 gate+up, dense N128 = 14.935 us / 123.411 GB/s; sparse N128 = 11.784 us /
95.588 GB/s. Both rotated >2 GiB and passed three references. Its apparent
21.1% improvement is **not** the main comparison against small-M dense N32.

Single-expert calls expose only 10 gate/up or 20 down tiles on 188 SMs. Thus
`48*D*delta_time` is the sum of **ungrouped** calls, not a prediction that a
production grouped kernel will save that absolute amount; expert grouping
fills waves and amortizes launch overhead. Finding 7's variable-term byte
budget is still the relevant budget for a future grouped implementation.

## Reproduction and raw artifacts

- Transform/tests/server harness: `$HOME/tools/sglang-s1/bench_s1/`.
- CPU tensor results: `bench_s1/results/checkpoint_mask.json`.
- Server queue: `LABEL=s1-20260908-b bench_s1/run_all.sh`; private copies of
  launcher text plus PYTHONPATH worktree overlay, all caches outside production.
- Per-request native `/generate` records exact completion_tokens/spec_verify_ct;
  warm-up excluded, 3 repeats per code-edit/prose-en/prose-ja/agent-loop.
  Reported decode t/s is the per-request server `meta_info.decode_throughput`,
  median over three repeats. Raw client streaming throughput is also saved,
  but native cumulative SSE plus bytewise reading can bottleneck the client
  (first code-edit example: server 383.9 t/s versus client 203.2 t/s), so it
  is not used as the kernel/serving-speed comparison. This choice applies to
  every arm using the same client protocol and all raw server timestamps.
- Quality rows use `bench/quality/runs/s1-20260908-b-{prod,sparse-all,sparse-down}`.
- Build recipe: `$HOME/tools/cutlass-s1/build_s1.sh`; `build_n32.sh`
  produces the dense N32 binary and records the expected sparse N32 compile failure.
  `bash build_n64.sh` builds the supported sparse N64 binary.
- Derived-source recipe: `cutlass-s1/prepare_s1.py`; original upstream files unchanged.
- Primitive queue: `cutlass-s1/run_all.sh`, raw `cutlass-s1/results/*.log`.
- Final report regeneration: `python3 bench_s1/report.py` in the S1 worktree.
- Strict CPU-only evidence audit: `python3 bench_s1/audit_run.py`; completed
  output: `bench_s1/results/s1-20260908-b/evidence-audit.json`.
- Primitive timing: CUDA event duration across 20 replays of a graph containing
  all expert banks; each bank contains D=19 or 52 separate GEMM launches. The
  ring exceeds 2 GiB even in compressed storage. Every slot has independent
  randomized values and separate scales/metadata addresses. Initialization,
  compression, copies, and three host-reference checks are outside timing.
  Reported bytes/s is effective weight bytes/s including allocated padding,
  **not a hardware DRAM counter**. Activation/output buffers are shared and
  reused across the weight ring; their traffic is excluded from the numerator.
  For scale, the M16 down BF16 output alone is 81,920 B/call (14.0% of sparse
  weight bytes), so the reported rate should not be read as total device
  bandwidth. Timing includes inter-kernel graph scheduling gaps.

<!-- S1-LIVE-BEGIN -->
## Measured results

Run label: `s1-20260908-b`. Missing entries are not passes.

| W4 arm | workload | tokens/verify | acceptance change | server decode t/s | acceptance gate |
|---|---|---:|---:|---:|---|
| prod | code-edit | 3.9344 | +0.00% | 383.92 | baseline |
| prod | prose-en | 2.2857 | +0.00% | 252.91 | baseline |
| prod | prose-ja | 2.1594 | +0.00% | 230.85 | baseline |
| prod | agent-loop | 3.3492 | +0.00% | 358.53 | baseline |
| sparse-all | code-edit | 3.8359 | -2.50% | 387.02 | within 3% screen |
| sparse-all | prose-en | 3.5294 | +54.41% | 359.18 | within 3% screen |
| sparse-all | prose-ja | 3.6269 | +67.96% | 398.38 | within 3% screen |
| sparse-all | agent-loop | 3.5556 | +6.16% | 379.52 | within 3% screen |
| sparse-down | code-edit | 3.9301 | -0.11% | 386.11 | within 3% screen |
| sparse-down | prose-en | 2.3033 | +0.77% | 250.88 | within 3% screen |
| sparse-down | prose-ja | 2.3960 | +10.96% | 258.11 | within 3% screen |
| sparse-down | agent-loop | 3.6364 | +8.58% | 386.97 | within 3% screen |
| scale-aware | code-edit | 3.8877 | -1.19% | 389.29 | within 3% screen |
| scale-aware | prose-en | 3.3426 | +46.24% | 347.00 | within 3% screen |
| scale-aware | prose-ja | 3.4545 | +59.98% | 372.74 | within 3% screen |
| scale-aware | agent-loop | 3.6600 | +9.28% | 387.98 | within 3% screen |

| arm | pre/post needle | minimum steady free VRAM (MiB) | quality completion |
|---|---|---:|---|
| prod | 6/6 passed | 11075 | Full battery complete |
| sparse-all | 6/6 passed | 11237 | Full battery complete |
| sparse-down | 6/6 passed | 11132 | Full battery complete |
| scale-aware | 6/6 passed | 11236 | Equivalent-mask acceptance/needle control only |

| quality benchmark | prod | sparse-all | sparse-down |
|---|---:|---:|---:|
| gsm8k | 1166/1319 (88.40%) | 1119/1319 (84.84%) | 1138/1319 (86.28%) |
| mmlu | 1189/1400 (84.93%) | 945/1400 (67.50%) | 1149/1400 (82.07%) |
| humaneval | 158/164 (96.34%) | 147/164 (89.63%) | 155/164 (94.51%) |
| jcqa | 1088/1119 (97.23%) | 1068/1119 (95.44%) | 1080/1119 (96.51%) |

| quality benchmark | arm | mean generated tokens | truncated | transport errors |
|---|---|---:|---:|---:|
| gsm8k | prod | 345.4 | 155 | 0 |
| mmlu | prod | 2.1 | 12 | 0 |
| humaneval | prod | 230.4 | 2 | 0 |
| jcqa | prod | 2.0 | 0 | 0 |
| gsm8k | sparse-all | 310.6 | 183 | 0 |
| mmlu | sparse-all | 4.1 | 212 | 0 |
| humaneval | sparse-all | 264.8 | 10 | 0 |
| jcqa | sparse-all | 2.0 | 0 | 0 |
| gsm8k | sparse-down | 350.0 | 197 | 0 |
| mmlu | sparse-down | 2.0 | 2 | 0 |
| humaneval | sparse-down | 248.2 | 4 | 0 |
| jcqa | sparse-down | 2.0 | 0 | 0 |

| sanity workload | arm | generated tokens | unique word ratio | repeated 4-grams |
|---|---|---:|---:|---:|
| prose-en | prod | 1500 | 0.476 | 0.03 |
| prose-ja | prod | 1091 | 1.0 | 0.053 |
| code-edit | prod | 1500 | 0.211 | 0.763 |
| agent-loop | prod | 401 | 0.579 | 0.454 |
| essay | prod | 1500 | 0.54 | 0.008 |
| prose-en | sparse-all | 1500 | 0.063 | 0.919 |
| prose-ja | sparse-all | 103 | 1.0 | 0.0 |
| code-edit | sparse-all | 1500 | 0.23 | 0.757 |
| agent-loop | sparse-all | 160 | 0.873 | 0.109 |
| essay | sparse-all | 1500 | 0.314 | 0.284 |
| prose-en | sparse-down | 1500 | 0.44 | 0.049 |
| prose-ja | sparse-down | 748 | 1.0 | 0.1 |
| code-edit | sparse-down | 1500 | 0.211 | 0.763 |
| agent-loop | sparse-down | 77 | 1.0 | 0.029 |
| essay | sparse-down | 1500 | 0.475 | 0.012 |

Non-inferiority of s1-20260908-b-sparse-all vs each other arm (paired BASE minus other; Tango score bounds, z=1.644854; recomputed from the raw paired runs, replacing the original Wald bounds with no verdict change).
Lower and upper bounds are each one-sided 95% bounds (not a two-sided 95% interval). PASS: lower > -margin; FAIL: upper < -margin; otherwise INCONCLUSIVE.
Tango score bounds stay valid with few or zero discordant pairs.

| benchmark | arm | BASE - other (pp) | lower 95% (pp) | upper 95% (pp) | margin (pp) | status | n |
|---|---|---|---|---|---|---|---|
| gsm8k | s1-20260908-b-prod | -3.56 | -5.22 | -1.94 | 0.5 | FAIL | 1319 |
| mmlu | s1-20260908-b-prod | -17.43 | -19.27 | -15.68 | 0.5 | FAIL | 1400 |
| humaneval | s1-20260908-b-prod | -6.71 | -10.88 | -3.54 | 0.5 | FAIL | 164 |
| jcqa | s1-20260908-b-prod | -1.79 | -2.70 | -1.00 | 0.5 | FAIL | 1119 |


Non-inferiority of s1-20260908-b-sparse-down vs each other arm (paired BASE minus other; Tango score bounds, z=1.644854; recomputed from the raw paired runs, replacing the original Wald bounds with no verdict change).
Lower and upper bounds are each one-sided 95% bounds (not a two-sided 95% interval). PASS: lower > -margin; FAIL: upper < -margin; otherwise INCONCLUSIVE.
Tango score bounds stay valid with few or zero discordant pairs.

| benchmark | arm | BASE - other (pp) | lower 95% (pp) | upper 95% (pp) | margin (pp) | status | n |
|---|---|---|---|---|---|---|---|
| gsm8k | s1-20260908-b-prod | -2.12 | -3.46 | -0.82 | 0.5 | FAIL | 1319 |
| mmlu | s1-20260908-b-prod | -2.86 | -4.17 | -1.59 | 0.5 | FAIL | 1400 |
| humaneval | s1-20260908-b-prod | -1.83 | -4.49 | -0.18 | 0.5 | INCONCLUSIVE | 164 |
| jcqa | s1-20260908-b-prod | -0.71 | -1.39 | -0.14 | 0.5 | INCONCLUSIVE | 1119 |


| projection | M | D | dense us/expert | sparse us/expert | sparse time change | dense GB/s | sparse GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate+up | 1 | 19 | 7.713 | 8.643 | +12.05% | 238.977 | 130.332 |
| down | 1 | 19 | 4.562 | 5.404 | +18.47% | 202.034 | 108.005 |
| gate+up | 4 | 19 | 7.686 | 8.642 | +12.45% | 239.826 | 130.339 |
| down | 4 | 19 | 4.563 | 5.388 | +18.08% | 201.966 | 108.326 |
| gate+up | 8 | 19 | 7.733 | 8.652 | +11.89% | 238.352 | 130.182 |
| down | 8 | 19 | 4.580 | 5.402 | +17.94% | 201.226 | 108.055 |
| gate+up | 16 | 19 | 7.747 | 8.670 | +11.92% | 237.935 | 129.922 |
| down | 16 | 19 | 4.571 | 5.388 | +17.87% | 201.607 | 108.328 |
| gate+up | 1 | 52 | 7.721 | 8.632 | +11.80% | 238.717 | 130.487 |
| down | 1 | 52 | 4.562 | 5.392 | +18.19% | 202.029 | 108.257 |
| gate+up | 4 | 52 | 7.702 | 8.635 | +12.12% | 239.306 | 130.439 |
| down | 4 | 52 | 4.563 | 5.379 | +17.88% | 201.974 | 108.515 |
| gate+up | 8 | 52 | 7.712 | 8.641 | +12.05% | 239.017 | 130.357 |
| down | 8 | 52 | 4.555 | 5.388 | +18.28% | 202.324 | 108.334 |
| gate+up | 16 | 52 | 7.703 | 8.627 | +12.00% | 239.294 | 130.566 |
| down | 16 | 52 | 4.548 | 5.372 | +18.12% | 202.624 | 108.646 |

Native sparse is block32; dense and Part A are block16. Bandwidth above counts native physical bytes including down metadata padding.

| D | M | D-expert ungrouped saving per layer (us) | ungrouped 48-layer saving (us) |
|---:|---:|---:|---:|
| 19 | 1 | -33.67 | -1616.32 |
| 19 | 4 | -33.85 | -1624.78 |
| 19 | 8 | -33.08 | -1587.96 |
| 19 | 16 | -33.06 | -1586.87 |
| 52 | 1 | -90.53 | -4345.29 |
| 52 | 4 | -90.95 | -4365.54 |
| 52 | 8 | -91.63 | -4398.10 |
| 52 | 16 | -90.91 | -4363.88 |

**Final verdict: NO-GO without recovery training (numerical gate failed).**

- sparse-all: NI FAIL at 0.5 pp
- sparse-down: NI FAIL at 0.5 pp

Recovery training could change the numerical result; none was run.

Direct use of the original checkpoint scales with this upstream sparse primitive is additionally incompatible (block16 versus block32); a primitive win cannot remove that missing numerical gate.
<!-- S1-LIVE-END -->
