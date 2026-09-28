# Q1 — Quality audit of the Qwen3.8-Flash-Next production serving stack (2026-09-07)

Purpose: check that the singleton-expert pruning shipped 2026-09-06 (P1/P2: routes to experts used by exactly one verify
row with routing weight < 0.08 are dropped — the first optimisation that changes the target model's computation) causes no
measurable quality loss, and that the whole 2026-09-06 optimisation stack matches the pre-optimisation configuration.
Harness: `bench/quality/` (run_all.sh -> run_arm.sh -> run_bench.py / analyze.py / he_exec.py / write_report.py).
Per-prompt logs: `bench/quality/runs/<arm>/<bench>.jsonl` (prompt, output, prediction, correctness, finish reason, tokens).

## Setup

- Server: `serve-fast.sh wa` (adaptive profile, production launcher, one server per arm, arms run back to back on the
  same machine under `flock ~/.gpu.lock`). Target `Qwen3.8-Flash-Next-NVFP4-mtpft5` (the v5 MTP head became the launcher
  default at 00:21 while this audit was queued). All four arms used the same head, so it is held constant across the
  comparison. It is part of the tested configuration, not neutral: with the default singleton pruning, which experts the
  target drops depends on every token in the verify batch, so a different draft can change the target's output.
- Arms (env passed on top of the launcher defaults):
- `prod`: Q1_ARM=prod
- `noprune`: SGLANG_MOE_PRUNE_SINGLETON_TAU=0
- `legacy`: SGLANG_MOE_PRUNE_SINGLETON_TAU=0 SGLANG_TRITON_PDL=0 SGLANG_NORM_INTO_GEMV=0 SGLANG_HC_LAYER_APPLY_FUSED=0 SGLANG_HC_APPLY_MIX_FUSED=0 SGLANG_SHARED_GATE_EARLY=0 SGLANG_HC_GATE_EARLY=0 SGLANG_GDN_CONV_CHAIN_PARALLEL=0 SGLANG_GDN_PROJ_DIRECT_LAYOUT=0 SGLANG_GDN_AB_STASH_DIRECT=0 SGLANG_ADAPTIVE_POLICY=ema
- `prod2`: Q1_ARM=prod2
- Decoding: greedy (temperature 0, top_k 1), `chat_template_kwargs: {"enable_thinking": false}` (no reasoning; same in
  every arm), client concurrency 8 (server `max_running_requests` = 2 in this profile, so requests are batched 2 at a time).
- Benchmarks (fixed subsets, `bench/quality/data/*.jsonl`, built once by `prep_data.py`, seed 0):
  GSM8K test (1319, "#### <number>" extraction, max_tokens 512); MMLU 14 subjects x 100 (zero-shot letter answer,
  max_tokens 16; abstract_algebra, anatomy, astronomy, college_computer_science, college_mathematics, high_school_biology,
  high_school_chemistry, high_school_physics, high_school_world_history, machine_learning, moral_scenarios, philosophy,
  professional_law, world_religions); HumanEval (164, pass@1, code block extracted and executed with the official
  `check()` inside `bwrap --unshare-all` with a 10 s timeout, max_tokens 768); JCommonsenseQA v1.3 validation (1119,
  letter answer, max_tokens 16); needle 18.5k at depths 0.1/0.5/0.9 before and after the battery (`prof/needle_test.py`);
  long-generation sanity 5 prompts x 1500 tokens (prose-en, prose-ja, code-edit, agent-loop, essay; the prompts are in
  `bench/bench/quality/run_bench.py`) with max character
  run, unique-word ratio and repeated-4-gram rate.
- Statistics: 95 % Wilson interval per cell; paired comparison vs `prod` with an exact two-sided McNemar test on the
  discordant pairs (same questions), flips reported as prod-right/other-wrong : prod-wrong/other-right.
- Timeline (JST):
- prod: [prod] start 00:43:58 -> [prod] benchmarks done 01:08:49 total 1379s | [prod] stopped 01:08:57
- noprune: [noprune] start 01:10:13 -> [noprune] benchmarks done 01:37:00 total 1499s | [noprune] stopped 01:37:08
- legacy: [legacy] start 01:38:44 -> [legacy] benchmarks done 02:05:56 total 1520s | [legacy] stopped 02:06:04
- prod2: [prod2] start 02:08:15 -> [prod2] benchmarks done 02:33:23 total 1397s | [prod2] stopped 02:33:31

## Results

| benchmark | prod | noprune | legacy | prod2 |
|---|---|---|---|---|
| gsm8k | 88.63% [86.8, 90.2] (1169/1319) | 89.31% [87.5, 90.9] (1178/1319) | 89.46% [87.7, 91.0] (1180/1319) | 89.31% [87.5, 90.9] (1178/1319) |
| mmlu | 85.36% [83.4, 87.1] (1195/1400) | 85.07% [83.1, 86.8] (1191/1400) | 84.93% [83.0, 86.7] (1189/1400) | 84.86% [82.9, 86.6] (1188/1400) |
| humaneval | 96.95% [93.1, 98.7] (159/164) | 96.34% [92.2, 98.3] (158/164) | 96.95% [93.1, 98.7] (159/164) | 95.73% [91.5, 97.9] (157/164) |
| jcqa | 97.14% [96.0, 98.0] (1087/1119) | 97.14% [96.0, 98.0] (1087/1119) | 97.14% [96.0, 98.0] (1087/1119) | 97.14% [96.0, 98.0] (1087/1119) |
| wall s (gsm8k/mmlu/he/jcqa) | 1089.7/93.3/74.2/67.6 | 1181.9/110.2/81.8/70.0 | 1202.2/108.7/81.0/68.5 | 1082.5/119.2/74.7/66.0 |
| mean gen tokens | 342.9/2.1/230.6/2.0 | 339.4/2.1/230.9/2.0 | 339.9/2.1/229.3/2.0 | 343.5/2.2/234.5/2.0 |

Paired vs prod (McNemar exact, two-sided; flips = base-right/other-wrong : base-wrong/other-right):

| benchmark | arm | diff (pp) | diff 95% CI (pp) | flips +/- | p | n |
|---|---|---|---|---|---|---|
| gsm8k | noprune | +0.68 | [-0.62, +1.99] | 34/43 | 0.362 | 1319 |
| mmlu | noprune | -0.29 | [-0.97, +0.40] | 14/10 | 0.541 | 1400 |
| humaneval | noprune | -0.61 | [-1.80, +0.58] | 1/0 | 1.000 | 164 |
| jcqa | noprune | +0.00 | [+0.00, +0.00] | 0/0 | 1.000 | 1119 |
| gsm8k | legacy | +0.83 | [-0.50, +2.17] | 35/46 | 0.266 | 1319 |
| mmlu | legacy | -0.43 | [-1.24, +0.39] | 20/14 | 0.392 | 1400 |
| humaneval | legacy | +0.00 | [-1.69, +1.69] | 1/1 | 1.000 | 164 |
| jcqa | legacy | +0.00 | [+0.00, +0.00] | 0/0 | 1.000 | 1119 |
| gsm8k | prod2 | +0.68 | [-0.60, +1.97] | 33/42 | 0.356 | 1319 |
| mmlu | prod2 | -0.50 | [-1.44, +0.44] | 26/19 | 0.371 | 1400 |
| humaneval | prod2 | -1.22 | [-2.90, +0.46] | 2/0 | 0.500 | 164 |
| jcqa | prod2 | +0.00 | [+0.00, +0.00] | 0/0 | 1.000 | 1119 |

GSM8K restricted to questions that finished (finish_reason=stop) in BOTH arms (removes the 512-token-cap coin flips):

| pair | n both finished | base acc | other acc | diff (pp) | flips +/- | p |
|---|---|---|---|---|---|---|
| prod vs noprune | 1134 | 98.41% | 98.59% | +0.18 | 1/3 | 0.625 |
| prod vs legacy | 1135 | 98.24% | 98.24% | +0.00 | 2/2 | 1.000 |
| prod vs prod2 | 1128 | 98.23% | 98.23% | +0.00 | 2/2 | 1.000 |

Output identity vs prod (greedy; differences come from the non-deterministic MoE finalize and, for prod, pruning):

| arm | gsm8k identical | mmlu identical | humaneval identical | jcqa identical | gsm8k truncated (base/other) |
|---|---|---|---|---|---|
| noprune | 83/1319 | 1366/1400 | 96/164 | 1119/1119 | 155/149 |
| legacy | 72/1319 | 1349/1400 | 94/164 | 1119/1119 | 155/137 |
| prod2 | 82/1319 | 1337/1400 | 92/164 | 1119/1119 | 155/147 |

| sanity | prod | noprune | legacy | prod2 |
|---|---|---|---|---|
| agent-loop tokens/maxrun/uniq/rep4 | 456/31/0.531/0.461 | 298/8/0.508/0.398 | 463/31/0.549/0.444 | 468/31/0.537/0.45 |
| code-edit tokens/maxrun/uniq/rep4 | 1500/4/0.211/0.763 | 1500/4/0.211/0.763 | 1500/4/0.211/0.763 | 1500/4/0.211/0.763 |
| essay tokens/maxrun/uniq/rep4 | 1500/2/0.509/0.005 | 1500/2/0.514/0.005 | 1500/2/0.514/0.001 | 1500/2/0.511/0.006 |
| prose-en tokens/maxrun/uniq/rep4 | 1500/2/0.478/0.023 | 1434/2/0.471/0.046 | 1500/2/0.475/0.014 | 1500/2/0.486/0.032 |
| prose-ja tokens/maxrun/uniq/rep4 | 911/1/1.0/0.0 | 988/2/1.0/0.014 | 1080/2/1.0/0.006 | 959/2/1.0/0.026 |

Needle 18.5k (pre-run / post-run, depths 0.1/0.5/0.9):

- prod: PASS(6.7s), PASS(1.4s), PASS(1.3s), PASS(1.4s), PASS(1.4s), PASS(0.7s)
- noprune: PASS(6.7s), PASS(1.3s), PASS(1.2s), PASS(1.4s), PASS(1.4s), PASS(0.8s)
- legacy: PASS(6.8s), PASS(1.3s), PASS(1.2s), PASS(1.4s), PASS(1.4s), PASS(0.8s)
- prod2: PASS(6.5s), PASS(1.3s), PASS(1.3s), PASS(1.4s), PASS(1.4s), PASS(0.8s)

VERDICT: no significant degradation


Truncated generations (finish_reason = length) per arm:
- prod: gsm8k 155, mmlu 12, humaneval 2, jcqa 0
- noprune: gsm8k 149, mmlu 13, humaneval 2, jcqa 0
- legacy: gsm8k 137, mmlu 10, humaneval 1, jcqa 0
- prod2: gsm8k 147, mmlu 16, humaneval 3, jcqa 0

## Reading the numbers

- `prod2` is a second run of the production configuration and measures the run-to-run scatter of the benchmarks
  themselves (the MoE finalize is non-deterministic, so greedy outputs diverge between identical servers).
  Any prod-vs-noprune or prod-vs-legacy difference should be read against the prod-vs-prod2 line.
- GSM8K at max_tokens 512 with thinking off: roughly 11 % of the chains hit the cap, and almost every flipped answer
  involves a truncated chain in one of the two arms (the chain diverges by a few tokens and either fits under the cap or
  not). Restricted to questions that finished in both arms, accuracy is ~98 % in every arm (the model card reports 97.3 %
  with thinking on, 8192 tokens), so the headline 88-89 % is a cap artefact shared by all arms, not a model property.
- MMLU/JCQA outputs are single letters; JCQA is byte-identical across all arms, MMLU differs on ~2-4 % of questions with
  flips in both directions.
- Sanity generations are all healthy (no character runs beyond structural separators, unique-word ratio and 4-gram
  repetition in the normal band for each prompt type; code-edit legitimately repeats record dictionaries in every arm).

## Verdict

**No significant degradation.** Every prod-minus-noprune difference is inside its 95 % interval and far from
significance (GSM8K -0.68 pp, p=0.36, flips 34/43; MMLU +0.29 pp, p=0.54, flips 14/10; HumanEval +0.61 pp (1 item),
p=1.0; JCQA identical 1087/1119), and the differences are the same size as the prod-vs-prod2 run-to-run scatter
(GSM8K -0.68 pp, MMLU +0.50 pp, HumanEval +1.22 pp). On the GSM8K questions that finished under the cap in both arms,
prod vs noprune is 98.41 % vs 98.59 % (flips 1/3, p=0.63). Needle 18.5k passed at all three depths before and after the
battery in every arm; no arm shows degeneration in the 1500-token generations.

The full 2026-09-06 stack (prod) vs the pre-optimisation configuration (legacy) is likewise within noise on every
benchmark (GSM8K -0.83 pp p=0.27; MMLU +0.43 pp p=0.39; HumanEval 0; JCQA identical; both-finished GSM8K 98.24 % vs
98.24 %, flips 2/2).

Effect sizes to keep in mind: the audit can only bound the pruning effect to roughly +-1.5 pp on GSM8K/MMLU (n=1319/1400)
and +-3 pp on HumanEval (n=164); a sub-percent systematic loss would not be detectable with these sizes.

## Caveats

- Launch workaround, applied identically to every arm: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see
  `run_arm.sh`). From ~00:15 every `serve-fast.sh` start on this machine (mine under `wa`, and another agent's `w4`
  at 00:36) died at the KV-pool check ("Loaded weights leave no GPU memory for the KV cache under
  --mem-fraction-static=0.925"). A `torch.cuda.memory_stats` probe (sitecustomize hook on `mem_get_info`) showed the
  cause: after the draft load the native caching allocator held 86.9 GB reserved against 77.7 GB allocated, with 9.0 GB
  in inactive split blocks that `empty_cache` cannot return, so the draft phase "freed" only 4.2 GB instead of the usual
  10.8 GB. With expandable segments the pools are identical to the known-good runs (draft phase -10.76 GB, 9.38 GB
  free after the pools, max_total_num_tokens 378112). The allocator setting does not touch any kernel or result; it only
  changes how freed blocks are returned. Why the fragmentation pattern changed at ~00:15 (all earlier starts were fine)
  was not identified — worth a look by whoever owns the launcher, because production `wa` at 0.925 will not start
  without it while the condition persists.
- Concurrency: the audit ran with client concurrency 8 but the profile's `max_running_requests` is 2, so batches were of
  size <= 2; the pruning only acts on experts that are singletons within a verify batch, so larger batches (more rows per
  expert) prune *less*. The audit therefore pruned less than a single-request (bs = 1) server does and is **not** a
  conservative test of the single-request setting that the speed numbers use.
- Thinking off / greedy: the audit measures the non-reasoning path only. Reasoning-mode quality (long chains, where a
  pruned low-weight expert could compound) was not measured.
- GSM8K headline numbers are cap-limited (see above); the both-finished subset is the informative comparison.
- HumanEval pass@1 with 164 items has a wide interval (+-3 pp); one flipped item is 0.6 pp.
- The sanity metrics are single greedy generations per prompt; they detect degeneration, not subtle quality change.
