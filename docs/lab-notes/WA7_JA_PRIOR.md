# WA7 — Japanese-only request pin

Status: COMPLETE — requested numerical gate FAIL; do not adopt. Four valid ABBA arms, 128 requests, 57.057 occupied GPU minutes. No production change, commit or push.

## Frozen contract

- Worktree `$HOME/tools/sglang-wa7`, branch `codex/wa7-ja-prior`, from production `codex/perf-v1` 7b4d539f9b. WA6 ff628dcd79430437eb095792a77bf62b610bf606 applied with `git cherry-pick --no-commit` to honor the no-commit instruction. No new cherry-pickable WA7 commit will be fabricated.
- A: production three-width wa config, B: identical config plus BS1 `ja_prior_threshold: 0.5` in `adaptive/w16_3_7_15_c_ja.json`. Both use the same WA7 PYTHONPATH overlay; default configuration is inert. No measured-cost table, request_prior, frozen predictor, or caller/env classification in B.
- Ratio: kana/Han code points divided by all code points in the full input prompt (including whitespace, punctuation, and any serialized chat template). No head/tail sampling. Exact >= threshold comparison. Empty/missing text abstains; token-only input decodes the complete prompt once using its existing tokenizer. Explicit Unicode ranges in source; script detection also matches Han-only Chinese and is not semantic language identification.
- A matched BS1 request owns a single-candidate S=3 slot through its entire decode, bypassing promotion. Nonmatching requests retain the ordinary shared confidence slot; Japanese statistics cannot replace it. Mixed BS>=2 uses the existing batch-wide policy; on return to BS1 the pinned slot resumes. This is the production single-stream scope; heterogeneous batch per-row widths are not implemented.
- One chronological A1/B1/B2/A2 cycle, 32 requests per arm, exactly BN1 `workloads/sets/*-v1` round-robin code/en/ja/agent, 8 parents/domain, one repeat, greedy, original budgets and EOS. One discarded BN1 1024-token neutral warmup/restart. No tuning after holdout results.
- Same .900 memory fractions, `SERVE_DISPLAY_HZ=` empty, production STEP 7.943/0.5554 and X3 autotune. Every server lifetime is enclosed by `flock -w 28800 $HOME/.gpu.lock`, released only after owned-group cleanup and NVML context drain. Production tree/launchers/venv read-only via bwrap, private cache bind, no production changes. >=4096 MiB free watchdog; no unowned process signals.
- Expected occupied time about one hour. Safety cap 4200 seconds including all four startups/cleanup (headroom over BN1's ~14 minutes per arm); queue time excluded; exactly four completed arms, no extra cycles.
- Width occupancy is CPU-only count of actual decode submissions by request and S, identical opt-in instrumentation in both arms. No CUDA trace or GPU events. Aggregates flush only at the next request boundary; one excluded 8-token request after measurement flushes the final aggregate. Occupancy includes speculative run-ahead submissions and is not assumed identical to finished verify counters.
- Primary estimand: equal-parent mean log(B/A) t/s, averaged across two adjacent blocks. `bench/stats/paired_ab.py`, 20000 parent bootstrap draws, seed 20260908; two-sided 95% CI, one-sided lower bound, block means, all paired log-ratios, A2/A1 drift.
- Requested screening gate: Japanese point effect >=+3% and two-sided 95% lower CI >0. Other domains |point effect| within BN1 wa n=8 conditional throughput MDE: code 7.72%, English 3.14%, agent 3.86%. Also report restart-aware thresholds 11.29%, 7.60%, 5.59%. Conditional thresholds are the stricter reading of the ambiguous MDE requirement, frozen before running.
- A single ABBA cannot establish a general 3% improvement under BN1's measured restart variance. Report requested numerical screening result separately from the adoption recommendation and restart/drift limitations.

## Incremental evidence

CPU preflight: 81 adaptive tests PASS, 22 fnbench tests PASS, 6 BN1 tests PASS. The review venv lacks numpy; BN1 statistical tests ran with the existing production Python (CUDA_VISIBLE_DEVICES=999), without installing anything. 7319 source/config/prompt files frozen. Raw detector audit (before chat template): Japanese 8/8 match; code/en/agent 0/24 match. See study-20260908/detector-audit.json and cpu-tests.txt.

Completed `wa7-20260908-110805-A1`: 32 valid requests; lock 838.8 seconds.

Completed `wa7-20260908-112205-B1`: 32 valid requests; lock 836.9 seconds.

Completed `wa7-20260908-113610-B2`: 32 valid requests; lock 845.0 seconds.

Completed `wa7-20260908-115025-A2`: 32 valid requests; lock 902.7 seconds.

## Final paired results

Equal-parent geometric means; effect = exp(mean paired log(B/A)) - 1. CI is two-sided 95%, conditional on these four restarts.

| Domain | A t/s | B t/s | Effect | 95% CI | 1-sided lower | MDE tolerance | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| agent-loop | 344.466 | 354.887 | +3.025% | [-0.712, +7.431]% | -0.229% | 3.86% | PASS |
| code-edit | 556.885 | 580.704 | +4.277% | [-0.494, +9.773]% | +0.161% | 7.72% | PASS |
| prose-en | 245.416 | 257.271 | +4.831% | [+0.888, +8.577]% | +1.505% | 3.14% | FAIL |
| prose-ja | 270.616 | 277.975 | +2.720% | [-5.684, +9.342]% | -4.206% | >=+3%, CI low>0 | FAIL |

### Block means and control drift

| Domain | Block 1 mean log B1/A1 | Block 1 effect | Block 2 mean log B2/A2 | Block 2 effect | A2/A1 drift | Drift 95% CI | Flag |
|---|---:|---:|---:|---:|---:|---:|---|
| agent-loop | -0.010260 | -1.021% | +0.069868 | +7.237% | -8.595% | [-11.974, -5.056]% | True |
| code-edit | -0.041663 | -4.081% | +0.125427 | +13.363% | -13.130% | [-18.742, -7.331]% | True |
| prose-en | +0.041379 | +4.225% | +0.052976 | +5.440% | -4.416% | [-7.621, -1.127]% | True |
| prose-ja | -0.032645 | -3.212% | +0.086310 | +9.014% | -12.263% | [-24.598, +0.805]% | True |

### Width occupancy by arm and domain

Counts and fractions are actual decode submissions, including overlap run-ahead. They are not target bonus token counts, wall-time occupancy, or completed verifies. B Japanese is asserted to be 100% W4 for every measured prompt.

| Arm | Domain | W4 count / % | W8 count / % | W16 count / % |
|---|---|---:|---:|---:|
| A1 | agent-loop | 1243 / 8.46 | 12923 / 87.94 | 530 / 3.61 |
| A1 | code-edit | 78 / 0.88 | 1091 / 12.26 | 7730 / 86.86 |
| A1 | prose-en | 19196 / 94.91 | 410 / 2.03 | 620 / 3.07 |
| A1 | prose-ja | 17419 / 96.08 | 400 / 2.21 | 310 / 1.71 |
| B1 | agent-loop | 1177 / 8.07 | 13118 / 89.97 | 285 / 1.95 |
| B1 | code-edit | 97 / 1.02 | 2201 / 23.24 | 7173 / 75.74 |
| B1 | prose-en | 17835 / 92.13 | 787 / 4.07 | 737 / 3.81 |
| B1 | prose-ja | 18306 / 100.00 | 0 / 0.00 | 0 / 0.00 |
| B2 | agent-loop | 1284 / 8.77 | 13362 / 91.23 | 0 / 0.00 |
| B2 | code-edit | 168 / 1.82 | 1620 / 17.55 | 7443 / 80.63 |
| B2 | prose-en | 19229 / 94.77 | 577 / 2.84 | 485 / 2.39 |
| B2 | prose-ja | 19240 / 100.00 | 0 / 0.00 | 0 / 0.00 |
| A2 | agent-loop | 1167 / 7.84 | 13520 / 90.83 | 198 / 1.33 |
| A2 | code-edit | 49 / 0.49 | 2084 / 20.99 | 7797 / 78.52 |
| A2 | prose-en | 19252 / 95.19 | 470 / 2.32 | 503 / 2.49 |
| A2 | prose-ja | 18110 / 96.96 | 376 / 2.01 | 192 / 1.03 |

### Acceptance (completed generation tokens / verify)

| Domain | Effect | 95% CI | Block 1 / 2 effect |
|---|---:|---:|---:|
| agent-loop | -0.391% | [-4.423, +3.344]% | -1.875% / +1.115% |
| code-edit | +0.723% | [-3.920, +6.297]% | -5.602% / +7.473% |
| prose-en | +2.603% | [-0.789, +7.367]% | +5.602% / -0.310% |
| prose-ja | -2.398% | [-12.936, +7.200]% | -5.247% / +0.536% |

### All paired throughput log-ratios

| Domain | Prompt | log B1/A1 | log B2/A2 | Mean log ratio |
|---|---|---:|---:|---:|
| agent-loop | bn1-agent-loop-01 | -0.090193 | +0.032038 | -0.029078 |
| agent-loop | bn1-agent-loop-02 | -0.111400 | +0.046467 | -0.032466 |
| agent-loop | bn1-agent-loop-03 | +0.038446 | +0.075105 | +0.056776 |
| agent-loop | bn1-agent-loop-04 | -0.130274 | +0.179906 | +0.024816 |
| agent-loop | bn1-agent-loop-05 | +0.194011 | +0.112385 | +0.153198 |
| agent-loop | bn1-agent-loop-06 | -0.034031 | -0.003358 | -0.018695 |
| agent-loop | bn1-agent-loop-07 | +0.018707 | +0.028941 | +0.023824 |
| agent-loop | bn1-agent-loop-08 | +0.032655 | +0.087458 | +0.060056 |
| code-edit | bn1-code-edit-01 | -0.058585 | +0.159206 | +0.050310 |
| code-edit | bn1-code-edit-02 | -0.113345 | +0.029366 | -0.041990 |
| code-edit | bn1-code-edit-03 | -0.053927 | +0.064852 | +0.005462 |
| code-edit | bn1-code-edit-04 | +0.079579 | -0.163746 | -0.042084 |
| code-edit | bn1-code-edit-05 | -0.106121 | +0.122179 | +0.008029 |
| code-edit | bn1-code-edit-06 | -0.021649 | +0.151076 | +0.064714 |
| code-edit | bn1-code-edit-07 | -0.069333 | +0.298531 | +0.114599 |
| code-edit | bn1-code-edit-08 | +0.010074 | +0.341952 | +0.176013 |
| prose-en | bn1-prose-en-01 | +0.227992 | +0.026875 | +0.127433 |
| prose-en | bn1-prose-en-02 | +0.094139 | -0.022446 | +0.035847 |
| prose-en | bn1-prose-en-03 | +0.013705 | +0.046759 | +0.030232 |
| prose-en | bn1-prose-en-04 | -0.060344 | -0.043924 | -0.052134 |
| prose-en | bn1-prose-en-05 | +0.055865 | +0.107446 | +0.081655 |
| prose-en | bn1-prose-en-06 | -0.000457 | +0.171900 | +0.085721 |
| prose-en | bn1-prose-en-07 | -0.036525 | +0.026803 | -0.004861 |
| prose-en | bn1-prose-en-08 | +0.036654 | +0.110398 | +0.073526 |
| prose-ja | bn1-prose-ja-01 | +0.130202 | +0.092444 | +0.111323 |
| prose-ja | bn1-prose-ja-02 | +0.086392 | -0.021527 | +0.032433 |
| prose-ja | bn1-prose-ja-03 | -0.600839 | +0.136357 | -0.232241 |
| prose-ja | bn1-prose-ja-04 | +0.048865 | +0.077818 | +0.063341 |
| prose-ja | bn1-prose-ja-05 | -0.111368 | +0.185463 | +0.037048 |
| prose-ja | bn1-prose-ja-06 | +0.077003 | -0.119230 | -0.021113 |
| prose-ja | bn1-prose-ja-07 | -0.019914 | +0.290163 | +0.135125 |
| prose-ja | bn1-prose-ja-08 | +0.128503 | +0.048989 | +0.088746 |

### Execution audit

| Arm | Lock seconds | Min free MiB | Steady min MiB | Duration exceptions | Seed |
|---|---:|---:|---:|---:|---:|
| A1 | 838.8 | 6292 | 6315 | 1 | 350526237 |
| B1 | 836.9 | 5420 | 5420 | 0 | 442184191 |
| B2 | 845.0 | 4227 | 5305 | 0 | 944069257 |
| A2 | 902.7 | 4285 | 4596 | 0 | 111132278 |

Total occupied GPU time: 3423.4 seconds = 0.9509 hours. Queue time excluded. All 128 request records/counters and frozen hashes validated (32 unique parents, 64 adjacent paired contrasts).

Requested numerical screen: **FAIL**. No production change, commit or push.

## Interpretation, gate, and recommendation

**Do not adopt WA7. Keep production `adaptive/w16_3_7_15_c.json`, candidates [3,7,15], STEP 7.943/0.5554, and the X3 autotune fix.** The script pin works, but this experiment does not establish the required improvement.

- Japanese point effect is +2.720%, below +3%; its two-sided 95% CI [-5.684%, +9.342%] and one-sided 95% lower bound -4.206% both include non-improvement. Block effects disagree (-3.212%, +9.014%). This is a failed adoption screen, not proof that the true effect is negative.
- Code and agent meet the predeclared absolute-point-effect tolerances. English +4.831% exceeds the stricter BN1 wa n=8 conditional MDE of 3.14%; this violates the literal “unchanged” criterion in the beneficial direction, not a measured regression. If “MDE from BN1” is interpreted as the restart-aware values (code 11.29%, English 7.60%, agent 5.59%), all three non-Japanese domains pass; the overall gate remains FAIL because Japanese still fails.
- Every domain has a control-drift flag: A2 versus A1 is -13.130% code, -4.416% English, -12.263% Japanese, -8.595% agent. A2 is slower across domains. The prompt bootstrap is conditional on these restarts and does not resample a common restart shift. ABBA balances ideal linear drift with equally spaced windows; it cannot remove arbitrary restart variation, nonlinear drift, or carryover. These measurements do not identify the cause of the drift or attribute it to Firefox.
- A Japanese request already spends most of its decode submissions at W4 in A (A1 96.08%, A2 96.96%). B is exactly W4 for every one of the 16 Japanese requests. The pin removes the remaining wide submissions, but the throughput result is not a reproducible >=3% win. The ordinary slot is preserved across pinned requests, but changing Japanese execution/history can still change later ordinary requests; zero cross-domain effects cannot be guaranteed by a detector alone.
- All valid requests are retained. A1 `bn1-prose-ja-03` uses all 6144 allowed tokens in 14.636 seconds: 422.073 t/s and 4.637 tokens/verify. This is the sole 15–60 second duration-target exception (127/128 meet the target), not an invalid counter or early EOS. Its B1 counterpart is 231.444 t/s, 5719 tokens, 24.790 seconds, and 2.113 tokens/verify; B2/A2 are 290.849/253.775 t/s. The strong first-block log contrast -0.600839 is retained, with no post-hoc prompt exclusion or retuning.
- Occupancy instrumentation records one additional speculative submission beyond completed verify calls for every measured request (range exactly +1). All 128 aggregates were matched uniquely by request timestamps. Startup probe and neutral warmup aggregates are excluded; occupancy raw counts and per-prompt records are preserved.
- Both arms retained .900 memory fractions and empty display setting. Across startup and steady samples the minimum was 4227 MiB; steady minimum was 4596 MiB. The required 4096 MiB floor was met throughout sampled monitoring. All servers ended by owned-process-group cleanup; each lock remained held through NVML compute-context drain. Four completed arms, no invalid arms, no extra GPU cycles. Saved study size including private caches: 2.4 GiB (`du -sh`).
- 7319 frozen source/config/prompt hashes and model size/mtime/resolved-path inventory match after all arms. Resolved server configuration differs only in the intended adaptive config and restart seed/timing/internal state. Production source/launchers/venv were read-only during serving; no package installation or production edit was made. CPU results: 81 adaptive tests, 22 fnbench tests, 6 BN1 tests PASS. Final staged and unstaged diff whitespace checks PASS.

## Exact experimental change and handoff (not an adoption instruction)

`$HOME/tools/sglang-wa7` is on `codex/wa7-ja-prior`, HEAD remains `7b4d539f9bd896265f498fabccc6466e45ffe818`. WA6 prerequisite `ff628dcd79430437eb095792a77bf62b610bf606` was applied with `git cherry-pick --no-commit`; WA7 edits and tests are uncommitted. No WA7 commit hash exists because commits were explicitly prohibited. WA6 alone does not implement WA7.

- `specs/wa7/study-20260908/wa7-on-wa6.patch`: complete WA7 source/test delta on WA6 ff628dcd79; `git apply --check` succeeds against the unchanged WA6 worktree. SHA-256 `f9bade59e38f99101258130213903e461013da52db9f958eb49f461e63377575`.
- `specs/wa7/study-20260908/wa7-combined.patch`: WA6 + WA7 source/test delta on production 7b4d539f9b. SHA-256 `e4a3f6a7952a155615446c25cfe8844f826284c981ab3212224f119b64c7f875`.
- Experimental config `adaptive/w16_3_7_15_c_ja.json` differs from production only by `"ja_prior_threshold": 0.5` in slot `"1"`. Omission or null disables it. Enabled values must be finite numeric (0,1], S=3 must exist, and enabling `request_prior` or `step_cost_ms` simultaneously is rejected.
- The scope is BS1. Mixed batches retain the existing BS>=2 policy; pinned requests resume W4 when they return to BS1. Unicode script detection is not semantic language detection and can match Chinese. These limitations remain relevant even if a later study passes.

The exact experimental launcher environment is saved separately for every arm in `command.json`. The arm runner uses the original production `serve-fast.sh wa` through a read-only bwrap bind and a private cache. Reproduction uses a fresh study and fresh labels:

```bash
cd $HOME/tools/flash-next-bench
.venv-review/bin/python -B bench/stats/run_wa7.py freeze specs/wa7/<fresh-study>
.venv-review/bin/python -B bench/stats/run_wa7.py sequence specs/wa7/<fresh-study>
```

Each `sequence` arm executes `flock -w 28800 $HOME/.gpu.lock env BN1_LOCKED=1 ... run_wa7.py arm ...`; B uses `PYTHONPATH=$HOME/tools/sglang-wa7/python`, `WA_MEM_FRACTION=0.900`, `SERVE_DISPLAY_HZ=`, and `ADAPTIVE_CONFIG=$HOME/tools/flash-next-bench/adaptive/w16_3_7_15_c_ja.json`. This completed task authorizes no additional cycle through these reproduction instructions.

Machine-readable evidence: `paired.json` (all paired log-ratios, block means, CIs, drift, acceptance, conditional MDE), `audit.json` (configuration, memory, time, gates), `occupancy-by-prompt.json`, `detector-audit.json`, `frozen.json`, and four arm directories. `bench/stats/wa7_report.py` regenerates tables and validates complete pairing and occupancy. All documents and raw evidence are left uncommitted.
