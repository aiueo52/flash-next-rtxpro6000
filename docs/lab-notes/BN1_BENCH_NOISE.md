# BN1 — held-out prompt sets and paired block benchmark

2026-09-08. Branch `codex/harness-v1`; uncommitted. CPU implementation and six production restart arms are complete: 192/192 requests have valid paired metrics, 191/192 meet the 15–60 second duration target. One valid fast response (9.847 seconds) is retained and documented below. Production source, launchers and venv were read-only during serving. Total occupied GPU lock time including the excluded pilot and failed preflight: **5626.246 seconds = 93.771 minutes = 1.563 hours**. Shared-lock queue time is excluded.

## Frozen measurement contract

- 8 original prompts per domain, 32 total, in `workloads/sets/<domain>-v1/`. The original one-prompt workloads remain available.
- Stable round-robin domain order: code-edit, prose-en, prose-ja, agent-loop for prompt 01, then 02, through 08. Identical in every restart/arm. No label is inserted in a request.
- One discarded, non-holdout, 1024-token general workshop warmup per restart. No holdout-specific warmup; greedy sampling, one request at a time, one repeat per prompt.
- Budgets: code-edit 12288, prose-en 6144, prose-ja 6144, agent-loop 8192. These are fixed caps, not forced output lengths. The 15–60 second request-duration target requires GPU calibration; early EOS is retained, never padded or silently excluded.
- Throughput preserves fnbench's existing `(completion_tokens - 1) / (last_token_time - first_token_time)` definition. This is the historical client decode metric; speculative SSE chunks may contain multiple tokens. TTFT, decode duration, completion tokens, output hash and raw counters are saved.
- Acceptance is **completed generation tokens / verify calls**, including target bonus tokens, from request-bracketed `sglang:generation_tokens_total` and `sglang:spec_verify_calls_total`. These are counters updated by `observe_one_finished_request` in production `observability/metrics_collector.py`. It is not a draft-acceptance probability.
- The current `bench/quality/run_bench.py` is a quality-battery driver and does not actually implement the cited counter extraction. The matching implementation tonight is `specs/dh6/analyze.py` (also DH5); `calib/wa*_report.py` uses trace `1 + accepted drafts` and SSE cross-checks. Server log `accept` lines summarize intervals and cannot assign one acceptance value to each prompt reliably. BN1 uses counters as its primary measurement and retains raw before/after snapshots.
- Counter publication can lag SSE completion: retry only the **scrape**, for at most 2 seconds. Never retry a generation. Missing/changed series, negative deltas, no verifies, missing usage, or a completion-count mismatch are invalid, not zero acceptance. Strict study mode writes the bad record then fails the arm. Isolated server ownership is required; counters cannot disambiguate arbitrary concurrent clients.

## Provenance and token accounting

`bench/stats/build_sets.py` authors fixtures deterministically without sampling any training parent. The code prompts contain newly generated literal inventories and distinct edit rules; the two prose domains use independently authored settings; agent prompts use distinct debugging fixtures with explicitly simulated future tool results. Domain style follows the legacy workloads, not their text or parents.

`bench/stats/audit_sets.py` checks NFKC/whitespace/case-normalized full-prompt equality/containment and 50-character shingle overlap against these five frozen inputs:

1. `$HOME/mtp-gen-v5/gen.jsonl` (decode prompt_ids)
2. `$HOME/mtp-gen-v5/prompts.jsonl`
3. `$HOME/mtp-dump-mt1/prompts.json` (decode input_ids)
4. `$HOME/mtp-dump-mt1/prompts-expanded.json`
5. `$HOME/tools/mtp-train/results/dh7/prompts.json`

The audit was run for all 32 prompts against every prompt text extracted from these five files, which are the author's private MTP-training prompt files; the predeclared review threshold was 20% shingle overlap. Its result is not published (per-prompt overlap figures and all details of the private sources are likewise not published). It is not an assertion about unavailable corpora or future training. All BN1 parents are reserved for evaluation and must never enter MTP training.

Manifest `input_tokens` is the exact production `tokenizer.json` count of the stripped prompt with `add_special_tokens=False`, excluding chat-template overhead. The tokenizer hash and per-prompt SHA-256 are included. Prompt token ranges: code-edit 12511–13641, prose-en 131–138, prose-ja 180–191, agent-loop 284–318. Runtime usage provides the templated input count. A unique fixture is one parent; technical repeats are not additional parents.

## Paired analysis

`bench/stats/paired_ab.py --schedule ABAB|ABBA --arms <four chronological JSONLs>` requires the same complete prompt/repeat grid, hashes, budgets, model, sampling, engine and set version. Missing pairs, duplicate records or arm files, identical prompts under different IDs, invalid acceptance and unbalanced repeats fail. It never silently intersects incomplete arms.

For each metric/domain, average technical repeats in log space within each prompt and arm. Form the adjacent block contrasts `log(B1/A1)` and `log(B2/A2)` (the second pair reverses chronological order for ABBA), then average the two contrasts within prompt. Report each prompt's two log-ratios, both block means and the equal-parent pooled effect. Bootstrap whole prompts with both blocks attached (20000 draws, seed 20260908), providing a two-sided 95% percentile CI and a one-sided 95% lower bound. The drift diagnostic is `A2/A1`, with a flag for absolute drift above 3% or a CI excluding zero; repeated schedules report this separately for every cycle.

Multiple complete four-arm cycles can be supplied to the same analyzer in chronological order. It retains the prompt bootstrap across all blocks and additionally reports a crossed cycle+parent variance-component t interval (conservative min(cycles−1, prompts−1) df); repeated cycles are not silently counted as new prompts.

The prompt bootstrap is **conditional on the observed restarts**. A restart shift shared by every prompt is not resampled by a prompt bootstrap; a narrow conditional CI is not sufficient evidence against restart noise. ABBA balances a linear chronological drift with equally spaced arm windows; it cannot cancel arbitrary drift or carryover. Actual timestamps and gaps are retained.

MDE uses a one-sided noncentral-t power calculation at alpha=.05, power=.80; effect is expressed as `100 * expm1(delta_log)`. `paired_ab.py` reports conditional MDE at observed n and plug-in power at 3%. These are model-based planning estimates, not measured power. See [NIST paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) and [NIST sample-size planning](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm).

## Variance study design

Production w4: A1 → A2 → A3; production wa: A1 → A2 → A3. Each is a new server process, same frozen source/configuration, 32 requests, one repeat. Freeze includes 3788 source/prompt/config inputs, production revision/diff, tokenizer/config hashes, token map hash and model file size/mtime/resolved-path inventory. Weight bytes are not rehashed (model inventory is a drift check, not a cryptographic weight attestation). Launcher cache is copied to a private runtime directory; a read-only production bind mount plus private cache bind prevents launcher/venv/source writes. Each arm is enclosed by `flock -w 28800 $HOME/.gpu.lock`. Memory fraction .920, `SERVE_DISPLAY_HZ=`; monitor requires >=4096 MiB free during startup and serving. Only owned process groups are signaled at cleanup. No broad process-name termination.

Explicit launcher arguments and environment are identical within each profile. Production leaves `random_seed` unset, and SGLang generates a new server seed at each startup (`server_args.py`); these resolved seed values are retained in every `server-info.json`. Greedy requests do not imply deterministic inference (`enable_deterministic_inference=False`). The study estimates ordinary production restart noise, including this default seed behavior; it is not a fixed-seed determinism experiment. Runtime `startup_time` and `internal_states` also vary and are not configuration drift.

For log metric y[r,p], fit crossed restart + prompt effects. Residual variance is `sum((y - restart_mean - prompt_mean + grand_mean)^2) / ((R-1)(n-1))`. Restart variance component is `max(0, var(restart_mean) - residual_variance/n)`. Report raw across-prompt within-restart CV (difficulty + noise), raw CV of restart means, and lognormal residual/restart component CVs. With one observation/cell, request noise and prompt×restart interaction cannot be separated. A three-restart estimate has only two restart degrees of freedom.

For a future 2A+2B schedule, conditional SE is `sqrt(residual_variance/n)`; restart-aware SE is `sqrt(restart_variance + residual_variance/n)`. The latter retains 2 pilot restart df in planning rather than pretending n prompts estimate restart uncertainty. n=16 extrapolates exchangeable new independent parents. Technical repeats may reduce request noise but not the common restart term. Independent four-arm cycles reduce both terms; prospective options are reported with this explicit plug-in assumption.

Both profiles are complete; measured tables and the precision-based protocol appear below. No candidate optimization was tested and no quality/adoption verdict is inferred from this noise study.

## How to run

From the benchmark root, use the existing `.venv-review/bin/python -B bench/stats/run_variance.py freeze specs/bn1/<unique-study>` once for CPU preparation, then the same Python with `sequence specs/bn1/<unique-study>` on the GPU-visible host; it obtains the shared flock separately for each server and holds it through cleanup. Generate the variance report with the existing production Python, `-B bench/stats/variance.py specs/bn1/<unique-study> --out specs/bn1/<unique-study>/variance.json` (numpy/scipy are already present; no venv changes). For a future A/B task, reuse the same sets via `python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang --workloads code-edit,prose-en,prose-ja,agent-loop --prompt-sets workloads/sets --prompt-limit 8 --repeats 1 --sampling greedy --require-acceptance --allow-proc sglang --out <arm>.jsonl` after the one fixed non-holdout warmup, with each task's launcher under the same per-server lock. Analyze four chronological files with the production Python `-B bench/stats/paired_ab.py --schedule ABBA --arms A1.jsonl B1.jsonl B2.jsonl A2.jsonl --out paired.json`. Predeclare the effect/non-regression thresholds and primary domains; a detected nonzero effect and proof of a lower bound above 3% are different hypotheses.

## CPU validation

- Full existing test suite: 22 passed.
- `bench/stats/test_bn1.py`: 6 passed. Checks exact counter ratios, missing/reset/lag/contamination cases, set expansion, no per-holdout warmup, strict missing-pair rejection, ABBA cancellation of linear drift, preservation of shared uncertainty across repeated cycles, power inversion, and a synthetic common-restart shift that does not disappear at n=16.
- Integrated artifact validation: 192 valid requests; all 3788 frozen input hashes and 206 model-file metadata entries unchanged; all five provenance source hashes unchanged. Identical explicit launcher command/environment within each profile.
- GPU memory: minimum 6465 MiB free across all recorded startup and serving samples (required minimum 4096 MiB); fraction .920 and empty display setting verified in every arm.
- Duration validation: 191/192 within 15–60 seconds. The 9.847-second wa Japanese response is a disclosed target failure, not an invalid counter measurement. `validation.json` reports `PASS_WITH_DURATION_EXCEPTION`.

## Duration calibration pilot

`specs/bn1/study-v1/w4-A1` is an excluded, intentionally interrupted length pilot, not a variance arm. The first code/en/ja/agent requests took 34.13/16.24/14.10/17.15 seconds. The first Japanese output ended naturally at 3532 tokens. Final prose and agent requests use longer authored output-length instructions and higher fixed budgets; EOS is respected (no minimum-token masking, padding, or forced continuation). The original pilot prompts and generator are preserved under `study-v1/prompt-snapshot` and `build_sets_pilot.py`. The final study uses `study-v2`; its budget is 6880 seconds after deducting the pilot's 318.4 seconds from the 2-hour total.

## Completed arms

| Arm | Requests | Lock seconds | Duration range (s) | Invalid acceptance |
|---|---:|---:|---:|---:|
| w4-A1 | 32 | 916.8 | 18.70–34.18 | 0 |
| w4-A2 | 32 | 927.6 | 19.31–34.49 | 0 |
| w4-A3 | 32 | 927.0 | 16.69–34.84 | 0 |
| wa-A1 | 32 | 869.6 | 19.66–25.99 | 0 |
| wa-A2 | 32 | 827.1 | 9.85–25.54 | 0 |
| wa-A3 | 32 | 839.8 | 17.55–24.97 | 0 |

### wa A2 duration exception (retained)

`bn1-prose-ja-02` completed all 6144 allowed tokens in 9.847 seconds (629.399 t/s, 7.623 tokens/verify). This is **not** an early-EOS response: the previous wa A1 value for the same prompt was 262.234 t/s and 2.440 tokens/verify over 23.482 seconds. Its valid counters and completed request remain in every estimate. A fixed token cap cannot guarantee a 15-second minimum when throughput changes this much. The duration target therefore has an observed exception; no prompt is dropped, padded or retrospectively retuned.

### A2 preflight drain adjustment

The first A2 lock acquisition saw the preceding owner's PIDs as `[No data]` in NVML and aborted before starting any BN1 server or request. Evidence is preserved in `study-v2/w4-A2-preflight-1`. The driver now waits up to120 seconds under the lock for previous compute processes to disappear, without signaling any unowned PID. This changes pre-start ownership checking only, not the frozen server or request protocol. Failed preflight occupied time is counted from cleanup events. A1 remains valid and the same A2 is retried.

The completed w4 A1/A2 runs also kept startup free VRAM above the 4GiB floor: minimum6465/6505MiB, versus steady minima10446/10471MiB. The watchdog now enforces the same floor during startup as well as serving; this guard-only tightening changes no server/request configuration.

For subsequent arms the lock is also held after owned process-group exit until NVML reports no remaining compute contexts (bounded120-second drain, no unowned signals), preventing the same asynchronous teardown race from being passed to the next lock owner. This occurs after all timed requests.

## Measured variance results

All values below are percentages; acceptance is completed tokens per verify. Within-restart CV is the mean of the three per-restart raw CVs (each retained in JSON); it includes parent difficulty. Residual CV removes the fixed parent effect.

| Profile | Domain | Mean within raw CV t/s / accept | Between restart mean CV t/s / accept | Residual CV t/s / accept | Restart component CV t/s / accept |
|---|---|---:|---:|---:|---:|
| w4 | agent-loop | 3.06 / 3.19 | 0.36 / 0.56 | 2.72 / 2.87 | 0.00 / 0.00 |
| w4 | code-edit | 1.93 / 1.82 | 0.31 / 0.18 | 1.39 / 1.46 | 0.00 / 0.00 |
| w4 | prose-en | 5.24 / 5.53 | 2.62 / 2.64 | 5.86 / 6.19 | 1.26 / 1.03 |
| w4 | prose-ja | 5.42 / 6.15 | 2.42 / 2.66 | 5.90 / 6.62 | 1.10 / 1.11 |
| wa | agent-loop | 3.82 / 5.53 | 0.41 / 1.44 | 3.87 / 5.99 | 0.00 / 0.00 |
| wa | code-edit | 7.99 / 9.08 | 2.55 / 3.48 | 7.61 / 8.78 | 0.00 / 1.68 |
| wa | prose-en | 4.99 / 5.50 | 1.90 / 2.22 | 3.16 / 3.59 | 1.46 / 1.69 |
| wa | prose-ja | 18.09 / 25.83 | 9.48 / 14.23 | 19.59 / 26.10 | 0.00 / 2.45 |

### MDE at 80% plug-in power, one-sided alpha=.05

Each cell is throughput / acceptance. Conditional = prompt residual only. Restart-aware includes common restart variance with 2 pilot restart df. n=16 is extrapolated. These estimates do not certify achieved power.

| Profile | Domain | n=8 conditional | n=16 conditional | n=8 restart-aware | n=16 restart-aware |
|---|---|---:|---:|---:|---:|
| w4 | agent-loop | 2.69 / 2.85 | 1.78 / 1.89 | 3.89 / 4.12 | 2.74 / 2.90 |
| w4 | code-edit | 1.37 / 1.44 | 0.91 / 0.96 | 1.97 / 2.08 | 1.39 / 1.47 |
| w4 | prose-en | 5.90 / 6.23 | 3.89 / 4.11 | 10.13 / 10.10 | 8.00 / 7.68 |
| w4 | prose-ja | 5.94 / 6.68 | 3.92 / 4.40 | 9.82 / 10.84 | 7.58 / 8.24 |
| wa | agent-loop | 3.86 / 6.03 | 2.55 / 3.98 | 5.59 / 8.79 | 3.92 / 6.14 |
| wa | code-edit | 7.72 / 8.95 | 5.08 / 5.88 | 11.29 / 15.06 | 7.86 / 11.61 |
| wa | prose-en | 3.14 / 3.58 | 2.08 / 2.37 | 7.60 / 8.79 | 6.84 / 7.93 |
| wa | prose-ja | 20.90 / 28.54 | 13.48 / 18.21 | 31.39 / 45.36 | 21.29 / 31.44 |

### Prospective 3% planning options

Each cell is the required independent four-arm cycles for throughput / acceptance, reaching 80% model-based power with conservative pilot df=2. Requests per domain = 4 × prompts × cycles (one repeat); running all four domains multiplies this by four. These are planning options, not authorization for additional GPU runs or a guarantee of real-world power.

| Profile | Domain | 8 prompts: cycles t/s / accept | 16 prompts: cycles t/s / accept |
|---|---|---:|---:|
| w4 | agent-loop | 2 / 2 | 1 / 1 |
| w4 | code-edit | 1 / 1 | 1 / 1 |
| w4 | prose-en | 11 / 11 | 7 / 7 |
| w4 | prose-ja | 11 / 13 | 7 / 8 |
| wa | agent-loop | 4 / 9 | 2 / 5 |
| wa | code-edit | 14 / 23 | 7 / 14 |
| wa | prose-en | 7 / 9 | 6 / 7 |
| wa | prose-ja | 86 / 161 | 43 / 86 |


## Recommended default gate protocol

Use **eight frozen independent parents per domain, one measured repeat, and fresh-server ABBA cycles**. Keep the exact round-robin prompt order, greedy sampling, caps, one discarded non-holdout warmup, per-server lock, allocator/memory settings and counter validation. Predeclare the primary domain/metric, cycle count, alpha and engineering threshold before evaluating a candidate. Technical repeats are not additional independent parents. Keep all valid requests, including the duration exception; a broken counter or incomplete arm invalidates the complete pairing rather than silently reducing n.

A single ABBA is a useful screening unit but **does not support a blanket 3% gate**. For a 3% throughput effect versus zero at one-sided alpha=.05 and >=80% plug-in power, a common schedule covering any of the four domains needs **11 cycles for w4** (1408 total requests, minimum modeled power 80.93%) or **86 cycles for wa** (11008 requests, minimum modeled power 80.25%). If acceptance is also eligible to be the predeclared 3% primary metric, use **13 w4 cycles** (1664 requests, minimum 82.05%) or **161 wa cycles** (20608 requests, minimum 80.16%). These are per-endpoint power statements, not simultaneous family-wise power guarantees. Domain-specific counts in the preceding table reduce work for a predeclared narrow primary endpoint; use at least three cycles when relying on the crossed-cycle interval.

Measured full-suite arm durations imply approximately **11.3 / 80.8 occupied GPU-hours** for the w4 / wa throughput plans, or **13.3 / 151.2 hours** for the either-metric plans. These costs are the consequence of the observed variance, especially wa Japanese, and are not an instruction to run them automatically. The practical default is a **budgeted, precision-based gate**: select the primary endpoint and fixed cycle count from the pilot before starting; if the budget cannot support that count, label the 3% decision **inconclusive**. One or two lock-hours cannot substantiate a universal 3% claim under the measured production noise. Noise-control work needs a new frozen pilot before advertising a cheaper 3% protocol.

Sixteen independent parents project to seven / 43 cycles for w4 / wa throughput, or eight / 86 for either metric. Only eight parents/domain were authored and measured in BN1; n=16 requires eight additional independent held-out parents per domain, a new frozen/audited set, and duration validation. It is not sixteen repetitions or variants of these eight parents. With the observed wa Japanese residual, doubling n halves throughput cycles while leaving essentially the same request count.

For final inference, report the required prompt bootstrap (20000 draws, fixed seed20260908), both block means, per-prompt log ratios and each cycle's A2/A1 drift. With repeated cycles, use the analyzer's **crossed cycle + parent one-sided 95% lower bound above zero** for the predeclared improvement test; the prompt-only bootstrap remains conditional evidence. Do not erase flagged drift blocks or stop at the first favorable result. A true 3% effect being detectable against zero is different from establishing a lower bound above +3%; the latter needs a separately powered hypothesis. Acceptance non-inferiority at a 1% margin and multiple-endpoint adoption rules likewise require their own planning.

Power estimates assume independent future cycles and comparable log variance. Only three restart controls were observed, the wa Japanese outlier makes tail extrapolation fragile, and a candidate's parent-specific effects are unknown. Retaining two pilot restart degrees of freedom is a conservative planning choice, not a proof that actual future power is >=80%. Negative estimated restart components are truncated to zero; a displayed 0.00 is not evidence of zero restart noise.

The ordinary production launcher generates a server seed on each restart: w4 seeds390163817 / 119776054 / 515489276; wa seeds534201594 / 734711191 / 761618367. This is the same explicit production configuration, not fixed-seed execution. wa A2/A1 geometric drift was +14.08% throughput and +20.27% tokens/verify for Japanese; the same prompt02 returned to252.61 t/s and2.366 tokens/verify in A3. The full event and counter evidence is retained.

## Artifact index and reproducibility

- [Frozen source/config/model inventory](bn1/study-v2/frozen.json)
- [Variance components, aggregate counters, drift and MDE](bn1/study-v2/variance.json)
- [Integrated validation, seeds, memory minima and per-prompt durations](bn1/study-v2/validation.json)
- [Machine-readable protocol sizes, powers and occupied-time projections](bn1/study-v2/protocol.json)
- [w4 A2/A1 prompt bootstrap](bn1/study-v2/w4-A2-over-A1.json), [wa A2/A1 prompt bootstrap](bn1/study-v2/wa-A2-over-A1.json)
- Every `bn1/study-v2/{w4,wa}-A{1,2,3}/` contains `requests.jsonl`, `command.json`, `server-info.json`, `server.log`, `events.jsonl`, `memory.jsonl`, `durations.json` and completion/validation markers. The private runtime cache is ignored by Git.
- [Prompt manifests and provenance instructions](../../bench/workloads/sets/README.md), [corpus-audit method note](../../bench/workloads/sets/provenance-audit.json) (the audit ran against private prompt files; its result is not published)

To reproduce CPU aggregation, use `$HOME/tools/sglang-rtxpro6000/.venv/bin/python -B bench/stats/variance.py specs/bn1/study-v2 --out specs/bn1/study-v2/variance.json` and the same Python with `bench/stats/validate_study.py specs/bn1/study-v2 --out specs/bn1/study-v2/validation.json`. The production venv is used read-only and requires no installation. Future multi-cycle paired analysis passes all chronological arm paths to `paired_ab.py --schedule ABBA --arms ... --out paired.json`; each four-path group must be A1,B1,B2,A2. Use the earlier run paragraph for the exact fnbench command and per-server lock contract.
