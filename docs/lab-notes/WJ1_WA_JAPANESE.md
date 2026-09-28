# WJ1 — Japanese prose diagnostic

Status: COMPLETE, 2026-09-08. Japanese regression inconclusive; recommend retaining current wa and X3. No production changes, commits or pushes.

## Predeclared protocol

Read BN1_BENCH_NOISE.md including its per-prompt exception, artifacts and statistical limitations, and WA5_LOG.md. Current launcher confirms A = three candidates [3,7,15], STEP_A/B 7.943/0.5554. B = previous [3,15] config, STEP_A/B 9.74/0.70. X3 target autotune remains enabled in both via the current launcher. Only these three policy environment values differ; tracing/debug are identical in both.

One fresh-server ABBA cycle A1,B1,B2,A2. Eight frozen prose-en and eight prose-ja parents, interleaved en01,ja01,...,en08,ja08; one greedy request each, cap 6144. Same single discarded 1024-token non-holdout BN1 warmup per restart. No per-prompt warmup, retries or output padding. Acceptance is completed tokens / verify calls from BN1 request-bracketed counters, not a draft acceptance probability.

Primary endpoint: equal-parent mean log B/A Japanese decode t/s. Report both blocks, every prompt, 20,000 parent bootstrap draws seed 20260908 and two-sided 95% conditional CI. B improvement CI wholly above zero supports Japanese regression in this diagnostic; +3% is an engineering reference, not a powered universal gate. English and tokens/verify are controls/secondary endpoints. A2/A1 drift and arbitrary restart uncertainty remain explicit; one cycle cannot estimate independent cycle variance.

Budget: at most 3600 occupied lock seconds, queue excluded. Each server runs under flock -w 28800 $HOME/.gpu.lock, including cleanup/drain. Fraction .920, MAX_TOTAL_TOKENS=131072, SERVE_DISPLAY_HZ empty. Private launcher cache and read-only production bind, PYTHONDONTWRITEBYTECODE=1. VRAM watchdog >=4096 MiB during startup and steady state. Only owned process groups signaled. No production tree, launcher or venv modification.

Historical BN1 driver did not enable SGLANG_ADAPTIVE_TRACE or DEBUG and no per-step trace exists in its wa A1-A3 artifact directories. Width switch logs have whole-second timestamps; exact mixed-width verify counts cannot be reconstructed from timestamps alone. Report exact total verify counters, width paths/switches and explicitly bounded/estimated occupancy where necessary. New WJ1 arms enable traces for direct per-step counts.

A second ABBA for a minimal fix is conditional on demonstrated harm and sufficient remaining occupied budget for a complete cycle; no partial candidate cycle or favorable-result stopping.

## Historical BN1 Japanese audit

No step traces were recorded by BN1. Pure-width rows are exact (total request verify counter plus no switch in the whole-second boundary envelope). Mixed rows show conservative interior 40-step log-interval counts plus unallocated boundary/switch steps U; these are not exact occupancy counts. An extra log interval either side of a switch is excluded to accommodate pipeline ordering. Switches are certain–possible counts under whole-second timestamp uncertainty. W is draft steps + 1.

| Arm | Prompt | t/s | tokens/verify | W4 / W8 / W16 known | U | Switches | Width path |
|---|---|---:|---:|---|---:|---|---|
| wa-A1 | 01 | 290.99 | 2.814 | 1560 / 80 / 0 | 543 | 3–5 | 4→8→4→8→4→8 |
| wa-A1 | 02 | 262.23 | 2.440 | 2515 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A1 | 03 | 269.23 | 2.501 | 2457 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A1 | 04 | 272.11 | 2.520 | 2438 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A1 | 05 | 261.37 | 2.483 | 2000 / 0 / 0 | 474 | 4–5 | 4→8→4→8→4→8 |
| wa-A1 | 06 | 270.62 | 2.507 | 2451 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A1 | 07 | 282.43 | 2.653 | 2000 / 40 / 0 | 276 | 1–1 | 4→8 |
| wa-A1 | 08 | 270.91 | 2.534 | 2120 / 0 / 0 | 305 | 2–3 | 4→8→4→8 |
| wa-A2 | 01 | 286.43 | 2.789 | 1840 / 0 / 0 | 363 | 3–3 | 8→4→8→16 |
| wa-A2 | 02 | 629.40 | 7.623 | 360 / 0 / 160 | 286 | 2–2 | 4→8→16 |
| wa-A2 | 03 | 266.64 | 2.467 | 2320 / 0 / 0 | 170 | 0–1 | 4→8 |
| wa-A2 | 04 | 264.84 | 2.467 | 2440 / 0 / 0 | 50 | 0–1 | 4→8 |
| wa-A2 | 05 | 267.37 | 2.471 | 2486 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A2 | 06 | 347.31 | 3.907 | 640 / 400 / 0 | 328 | 2–2 | 16→4→8 |
| wa-A2 | 07 | 273.16 | 2.553 | 2407 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A2 | 08 | 268.82 | 2.484 | 2200 / 0 / 0 | 122 | 0–1 | 4→8 |
| wa-A3 | 01 | 284.27 | 2.654 | 2120 / 0 / 0 | 195 | 1–1 | 4→8 |
| wa-A3 | 02 | 252.61 | 2.366 | 2200 / 0 / 0 | 397 | 2–2 | 4→8→4 |
| wa-A3 | 03 | 304.98 | 2.955 | 1720 / 120 / 0 | 239 | 1–1 | 4→8 |
| wa-A3 | 04 | 264.59 | 2.450 | 2360 / 0 / 0 | 148 | 0–1 | 4→8 |
| wa-A3 | 05 | 252.60 | 2.338 | 2240 / 0 / 0 | 0 | 0–0 | 4 |
| wa-A3 | 06 | 351.56 | 3.911 | 760 / 0 / 0 | 811 | 7–7 | 8→4→8→4→8→16→8→16 |
| wa-A3 | 07 | 261.94 | 2.442 | 2400 / 0 / 0 | 116 | 0–1 | 4→8 |
| wa-A3 | 08 | 299.71 | 2.908 | 1640 / 160 / 0 | 262 | 1–1 | 4→8 |

Across 24 prompt×restart rows, correlation of t/s with the conservative known W8+W16 verify fraction: Pearson r=0.668, Spearman rho=0.670. This lower-bound proxy is descriptive, not exact occupancy or a causal policy estimate; repeated parents are not independent samples.


## Trace accounting clarification during A1

Production `batch_result_processor.py::_resolve_spec_v2_output` invokes `on_verify_complete_cpu` before checking `req.finished()`. The trace therefore includes an optional extra overlap result for an already-finished request, while `req.spec_verify_ct` is incremented only for unfinished requests. WJ1 aligns the first N traced results with the completed-request counter N and records any single trailing executed result separately. It checks that the extra result is near/after client completion, all widths and accepted-draft counts are valid, and token sums differ from client completion only by the allowed final cap/bonus boundary. Both raw executed and counter-aligned occupancy are retained. This is an analysis clarification, not a server/harness/protocol change.

The production default random seed remains unset, as in BN1; actual per-restart seeds are retained in server-info.json. Greedy sampling is not deterministic-inference mode. The experiment compares the complete old and current policy configurations, including different STEP cost models and candidate graph sets, not an isolated causal effect of candidate count alone.

Completed `wj1-20260908-101958-A1`: 16 valid requests; lock 482.9 seconds.

Completed `wj1-20260908-102801-B1`: 16 valid requests; lock 502.8 seconds.

Completed `wj1-20260908-103624-B2`: 16 valid requests; lock 489.2 seconds.

## Three-arm checkpoint (not the final paired estimate)

| Arm | Japanese geometric t/s | English geometric t/s | Lock seconds |
|---|---:|---:|---:|
| A1 | 278.81 | 252.69 | 482.9 |
| B1 | 296.22 | 221.42 | 502.8 |
| B2 | 274.18 | 240.30 | 489.2 |

A2 is running. No decision or candidate change is made from this checkpoint. B1 Japanese prompt06 completed quickly enough to miss the BN1 15-second duration target; it is retained with valid counters.

Completed `wj1-20260908-104433-A2`: 16 valid requests; lock 472.3 seconds.


## WJ1 measured arms

Width counts below cover the first N verified results, where N is the exact completed-request verify counter. The existing CPU trace is emitted before the req.finished() guard; an optional single trailing overlap result after completion is recorded separately as executed work and excluded from these counter-aligned counts. Source: batch_result_processor.py _resolve_spec_v2_output and eagle_worker_v2.py on_verify_complete_cpu. Switches count changes between consecutive measured verify steps within a request; a policy decision after its final verify is not a measured switch. Trace token sums may exceed capped client completion by up to one verify width; exact counter acceptance remains primary.

| Arm | Domain | t/s geometric mean | acceptance geometric mean | W4 / W8 / W16 verifies | Width % 4 / 8 / 16 | Switches |
|---|---|---:|---:|---|---|---:|
| A1 | prose-ja | 278.81 | 2.699 | 16750 / 1277 / 247 | 91.66 / 6.99 / 1.35 | 10 |
| A1 | prose-en | 252.69 | 2.424 | 19405 / 752 / 101 | 95.79 / 3.71 / 0.50 | 10 |
| B1 | prose-ja | 296.22 | 2.920 | 16779 / 0 / 793 | 95.49 / 0.00 / 4.51 | 5 |
| B1 | prose-en | 221.42 | 2.450 | 14997 / 0 / 5072 | 74.73 / 0.00 / 25.27 | 5 |
| B2 | prose-ja | 274.18 | 2.609 | 18491 / 0 / 181 | 99.03 / 0.00 / 0.97 | 2 |
| B2 | prose-en | 240.30 | 2.407 | 18785 / 0 / 1647 | 91.94 / 0.00 / 8.06 | 4 |
| A2 | prose-ja | 267.36 | 2.571 | 15938 / 1501 / 0 | 91.39 / 8.61 / 0.00 | 12 |
| A2 | prose-en | 250.35 | 2.394 | 19515 / 621 / 0 | 96.92 / 3.08 / 0.00 | 9 |

| Arm | Lock seconds | Min free MiB startup+steady | Min free MiB steady |
|---|---:|---:|---:|
| wj1-20260908-101958-A1 | 482.9 | 6259 | 9355 |
| wj1-20260908-102801-B1 | 502.8 | 6258 | 10568 |
| wj1-20260908-103624-B2 | 489.2 | 6290 | 10600 |
| wj1-20260908-104433-A2 | 472.3 | 6290 | 9370 |

Occupied lock seconds: 1947.2; queue time excluded.


## Paired policy comparison

Percent changes are exp(mean log B/A) − 1, equal weighting of eight parents and two blocks. CI is the 20,000-draw two-sided 95% parent bootstrap, conditional on these four restarts.

| Domain | Metric | Block 1 mean ln(B1/A1) / % | Block 2 mean ln(B2/A2) / % | Pooled B/A % | 95% CI % | A2/A1 drift % [95% CI] | Drift flag |
|---|---|---:|---:|---:|---|---|---|
| prose-en | tps | -0.13210 / -12.37 | -0.04095 / -4.01 | -8.29 | [-15.85, -1.20] | -0.93 [-3.14, +1.45] | False |
| prose-en | acceptance | +0.01097 / +1.10 | +0.00530 / +0.53 | +0.82 | [-1.54, +3.06] | -1.21 [-3.83, +1.52] | False |
| prose-ja | tps | +0.06058 / +6.24 | +0.02518 / +2.55 | +4.38 | [-3.31, +16.88] | -4.11 [-12.40, +5.39] | True |
| prose-ja | acceptance | +0.07869 / +8.19 | +0.01479 / +1.49 | +4.78 | [-4.96, +20.69] | -4.76 [-15.93, +8.74] | True |

| Domain / prompt | ln B1/A1 t/s | ln B2/A2 t/s | Mean ln B/A t/s | ln B1/A1 acceptance | ln B2/A2 acceptance | Mean ln B/A acceptance |
|---|---:|---:|---:|---:|---:|---:|
| prose-en/01 | -0.05844 | -0.00675 | -0.03260 | -0.05716 | -0.00771 | -0.03244 |
| prose-en/02 | -0.00933 | +0.05745 | +0.02406 | -0.03244 | +0.06402 | +0.01579 |
| prose-en/03 | +0.03537 | -0.02909 | +0.00314 | +0.06182 | -0.02227 | +0.01977 |
| prose-en/04 | +0.04111 | -0.07925 | -0.01907 | +0.02467 | -0.00941 | +0.00763 |
| prose-en/05 | +0.03392 | -0.00501 | +0.01446 | +0.02348 | -0.01390 | +0.00479 |
| prose-en/06 | -0.30679 | -0.05178 | -0.17928 | -0.02219 | -0.07560 | -0.04889 |
| prose-en/07 | -0.39073 | -0.24585 | -0.31829 | +0.01408 | +0.07599 | +0.04503 |
| prose-en/08 | -0.40193 | +0.03271 | -0.18461 | +0.07548 | +0.03129 | +0.05338 |
| prose-ja/01 | -0.09510 | +0.08969 | -0.00271 | -0.17887 | +0.08254 | -0.04817 |
| prose-ja/02 | +0.17671 | -0.06047 | +0.05812 | +0.27607 | -0.09579 | +0.09014 |
| prose-ja/03 | -0.09787 | +0.03225 | -0.03281 | -0.02259 | +0.04797 | +0.01269 |
| prose-ja/04 | -0.01769 | +0.04806 | +0.01519 | -0.01306 | +0.04403 | +0.01548 |
| prose-ja/05 | -0.10419 | +0.11068 | +0.00324 | -0.16797 | +0.12716 | -0.02041 |
| prose-ja/06 | +0.72861 | +0.08400 | +0.40631 | +0.90485 | +0.08165 | +0.49325 |
| prose-ja/07 | -0.03908 | +0.03525 | -0.00192 | -0.08816 | +0.02328 | -0.03244 |
| prose-ja/08 | -0.06680 | -0.13802 | -0.10241 | -0.08074 | -0.19251 | -0.13662 |

## Exact per-request WJ1 occupancy

| Arm | Domain / prompt | t/s | tokens/verify | W4 / W8 / W16 | Switches |
|---|---|---:|---:|---|---:|
| A1 | prose-en/01 | 265.11 | 2.527 | 2383 / 48 / 0 | 1 |
| A1 | prose-ja/01 | 302.29 | 3.094 | 1610 / 221 / 155 | 3 |
| A1 | prose-en/02 | 262.87 | 2.564 | 2212 / 140 / 44 | 2 |
| A1 | prose-ja/02 | 260.70 | 2.446 | 2512 / 0 / 0 | 0 |
| A1 | prose-en/03 | 247.61 | 2.346 | 2619 / 0 / 0 | 0 |
| A1 | prose-ja/03 | 269.68 | 2.553 | 2314 / 93 / 0 | 1 |
| A1 | prose-en/04 | 239.21 | 2.304 | 2521 / 146 / 0 | 1 |
| A1 | prose-ja/04 | 263.55 | 2.448 | 2510 / 0 / 0 | 0 |
| A1 | prose-en/05 | 248.68 | 2.376 | 2506 / 80 / 0 | 2 |
| A1 | prose-ja/05 | 303.07 | 3.054 | 1690 / 230 / 92 | 2 |
| A1 | prose-en/06 | 251.14 | 2.419 | 2470 / 13 / 57 | 2 |
| A1 | prose-ja/06 | 278.02 | 2.720 | 1910 / 349 / 0 | 2 |
| A1 | prose-en/07 | 254.15 | 2.437 | 2302 / 186 / 0 | 1 |
| A1 | prose-ja/07 | 278.68 | 2.722 | 1977 / 280 / 0 | 1 |
| A1 | prose-en/08 | 253.74 | 2.427 | 2392 / 139 / 0 | 1 |
| A1 | prose-ja/08 | 277.63 | 2.636 | 2227 / 104 / 0 | 1 |
| B1 | prose-en/01 | 250.06 | 2.387 | 2574 / 0 / 0 | 0 |
| B1 | prose-ja/01 | 274.87 | 2.587 | 2375 / 0 / 0 | 0 |
| B1 | prose-en/02 | 260.43 | 2.482 | 2475 / 0 / 0 | 0 |
| B1 | prose-ja/02 | 311.09 | 3.224 | 1711 / 0 / 195 | 1 |
| B1 | prose-en/03 | 256.53 | 2.496 | 2380 / 0 / 82 | 2 |
| B1 | prose-ja/03 | 244.54 | 2.496 | 2151 / 0 / 311 | 1 |
| B1 | prose-en/04 | 249.25 | 2.361 | 2602 / 0 / 0 | 0 |
| B1 | prose-ja/04 | 258.93 | 2.416 | 2543 / 0 / 0 | 0 |
| B1 | prose-en/05 | 257.26 | 2.432 | 2526 / 0 / 0 | 0 |
| B1 | prose-ja/05 | 273.08 | 2.582 | 2354 / 0 / 26 | 1 |
| B1 | prose-en/06 | 184.79 | 2.366 | 1344 / 0 / 1253 | 1 |
| B1 | prose-ja/06 | 576.11 | 6.722 | 655 / 0 / 259 | 1 |
| B1 | prose-en/07 | 171.95 | 2.471 | 746 / 0 / 1740 | 1 |
| B1 | prose-ja/07 | 268.00 | 2.492 | 2463 / 0 / 2 | 1 |
| B1 | prose-en/08 | 169.76 | 2.618 | 350 / 0 / 1997 | 1 |
| B1 | prose-ja/08 | 259.69 | 2.431 | 2527 / 0 / 0 | 0 |
| B2 | prose-en/01 | 247.85 | 2.360 | 2603 / 0 / 0 | 0 |
| B2 | prose-ja/01 | 263.25 | 2.488 | 2469 / 0 / 0 | 0 |
| B2 | prose-en/02 | 270.80 | 2.571 | 2390 / 0 / 0 | 0 |
| B2 | prose-ja/02 | 299.53 | 2.970 | 1941 / 0 / 128 | 1 |
| B2 | prose-en/03 | 239.75 | 2.333 | 2532 / 0 / 101 | 1 |
| B2 | prose-ja/03 | 275.90 | 2.648 | 2267 / 0 / 53 | 1 |
| B2 | prose-en/04 | 232.46 | 2.380 | 2316 / 0 / 266 | 1 |
| B2 | prose-ja/04 | 276.53 | 2.609 | 2355 / 0 / 0 | 0 |
| B2 | prose-en/05 | 245.07 | 2.329 | 2638 / 0 / 0 | 0 |
| B2 | prose-ja/05 | 268.94 | 2.555 | 2196 / 0 / 0 | 0 |
| B2 | prose-en/06 | 245.75 | 2.330 | 2637 / 0 / 0 | 0 |
| B2 | prose-ja/06 | 295.17 | 2.795 | 2198 / 0 / 0 | 0 |
| B2 | prose-en/07 | 192.38 | 2.526 | 1152 / 0 / 1280 | 2 |
| B2 | prose-ja/07 | 261.96 | 2.457 | 2501 / 0 / 0 | 0 |
| B2 | prose-en/08 | 256.56 | 2.441 | 2517 / 0 / 0 | 0 |
| B2 | prose-ja/08 | 255.25 | 2.396 | 2564 / 0 / 0 | 0 |
| A2 | prose-en/01 | 249.53 | 2.379 | 2583 / 0 / 0 | 0 |
| A2 | prose-ja/01 | 240.67 | 2.291 | 1812 / 100 / 0 | 2 |
| A2 | prose-en/02 | 255.68 | 2.411 | 2548 / 0 / 0 | 0 |
| A2 | prose-ja/02 | 318.20 | 3.268 | 1449 / 431 / 0 | 1 |
| A2 | prose-en/03 | 246.82 | 2.386 | 2427 / 148 / 0 | 1 |
| A2 | prose-ja/03 | 267.14 | 2.524 | 2344 / 90 / 0 | 2 |
| A2 | prose-en/04 | 251.63 | 2.402 | 2447 / 80 / 0 | 2 |
| A2 | prose-ja/04 | 263.55 | 2.497 | 2319 / 142 / 0 | 3 |
| A2 | prose-en/05 | 246.30 | 2.362 | 2130 / 107 / 0 | 1 |
| A2 | prose-ja/05 | 240.77 | 2.250 | 2140 / 0 / 0 | 0 |
| A2 | prose-en/06 | 258.81 | 2.513 | 2355 / 90 / 0 | 2 |
| A2 | prose-ja/06 | 271.39 | 2.576 | 2302 / 83 / 0 | 1 |
| A2 | prose-en/07 | 246.00 | 2.341 | 2508 / 116 / 0 | 1 |
| A2 | prose-ja/07 | 252.89 | 2.400 | 1952 / 160 / 0 | 2 |
| A2 | prose-en/08 | 248.30 | 2.366 | 2517 / 80 / 0 | 2 |
| A2 | prose-ja/08 | 293.02 | 2.905 | 1620 / 495 / 0 | 1 |

Japanese within-policy descriptive correlations (wide verify fraction vs t/s; repeated parents, no causal attribution): `{"A": {"pearson": 0.8231194714950028, "spearman": 0.7168172780381192}, "B": {"pearson": 0.8572885743779092, "spearman": 0.39864528998458104}}`.


## Final verdict and recommendation

**Japanese regression: INCONCLUSIVE, not established by this diagnostic. Do not revert the wa default to the previous two-candidate configuration on this evidence. Retain the current three-width configuration and X3 fix. No production change was made.**

The predeclared Japanese throughput effect of B versus A is **+4.38%**, conditional two-sided 95% bootstrap CI **[-3.31%, +16.88%]**. Both block means favor B (+6.24%, +2.55%), but their parent-paired uncertainty includes no benefit and a loss. Japanese tokens/verify is +4.78% [-4.96%, +20.69%]. This is not proof of equivalence or no harm: a material Japanese regression remains compatible with this small experiment. A2/A1 Japanese throughput drift is -4.11% [-12.40%, +5.39%], flagged by the predeclared absolute 3% rule; acceptance drift is -4.76% and also flagged. One cycle provides no independent cycle-variance estimate. All CIs here are conditional on these four restarts, not restart-aware guarantees.

The English control argues against the proposed fallback: B/A throughput is **-8.29% [-15.85%, -1.20%]**, with both blocks negative (-12.37%, -4.01%), while acceptance changes only +0.82% [-1.54%, +3.06%]. This is conditional evidence of an English performance loss under the old full configuration, not a globally certified multi-domain gate.

Flapping exists: A has 22 within-request Japanese width switches across its two arms, B has 7. However, measured wide-width occupancy correlates positively with Japanese t/s within both policies: A Pearson r=0.823 / Spearman rho=0.717; B r=0.857 / rho=0.399. These are descriptive 16-row correlations with repeated parents, not independent causal tests. Historical BN1's conservative wide-occupancy proxy also correlates positively (r=0.668, rho=0.670). High across-parent CV alone is not a signed regression estimate. This run's Japanese raw t/s CVs are A1 5.72%, A2 9.85%, B1 35.65%, B2 5.75%; the high-speed tail occurs under B as well.

B1 Japanese06 is 576.11 t/s versus 278.02 / 271.39 in A1 / A2 and 295.17 in B2. Its paired mean log B/A is +0.40631, the main positive contribution to the pooled Japanese result. It remains in the primary estimate and CI. As a labeled influence diagnostic only, omitting this one parent gives -0.90% B/A; this does not replace the registered analysis. Output hashes are retained but full generated text is not saved by the BN1 client, so this experiment does not establish a same-text counterfactual or output-quality equivalence.

## Mechanism observations and the minimal-fix decision

B1 English06 contains 1,253 W16 and 1,344 W4 counter-aligned verify steps. Within this request, trace token yield is 2.417 at W16 versus 2.318 at W4, while mean adjacent same-width CPU trace callback gaps are 16.403 versus 9.453 ms. Its overall throughput is 184.79 t/s. Thus prolonged expensive W16 residence with little extra token yield is observed in the **old B configuration's English control**. These callback intervals include host scheduling and are not kernel-only timings; the two portions are different text, so they are not a fixed-passage intervention.

The B1 switch log and the unchanged policy arithmetic explain a concrete persistence mechanism. After alternating 15→3, 3→15, 15→3, 3→15, 15→3, the 10:33:35 promotion 3→15 raises reversal grace to 1,280 batches. English06 begins at 10:33:36.359 and retains W16 for 1,253 counted steps before the 10:33:56 demotion. The old config omits max_grace_batches and receives the implementation default 2,000; A explicitly caps it at 80. Each arm's grace-audit.json preserves the decisions and the grace derived from those decisions. This supports carryover/grace as a mechanism; it does not assign the entire aggregate policy effect to that one parameter.

Conversely, B1 Japanese06's 259 W16 steps yield 15.946 trace tokens/verify, compared with 3.087 over 655 W4 steps. Historical wa-A2 Japanese02 also reaches 629.40 t/s and 7.623 counter tokens/verify after 4→8→16. Wide-width acceptance does grow in these observed Japanese trajectories, so the blanket premise that Japanese prose gains no acceptance at W8/W16 is false for this set.

No third configuration was tested: the predeclared condition of established Japanese harm was not met. The first cycle occupied 1,947.2 seconds (32.453 minutes, 0.541 GPU-hours), leaving 1,652.8 seconds of the 3,600-second budget. Another comparable four-arm cycle would require about 1,947 seconds and would not fit. No partial second cycle was started.

A stronger numeric down_margin["3"] is not a justified fix from this experiment: it raises the hurdle for W8/W16→W4 and can prolong exactly the expensive wide-width residence being investigated. No Japanese/prose prior is justified by these measurements either. If a future two-candidate fallback study is desired, the smallest evidence-directed candidate is the old config **with max_grace_batches=80**, retaining X3 and explicitly freezing STEP_A/B; that addresses the observed legacy carryover defect. It is an untested future candidate, not a recommendation to change production or a demonstrated Japanese fix. For the current production A policy, retain the current config pending additional independent paired cycles if a tighter Japanese decision is required.

## Raw within-restart variation (eight different parents)

| Arm | Domain | Raw CV t/s % | Raw CV tokens/verify % |
|---|---|---:|---:|
| A1 | prose-ja | 5.72 | 9.20 |
| A1 | prose-en | 3.30 | 3.60 |
| B1 | prose-ja | 35.65 | 47.43 |
| B1 | prose-en | 18.39 | 3.48 |
| B2 | prose-ja | 5.75 | 7.25 |
| B2 | prose-en | 9.47 | 3.92 |
| A2 | prose-ja | 9.85 | 13.16 |
| A2 | prose-en | 1.87 | 2.20 |

## Raw executed occupancy including finished overlap tails

The main tables align to the completed-request counters. This complementary table counts all traced executed results, including the final result ignored by the finished-request guard. It makes that accounting choice fully reviewable.

| Arm | Domain | Raw W4 / W8 / W16 | Excluded finished tails |
|---|---|---|---:|
| A1 | prose-ja | 16752 / 1281 / 249 | 8 |
| A1 | prose-en | 19411 / 754 / 101 | 8 |
| B1 | prose-ja | 16783 / 0 / 797 | 8 |
| B1 | prose-en | 15004 / 0 / 5073 | 8 |
| B2 | prose-ja | 18497 / 0 / 183 | 8 |
| B2 | prose-en | 18793 / 0 / 1647 | 8 |
| A2 | prose-ja | 15942 / 1505 / 0 | 8 |
| A2 | prose-en | 19523 / 621 / 0 | 8 |

## Final validation and artifact index

64/64 complete, valid paired requests; all exact width audits pass. 63/64 durations meet the inherited BN1 15–60 second target. B1 Japanese06 takes 10.752 seconds and uses all 6144 allowed tokens; it is a retained fast-response exception. Minimum sampled free VRAM is 6258 MiB across startup/steady samples, and 9355 MiB at steady state, both above 4096 MiB. Fraction .920, display setting empty, and the 131072 token cap match in every arm.

3790 frozen source/config/input hashes and 206 model metadata entries are unchanged. Production HEAD and preexisting diff are unchanged; X3 446c801189 remains an ancestor. Runtime arguments differ only by the intended config file, ordinary random seeds and startup timing/internal state. STEP overrides are recorded in command.json and default A values are attested by the frozen launcher. Host-level process-group verification found no remaining owned processes. No commit, push, production/launcher/venv edit or installation was performed. The GPU lock was held separately through each server and its context cleanup.

| Arm | Label | Seed | Lock acquired JST | Cleanup JST |
|---|---|---:|---|---|
| A1 | wj1-20260908-101958-A1 | 126245161 | 2026-09-08T10:19:58+09:00 | 2026-09-08T10:28:01+09:00 |
| B1 | wj1-20260908-102801-B1 | 809840030 | 2026-09-08T10:28:01+09:00 | 2026-09-08T10:36:24+09:00 |
| B2 | wj1-20260908-103624-B2 | 999140617 | 2026-09-08T10:36:24+09:00 | 2026-09-08T10:44:33+09:00 |
| A2 | wj1-20260908-104433-A2 | 82752690 | 2026-09-08T10:44:33+09:00 | 2026-09-08T10:52:25+09:00 |

- [Paired effect, per-parent log ratios, block means and conditional bootstrap](wj1/study-20260908/paired.json); independently invoked paired_ab.py CLI produced identical parsed JSON in paired-cli.json.
- [Validation including precise per-request windows and runtime parity](wj1/study-20260908/validation.json).
- [Historical BN1 audit](wj1/study-20260908/historical.json), [correlations](wj1/study-20260908/correlations.json), [mechanism](wj1/study-20260908/mechanism.json), [leave-one-parent-out influence diagnostic](wj1/study-20260908/leave-one-parent-out-sensitivity.json).
- [Frozen inputs and production state](wj1/study-20260908/frozen.json).
- Each timestamped arm directory contains requests.jsonl, trace.jsonl.rank0, server.log, server-info.json, command.json, memory.jsonl, events.jsonl, durations.json, width-audit.json, grace-audit.json and completion markers. Historical paths remain specs/bn1/study-v2/wa-A{1,2,3}/server.log and requests.jsonl; BN1 has no step traces.
- WJ1-specific reproducibility scripts: bench/stats/run_wj1.py, wj1_history.py, wj1_report.py, wj1_mechanism.py, wj1_validate.py and wj1_finalize.py. The production runner/harness and original paired_ab.py were not edited.

CPU re-analysis: run the production Python with -B bench/stats/wj1_report.py specs/wj1/study-20260908. The registered paired command is:

```sh
$HOME/tools/sglang-rtxpro6000/.venv/bin/python -B bench/stats/paired_ab.py \
  --schedule ABBA --arms \
  specs/wj1/study-20260908/wj1-20260908-101958-A1/requests.jsonl \
  specs/wj1/study-20260908/wj1-20260908-102801-B1/requests.jsonl \
  specs/wj1/study-20260908/wj1-20260908-103624-B2/requests.jsonl \
  specs/wj1/study-20260908/wj1-20260908-104433-A2/requests.jsonl \
  --out /tmp/wj1-paired-reproduction.json
```
