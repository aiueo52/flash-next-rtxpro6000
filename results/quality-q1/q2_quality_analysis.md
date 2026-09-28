| benchmark | prod | noprune | legacy | prod2 |
|---|---|---|---|---|
| gsm8k | 88.63% [86.8, 90.2] (1169/1319) | 89.31% [87.5, 90.9] (1178/1319) | 89.46% [87.7, 91.0] (1180/1319) | 89.31% [87.5, 90.9] (1178/1319) |
| mmlu | 85.36% [83.4, 87.1] (1195/1400) | 85.07% [83.1, 86.8] (1191/1400) | 84.93% [83.0, 86.7] (1189/1400) | 84.86% [82.9, 86.6] (1188/1400) |
| humaneval | 96.95% [93.1, 98.7] (159/164) | 96.34% [92.2, 98.3] (158/164) | 96.95% [93.1, 98.7] (159/164) | 95.73% [91.5, 97.9] (157/164) |
| jcqa | 97.14% [96.0, 98.0] (1087/1119) | 97.14% [96.0, 98.0] (1087/1119) | 97.14% [96.0, 98.0] (1087/1119) | 97.14% [96.0, 98.0] (1087/1119) |
| wall s, summary.json timing (gsm8k/mmlu/he/jcqa) | 1089.7/93.3/74.2/67.6 | 1181.9/110.2/81.8/70.0 | 1202.2/108.7/81.0/68.5 | 1082.5/119.2/74.7/66.0 |
| mean gen tokens | 342.9/2.1/230.6/2.0 | 339.4/2.1/230.9/2.0 | 339.9/2.1/229.3/2.0 | 343.5/2.2/234.5/2.0 |

Paired vs prod (McNemar exact, two-sided; flips = base-right/other-wrong : base-wrong/other-right):

| benchmark | arm | diff (pp) | diff 95% CI (pp) | flips +/- | p | n |
|---|---|---|---|---|---|---|
| gsm8k | noprune | +0.68 | [-0.64, +2.03] | 34/43 | 0.362 | 1319 |
| mmlu | noprune | -0.29 | [-1.04, +0.43] | 14/10 | 0.541 | 1400 |
| humaneval | noprune | -0.61 | [-3.37, +1.69] | 1/0 | 1.000 | 164 |
| jcqa | noprune | +0.00 | [-0.34, +0.34] | 0/0 | 1.000 | 1119 |
| gsm8k | legacy | +0.83 | [-0.51, +2.21] | 35/46 | 0.266 | 1319 |
| mmlu | legacy | -0.43 | [-1.31, +0.41] | 20/14 | 0.392 | 1400 |
| humaneval | legacy | +0.00 | [-2.82, +2.82] | 1/1 | 1.000 | 164 |
| jcqa | legacy | +0.00 | [-0.34, +0.34] | 0/0 | 1.000 | 1119 |
| gsm8k | prod2 | +0.68 | [-0.62, +2.01] | 33/42 | 0.356 | 1319 |
| mmlu | prod2 | -0.50 | [-1.49, +0.46] | 26/19 | 0.371 | 1400 |
| humaneval | prod2 | -1.22 | [-4.34, +1.10] | 2/0 | 0.500 | 164 |
| jcqa | prod2 | +0.00 | [-0.34, +0.34] | 0/0 | 1.000 | 1119 |

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

Non-inferiority of prod vs each other arm (paired BASE minus other; Tango score bounds, z=1.644854).
Lower and upper bounds are each one-sided 95% bounds (not a two-sided 95% interval). PASS: lower > -margin; FAIL: upper < -margin; otherwise INCONCLUSIVE.
Tango score bounds stay valid with few or zero discordant pairs (a Wald interval collapses there). The significance verdict above is not a non-inferiority verdict.

| benchmark | arm | BASE - other (pp) | lower 95% (pp) | upper 95% (pp) | margin (pp) | status | n |
|---|---|---|---|---|---|---|---|
| gsm8k | noprune | -0.68 | -1.80 | +0.42 | 0.5 | INCONCLUSIVE | 1319 |
| mmlu | noprune | +0.29 | -0.31 | +0.90 | 0.5 | PASS | 1400 |
| humaneval | noprune | +0.61 | -1.02 | +2.69 | 0.5 | INCONCLUSIVE | 164 |
| jcqa | noprune | -0.00 | -0.24 | +0.24 | 0.5 | PASS | 1119 |
| gsm8k | legacy | -0.83 | -1.98 | +0.29 | 0.5 | INCONCLUSIVE | 1319 |
| mmlu | legacy | +0.43 | -0.27 | +1.15 | 0.5 | PASS | 1400 |
| humaneval | legacy | -0.00 | -2.14 | +2.14 | 0.5 | INCONCLUSIVE | 164 |
| jcqa | legacy | -0.00 | -0.24 | +0.24 | 0.5 | PASS | 1119 |
| gsm8k | prod2 | -0.68 | -1.79 | +0.40 | 0.5 | INCONCLUSIVE | 1319 |
| mmlu | prod2 | +0.50 | -0.30 | +1.32 | 0.5 | PASS | 1400 |
| humaneval | prod2 | +1.22 | -0.42 | +3.62 | 0.5 | PASS | 164 |
| jcqa | prod2 | -0.00 | -0.24 | +0.24 | 0.5 | PASS | 1119 |
