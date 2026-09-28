# WS1: fixed speculation-width frontier, production stack (2026-09-07)

Stack: sglang-rtxpro6000 `serve-fast.sh w4` (mtpft5 head, tau=0.08 singleton-expert pruning, PDL, expandable_segments), widths overridden with `--speculative-num-steps N-1 --speculative-num-draft-tokens N`; W16 = the `w16` profile (mem fraction 0.93). One session, 13:37-14:04 local, desktop monitor on. Labels `ws11337-<arm>`; W6/W8/W12 started at the w4 mem fraction without any `MEM_FRACTION` override. Needle 18.5k: PASS on all six arms.

Harness: `calib/ws1_sweep.sh` -> `calib/ws1_validate.sh` (copy of prof/validate.sh with WORKLOADS/REPEATS as env vars, server args via SERVER_ARGS) -> profile_decode2 (code-edit, prose-en; 20 steps) + fnbench (code-edit, prose-en, prose-ja, agent-loop; repeats 2; greedy) + needle. Numbers extracted by `calib/ws1_report.py` (raw output in `calib/logs/ws1_report.txt`, sweep log in `calib/logs/ws1_sweep.log`).

Acceptance = tokens emitted per verify step (bonus token included) = completion_tokens / SSE chunks of each measured request, averaged over the 2 repeats. t/s = fnbench client decode t/s, mean (min-max) over the 2 repeats.

## Table 1: acceptance / t/s per arm x workload

| arm | code-edit acc / t/s | prose-en acc / t/s | prose-ja acc / t/s | agent-loop acc / t/s |
|---|---|---|---|---|
| w4 | 3.84 / 363 (360-366) | 2.59 / 264 (255-272) | 2.35 / 242 (237-246) | 3.13 / 312 (305-318) |
| w6 | 5.56 / 440 (432-449) | 2.90 / 252 (235-268) | 2.44 / 214 (204-224) | 4.03 / 346 (343-348) |
| w8 | 7.04 / 533 (522-544) | 2.93 / 231 (223-238) | 2.63 / 210 (207-212) | 4.77 / 365 (338-393) |
| w12 | 9.92 / 591 (591-591) | 3.13 / 199 (198-201) | 2.59 / 167 (164-170) | 5.27 / 331 (313-350) |
| w16 | 11.68 / 620 (596-643) | 3.17 / 182 (180-185) | 2.63 / 153 (153-153) | 5.32 / 298 (275-320) |
| w4b (repeat) | 3.83 / 366 (359-373) | 2.60 / 264 (250-279) | 2.30 / 236 (236-237) | 3.24 / 324 (309-339) |

Cross-check, acceptance from the server's windowed `accept len` log lines (40 verify steps per line, first 1.5 s of each request dropped so the previous request's tail is excluded). Agrees with the chunk-based figure within 2-8 %; the log figure runs high on code-edit at wide widths because it drops the low-acceptance start of each request, the chunk figure runs slightly high on prose-ja (multi-byte characters are held back and merged into the next chunk).

| arm | code-edit | prose-en | prose-ja | agent-loop |
|---|---|---|---|---|
| w4 | 3.89 | 2.52 | 2.33 | 3.22 |
| w6 | 5.81 | 2.98 | 2.33 | 4.17 |
| w8 | 7.44 | 3.00 | 2.46 | 4.35 |
| w12 | 10.79 | 3.04 | 2.48 | 5.41 |
| w16 | 13.59 | 3.14 | 2.46 | 4.82 |
| w4b | 3.89 | 2.64 | 2.26 | 3.33 |

## Table 2: step time from the profiling traces (prof/trimmed_step.py, 20 steps, bs=1)

wall = full step wall (draft + verify + draft_extend, incl. idle); trimmed = `legacy_trimmed_ms` TOTAL.

| arm | code-edit wall ms | code-edit trimmed ms | prose-en wall ms | prose-en trimmed ms |
|---|---|---|---|---|
| w4 | 10.51 | 9.47 | 11.11 | 9.16 |
| w6 | 12.70 | 11.18 | 12.41 | 10.82 |
| w8 | 12.82 | 12.01 | 12.46 | 11.49 |
| w12 | 16.18 | 14.63 | 15.54 | 14.16 |
| w16 | 17.43 | 16.05 | 17.63 | 16.00 |
| w4b (repeat) | 10.36 | 9.47 | 9.96 | 9.24 |

Marginal step cost per extra draft token (wall, code-edit): W4->W6 +1.10 ms/token, W6->W8 +0.06 ms/token, W8->W12 +0.84 ms/token, W12->W16 +0.31 ms/token. W6 and W8 cost the same step time: W8 dominates W6 at every workload.

## Drift control (W4 first vs W4 last, 27 min apart)

t/s: code-edit +0.8 %, prose-en 0.0 %, prose-ja -2.5 %, agent-loop +3.8 %. Acceptance within 0.1. Trimmed step: 9.47/9.47 (code-edit), 9.16/9.24 (prose-en); the first W4 prose-en trace carried ~1.1 ms extra wall (unattributed/idle, monitor noise), the trimmed figure did not move. Session drift is under the n=2 noise (+-5 %), so between-arm differences above ~5 % are real.

## Arithmetic check (throughput ∝ acceptance / step wall; W4 = mean of both W4 arms; agent-loop and prose-ja use the prose-en step time as proxy)

| arm | code-edit pred / meas | prose-en pred / meas | prose-ja pred / meas | agent-loop pred / meas |
|---|---|---|---|---|
| w6 | 1.19x / 1.21x | 0.95x / 0.95x | 0.89x / 0.90x | 1.08x / 1.09x |
| w8 | 1.50x / 1.46x | 0.96x / 0.88x | 0.96x / 0.88x | 1.27x / 1.15x |
| w12 | 1.67x / 1.62x | 0.82x / 0.75x | 0.76x / 0.70x | 1.12x / 1.04x |
| w16 | 1.82x / 1.70x | 0.73x / 0.69x | 0.68x / 0.64x | 1.00x / 0.94x |

The model tracks measurement to within ~5 % except at W8 on prose/agent (predicted better than measured by 8-12 %; the prose-en/agent-loop step time was profiled on a 20-step window and W8's acceptance is the noisiest column, 338-393 t/s).

## Conclusion

1. code-edit: W16 wins, 620 t/s = +70 % over W4 (acceptance 3.04x for 1.68x step time); W12 +62 %, W8 +46 %, W6 +21 % -- monotone, the wider the better.
2. prose-en and prose-ja: W4 wins. Acceptance saturates at ~3.0 (en) / ~2.5 (ja) already at W6-W8 while step time keeps growing, so every wider width loses: W6 -5 % / -10 %, W8 -12 % / -12 %, W12 -25 % / -30 %, W16 -31 % / -36 %.
3. agent-loop: W8 wins at 365 t/s = +15 % over W4 (acceptance 1.50x for 1.22x step time; n=2 range 338-393, so call it +10-15 %), W6 +9 %, W12 +4 %, W16 -6 %.
4. Measured against W4: W6/W8 beat it by ~9-15 % on agent-loop, and are 5-12 % slower on prose-en/prose-ja, because prose acceptance does not grow with width on this head.
5. W6 was not the best fixed width on any of the four measured workloads (W8 has the same step time and more acceptance). The fixed-width frontier is W4 for prose, W8 for agent-loop, W16 for code; a fixed W8 costs -12 % on prose against +15 % on agent and +46 % on code, which is the trade the adaptive (`wa`) policy exists to avoid.
