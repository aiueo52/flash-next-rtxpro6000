# BetterBench run, 2026-10-07

A third-party benchmark, [GGZ14/BetterBench](https://github.com/GGZ14/BetterBench) at commit
`7696bf2`, run once against my own build at a Reddit commenter's request. Default config:
8 categories × (3 warmup + 20 measured passes), concurrency 1/2/4/8 × 48 requests, prefill
depths 2k–64k. The benchmark took about 9.5 minutes (05:01–05:10 JST),
about 15 minutes with server start and stop.

- `results.html`: the report BetterBench writes (open it in a browser, no external scripts)
- `results.json`: the raw numbers (no prompts or generated text, only timings and token counts)
- `run_flashnext.sh`: how it was run (server start, BetterBench with the update check off, server stop)

## Setup

- Build: my production build (the `wa` profile, adaptive draft width 3/7/15) from 2026-10-02,
  with the **private** fine-tuned MTP head and the **private** 49k token map. This is the build
  behind the "mine" rows in the README, not the "public components only" one.
- Server: `serve-fast.sh wa` with the launcher defaults (262,144 context, `MAMBA_SLOTS=10`,
  `--max-running-requests 4`). Sampling was BetterBench's default (temperature 0.7, top-p 0.95,
  top-k 20), thinking on.
- GPU: one RTX PRO 6000 Blackwell Max-Q at a 325 W power limit, also driving a 4K desktop.
  Another job kept about 16 of 48 CPU threads busy during the run.

## Results

Single-stream (batch 1) decode, median tok/s over 20 passes:

| chat | code | file_edit | json | math | prose | reasoning | summarization | weighted |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 301 | 370 | 520 | 452 | 455 | 270 | 297 | 372 | 368 |

Concurrency sweep (aggregate tok/s across all requests):

| level | 1 | 2 | 4 | 8 |
|---|--:|--:|--:|--:|
| aggregate t/s | 314 | 306 | 310 | 317 |
| TTFT p50 (ms) | 112 | 141 | 2,412 | 6,866 |

Prefill: about 9.1k tok/s at ~1.5k prompt tokens and 11.7k–11.8k tok/s from ~6k to ~47k tokens.

## How to read it

- 132 of the 160 single-stream runs (82 %) stopped at the corpus `max_tokens` (120–900), and in
  55 of them the model was still thinking and never reached an answer. BetterBench estimates the
  thinking share of the output at 58–90 % per category, so these rates are mostly thinking tokens.
- The launchers are tuned for one request at a time. With speculation one request uses about 9
  GDN state slots and `MAMBA_SLOTS` is 10, so concurrent requests queue: aggregate throughput
  stays flat at about 310 tok/s from c1 to c8 and waiting time grows. Concurrency was never
  optimised (see Limitations in the README).
- One run, one server start. In earlier measurements on this machine, identical configurations
  varied by ±10–15 % across sessions (see Limitations in the README).
