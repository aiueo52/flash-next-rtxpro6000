> **About this copy.** This directory is the benchmark/profiling harness repository used for every
> measurement in this project (`flash-next-bench`, branch `codex/harness-v1`), published with these
> omissions: raw run logs (stripped copies are in `../results/`), CUPTI traces, the private token maps and
> token counts, the quality-benchmark data files (regenerate with `bench/quality/prep_data.py`), large
> per-experiment artefact directories, and a compiled `.so`. Home-directory paths inside scripts were
> genericised (`$HOME/...` or `/home/user/...`); many scripts under `prof/`, `calib/` and `bench/` were
> written for the author's machine and need their paths adjusted. The engineering logs that used to live
> in `specs/` are in `../docs/lab-notes/`. Layout: `fnbench/` client harness (+ `tests/`), `workloads/`
> synthetic prompts (single-prompt workloads and the 8-prompt held-out sets), `prof/` server profiling and
> analysis, `bench/` per-experiment micro-benchmarks and statistics, `calib/` adaptive-width experiment
> drivers, `adaptive/` adaptive-policy configs. The 12-task quality battery (`fnbench quality`,
> `prof/battery12.py`) the lab notes mention is not included: its questions came from a set of unshown
> origin, so it and every result on it were removed.

# flash-next-bench

`fnbench` is a CPU-only client harness for measuring decode performance on an
already-running local OpenAI-compatible endpoint. It never starts a server,
loads a model, or imports PyTorch/CUDA libraries.

## Setup (Python 3.12)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m fnbench --help
```

The only runtime dependency is `requests`; `pytest` and `ruff` are included for
development and verification. Tests are completely offline.

## Run

```bash
python -m fnbench run \
  --endpoint http://127.0.0.1:8001/v1 \
  --engine sglang \
  --workloads code-edit,prose-ja,prose-en,agent-loop,long-ctx \
  --repeats 3 \
  --out runs/experiment-a.jsonl \
  --label experiment-a
```

`--engine` selects `sglang`, `generic`, or `llamacpp` response parsing. The
default sampling mode is `greedy` (`temperature=0`). Use
`--sampling recommended` to apply each manifest entry's settings. By default,
the first model ID from the endpoint's `GET /models` response is used;
`--model` overrides that request field.

Each workload gets one warm-up request that is not written to JSONL, followed
by `--repeats` measured requests. Output defaults to
`runs/YYYYMMDD-HHMMSS.jsonl`; existing files are never overwritten.

The long-context prompt is a fixed committed template expanded 24 times with
deterministic block labels. This keeps the repository reviewable while sending
roughly 10k–30k tokenizer-dependent tokens. `prompt_chars` in each record makes
the actual expanded input auditable.

## GPU guard

Before a run, `fnbench` asks `nvidia-smi` for active compute processes. An
unknown process using more than 2048 MiB blocks the run. Recognized SGLang and
llama-server process names are allowed. If `nvidia-smi` is absent
or unusable, the CLI warns and continues. `--force` explicitly overrides a
detected conflict. Repeatable `--allow-proc SUBSTRING` options add
case-insensitive executable-path substrings to the built-in allowlist.

The experiment scripts that start their own servers (`quality/run_arm.sh`, `n1/`, `prof/`,
`calib/ws1_validate.sh`, `adaptive/c1/`) use `lib/owned_group.sh`: each server is started in its own
process group, and only that group is stopped (TERM, then KILL) on exit or failure. They never kill
other processes. If the GPU is not idle when a script starts, it waits up to 120 s and then refuses
to run, listing the processes it found; free the GPU yourself and rerun. A failed step stops the
script.

## JSONL fields

Every line is one measured request. Important fields are:

- `client.ttft_seconds`: request start to first non-empty content delta.
- `client.decode_tps`: `(completion_tokens - 1) / (last token chunk - first token chunk)`.
- `client.usage`: server-provided token counts from `stream_options.include_usage`.
- `client.timeline`: compact `[delta_ms, chunk_chars]` pairs; the first delta is
  measured from request start and later deltas from the preceding token-bearing
  chunk. Both visible content and `reasoning_content` chunks count for timing.
- `client.output_chars` and `client.reasoning_chars`: visible and reasoning text
  sizes. Only visible output contributes to `output_sha256`.
- `server.timings`: llama.cpp or generic `timings` fields, when present.
- `server.meta_info`: SGLang metadata, including speculative fields when present.
- `server.metrics`: SGLang `/metrics` before/after raw text, errors, and numeric deltas.
- `derived.effective_forward_per_second`: `decode_tps / accept_length`, when an
  acceptance length is available.

If the endpoint is down, returns a non-2xx response, emits malformed SSE, or
stops mid-stream, the CLI exits with a clear error. Metrics scraping is
best-effort and does not invalidate an otherwise successful SGLang request.

## Report

```bash
python -m fnbench report runs/experiment-a.jsonl
python -m fnbench report runs/experiment-a.jsonl --baseline runs/baseline.jsonl
```

The Markdown table reports per-workload median/p10/p90 decode throughput,
median TTFT and tokens, speculative acceptance length, effective forward rate,
and median throughput delta versus the baseline.

## Benchmark quality audit (GSM8K / MMLU / HumanEval / JCommonsenseQA)

`bench/quality/` (`run_all.sh` -> `run_arm.sh` -> `run_bench.py`) is the harness behind
`results/quality-q1/`. **Warning: its HumanEval step executes model-generated code.**
`bench/quality/he_exec.py` runs each program in bubblewrap (no network, cleared environment,
read-only root and `/dev`, 32 MiB `/tmp` and `/dev/shm` tmpfs, no nested user namespaces), inside
a transient systemd user scope that caps the whole job's memory, task count and CPU, plus
per-process rlimits and a timeout (`python3 bench/quality/he_exec.py --self-test` runs a
self-test). **The per-job cgroup is required by default:** if `systemd-run --user --scope` is
unavailable, nothing is executed (`SandboxUnavailable`), and a launcher failure during a job
aborts the run (`SandboxInfraError`) instead of being scored as a pass or fail.
`HE_EXEC_ALLOW_NO_CGROUP=1` (`--allow-no-cgroup` for the self-test) is an **unsafe** opt-out
that runs with per-process rlimits only, with no limit on the whole job's memory. This is
best-effort isolation that still shares the host kernel, not a security boundary: **run the
quality audit only in a disposable VM or container.** The datasets are not included; `bench/quality/prep_data.py` downloads them.

## Verification

```bash
pytest
ruff check .
```
