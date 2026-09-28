# QL1 frozen long-context workloads

`manifest.json` lists 24 prompts at exactly 8192 / 32768 / 65536 / 98304 input tokens, including the production chat template. Each `.txt` is the human-readable user content; the matching `.ids.json` is the complete rendered token sequence supplied to native `/generate`. Do not wrap the IDs in another chat template.

Two synthetic archive documents contain signed directives at separated depths and a substantive report-writing task at the end. The historical archive is deterministic context, not an external factual source. The needle vocabulary, seed, and passphrase follow `prof/needle_test.py`. The performance needle asks for a long audit after quoting the passphrase; separate strict retrieval prompts ask only for the passphrase at 10%, 40%, and 90% depth.

## Reproduction

CPU preparation (no model loading onto GPU):

```bash
$HOME/tools/sglang-rtxpro6000/.venv/bin/python -B prof/ql1_prepare.py
```

The private runtime cache is a copy of the production `.cache`, bound into a read-only production mount by bwrap. `frozen.json` records production source and measurement-file hashes. The runner checks production-file hashes at lock acquisition. Use a fresh run ID when resuming a failed server arm; the runner refuses to overwrite an existing arm directory.

Each server must be launched from the bench checkout under this lock (replace RUN_ID with a new identifier for reruns):

```bash
flock -w 28800 $HOME/.gpu.lock env QL1_GPU_LOCKED=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python -B prof/ql1_run.py w4 RUN_ID
flock -w 28800 $HOME/.gpu.lock env QL1_GPU_LOCKED=1 $HOME/tools/sglang-rtxpro6000/.venv/bin/python -B prof/ql1_run.py w16 RUN_ID
```

CPU analysis:

```bash
python3 -B prof/ql1_analyze.py RUN_ID
```

This writes `results.json`, `tables.md`, and an `exclusive.txt` / `ql1-summary.json` beside every captured trace. Original CUPTI traces and per-request measurements live under `prof/traces/ql1-*`; the incremental decision document is `specs/QL1_LONG_CONTEXT_MAP.md`.

Do not treat a runner exit code as trace validation. Each accepted trace must contain exactly 20 draft annotations and GPU events. Report any skipped contexts, partial arms, errors, observed KV capacity, minimum steady VRAM, needle failures and actual lock-hours. Raw deletion ceilings assume unchanged acceptance and zero replacement cost; they are not achieved speedups.
