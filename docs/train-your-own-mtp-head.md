# Train your own MTP draft head

The measurements in this repository used a fine-tuned MTP draft head. That head was trained on the
author's private data (own coding-agent sessions and chats, plus the model's continuations of prompts
drawn from them), so it is **not published**, and neither is the data. This guide shows how to
train your own head on your own data with the pipeline in [`mtp-train/`](../mtp-train/). The
pipeline is the one that produced the measured head.

Only train on data you are entitled to use: your own sessions, text you wrote, or datasets whose
licence allows it. Before training on Codex CLI or Claude Code transcripts, check the provider's
terms (OpenAI, Anthropic, etc.): they may restrict using model outputs to train other models. Session logs often contain secrets (API keys, tokens, customer data) that end up
in the dumps and in the weights. Filter them out first.

Contents:

1. [What the MTP head is and why fine-tuning helps](#1-what-the-mtp-head-is-and-why-fine-tuning-helps)
2. [What you need](#2-what-you-need)
3. [Pipeline overview](#3-pipeline-overview)
4. [Build the server with the dump hook](#4-build-the-server-with-the-dump-hook)
5. [Assemble a corpus](#5-assemble-a-corpus)
6. [Extract: teacher-forced dump of the corpus](#6-extract-teacher-forced-dump-of-the-corpus)
7. [Self-generate target continuations](#7-self-generate-target-continuations)
8. [Index and held-out split](#8-index-and-held-out-split)
9. [Parity check (not included)](#9-parity-check-not-included)
10. [Train](#10-train)
11. [Evaluate offline with the renewal evaluator](#11-evaluate-offline-with-the-renewal-evaluator)
12. [Write back into a checkpoint copy](#12-write-back-into-a-checkpoint-copy)
13. [Validate with a server A/B, the only gate that counts](#13-validate-with-a-server-ab-the-only-gate-that-counts)
14. [Input checks](#14-input-checks)
15. [Pitfalls](#15-pitfalls)

---

## 1. What the MTP head is and why fine-tuning helps

Qwen3.8-Flash-Next ships one extra decoder layer, the **MTP (multi-token prediction) layer**, stored
as the 31 `mtp.*` tensors of the checkpoint (about 2.6 B parameters, BF16 even inside the NVFP4
checkpoint). SGLang serves it as a NEXTN speculative draft head with `topk = 1` chains:

* **Draft step 1.** The input is the target's final-layer hyper-connection state `hc_hidden[t]`
  (4 streams × 2560 = 10240 values, taken *before* the target's own mixer) and the embedding of the
  token just produced. The output is logits for the token after it. The layer has QSA attention,
  a 512-expert MoE with top-10 routing plus a shared expert, and hyper-connections. It uses the
  target's embedding and `lm_head`.
* **Draft steps 2..S.** The head feeds on its own output HC state and its own previous prediction.
  Only the first step sees the target's state. W4 uses S = 3 and W16 uses S = 15.
* **Verify.** The target checks all draft tokens in one forward. It accepts the longest matching
  prefix and adds one bonus token. The server's *acceptance length* is tokens emitted per verify.

Throughput at batch size 1 is roughly acceptance length ÷ step time. The step time is fixed by the
serving stack, so a head that agrees with the target more often, especially deep into the chain, is
directly worth tokens per second.

The released head is a general-purpose predictor. Fine-tuning helps for three reasons:

* **It learns the serving target's own behaviour.** The labels are the target's greedy argmax (and
  its top-K distribution) computed by the *quantised* serving stack on the *serving* chat template,
  not the original training text.
* **It concentrates on your traffic.** Your mix of code, prose, languages and agent tool calls has
  its own distribution. The head is small, so capacity spent on your distribution matters.
* **It learns the chain.** A K-step rollout loss trains the recursive steps on the head's own
  states, which is what steps 2..S see at serving time.

What to expect: in this project's server A/Bs the first adopted head (v3) raised W4 acceptance by
a few percent on prose and agent traffic, left code flat and was neutral at W16. The second adopted
head (v5), the same recipe on a broader re-extraction, added a few percent more. A rerun on the
same data at three times the budget (v4) was evaluated only offline, on private data, and was never
served; nothing about its outcome is published. See [`optimizations.md`](optimizations.md) A10
and [`rejected.md`](rejected.md) §1.4–1.6. Only the server A/B in §13 decides.

## 2. What you need

### Hardware

| stage | GPU | host | notes |
|---|---|---|---|
| dump server (§6, §7) | the serving GPU (96 GB class) | as for serving | the normal serving footprint. The hook adds one `lm_head` chunk (default 1024 rows × 248,320 × fp32 ≈ 1 GB) and a 4-deep write queue in host RAM (≤ 8192 rows × 20 KB ≈ 170 MB per queued forward) |
| training (§10) | one 96 GB GPU, **nothing else resident** | ≥ 64 GB RAM | see the VRAM budget below |
| offline eval, write-back | GPU optional | ≈ 8 GB RAM per CPU-loaded head | `eval_renewal.py` runs on CPU too, only slowly |

**Training VRAM**, from the code's defaults (FP32 master weights, AdamW, BF16 autocast):

```
trainable MTP weights, FP32       2.6 B × 4 B  ≈ 10.4 GB
gradients, FP32                   2.6 B × 4 B  ≈ 10.4 GB
AdamW m and v, FP32               2.6 B × 8 B  ≈ 20.9 GB
frozen embed + lm_head, BF16      2 × 248,320 × 2560 × 2 B ≈ 2.5 GB
activations (+ rollout, chunked logits)   grows with --tokens-per-step
```

That is about 45 GB before activations. With `--tokens-per-step 8192` (8192 padded tokens per
micro-batch) the observed peak was about 65 GB. A single 24,576-token micro-batch nearly filled a
96 GB card, so reach large batches with `--grad-accum` instead.
`--optimizer adamw8bit` (bitsandbytes) cuts the optimizer state from about 21 GB to about 5 GB. The
logits are never materialised: the loss recomputes them chunk by chunk (`--logit-chunk`) in the
backward pass.

### Disk

Each dumped token (a "row") stores:

```
hc_hidden   10240 × bf16          20,480 B
input_ids, positions, argmax      3 × int32 = 12 B
top-K (SGLANG_MTP_DUMP_TOPK=K)    K × (int32 id + fp16 logit) = 6K B
lse                               fp16 = 2 B
taps (SGLANG_MTP_DUMP_TAPS)       20,480 B per extra layer
```

With K = 8 and no taps that is **20,542 B/row ≈ 20.5 GB per million tokens** (19.1 GiB). So:

| dumped tokens | disk |
|---:|---:|
| 1 M | ≈ 21 GB |
| 5 M | ≈ 103 GB |
| 10 M | ≈ 205 GB |
| 15 M | ≈ 308 GB |

Also count:

* **Forwards the manifest does not list.** These are the server's warm-up and, in the
  self-generation of §7, the prefill of each prompt before it is generated from (about the prompt's
  length again per self-generated document).
* **The hook's cap.** `SGLANG_MTP_DUMP_MAX_GB` (default 400) stops the hook when it is reached.
* **Training outputs.** `latest.pt` holds the FP32 weights and the optimizer state, about 31 GB. Each
  BF16 snapshot (`best-mtp.safetensors`, `mtp-step*.safetensors` with `--snapshot-every-eval`) is about 5.2 GB.
* **Write-back.** The three shards that hold `mtp.*` (≈ 14 GB) are rewritten. Every other shard is
  a symlink.

Use a fast local disk. The trainer streams the dump every epoch through mmap, and a slow disk
starves the GPU.

### Time

All of these are "measure a small run first, then extrapolate". Every tool prints its own rate.

* **Extraction** is prefill-bound. It is capped by disk write bandwidth: at ~20 KB/row, 1 GB/s
  sustained write is at most ~50k rows/s. Run `client.py` with a small `--token-budget` (for example
  200k), read the `tok/s` it prints, and divide your budget by it.
* **Self-generation** is decode-bound. It takes generated tokens ÷ aggregate t/s at your
  `--concurrency` on the dump server, plus one prefill-only pass over prompt + continuation per
  document.
* **Training** processes `steps × tokens-per-step × grad-accum` rows. With K = 3 rollout each batch
  runs up to three extra draft forwards. Time a short run (for example 20 steps with the §10
  settings), multiply by the step count, and add the periodic evals. The renewal eval is the
  expensive one, so keep `--renewal-docs` small.

### Software

* The patched SGLang from [`reproduce.md`](reproduce.md) §1–2, plus the dump branch (§4).
* The SGLang virtualenv also runs `mtp-train/` (torch, safetensors, transformers, requests).
  `pyarrow` is only needed for `--arrow` corpora and `bitsandbytes` only for `--optimizer adamw8bit`.
* CPU tests: `cd mtp-train && CUDA_VISIBLE_DEVICES="" python -m pytest -q`.

### Status of the published code

The trainer (`mtptrain/`), the renewal evaluator, the dump hook and the write-back are the code the
published head was trained with. For publication, paths became arguments, comments were cleaned, and
one option was added (`--renewal-greedy-only`). The following parts were rewritten or generalised
for publication and are **untested against a live server**. They pass the CPU tests (and, for
extraction, `--dry-run`) only:

* `corpus/formats.py` (the corpus readers, moved out of a private script);
* the source-selection flags of `extract/client.py` (`--codex-sessions`, `--claude-sessions`,
  `--lmstudio-conversations`, `--files`, `--jsonl`, `--arrow*`);
* the self-generation route of §7, `extract/client.py --mode selfgen`. The published head's
  self-generated data came from a two-stage prompt-bank / generate / dump tool chain that is not
  included;
* `parity/serve.sh` (made generic; `stop` now kills only its own process group).

Run them on a small budget first and check the output with `extract/verify_dump.py` before a full
run.

The repository ships the main path only. Parity tooling against the served head, the calibration of
the offline evaluator against the server, dump replay and the other peripheral tools the project
used are not included.

## 3. Pipeline overview

```
 your sessions / text ──► corpus readers ──► extract/client.py --mode extract  (prefill only)          ─┐
                                          └──► extract/client.py --mode selfgen  (generate, then prefill) ─┴─► hook server
 hook server ──► one dump dir + manifest.jsonl ──► verify_dump / build_index / build_eval_set
      ──► mtptrain.train ──► best-mtp.safetensors
      ──► scripts/eval_renewal.py (offline, held-out) ──► writeback/write_mtp.py ──► model dir copy
      ──► server A/B on bench/ workloads (fnbench, measurement.md §6) ──► keep or discard
```

Commands below assume:

```bash
export REPO=/path/to/flash-next-rtxpro6000
export SGLANG_DIR=/path/to/sglang-rtxpro6000          # serving checkout with .venv (reproduce.md)
export MTP_MODEL_DIR=$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
export TARGET_MODEL="$MTP_MODEL_DIR"
export PY=$SGLANG_DIR/.venv/bin/python
export WORK=/fast/disk/mtp                             # dumps and runs
export TOKEN_MAP=${TOKEN_MAP:-none}                    # the draft token map you serve with, or none
if [ "$TOKEN_MAP" = none ]; then TM=(); else TM=(--token-map "$TOKEN_MAP"); fi
cd "$REPO"/mtp-train
```

`TM` passes your serving token map to the offline evaluators (`"${TM[@]}"` below), so they restrict
the draft argmax exactly as `--speculative-token-map` does. With `TOKEN_MAP=none` they use the full
vocabulary.

The server block in §4 starts the dump server and waits until it is ready; §6 and §7 run the client
against it; then it is stopped. `parity/serve.sh` refuses to start while any process is on the GPU,
and `stop` signals only the process group it started. Chain the steps with `&&`, so a server that
fails to come up stops the block; run `stop` in any case.

## 4. Build the server with the dump hook

The hook is a separate 9-patch series on top of the serving branch:

```bash
"$REPO"/patches/sglang/mtp-dump/apply.sh "$SGLANG_DIR" "$WORK/sglang-dump"
export SGLANG_DUMP_TREE=$WORK/sglang-dump
```

`apply.sh` creates branch `flash-next-mtp-dump` in a new git worktree of the serving clone and
applies the patches there. The serving checkout is not touched and stays on `flash-next-fast`.
There is no second virtualenv: the dump server runs from the serving checkout's `.venv`, and
setting `PYTHONPATH=$SGLANG_DUMP_TREE/python` makes the worktree's patched files the ones that get
imported (the editable install's finder is appended to `sys.meta_path`, so `PYTHONPATH` wins).
`mtp-train/parity/serve.sh` sets it for you (the script itself is untested against a live server
in this published form; see §2). This is the setup the published head was dumped with.

What the hook does (`python/sglang/srt/models/mtp_dump.py` on that branch):

| variable | effect |
|---|---|
| `SGLANG_MTP_DUMP_DIR=DIR` | enables the **target-side dump**. Every target *extend* (prefill) forward writes one safetensors file to DIR with `input_ids` [T] int32, `positions` [T], `hc_hidden` [T, 10240] bf16 (the residual after the last layer and before `hyper_connection_mixer`, i.e. exactly what the draft consumes), `target_argmax` [T] int32 (the target's argmax at *every* position, recomputed chunk-wise through the target `lm_head` with the same quant method the server uses), `seq_offsets` [S+1] and `prefix_lens` [S] |
| `SGLANG_MTP_DUMP_TOPK=K` | also `topk_ids` [T, K] int32, `topk_logits` [T, K] fp16 and `lse` [T] fp16, the full-vocabulary log-sum-exp, so the true top-K probability mass is known. **Use K = 8**: the soft-target loss needs it |
| `SGLANG_MTP_DUMP_TAPS=3,23` | also `tap_hidden` [T, N × 10240]: the output HC state of extra target layers, for the experimental EAGLE-3-style tap-fusion entry (`--tap-layers`, [`lab-notes/DRAFT_V2_SPEC.md`](lab-notes/DRAFT_V2_SPEC.md)). Not part of the recommended recipe |
| `SGLANG_MTP_DUMP_MAX_GB` (400), `SGLANG_MTP_DUMP_CHUNK` (1024), `SGLANG_MTP_DUMP_TAG` (`dump`) | size cap, lm_head chunk rows, file-name prefix |
| `SGLANG_MTP_DRAFT_DUMP_DIR=DIR` | **draft-side dump**: for every draft-extend forward, the draft's exact inputs (`input_ids`, `positions`, `input_hc`) and outputs (`own_hc`, `mixed`, `post_attn_hc`, `argmax`, `topk_ids`/`topk_logits`). Knobs: `SGLANG_MTP_DRAFT_DUMP_TOPK` (8), `_MAX_GB` (40), `_CHUNK`, `_TAG`. For parity checks (§9); not needed for training, and the tools that read it are not included |

Decode, target-verify and CUDA-graph capture paths return immediately, so the graph replay path is
untouched. Files are written by a background thread through a bounded queue. With no variable set
the branch behaves exactly like the serving branch.

Start a dump server (production W4 config, radix cache **off** so every request is dumped from
position 0, short context for a small KV pool):

```bash
export SGLANG_MTP_DUMP_DIR=$WORK/dump SGLANG_MTP_DUMP_TOPK=8
parity/serve.sh start-extract $WORK/server-extract.log && parity/serve.sh wait $WORK/server-extract.log
# ... run §6 and §7 against this server ...
parity/serve.sh stop $WORK/server-extract.pgid
unset SGLANG_MTP_DUMP_DIR
```

`start-extract` runs `launch/serve-fast.sh w4 --disable-radix-cache` with `CONTEXT_LENGTH=16384`
and `MAMBA_SLOTS=10`. If you use a token map for serving, set `TOKEN_MAP` as usual; it does not
affect the target-side dump. The corpus extraction (§6) and the self-generation (§7) write into this
one directory and one `manifest.jsonl`; use a new directory for a new, separate dataset.

## 5. Assemble a corpus

The head should see text like the text it will be asked to draft. Useful sources, in rough order of
value (the readers and their flags are untested against a live server; see §2):

1. **Your own sessions** with this or a similar model. The readers in `mtp-train/corpus/formats.py`
   take:
   * **Codex CLI rollouts**: a directory of `*.jsonl` session files → `--codex-sessions DIR`.
   * **Claude Code transcripts**: a directory tree of session `*.jsonl` → `--claude-sessions DIR`.
     Thinking blocks become `reasoning_content`, tool calls are rendered in the Qwen tool-call
     syntax, and tool results become `tool` turns.
   * Before using Codex CLI or Claude Code transcripts, check the provider's terms (OpenAI,
     Anthropic, etc.); they may restrict using outputs to train models.
   * **LM Studio chats**: a directory of `*.conversation.json` → `--lmstudio-conversations DIR`.
2. **Your own documents and code**: `--files '[NAME=]GLOB'`. It reads `.jsonl`/`.json` in the usual
   chat shapes (`messages` with `role`/`content`, ShareGPT `conversations` with `from`/`value`,
   `instruction`/`input`/`output`, `prompt`/`response`, bare `text`) and `.txt`/`.md`. Long text files
   are split into ~20 KB documents so the held-out split has something to hold out. `--jsonl FILE`
   takes plain-text JSONL with one document per record.
3. **Public datasets** whose licence allows training, from a local Hugging Face `datasets` cache:
   `--arrow GLOB --arrow-text-field ... [--arrow-aux-field ... --arrow-instruction ...]`. For example
   a news-summarisation set becomes user = instruction + article, assistant = summary.
4. **The target's own continuations** of prompts drawn from all of the above (§7). This is the
   most on-distribution text there is: it is exactly what the verifier will compare the draft
   against.

**Mix guidance.** A reasonable first run is roughly **10 M teacher-forced tokens**, mostly your
dominant workload, with some of every language and task type you serve, **plus self-generated
continuations** of prompts from the same sources: a few thousand documents of 384–1024 generated
tokens (`--gen-tokens`, one value per run and `--bucket`). The measured second head (v5) was trained on a re-extraction that added new documents and
prompt families.

The Codex / LM Studio readers are the n-gram simulator's parsers ([`sim/ngramsim`](../sim/)), imported
from `../sim` (override with `NGRAMSIM_PATH`). Every reader takes explicit paths. Nothing is read
from a default location.

## 6. Extract: teacher-forced dump of the corpus

`extract/client.py` renders every document with the **serving chat template and kwargs**
(`--chat-kwargs`, default = `launch/serve-local.sh`'s
`{"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "medium"}`). It tokenises
locally, cuts each document into windows of at most **2048 tokens** (`--max-len`: up to that length
QSA attention selects every token, so the dense attention the trainer uses is exact), and posts each
window as raw `input_ids` with `max_new_tokens = 1`. Sending ids guarantees that the dumped tokens are
exactly what the manifest hashes.

```bash
$PY extract/client.py --mode extract --url http://127.0.0.1:8001 \
    --codex-sessions ~/path/to/codex-sessions --claude-sessions ~/path/to/claude-transcripts \
    --files 'docs=~/my-notes/**/*.md' \
    --manifest "$SGLANG_MTP_DUMP_DIR/manifest.jsonl" \
    --token-budget 10000000 --max-chunks-per-doc 8 --max-tool-chars 2048 --concurrency 2
```

Useful options:

* `--max-tool-chars 2048` (default): long tool outputs are cut to head + tail. In agent transcripts
  most tokens are tool output, which the model never generates.
* `--max-chunks-per-doc N` samples N windows from very long sessions instead of all of them. This keeps
  the document count, and with it the diversity, high.
* `--lang ja|en` and `--workload code,mixed,prose` filter.
* `--bucket NAME` labels the manifest rows. Buckets are what the per-bucket reports and
  `--exclude-buckets` use.
* `--dry-run` (extract mode) tokenises and writes a manifest without contacting a server, as a budget check. Give
  it a manifest of its own: a real run refuses a manifest that holds dry-run rows, and vice versa.
* Re-running the same command resumes: chunks (extract) or documents (selfgen) already recorded for
  the same mode and `--bucket` are skipped and count toward `--token-budget`; a line torn by a crash
  is dropped and redone. A resume with other chunking, chat kwargs or model is refused; the model
  is identified by the content of every file in `--model-dir` (shards, config, tokenizer files,
  chat template), so a swapped shard or tokenizer is caught even at the same name and size.
  Every row records `input_hash`, a hash of the document's source and messages. Each run checks
  every loaded document against the manifest *before* dedup, `--lang`/`--workload`, resume skips,
  `--limit-docs` or the budget can drop it, and refuses, naming the key: a document whose content
  changed under the same `split_key` (including one edited into a copy of another); a manifest
  row of a source this run loads (same `source` label) whose document is gone or is now dropped
  by the loader (emptied, too short, over `--max-file-mb`, beyond `--limit-files`); and a manifest
  written before `input_hash` existed. Use a new dump directory, or restore the source. Keep one
  set of sources per source label: a later run that loads another set under the same label is
  refused for the same reason.

The manifest row keys (`doc_hash` = hash of the window's ids, `split_key` = hash of source +
document name) decide the held-out split **by document**: all windows of a document land on the
same side. `--files` names are paths relative to the glob's directory, so `a/x.md` and `b/x.md`
are two documents; two documents of one run with the same `split_key` but different content are
refused.

The dump is checked once, after §7.

## 7. Self-generate target continuations

The renewal evaluator and the checkpoint selection (§10, §11) need **held-out self-generated greedy
documents**: prompts from your corpus, continued by the served target itself. `client.py --mode
selfgen` makes them against the same hook server, into the same dump directory and manifest as §6:

```bash
$PY extract/client.py --mode selfgen --url http://127.0.0.1:8001 \
    --codex-sessions ~/path/to/codex-sessions --claude-sessions ~/path/to/claude-transcripts \
    --manifest "$SGLANG_MTP_DUMP_DIR/manifest.jsonl" \
    --token-budget 2000000 --gen-tokens 512 --concurrency 8
```

Per corpus document it:

* takes every message before the last assistant turn as the prompt, rendered with the serving chat
  template and a generation prompt, and keeps its last `--max-len` − `--gen-tokens` tokens
  (≤ 1536 with the defaults, so prompt + generation ≤ 2048);
* generates `--gen-tokens` tokens at `--temperature` (default 0, greedy: the documents the renewal
  evaluator can replay exactly) and drops outputs shorter than `--min-len`;
* sends prompt + continuation once more as a prefill-only request, which is the dump the trainer and
  the evaluator read, and records it with `mode: selfgen`, `prompt_tokens` and `generated_tokens`.

Documents without a prompt of their own (flat `.txt`/`.md` files, `--jsonl`, `--arrow` without
`--arrow-aux-field`) are skipped: each would get the same placeholder prompt and the same greedy
continuation. Identical continuations are recorded once, and a run that self-generates nothing is
an error.

The prefill of the prompt before generation is dumped too. The manifest does not list it, so every
reader ignores it; it only costs disk (§2). One self-generated document is made per source
document and `--bucket`; `--token-budget` counts generated tokens; `--limit-docs N` caps the
documents. Each row inherits its source document's `split_key`, so a held-out document's
continuation is held out too.

The measured second head (v5) also used sampled continuations and more prompt families; this route
generates greedily, which is what the evaluator needs.

Verify the dump (CPU). First every manifest document found in the dump is re-hashed and must match
its manifest hash (a mismatch is an error; documents not on disk are counted and ignored). Then, on
greedy self-generated documents, the dumped `target_argmax` at row t must equal the token the server
emitted at t+1. It need not be exactly 100 %: generation ran with speculative decoding over a
growing KV cache, while the dump is one fresh prefill.

```bash
$PY extract/verify_dump.py $WORK/dump --docs 200
```

## 8. Index and held-out split

```bash
# frozen teacher-forced eval subset (held-out documents only, stratified by bucket)
$PY extract/build_eval_set.py $WORK/dump --rows 200000     # writes $WORK/dump/eval_fixed.json
```

The split is `blake2b(f"{seed}:{split_key}") % 10000 < holdout × 10000`, with `seed 20260903` and
`holdout 0.1` everywhere by default (`DumpDataset`, `select.py`, `build_eval_set.py`,
`eval_renewal.py`). **Keep the split seed and holdout identical across the eval set, training and
evaluation.** They are `--seed` / `--holdout` in `build_eval_set.py` and `mtptrain.train` (where
`--seed` is also the split seed; an eval set built with another seed or holdout is refused), but
`--split-seed` / `--holdout-frac` in `eval_renewal.py`, whose `--seed` only sets document sampling.
`eval_renewal.py` refuses a non-default `--seed` unless `--split-seed` is given explicitly. A document listed under several split keys (the same ids in two
manifest rows) is held out if any of its keys is, so it can never be on both sides.

Before anything is written, `mtptrain.train` intersects every eval source it uses (`--eval-set`,
or the held-out split of `--dump-dir`, and the `--eval-renewal` documents) with the training
documents, by `doc_hash` and by split key, and refuses a non-empty overlap. This catches an eval
set or renewal dump made from another dump of overlapping data, where the seed check cannot.
`eval_fixed.json` lists each document's `split_keys` for this check; an older eval set without
them is matched against its dump's manifest, or by content only (with a note).

`build_index.py` (run automatically when needed) maps each `doc_hash` to its shard(s). It joins
chunked-prefill pieces of one request, and refuses orphan continuations (a chunk without its
start), which only occur if the radix cache was on. Pieces are joined only within one write stream (one server process, from the
hook's `{tag}-{pid}-{counter}` file names); an unreadable file (for example a partial last file
left by a crash) is refused, not skipped: move or repair it and index again.

Every loader (training and eval splits, `build_eval_set.py`, the renewal evaluator's `select.py`,
`verify_dump.py`) reads documents through the same joined index (`mtptrain/dumpindex.py`), so a
document split over several dump files is loaded whole and matched against its manifest hash as a
whole. The index is always rebuilt from a full scan; it is a cache keyed on each file's name, size,
`mtime_ns`, `ctime_ns`, inode, device and symlink target (recorded in `index.files.json`), rebuilt
whenever any of that changes. A file rewritten in place with its size and mtime put back still
has a new ctime. Every loaded document is re-hashed against its manifest hash, and a mismatch
stops `DumpDataset`, `build_eval_set.py` and `verify_dump.py` with an error. Manifest rows whose request never reached the disk (for example
after the hook's size cap) are simply absent from every split.
`eval_fixed.json` records its dump directory and lists each eval document's pieces under `docs`; a
missing piece or a hash mismatch is an error, so rebuild the eval set after changing the dump.

## 9. Parity check (not included)

Checking the plain-PyTorch re-implementation in `mtptrain/` against the served head on identical
inputs (the hook's draft-side dump, §4) was part of the project, but its tools are not included.

## 10. Train

The recipe behind the measured head. The published description is in
[`optimizations.md`](optimizations.md) A10.

**Stage 1: from the released head.**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m mtptrain.train \
  --model-dir "$MTP_MODEL_DIR" --dump-dir $WORK/dump \
  --eval-set $WORK/dump/eval_fixed.json --out $WORK/run1 \
  --device cuda --bf16 \
  --tokens-per-step 8192 --grad-accum 3 --max-len 2048 --prefetch 6 \
  --soft-alpha 0.5 --soft-temp 1.0 --soft-steps first \
  --rollout-k 3 --rollout-weight 0.5 --rollout-mask accepted \
  --epochs 20 --steps 1000 --lr 1e-5 --warmup 50 --min-lr-ratio 0.1 --grad-clip 1.0 --logit-chunk 512 \
  --eval-every 100 --eval-batch 8 --eval-starts 16 --eval-batches 20 --chain-ks 3 15 \
  --eval-renewal --renewal-ks 3 15 --renewal-docs 48 --renewal-batch 32 --renewal-min-gen 32 \
  --renewal-greedy-only "${TM[@]}" \
  --snapshot-every-eval --ckpt-every 200 --log-every 20
```

What the options mean:

* **Objective.** Row t consumes `hc_hidden[t]` and `embed(input_ids[t+1])` and is labelled
  `target_argmax[t+1]`. The router and the QSA indexer are frozen. The other ~2.6 B parameters train
  in FP32 master weights under BF16 autocast.
* **Soft targets** (`--soft-alpha 0.5 --soft-temp 1.0`). The loss is
  `(1 − α)·CE(hard label) + α·(−Σ_k q_k log p_draft(i_k))` over the dumped top-8.
  `q_k = softmax(topk_logits)_k × exp(logsumexp(topk_logits) − lse)` is the target's *true*
  probability on those ids. The rest of the vocabulary is left unsupervised rather than forced to
  zero. `--soft-steps first` applies it to the teacher-forced step only, where the dumped
  distribution has the right conditioning. This needs `SGLANG_MTP_DUMP_TOPK=8` dumps: with
  `--soft-alpha` on a dump without top-K the trainer refuses to start.
* **Rollout** (`--rollout-k 3 --rollout-weight 0.5 --rollout-mask accepted`). After the
  teacher-forced step the head is rolled 3 more steps in an approximation of serving: own argmax as
  next token, own HC state, position + 1, but each rolled step attends only to the teacher-forced
  KV plus its own key, not to the chain's earlier rolled rows as the served draft does
  (`mtptrain/rollout.py`, "One approximation"). Each rolled step is scored against the target's argmax, only
  while the chain is still on an accepted prefix and the corpus continuation equals the target's
  greedy continuation. Rolled inputs are detached, so memory stays flat in K.
* **Schedule.** LR 1e-5, 50 warm-up steps, cosine to 10 %, about 1000 optimizer steps of 24,576
  tokens. `--epochs` is only an upper bound: training stops at `--steps`. For the measured head
  the served checkpoint was from around step 400, not the last one; snapshot every eval so that
  earlier checkpoints stay available.
  The measured stage-1 run used `--tokens-per-step 24576 --grad-accum 1`. `8192 × 3` gives the same
  tokens per step with a third of the activation memory.
* **`--renewal-greedy-only`** was added for publication and was not used in the measured runs. It
  restricts the renewal eval to greedily generated documents, the ones it can replay exactly
  (§11). Drop it to match the measured runs.
* **Selection.** With `--eval-renewal` the best checkpoint (`best-mtp.safetensors`, `best.json`) is
  chosen by the **renewal** acceptance length on held-out self-generated documents (§11), not by
  teacher-forced agreement. The two can diverge as the head's distribution shifts. `--select-metric renewal@7`
  (or `@3`, `@15`) picks one chain length; the default is the mean of `--renewal-ks`.
  `--token-map` restricts the draft argmax in that eval exactly as `--speculative-token-map` does at
  serving time. Use the map you serve with.
* **Crash safety.** Every checkpoint and output file is written atomically (temp file, fsync,
  rename). Each `latest.pt`, periodic or at an epoch end, holds the model and optimizer, the step,
  the epoch and the position inside it, any half-full gradient accumulation, the random-number
  states, the best checkpoint's score and the eval history. `--resume $WORK/run1/latest.pt`
  restores all of it and skips the micro-batches already consumed, so history continues and
  `best-mtp.safetensors` is only replaced by a strictly better score (the file records its own
  score, so an eval that ran after the last checkpoint is kept). The position is exact only if the
  dump, its manifest and the flags that shape the data stream are unchanged (a difference prints a
  warning); skipped micro-batches are re-read, which costs some I/O; GPU arithmetic is not bitwise
  deterministic. `--stop-after-epochs 1` splits a long run into GPU stretches that each end on an
  epoch boundary. A `--resume` path that does not exist is an error. The trainer
  refuses an `--out` that already holds a run (`latest.pt`, `best*`, `history.json`, snapshots)
  unless you resume from that directory's `latest.pt` or pass `--overwrite`. `--overwrite` moves
  the earlier run's files into `--out/.previous-<time>/` (nothing is deleted) and cannot be combined
  with `--resume`. `--resume` must point at a checkpoint inside `--out`; to branch a run, copy its
  directory and resume the copy. Each run has a run id, stored in `latest.pt`, `best.json` and the
  header of every weights file it writes; on resume, files from another run are moved aside, so
  `best.json`, the best file and the checkpoint always describe the same run.
  A resume also refuses any change to a setting that makes scores incomparable: the selection
  metric, the eval and renewal data (by a digest of every tensor the evals read: hidden states,
  labels, top-K, lse, masks; the same token ids with other hidden states or labels are other
  data), the renewal settings, the token map (by content), `--chain-ks`, the split seed and
  holdout, and the base model (its loaded weights and architecture config). Paths are not
  compared: a moved dump with the same content resumes. `--reset-best` accepts such a change and
  restarts best selection (the earlier best is moved aside); a checkpoint from a trainer without
  this record needs it too. `--init` cannot be combined with `--resume`. A best file is adopted
  only if its tensors are the ones its header was written for (`mtp_tensors_digest`).
* **Unmeasured is not zero.** A metric with nothing to average over (an `accept@k` with no
  scorable k-step window, a renewal k with no verify) is `null` in the eval log, `history.json`
  and `best.json`, next to its window or verify count. An eval whose selection metric is unmeasured
  is recorded with `"unscored"` and a warning, and neither becomes nor is compared with the best.
  If the eval data cannot measure the selected metric at all (for example `accept@15` with
  sequences too short for a 15-step window), the trainer refuses to start; an unselected
  `--chain-ks` value only gets a warning.

**Stage 2 (optional): warm start on broader data.** The measured final head (v5) was stage 1's
head retrained on a broader re-extraction (more documents and prompt families, partly sampled
self-generation). It used a lower LR:

```bash
$PY -m mtptrain.train --model-dir "$MTP_MODEL_DIR" --init $WORK/run1/best-mtp.safetensors \
  --dump-dir $WORK/dump-v2 --eval-set $WORK/dump-v2/eval_fixed.json --out $WORK/run2 \
  --device cuda --bf16 --tokens-per-step 8192 --grad-accum 3 --max-len 2048 --prefetch 6 \
  --soft-alpha 0.5 --soft-temp 1.0 --soft-steps first \
  --rollout-k 3 --rollout-weight 0.5 --rollout-mask accepted \
  --epochs 12 --steps 1250 --lr 5e-6 --warmup 100 --min-lr-ratio 0.1 --grad-clip 1.0 \
  --eval-every 200 --eval-batch 8 --eval-starts 16 --eval-batches 20 --chain-ks 3 15 \
  --eval-renewal --renewal-dump-dir $WORK/dump --renewal-ks 3 7 --renewal-docs 48 \
  --renewal-greedy-only "${TM[@]}" --select-metric renewal@7 \
  --snapshot-every-eval --ckpt-every 100
```

* `--init FILE` loads a BF16 `mtp.*` snapshot on top of `--model-dir`, with a fresh optimizer and a
  fresh schedule. Pointing `--model-dir` at a written-back checkpoint (§12) has the same effect.
* `--renewal-dump-dir` keeps scoring the *previous* held-out documents, so stage-2 numbers compare
  directly with stage 1's.
* The measured stage 2 (v5) was trained on a re-extraction with new documents and prompt families,
  not on more passes over stage 1's data. `$WORK/dump-v2` is such a second dump directory, made
  with §6 and §7 against a new `SGLANG_MTP_DUMP_DIR`.

Other options, all off by default: `--exclude-buckets` (train without some buckets),
`--rollout-mask consistent`, `--soft-steps all`, `--optimizer adamw8bit`, and `--tap-layers ...` /
`--train-taps-only` (experimental tap fusion; needs a `SGLANG_MTP_DUMP_TAPS` dump, and the served
head must support it; see DRAFT_V2_SPEC). CPU smoke test:
`python -m mtptrain.train --tiny --synthetic 8 --steps 4 --soft-alpha 0.5 --rollout-k 2 --log-every 1`.

## 11. Evaluate offline with the renewal evaluator

`scripts/eval_renewal.py` replays the server's verify-and-advance loop on **held-out
self-generated** documents:

1. Start at the first generated token.
2. Roll the draft chain k steps.
3. Count the leading matches `a` against the target's argmax.
4. Advance `a + 1` and repeat.

It reports `tokens / verifies`, the same quantity as the server's
`completion_tokens / spec_verify_calls`, per bucket.

```bash
for CK in "" "--ckpt $WORK/run1/best-mtp.safetensors"; do
  $PY scripts/eval_renewal.py --dump-dir $WORK/dump --split eval --greedy-only \
      --ks 3 15 --per-bucket 200 "${TM[@]}" --device cuda $CK \
      --append $WORK/evals.jsonl
done
```

Run it without `--ckpt` first: the released head is the baseline for every comparison.
`--split` defaults to `eval`. Each output line records `ckpt_digest`, the content hash of the
checkpoint file, so a file replaced under the same name is not mistaken for the same candidate. A
k or bucket without a single verify reports `accept_len: null`, not 0.

* **Why this evaluator.** The teacher-forced `accept@k` the trainer also reports starts chains at
  uniformly random rows and scores only positions where the corpus agrees with the target. By
  construction that over-samples easy, long accepted runs, whereas the server weights every verify
  equally. Its agreement@1 remains a useful sanity check.
* **Held out means held out.** Only `--split eval` documents count. The training documents reward
  memorisation.
* **Calibration.** Two different things were done for the measured heads, and only the second is
  published. (1) The offline evaluator was compared with the server's acceptance on **private**
  data during development; no result or conclusion from that comparison is published, and the tool
  for it is not included. (2) Every head was then judged by a **server A/B on the public workloads**
  (§13); every acceptance number in this repository is such a server measurement. For your own head,
  check offline numbers against a §13 A/B before relying on them.
* The evaluator is still offline: it runs the BF16 PyTorch head with dense attention and no
  FP8/NVFP4 serving kernels. It ranks candidates; it does not replace §13.

## 12. Write back into a checkpoint copy

```bash
$PY writeback/write_mtp.py --src "$MTP_MODEL_DIR" \
    --dst $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mine \
    --ckpt $WORK/run1/best-mtp.safetensors --verify
```

The new directory symlinks every file that does not hold `mtp.*` tensors, pointing straight at the
real source file. It rewrites the three shards that do, with the trained tensors in place
(`--strategy rewrite`). `--strategy new-shard` instead moves all 31 into one new shard and repoints
the index.

`write_mtp.py` refuses when `--src` and `--dst` overlap, when any source shard resolves into or
through `--dst` (for example a source that was itself written back from `--dst`), and when a shard
name in an index is not a plain file name. It builds the output in a temporary directory and
verifies it there before anything replaces `--dst`, so a failed run leaves an existing `--dst`
untouched (`--force` is needed to replace one). Replacing is two renames; if the process dies
exactly between them, both trees are left next to `--dst` (`.<dst>.old-*` and the verified
`.<dst>.tmp-*`) and nothing is lost. The verification checks:

* every indexed tensor resolves and appears exactly once, with the source's dtype and shape;
* the trained tensors are bit-identical to the checkpoint you passed;
* every other tensor of a rewritten shard is bit-identical to the source's;
* every other file (symlinked shards, config, tokenizer, chat template) has the same full content
  as the source's file of that name, and no file is missing or extra;
* the index matches the shards. This matters because SGLang loads every tensor of every file the
  index references.

`--verify` re-runs the same checks on the final directory.

The written-back checkpoint is a derivative of Qwen3.8-Flash-Next, which is under the Qwen Community
License 1.0 (https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE), not Apache-2.0. If you
share it, include Qwen's LICENSE and notice with it, and check the licence's commercial conditions
(condition 2) first.

Serve it with `TARGET_MODEL=$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mine launch/serve-fast.sh ...`.
Nothing else changes. The token map and launch flags stay the same.

## 13. Validate with a server A/B, the only gate that counts

The offline numbers pick a candidate. Whether it ships is decided by the server on the public
workloads, with the protocol in [`measurement.md`](measurement.md) §6:

* a fresh server per arm, in ABBA order: original (or current) head vs new head, same launcher,
  same token map;
* greedy, the 8-prompt held-out sets per domain
  (`fnbench run ... --prompt-sets workloads/sets --prompt-limit 8 --require-acceptance`);
* acceptance *and* t/s per domain, analysed with `bench/bench/stats/paired_ab.py`.

```bash
cd "$REPO"/bench
TARGET_MODEL=$MTP_MODEL_DIR ../launch/serve-fast.sh w4 &        # arm A (wait for "ready")
.venv/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
    --workloads code-edit,prose-en,prose-ja,agent-loop --prompt-sets workloads/sets --prompt-limit 8 \
    --repeats 1 --sampling greedy --require-acceptance --allow-proc sglang --label A1 --out runs/A1.jsonl
# stop; restart with TARGET_MODEL=...-mine for B1, then B2, then A2; compare with paired_ab.py
```

Acceptance moves more reliably than t/s, because the head does not change the step time. Declare
the primary domain and the cycle budget before running. Prose effects of a few percent need several
ABBA cycles to resolve. Check every profile you serve, W4, W16 and the adaptive `wa`, because a head
can help short chains and be neutral or worse on long ones. Keep the new head only if acceptance
improves on your primary domain and nothing else regresses beyond noise. Also check that your own
traffic, not only the benchmark prompts, looks better.

The target verifies every drafted token, so the draft never decides on its own what is emitted. The
target's output is still not independent of the draft: with the default singleton expert pruning,
which experts are dropped depends on every token in the verify batch, so a new head can change the
target's arithmetic and occasionally its greedy output. Run the quality checks you rely on (at least
the needle test) with the new head as well, not only the acceptance A/B.

## 14. Input checks

Every tool refuses rather than guesses:

* An explicit path that does not exist (an `--eval-set`, a corpus source, a token map, a dump or
  model directory) is an error before anything is written. A corpus source that contributes no
  document is also an error.
* An existing but empty manifest selects no documents, and training or evaluation with zero train,
  eval or renewal documents is refused. `extract/build_index.py` exits 1 on an empty dump.
* Options the chosen mode would ignore are refused, and `--optimizer adamw8bit` without
  bitsandbytes is an error rather than a fallback.
* Re-runs are checked against what is already on disk: `build_eval_set.py` refuses to replace a
  different existing eval set without `--overwrite` (an identical re-run is fine);
  `eval_renewal.py --append` refuses a file written with other settings (the checkpoint, token map
  and dtype may differ; they are what a comparison varies); `eval_renewal.py --per-seq` must name a
  new file; `client.py` refuses to resume a mode and bucket with other chunking or generation
  settings.
* Content changed under an unchanged key is refused: `client.py` compares each manifest row's
  `input_hash` with the document its `split_key` names now, before anything can drop that
  document (§6), and a `train.py` resume compares the eval and renewal data, token map and base
  weights by content (§10).
* Identities are content, never a name, size, path or id list: model directories (every file,
  tokenizer included), checkpoints and the files `write_mtp.py` links or copies are compared by
  full-content digests. Hashing a multi-GB checkpoint the first time takes minutes; the digests
  are cached in `~/.cache/mtp-train/digests.json` (`$MTP_DIGEST_CACHE` overrides the path), keyed
  on each file's size, mtime, ctime, inode and device and recomputed whenever any of them changes.
* An option given with no value or an empty one (`--buckets` alone, `--eval-set ""`,
  `--exclude-buckets a,,b`) is refused, never read as "all", "none" or the default. So is an
  environment default that is set but empty (`MTP_MODEL_DIR=`).
* Eval documents that are training documents (same `doc_hash` or split key) are refused for
  every eval source (§8).
* A missing measurement is `null`, never 0 (§10); a selection metric the eval data cannot measure
  is refused at start-up.
* `extract/client.py` exits non-zero if any request failed; re-run it to retry only what is
  missing.

`mtp-train/README.md` has the full list.

## 15. Pitfalls

* **Radix cache on during extraction.** Requests that share a prefix are then dumped from the middle
  (`prefix_lens > 0`). The index refuses them as orphan continuations, so the build stops. Always
  `--disable-radix-cache` on the dump server.
* **Chat template mismatch.** The client must render with the same kwargs the server uses by
  default (`--chat-kwargs`), or the training text differs from served text.
* **Sequences over 2048 tokens.** Above the QSA budget the served attention is sparse, but the
  trainer's dense attention is not. `--max-len 2048` everywhere.
* **No top-K in the dump.** Without `SGLANG_MTP_DUMP_TOPK` the soft-target half of the recipe
  cannot run; `--soft-alpha` is refused on such a dump.
* **Changing `--seed` or `--holdout`** moves documents between train and eval.
* **Scoring on training documents** or on the corpus's teacher-forced agreement alone rewards
  memorisation and easy positions. The trainer refuses overlapping eval documents, but
  `eval_renewal.py --split all/train` still scores them if you ask.
* **Editing the corpus between client runs.** Resuming into the same manifest after a source
  changed is refused; start a new dump directory.
* **Taking the last step.** Snapshot every eval and select on held-out renewal acceptance; the
  last checkpoint is not necessarily the best.
* **Disk cap.** The hook stops at `SGLANG_MTP_DUMP_MAX_GB`. Manifest rows whose request is not
  completely on disk are ignored by every reader; `verify_dump.py` reports how many there are.
* **Secrets in session logs** end up in the dumps and the weights. Scrub them before extraction.
