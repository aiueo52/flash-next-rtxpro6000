# mtp-train

Fine-tuning pipeline for the MTP (NEXTN) draft head of Qwen3.8-Flash-Next: dump the serving
target's own hidden states with an SGLang hook, train the 2.6 B-parameter draft layer on them
(target frozen), score candidates offline with an evaluator that replays the server's
verify-and-advance process, and write the result back into a copy of the checkpoint that the
server loads unchanged.

The step-by-step guide, including hardware needs and the recipe that produced the measured head,
is [`../docs/train-your-own-mtp-head.md`](../docs/train-your-own-mtp-head.md). The fine-tuned heads
used for the published measurements were trained on the author's private data and are **not**
included; this directory contains only code, and tests that use synthetic data.

## Layout

This directory holds the guide's main path only: one hook server, one dump directory, the trainer,
the renewal evaluator and the write-back.

| path | what |
|---|---|
| `corpus/formats.py` | readers: Codex CLI rollout JSONL and LM Studio conversation JSON (via `../sim/ngramsim`), Claude Code session JSONL, generic JSONL/JSON/TXT/MD, Hugging Face Arrow cache files |
| `extract/client.py` | streams corpus documents to a hook-enabled server and writes `manifest.jsonl`: `--mode extract` sends chat-templated windows of ≤ 2048 tokens as prefill-only requests; `--mode selfgen` lets the server continue prompts from the same documents greedily, then sends prompt + continuation as one prefill-only request |
| `extract/build_index.py`, `build_eval_set.py` | doc_hash → shard index (joins chunked-prefill pieces; logic in `mtptrain/dumpindex.py`, which every loader uses); frozen held-out subset for teacher-forced eval |
| `extract/verify_dump.py` | dump sanity check: every manifest document on disk re-hashed, target argmax vs the tokens the server emitted |
| `mtptrain/` | the trainable head (`model.py`, `layers.py`), weight I/O (`weights.py`), dump loader (`data.py`), chunked hard/soft losses (`loss.py`), K-step rollout loss (`rollout.py`), teacher-forced eval (`evaluate.py`), renewal evaluator (`renewal.py`, `select.py`), training loop with exact mid-epoch resume (`train.py`), crash-safe file writes and resumable JSONL logs (`fileio.py`) |
| `scripts/eval_renewal.py` | offline acceptance length computed like the server's metric (tokens per verify) on held-out self-generated documents |
| `writeback/write_mtp.py` | writes trained `mtp.*` tensors into a new model dir (untouched shards symlinked) and verifies it |
| `parity/serve.sh` | starts, waits for and stops the hook-enabled dump server (`start-extract`) or the plain serving config (`start-prod`) |
| `tests/` | CPU unit tests (tiny config, synthetic data) |

The SGLang side of the pipeline is the dump hook in [`../patches/sglang/mtp-dump/`](../patches/sglang/mtp-dump/)
(environment variables `SGLANG_MTP_DUMP_DIR`, `SGLANG_MTP_DUMP_TOPK`, `SGLANG_MTP_DUMP_TAPS`;
see the guide). The hook's draft-side dump (`SGLANG_MTP_DRAFT_DUMP_DIR`) is for parity checks,
whose tools are not included here.

## Install

Use the SGLang virtualenv from [`../docs/reproduce.md`](../docs/reproduce.md); it has everything
(`torch`, `safetensors`, `transformers`, `requests`). For the tests alone, any Python ≥ 3.10 with
`torch`, `safetensors` and `pytest` works (see `requirements.txt`). The Codex / LM Studio readers
import `../sim/ngramsim` (needs `tokenizers`); set `NGRAMSIM_PATH` if `sim/` is elsewhere.

Paths are always arguments. The only defaults are `MTP_MODEL_DIR` (the serving checkpoint,
default `$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`) and, for `scripts/eval_renewal.py`,
`MTP_DUMP_DIR`.

Every tool uses what it is given or stops: a path that does not exist, an option the chosen mode
would ignore, an input that selects nothing (an empty manifest, a glob matching no file, filters
that remove everything) or a resume/append with settings other than the ones already recorded is
an error with the reason, never a silent default or an empty result. `train.py --resume` refuses
changed evaluation or selection settings unless `--reset-best` is given. `train.py` also refuses
eval documents (the eval set, the held-out split, the renewal documents) that are training
documents by content hash or split key, and a `--select-metric` its eval data cannot measure;
`client.py` refuses to resume when a document changed under the same split key. A metric with
nothing to measure is `null` next to its zero count, never 0, and never selects a best.

Identities are content, never names, sizes, paths or id lists: a model directory by the digest of
every file in it (shards, config, tokenizer, chat template), eval and renewal data by a digest of
every tensor the evals read, checkpoints and write-back outputs by their full content. File digests
are cached in `~/.cache/mtp-train/digests.json` (override with `MTP_DIGEST_CACHE`), keyed on size,
mtime, ctime, inode and device; the first hash of a large checkpoint takes minutes. An option given
with no value or an empty value (or an empty item in a comma list) is refused, never widened to
"all" or "none"; so is an environment default that is set but empty.

## Tests

```bash
cd mtp-train
CUDA_VISIBLE_DEVICES="" "$SGLANG_DIR/.venv/bin/python" -m pytest -q
```

`tests/conftest.py` points `MTP_DIGEST_CACHE` at a temporary directory, so the suite never writes
to `~/.cache`.

The tests need `torch`, `safetensors` and `pytest`, which a system Python usually lacks; the SGLang
virtualenv (`$SGLANG_DIR/.venv`, see [`../docs/reproduce.md`](../docs/reproduce.md)) has all three.

Everything runs on CPU with synthetic data. Two tests also check the real checkpoint's tensor names,
shapes and one CPU forward (about 8 GiB RAM); they are skipped unless `MTP_MODEL_DIR` points at a
local copy of the checkpoint.

## Licence

Apache-2.0, like the rest of the repository. Qwen3.8-Flash-Next is under the Qwen Community License 1.0 (https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE), not Apache-2.0; this repo contains no model files. A checkpoint written back with a
fine-tuned head is a derivative of the model; see §12 of
[the training guide](../docs/train-your-own-mtp-head.md) before sharing one. `mtptrain/layers.py` and `mtptrain/model.py` re-implement
the Qwen4-Exp MTP layer following SGLang's model code (Apache-2.0); see their headers and
[`../NOTICE`](../NOTICE).
