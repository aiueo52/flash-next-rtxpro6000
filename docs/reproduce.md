# Reproducing the setup

This is the recipe behind the numbers in `results/TABLES.md`, written for someone with the same
class of machine. It builds the 2026-10-02 production state by default; the notes in each step say how
to get the 2026-09-08 state that sections 1-5 of `TABLES.md` measured. The code is exact (patch series, FlashInfer patches). The launch flags are those of
`launch/serve-fast.sh`; its memory defaults differ from some measured runs, and section 4 shows the
settings of the final measured tables. Two inputs of the measured runs are **not included**: the privately fine-tuned MTP draft head and the
reduced draft vocabulary (token map), both built from private data (see the end of section 4). With
the public MTP head and a token map you build yourself, expect lower acceptance and lower t/s than
the published tables. Nothing here downloads weights for you or changes system settings.

The commands below use two absolute paths; set them once in your shell:

```bash
export REPO=/path/to/flash-next-rtxpro6000        # this repository
export WORK=/path/to/workdir                       # where the SGLang clone goes
export SGLANG_DIR=$WORK/sglang-rtxpro6000
```

## 0. What you need

| item | measured with | notes |
|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell **Max-Q** Workstation Edition, 96 GB GDDR7, SM120 | power limit 325 W (the card's maximum; default 300 W) from 2026-09-03 on; every 2026-09-02 run, the baseline included, used the default 300 W. No power-cap throttling was seen in 39 nvidia-smi samples taken at 100 ms intervals during decode (2026-09-07, X3; per-arm median SM clock 2242–2287 MHz); throttling shorter than the sampling interval cannot be ruled out |
| driver | 595.84 (open kernel module) | the base fork was qualified on 610.57.04 |
| OS / CPU / RAM | Ubuntu 24.04.4, kernel 7.0, AMD Threadripper 9960X, 256 GB | the PLE embedding table (~28 GB) is pinned in host RAM (`--ple-offload-embedding`) |
| CUDA toolkit | 13.3 (nvcc 13.3.73), conda-packaged | needed for FlashInfer / SGLang JIT |
| Python stack | Python 3.12.13, torch 2.13.0+cu130, flashinfer-python 0.6.17, sgl-kernel 0.4.6.post1, triton 3.7.1, transformers 5.12.1 | the base fork's pinned set |
| checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (≈126 GB) | NVFP4 routed experts, BF16 dense weights, native MTP layer |
| display | the same GPU also drove a 4K 160 Hz KDE desktop | this costs 5–7 % step time, see `measurement.md` |

## 1. Build the patched SGLang

```bash
git clone https://github.com/jpezzulli/sglang-rtxpro6000 "$SGLANG_DIR"
"$REPO"/patches/sglang/apply.sh "$SGLANG_DIR"       # branch flash-next-fast = base + 121 patches
```

Patches 0001-0105 are the 2026-09-08 tree, 0106-0121 the 2026-10-02 additions. Every 2026-10 feature is
behind its own flag (default off in the code, on in `launch/serve-fast.sh`), so with those flags set to 0 the full
series serves the September SGLang configuration (`docs/optimizations.md`, appendix). The FlashInfer
prologue changes (patches 08 and 10 below) have no flag; stop at patch 07 for the September FlashInfer.

Then build the venv inside `$SGLANG_DIR` exactly as the base fork's `BUILD.md` describes (uv,
Python 3.12.13, editable install of `python/`), so that it ends up at `$SGLANG_DIR/.venv`. Pitfalls we hit on a machine without root:

- the venv needs `ninja` and `cmake` for the JIT builds;
- a conda-packaged CUDA has `lib/` but no `lib64/` (`ln -s lib lib64` inside that env) and keeps its
  headers under `targets/x86_64-linux/include` (export `CPATH`; `launch/serve-local.sh` does this);
- put the venv's `bin/` on `PATH` (the launcher does).

## 2. Patch FlashInfer inside that venv

```bash
"$REPO"/patches/flashinfer/apply.sh "$SGLANG_DIR/.venv/lib/python3.12/site-packages"          # 01-10
# "$REPO"/patches/flashinfer/apply.sh "$SGLANG_DIR/.venv/lib/python3.12/site-packages" 07     # the 2026-09-08 stack
```

The optional second argument is the last patch to apply. The default (10) gives the 2026-10-02
production behaviour from a normal JIT build; `apply.sh`'s header explains 09 vs 10. Run it after the
venv is complete: a later reinstall of `flashinfer-python` silently undoes it, and the script refuses
to run on files that are not pristine 0.6.17. The first MoE call after this rebuilds FlashInfer's JIT
module (about two minutes).

Production serves a frozen copy of the patched FlashInfer and its JIT cache (`FLASHINFER_DIR`,
`FLASHINFER_WORKSPACE_BASE`, `FLASHINFER_P2_NO_NINJA=1`; see the comments in `launch/serve-fast.sh`), so
that later development cannot change the served build. With a single venv you do not need this.

## 3. Get the model and build a draft token map

Download `RadixArk/Qwen3.8-Flash-Next-NVFP4` to e.g. `$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`.
Use the published public-data map `tokenmaps/public/public_49152.pt` (49,152 rows, built only from
public datasets; licence CC BY-SA 4.0, see `tokenmaps/README.md` and `NOTICE`), or build your own as
described in `tokenmaps/README.md` (49,152 rows recommended).
Without one the launcher still works, but the draft head runs over the full vocabulary and is much
slower at wide draft widths.

## 4. Launch

```bash
export TOKEN_MAP=$REPO/tokenmaps/public/public_49152.pt   # or the map you built in step 3
"$REPO"/launch/serve-fast.sh wa                   # or w4 / w16; uses $SGLANG_DIR
```

The final measured tables (`results/TABLES.md` section 2, the BN1 study of 2026-09-08) used memory
fraction 0.920 and `MAX_TOTAL_TOKENS=131072` for every profile, with no display override
(`SERVE_DISPLAY_HZ` unset). To launch with those settings:

```bash
W4_MEM_FRACTION=0.920 W16_MEM_FRACTION=0.920 WA_MEM_FRACTION=0.920 MAX_TOTAL_TOKENS=131072 \
  "$REPO"/launch/serve-fast.sh wa                 # or w4 / w16
```

The headline table (section 1, 2026-09-06) ran with the launcher's memory defaults of that day.
`launch/as-measured/serve-fast.sh` records the final defaults (W4 0.935, W16 0.93, wa 0.925) and, in
its dated comments, the changes made along the way; the exact per-run values of that session are not
separately recorded.

The server listens on `127.0.0.1:8001` (OpenAI-compatible, model name `flash-next`). Start-up takes a
few minutes (weights, CUDA-graph capture for every draft width, JIT on the first run). If the KV pool
check fails at start-up, lower `W4_MEM_FRACTION` / `W16_MEM_FRACTION` / `WA_MEM_FRACTION`; the defaults
leave about 4 GB free for a desktop on the same GPU.

Differences from the measured configuration that you cannot remove:

- **MTP head.** The measurements used privately fine-tuned MTP heads (`mtpft3` for the 2026-09-06
  headline table, `mtpft5` from 2026-09-07 on). They are not published because their training data
  came from private sources. Measured in-server (greedy, 4-repeat A/Bs): v3 vs the original head
  raised W4 acceptance by 0–7 % (prose-en +6 %, agent-loop +7 %, prose-ja +2 %, code ±0) and was
  neutral at W16; v5 vs v3 added +1–5 % at W4 and +2–8 % at W16. With the public head expect
  correspondingly lower acceptance and t/s (roughly 5–10 % on prose/agent), or train your own head
  with [`train-your-own-mtp-head.md`](train-your-own-mtp-head.md).
- **Token map.** The measured map was built partly from private data and is not published. The public
  map shares 72.0 % of its ids. With the original head and the public map together (2026-10-06,
  `results/TABLES.md` section 8), `wa` greedy ran 468 / 232 / 248 / 307 t/s (code-edit / prose-en /
  prose-ja / agent-loop) vs 584 / 253 / 290 / 359 with the private pair, same build and launcher.
- **2026-10 flags.** The launcher turns on every 2026-10-02 feature by default, including rejection
  sampling for sampling requests (`SGLANG_OPT_SPEC_SPARSE_RS=1`). Set the flags listed in the appendix of
  `docs/optimizations.md` to 0 to compare with the September tables.
- **Context length.** The launcher defaults to 262,144 tokens. `CONTEXT_LENGTH=524288` turns on factor-2
  YaRN and, for `wa`, a 0.96 memory fraction; it needs a nearly idle desktop on the same GPU
  (`docs/optimizations.md` H5).

## 5. Measure

The harness is `bench/` (a copy of the `flash-next-bench` repository; `fnbench` needs only `requests`).

```bash
cd "$REPO"/bench
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# the headline workloads: 1 warm-up + 2 greedy repeats per workload
.venv/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
    --workloads code-edit,prose-en,agent-loop --repeats 2 --sampling greedy \
    --allow-proc sglang --label mine-wa --out runs/mine-wa.jsonl
.venv/bin/python -m fnbench report runs/mine-wa.jsonl
# the 8-prompt held-out sets used for the robust numbers
.venv/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
    --workloads code-edit,prose-en,prose-ja,agent-loop --prompt-sets workloads/sets --prompt-limit 8 \
    --repeats 1 --sampling greedy --require-acceptance --allow-proc sglang --out runs/mine-sets.jsonl
# long-context retrieval sanity check (18.5k-token needle)
.venv/bin/python prof/needle_test.py 18500 0.4     # approx tokens, depth (thinking disabled)
```

- `--allow-proc sglang` lets the GPU guard accept the server's own process.
- The server must run with `--enable-metrics` (the launcher sets it) so that acceptance can be read
  from `/metrics`.
- Compare with `results/runs/p2m2332-*.jsonl` (headline) and `results/bn1-study-v2/` (sets), e.g. by
  copying your stripped files next to them (`results/strip_runs.py in.jsonl out.jsonl`).
- For A/B decisions use the protocol in `measurement.md` (fresh server per arm, ABBA, 8 prompts per
  domain, `bench/bench/stats/paired_ab.py`), not single runs. The 2026-10 round used 8 server starts and
  `bench/bench/stats/ancova_ab.py` (`measurement.md` §9).
- `--sampling lmstudio` sends LM Studio's default sampler (temperature 0.8, top_p 0.95, top_k 40,
  min_p 0.05). Measure both modes: several 2026-10 changes act only on sampling requests.

## 6. Optional: profiling

`bench/prof/validate.sh <label> <w4|w16>` starts a server, captures a 20-step torch-profiler (CUPTI)
trace with `profile_decode2.py`, and summarises it with `trimmed_step.py` (kernels/step, busy time)
and `exclusive_time.py` (critical-path time per kernel family). These scripts were written for this
machine: read them and adjust the paths (`$HOME/...`) before running.
