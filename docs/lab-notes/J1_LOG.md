# J1 — deleting `_hc_branch_stats_kernel` (K0) from the HC mix chain: **NO-GO**

Date: 2026-09-06. Branch `opus/hc-stats-fold` (worktree `~/tools/sglang-j1`), on
`codex/perf-v1` @ `ef31f26346` (PDL + R1/R6/R7 + H2). RTX PRO 6000 Blackwell
Max-Q, GPU shared, every run under `flock ~/.gpu.lock`.

`EXCLUSIVE_TIME_MAP.md` §6 item 4 ranked this at **−185 µs/step (W16) /
−147 µs (W4)**: K0 is 1.5–1.9 µs × 92 launches, 98 % exclusive, and
`MEGAKERNEL_SPEC.md` §3.1 measures 1.41 µs of that as fixed cost. The estimate
treats K0 as a pure launch.

**It is not.** Measured, deleting K0 costs **+2.4 µs/call** at the plain
boundaries and **+7.4 µs/call** at the fused ones, against the 1.6–2.1 µs it
costs to keep. Per step that is **+0.36 ms (plain) to +0.83 ms (production, W4)**
— the wrong sign, by a factor of 4–5.

---

## 1. What K0 actually does

Grid = M × hc (16 CTAs at W4, 64 at W16). Per (row, branch) CTA:

1. the sum of squares and the `inv_rms` it implies;
2. **`normed` = `(x * inv_rms * (1+w))` rounded to bf16 and stored** — this is
   the part the estimate missed;
3. a slice of the fp32 split-K workspace cleared for K1's atomics (which is why
   the graph carries no memset node);
4. with `SGLANG_HC_APPLY_MIX_FUSED` / `SGLANG_HC_LAYER_APPLY_FUSED` (R7/H2, on
   in production at 46/48 boundaries), the *previous* boundary's combine apply
   as a prologue.

K1's fast body (`stats_mode="norm"`) reads that materialized `normed` straight
into its `tl.dot`. The module docstring already said this was worth more than
the 327 KB round trip; nobody had priced it against K0's launch.

## 2. The three options, and why only one was worth measuring

**Option 3 — merge K1 and K2** is dead on counting, no GPU needed. Every K2 CTA
needs the complete `t[m, 0:320]`, i.e. the whole k-reduction over all 10240
columns, so each of the 160 CTAs would have to re-stream all of `w_down`:
160 × 3.28 MB = **525 MB/call** against 3.28 MB today. Shrinking K2's grid to
make that affordable takes it below the CTA count it needs to stream `w_up`.
The down→up boundary is a rank-320 reduction across every CTA; only a grid
barrier crosses it, and `MEGAKERNEL_SPEC.md` §3.2 prices that at +0.44 µs.

**Option 2 — emit the stats from the kernel that produces the residual.** With
R7/H2 on, K0 *is* that kernel (the apply is already its prologue). One step
further back is the block's last kernel — the attention `out_proj` GEMV
`[40,4,1]` or the MoE chain — and no CTA there owns a whole branch row
(the GEMV is split 40 ways in j and 4 in k), so none can form
`sum_j x[m,c,j]^2`. The boundaries that still run a standalone
`hc_combine_apply` are the PLE and last layers, 5–17 launches/step, which the
map already prices as low value.

**Option 1 — K0 into K1** (implemented, `SGLANG_HC_STATS_FOLD=1`). K1 keeps its
`[20,5]` grid; every CTA re-derives `inv_rms` for the branch its k-group lies
in, re-runs the apply prologue for that branch, and one designated CTA per
branch writes `applied` back. This is the only fold that leaves K2 and `t_raw`
untouched and is arithmetically identical stage by stage.

## 3. The measurement

`bench/hc_mix2/bench_j1.py` (fork). CUDA-graph replay of `len(sets)` chained
calls over 82 rotating weight copies (512 MB working set, ≥ 4× L2); **`chain` is
the graph's wall time per call**, which is the metric a launch deletion has to
move — a deleted node's fixed cost never shows up in the surviving kernels'
CUPTI medians. `K0/K1/K2` are CUPTI medians for the same configuration.
`SGLANG_TRITON_PDL=1` (the W4 production setting); the PDL=0 table differs by
under 0.5 µs and does not change any sign.

`stats` is the decomposition control: K0 still runs but emits only `inv_rms`, so
K1 builds `normed` itself. It isolates "K1 lost its pre-normalized tile" from
"K1 re-reads the branch".

| M | boundary | variant | K0 | K1 | K2 | **chain** |
|--:|---|---|--:|--:|--:|--:|
| 4 | plain | norm (production) | 1.60 | 3.87 | 4.77 | **9.59** |
| 4 | plain | stats | 1.21 | 5.54 | 4.77 | 10.96 |
| 4 | plain | **fold (no K0)** | – | 7.94 | 4.64 | **13.52** |
| 4 | apply+shared | norm (production) | 2.05 | 3.87 | 4.77 | **10.13** |
| 4 | apply+shared | stats | 1.54 | 5.54 | 4.77 | 11.38 |
| 4 | apply+shared | **fold (no K0)** | – | 13.31 | 4.61 | **19.10** |
| 16 | plain | norm | 1.60 | 3.97 | 4.77 | **9.76** |
| 16 | plain | fold | – | 9.06 | 4.61 | **14.69** |
| 16 | apply+shared | norm | 2.02 | 3.97 | 4.77 | **10.19** |
| 16 | apply+shared | fold | – | 16.54 | 4.61 | **22.17** |
| 1 | plain | norm | 1.57 | 3.84 | 4.61 | **9.57** |
| 1 | plain | fold | – | 7.20 | 4.51 | **12.69** |
| 1 | apply+shared | norm | 2.08 | 3.87 | 4.64 | **9.90** |
| 1 | apply+shared | fold | – | 10.21 | 4.51 | **15.66** |

### 3.1 The two costs, separated

Take M=4, plain. K0 costs **1.60 µs**. Deleting it costs:

| | µs | what |
|---|--:|---|
| K1 3.87 → 5.54 | **+1.67** | K1 builds `normed` instead of reading K0's tile |
| K1 5.54 → 7.94 | **+2.40** | K1 re-reads its whole 2560-wide branch for the stats |
| K0 1.60 → 0 | −1.60 | the launch that was the point |
| **net** | **+2.47** | |

**The first row alone is bigger than K0.** That is the finding: 1.6 µs of K0
buys 1.7 µs in K1, so K0's *fixed* launch cost is not recoverable by any design
that moves the statistics elsewhere — the normalized tile has to be produced by
someone, and only a kernel with one CTA per (row, branch) can produce it in the
layout `tl.dot` wants. This also disposes of the two variants not implemented
(K1 emitting split partials with the `inv_rms` scaling deferred to K2, and
per-branch k-groups with a plain-store `t_raw`): both leave K1 without
`inv_rms` at dot time, so both pay the +1.67 µs, and both then add a 4×-wider
`t_raw` read to K2 (160 CTAs × 82 KB = 13 MB at M=16) on top.

At the fused boundaries the second row is far worse — `x` is not read but
*computed*, so each of the 25 CTAs per branch re-loads both the residual and the
block output over the full branch: K1 3.87 → 13.31 µs at M=4, 16.54 at M=16.

### 3.2 Tuning does not rescue it

`--sweep-n` over `block_n` × `block_s` (`bench_j1.py`). `block_n=64` (the pinned
fp8 value) stays best; `block_n=32` is 2–13 µs worse and `160` is not a legal
`tl.arange`. The stats tile wants to be **smaller** than the k-tile: `block_s`
128 is the optimum, 1.5–3.6 µs better than the 512 default at M=16, and 2048
spills. Best point measured, M=16 apply+shared: **19.16 µs** against the
production chain's 10.19.

## 4. Numerics (they were fine; the kernel is correct, just slow)

`bench/hc_mix2/j1_numerics.py` — real HC weights out of the `mtpft3` shards
(layer 6 attention HC + layer 5 inject), M ∈ {1,4,16} × {bf16, fp8 mix weights}
× {plain, apply, apply+shared} = 18 cases, fold vs the K0 path:

* **`applied`: bit-identical in 18/18.** The apply prologue reproduces K0's
  arithmetic exactly, including R6's round-through-bf16 of the shared join.
* **`normed`: bit-identical in 18/18** on real weights. On synthetic
  `randn` inputs it differs on 1–2 elements out of 40 960 / 163 840 in 2 of 9
  cases (`maxrel 4.35e-3`, i.e. one bf16 ulp): `inv_rms` is a different fp32
  *summation order* (K0 reduces one 4096-lane tile, K1 five 512-lane chunks),
  which lands on a rounding boundary occasionally.
* **`mixed`: 17/18 identical**, the exception 41/40 960 elements at
  `maxabs 3.9e-3`. `t_raw` is accumulated with device-scope atomics, and
  `bench/hc_layer_boundary/mix_nondeterminism.py` already showed the *unchanged*
  call differs from itself 5 runs out of 5 at M=16.

Default path (`SGLANG_HC_STATS_FOLD` unset): the fold adds five pointer
parameters and four constexprs to `_hc_down_kernel`, so the flag-off kernel is
not textually the pre-J1 one. `bench/hc_mix2/j1_default_path.py` dumps
`normed`/`applied` for 3 widths × 2 weight dtypes × 3 apply shapes from each
build and `torch.equal`s them: **30/30 byte-identical** between
`ef31f26346` and this branch.

## 5. Verdict and what it means for the map

**NO-GO.** The flag stays default-off and the code stays as the evidence, the
way `SGLANG_HC_GATE_EARLY=FUSED` did.

Per step, using the map's 91.9 K0 launches/step in verify plus the draft ones,
and the production boundary mix (46/48 fused):

| | measured Δ/call | Δ/step at ~92 calls |
|---|--:|--:|
| plain boundaries | +2.5 µs | +0.23 ms |
| fused boundaries (production) | +7.4 … +9.0 µs | **+0.68 … +0.83 ms** |

`EXCLUSIVE_TIME_MAP.md` §6 item 4 should be struck, and its §2.1 framing —
"~1.1 µs of critical path per tiny launch, and this 0.4–0.75 ms is the total
prize for all launch-deletion work" — needs the qualifier this task establishes:
**that prize is only collectable where the deleted kernel's work is genuinely
free somewhere else.** K0 is 98 % exclusive and 1.41 µs of fixed cost, and it is
still not deletable, because it is the producer of a tile the next kernel reads
in the layout its `tl.dot` wants. The same test — "what does the surviving
consumer lose when the producer's output stops being materialized?" — should be
applied to the remaining sub-2 µs families before any of them is costed as a
launch.

## 6. Reproduction

```bash
cd ~/tools/sglang-j1 && . ~/tools/flash-next-bench/bench/megakernel/env.sh
export PYTHONPATH=~/tools/sglang-j1/python
flock -w 21600 ~/.gpu.lock env SGLANG_TRITON_PDL=1 $PY bench/hc_mix2/bench_j1.py \
    --cupti --variants norm stats fold          # section 3
flock -w 21600 ~/.gpu.lock $PY bench/hc_mix2/j1_numerics.py        # section 4
flock -w 21600 ~/.gpu.lock env SGLANG_TRITON_PDL=1 $PY bench/hc_mix2/bench_j1.py \
    --sweep-n --variants fold --rows 1 16       # section 3.2
```

No in-server A/B was run: a change that is 2.5–9 µs/call slower in a chained
CUDA-graph replay has no path to being faster in the server, and the GPU is
shared.
