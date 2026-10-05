# FG1: GDN verify front overlap (fc-glue ideas 1 + 2)

Status (2026-10-01 13:20): prototype, commit dbf09dabe7 on `opus/fc-glue` (`~/tools/sglang-fc-glue`, based on production 7b4d539f9b); in the stack as ef95a68b66. Smoke PASS, ABBA inconclusive (§4.2); judged inside the 8-start stack ABBA.

## 0. Summary

- **What:** two stream-placement changes in front of every GDN layer's input projection, inside the verify CUDA graph. Kernels, inputs and outputs are unchanged, so the result is bit-exact.
  1. **Idea 1 (swap):** qkvz (fp8 GEMV, 256–512 CTAs) goes to the GDN alt stream and b/a (bf16 GEMV, 3 CTAs) to the main stream. Today b/a sits on the alt stream and cannot get an SM until the qkvz wave drains: it starts ~26 µs late and ends 1.7–3.4 µs after qkvz (FC_fc-glue §1.4).
  2. **Idea 2 (gate fork):** the MOVED combine gate of the layer's attention HC (`hc_combine_gate`, 1.5 µs) forks onto the same alt stream right after HC mix K0, so it overlaps K1/K2 instead of running alone between K2 and qkvz (FC_fc-glue §1.5).
- **Why:** microbench (FC_fc-glue §4b, graph replay, paired median per GDN layer front): **−3.57 µs at M=4, −3.15 µs at M=16**. Times 36 GDN layers: **about −0.13 ms/step at W4 (−1.5%) and −0.11 ms/step at W16 (−0.7%)**.
- **Switch:** `SGLANG_OPT_GDN_FRONT_OVERLAP=1`, default off.
- **Scope:** greedy and sampling alike (the verify forward is shared). Prefill and eager runs are untouched: the direct two-stream path only exists under CUDA-graph capture with at most 16 rows.

## 1. Code (dbf09dabe7)

| File | Change |
|---|---|
| `srt/environ.py` | `SGLANG_OPT_GDN_FRONT_OVERLAP = EnvBool(False)` |
| `srt/layers/hc_mix2_triton.py` | `hc_norm_mix2(..., *, after_normed=None)`: a hook called right after the launch that completes `normed` (K0 in the default "norm" stats mode, K1 otherwise) |
| `srt/layers/hyperconnection.py` | `GatedResidual._side_gate(hyper_input, gate_stream)` returns `(hook, partials)`; `mix()` and `combine_then_mix()` (the H2 path) take `gate_stream` and return the side partials in the residual tuple. The base class `mix()` accepts and ignores `gate_stream` |
| `srt/models/qwen3_5.py` | `Qwen3_5GatedDeltaNet._direct_use_alt()` (the existing dual-stream condition), `front_gate_stream(num_tokens)`, and the swapped branch in `_forward_input_proj_direct` |
| `srt/models/qwen4_exp.py` | `Qwen4ExpLinearDecoderLayer._attn_gate_stream()` asks `linear_attn.front_gate_stream(rows)` and passes the answer through `_prepare_qwen4_exp_attn` |

## 2. Why every fork is joined

- The gate uses the GDN alt stream itself, so the `current_stream.wait_stream(alt_stream)` that already ends the two-stream input projection also joins the gate. On the alt stream the gate sits ahead of qkvz.
- The gate partials are read at the MLP seam (`combine_then_mix` into the MLP HC). That runs after `linear_attn`, so after the join.
- `front_gate_stream` offers the stream only when the flag is on, the direct layout applies, 0 < rows ≤ 16 and `_direct_use_alt` holds (alt stream present, graph capture, the dual-stream threshold). Under the same conditions the fallback `_forward_input_proj` also forks and joins the alt stream, so even a qkvz GEMV refusal still joins.
- Idle batches get no stream (`_attn_gate_stream` checks `is_idle()`), since they skip `linear_attn`.
- A missing join would make capture fail loudly (unjoined stream), not corrupt data.
- The gate kernel calls `PDLWaitPrimary` before reading `normed`, so it is safe even if the cross-stream edge became programmatic.
- Server graphs are all `backend=full` (no piecewise), so the fork and the join are in the same graph.

## 3. Validation so far

- `ast` parse and a CPU import of the five modules (`CUDA_VISIBLE_DEVICES=""`): pass; the signatures are as intended.
- JIT cache (`.cache` copied from `~/tools/sglang-sv`): `find_prebuilt` probe hit=15 miss=3, the same three stale `hc_combine` variants as sv and rs-dev, whose servers compile nothing.
- Microbench of the same kernels in both placements: FC_fc-glue §4b.

## 4. Server smoke and ABBA (`bench/fg1/`)

- `arm_fg.sh <label> <off|on> [N]`: a fresh `serve-fast.sh wa` from the fc-glue worktree on port 8021, then:
  - per workload (code-edit, prose-en, prose-ja, agent-loop) × {greedy, sampling}: N = 4 unprofiled 600-token requests through `prof/fc_sampling_probe.py` (ms/step and tok/step per request);
  - one profiled greedy code-edit request (20 steps);
  - the server-log error count;
  - `gate_streams.py`: per-stream counts of `hc_combine_gate_kernel` and `_w8a16_gemv_kernel`, and how many gates on the alt stream overlapped the main stream's K1/K2. Production trace (fc1002-wa): 0 gates on the alt stream (stream 165).
- Smoke: `arm_fg.sh smoke-fg on 0`. It must show 0 error lines and gates on the alt stream.
- `abba_fg.sh`: A1 off, B1 on, B2 on, A2 off, then `fg_table.py` (per cell B/A change with a bootstrap CI over requests, ABBA block ratios, A2/A1 drift, pooled per mode).
- BN1 tok/s is not used: its per-domain CI is about ±5% (RS1 ABBA), far wider than the expected ~1%. Greedy requests repeat the same steps in every arm, so greedy ms/step pairs cleanly.

### 4.1 Smoke (12:23-12:29, `runs/fg1/smoke-fg.log`): PASS

- `SGLANG_OPT_GDN_FRONT_OVERLAP=1`, commit dbf09dabe7: server up, 0 error lines.
- Profiled 20-step greedy code-edit trace, `hc_combine_gate_kernel` per stream: alt (13) 720, main (173) 1000,
  draft (174) 480. 720 = 36 GDN layers x 20 steps, so every GDN verify layer forked its gate.
- ~~Only 220 of the 720 forked gates overlapped the main stream's K1/K2.~~ **Correction (14:35):** that count
  compared against the K1/K2 of one stream id, but wa's graphs (one per width) put the main role on more than one
  stream id. Counting any K1/K2 on another stream that overlaps the gate (`bench/fg1/gate_overlap.py`; one graph
  runs at a time, so it is the gate's own graph) gives **720 of 720** here, in both ABBA B arms and in the stack
  arms, and 0 with FG1 off. Every forked gate overlaps K1/K2 as designed.
- `_w8a16_gemv_kernel` per stream: alt 2300, main 2480, draft 1460 (qkvz now on the alt stream).
- Unprofiled single requests: greedy prose-en 15.96 ms/step at 4.35 tok/step, sampling code-edit 13.41 at 6.90
  (no comparison arm).

### 4.2 ABBA (12:33-13:01, `runs/fg1/abba-fg.log`): inconclusive

- All four arms: 0 error lines. B arms: 736 / 720 gates on the alt stream, all 720 per 20 steps overlapping K1/K2 (`gate_overlap.py`; `gate_streams.py` said 209 / 220, §4.1). A arms: 0.
- `fg_table.py`, B/A with a 95% CI over requests, pooled:

| mode | ms/step | tok/step |
|---|---|---|
| greedy | −0.84% [−5.91, +4.44] | −1.98% [−9.80, +6.29] |
| sampling | −0.01% [−4.72, +4.69] | −0.03% [−7.85, +8.92] |

- `bench/stats/ancova_ab.py` (log ms/step on per-workload intercepts and slopes of log tok/step, drift, B):
  greedy +0.33% [−2.44, +3.17], sampling −0.10% [−2.76, +2.62], residual sd 5.5% per request.
- The covariate-adjusted arm means (greedy ms/step) are A1 −0.02%, B1 +3.55%, B2 −3.11%, A2 −0.30%. The two
  B arms differ by 6.7% with the same code, so one server start moves the result by a few percent. Four
  starts cannot resolve a 1% change, however many requests each arm has.
- §5's rule is not met (the CI does not exclude 0). Nothing says FG1 hurts either. FG1 is bit-exact and
  the trace shows it working (gates forked and overlapping), so it rides in the stack ABBA. That ABBA uses
  8 server starts (`bench/stack/chain_stack8.sh`) and judges the set, not FG1 alone.

## 5. Decision

- **Adopt (default on in the fork)** if:
  - greedy pooled ms/step drops with a CI that excludes 0;
  - tok/step is unchanged within noise (the change is bit-exact);
  - 0 error lines in all arms, and the B arms show gates on the alt stream.
- The expected size is −0.7..−1.5% ms/step; less than half of that in the server means contention eats the overlap, and the trace timeline should say where.
