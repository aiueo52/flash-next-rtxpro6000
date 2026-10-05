# Result files of the 2026-10 round

Raw results behind the lab notes in `docs/lab-notes/` (named below), `docs/optimizations.md` (2026-10 section), `docs/rejected.md` and sections 6-7 of
`../TABLES.md`. All runs: one RTX PRO 6000 Blackwell Max-Q, the `wa` profile unless the
directory says otherwise, `fnbench` on the BN1 held-out prompts of the four workloads (code-edit, prose-en, prose-ja, agent-loop;
4 per workload in `stack8/`, `st1/`, `xa1-st1/`, 8 in most others).
`greedy` = temperature 0; `lmstudio` = the LM Studio default sampling (T 0.8, top_p 0.95, top_k 40,
min_p 0.05). `*-probe.jsonl` are the per-request step counters (ms/step, tok/step).

| directory | experiment | lab note |
|---|---|---|
| `stack8/` | the 2026-10-01 stack, 8-start ABBA (A1 B1 B2 A2 B3 A3 A4 B4); `C1` = DT1 arm after it; `exact/` = w4 greedy exactness check (token counts and text hashes only) | `STACK_2026-10-01.md`, `DT1_DRAFT_TAIL_2026-10-01.md` |
| `st1/` | ST1 alone, 8 starts | `ST1_SHARED_TAIL_2026-10-01.md` |
| `xa1-st1/` | XA1 with ST1 in both arms, 8 starts | `XA1_DECODE_ATTN_2026-10-01.md` |
| `rs1/`, `rs1-minp/` | RS1 (dense rejection sampling), without / with the min_p fix | `RS1_REJECTION_SAMPLING.md` |
| `rs2/` | RS2 sparse RS alone, 8 starts | `RS2_SPARSE_RS_2026-10-01.md` |
| `rs2d-offline/`, `rs2d-abba/` | RS2d knobs offline grid; the RS package vs the stack (target-only), 8 starts | `RS2D_DRAFT_SHARPEN_2026-10-01.md`, `RS3_BLOCK_VERIFY_2026-10-01.md` |
| `rs4-k16/` | K=16 draft top-k, support timing and proposal kernels | `RS2D_DRAFT_SHARPEN_2026-10-01.md` (§K=16) |
| `sv/`, `sv2/` | sparse verify (SV1) ABBA; sparse top-k (SV2) GPU unit runs | `SV1_SPARSE_VERIFY.md`, `SV2_SPARSE_TOPK_2026-10-01.md` |
| `fg1/` | FG1 single ABBA (inconclusive) | `FG1_GDN_FRONT_OVERLAP.md` |
| `rq2-u2h/` | RQ2 u2h prologue, 4 starts | `FC_fc-moe_2026-10-01.md` |
| `ship/` | the shipped build, one start per profile (wa, w4, w16), single-prompt smokes | `STACK_2026-10-01.md` |
| `long-524k/` | 524,288-token context: needle tests, prefill and decode speed, GPU memory | `LONGCTX_2026-10-02.md` |

What was removed before publishing (`../strip_runs.py` and a path rewrite):

- every generated text field (completions, needle answers); only token counts and timings remain;
- absolute paths (rewritten to `/home/user/...`).

The `paired-*.json` files record sha256 sums of the *unstripped* run files they were computed from, so
those sums do not match the files here. The numbers in them are unchanged; `bench/bench/stats/ancova_ab.py`
on the stripped files reproduces the logged results (checked for `stack8/`).
