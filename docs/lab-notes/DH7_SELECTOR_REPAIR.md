# DH7 — production-state selector repair and restricted W16 eligibility

Status: COMPLETE. The candidate was evaluated offline and in timed serving runs on private data; no result or conclusion from that evaluation is published. It is not part of the production configuration; no production changes, commit or push. See Closure.

> Publication note: every capture, offline evaluation and timed serving run in DH7 used prompts drawn from the author's private MTP-training prompt set ("v5 holdout parents"). All numbers measured on those inputs (error tables, fallback rates, parent/chain/state/row counts, context ranges, head-region timings, TPS / acceptance tables, bootstrap bounds, selector-call counts) have been removed. Qualitative outcomes and pass/fail results measured on those inputs have also been removed. This note keeps only the contract, the implementation, which checks were run, and the resulting state of the software.

## Frozen contract (2026-09-08)

Task 1 in ASTRA_REVIEW_2026-09-08.md section 5, including sections 1–3, is the experiment contract. v5 MTP, P2 tau=.08, BS=1, current production stack, unchanged target weights and initial target-fed full head. Only explicitly caller-designated code at W16 is eligible; W4/W8, prose, agent, unknown requests bypass before selector execution. Multilingual code is included. Diagnostic capture/oracle is excluded from timed requests.

Gate: eligible W16 code delivered decode throughput >=1.03x production; acceptance loss <=1% on every supported workload; matched A/T inequality using mean head-region cost including fallback; bypassed workloads no material regression or selector execution; scope-appropriate quality/needle checks. Inconclusive statistical bounds remain inconclusive. Kill the candidate if actual-state repair cannot fit latency/acceptance budget. Never tune on the final holdout or sweep more margins after failure.

Resources: 3–6 GPU lock-hours maximum, per-run/arm `flock -w 28800 $HOME/.gpu.lock`, release between arms. Identical server memory fractions .920, empty SERVE_DISPLAY_HZ, >=4 GiB steady free VRAM (startup dips allowed), training >=6 GiB free. No deletion, production tree/launcher/venv edits, commit, push, or additional branches.

## Source inheritance

Created $HOME/tools/sglang-dh7, branch codex/dh7-selector-repair, from current codex/perf-v1 at 7b4d539f9bd896265f498fabccc6466e45ffe818. DH4 remains at 3d7caf070d with its uncommitted DH5 changes untouched. No checkpoint commit or cherry-pick was performed.

Applied a three-way binary patch from common base 14d4c4c985 to the DH4 *working files* for scheduler.py, forward_batch_info.py, qwen4_exp_mtp.py and eagle_worker_v2.py; copied DH runtime/bench files including uncommitted DH5 files. This preserves later production adaptive-controller/graph-warmup changes. DH7 integration changes remain uncommitted.

## Preregistered single recipe

One bias-free r128 BF16 two-factor architecture, initialized from DH2 learned_r128; K2048 and strict margin<0.25 conditional fallback fixed in advance (no margin search). Repair 4,000 AdamW updates, seed 20260908, batch64, peak lr 0.0001 with 200-step warmup and cosine decay, wd 1e-4, gradient norm 1; final update checkpoint only. Loss: argmax CE (weight 1.0) plus top-2048 sparse KL (temperature 2) against captured production BF16 full-head labels, position weights 4 / 2 / 1 for surviving-prefix / first-rejection / later-rejection positions. A single 20-bin monotone interpolation maps raw shortlist probability to production probability of the selected token; current wa controller settings are kept. All coefficients were fixed before any actual-state capture.

Training, calibration, test and final-serving prompts were parent-disjoint splits of the private prompt set, frozen before GPU capture.

## Implementation

The caller API is native `/generate`, `sampling_params.custom_params.dh7_task="code"`; width alone never authorizes selection. Unknown designation, non-greedy sampling, structured-output constraints and batch sizes other than 1 bypass. W16 captures an ordinary full graph plus an additional same-buffer shortlist graph. Host graph-key selection occurs before replay, with no selector/conditional kernels in bypass graphs; W4/W8 capture no shortlist variant. No draft/target KV state is reset when switching eligibility.

The comparable production head-region boundary is recursive hot-vocab GEMV start through `_draft_topk1_finalize_kernel` end, including conversion, winner/hot-map output and publication. The candidate boundary is `selector_down` start through `dh4_publish` end, including conditional fallback.

Integration corrections made during the run (implementation facts, no data statistics):

- Overlap execution leaves `ScheduleBatch.seq_lens_cpu=None`; the untimed capture collector was corrected to read the actual `batch.seq_lens` GPU tensor when the CPU mirror is absent. The failed smoke request produced no training or timed sample.
- Training labels are produced by replaying the *production* `draft_topk1_postprocess` reducer on captured BF16 logits, with exact winner comparison and FP64 probability checks.
- In WA traces, the initial target-fed hot head in each draft-extend graph was initially miscounted as a recursive head by the analysis-only width parser; the parser was corrected to require recursive finalize or shortlist publication in the graph and to exclude full GEMV calls nested inside conditional fallback.
- One WA arm's preflight saw leftover compute PIDs from the preceding arm and exited before launching a server; unmeasured arms were retried with a short settle interval, unchanged otherwise.

## Correctness checks

- CPU eligibility matrix and preservation of current production adaptive implementations: PASS.
- GPU numerical/conditional preflight (K1024/K2048) and the repaired-asset checks (candidate logits versus production BF16 GEMV / teacher, included/fallback winner recovery, hot-token mapping/publication, frozen/calibrated interpolation, strict finite-threshold conditional execution, ties, 14-position graph replay, production reducer labeling) were run on captured states from the private prompt set; their results are not published.
- Candidate arms were checked for selector execution on bypassed workloads (prose-en, prose-ja, agent) and for shortlist-counter increments on unknown-designation code requests; these checks ran on the private prompts and their results are not published. W4/W8 graphs are captured without a selector variant.
- Scoped multilingual code-output checks and DH7-specific code needles were run (their inputs are not public; results not published). The repository's public prose needle test (`bench/prof/needle_test.py`, context `bench/quality/tasks/needle-context.txt`) PASSed in every arm.
- Production / model / DH4 preservation audit PASS; owned process-group cleanup PASS; no commit / push / production edit.

## Closure

**No production change.** The actual-state repair, numerical checks, restricted eligibility, fixed-W16 ABBA experiment and unchanged current-WA ABBA integration were carried out on the private prompt set; no result or conclusion from them is published. The DH7 candidate is not part of the production configuration. No additional checkpoint, calibration recipe or margin sweep was run on the final holdout.

Resource summary: memory fraction 0.920 and empty SERVE_DISPLAY_HZ in every server arm; steady free VRAM stayed above the 4 GiB floor. GPU work ended with the planned experiment; the allocation is a ceiling, not a GPU-time target.

Limitations: few independent final prompts per domain and only two paired blocks; long shared-lock gaps and ordinary shared autotune caches; head-region timing on one separately profiled prompt per arm; short capture contexts with long contexts screened only by needles. Offline consistency proxies are not population serving guarantees.

**Exact production change: none.** Keep the existing production checkout, `serve-fast.sh`, `.venv` and current `wa` unchanged. The DH7 working tree remains a reviewable, unmerged experimental candidate.
