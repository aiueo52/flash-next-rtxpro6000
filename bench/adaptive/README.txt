w16_3_15.json -- two-state adaptive speculative config for Qwen3.8-Flash-Next
(NEXTN, topk=1), tuned 2026-09-05 against the measured per-workload acceptance.

  cd ~/tools/sglang-rtxpro6000 && PYTHONPATH=~/tools/sglang-adaptive/python \
    ./serve-fast.sh w16 --speculative-adaptive \
      --speculative-adaptive-config ~/tools/flash-next-bench/adaptive/w16_3_15.json

Thresholds (accepted drafts per verify, i.e. fnbench acc - 1):
  at steps=15  drop to 3 when EMA <= 5.0   (measured: code 9.87, agent 3.82,
                                            prose-en 1.83, prose-ja 1.55)
  at steps=3   rise to 15 when EMA >  2.5  (measured: code 2.86, agent 2.29,
                                            prose-en 1.51, prose-ja 1.31)
  The EMA is re-seeded at the new neutral value on a step DOWN only; slot "2"
  pins bs>=2 to steps=15 so the steps=3 state captures bs=1 graphs only.

Validated 2026-09-05 21:31-21:40, one window, fnbench greedy repeats=2,
needle PASS on all three:

  workload     W4 acc/tps    W16 acc/tps    adaptive acc/tps   vs better fixed
  code-edit    3.88 / 318    11.55 / 528     8.98 / 463         -12%
  prose-en     2.57 / 220     3.10 / 156     2.54 / 229          +4%
  prose-ja     2.29 / 208     3.31 / 170     2.25 / 205          -1%
  agent-loop   3.19 / 282     4.87 / 235     4.05 / 306          +9%

One server replaces the hand-picked profile: +46% over W4 on code-edit and
+47/+21/+30% over W16 on prose-en/prose-ja/agent-loop. 10 switches per run.

Free VRAM while serving at mem fraction 0.93: 4.12 GB (sglang) / 3.94 GiB
(nvidia-smi), vs 4.87 / 4.68 for plain W16 -- the steps=3 state costs 0.67 GB.
If you need a strict 4 GiB floor, launch with W16_MEM_FRACTION=0.925
(measured 4.61 GB / 4.40 GiB free; ~8% slower on code-edit).

Caveat: code-edit still spends about a quarter of its verifies at steps=3.
See ~/tools/sglang-adaptive/dbg/FINDINGS.txt.
