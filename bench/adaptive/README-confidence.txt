w16_conf.json -- confidence/throughput adaptive config for Qwen3.8-Flash-Next
(NEXTN, topk=1).  Requires SGLANG_ADAPTIVE_POLICY=confidence and the C1 code
(branch opus/spec-confidence in ~/tools/sglang-c1).  Without that env var the
server runs the shipped EMA controller and ignores the extra keys here.

  cd ~/tools/sglang-rtxpro6000 && \
    PYTHONPATH=~/tools/sglang-c1/python SGLANG_ADAPTIVE_POLICY=confidence \
    ./serve-fast.sh wa --speculative-adaptive-config \
      ~/tools/flash-next-bench/adaptive/w16_conf.json

What it decides on
------------------
argmax over the candidate step counts of

    sum_b w_b (1 + E[accepted | bucket b, S]) / step_time(S)

  w_b        occupancy EMA of the draft's position-0 top-1 probability bucket
             (weight_alpha), i.e. what fraction of chains currently start from
             each confidence level
  E[.|b,S]   for S <= the live chain: mean of min(accepted, S) -- exact, because
             a greedy chain is accepted as a prefix
             for S >  the live chain: censored, so extrapolated from the
             boundary hazard r = P(a>=live)/P(a>=live-1) with a flat tail

The shipped w16_3_15.json instead compares ONE EMA of accepted drafts against a
threshold.  That is why it needs the "re-seed on step-down only" asymmetry (its
statistic changes scale with every switch, this one does not) and why it dips
on code-edit (its accept distribution has no mass near its own mean).

Keys
----
  buckets            position-0 confidence bucket edges, packed against 1.0
                     because that is where the mass is
  rate_alpha         EMA on every acceptance statistic
  weight_alpha       EMA on bucket occupancy (slower: it is a mixture weight)
  update_interval    batches between decisions
  switch_grace_batches  batches after a switch during which no decision is taken
  grace_backoff      multiplies the grace on each REVERSAL of the last switch;
                     stops ping-pong where the two candidates genuinely tie
  switch_margin      relative margin a challenger must clear; also covers the
                     upward estimate's known ~10% optimism
  tail_bias          discount on the hazard extrapolation (measured +8..17%)
  confidence_weight  0 pools all steps into one estimate (a better statistic,
                     no confidence -- useful as an ablation); 1 uses the buckets
  min_bucket_samples shrinkage: a bucket is trusted in proportion to its count

Slot "2" pins bs>=2 to steps=15, same as the shipped config, so the steps=3
state only needs bs=1 graphs.

Measured (2026-09-06 16:47-17:28, fnbench 4 workloads x 2 repeats greedy,
needle PASS, all four runs inside one hour):

  workload     fixed W4     fixed W16    EMA (shipped)   this config
  code-edit    3.88 / 330   10.48 / 493   8.92 / 480    11.80 / 586
  prose-en     2.55 / 227    3.10 / 156   2.56 / 240     2.50 / 236
  prose-ja     2.37 / 214    2.56 / 133   2.25 / 216     2.30 / 219
  agent-loop   3.21 / 284    5.70 / 281   3.24 / 287     3.26 / 290

  vs the BETTER fixed profile:  +18.9%  +4.0%  +2.3%  +2.1%
  vs the shipped EMA config:    +22.1%  -1.7%  +1.4%  +1.0%
  switches per benchmark:  4 (the shipped EMA config: 13)
  free VRAM at WA_MEM_FRACTION=0.925: 4.59 GB (EMA arm: 4.56 GB)

Tuned by joint selection over recorded per-step traces (adaptive/c1/sim.py,
3000-batch in-order replay with the exact min(a,S) counterfactual) under two
criteria at once: throughput, and convergence to the better fixed profile from
BOTH starting states on all four workloads.  Regression guard:
adaptive/c1/test_policy.py (CPU only).  See specs/C1_LOG.md for the
derivation, the option-2 investigation, and the caveats -- in particular that
code-edit acceptance varies 8.3-11.6 run to run at fixed W16.
