# FP8C CPU survey

Read the final report at `../../specs/FP8_CODING_SURVEY.md`.

```bash
cd $HOME/tools/flash-next-bench
bash bench/fp8coding/run_all.sh
```

The launcher uses `$HOME/tools/sglang-rtxpro6000/.venv/bin/python`,
`nice -n 19`, `OMP_NUM_THREADS=12`, and empty `CUDA_VISIBLE_DEVICES`. It never
launches a server, imports the serving runtime, installs dependencies, or uses a
GPU. It reads the checkpoint, token map, serving sources, and existing traces.
Outputs and the Numba CPU cache remain under this directory. No complete FP8
weight copy or compressed model archive is retained.

Scripts, in execution order:

- `survey.py`: assemble the serving matrices, reconstruct FP8, compare the DH1
  formula, collect byte/field/scale-conditional histograms, run exhaustive
  zlib/zstd block round trips, calculate static Huffman and ANS size models.
- `validate_huffman.py`: independent canonical Huffman encode/decode reference,
  full byte and exponent/raw-sign-mantissa round trips with 256-weight restarts;
  synthetic edge cases and an independent nearest-even E4M3 reference check.
- `audit_conversion.py`: Torch versus NumPy FP32 division over every finite
  nonnegative BF16 value, FP8 NaN check, detailed audit of below-floor rows.
- `time_weights.py`: read existing X2 traces, preserve the published map's
  primary weights, disclose the published-total versus trace-sum discrepancy.
- `report.py`: require all tensor results and hashes, generate all tables and
  the report, and write aggregate and completeness evidence.

`run.sh` runs only the primary survey. Existing complete tensor outputs are
reused. The primary survey refuses changed source hashes. To intentionally
rerun after changing its inputs or implementation, preserve/rename `results/`
and use a fresh results directory; do not silently mix runs.

`results/manifest.json` lists the 200 reconstructed tensors and source hashes.
198 are actual serving FP8 matrices; the two MTP `fc_*` matrices are supplemental
hypothetical FP8 conversions because their current constructors remain BF16.
Per-tensor `.json` files contain counts, entropies, all block rates, and hashes;
`.huffman.json` files contain exhaustive restartable-Huffman verification and
sizes. `aggregate.json`, `time_weights.json`, `conversion_audit.json`, and
`audit.json` provide machine-readable totals and limitations.

ANS figures are normalized cross-entropy estimates, not encoded ANS streams.
Huffman no-restart sizes are exact code-length models. Restartable Huffman
payloads and all zlib/zstd blocks are actually encoded/decoded; container/index
bytes are explicitly counted rather than emitted as a full persistent archive.
CPU byte preservation is verified; CUDA parity and fused-GEMV throughput are
outside this CPU-only task and remain unverified.
