#!/usr/bin/env python3
"""Regenerate results/TABLES.md from the stripped result files in this directory.

Every throughput / acceptance number quoted in README.md, README.ja.md and docs/ that
comes from a benchmark run is produced here.  No GPU, no network, stdlib only.

    python3 results/make_tables.py            # writes results/TABLES.md
    python3 results/make_tables.py --check    # exit 1 if TABLES.md is stale
    python3 results/make_tables.py --out F    # write the tables to F instead

Definitions
  t/s          fnbench client decode rate = (completion_tokens - 1) / (last token chunk time
               - first token chunk time), reasoning ("thinking") tokens included.
  acceptance   tokens emitted per target verify step (bonus token included)
               = completion tokens / sglang:spec_verify_calls_total (request-bracketed delta).
  mean         arithmetic mean over the measured requests (warm-up excluded by fnbench).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JST = dt.timezone(dt.timedelta(hours=9))
ORDER = ["code-edit", "prose-en", "prose-ja", "agent-loop", "long-ctx"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def acceptance(r: dict) -> float | None:
    s = r.get("server", {})
    if isinstance(s.get("acceptance"), dict) and s["acceptance"].get("tokens_per_verify"):
        return float(s["acceptance"]["tokens_per_verify"])
    d = s.get("metrics_delta", {})
    ver = d.get("sglang:spec_verify_calls_total")
    gen = d.get("sglang:generation_tokens_total") or r["client"]["usage"]["completion_tokens"]
    if ver:
        return float(gen) / float(ver)
    return None


def by_workload(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["workload"], []).append(r)
    return out


def jst(ts: str) -> str:
    return dt.datetime.fromisoformat(ts).astimezone(JST).strftime("%Y-%m-%d %H:%M")


def cell(rs: list[dict], with_acc: bool = True, with_range: bool = False) -> str:
    if not rs:
        return "–"
    t = [r["client"]["decode_tps"] for r in rs]
    s = f"{st.mean(t):.0f}"
    if with_range and len(t) > 1:
        s += f" ({min(t):.0f}–{max(t):.0f})"
    if with_acc:
        a = [x for x in (acceptance(r) for r in rs) if x]
        if a:
            s += f" / {st.mean(a):.2f}"
    return s


def workload_table(files: list[tuple[str, str]], workloads: list[str], with_range=False) -> list[str]:
    head = "| run | profile / note | sampling | " + " | ".join(workloads) + " |"
    lines = [head, "|" + "---|" * (3 + len(workloads))]
    for fname, note in files:
        rows = load(HERE / fname)
        bw = by_workload(rows)
        n = max(len(v) for v in bw.values())
        samp = rows[0]["sampling_mode"]
        cells = [cell(bw.get(w, []), with_range=with_range) for w in workloads]
        lines.append(f"| `{fname}` ({jst(rows[0]['timestamp'])} JST, n={n}) | {note} | {samp} | " + " | ".join(cells) + " |")
    return lines


def conditions(fname: str) -> list[str]:
    rows = load(HERE / fname)
    out = []
    for w, rs in sorted(by_workload(rows).items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99):
        u = rs[0]["client"]["usage"]
        rt = [r["client"]["usage"].get("reasoning_tokens") or 0 for r in rs]
        ct = [r["client"]["usage"]["completion_tokens"] for r in rs]
        out.append(f"| {w} | {u['prompt_tokens']} | {rs[0]['max_tokens']} | {st.mean(ct):.0f} | {st.mean(rt):.0f} |")
    return out


def bn1_table() -> list[str]:
    lines = ["| profile | domain | requests | mean t/s | geo-mean t/s | min–max t/s | mean acceptance | mean completion tokens |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for prof in ("w4", "wa"):
        rows = []
        for arm in ("A1", "A2", "A3"):
            rows += load(HERE / "bn1-study-v2" / f"{prof}-{arm}.jsonl")
        bw = by_workload(rows)
        for w in ORDER:
            if w not in bw:
                continue
            rs = bw[w]
            t = [r["client"]["decode_tps"] for r in rs]
            a = [acceptance(r) for r in rs]
            gm = math.exp(st.mean(math.log(x) for x in t))
            ct = st.mean(r["client"]["usage"]["completion_tokens"] for r in rs)
            lines.append(f"| {prof} | {w} | {len(rs)} | {st.mean(t):.0f} | {gm:.0f} | {min(t):.0f}–{max(t):.0f} | {st.mean(a):.2f} | {ct:.0f} |")
    return lines


def wa5_table() -> list[str]:
    arms = sorted((HERE / "wa5").glob("*.jsonl"))
    ws = ["code-edit", "prose-en", "prose-ja", "agent-loop"]
    lines = ["| arm | policy | " + " | ".join(ws) + " |", "|---|---|" + "---|" * len(ws)]
    for p in arms:
        rows = load(p)
        pol = "A = [3,15] (previous production)" if p.stem.endswith(("A1", "A2")) else "C = [3,7,15] config c (adopted)"
        bw = by_workload(rows)
        lines.append(f"| `{p.name}` ({jst(rows[0]['timestamp'])} JST) | {pol} | " + " | ".join(cell(bw.get(w, [])) for w in ws) + " |")
    # pooled C vs pooled A
    def pooled(tag):
        out = {}
        for p in arms:
            if p.stem[-2] == tag:
                for w, rs in by_workload(load(p)).items():
                    out.setdefault(w, []).append(st.mean(r["client"]["decode_tps"] for r in rs))
        return {w: st.mean(v) for w, v in out.items()}
    a, c = pooled("A"), pooled("C")
    lines.append("| **C vs A (equal-weight arm means)** | | " + " | ".join(f"{100*(c[w]/a[w]-1):+.1f}%" for w in ws) + " |")
    return lines


def stack8_table(mode: str) -> list[str]:
    ws = ["code-edit", "prose-en", "prose-ja", "agent-loop"]
    order = ["A1", "B1", "B2", "A2", "B3", "A3", "A4", "B4"]
    lines = ["| arm (start order) | arm | " + " | ".join(ws) + " |", "|---|---|" + "---|" * len(ws)]
    means: dict[str, dict[str, list[float]]] = {"A": {}, "B": {}}
    for i, arm in enumerate(order, 1):
        f = f"runs-1002/stack8/{arm}-{mode}.jsonl"
        rows = load(HERE / f)
        bw = by_workload(rows)
        for w, rs in bw.items():
            means[arm[0]].setdefault(w, []).append(st.mean(r["client"]["decode_tps"] for r in rs))
        what = "A: all off" if arm[0] == "A" else "B: stack on"
        lines.append(f"| {i}. `{f}` ({jst(rows[0]['timestamp'])} JST) | {what} | " + " | ".join(cell(bw.get(w, [])) for w in ws) + " |")
    a = {w: st.mean(v) for w, v in means["A"].items()}
    b = {w: st.mean(v) for w, v in means["B"].items()}
    lines.append("| **A mean -> B mean (equal-weight arm means, t/s)** | | " + " | ".join(f"{a[w]:.0f} -> {b[w]:.0f} ({100*(b[w]/a[w]-1):+.1f}%)" for w in ws) + " |")
    return lines


def ship_table() -> list[str]:
    ws = ["code-edit", "prose-en", "prose-ja", "agent-loop"]
    lines = ["| run | profile | sampling | " + " | ".join(ws) + " |", "|---|---|---|" + "---|" * len(ws)]
    for prof in ["wa", "w4", "w16"]:
        for mode in ["lmstudio", "greedy"]:
            f = f"runs-1002/ship/{prof}-{mode}.jsonl"
            rows = load(HERE / f)
            bw = by_workload(rows)
            lines.append(f"| `{f}` ({jst(rows[0]['timestamp'])} JST) | `{prof}` | {mode} | " + " | ".join(cell(bw.get(w, [])) for w in ws) + " |")
    return lines


def build() -> str:
    L: list[str] = []
    L += ["# Result tables (generated)", "",
          "Generated by `results/make_tables.py` from the stripped result files in `results/`. Do not edit by hand.",
          "Cells are `mean t/s / mean acceptance` unless stated otherwise. t/s is the client-side decode rate and",
          "includes reasoning (\"thinking\") tokens; acceptance is tokens per verify step including the bonus token.",
          "All runs: one RTX PRO 6000 Blackwell Max-Q, batch size 1 (one request at a time), FP8 KV cache.", ""]

    L += ["## 1. Headline: all three production profiles, same session (2026-09-06 23:3x JST)", "",
          "Single prompt per workload (`bench/workloads/*.txt`), 2 measured repeats after 1 warm-up, greedy.",
          "`baseline-nextn` is the unmodified base fork (jpezzulli/sglang-rtxpro6000 @ 16e5682aad, NEXTN steps 3)",
          "measured with the same harness on 2026-09-02, but with each workload's recommended sampling",
          "(temperature 0.2–0.7), so it is indicative rather than a strictly paired baseline.", ""]
    L += workload_table([
        ("runs/baseline-nextn.jsonl", "starting point: unmodified base fork, W4 (NEXTN steps 3)"),
        ("runs/p2m2332-w4.jsonl", "`serve-fast.sh w4` (draft width 4)"),
        ("runs/p2m2332-w16.jsonl", "`serve-fast.sh w16` (draft width 16)"),
        ("runs/p2m2332-wa.jsonl", "`serve-fast.sh wa` (adaptive 3/15, 2026-09-06 policy)"),
    ], ["code-edit", "prose-en", "prose-ja", "agent-loop", "long-ctx"], with_range=True)
    L += ["", "Prompt / output sizes of the headline runs (from `runs/p2m2332-w4.jsonl`; the same prompts are used by every run in this section):", "",
          "| workload | prompt tokens (templated) | max_tokens | mean completion tokens | mean reasoning tokens |",
          "|---|---:|---:|---:|---:|"] + conditions("runs/p2m2332-w4.jsonl")
    L += ["", "Note: completion and reasoning tokens come from two different server counters. `completion_tokens`",
          "is the generated count capped at `max_tokens`; the reasoning count is reported separately and is not",
          "clipped to that cap, so it can exceed it slightly (code-edit: 2402 reasoning vs 2400 completion). This is",
          "not an arithmetic error; t/s uses `completion_tokens`."]

    L += ["", "## 2. Final production state, 8 held-out prompts x 3 server restarts (2026-09-08 JST)", "",
          "BN1 noise study: 32 synthetic held-out prompts (`bench/workloads/sets/*-v1`), one request each, greedy,",
          "new server process per restart (A1–A3), memory fraction 0.920, `MAX_TOTAL_TOKENS=131072`.",
          "Longer outputs than section 1 (caps: code-edit 12288, prose 6144, agent-loop 8192 tokens).",
          "`wa` here is the adopted three-width policy [3,7,15]. Source: `bn1-study-v2/*.jsonl`.", ""]
    L += bn1_table()

    L += ["", "## 3. Fixed draft-width sweep on the production stack (2026-09-07 13:37–14:04 JST)", "",
          "WS1: same session, `w4` stack with `--speculative-num-steps N-1 --speculative-num-draft-tokens N`",
          "(W16 = the `w16` profile). n=2 per cell, greedy. `w4b` repeats W4 at the end as a drift check.", ""]
    L += workload_table([
        ("runs/ws11337-w4-w4.jsonl", "W4"), ("runs/ws11337-w6-w4.jsonl", "W6"),
        ("runs/ws11337-w8-w4.jsonl", "W8"), ("runs/ws11337-w12-w4.jsonl", "W12"),
        ("runs/ws11337-w16-w16.jsonl", "W16"), ("runs/ws11337-w4b-w4.jsonl", "W4 (repeat)"),
    ], ["code-edit", "prose-en", "prose-ja", "agent-loop"], with_range=True)

    L += ["", "## 4. Three-width adaptive policy A/B (WA5, quiet night 2026-09-08 01:58–02:10 JST) and post-adoption smoke", "",
          "ABAB, n=3 greedy per workload per arm. Source: `wa5/*.jsonl`.", ""]
    L += wa5_table()
    L += ["", "Post-adoption smoke of the final `wa` profile (mtpft5 head, three-width policy):", ""]
    L += workload_table([("runs/wa5prod0221-wa.jsonl", "`serve-fast.sh wa`, final")],
                        ["code-edit", "prose-en", "agent-loop"], with_range=True)

    L += ["", "## 5. Progression (single-prompt workloads, n=2–4 per cell)", "",
          "Selected milestone runs; each used whatever `serve-fast.sh` / flag set was current at the time (see",
          "`docs/timeline.md`). Early runs used recommended sampling, later ones greedy. Numbers from different",
          "sessions carry roughly ±10 % desktop-compositor noise (see `docs/measurement.md`).", ""]
    L += workload_table([
        ("runs/baseline-nextn.jsonl", "unmodified base fork"),
        ("runs/hot32k.jsonl", "+ reduced draft vocab (32k)"),
        ("runs/w8a16-hot32k.jsonl", "+ Triton W8A16 GEMV for FP8 dense weights"),
        ("runs/nextn15-v3.jsonl", "first W16 after QSA ring fix"),
        ("runs/w4-greedy.jsonl", "W4 greedy reference (v3)"),
        ("runs/final2-w4.jsonl", "W4 after glue-fusion round 1"),
        ("runs/final2-w16.jsonl", "W16 after glue-fusion round 1"),
        ("runs/final4-w4.jsonl", "W4 after round 2 (HC FP8, GDN strided, router GEMV ...)"),
        ("runs/mtpft3-W4_MTPFT3.jsonl", "W4 + fine-tuned MTP head v3 + blended 49k map"),
        ("runs/mtpft3-W16_MTPFT3.jsonl", "W16 + fine-tuned MTP head v3"),
        ("runs/final0906b-w4.jsonl", "W4 after round 3 (PDL, FlashInfer G1/G2, HC boundary, H1/H2)"),
        ("runs/final0906b-w16.jsonl", "W16 after round 3"),
        ("runs/final0906b-wa.jsonl", "adaptive wa (two-state)"),
        ("runs/p1m2124-w4.jsonl", "W4 + P1 singleton pruning"),
        ("runs/p2m2332-w4.jsonl", "W4 + P2 (pruning inside FlashInfer prologue)"),
        ("runs/p2m2332-w16.jsonl", "W16 + P2"),
        ("runs/p2m2332-wa.jsonl", "wa + C1 confidence policy + P2"),
        ("runs/wa5prod0221-wa.jsonl", "wa final (mtpft5 head, three widths)"),
    ], ["code-edit", "prose-en", "prose-ja", "agent-loop", "long-ctx"])
    L += ["## 6. 2026-10-01 stack, 8-start ABBA (2026-10-01 JST)", "",
          "A = the 2026-09-08 production code with every 2026-10 switch off; B = RT1 + SV1 + SV2 + FG1 + DG1 + the RQ2",
          "u2h FlashInfer build (min_p filtering off in both arms; see `docs/optimizations.md`). Profile `wa`, 16 prompts",
          "(the first 4 per workload of `bench/workloads/sets/*-v1`), one request each per arm and mode, 8 server starts in the order",
          "A1 B1 B2 A2 B3 A3 A4 B4. The effect sizes with confidence intervals come from `bench/bench/stats/ancova_ab.py`",
          "(`runs-1002/stack8/ancova8.log`); the t/s cells below are plain means and move by several percent between",
          "server starts of the same arm. Cells: mean t/s / mean acceptance.", "",
          "LM Studio sampling (temperature 0.8, top_p 0.95, top_k 40, min_p 0.05):", ""]
    L += stack8_table("lmstudio")
    L += ["", "Greedy:", ""]
    L += stack8_table("greedy")
    L += ["", "## 7. Shipped build, single-prompt smokes per profile (2026-10-02 JST)", "",
          "One prompt per workload, one repeat, after the 2026-10-02 build was installed. Sanity check only, not a",
          "measurement; the shipped server ran mem fraction 0.925, chunked prefill 4096 and one running request.", ""]
    L += ship_table()
    L += ["", "## 8. Shipped build, 8 held-out prompts, private vs public components (2026-10-06 JST)", "",
          "The 2026-10-02 build (`launch/as-measured/serve-fast.sh`, worktree 7118260ce3) as LM Studio launches it: mem fraction",
          "0.925, `MAMBA_SLOTS=16`, context 262,144, chunked prefill 4096, one running request; the desktop stayed at 160 Hz (the LM Studio",
          "entry sets `SERVE_DISPLAY_HZ=60`, see `preflight.json`; this run cleared it, as the 2026-10 A/B drivers in `bench/bench/` do). 32 BN1 held-out prompts",
          "(`bench/workloads/sets/*-v1`, 8 per workload, same caps as section 2), thinking on, one request each,",
          "**one server start per run** (no restarts, unlike section 2). `public` = the checkpoint's original MTP head",
          "(`RadixArk/Qwen3.8-Flash-Next-NVFP4`, all 419 files hash-checked against the Hub, `verify-base.json`) +",
          "`tokenmaps/public/public_49152.pt`; everything else is identical (`effective-*.txt`). `private` = fine-tuned",
          "MTP head v5 + private 49k map, as in sections 1–7. Cells: mean t/s (min–max) / mean acceptance.", ""]
    L += workload_table([
        ("runs-1002/pub1006/A-prod-wa-greedy.jsonl", "`wa`, private head + map"),
        ("runs-1002/pub1006/B-prod-w4-greedy.jsonl", "`w4`, private head + map"),
        ("runs-1002/pub1006/C-prod-wa-lmstudio.jsonl", "`wa`, private head + map"),
        ("runs-1002/pub1006/D-public-wa-greedy.jsonl", "`wa`, **public components only**"),
        ("runs-1002/pub1006/E-public-w4-greedy.jsonl", "`w4`, **public components only**"),
    ], ["code-edit", "prose-en", "prose-ja", "agent-loop"], with_range=True)
    L.append("")
    return "\n".join(L)


def main() -> int:
    text = build()
    out = HERE / "TABLES.md"
    if "--check" in sys.argv:
        return 0 if out.exists() and out.read_text() == text else 1
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        if i + 1 >= len(sys.argv):
            raise SystemExit("usage: make_tables.py [--check | --out FILE]")
        out = Path(sys.argv[i + 1])
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
