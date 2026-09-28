from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cost import hybrid_tps, mtp_chain_tps, ngram_tps


def _f(value: object, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _pct(value: object, digits: int = 2) -> str:
    """A rate as a percentage; "not measured" when there were no positions."""
    return "not measured" if value is None else f"{100 * float(value):.{digits}f}%"


def _primary_alpha(alphas: list[float]) -> float:
    return min(alphas, key=lambda value: abs(value - 0.75))


def _corpora(data: dict[str, Any]) -> list[str]:
    """Every corpus the run recorded, including one with no generated positions (and so no statistics rows)."""
    names = {str(row["corpus"]) for row in data["statistics"]}
    names |= {str(k) for k in data.get("parser_accounting", {})}
    names |= {str(k) for k in data.get("token_counts", {}).get("by_corpus", {})}
    names |= {str(k) for k in data.get("truncations", {})}
    return sorted(names)


def _strategy_rows(
    data: dict[str, Any], scope: str, corpus: str, lookup_mode: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in data["statistics"]
        if row["scope"] == scope
        and row["corpus"] == corpus
        and row.get("lookup_mode", "session-local") == lookup_mode
    ]


def _lookup_modes(data: dict[str, Any], corpus: str) -> list[str]:
    return sorted(
        {
            str(row.get("lookup_mode", "session-local"))
            for row in data["statistics"]
            if row["corpus"] == corpus
        },
        reverse=True,
    )


def _throughput_rows(
    data: dict[str, Any], corpus: str, lookup_mode: str, alpha: float, delta: float
) -> list[tuple[str, str, float]]:
    config = data["config"]
    c1 = float(config["c1_ms"])
    draft_ms = float(config["draft_ms"])
    fallback = int(config["hybrid_mtp_steps"])
    stats = _strategy_rows(data, "workload", corpus, lookup_mode)
    groups = sorted({str(row["group"]) for row in stats})
    output: list[tuple[str, str, float]] = []
    for group in groups:
        output.append(
            ("MTP-chain(steps=3)", group, mtp_chain_tps(3, alpha, c1, delta, draft_ms))
        )
        output.append(
            ("MTP-chain(steps=7)", group, mtp_chain_tps(7, alpha, c1, delta, draft_ms))
        )
        for row in stats:
            if row["group"] != group or row["draft_steps"] not in {7, 15}:
                continue
            if row["hit_rate"] is None:  # no positions in this group: not measured
                continue
            steps = int(row["draft_steps"])
            width = steps + 1
            dedicated = ngram_tps(
                hit_rate=float(row["hit_rate"]),
                expected_confirmed=float(row["expected_confirmed_per_verification"]),
                steps=steps,
                c1_ms=c1,
                delta_ms=delta,
            )
            hybrid = hybrid_tps(
                hit_rate=float(row["hit_rate"]),
                hit_confirmed_sum_per_position=float(
                    row["hit_confirmed_sum_per_position"]
                ),
                ngram_steps=steps,
                mtp_steps=fallback,
                alpha=alpha,
                c1_ms=c1,
                delta_ms=delta,
                draft_ms=draft_ms,
            )
            output.append(
                (f"n-gram-only(width={width}) {row['strategy']}", group, dedicated)
            )
            output.append(
                (f"n-gram+MTP(width={width}) {row['strategy']}", group, hybrid)
            )
    return output


def _break_even(
    data: dict[str, Any], corpus: str, lookup_mode: str, alpha: float
) -> str:
    config = data["config"]
    c1 = float(config["c1_ms"])
    delta = float(config["delta_ms"])
    draft_ms = float(config["draft_ms"])
    candidates = [
        row
        for row in _strategy_rows(data, "overall", corpus, lookup_mode)
        if row["draft_steps"] in {7, 15} and row["hit_rate"] is not None
    ]
    if not candidates:
        return "No width-8/16 measurement is available for break-even analysis."

    def predicted(row: dict[str, Any]) -> float:
        return ngram_tps(
            hit_rate=float(row["hit_rate"]),
            expected_confirmed=float(row["expected_confirmed_per_verification"]),
            steps=int(row["draft_steps"]),
            c1_ms=c1,
            delta_ms=delta,
        )

    best = max(candidates, key=predicted)
    steps = int(best["draft_steps"])
    hit_rate = float(best["hit_rate"])
    match_mean = best["hit_match_mean"]  # None when no hit: not measured
    expected = float(best["expected_confirmed_per_verification"])
    mtp_tps = mtp_chain_tps(3, alpha, c1, delta, draft_ms)
    rate_per_ms = mtp_tps / 1000.0
    numerator = rate_per_ms * c1 - 1.0
    match_threshold = (
        rate_per_ms * steps * delta + numerator / hit_rate
        if hit_rate > 0
        else float("inf")
    )
    if match_mean is None:  # no hit at all: the match mean was not measured
        hit_text = "not computable (no hit, match mean not measured)"
    else:
        denominator = match_mean - rate_per_ms * steps * delta
        hit_threshold = numerator / denominator if denominator > 0 else float("inf")
        hit_text = f"{100 * hit_threshold:.1f}%" if 0 <= hit_threshold <= 1 else "not reachable at that match mean"
    match_text = f"{match_threshold:.2f}" if match_threshold >= 0 else "0.00"
    return (
        f"With α={alpha:.2f}, MTP-3 projects {mtp_tps:.1f} t/s. The best observed provider row is "
        f"`{best['strategy']}` at width {steps + 1}: hit rate {100 * hit_rate:.2f}%, "
        f"hit-conditional match {_f(match_mean, 2) if match_mean is not None else 'not measured'}, E[confirmed]={expected:.3f}, and "
        f"{predicted(best):.1f} t/s. Holding its match mean fixed, n-gram breaks even at hit rate "
        f"{hit_text}; holding its hit rate fixed, it needs mean consecutive match {match_text}. "
        "The threshold includes wide-verification cost, so hit rate alone is insufficient."
    )


def _append_acceptance(
    lines: list[str],
    data: dict[str, Any],
    corpus: str,
    lookup_mode: str,
    alphas: list[float],
) -> None:
    config = data["config"]
    lines.extend(
        [
            "### Acceptance by workload",
            "",
            f"Hybrid values use MTP-{config['hybrid_mtp_steps']} only on no-proposal positions.",
            "",
            "| Strategy | Steps | Workload | Positions | Hit rate | Hit match mean | Median | P90 | E[confirmed] | "
            + " | ".join(f"Hybrid α={value:.2f}" for value in alphas)
            + " |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|"
            + "---:|" * len(alphas),
        ]
    )
    for row in _strategy_rows(data, "workload", corpus, lookup_mode):
        hybrid = row["hybrid_expected_confirmed"]
        lines.append(
            f"| {row['strategy']} | {row['draft_steps']} | {row['group']} | "
            f"{row['positions']:,} | {_pct(row['hit_rate'])} | "
            f"{_f(row['hit_match_mean'])} | {_f(row['hit_match_median'], 0)} | "
            f"{_f(row['hit_match_p90'], 0)} | "
            f"{_f(row['expected_confirmed_per_verification'])} | "
            + " | ".join(_f(hybrid[f"{value:.2f}"]) for value in alphas)
            + " |"
        )


def _append_throughput(
    lines: list[str],
    data: dict[str, Any],
    corpus: str,
    lookup_mode: str,
    alpha: float,
    delta: float,
) -> None:
    config = data["config"]
    lines.extend(
        [
            "",
            "### Predicted throughput",
            "",
            f"Uses α={alpha:.2f}, c1={config['c1_ms']} ms, δ={delta} ms/row, and d={config['draft_ms']} ms/MTP step.",
            "",
            "| Configuration | Workload | Predicted tokens/s |",
            "|---|---|---:|",
        ]
    )
    for name, group, tps in _throughput_rows(
        data, corpus, lookup_mode, alpha, delta
    ):
        lines.append(f"| {name} | {group} | {tps:.1f} |")


def _append_sensitivity(
    lines: list[str],
    data: dict[str, Any],
    corpus: str,
    lookup_mode: str,
    alphas: list[float],
) -> None:
    config = data["config"]
    c1 = float(config["c1_ms"])
    draft_ms = float(config["draft_ms"])
    fallback = int(config["hybrid_mtp_steps"])
    deltas = config.get("sensitivity_deltas") or [0.25, 0.5, 0.75, 1.0]
    length_15 = [
        row
        for row in _strategy_rows(data, "overall", corpus, lookup_mode)
        if row["draft_steps"] == 15 and row["hit_rate"] is not None
    ]  # width 16 not requested (e.g. --lengths 3 7) or no positions: shown as "not measured", never 0.0
    lines.extend(
        [
            "",
            "### Sensitivity (overall)",
            "",
            "| α | δ ms/row | MTP-3 t/s | MTP-7 t/s | Best n-gram width-16 t/s | Best hybrid width-16 t/s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alpha in alphas:
        for delta in deltas:
            dedicated = [
                ngram_tps(
                    hit_rate=float(row["hit_rate"]),
                    expected_confirmed=float(row["expected_confirmed_per_verification"]),
                    steps=15,
                    c1_ms=c1,
                    delta_ms=float(delta),
                )
                for row in length_15
            ]
            hybrid = [
                hybrid_tps(
                    hit_rate=float(row["hit_rate"]),
                    hit_confirmed_sum_per_position=float(
                        row["hit_confirmed_sum_per_position"]
                    ),
                    ngram_steps=15,
                    mtp_steps=fallback,
                    alpha=alpha,
                    c1_ms=c1,
                    delta_ms=float(delta),
                    draft_ms=draft_ms,
                )
                for row in length_15
            ]
            lines.append(
                f"| {alpha:.2f} | {float(delta):.2f} | "
                f"{mtp_chain_tps(3, alpha, c1, float(delta), draft_ms):.1f} | "
                f"{mtp_chain_tps(7, alpha, c1, float(delta), draft_ms):.1f} | "
                f"{_best(dedicated)} | {_best(hybrid)} |"
            )


def _best(values: list[float]) -> str:
    return f"{max(values):.1f}" if values else "not measured"


def _rec(mapping: dict[str, Any], key: str, fmt: str = "") -> str:
    """A recorded count, or "not recorded" when the key is absent (never a silent 0)."""
    value = mapping.get(key)
    return "not recorded" if value is None else format(value, fmt)


def _append_models(
    lines: list[str], data: dict[str, Any], corpus: str, lookup_mode: str
) -> None:
    config = data["config"]
    max_length = max(int(value) for value in config["draft_lengths"])
    lines.extend(
        [
            "",
            f"### Model-tag breakdown (steps={max_length})",
            "",
            "| Strategy | Model tag | Positions | Hit rate | Hit match mean | P90 | E[confirmed] |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    shown = 0
    for row in _strategy_rows(data, "model", corpus, lookup_mode):
        if row["draft_steps"] != max_length:
            continue
        shown += 1
        safe_tag = str(row["group"]).replace("|", "\\|")
        lines.append(
            f"| {row['strategy']} | {safe_tag} | {row['positions']:,} | "
            f"{_pct(row['hit_rate'])} | {_f(row['hit_match_mean'])} | "
            f"{_f(row['hit_match_p90'], 0)} | "
            f"{_f(row['expected_confirmed_per_verification'])} |"
        )
    if not shown:
        lines.append("| not measured | | | | | | |")


def render_report(data: dict[str, Any]) -> str:
    config = data["config"]
    alphas = [float(value) for value in config["alphas"]]
    alpha = _primary_alpha(alphas)
    delta = float(config["delta_ms"])
    lines = [
        "# n-gram acceptance simulation report",
        "",
        "`E[confirmed] = min(consecutive match, steps) + 1` over all generated decode positions.",
        "",
        "| Corpus | Conversations | Total tokens | Generated positions | Truncated |",
        "|---|---:|---:|---:|---:|",
    ]
    accounting = data.get("parser_accounting", {})
    by_corpus = data["token_counts"].get("by_corpus", {})
    truncations = data.get("truncations", {})
    for corpus in _corpora(data):
        summary = accounting.get(corpus, {})
        counts = by_corpus.get(corpus, {})
        lines.append(
            f"| {corpus} | {_rec(summary, 'conversations_loaded')} | "
            f"{_rec(counts, 'all', ',')} | {_rec(counts, 'assistant', ',')} | "
            f"{_rec(truncations, corpus)} |"
        )
    lines.extend(
        [
            "",
            f"Searchable window={config['search_window_tokens']:,} tokens; session cap={config['max_tokens_per_conv']:,} tokens. Available lookup modes are listed per corpus below.",
        ]
    )

    for corpus in _corpora(data):
        summary = accounting.get(corpus, {})
        lines.extend(["", f"## Corpus: {corpus}", ""])
        if not _lookup_modes(data, corpus):
            lines.append(
                "No generated positions: acceptance, throughput, break-even, sensitivity and model-tag "
                "results are not measured for this corpus."
            )
        for lookup_mode in _lookup_modes(data, corpus):
            lines.extend([f"### Lookup mode: {lookup_mode}", ""])
            _append_acceptance(lines, data, corpus, lookup_mode, alphas)
            _append_throughput(lines, data, corpus, lookup_mode, alpha, delta)
            lines.extend(
                [
                    "",
                    "### Break-even",
                    "",
                    _break_even(data, corpus, lookup_mode, alpha),
                ]
            )
            _append_sensitivity(lines, data, corpus, lookup_mode, alphas)
            _append_models(lines, data, corpus, lookup_mode)
        warning_counts = summary.get("warning_counts", {})
        lines.extend(
            [
                "",
                "### Parser accounting",
                "",
                f"- Files: seen={_rec(summary, 'files_seen')}, selected={_rec(summary, 'files_selected')}, oversize={_rec(summary, 'skipped_oversize')}, limited-out={_rec(summary, 'skipped_by_limit')}",
                f"- Skips: corrupt={_rec(summary, 'skipped_corrupt')}, empty={_rec(summary, 'skipped_empty')}, messages={_rec(summary, 'skipped_messages')}",
                "- Counters: "
                + (
                    ", ".join(f"`{key}`={value}" for key, value in warning_counts.items())
                    or "none"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is token replay, not a live serving benchmark. It omits kernel-shape effects, batching, KV-cache traffic, scheduler overhead, chat-template boundaries, and index-maintenance cost. Replay can be optimistic, especially at temperature > 0. Codex JSONL canonical response items are used while duplicate event messages, encrypted reasoning, and embedded binary/base64 blobs are excluded. Tool-call names are lookup context; only their arguments are decode targets. The MTP fallback is an independent per-step geometric α model. Predicted tokens/s ranks hypotheses but is not a production promise.",
            "",
        ]
    )
    return "\n".join(lines)


def load_results(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser()
    json_path = source / "raw_stats.json" if source.is_dir() else source
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), json_path


def regenerate_report(path: str | Path) -> Path:
    data, json_path = load_results(path)
    destination = json_path.parent / "report.md"
    destination.write_text(render_report(data), encoding="utf-8")
    return destination
