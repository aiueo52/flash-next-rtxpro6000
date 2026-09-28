from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .codex_parser import load_codex_sessions
from .models import ParseSummary
from .parser import load_conversations
from .report import regenerate_report, render_report
from .simulator import SimulationConfig, simulate
from .tokenization import LocalTokenizer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _nonnegative(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _repository_root() -> Path:
    """The git work tree containing this package (nearest ancestor with .git), else the sim/ directory."""
    package_root = Path(__file__).resolve().parent.parent
    for candidate in (package_root, *package_root.parents):
        if (candidate / ".git").exists():
            return candidate
    return package_root


def _output_dir(value: str, allow_in_repo: bool = False) -> Path:
    """Resolve an output directory. Outputs are derived from private conversations (names, input paths,
    per-conversation statistics), so a directory inside the repository is refused unless allow_in_repo."""
    destination = Path(value).expanduser().resolve()
    root = _repository_root()
    try:
        destination.relative_to(root)
    except ValueError:
        return destination
    if not allow_in_repo:
        raise ValueError(
            f"refusing to write simulator output inside the repository ({root}): {destination}; "
            "outputs are derived from private conversations. Choose a directory outside the repository "
            "or pass --allow-in-repo (and do not commit it)"
        )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ngramsim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="parse conversations and run replay simulation")
    run.add_argument("--conversations", action="append", help="LM Studio file/directory; repeatable")
    run.add_argument("--codex-sessions", action="append", help="Codex JSONL file/directory; repeatable")
    run.add_argument("--tokenizer", required=True)
    run.add_argument(
        "--out",
        required=True,
        help="output directory for raw_stats.json and report.md; must be outside the repository",
    )
    run.add_argument(
        "--allow-in-repo",
        action="store_true",
        help="allow --out inside the repository (the output is private-derived; do not commit it)",
    )
    run.add_argument("--n-min", nargs="+", type=_positive_int, default=[2, 3, 4, 5])
    run.add_argument("--n-max", type=_positive_int, default=12)
    run.add_argument("--ngram-n", nargs="+", type=_positive_int, default=[2, 3, 4])
    run.add_argument("--lengths", "--L", nargs="+", type=_positive_int, default=[3, 7, 15])
    run.add_argument("--alpha", nargs="+", type=_probability, default=[0.65, 0.75, 0.85])
    run.add_argument("--strategy", nargs="+", choices=["both", "suffix-match", "ngram-mod"], default=["both"])
    run.add_argument("--cross-conv", action="store_true")
    run.add_argument(
        "--also-cross-codex",
        action="store_true",
        help="append a second global-window replay for the Codex corpus",
    )
    run.add_argument("--c1", type=_nonnegative, default=7.46, help="one-row target forward cost in ms")
    run.add_argument("--delta", type=_nonnegative, default=0.5, help="cost per additional verification row in ms")
    run.add_argument("--d", type=_nonnegative, default=1.2, help="MTP draft step cost in ms")
    run.add_argument("--hybrid-mtp-steps", type=_positive_int, default=3)
    run.add_argument("--sensitivity-delta", nargs="+", type=_nonnegative, default=[0.25, 0.5, 0.75, 1.0])
    run.add_argument("--codex-limit-files", type=_positive_int, default=200)
    run.add_argument("--max-file-mb", type=_nonnegative, default=20.0)
    run.add_argument("--max-tokens-per-conv", type=_positive_int, default=60000)
    run.add_argument("--search-window-tokens", type=_positive_int, default=60000)

    report = subparsers.add_parser("report", help="regenerate markdown from raw_stats.json")
    report.add_argument("results")
    report.add_argument(
        "--allow-in-repo",
        action="store_true",
        help="allow rewriting report.md inside the repository",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    if not args.conversations and not args.codex_sessions:
        raise ValueError("provide --conversations and/or --codex-sessions")
    destination = _output_dir(args.out, args.allow_in_repo)  # checked before the (long) simulation
    strategies = ["suffix-match", "ngram-mod"] if "both" in args.strategy else list(dict.fromkeys(args.strategy))
    config = SimulationConfig(
        n_mins=sorted(set(args.n_min)),
        n_max=args.n_max,
        ngram_ns=sorted(set(args.ngram_n)),
        draft_lengths=sorted(set(args.lengths)),
        alphas=sorted(set(args.alpha)),
        strategies=strategies,
        cross_conv=args.cross_conv,
        c1_ms=args.c1,
        delta_ms=args.delta,
        draft_ms=args.d,
        hybrid_mtp_steps=args.hybrid_mtp_steps,
        sensitivity_deltas=sorted(set(args.sensitivity_delta)),
        max_tokens_per_conv=args.max_tokens_per_conv,
        search_window_tokens=args.search_window_tokens,
    )
    conversations = []
    parse_summaries: dict[str, ParseSummary] = {}
    if args.conversations:
        loaded_groups = [load_conversations(source) for source in args.conversations]
        for loaded, _ in loaded_groups:
            conversations.extend(loaded)
        parse_summaries["lmstudio-chat"] = _merge_summaries(
            [summary for _, summary in loaded_groups]
        )
    if args.codex_sessions:
        loaded, summary = load_codex_sessions(
            args.codex_sessions,
            limit_files=args.codex_limit_files,
            max_file_mb=args.max_file_mb,
        )
        conversations.extend(loaded)
        parse_summaries["codex-agent"] = summary
    for corpus, summary in parse_summaries.items():
        print(
            f"{corpus}: loaded {summary.conversations_loaded}/{summary.files_seen} files "
            f"(selected={summary.files_selected or summary.files_seen}, "
            f"accounting_events={sum(summary.warning_counts.values())})",
            file=sys.stderr,
        )
    tokenizer = LocalTokenizer(args.tokenizer)
    results = simulate(
        conversations,
        parse_summaries,
        tokenizer,
        config,
        tokenizer_path=args.tokenizer,
        conversations_path={
            "lmstudio-chat": args.conversations or [],
            "codex-agent": args.codex_sessions or [],
        },
    )
    if args.also_cross_codex:
        if args.cross_conv:
            raise ValueError("--also-cross-codex cannot be combined with --cross-conv")
        codex_conversations = [
            conversation
            for conversation in conversations
            if conversation.corpus == "codex-agent"
        ]
        if not codex_conversations:
            raise ValueError("--also-cross-codex requires --codex-sessions")
        print("codex-agent: running additional global-window replay", file=sys.stderr)
        cross_results = simulate(
            codex_conversations,
            {"codex-agent": parse_summaries["codex-agent"]},
            tokenizer,
            replace(config, cross_conv=True),
            tokenizer_path=args.tokenizer,
            conversations_path={"codex-agent": args.codex_sessions or []},
        )
        results["statistics"].extend(cross_results["statistics"])
        results["extra_runs"] = [
            {
                "corpus": "codex-agent",
                "lookup_mode": "global-window",
                "search_window_tokens": config.search_window_tokens,
            }
        ]
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw_stats.json"
    report_path = destination / "report.md"
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")
    print(f"wrote {raw_path} and {report_path}", file=sys.stderr)
    return 0


def _merge_summaries(summaries: list[ParseSummary]) -> ParseSummary:
    merged = ParseSummary()
    for summary in summaries:
        for name in (
            "files_seen",
            "files_selected",
            "conversations_loaded",
            "skipped_corrupt",
            "skipped_empty",
            "skipped_messages",
            "skipped_oversize",
            "skipped_by_limit",
        ):
            setattr(merged, name, getattr(merged, name) + getattr(summary, name))
        for key, value in summary.warning_counts.items():
            merged.warning_counts[key] = merged.warning_counts.get(key, 0) + value
        merged.warnings.extend(summary.warnings)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        source = Path(args.results).expanduser().resolve()
        _output_dir(str(source if source.is_dir() else source.parent), args.allow_in_repo)
        destination = regenerate_report(args.results)
        print(f"wrote {destination}", file=sys.stderr)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
