from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .report import ReportError, render_report
from .runner import EndpointError, RunConfig, run_benchmark
from .workloads import WorkloadError, select_prompt_sets, select_workloads

DEFAULT_WORKLOADS = "code-edit,prose-ja,prose-en,agent-loop,long-ctx"


def _default_output() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fnbench",
        description="Benchmark a local OpenAI-compatible streaming endpoint.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run benchmark workloads")
    run.add_argument("--endpoint", required=True, help="OpenAI-compatible /v1 URL")
    run.add_argument("--engine", required=True, choices=("sglang", "generic", "llamacpp"))
    run.add_argument("--workloads", default=DEFAULT_WORKLOADS, help="comma-separated names")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--prompt-sets", type=Path, help="root containing <domain>-v1/manifest.json")
    run.add_argument("--set-version", default="v1")
    run.add_argument("--prompt-limit", type=int)
    run.add_argument("--require-acceptance", action="store_true")
    run.add_argument("--out", type=Path, help="output JSONL (default: runs/<timestamp>.jsonl)")
    run.add_argument("--label")
    run.add_argument("--force", action="store_true", help="override the GPU process guard")
    run.add_argument(
        "--allow-proc",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="allow an additional GPU process path substring (repeatable)",
    )
    run.add_argument("--sampling", choices=("greedy", "recommended"), default="greedy")
    run.add_argument(
        "--model",
        help="model field sent to the endpoint (default: first item from GET /models)",
    )

    report = subparsers.add_parser("report", help="render a Markdown aggregate report")
    report.add_argument("path", type=Path)
    report.add_argument("--baseline", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = RunConfig(
                endpoint=args.endpoint,
                engine=args.engine,
                workloads=(select_prompt_sets(args.workloads, args.prompt_sets, args.set_version, args.prompt_limit)
                           if args.prompt_sets else select_workloads(args.workloads)),
                require_acceptance=args.require_acceptance,
                repeats=args.repeats,
                out=args.out or _default_output(),
                label=args.label,
                force=args.force,
                sampling_mode=args.sampling,
                model=args.model,
                allow_proc=tuple(args.allow_proc),
            )
            return run_benchmark(config)
        print(render_report(args.path, args.baseline), end="")
        return 0
    except (
        EndpointError,
        FileExistsError,
        ReportError,
        RuntimeError,
        ValueError,
        WorkloadError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
