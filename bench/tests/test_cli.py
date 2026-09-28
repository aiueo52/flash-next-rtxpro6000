from __future__ import annotations

from fnbench.cli import build_parser


def test_allow_proc_is_repeatable() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--endpoint",
            "http://127.0.0.1:8001/v1",
            "--engine",
            "sglang",
            "--allow-proc",
            "/srv/engine-a/",
            "--allow-proc",
            "engine-b",
        ]
    )
    assert args.allow_proc == ["/srv/engine-a/", "engine-b"]


def test_quality_subcommands_are_not_shipped() -> None:
    """The 12-task quality battery was removed (its questions were of unshown origin)."""
    import pytest

    for argv in (["quality", "--help"], ["quality-diff", "a.json", "b.json"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)
