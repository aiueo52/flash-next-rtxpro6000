#!/usr/bin/env python3
"""Collect the target model's own greedy outputs for building a token map (stdlib only).

Under greedy decoding the tokens the server emits are exactly the target's argmax path, which is
what the verifier compares the draft against -- the second ("self-generated argmax") source of the
production hot2_* maps.  Point this at a running server and a prompt file you are allowed to use
(one JSON object per line with a "prompt" field, or plain text with one prompt per line).

  python tokenmaps/collect_selfgen.py --endpoint http://127.0.0.1:8001/v1 \
      --prompts my_prompts.jsonl --out ~/tokenmap-work/gen.jsonl --max-tokens 2048

Output: JSONL with {"id", "text"} where text = reasoning_content + content, ready for
build_hot_vocab.py --jsonl selfgen=$HOME/tokenmap-work/gen.jsonl. The output is generated from your
prompts, so a path inside the repository is refused unless --allow-in-repo is given.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def repository_root() -> Path:
    """The checkout this script lives in (tokenmaps/ sits at its top level)."""
    return Path(__file__).resolve().parent.parent


def outside_repo(path: Path, allow_in_repo: bool, what: str) -> Path:
    """Refuse an output path inside the repository unless allow_in_repo: the output is generated from the
    user's prompts, which may be private."""
    destination = path.expanduser().resolve()
    root = repository_root()
    try:
        destination.relative_to(root)
    except ValueError:
        return destination
    if not allow_in_repo:
        raise SystemExit(
            f"refusing to write {what} inside the repository ({root}): {destination}. "
            "Choose a path outside the repository, or pass --allow-in-repo (and do not commit it).")
    return destination


def prompts(path: Path):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            yield i, obj["prompt"] if isinstance(obj, dict) else str(obj)
        except (json.JSONDecodeError, KeyError, TypeError):
            yield i, line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="flash-next")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSONL (generated text for your prompts; keep it outside the repository)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--allow-in-repo", action="store_true",
                    help="allow --out inside the repository (do not commit it)")
    args = ap.parse_args()
    args.out = outside_repo(args.out, args.allow_in_repo, "generated text")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    with args.out.open("w", encoding="utf-8") as g:
        for i, p in prompts(args.prompts):
            body = json.dumps({"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                               "messages": [{"role": "user", "content": p}]}).encode()
            req = urllib.request.Request(args.endpoint.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
            text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
            g.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")
            print(f"prompt {i}: {len(text)} chars")


if __name__ == "__main__":
    main()
