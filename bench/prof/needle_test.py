#!/usr/bin/env python3
"""Needle check, thinking disabled: [approx_tokens] [depth 0-1] [label] [--dry-run]."""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import argparse
import json
import os
import random
import time

import requests

NEEDLE = "The maintenance passphrase for the east cabinet is AURORA-CEDAR-7319 and must be quoted exactly when asked."
QUESTION = "\n\nQuestion: What is the maintenance passphrase for the east cabinet? Reply with the passphrase only."
SUBJECTS = ["the harbor authority", "a regional archive", "the observatory staff", "the night shift", "a cartography guild", "the tram depot", "the river commission", "an apiary cooperative"]
VERBS = ["recorded", "reviewed", "catalogued", "postponed", "inspected", "reconciled", "transcribed", "audited"]
OBJECTS = ["the tide tables", "a set of brass gauges", "the quarterly ledgers", "several lantern housings", "the drainage survey", "a batch of glass plates", "the timber manifests", "the signal logs"]


def build_context(target_tokens, depth):
    """Reproduce the original seed-1234 paragraphs and needle insertion exactly."""
    rng = random.Random(1234)
    paras = []
    total = 0
    while total < target_tokens * 3.9:
        sentences = []
        for _ in range(6):
            sentences.append(f"On day {rng.randint(1,365)} {rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}, noting that entry {rng.randint(100,9999)} remained consistent with the earlier notes.")
        paragraph = " ".join(sentences)
        paras.append(paragraph)
        total += len(paragraph)
    paras.insert(int(len(paras) * depth), NEEDLE)
    return "\n\n".join(paras)


def parse_args(description, default_tokens=18500, default_depth=0.4):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("target_tokens", nargs="?", type=int, default=default_tokens)
    parser.add_argument("depth", nargs="?", type=float, default=default_depth)
    parser.add_argument("label", nargs="?", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_tokens <= 0 or not 0 <= args.depth <= 1:
        parser.error("target_tokens must be positive and depth must be between 0 and 1")
    return args


def dry_run(kind, args, context, prompt):
    row = dict(kind=kind, label=args.label, target_tokens=args.target_tokens,
               depth=args.depth, seed=1234, context_chars=len(context),
               prompt_chars=len(prompt), approx_tokens=round(len(prompt) / 3.9, 1),
               dry_run=True)
    print(f"[{args.label}] dry-run context_chars={len(context)} prompt_chars={len(prompt)} "
          f"approx_tokens={row['approx_tokens']} (chars/3.9; not tokenizer count)")
    print("JSON " + json.dumps(row))


def main():
    args = parse_args(__doc__)
    context = build_context(target_tokens=args.target_tokens, depth=args.depth)
    prompt = context + QUESTION
    if args.dry_run:
        dry_run(kind="needle", args=args, context=context, prompt=prompt)
        return 0
    port = int(os.environ.get("NEEDLE_PORT", "8001"))
    timeout = float(os.environ.get("NEEDLE_TIMEOUT", "600"))
    row = dict(kind="needle", label=args.label, target_tokens=args.target_tokens,
               depth=args.depth, prompt_tokens=None, completion_tokens=None,
               wall_s=None, passed=False, answer=None, error=None)
    t0 = time.perf_counter()
    try:
        with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
            "model": "flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 48, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=timeout) as response:
            response.raise_for_status()
            data = response.json()
        out = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        row.update(prompt_tokens=usage.get("prompt_tokens"),
                   completion_tokens=usage.get("completion_tokens"), answer=out,
                   passed="AURORA-CEDAR-7319" in (out or ""))
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["wall_s"] = time.perf_counter() - t0
    print(f"[{args.label}] prompt_tokens={row['prompt_tokens']} completion={row['completion_tokens']} "
          f"time={row['wall_s']:.1f}s PASS={row['passed']} out={row['answer']!r:.80}")
    if row["error"]:
        print(f"[{args.label}] error={row['error']}")
    print("JSON " + json.dumps(row))
    return 1 if row["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
