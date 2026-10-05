#!/usr/bin/env python3
"""Stream a summary of the needle haystack: [approx_tokens] [depth] [label] [--dry-run]."""
import json
import os
import time

import requests

from needle_test import build_context, dry_run, parse_args

QUESTION = "\n\nSummarize the records above in about 300 words."


def main():
    args = parse_args(__doc__, default_tokens=2000, default_depth=0.75)
    context = build_context(target_tokens=args.target_tokens, depth=args.depth)
    prompt = context + QUESTION
    if args.dry_run:
        dry_run(kind="decode", args=args, context=context, prompt=prompt)
        return 0
    port = int(os.environ.get("NEEDLE_PORT", "8001"))
    timeout = float(os.environ.get("NEEDLE_TIMEOUT", "600"))
    row = dict(kind="decode", label=args.label, target_tokens=args.target_tokens,
               depth=args.depth, prompt_tokens=None, completion_tokens=None,
               ttft_s=None, decode_tokens=None, decode_s=None, decode_tps=None,
               wall_s=None, answer="", error=None)
    first = last = first_tokens = last_tokens = None
    done = False
    t0 = time.perf_counter()
    try:
        with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
            "model": "flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.8, "top_p": 0.95,
            "top_k": 40, "min_p": 0.05, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                data = json.loads(payload)
                if data.get("error"):
                    raise ValueError(f"stream error: {data['error']}")
                usage = data.get("usage") or {}
                if usage:
                    row.update(prompt_tokens=usage.get("prompt_tokens"),
                               completion_tokens=usage.get("completion_tokens"))
                choices = data.get("choices") or []
                content = choices[0].get("delta", {}).get("content") if choices else None
                if content:
                    now = time.perf_counter()
                    tokens = usage.get("completion_tokens")
                    if tokens is None:
                        raise ValueError("content chunk lacks cumulative completion_tokens")
                    if first is None:
                        first, first_tokens = now, tokens
                        row["ttft_s"] = first - t0
                    last, last_tokens = now, tokens
                    row["answer"] += content
        if not done:
            raise ValueError("stream ended without [DONE]")
        if first is None or last <= first or last_tokens <= first_tokens:
            raise ValueError("not enough token-bearing chunks to measure decode speed")
        # Exclude tokens already delivered in the first speculative chunk.
        row.update(decode_tokens=last_tokens - first_tokens, decode_s=last - first,
                   decode_tps=(last_tokens - first_tokens) / (last - first))
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["wall_s"] = time.perf_counter() - t0
    print(f"[{args.label}] prompt_tokens={row['prompt_tokens']} TTFT={row['ttft_s']}s "
          f"completion={row['completion_tokens']} decode_tokens={row['decode_tokens']} "
          f"decode_tps={row['decode_tps']} time={row['wall_s']:.1f}s")
    if row["error"]:
        print(f"[{args.label}] error={row['error']}")
    print("JSON " + json.dumps(row))
    return 1 if row["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
