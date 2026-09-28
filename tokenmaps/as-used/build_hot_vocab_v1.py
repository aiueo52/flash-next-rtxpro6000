#!/usr/bin/env python3
"""Build FR-Spec style hot-vocab token maps from the author's own agent sessions and chat
logs (private, not published; paths come from AGENT_SESSIONS_DIR / CHAT_EXPORTS_DIR).
Counts token frequencies over generated (assistant) text, reports held-out coverage,
and saves top-K id lists as .pt for --speculative-token-map."""
import sys, json, random, collections, os
sys.path.insert(0, os.path.expanduser("~/tools/ngram-accept-sim"))
from ngramsim.codex_parser import load_codex_sessions
from ngramsim.parser import load_conversations
from ngramsim.tokenization import LocalTokenizer
import torch
tok = LocalTokenizer(os.path.expanduser("~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"))
codex, s1 = load_codex_sessions([os.environ["AGENT_SESSIONS_DIR"]], limit_files=600, max_file_mb=20)
lms, s2 = load_conversations(os.environ["CHAT_EXPORTS_DIR"])
print("agent sessions", len(codex), "chat conversations", len(lms))
random.seed(7)
train = collections.Counter(); held = collections.Counter()
ntrain = nheld = 0
for conv in codex + lms:
    is_held = random.random() < 0.1
    for m in conv.messages:
        if not m.decode_target or not m.text: continue
        ids = tok.encode(m.text)
        (held if is_held else train).update(ids)
        if is_held: nheld += len(ids)
        else: ntrain += len(ids)
print(f"train tokens {ntrain}  held tokens {nheld}  distinct train {len(train)}")
# special/added tokens always included
tj = json.load(open(os.path.expanduser("~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/tokenizer.json")))
special = [t["id"] for t in tj.get("added_tokens", [])]
print("added tokens", len(special))
ranked = [t for t, _ in train.most_common()]
out_dir = os.path.expanduser("~/tools/flash-next-bench/tokenmaps")
for K in (16384, 32768, 49152, 65536, 98304):
    hot = []
    seen = set()
    for t in special + ranked:
        if t not in seen:
            seen.add(t); hot.append(t)
        if len(hot) >= K: break
    hs = set(hot)
    cov = sum(c for t, c in held.items() if t in hs) / max(1, nheld)
    covt = sum(c for t, c in train.items() if t in hs) / max(1, ntrain)
    torch.save(sorted(hot), f"{out_dir}/hot_{K}.pt")
    print(f"K={K:6d} held-out coverage {cov*100:.2f}%  train {covt*100:.2f}%  -> hot_{K}.pt")
