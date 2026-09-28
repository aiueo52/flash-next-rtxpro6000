#!/usr/bin/env python3
"""Hot-vocab maps v2: blend (a) generated text of the author's own agent sessions and
chat logs (private, not published; as build_hot_vocab_v1.py) with (b) the target's own argmax over every row of the
self-generated dumps in ~/mtp-dump (what the verifier actually compares
against).  Frequencies are normalised per source and averaged 50/50, so neither
source dominates the blend by sheer token count.

    python build_hot_vocab2.py            # writes hot2_32768.pt, hot2_49152.pt
"""
import sys, json, random, collections, os, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.expanduser("~/tools/ngram-accept-sim"))
import torch
from safetensors import safe_open

OUT = os.path.expanduser("~/tools/flash-next-bench/tokenmaps")
CACHE = os.path.join(OUT, "counts_private_corpus.json")
MODEL = os.path.expanduser("~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4")

# (a) corpora counts (cached)
if os.path.exists(CACHE):
    codex = collections.Counter({int(k): v for k, v in json.load(open(CACHE)).items()})
    print("corpora counts from cache:", sum(codex.values()), "tokens")
else:
    from ngramsim.codex_parser import load_codex_sessions
    from ngramsim.parser import load_conversations
    from ngramsim.tokenization import LocalTokenizer
    tok = LocalTokenizer(MODEL)
    cx, _ = load_codex_sessions([os.environ["AGENT_SESSIONS_DIR"]], limit_files=600, max_file_mb=20)
    lms, _ = load_conversations(os.environ["CHAT_EXPORTS_DIR"])
    codex = collections.Counter()
    for conv in cx + lms:
        for m in conv.messages:
            if m.decode_target and m.text:
                codex.update(tok.encode(m.text))
    json.dump({str(k): v for k, v in codex.items()}, open(CACHE, "w"))
    print("corpora tokens", sum(codex.values()), "distinct", len(codex))

# (b) self-generated dumps: target argmax over all rows + generated tokens
man = [json.loads(l) for l in open(os.path.expanduser("~/mtp-dump/manifest.jsonl"))]
idx = json.load(open(os.path.expanduser("~/mtp-dump/index.json")))
sg = collections.Counter(); nrows = 0
per_bucket = collections.defaultdict(collections.Counter)
for r in man:
    if r.get("mode") != "selfgen" or r["doc_hash"] not in idx:
        continue
    ent = idx[r["doc_hash"]]
    with safe_open(os.path.join(os.path.expanduser("~/mtp-dump"), ent["file"]), "pt", device="cpu") as f:
        tgt = f.get_tensor("target_argmax").tolist()
    sg.update(tgt); per_bucket[r.get("bucket") or "selfgen"].update(tgt); nrows += len(tgt)
print("selfgen argmax rows", nrows, "distinct", len(sg), {b: len(c) for b, c in per_bucket.items()})

# blend
ta, tb = sum(codex.values()), sum(sg.values())
score = collections.Counter()
for t, c in codex.items(): score[t] += 0.5 * c / ta
for t, c in sg.items(): score[t] += 0.5 * c / tb
tj = json.load(open(os.path.join(MODEL, "tokenizer.json")))
special = [t["id"] for t in tj.get("added_tokens", [])]
ranked = [t for t, _ in score.most_common()]
print("distinct union", len(ranked))
for K in (32768, 40960, 45056, 49152):
    hot, seen = [], set()
    for t in special + ranked:
        if t not in seen:
            seen.add(t); hot.append(t)
        if len(hot) >= K: break
    hs = set(hot)
    cov_sg = sum(c for t, c in sg.items() if t in hs) / tb
    cov_cx = sum(c for t, c in codex.items() if t in hs) / ta
    covb = {b: round(100 * sum(v for t, v in c.items() if t in hs) / sum(c.values()), 2) for b, c in per_bucket.items()}
    torch.save(sorted(hot), f"{OUT}/hot2_{K}.pt")
    print(f"K={K} rows={len(hot)} coverage selfgen {cov_sg*100:.2f}% corpora {cov_cx*100:.2f}% by bucket {covb} -> hot2_{K}.pt")
# for reference: old map coverage on the selfgen argmax
for name in ("hot_32768", "hot_49152"):
    hs = set(torch.load(f"{OUT}/{name}.pt"))
    covb = {b: round(100 * sum(v for t, v in c.items() if t in hs) / sum(c.values()), 2) for b, c in per_bucket.items()}
    print(f"{name}: coverage selfgen {100*sum(c for t,c in sg.items() if t in hs)/tb:.2f}% by bucket {covb}")
