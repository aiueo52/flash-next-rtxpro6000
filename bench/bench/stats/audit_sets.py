#!/usr/bin/env python3
"""Audit exact and normalized containment plus 50-character source overlap.
Only prompt-bearing fields are examined; generated answers are not parents.

The report names the (private) source files and their hashes, so the output path is explicit:
  audit_sets.py --out PATH
The published copy is workloads/sets/provenance-audit.json; pass that path only to deliberately replace it.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [Path(p) for p in (
    '/home/user/mtp-gen-v5/gen.jsonl', '/home/user/mtp-gen-v5/prompts.jsonl',
    '/home/user/mtp-dump-mt1/prompts.json', '/home/user/mtp-dump-mt1/prompts-expanded.json',
    '/home/user/tools/mtp-train/results/dh7/prompts.json')]
TOKENIZER = Path('/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5/tokenizer.json')

def normalize(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s)).strip().casefold()

def texts(x, tokenizer):
    if isinstance(x, list):
        for v in x:
            yield from texts(v, tokenizer)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k in ('prompt_ids', 'input_ids') and isinstance(v, list) and all(isinstance(a, int) for a in v):
                yield tokenizer.decode(v, skip_special_tokens=True)
            elif k in ('prompt', 'source_prompt', 'text', 'content') and isinstance(v, str):
                yield v
            elif isinstance(v, (dict, list)):
                yield from texts(v, tokenizer)

def audit_status(prompts, files, collisions, suspicious):
    """PASS needs something to have been compared: no prompts or no source files verifies nothing."""
    if not prompts or not files:
        return 'UNVERIFIED'
    return 'PASS' if not collisions and not suspicious else 'REVIEW_REQUIRED'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='where to write the audit JSON')
    args = ap.parse_args()
    missing = [str(p) for p in [*SOURCES, TOKENIZER] if not p.exists()]
    if missing:
        raise SystemExit('audit_sets.py: needs the private source prompt files and the tokenizer, which are not '
                         'published in this repository; missing: ' + ', '.join(missing))
    tok = Tokenizer.from_file(str(TOKENIZER))
    prompts = {}
    for manifest in sorted((ROOT/'workloads/sets').glob('*-v1/manifest.json')):
        for p in json.loads(manifest.read_text())['prompts']:
            prompts[p['id']] = normalize((manifest.parent / p['file']).read_text())
    n = 50
    shingles = {k: {s[i:i+n] for i in range(len(s)-n+1)} for k,s in prompts.items()}
    index = {}
    for pid, ss in shingles.items():
        for s in ss:
            index.setdefault(s, []).append(pid)
    nearest = {k: dict(shared_shingles=0, fraction=0, source=None) for k in prompts}
    collisions, files = [], []
    for path in SOURCES:
        raw = path.read_bytes()
        rows = (json.loads(l) for l in raw.splitlines() if l.strip()) if path.suffix == '.jsonl' else [json.loads(raw)]
        count = 0
        for row in rows:
            for source in texts(row, tok):
                source = normalize(source)
                if not source: continue
                count += 1
                shared = {}
                for sh in {source[i:i+n] for i in range(max(0,len(source)-n+1))}:
                    for pid in index.get(sh, []):
                        shared[pid] = shared.get(pid, 0) + 1
                for pid, value in prompts.items():
                    if value in source or (len(source) >= 200 and source in value):
                        collisions.append(dict(id=pid, source=str(path), ordinal=count, kind='normalized_containment'))
                for pid, overlap in shared.items():
                    if overlap > nearest[pid]['shared_shingles']:
                        fraction = overlap / max(1, len(shingles[pid]))
                        nearest[pid] = dict(shared_shingles=overlap, fraction=fraction, source=str(path), ordinal=count)
        if count == 0: raise ValueError(f'No prompt fields found in {path}')
        files.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), prompt_texts=count))
    suspicious = [k for k,v in nearest.items() if v['fraction'] >= .20]
    status = audit_status(prompts, files, collisions, suspicious)
    report = dict(status=status,
        sources=files, prompts=len(prompts), exact_or_containment_collisions=collisions,
        near_overlap_review_threshold=.20, nearest_source_overlap=nearest, suspicious=suspicious,
        scope='Original authorship plus normalized prompt containment and character-shingle overlap against the listed frozen sources. Not a proof against unavailable corpora or later training additions. All BN1 parents reserved for evaluation; never train on them.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], sources=len(files), prompts=len(prompts), collisions=len(collisions), suspicious=suspicious)))
    if report['status'] != 'PASS': raise SystemExit(1)

if __name__ == '__main__': main()
