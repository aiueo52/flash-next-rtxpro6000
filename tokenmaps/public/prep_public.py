#!/usr/bin/env python3
"""Turn the files fetched by fetch_public.sh into one JSONL per source ({"text": ...} per document) for
tokenmaps/build_hot_vocab.py. Only the *generated side* is kept where a source is a conversation
(assistant / "gpt" / "ai" turns), because the map has to cover what the model writes, not what it reads.
Deterministic: rows are taken in file order up to a per-source character cap. CPU only.

  prep_public.py [--raw ~/datasets/public-tokenmap/raw] [--out ~/datasets/public-tokenmap/text]
Needs pyarrow (any env that has it; the SGLang venv does).
"""
import argparse, ast, json, os, tarfile
from pathlib import Path

CODE_EXT = {'.py', '.pyi', '.go', '.js', '.jsx', '.mjs', '.ts', '.tsx', '.vue', '.rs', '.c', '.h', '.cc', '.cpp',
            '.hpp', '.sh', '.toml', '.yml', '.yaml', '.md', '.rst', '.html', '.css', '.cmake', '.s'}
TARBALLS = ['cpython-v3.13.0.tar.gz', 'go-go1.23.0.tar.gz', 'react-v18.3.1.tar.gz', 'vuejs-core-v3.5.12.tar.gz',
            'tokio-1.40.0.tar.gz', 'curl-8_10_1.tar.gz']
CAPS = dict(code=120_000_000, enwiki=80_000_000, jawiki=40_000_000, en_chat=60_000_000, ja_chat=40_000_000,
            reasoning=80_000_000, agent_swe=60_000_000, agent_tools=40_000_000)
PER_TARBALL = 30_000_000


class Sink:
    def __init__(self, path, cap):
        self.f = open(path, 'w', encoding='utf-8'); self.cap = cap; self.chars = 0; self.docs = 0
    def add(self, text):
        if not isinstance(text, str) or not text.strip() or self.full(): return
        self.f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        self.chars += len(text); self.docs += 1
    def full(self): return self.chars >= self.cap
    def close(self): self.f.close(); return dict(docs=self.docs, chars=self.chars)


def as_list(v):  # parquet list columns come back as lists; tolerate stringified ones
    if isinstance(v, str):
        try: return ast.literal_eval(v)
        except Exception: return []
    return v or []


def parquet_rows(path, columns):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        for row in pf.read_row_group(i, columns=columns).to_pylist():
            yield row


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--raw', type=Path, default=Path.home() / 'datasets/public-tokenmap/raw')
    a.add_argument('--out', type=Path, default=Path.home() / 'datasets/public-tokenmap/text')
    o = a.parse_args(); o.out.mkdir(parents=True, exist_ok=True); R = o.raw
    stats = {}

    s = Sink(o.out / 'code.jsonl', CAPS['code']); per = {}
    for tb in TARBALLS:
        got = 0
        with tarfile.open(R / tb) as t:
            for m in t:
                if not m.isfile() or m.size > 256_000 or os.path.splitext(m.name)[1].lower() not in CODE_EXT: continue
                try: text = t.extractfile(m).read().decode('utf-8')
                except UnicodeDecodeError: continue
                if '\0' in text: continue
                s.add(text); got += len(text)
                if got >= PER_TARBALL or s.full(): break
        per[tb] = got
    stats['code'] = dict(s.close(), per_tarball_chars=per)

    for name, f in (('enwiki', 'enwiki-20231101-00007.parquet'), ('jawiki', 'jawiki-20231101-00007.parquet')):
        s = Sink(o.out / f'{name}.jsonl', CAPS[name])
        for r in parquet_rows(R / f, ['text']):
            s.add(r['text'])
            if s.full(): break
        stats[name] = s.close()

    s = Sink(o.out / 'en_chat.jsonl', CAPS['en_chat'])
    for r in parquet_rows(R / 'ultrachat-test_sft.parquet', ['messages']):
        for m in as_list(r['messages']):
            if m.get('role') == 'assistant': s.add(m.get('content'))
        if s.full(): break
    stats['en_chat'] = s.close()

    s = Sink(o.out / 'ja_chat.jsonl', CAPS['ja_chat'])
    with open(R / 'oasst2-33k-ja.jsonl', encoding='utf-8') as f:
        for line in f:
            for m in json.loads(line).get('conversations', []):
                if m.get('role') == 'assistant': s.add(m.get('content'))
            if s.full(): break
    stats['ja_chat'] = s.close()

    s = Sink(o.out / 'reasoning.jsonl', CAPS['reasoning'])
    for r in parquet_rows(R / 'openthoughts-00005.parquet', ['conversations']):
        for m in as_list(r['conversations']):
            if m.get('from') == 'assistant': s.add(m.get('value'))
        if s.full(): break
    stats['reasoning'] = s.close()

    s = Sink(o.out / 'agent_swe.jsonl', CAPS['agent_swe'])
    for r in parquet_rows(R / 'swe-agent-traj-00000.parquet', ['trajectory']):
        for m in as_list(r['trajectory']):
            if m.get('role') == 'ai': s.add(m.get('text'))
        if s.full(): break
    stats['agent_swe'] = s.close()

    s = Sink(o.out / 'agent_tools.jsonl', CAPS['agent_tools'])
    for f in ('hermes-func-calling.json', 'hermes-json-mode-agentic.json'):
        for r in json.load(open(R / f, encoding='utf-8')):
            for m in r.get('conversations', []):
                if m.get('from') == 'gpt': s.add(m.get('value'))
    stats['agent_tools'] = s.close()

    (o.out / 'prep-stats.json').write_text(json.dumps(stats, indent=1) + '\n')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
