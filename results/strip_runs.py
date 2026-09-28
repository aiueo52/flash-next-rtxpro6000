"""Strip fnbench JSONL records to timing/acceptance/config fields (no text, no raw metrics)."""
import json, sys, re, pathlib
KEEP_TOP = ['schema_version','timestamp','label','engine','model','workload','prompt_id','prompt_set',
            'prompt_sha256','input_tokens_manifest','repeat','sampling_mode','sampling','max_tokens','prompt_chars']
KEEP_CLIENT = ['ttft_seconds','decode_seconds','decode_tps','usage','output_chars','reasoning_chars']
DELTA_RE = re.compile(r'spec_|accept|generation_tokens|verify')
def strip(r):
    o = {k: r[k] for k in KEEP_TOP if k in r}
    c = r.get('client', {})
    oc = {k: c[k] for k in KEEP_CLIENT if k in c}
    if 'timeline' in c:
        oc['sse_chunks'] = len(c['timeline'])  # number of token-bearing SSE chunks (~ verify steps)
    o['client'] = oc
    s = r.get('server', {}) or {}
    os_ = {}
    d = ((s.get('metrics') or {}).get('delta')) or {}
    dd = {}
    for k, v in d.items():
        if DELTA_RE.search(k) and '_bucket' not in k:
            name = k.split('{')[0]
            dd[name] = v
    if dd: os_['metrics_delta'] = dd
    for k in ('acceptance', 'accept_length'):
        if k in s: os_[k] = s[k]
    if 'meta_info' in s and isinstance(s['meta_info'], dict):
        mi = {k: v for k, v in s['meta_info'].items() if 'spec' in k or 'accept' in k}
        if mi: os_['meta_info'] = mi
    o['server'] = os_
    if 'derived' in r: o['derived'] = r['derived']
    return o
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
n = 0
with open(src) as f, open(dst, 'w') as g:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if 'schema_version' not in r: continue
        g.write(json.dumps(strip(r), ensure_ascii=False) + '\n'); n += 1
print(dst.name, n)
