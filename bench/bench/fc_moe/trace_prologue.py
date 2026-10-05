"""fc-moe: prologue (fusedBuildExpertMapsSort...) median duration by grid size, per trace dir.
grid 26 = draft T=1, 56 = W4 verify T=4, 176 = W16 verify T=16, 16 = pre-G2 builds (all calls)."""
import collections, glob, gzip, json, statistics as st, sys
for d in sys.argv[1:]:
    fs = sorted(glob.glob(d + '/*.json.gz'))
    if not fs:
        continue
    ev = json.load(gzip.open(fs[0]))['traceEvents']
    c = collections.defaultdict(list)
    for e in ev:
        if e.get('cat') == 'kernel' and 'fusedBuildExpertMapsSortF' in e['name']:
            c[e['args']['grid'][0]].append(e['dur'])
    print(d.rstrip('/').split('/')[-1], {g: (len(v), round(st.median(v), 2)) for g, v in sorted(c.items())})
