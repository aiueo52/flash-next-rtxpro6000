#!/usr/bin/env python3
"""CPU-only, deterministic QL1 prompt construction with exact rendered lengths."""
import hashlib
import json
import random
from pathlib import Path
from transformers import AutoTokenizer

B = Path(__file__).resolve().parents[1]
R = B.parent / 'sglang-rtxpro6000'
M = Path('/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5')
O = B / 'workloads/longctx'
O.mkdir(exist_ok=True)
t = AutoTokenizer.from_pretrained(M, local_files_only=True)
t.chat_template = (M / 'chat_template.jinja').read_text()

def render(s):
    return t.apply_chat_template([{'role': 'user', 'content': s}], tokenize=True,
                                 add_generation_prompt=True, enable_thinking=False, return_dict=False)

rng = random.Random(1234)
subjects = ['the harbor authority', 'a regional archive', 'the observatory staff', 'the night shift',
            'a cartography guild', 'the tram depot', 'the river commission', 'an apiary cooperative']
verbs = ['recorded', 'reviewed', 'catalogued', 'postponed', 'inspected', 'reconciled', 'transcribed', 'audited']
objects = ['the tide tables', 'a set of brass gauges', 'the quarterly ledgers', 'several lantern housings',
           'the drainage survey', 'a batch of glass plates', 'the timber manifests', 'the signal logs']
hay = '\n\n'.join(' '.join(
    f'On day {rng.randint(1,365)} {rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}, '
    f'noting that entry {rng.randint(100,9999)} remained consistent with the earlier notes.'
    for _ in range(6)) for i in range(1000))
needle = '\n\nThe maintenance passphrase for the east cabinet is AURORA-CEDAR-7319 and must be quoted exactly when asked.\n\n'
doc = '\n\n'.join(
    f'Record {i:05d}: {subjects[i % 8]} inspected asset {i % 173:03d} on day {i % 365 + 1}. '
    f'The authorized limit was {100 + i % 19} units and the recorded consumption was {87 + i % 23} units. '
    f'Ticket H-{i:05d} remains {("open", "closed", "pending review")[i % 3]}. '
    'The inspection log records intact housing, legible labels, and a signed inventory sheet. '
    'Local discrepancies require an independent recount before the next handover; routine observations do not supersede signed directives.'
    for i in range(3500))
doc_ids = t.encode(doc, add_special_tokens=False)
hay_ids = t.encode(hay, add_special_tokens=False)
task_a = ('\n\nTask: Write a detailed operational handover report of at least 900 words for the next harbor shift. '
          'Use the signed directives in this document to identify the affected assets, deadlines, and required safety actions. '
          'Explain the distinction between historical inspection records and current instructions. Include a numbered action plan, '
          'an evidence table quoting each current directive, and a verification checklist. Begin with the most urgent action.')
task_b = ('\n\nTask: Write a detailed incident reconciliation report of at least 900 words. Compare the signed directives '
          'in this archive, identify which instruction supersedes which, and propose a traceable resolution for each affected asset. '
          'Cite the identifiers, explain the ordering and resource constraints, and finish with an escalation and validation plan. '
          'Begin with a concise account of the conflict and its resolution.')
directives = {
    'doc-a': [(0.12, '\n\nSIGNED DIRECTIVE A-17: Suspend use of crane Cobalt until its brake test passes; supervisor Mira owns the test before the dawn shift.\n\n'),
              (0.51, '\n\nSIGNED DIRECTIVE B-29: Reserve two spare lanterns for pier Seven and verify their seals before the evening ferry; inspector Arun signs the handover.\n\n'),
              (0.86, '\n\nSIGNED DIRECTIVE C-41: Deliver the corrected tide-table packet to the west control room by 18:00; clerk Sora confirms receipt.\n\n')],
    'doc-b': [(0.12, '\n\nSIGNED DIRECTIVE R-10 revision 1: Allocate pump Amber to the north basin by 15:00; owner Imani.\n\n'),
              (0.51, '\n\nSIGNED DIRECTIVE R-10 revision 2: Supersedes revision 1. Allocate pump Amber to the east basin by 14:00 because the east gate is leaking; owner Imani.\n\n'),
              (0.86, '\n\nSIGNED DIRECTIVE S-20: Pump Birch is available for the north basin after its 13:00 seal inspection; owner Leon. Record test results before transfer.\n\n')]
}

def make(n, family, depth=None):
    if family.startswith('doc'):
        source = doc_ids; inserts = directives[family]; task = task_a if family == 'doc-a' else task_b
    else:
        source = hay_ids; inserts = [(depth, needle)]
        task = ('\n\nQuestion: What is the maintenance passphrase for the east cabinet? Reply with the passphrase only.'
                if family == 'needle-qa' else
                '\n\nTask: First quote the maintenance passphrase for the east cabinet exactly. Then write an archival audit report '
                'of at least 900 words explaining how to retrieve and verify the cabinet instruction from the surrounding records. '
                'Discuss provenance, similarly named records, exact transcription, handover checks, and a numbered verification workflow.')
    def content(k):
        pieces = []; last = 0
        for d, s in inserts:
            ix = int(k * d); pieces += [t.decode(source[last:ix]), s]; last = ix
        pieces += [t.decode(source[last:k]), '\n\nArchive padding:']
        return ''.join(pieces)
    # Reserve space for the complete task and insertions, then pad before the task.
    k = n - len(render(content(0) + task)) - 100
    body = content(k)
    while len(render(body + task)) > n:
        k -= 128; body = content(k)
    remain = n - len(render(body + task))
    body += ' x' * remain
    ids = render(body + task)
    assert len(ids) == n, (family, n, len(ids))
    name = f'{family}-{n//1024}k' + (f'-d{int(depth*100):02d}' if depth is not None else '')
    prompt = body + task
    (O / f'{name}.txt').write_text(prompt)
    (O / f'{name}.ids.json').write_text(json.dumps(ids))
    insertion_positions = []
    for d, s in inserts:
        pos = prompt.index(s)
        insertion_positions.append({'target_depth': d, 'body_token_offset': len(t.encode(prompt[:pos], add_special_tokens=False))})
    return dict(id=name, family=family, context_tokens=n, depth=depth,
                text=f'{name}.txt', input_ids=f'{name}.ids.json',
                sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                rendered_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                insertion_positions=insertion_positions, perf=family != 'needle-qa')

rows = []
for n in [8192, 32768, 65536, 98304]:
    for family in ['doc-a', 'doc-b', 'needle-perf']:
        rows.append(make(n, family, 0.4 if family.startswith('needle') else None))
    for depth in [0.1, 0.4, 0.9]:
        rows.append(make(n, 'needle-qa', depth))
manifest = dict(model=str(M), seed=1234, thinking=False, tokenizer_sha256=hashlib.sha256((M/'tokenizer.json').read_bytes()).hexdigest(),
                chat_template_sha256=hashlib.sha256((M/'chat_template.jinja').read_bytes()).hexdigest(),
                convention='k=1024; exact rendered prompt tokens including chat template; input_ids used without re-templating', prompts=rows)
(O/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Built {len(rows)} exact-length prompts in {O}')
