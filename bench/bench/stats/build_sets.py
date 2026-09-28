#!/usr/bin/env python3
"""Deterministically author BN1 synthetic holdouts; never sample a training parent.
Run with the existing production Python for tokenizers (CPU only, -B).
"""
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    groups = {k: [] for k in ('code-edit', 'prose-en', 'prose-ja', 'agent-loop')}
    code_specs = [
        ('seed_packets', 'viability', 'unchecked', 'tested', 3, 11, 'greenhouse'),
        ('tram_parts', 'inspection', 'pending', 'cleared', 4, 9, 'workshop'),
        ('ceramic_tiles', 'glaze', 'raw', 'fired', 6, 13, 'kiln'),
        ('field_recorders', 'calibration', 'due', 'aligned', 8, 15, 'valley'),
        ('library_lamps', 'service', 'waiting', 'renewed', 9, 17, 'reading_room'),
        ('canal_gauges', 'survey', 'unread', 'checked', 11, 19, 'lockhouse'),
        ('fabric_rolls', 'dye', 'natural', 'tinted', 12, 7, 'weaving_shed'),
        ('telescope_filters', 'coating', 'blank', 'finished', 13, 5, 'observatory'),
    ]
    for i, (table, key, old, new, mod, priority, owner) in enumerate(code_specs, 1):
        records = '\n'.join('    '+repr({'serial': i*1000+j, key: old, 'rack': f'{owner}_{j%4}',
                                        'label': f'{table} lot {j:03d}', 'slots': 2+j%6})+','
                            for j in range(1, 281))
        groups['code-edit'].append(f'''Return the complete revised Python module, with every record in its original order. No ellipses, omitted records, or generated comprehensions replacing literal data.
This newly authored fixture describes {table} at a fictional {owner}. Apply these edits:
1. Rename the record key {key!r} to 'stage' throughout.
2. Set stage to {new!r} when serial is divisible by {mod}; retain {old!r} otherwise.
3. Add 'review_band': 4 when serial is divisible by {priority}, else 2.
4. Replace totals() with a nested count by stage and review_band, including the total slots in each group.
5. Preserve rack, label and slots exactly. Return only the full Python module.

```python
from collections import Counter

{table.upper()} = [
{records}
]


def totals():
    return Counter(row[{key!r}] for row in {table.upper()})
```
''')
    en = [
        ('an inland seed library', 'a retired map printer', 'returned envelopes sort themselves by journeys never taken', 'a bent paper clip'),
        ('a closed hillside tram depot', 'the last ticket clerk', 'the destination blinds show rooms from her childhood', 'a brass washer'),
        ('a municipal swimming pool in winter', 'a tile restorer', 'dry lanes retain the sounds of absent swimmers', 'a folded towel'),
        ('a rural instrument repair shop', 'a visiting piano tuner', 'metronomes disagree only while someone conceals a question', 'a chipped tea saucer'),
        ('a disused mountain cable station', 'a rope inspector', 'cargo labels arrive describing things still upstairs', 'a wool glove'),
        ('a glass recycling sorting shed', 'a night supervisor', 'bottles cast shadows of windows in other buildings', 'a yellow pencil stub'),
        ('a small railway plant nursery', 'a volunteer gardener', 'seedling labels fade in the order of unspoken farewells', 'a wooden peg'),
        ('a village costume store', 'an apprentice seamstress', 'old hems hold dust from tomorrow\'s streets', 'a spool of grey thread'),
    ]
    for setting, person, mystery, obj in en:
        groups['prose-en'].append(f'''Write an original literary story of 4500–5500 words, in English prose without headings, bullets or an afterword. The setting is {setting}; the viewpoint character is {person}. Introduce {obj} naturally in the first paragraph. During one ordinary shift, {mystery}. Develop several patient scenes with concrete work, hesitant dialogue and an understated emotional change. Vary sentence length and imagery. Avoid a repeated refrain or an explanation of the phenomenon. In the final scene the opening object should carry a different meaning. Give the full story rather than an outline or summary.''')
    ja = [
        ('山間の活版印刷所', '廃業前の整理を任された植字工', '使っていない活字の裏から遠い食卓の気配がする', '欠けた定規'),
        ('冬の市営温室', '夜勤を引き継ぐ園芸員', '誰も通らない通路だけ葉の向きが変わる', '麻ひもの結び目'),
        ('旧街道の靴修理店', '店を閉じる決断をした修理職人', '預かった靴底にまだ舗装されていない道の跡がある', '片方だけの木型'),
        ('郊外の舞台衣装庫', '衣装目録を書き直す縫製係', '裏地の縫い目から知らない駅のざわめきが届く', '銀色の指ぬき'),
        ('運休中の渡し船の待合所', '備品点検に来た船員', '時刻表の空欄に合わせて椅子が温かくなる', '青い琺瑯の湯のみ'),
        ('古い公営スケート場', '冷却設備の管理人', '溶けかけた氷に明日の忘れ物の形が浮かぶ', '鍵についた布の札'),
        ('山裾の繭の選別場', '建物の測量を手伝う元作業員', '空の籠の重さが話題によって変わる', '短い竹の物差し'),
        ('小さな天文教材の工房', '模型の塗り直しをする職人', '太陽系模型の影が町の路地をなぞる', '赤く塗られた留め金'),
    ]
    for setting, person, mystery, obj in ja:
        groups['prose-ja'].append(f'''既存作品の文章や登場人物に依存しない、日本語の新しい文学的短編を12000〜15000字で書いてください。
舞台は{setting}、視点人物は{person}です。冒頭に{obj}を作業の一部として出してください。ある日の勤務中、{mystery}という現象に気づきます。
複数の場面を省略せず、仕事の手触り、言いよどむ会話、過去への小さなためらいを描いてください。静かな緊張感を保ち、比喩や文末を反復せず、謎を理屈で解明しないでください。最後の場面で冒頭の物の受け止め方が変わる構成にしてください。見出し、箇条書き、あらすじ、あとがきは不要です。本文だけを出力してください。''')
    bugs = [
        ('reservation_dates', 'exclusive end date is incorrectly included', 'range(start_day, end_day + 1)', 'range(start_day, end_day)', 'list(days(4, 6)) == [4, 5]'),
        ('sensor_labels', 'labels containing a slash lose their suffix', "label.split('/')[0]", 'label', "clean('north/roof') == 'north/roof'"),
        ('archive_order', 'equal timestamps reverse the original order', 'sorted(rows, key=lambda r: (r[0], -r[1]))', 'sorted(rows, key=lambda r: r[0])', "order([(7, 1), (7, 2)]) == [(7, 1), (7, 2)]"),
        ('quota_window', 'a request at the reset boundary uses the previous window', 'now > reset_at', 'now >= reset_at', 'expired(10, 10) is True'),
        ('ticket_zero', 'a legitimate zero identifier is discarded', 'if ticket_id:', 'if ticket_id is not None:', 'keep(0) == [0]'),
        ('path_suffix', 'a backup suffix is removed from the middle of a name', "name.replace('.bak', '')", "name.removesuffix('.bak')", "base('a.bak.csv') == 'a.bak.csv'"),
        ('retry_header', 'zero retry delay is replaced with the default', 'delay or 5', '5 if delay is None else delay', 'wait_seconds(0) == 0'),
        ('inventory_alias', 'two bins share one mutable list', '[[]] * count', '[[] for _ in range(count)]', 'bins(2)[0] is not bins(2)[1]'),
    ]
    for name, issue, bad, good, test in bugs:
        groups['agent-loop'].append(f'''Continue this fictional tool-using coding-agent transcript. Produce exactly twelve assistant turns alternating with eleven tool-result turns, keeping terminal.exec calls as JSON. Write a detailed investigation, minimal complete patch, focused boundary tests, review of the diff and a final verification summary. The known defect is that {issue}. The fixture below is supplied evidence; future command outcomes are unknown. Future TOOL turns must explicitly say "SIMULATED (not executed)" and distinguish expected results from observed evidence. Never claim tests actually passed. Include complete proposed code and tests rather than an outline; aim for 5000–6500 words across the continuation. Preserve the user-facing caution about unavailable execution.

USER: In the {name} module, {issue}. Inspect it, propose a fix, and verify the boundaries.
ASSISTANT: I will first read the relevant expression and the supplied regression case.
ASSISTANT to=terminal.exec:
{{"cmd":"rg -n . src/{name}.py tests/test_{name}.py"}}
TOOL (supplied fixture, not a live execution):
The current implementation uses `{bad}`.
The regression assertion is `{test}`.
A reviewer suggested `{good}`, but the full function and unrelated behavior still need checking.
ASSISTANT:
''')
    budgets = {'code-edit': 12288, 'prose-en': 6144, 'prose-ja': 6144, 'agent-loop': 8192}
    for domain, prompts in groups.items():
        folder = ROOT / 'workloads/sets' / f'{domain}-v1'; folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for i, text in enumerate(prompts, 1):
            text = text.strip(); pid = f'bn1-{domain}-{i:02d}'; filename = f'{i:02d}.txt'
            (folder / filename).write_text(text + '\n')
            entries.append(dict(id=pid, parent_id=pid, file=filename, sha256=sha(text.encode()),
                                input_tokens=len(tokenizer.encode(text, add_special_tokens=False).ids),
                                max_tokens=budgets[domain], duration_seconds_target=[15, 60],
                                duration_status='GPU calibration pending'))
        (folder/'manifest.json').write_text(json.dumps(dict(domain=domain, version='v1',
            provenance='Original synthetic fixtures authored for BN1 on 2026-09-08; build_sets.py. No training parent sampled. Domain style only follows legacy workloads.',
            token_count_method='tokenizer.json encode(strip(prompt), add_special_tokens=False); excludes server chat template',
            tokenizer_path=str(MODEL/'tokenizer.json'), tokenizer_sha256=sha((MODEL/'tokenizer.json').read_bytes()),
            exclusion_audit='../../sets/provenance-audit.json', prompts=entries), ensure_ascii=False, indent=2)+'\n')

if __name__ == '__main__':
    main()
