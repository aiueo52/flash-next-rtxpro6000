# ngram-accept-sim (English summary)

Offline CPU simulator that replays recorded conversations token by token and measures how many draft
tokens an n-gram / suffix-match drafter (optionally hybrid with an MTP chain) would have had accepted,
plus a simple cost model that turns acceptance into projected tokens/s. It needs only the Qwen3.8
`tokenizer.json` and the `tokenizers` package; no GPU, no model weights, no network.

It was used on 2026-09-01 to decide whether request-local n-gram drafting was worth implementing.
The corpora it was run on (the author's own coding-agent session logs and chat exports) are private
and not included, nor are the result files, any statistics of them, or any result or conclusion
from the simulator's runs on them. N-gram drafting was implemented as `NGRAM_CHAIN` in the patch
series; on the public workloads that live implementation measured hit rates of 37–49 %, but lost to
plain wide MTP drafting; see `../docs/rejected.md`.

Usage and details follow in Japanese (the original README). Input parsers exist for LM Studio
conversation JSON and Codex CLI rollout JSONL; point `--conversations` / `--codex-sessions` at your
own data. Tests (`pytest -q`) use only synthetic fixtures.

---

# ngram-accept-sim

Qwen3.8-Flash-Next のローカル `tokenizer.json` と、LM Studio会話またはCodex CLI agent rolloutを使い、テキスト履歴自己投機のドラフト受理長をCPUでリプレイ測定するオフラインシミュレータです。GPU、PyTorch、ネットワークアクセス、モデル重みは使いません。

## セットアップ

Python 3.10以上を想定しています。依存はCPU版の Hugging Face `tokenizers` とテスト用 `pytest` だけです。ネットワーク禁止環境では、事前取得済みwheelまたは既存のローカルパッケージ環境から導入してください。

```bash
python -m venv .venv
.venv/bin/pip install --no-index --find-links /path/to/local/wheels -r requirements.txt
```

実行時に参照するのは指定した会話JSONとtokenizerだけです。出力（会話名・入力パス・会話別の統計を含む、非公開コーパス由来のデータ）は誤ってcommitしないよう、リポジトリの外に書きます。`--out` は必須で、リポジトリ配下を指定するとエラーになります（`--allow-in-repo` で明示的に許可できますが、その出力はcommitしないでください。旧既定の `sim/results/` は `.gitignore` 済みです）。

## 実行

```bash
python -m ngramsim run \
  --conversations /path/to/chat-exports \
  --codex-sessions /path/to/codex-sessions \
  --tokenizer ~/models/Qwen/Qwen3.8-Flash-Next-BF16 \
  --out ~/ngramsim-out/run1

python -m ngramsim report ~/ngramsim-out/run1
```

主なフラグは次の通りです。

- `--strategy both|suffix-match|ngram-mod`（複数指定も可）
- `--n-min 2 3 4 5`, `--n-max 12`: suffix-match の探索範囲
- `--ngram-n 2 3 4`: ngram-mod の固定n
- `--lengths 3 7 15`（別名 `--L`）: ドラフトステップ数。検証幅はそれぞれ4/8/16
- `--alpha 0.65 0.75 0.85`: MTPのステップ独立受理確率
- `--c1 7.46`, `--delta 0.5`, `--d 1.2`: ms単位のコストモデル
- `--sensitivity-delta 0.25 0.5 0.75 1.0`: 感度分析のδ
- `--hybrid-mtp-steps 3`: n-gram lookup miss時のMTPフォールバック長
- `--cross-conv`: 作成時刻順で、それ以前に処理済みの会話も検索コーパスへ加える
- `--also-cross-codex`: 通常のsession-local測定に加え、Codexだけglobal-windowで再生して同じreportへ追加
- `--conversations PATH`, `--codex-sessions PATH`: 片方だけでも併用でもよく、それぞれ反復指定可能
- `--codex-limit-files 200`: size条件を満たすCodex rolloutをmtimeの新しい順にN件選択
- `--max-file-mb 20`: これを超えるCodex JSONLを読まずにaccountingへ記録
- `--max-tokens-per-conv 60000`: session末尾を切り捨てるtoken上限。truncate session数を出力
- `--search-window-tokens 60000`: 各corpusのn-gram検索対象を直近N tokenへ制限
- `--out DIR`（必須）: 出力先。リポジトリ外のみ。`--allow-in-repo` でリポジトリ配下も許可（`report` も同じフラグを受け付ける）

`<out>/raw_stats.json` に設定、corpus別パーサ集計、truncate数、会話別token数、corpus／workload／model tag別の生集計を書き、`<out>/report.md` にcorpus別の表を生成します。`report` は保存済みJSONからMarkdownだけを再生成し、会話やtokenizerは読み直しません。

## 入力の解釈

### LM Studio (`corpus=lmstudio-chat`)

各 `messages[]` では `versions[currentlySelected]` だけを採用します。

- `singleStep`: `role` と `content[]` 内の `type: text` を順に連結
- `multiStep`: `role`、`senderInfo.senderName`、`steps[]` の `contentBlock` 内にある `type: text` を順に連結
- `style.type: thinking` のcontentBlock、tool call request/result、tool status、debug info、確認UIブロック、fileブロックは生成本文ではないため除外
- 未知type、壊れた選択、空本文はスキップし、警告カウンタと最大50件のサンプルをJSONへ保存
- トップレベル `systemPrompt` が非空ならsystemメッセージとして先頭に追加

壊れたJSONと本文のない会話はスキップし、件数を必ず残します。

### Codex CLI (`corpus=codex-agent`)

JSONLを記録順にstream parseします。実データで確認したcanonical `response_item` を使い、同じ内容をUI通知用に再掲する `event_msg` は二重計上しません。

- `session_meta.payload.base_instructions.text`、developer/user/system message、tool outputはlookup context
- `response_item.message(role=assistant)` の本文は生成decode位置
- `function_call.arguments` と `custom_tool_call.input` も生成decode位置。tool名は引数より前のlookup context
- `function_call_output` と `custom_tool_call_output` のtextはlookup context
- agent間 `response_item.agent_message` の平文もassistant生成として採用
- encrypted reasoning、world state、event bookkeeping、compaction metadataは本文から除外
- data URIおよび512文字以上のbase64 run、image/audio/encrypted blockは除外して件数を記録
- 未知outer record、未知response item、未知content typeはスキップしてcounterと最大50件のsampleへ記録

メッセージ／tool引数／tool出力はそれぞれ `add_special_tokens=False` で独立にtokenizeし、session内の生成順に連結します。CodexやLM Studio内部のchat template、tool-call特殊tokenそのものは再現しません。

## シミュレーション

LM Studio assistant本文、Codex assistant本文およびtool-call引数の各token位置をデコード位置とします。その位置を測定した後で正解tokenをコーパスへ追加するため、未来tokenはlookupに入りません。それ以外のcontextは順番に全tokenを追加します。

### suffix-match

現在履歴の末尾を `n_max` から `n_min` へ短くしながら検索します。最長の一致長を優先し、同じ長さに複数出現があれば最新の「後続tokenを持つ出現」を使います。各n-gram keyには最新2位置だけを保持します。最新位置が現在の末尾そのものなら後続がないため、1つ前の位置を選べばよく、全位置リストを走査せずに済みます。

### ngram-mod

固定nの `n-gram -> latest observed next token` 表をオンライン構築し、予測tokenを末尾へ足しながら同じ表を再帰参照して線形チェーンを作る、llama.cpp `spec-ngram-mod` の簡易近似です。確率分布、top-k分岐、複数候補は扱いません。

`--cross-conv` なしではsessionごとにindexをリセットします。ありの場合はcorpusごとにsession時刻、同値ならpath順に処理し、過去sessionを保持します。境界をまたぐtransitionとsource continuationはindexから除外します。LM StudioとCodexのcorpus同士は混ぜず、評価中sessionの未来も先読みしません。

`--also-cross-codex` は全corpusのsession-local run後にCodexだけをglobal-windowで再生し、`lookup_mode` が異なる追加統計行を同じJSON／Markdownへ保存します。SGLang NGRAMWorkerのserver-global corpusに近い比較用です。`--cross-conv` との同時指定は重複になるためエラーにします。

既定の検索窓は直近60,000 tokenです。index keyは最新2位置だけを保持し、窓外位置はlookup対象外です。保持tokenが窓の約150%へ達した時点で直近窓からindexを再構築するため、`--cross-conv` でも全session数に比例してメモリが増え続けません。再構築間の余剰index keyもlookup時の位置検査で除外され、検索範囲自体は60,000 tokenのままです。`--max-tokens-per-conv` は別の安全弁で、既定60,000 token以降のsession末尾を測定対象から外します。

## 統計

提案が1token以上あればhitです。hit時に実際のassistant継続と先頭から比較し、最初の不一致までを連続一致長とします。長さ `L` の検証で確定するtoken数は次です。

```text
confirmed = min(consecutive_match, L) + 1
```

hit率、hit時一致長のmean/median/p90/0..Lヒストグラム、全位置での期待確定token数を集計します。hybridは「提案なし」の位置だけMTPへフォールバックします。n-gramが提案した先頭tokenが不一致だった場合は同じ検証でMTPをやり直さず、確定1tokenとして扱います。MTPの期待値はステップごとに独立な受理率αの打ち切り幾何モデルです。

```text
E[MTP confirmed for K steps] = 1 + α + α² + ... + αᴷ
```

workload分類は再現可能な単純規則です。全メッセージ本文を連結し、コードフェンス数を文字数1000あたりへ正規化します。

- `code`: 完結したコードフェンスが1つ以上、かつフェンス密度が0.5/1000文字以上
- `mixed`: codeではないがコードフェンスがある、または12メッセージ以上
- `prose`: それ以外

## スループット射影

1行のtarget forwardを `c1`、検証の追加行を1行あたり `δ`、MTP draftを1stepあたり `d` とします。

- MTP-chain(K): 時間 `c1 + K(δ+d)`、token数 `1+α+...+αᴷ`
- n-gram専用(K): hit時だけwide検証するため平均時間 `c1 + hit_rate*K*δ`、missは素の1token
- hybrid: hit時はn-gram wide検証、no-hit時は指定step数のMTP draft+検証

レポートのn-gram `width=8/16` はdraft steps `7/15` に対応します。δとαを振った全体感度表も出します。

### ライブ実測へのcalibration recipe

自分のサーバーでMTP-chain(steps=3)の平均確定token数 `A` とスループット `T`(t/s)を公開ワークロードで測り、まず幾何モデルを合わせます。

```text
1 + α + α² + α³ = A
required total time = 1000 * A / T  [ms]
c1 + 3 * (δ + d) = required total time
```

最初の式から `--alpha` を求め、実測で分離できる `c1`、wide verification差分 `δ`、MTP draft差分 `d` を入力します。総時間式だけに合う組み合わせは一意ではないので、2値を固定して残り1値を総時間式から決めるか、M=1、wide verify、MTP draftを別々に測って3値を決めます。calibration後に `run` または保存済み設定を更新した再runを行い、MTP-3行が実測の `T` に近いことを確認します。

## 仮定と限界

これはライブ推論ベンチマークではありません。記録済み継続がtarget modelのgreedy経路であるとみなし、検証行コストを線形近似します。実際のkernel形状、batching、KV cache帯域、scheduler、index構築、chat template／特殊token境界、MTP受理の相関は含みません。

特にtemperatureが0より大きいライブ生成では、途中のsampleが1token変わるだけで後続履歴が変わります。記録済み正解を固定して最後までリプレイする本測定は、ライブ実行より楽観的になりえます。予測t/sは実装候補の順位付けとbreak-even検討用であり、本番性能の保証値ではありません。

## テスト

```bash
pytest -q
```

合成反復列／一意列、検索窓とsession境界、suffix-matchとngram-mod、LM Studio singleStep／multiStep、Codex message／tool call／tool output／base64除外／mtime制限、token truncate、corpus分割、手計算コストモデル、JSON／Markdown生成までをオフラインで検証します。品質確認には `ruff check .` も使用します。
