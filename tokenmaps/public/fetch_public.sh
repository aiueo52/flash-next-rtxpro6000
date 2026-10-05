#!/bin/bash
# fetch_public.sh [DEST] -- download the public corpora for the public 49,152-token draft map.
# Every file is pinned to a dataset revision (HF) or a release tag (GitHub); SHA256SUMS is written next
# to the files so a reader can check they got the same bytes. About 1.1 GB in total.
set -euo pipefail
DEST="${1:-$HOME/datasets/public-tokenmap}"; mkdir -p "$DEST/raw"; cd "$DEST/raw"
hf() {  # hf <repo> <revision> <path in repo> <local name>
  [ -s "$4" ] || curl -fL --retry 5 -C - -o "$4.part" "https://huggingface.co/datasets/$1/resolve/$2/$3" && { [ -s "$4" ] || mv "$4.part" "$4"; }
}
gh() {  # gh <owner/repo> <tag> <local name>
  [ -s "$3" ] || { curl -fL --retry 5 -o "$3.part" "https://codeload.github.com/$1/tar.gz/refs/tags/$2" && mv "$3.part" "$3"; }
}
# prose (CC BY-SA 3.0 / GFDL -- attribution: Wikipedia contributors, via wikimedia/wikipedia)
hf wikimedia/wikipedia b04c8d1ceb2f5cd4588862100d08de323dccfbaa 20231101.en/train-00007-of-00041.parquet enwiki-20231101-00007.parquet
hf wikimedia/wikipedia b04c8d1ceb2f5cd4588862100d08de323dccfbaa 20231101.ja/train-00007-of-00015.parquet jawiki-20231101-00007.parquet
# chat (MIT / Apache-2.0)
hf HuggingFaceH4/ultrachat_200k 8049631c405ae6576f93f445c6b8166f76f5505a data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet ultrachat-test_sft.parquet
hf llm-jp/oasst2-33k-ja df6f564ceaa48d3435a200f9f926aa9ab71768a9 oasst2-33k-ja.jsonl oasst2-33k-ja.jsonl
# reasoning traces (Apache-2.0)
hf open-thoughts/OpenThoughts-114k bd093c3994fd54d2390985b66988ddf282a55eb6 data/train-00005-of-00006.parquet openthoughts-00005.parquet
# agent / tool calls (CC BY 4.0 / Apache-2.0)
hf nebius/SWE-agent-trajectories 68195a1450865274106246d0d0296a1d6807b88e data/train-00000-of-00012.parquet swe-agent-traj-00000.parquet
hf NousResearch/hermes-function-calling-v1 dae3e1d28cfbcf4b915c04ea1e072030529b4bda func-calling.json hermes-func-calling.json
hf NousResearch/hermes-function-calling-v1 dae3e1d28cfbcf4b915c04ea1e072030529b4bda json-mode-agentic.json hermes-json-mode-agentic.json
# code (PSF-2.0, BSD-3-Clause, MIT, MIT, MIT, curl)
gh python/cpython v3.13.0 cpython-v3.13.0.tar.gz
gh golang/go go1.23.0 go-go1.23.0.tar.gz
gh facebook/react v18.3.1 react-v18.3.1.tar.gz
gh vuejs/core v3.5.12 vuejs-core-v3.5.12.tar.gz
gh tokio-rs/tokio tokio-1.40.0 tokio-1.40.0.tar.gz
gh curl/curl curl-8_10_1 curl-8_10_1.tar.gz
rm -f ./*.part
sha256sum $(ls | grep -v SHA256SUMS | sort) > SHA256SUMS
du -sh . ; cat SHA256SUMS
