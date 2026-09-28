#!/usr/bin/env python3
"""Materialise fixed benchmark subsets as JSONL under bench/quality/data (run once; seeded)."""
import json, os, random, sys, urllib.request
from datasets import load_dataset
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(D, exist_ok=True)
def dump(name, rows):
    p = os.path.join(D, name + ".jsonl")
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), p)

if not os.path.exists(os.path.join(D, "gsm8k.jsonl")):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    dump("gsm8k", [{"id": i, "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(ds)])

SUBJECTS = ["abstract_algebra", "anatomy", "astronomy", "college_computer_science", "college_mathematics",
            "high_school_biology", "high_school_chemistry", "high_school_physics", "high_school_world_history",
            "machine_learning", "moral_scenarios", "philosophy", "professional_law", "world_religions"]
if not os.path.exists(os.path.join(D, "mmlu.jsonl")):
    rows = []
    for s in SUBJECTS:
        ds = load_dataset("cais/mmlu", s, split="test")
        idx = list(range(len(ds))); random.Random(0).shuffle(idx); idx = sorted(idx[:100])
        for i in idx:
            r = ds[i]
            rows.append({"id": f"{s}/{i}", "subject": s, "question": r["question"], "choices": r["choices"], "answer": "ABCD"[r["answer"]]})
    dump("mmlu", rows)

if not os.path.exists(os.path.join(D, "humaneval.jsonl")):
    ds = load_dataset("openai/openai_humaneval", split="test")
    dump("humaneval", [{"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]} for r in ds])

if not os.path.exists(os.path.join(D, "jcqa.jsonl")):
    url = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jcommonsenseqa-v1.3/valid-v1.3.json"
    raw = urllib.request.urlopen(url, timeout=120).read().decode()
    rows = []
    for line in raw.splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        rows.append({"id": r["q_id"], "question": r["question"], "choices": [r[f"choice{k}"] for k in range(5)], "answer": "ABCDE"[int(r["label"])]})
    dump("jcqa", rows)
