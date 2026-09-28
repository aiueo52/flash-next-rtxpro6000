#!/usr/bin/env python3
"""Q1 quality audit runner. Usage: run_bench.py <arm> [bench,...]  (endpoint http://127.0.0.1:8001/v1)
Greedy, thinking off; per-item results (id, grade, token count, finish reason) in runs/<arm>/<bench>.jsonl; summary
runs/<arm>/summary.json. Prompts and raw model outputs are NOT written unless KEEP_TEXT=1 (local debugging only;
never publish such files). HumanEval failures record only the exception class, not stderr.
Any request error (retries exhausted, non-200 status, malformed response, missing choices/usage) or exception while
handling an item is recorded as finish="error" with the error class, counted in the benchmark's "errors", and makes the
run incomplete: summary.json keeps "incomplete": true (set at start, cleared only when every recorded benchmark has
zero errors) and the process exits 1 after writing everything it has. analyze.py refuses such arms."""
import hashlib, json, os, re, sys, time, collections, concurrent.futures as cf
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
EP = os.environ.get("EP", "http://127.0.0.1:8001/v1") + "/chat/completions"
BS = int(os.environ.get("BS", "1"))
CONC = min(int(os.environ.get("CONC", "8")), BS)
if BS < 1 or CONC < 1: raise SystemExit("BS and CONC must be positive integers")
ARM = sys.argv[1]; WHICH = sys.argv[2].split(",") if len(sys.argv) > 2 else ["gsm8k", "mmlu", "humaneval", "jcqa", "sanity"]
OUT = os.path.join(HERE, "runs", ARM); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE); import he_exec

LIMIT = int(os.environ.get("LIMIT", "0")); MOCK = os.environ.get("MOCK")
KEEP_TEXT = os.environ.get("KEEP_TEXT") == "1"
DROP = ("question", "choices", "prompt", "test", "canonical_solution", "answer")  # dataset text is not copied into results
RETRY_SLEEP = float(os.environ.get("RETRY_SLEEP", "5"))
class BadResponse(Exception):
    """The server answered, but not with a usable chat completion."""
def parse_completion(r):
    """(content, completion_tokens, finish_reason) from a response, or BadResponse (never the response text)."""
    status = getattr(r, "status_code", None)
    if status != 200: raise BadResponse(f"HTTP status {status}")
    j = r.json()
    if not isinstance(j, dict): raise BadResponse("response is not a JSON object")
    ch = j.get("choices")
    if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict): raise BadResponse("missing choices")
    m = ch[0].get("message")
    if not isinstance(m, dict) or not isinstance(m.get("content") or "", str): raise BadResponse("missing message")
    fr = ch[0].get("finish_reason")
    if not isinstance(fr, str) or not fr: raise BadResponse("missing finish_reason")
    usage = j.get("usage")
    ct = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(ct, int) or isinstance(ct, bool) or ct < 0: raise BadResponse("missing usage.completion_tokens")
    return (m.get("content") or ""), ct, fr
def chat(prompt, max_tokens, retries=3):
    """(content, completion_tokens, finish_reason, error); error is None on success, else the error class name
    and finish_reason is "error"."""
    if MOCK: return ("```python\n    return 1\n```\nThe answer is B.\n#### 18", 7, "stop", None)
    body = {"model": "flash-next", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "top_k": 1, "chat_template_kwargs": {"enable_thinking": False}}
    err = None
    for a in range(retries):
        try:
            return (*parse_completion(requests.post(EP, json=body, timeout=900)), None)
        except Exception as e:
            err = e
            if a + 1 < retries: time.sleep(RETRY_SLEEP)
    name = type(err).__name__ + (f": {err}" if isinstance(err, BadResponse) else "")
    return "", 0, "error", name[:120]
def output_hash(text):
    """sha256 of the model output: lets analyze.py compare outputs across arms without storing text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def error_row(r, e):
    """Mark an item as failed by an exception (grading/sandbox infrastructure, bad data); never scored."""
    r.update(finish="error", error=type(e).__name__[:80], correct=0)
    return r

def load(name):
    rows = [json.loads(l) for l in open(os.path.join(D, name + ".jsonl"))]
    return rows[:LIMIT] if LIMIT else rows
def run(name, rows, make_prompt, max_tokens, grade):
    t0 = time.time(); res = [None] * len(rows)
    def work(i):
        r = {k: v for k, v in rows[i].items() if k not in DROP}; r.setdefault("gen_tokens", 0)
        try:
            p = make_prompt(rows[i]); out, ct, fr, err = chat(p, max_tokens)
            if KEEP_TEXT: r.update(prompt=p, output=out)
            r.update(gen_tokens=ct, finish=fr)
            if not err: r["output_sha256"] = output_hash(out)   # identity across arms without keeping the text
            if err: r.update(error=err, correct=0)       # a failed request is not graded
            else: r.update(grade(rows[i], out))
        except Exception as e:
            error_row(r, e)
        return i, r
    with cf.ThreadPoolExecutor(CONC) as ex:
        for i, r in ex.map(work, range(len(rows))): res[i] = r
    wall = time.time() - t0
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(res); k = sum(r["correct"] for r in res); mt = sum(r["gen_tokens"] for r in res) / max(1, n)
    s = {"n": n, "correct": k, "acc": k / n, "wall_s": round(wall, 1), "mean_gen_tokens": round(mt, 1),
         "errors": sum(r["finish"] == "error" for r in res), "truncated": sum(r["finish"] == "length" for r in res)}
    print(f"[{ARM}] {name}: {k}/{n} = {100*k/n:.2f}%  wall {wall:.0f}s  mean_tokens {mt:.0f}  trunc {s['truncated']} err {s['errors']}", flush=True)
    return s

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def gsm_extract(out):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", out)
    if m: s = m.group(1)
    else:
        m = re.search(r"(?:answer is|Answer:|answer:)\s*\$?\s*(-?[\d,]*\.?\d+)", out)
        if m: s = m.group(1)
        else:
            a = NUM.findall(out); s = a[-1] if a else ""
    s = s.replace(",", "").rstrip(".")
    try: return str(int(float(s))) if float(s) == int(float(s)) else s
    except Exception: return s
def grade_gsm(r, out):
    p = gsm_extract(out); return {"pred": p, "correct": int(p == r["answer"])}
def grade_letter(letters):
    def g(r, out):
        m = re.search(rf"\b([{letters}])\b", out.strip().replace("*", " "))
        p = m.group(1) if m else ""; return {"pred": p, "correct": int(p == r["answer"])}
    return g

summary = {"arm": ARM, "started": time.strftime("%F %T"), "conc": CONC, "bench": {}}
sp = os.path.join(OUT, "summary.json")
if os.path.exists(sp): summary = json.load(open(sp))
summary.update(bs=BS, conc=CONC, incomplete=True)   # cleared at the very end only if nothing failed
summary.pop("finished", None)
def save(): json.dump(summary, open(sp, "w"), indent=1, ensure_ascii=False)
save()

if "gsm8k" in WHICH:
    summary["bench"]["gsm8k"] = run("gsm8k", load("gsm8k"),
        lambda r: r["question"] + "\n\nSolve the problem step by step, then write the final answer on the last line in the form '#### <number>'.",
        512, grade_gsm); save()
if "mmlu" in WHICH:
    def mk(r):
        ch = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(r["choices"]))
        return (f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n{r['question']}\n{ch}\n\n"
                "Answer with only the letter (A, B, C, or D) of the correct choice.")
    summary["bench"]["mmlu"] = run("mmlu", load("mmlu"), mk, 16, grade_letter("ABCD")); save()
if "humaneval" in WHICH:
    def grade_he(r, out):
        src = he_exec.build_program(r["prompt"], out, r["test"], r["entry_point"]); st, err = he_exec.run_program(src, 10.0)
        return {"status": st, "err": err, "correct": int(st == "pass")}
    summary["bench"]["humaneval"] = run("humaneval", load("humaneval"),
        lambda r: "Complete the following Python function. Return the complete function (with its signature and any needed imports) in a single ```python code block, no tests.\n\n```python\n" + r["prompt"] + "```",
        768, grade_he); save()
if "jcqa" in WHICH:
    def mkj(r):
        ch = "\n".join(f"{'ABCDE'[i]}. {c}" for i, c in enumerate(r["choices"]))
        return f"次の質問に対して最も適切な選択肢を選んでください。\n\n質問: {r['question']}\n{ch}\n\n正しい選択肢の記号（A〜E）だけを答えてください。"
    summary["bench"]["jcqa"] = run("jcqa", load("jcqa"), mkj, 16, grade_letter("ABCDE")); save()
if "sanity" in WHICH:
    W = os.path.join(HERE, "..", "..", "workloads")
    prompts = {"prose-en": open(os.path.join(W, "prose-en.txt")).read(), "prose-ja": open(os.path.join(W, "prose-ja.txt")).read(),
               "code-edit": open(os.path.join(W, "code-edit.txt")).read(), "agent-loop": open(os.path.join(W, "agent-loop.txt")).read(),
               "essay": "Write a detailed, well-structured essay of about 1500 words on how lighthouses were automated during the 20th century, covering technology, labour, and safety. Use varied vocabulary and avoid repeating phrases."}
    def metrics(out):
        run_ = max((len(m.group(0)) for m in re.finditer(r"(.)\1+", out)), default=1)
        # An output too short to have words / 4-grams is not measured (None), never scored as 0 or 1.
        words = out.split(); uniq = len(set(words)) / len(words) if words else None
        toks = re.findall(r"\w+|[^\w\s]", out); g = [tuple(toks[i:i+4]) for i in range(max(0, len(toks) - 3))]
        rep4 = 1 - len(set(g)) / len(g) if g else None
        return {"maxrun": run_, "uniq_words": None if uniq is None else round(uniq, 3),
                "rep_4gram": None if rep4 is None else round(rep4, 3)}
    t0 = time.time(); rows = []
    def work(kv):
        k, p = kv; r = {"id": k, "gen_tokens": 0, "finish": "error", "maxrun": None, "uniq_words": None, "rep_4gram": None}
        try:
            out, ct, fr, err = chat(p, 1500); r.update(gen_tokens=ct, finish=fr)
            if err: r["error"] = err
            if KEEP_TEXT: r.update(prompt=p, output=out)
            if not err: r.update(metrics(out))
        except Exception as e:
            error_row(r, e); r.pop("correct")
        return r
    with cf.ThreadPoolExecutor(min(5, CONC)) as ex: rows = list(ex.map(work, list(prompts.items())[:LIMIT] if LIMIT else prompts.items()))
    with open(os.path.join(OUT, "sanity.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary["bench"]["sanity"] = {"wall_s": round(time.time() - t0, 1), "errors": sum(r["finish"] == "error" for r in rows), "items": {r["id"]: {k: r[k] for k in ("gen_tokens", "finish", "maxrun", "uniq_words", "rep_4gram")} for r in rows}}
    for r in rows: print(f"[{ARM}] sanity {r['id']}: tokens {r['gen_tokens']} maxrun {r['maxrun']} uniq {r['uniq_words']} rep4 {r['rep_4gram']}", flush=True)
    save()
errors = {b: v.get("errors", 0) for b, v in summary["bench"].items() if v.get("errors", 0)}
summary.update(errors=sum(errors.values()), incomplete=bool(errors), finished=time.strftime("%F %T")); save()
if errors:
    print(f"[{ARM}] INCOMPLETE: request/item errors in {errors}; results written but the run failed", file=sys.stderr, flush=True)
    sys.exit(1)
