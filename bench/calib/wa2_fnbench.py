#!/usr/bin/env python3
"""Run the C1 fnbench CLI unchanged; record exact request wall-clock boundaries."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fnbench.http_client import OpenAIStreamClient
from fnbench.cli import main

original = OpenAIStreamClient.complete
calls = 0
workloads = ['code-edit', 'prose-en', 'prose-ja', 'agent-loop']
events = open(os.environ['WA2_EVENTS'], 'x', buffering=1)

def measured(self, *args, **kwargs):
    global calls
    i = calls
    calls += 1
    start = time.time()
    result = original(self, *args, **kwargs)
    end = time.time()
    events.write(json.dumps(dict(workload=workloads[i // 4], repeat=i % 4,
                                 start=start, end=end)) + '\n')
    return result

OpenAIStreamClient.complete = measured
raise SystemExit(main())
