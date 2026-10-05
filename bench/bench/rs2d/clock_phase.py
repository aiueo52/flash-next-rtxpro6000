"""GPU clocks, power and throttle state per arm phase, from bench/rs2d/abba_after_queue.sh's clocks.csv
(`nvidia-smi --query-gpu=... --format=csv -l 5`) and an ABBA log.

Usage: clock_phase.py <clocks.csv> <abba.log> [<abba.log> ...]
The RS2 ABBA's verify graph span rose 35% over one evening on the same code, so SM and memory clocks per phase are
the drift covariates. Phases as in bench/rs2/pmon_phase.py, except that lmstudio starts at "server up" (start-up and
graph capture excluded). "cap%" is the share of the phase spent at the SW power cap, from the driver's counter.
"""

import csv
import re
import sys
from datetime import datetime

sys.path.insert(0, "bench/rs2")
from pmon_phase import arm_phases  # noqa: E402


def load_clocks(path):
    rows = csv.reader(open(path), skipinitialspace=True)
    names = [name.split(" [")[0] for name in next(rows)]
    samples = []
    for row in rows:
        if len(row) != len(names):
            continue
        values = dict(zip(names, row))
        values["timestamp"] = datetime.strptime(values["timestamp"].split(".")[0], "%Y/%m/%d %H:%M:%S")
        samples.append(values)
    return samples


def number(text):
    return float(text.split()[0])


def main():
    samples = load_clocks(sys.argv[1])
    if not samples:
        print("no clock samples")
        return
    day = samples[0]["timestamp"].strftime("%Y%m%d")
    print(f"{'arm':4s} {'phase':9s} {'samples':>7s} {'sm MHz':>7s} {'min':>5s} {'mem MHz':>7s} {'W':>6s} "
          f"{'degC':>5s} {'cap%':>5s}  pstates")
    for abba in sys.argv[2:]:
        phases = arm_phases(abba, day)
        for line in open(abba):
            match = re.match(r"\[(\w+)\] server up (\S+)", line)
            if match and match.group(1) in phases:
                phases[match.group(1)]["server up"] = datetime.strptime(f"{day} {match.group(2)}", "%Y%m%d %H:%M:%S")
        for label, marks in phases.items():
            bounds = [("lmstudio", "server up", "fnbench lmstudio"), ("greedy", "fnbench lmstudio", "fnbench greedy")]
            for phase, begin, end in bounds:
                if begin not in marks or end not in marks:
                    continue
                window = [sample for sample in samples if marks[begin] <= sample["timestamp"] <= marks[end]]
                if len(window) < 2:
                    continue
                count = len(window)
                seconds = (window[-1]["timestamp"] - window[0]["timestamp"]).total_seconds()
                capped_us = (number(window[-1]["clocks_event_reasons_counters.sw_power_cap"])
                             - number(window[0]["clocks_event_reasons_counters.sw_power_cap"]))
                states = {}
                for sample in window:
                    states[sample["pstate"]] = states.get(sample["pstate"], 0) + 1
                print(f"{label:4s} {phase:9s} {count:7d} "
                      f"{sum(number(s['clocks.current.sm']) for s in window) / count:7.0f} "
                      f"{min(number(s['clocks.current.sm']) for s in window):5.0f} "
                      f"{sum(number(s['clocks.current.memory']) for s in window) / count:7.0f} "
                      f"{sum(number(s['power.draw']) for s in window) / count:6.1f} "
                      f"{sum(number(s['temperature.gpu']) for s in window) / count:5.1f} "
                      f"{100 * capped_us / 1e6 / seconds:5.1f}  "
                      + " ".join(f"{state} {n}" for state, n in sorted(states.items())))


if __name__ == "__main__":
    main()
