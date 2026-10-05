"""Display/other-process GPU load per arm phase, from `nvidia-smi pmon -s u -d 5 -o DT` and an ABBA log.

Usage: pmon_phase.py <pmon.log> <abba.log> [<abba.log> ...]
Graphics ("G") processes share the GPU by time-slicing; their SM% is the interference covariate for an arm (the
greedy B1/A1 traces showed it as 0.2-1.3 ms stalls inside otherwise unchanged kernels). Phases: lmstudio = arm
start .. "fnbench lmstudio rc=", greedy = that .. "fnbench greedy rc=".
"""

import re
import sys
from datetime import datetime


def load_pmon(path):
    # Every sample time counts, so a sample where no graphics process ran contributes 0.
    samples = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 6 or line.startswith("#"):
            continue
        stamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y%m%d %H:%M:%S")
        per_name = samples.setdefault(stamp, {})
        if parts[4] == "G" and parts[5] != "-":
            per_name[parts[-1]] = per_name.get(parts[-1], 0) + int(parts[5])
    return samples


def arm_phases(path, day):
    marks = {}
    for line in open(path):
        match = re.match(r"\[(\w+)\] (start|fnbench lmstudio rc=\d+|fnbench greedy rc=\d+) ?(\S*)", line)
        if not match:
            continue
        label, what = match.group(1), match.group(2).split(" rc=")[0]
        clock = match.group(3) if what == "start" else line.split()[-1]
        marks.setdefault(label, {})[what] = datetime.strptime(f"{day} {clock}", "%Y%m%d %H:%M:%S")
    return marks


def main():
    samples = load_pmon(sys.argv[1])
    if not samples:
        print("no graphics samples")
        return
    day = min(samples).strftime("%Y%m%d")
    print(f"{'arm':4s} {'phase':9s} {'samples':>7s} {'G sm% mean':>10s} {'p90':>5s}  top processes")
    for abba in sys.argv[2:]:
        for label, marks in arm_phases(abba, day).items():
            bounds = [("lmstudio", "start", "fnbench lmstudio"), ("greedy", "fnbench lmstudio", "fnbench greedy")]
            for phase, begin, end in bounds:
                if begin not in marks or end not in marks:
                    continue
                window = [stamp for stamp in samples if marks[begin] <= stamp <= marks[end]]
                if not window:
                    continue
                totals = sorted(sum(samples[stamp].values()) for stamp in window)
                per_name = {}
                for stamp in window:
                    for name, value in samples[stamp].items():
                        per_name[name] = per_name.get(name, 0) + value
                top = sorted(per_name.items(), key=lambda item: -item[1])[:3]
                print(f"{label:4s} {phase:9s} {len(window):7d} {sum(totals) / len(totals):10.1f} "
                      f"{totals[int(0.9 * (len(totals) - 1))]:5d}  "
                      + ", ".join(f"{name} {value / len(window):.1f}" for name, value in top))


if __name__ == "__main__":
    main()
