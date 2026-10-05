"""post_summary.py <post.log> [<abba.log>]: one line per ANCOVA fit from bench/rs2d/post_abba.sh's log (and the pooled
ANCOVA in abba_rs2d.sh's log), ms/token turned into t/s (1 / (1 + x) - 1, so the bounds swap) as RS2 §6 reads it.
"""
import re
import sys

NUMBER = r"([+-]\d+\.\d+)"
EFFECT = re.compile(rf"^  (ms/token|tok/step|ms/step\|a)\s+B effect\s+{NUMBER}%\s+\[\s*{NUMBER} \.\.\s+{NUMBER}\]"
                    r"(.*clock elasticity (\S+) \(se (\S+)\))?")
ARM = re.compile(rf"arm level B effect\s+{NUMBER}%\s+\[\s*{NUMBER} \.\.\s+{NUMBER}\]")
CLOCK = re.compile(rf"B - A clock\s+{NUMBER}%\s+\[\s*{NUMBER} \.\.\s+{NUMBER}\]")


def as_tps(effect, low, high):
    def flip(x):
        return 100 * (1 / (1 + x / 100) - 1)
    return flip(effect), flip(high), flip(low)


def sections(path, pooled):
    # abba_rs2d.sh's log: "== ancova <mode>" opens a pooled section; post_abba.sh's log: "#### <name>"
    section = None
    for line in open(path):
        match = re.match(r"^== ancova (\w+)$", line)
        if pooled and match:
            section = f"ancova {match.group(1)} / pooled"
        elif pooled and line.startswith("== clipped"):
            section = None
        elif not pooled and line.startswith("#### "):
            section = line[5:].strip()
        if section:
            yield section, line


def main():
    lines = list(sections(sys.argv[2], pooled=True)) if len(sys.argv) > 2 else []
    lines += list(sections(sys.argv[1], pooled=False))
    rows = []
    for section, line in lines:
        match = EFFECT.match(line)
        if match:
            fit = match.group(1)
            values = [float(match.group(i)) for i in (2, 3, 4)]
            name = "t/s" if fit == "ms/token" else fit
            if fit == "ms/token":
                values = as_tps(*values)
            extra = f"  elasticity {match.group(6)} (se {match.group(7)})" if match.group(5) else ""
            rows.append([section, name, values, None, extra])
            continue
        match = ARM.search(line)
        if match and rows and rows[-1][3] is None:
            values = [float(match.group(i)) for i in (1, 2, 3)]
            rows[-1][3] = as_tps(*values) if rows[-1][1] == "t/s" else values
            continue
        match = CLOCK.search(line)
        if match:
            rows.append([section, "B-A clock", [float(match.group(i)) for i in (1, 2, 3)], None, ""])
    for section, name, (effect, low, high), arm, extra in rows:
        arm_text = f"  arm {arm[0]:+6.2f} [{arm[1]:+6.2f} .. {arm[2]:+6.2f}]" if arm else ""
        print(f"{section:34s} {name:9s} {effect:+6.2f} [{low:+6.2f} .. {high:+6.2f}]{arm_text}{extra}")


if __name__ == "__main__":
    main()
