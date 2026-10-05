"""fc-moe: prologue (and chain) table from a session jsonl (s5.jsonl, s6.jsonl): one row per arm, warm and cold
prologue medians per T, with the p10-p90 spread; mirrored arms (x and xb) are shown separately and averaged.
With the prologue key it ends with a per-step estimate against prod for the in-server call mix (FC_fc-moe 4b.1):
  W4:     49 verify calls at T=4, 43% of them warm, + 2 draft calls at T=1 (cold);
  W16/wa: 49 verify calls at T=16 (in-server between the warm and the cold number: a range) + 14 draft calls at T=1.

  python bench/fc_moe/s_table.py runs/fc_moe/s5.jsonl [--key prologue|wall_us|chain_span]
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", default="prologue")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path) if l.startswith("{")]
    tab = collections.defaultdict(dict)  # (arm, mode) -> T -> row
    means = {}  # mode -> arm -> T -> mean of the mirrored arms
    Ts = sorted({r["T"] for r in rows})
    for r in rows:
        arm, _, mode = r["label"].rpartition("_")
        tab[(arm, mode)][r["T"]] = r
    for mode in ("warm", "cold"):
        print(f"\n{mode}: {a.key} median us (p10-p90)")
        print(f"{'arm':8s}" + "".join(f"{'T=' + str(t):>22s}" for t in Ts))
        base = {}
        for (arm, m), byT in tab.items():
            if m != mode:
                continue
            cells = []
            for t in Ts:
                r = byT.get(t)
                if r is None:
                    cells.append(f"{'-':>22s}")
                    continue
                v = r[a.key]
                spread = (f" ({r['prologue_p10']:.2f}-{r['prologue_p90']:.2f})" if a.key == "prologue"
                          else "")
                cells.append(f"{v:>8.2f}{spread:>14s}")
                base.setdefault(arm.rstrip("b"), {}).setdefault(t, []).append(v)
            print(f"{arm:8s}" + "".join(cells))
        print("mean of mirrored arms:")
        for arm, byT in base.items():
            print(f"{arm:8s}" + "".join(f"{sum(byT[t]) / len(byT[t]):>22.2f}" if t in byT else f"{'-':>22s}"
                                        for t in Ts))
        means[mode] = {arm: {t: sum(v) / len(v) for t, v in byT.items()} for arm, byT in base.items()}
    if a.key != "prologue" or "prod" not in means.get("warm", {}) or "prod" not in means.get("cold", {}):
        return
    w, c = means["warm"], means["cold"]
    print("\nper-step estimate against prod (ms; negative = faster)")
    print(f"{'arm':8s}{'W4':>10s}{'W16/wa':>18s}")
    for arm in w:
        if arm == "prod" or arm not in c:
            continue
        try:
            d = {m: {t: means[m][arm][t] - means[m]["prod"][t] for t in (1, 4, 16)} for m in ("warm", "cold")}
        except KeyError:
            continue
        w4 = 49 * (0.57 * d["cold"][4] + 0.43 * d["warm"][4]) + 2 * d["cold"][1]
        w16 = sorted(49 * d[m][16] + 14 * d["cold"][1] for m in ("warm", "cold"))
        print(f"{arm:8s}{w4 / 1000:>10.3f}{w16[0] / 1000:>9.3f}..{w16[1] / 1000:.3f}")


if __name__ == "__main__":
    main()
