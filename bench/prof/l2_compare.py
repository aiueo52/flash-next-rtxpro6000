"""Compare validate.sh prof traces for the L2 flags: median step wall + exclusive us/step of the
families the flags touch. Usage: python prof/l2_compare.py <profile w16|w4> <label>... (labels of
prof/traces/<label>-<profile>-{code-edit,prose-en})."""
import csv, io, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAMS = {
    "draft lm_head": ("draft", "_w8a16_gemv_kernel", "[1536,1,1]"),
    "draft_extend lm_head": ("draft_extend", "_w8a16_gemv_kernel", "[1536,1,1]"),
    "verify GEMM1": ("verify", "cutlass_moe_grouped_gemm1", None),
    "verify GEMM2": ("verify", "cutlass_moe_grouped_gemm2", None),
    "verify gemv (all)": ("verify", "_w8a16_gemv_kernel", None),
    "verify touch_table": ("verify", "_touch_table_kernel", None),
    "draft_extend touch_table": ("draft_extend", "_touch_table_kernel", None),
    "verify hc_up": ("verify", "_hc_up_kernel", None),
}


def run(trace_dir):
    out = subprocess.run([sys.executable, os.path.join(HERE, "exclusive_time.py"), trace_dir, "--top", "1",
                          "--csv", "/dev/stdout"], capture_output=True, text=True, check=True).stdout  # a failed trace must not become a NaN row
    m = re.search(r"MEDIAN step:\s+wall=\s*([\d.]+)\s+exclusive=\s*([\d.]+)", out)
    wall = float(m.group(1)) if m else float("nan")
    rows = [r for r in csv.DictReader(io.StringIO(out[out.index("trace,phase,family"):])) if r.get("trace")] if "trace,phase,family" in out else []
    fam = {}
    for name, (phase, pat, grid) in FAMS.items():
        tot_e = tot_r = 0.0; n = 0.0; meds = []
        for r in rows:
            if r["phase"] == phase and pat in r["family"] and (grid is None or r["grid"] == grid):
                tot_e += float(r["excl_us_per_step"]); tot_r += float(r["raw_us_per_step"]); n += float(r["n_per_step"]); meds.append((float(r["n_per_step"]), float(r["median_us"])))
        fam[name] = (tot_e, tot_r, n, max(meds)[1] if meds else 0.0)
    return wall, fam


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: l2_compare.py <w16|w4> <label>...  -- needs prof/traces/<label>-<profile>-{code-edit,prose-en} "
                 "torch-profiler traces (validate.sh MODE=prof); traces are not published in this repository")
    prof, labels = sys.argv[1], sys.argv[2:]
    missing = [f"{lab}-{prof}-{wl}" for wl in ("code-edit", "prose-en") for lab in labels
               if not os.path.isdir(os.path.join(HERE, "traces", f"{lab}-{prof}-{wl}"))]
    if missing:
        sys.exit("l2_compare.py: missing traces (not published in this repository): " + ", ".join(missing))
    for wl in ("code-edit", "prose-en"):
        print(f"\n== {prof} {wl} ==")
        base = None
        for lab in labels:
            d = os.path.join(HERE, "traces", f"{lab}-{prof}-{wl}")
            if not os.path.isdir(d):
                print(f"  {lab:14s} (no trace)"); continue
            wall, fam = run(d)
            if base is None:
                base = (wall, fam)
            dw = wall - base[0]
            line = f"  {lab:14s} wall {wall:8.0f} us ({dw:+6.0f}, {100*dw/base[0]:+5.1f}%)"
            for name, (e, r, n, med) in fam.items():
                if e or r:
                    be = base[1][name][0]
                    line += f" | {name}: excl {e:6.0f} ({e-be:+5.0f}) n {n:4.1f} med {med:6.1f}"
            print(line)


if __name__ == "__main__":
    main()
