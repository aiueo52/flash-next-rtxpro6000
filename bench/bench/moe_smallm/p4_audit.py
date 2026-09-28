"""Read-only final evidence audit and private reconstruction of the P4 patch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT.parent / "sglang-rtxpro6000"
PRIVATE = ROOT.parent / "flashinfer-p4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    args = ap.parse_args()
    if not (ROOT / "runs/p4/production-hashes.json").is_file():
        raise SystemExit("p4_audit.py: needs runs/p4/production-hashes.json and the P4 run directories, which are "
                         "not published in this repository")
    original = json.loads((ROOT / "runs/p4/production-hashes.json").read_text())
    assert original, "production-hashes.json lists no files: nothing to audit"
    site = next((PROD / ".venv/lib").glob("python*/site-packages"))
    production = {}
    patch = ROOT / "bench/moe_smallm/patches/p4-contrib-prune.patch"
    with tempfile.TemporaryDirectory(prefix="p4-patch-audit-") as tmp:
        tmp = Path(tmp)
        for name, expected in original.items():
            source = (site if name.startswith("flashinfer/") else PROD) / name
            assert sha(source) == expected, source
            production[name] = expected
            if name.startswith("flashinfer/"):
                target = tmp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                assert source.stat().st_ino != (PRIVATE / name).stat().st_ino
        subprocess.run(["patch", "-s", "-p1", "-i", str(patch)], cwd=tmp, check=True)
        for name in original:
            if name.startswith("flashinfer/"):
                assert (tmp / name).read_bytes() == (PRIVATE / name).read_bytes(), name
    manifest = json.loads((ROOT / "prune/p4-manifest.json").read_text())
    assert manifest["widths"], "p4-manifest.json lists no tables: nothing to audit"
    for item in manifest["widths"].values():
        assert sha(ROOT / "prune" / item["file"]) == item["sha256"]
    arms = []
    for width in (4, 16):
        for arm in ("control", "contrib", "control2"):
            path = ROOT / "runs/p4" / f"{args.label}-w{width}-{arm}"
            assert (path / "COMPLETE").exists(), path
            summary = json.loads((path / "arm-summary.json").read_text())
            assert summary["needle_pass"] == summary["needle_total"] == 3
            env = json.loads((path / "env.json").read_text())
            assert env["MEM_FRACTION"] == env["W16_MEM_FRACTION"] == env["WA_MEM_FRACTION"] == "0.920"
            assert env["SERVE_DISPLAY_HZ"] == ""
            assert env["SGLANG_CACHE_DIR"] == "/home/user/.cache/sglang-p4"
            assert (env.get("SGLANG_MOE_PRUNE_POLICY") == "contrib") == (arm == "contrib")
            assert len(list((path / "timing").glob("*-r[012].jsonl"))) == 12
            assert len(list((path / "timing").glob("trace-*/*.gz"))) == 6
            assert len(list((path / "census").glob("*.npz"))) == 4
            minima = {}
            for stage in ("timing", "census"):
                folder = path / stage
                assert not (folder / "memory-failed.txt").exists()
                free = [int(x) for x in re.findall(r"free_MiB=(\d+)", (folder / "memory.log").read_text())]
                assert free and min(free) >= 4096, (folder, free)
                minima[stage] = min(free)
            arms.append(dict(width=width, arm=arm, min_free_MiB=minima, needle="3/3"))
    result = dict(status="PASS", label=args.label, arms=arms,
                  production_hashes_unchanged=production, patch_sha256=sha(patch),
                  patch_reconstructs_private_files=True, no_production_hardlinks=True,
                  calibration_hashes_valid=True, measured_requests=72, traces=36,
                  census_files=24, needle="18/18")
    output = ROOT / "runs/p4" / f"{args.label}-audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
