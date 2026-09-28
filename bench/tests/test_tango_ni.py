"""Non-inferiority bounds use the Tango (1998) score interval in every published copy.

A Wald interval collapses with few or zero discordant pairs; Q1 HumanEval (1 favourable, 0 unfavourable of 164)
was once reported as a 0.5 pp non-inferiority PASS because of that.
"""
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "bench"
COPIES = ["quality/analyze.py", "hc1/quality/analyze.py", "hc1/report.py", "moe_smallm/p4_quality_report.py"]
Z = 1.6448536269514722


def load(rel):
    src = (ROOT / rel).read_text()
    m = re.search(r"^def tango_bounds.*?\n    return lower, upper\n", src, re.S | re.M)
    assert m, rel
    ns = {"math": math}
    exec(m.group(0), ns)
    return m.group(0), ns["tango_bounds"]


def test_all_copies_identical_and_no_wald():
    texts = {load(rel)[0] for rel in COPIES}
    assert len(texts) == 1
    for rel in COPIES:
        code = "\n".join(l for l in (ROOT / rel).read_text().splitlines() if not l.lstrip().startswith(("#", '"')))
        assert "math.sqrt((win+loss)" not in code and "diff-1.6448536269514722*se" not in code, rel


@pytest.mark.parametrize("rel", COPIES)
def test_one_zero_of_164_is_inconclusive(rel):
    lo, hi = (100 * v for v in load(rel)[1](1, 0, 164, Z))
    assert lo == pytest.approx(-1.023, abs=5e-3) and hi == pytest.approx(2.687, abs=5e-3)
    assert not lo > -0.5  # not a 0.5 pp PASS


def test_zero_discordance_symmetric_and_shrinks_with_n():
    tb = load(COPIES[0])[1]
    lo, hi = tb(0, 0, 164, Z)
    assert lo == pytest.approx(-hi) and 100 * hi == pytest.approx(1.623, abs=5e-3)
    assert tb(0, 0, 5000, Z)[0] > -0.005 > tb(0, 0, 100, Z)[0]


def test_bounds_bracket_estimate_and_monotone_in_z():
    tb = load(COPIES[0])[1]
    for b, c, n in [(60, 53, 1319), (0, 53, 164), (38, 66, 1400), (7, 0, 20)]:
        lo, hi = tb(b, c, n, Z)
        lo2, hi2 = tb(b, c, n, 1.959963984540054)
        assert -1 <= lo2 <= lo <= (b - c) / n <= hi <= hi2 <= 1
