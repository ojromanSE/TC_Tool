"""Tests for aries_export.py — ARIES lookup rows and economic lines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aries_export as A
import core
import se_report as R

D_LIM = core.D_LIM_DEFAULT


def test_reproduces_a_row_of_the_se_typecurves_lookup():
    """AVLN-ECHO in the SE lookup: QI 29664.4 B/M, B/1.2, DI_SEC 78.47, DMIN 4.88
    -> switch after 197.6 months, tail 402.4 months (600 in total)."""
    di = A._effective_to_nominal(0.7847, 1.2)
    d_lim = A._effective_to_nominal(0.0488)
    s = A.segments(29664.4 / A.DAYS_PER_MONTH, 1.2, di, d_lim)
    assert s.fields() == ["29664.4", "197.6", "B/1.2", "78.47", "402.4", "4.88"]


def test_conversions_round_trip():
    for b in (0.0, 0.5, 1.0, 1.2, 1.8):
        for de in (0.05, 0.3, 0.7847, 0.95):
            assert A._nominal_to_effective(A._effective_to_nominal(de, b), b) == pytest.approx(de, rel=1e-12)
    # the app's terminal decline (5 %/yr effective) is 5.00 in the lookup
    assert A._nominal_to_effective(D_LIM) * 100 == pytest.approx(5.0, abs=1e-9)


def test_b_is_written_with_four_significant_figures():
    assert [A.fmt_b(b) for b in (1.2, 1.127, 0.93294, 0.8053, 1.19304, 0.8)] == \
           ["1.2", "1.127", "0.9329", "0.8053", "1.193", "0.8"]


@pytest.mark.parametrize("qi,b,di", [(400.0, 1.1, 0.12), (1500.0, 0.8, 0.3), (90.0, 1.5, 0.05),
                                     (2200.0, 1.0, 0.2), (300.0, 0.3, 0.02)])
def test_aries_volume_matches_the_model(qi, b, di):
    """The volume ARIES integrates from the exported (rounded) cells matches the
    app's Modified Arps over 600 months."""
    s = A.segments(qi, b, di, D_LIM)
    assert s.switch_mos + s.tail_mos == pytest.approx(600.0, abs=1e-9)
    model = core.modified_arps_cum(qi * core.DAYS_PER_MONTH, b, di, D_LIM, np.array([600.0]))[0]
    assert A.segments_volume(s.fields()) == pytest.approx(model, rel=2e-3)


def test_exponential_and_low_decline_cases():
    s = A.segments(500.0, 0.0, 0.05, D_LIM)                 # b = 0: own decline for life
    assert s.method == "EXP" and s.switch_mos == 600 and s.tail_mos == 0
    assert float(s.fields()[3]) == pytest.approx(A._nominal_to_effective(0.05) * 100, abs=0.01)
    vol = core.exponential_cum(500 * core.DAYS_PER_MONTH, 0.05, np.array([600.0]))[0]
    assert A.segments_volume(s.fields()) == pytest.approx(vol, rel=2e-3)
    s = A.segments(500.0, 1.2, D_LIM * 0.5, D_LIM)          # starts below the floor: exponential at DMIN
    assert s.method == "EXP" and s.fields()[3] == s.fields()[5] == "5.00"
    s = A.segments(500.0, 1.9, 0.5, 0.0005)                 # floor reached after > 50 years: one segment
    assert s.method == "B/1.9" and s.switch_mos == 600 and s.tail_mos == 0
    s = A.segments(500.0, 1.9, D_LIM * 1.02, D_LIM)         # just above the floor: switches almost at once
    assert s.switch_mos == pytest.approx(2.4, abs=0.05) and s.tail_mos == pytest.approx(597.6, abs=0.05)


def test_lookup_rows_layout():
    groups = {"P50": dict(qi=400.0, b=1.1, di=0.12), "P10": dict(qi=500.0, b=1.2, di=0.11)}
    rows = A.lookup_rows("WCA-G1", 10000, {"Oil": groups, "Gas": groups}, {"Oil": D_LIM, "Gas": D_LIM}, ["P50"])
    assert len(rows) == 1 and len(rows[0]) == 14 == len(A.LOOKUP_COLUMNS)
    assert rows[0][:2] == ["WCA-G1", "10000"] and rows[0][4] == "B/1.1" and rows[0][10] == "B/1.1"
    assert A.LOOKUP_COLUMNS == ["TC_ID", "LL_EFF", "OIL_QI_BPM", "OIL_SWITCH_MOS", "OIL_METHOD", "OIL_DI_SEC_PCT",
                                "OIL_TAIL_MOS", "OIL_DMIN_PCT", "GAS_QI_MCFPM", "GAS_SWITCH_MOS", "GAS_METHOD",
                                "GAS_DI_SEC_PCT", "GAS_TAIL_MOS", "GAS_DMIN_PCT"]
    rows = A.lookup_rows("WCA", 10000, {"Oil": groups}, {"Oil": D_LIM}, ["P10", "P50"], water=True)
    assert [r[0] for r in rows] == ["WCA-P10", "WCA-P50"]
    assert all(len(r) == 20 for r in rows) and rows[0][8:] == [""] * 12      # no gas / water forecast
    tsv = A.to_tsv(rows, with_header=True, water=True)
    lines = tsv.split("\n")
    assert lines[0].split("\t")[-1] == "WTR_DMIN_PCT" and len(lines) == 3
    assert all(len(l.split("\t")) == 20 for l in lines)


def test_economic_lines_follow_the_lookup_template():
    groups = {"P50": dict(qi=400.0, b=1.1, di=0.12)}
    text = A.economic_lines({"Oil": groups, "Gas": groups}, {"Oil": D_LIM, "Gas": D_LIM}, "P50")
    lines = text.split("\n")
    assert len(lines) == 4
    kw, expr = lines[0].split("\t")
    assert kw == "OIL" and expr.split()[1:4] == ["X", "B/M", expr.split()[3]] and expr.split()[4] == "MOS"
    assert expr.split()[5] == "B/1.1"
    kw2, expr2 = lines[1].split("\t")
    assert kw2 == '"' and expr2.split()[:3] == ["X", "X", "B/M"] and expr2.split()[5] == "EXP"
    assert lines[2].startswith("GAS\t") and " M/M " in lines[2]
    sw, tail = float(expr.split()[3]), float(expr2.split()[3])
    assert sw + tail == pytest.approx(600.0)


def test_rows_from_the_report_type_wells_reproduce_the_eurs():
    """End to end: the type wells the app plots -> lookup cells -> ARIES volume = EUR."""
    rng = np.random.default_rng(3)
    n = 25
    ol = pd.DataFrame({"WellID": [f"W{i}" for i in range(n)], "LateralLength": 10000.0,
                       "qi (per day)": rng.uniform(300, 900, n), "b": rng.uniform(0.8, 1.2, n),
                       "di (per month)": rng.uniform(0.06, 0.2, n), "EUR (Mbbl)": rng.lognormal(6.1, 0.3, n)})
    f = R.fluid_report("Oil", ol, None, "EUR (Mbbl)", None, None, D_LIM, 10000, True, 0.8, 1.2)
    for pct in ("P10", "P50", "P90", "Mean"):
        g = f.groups[pct]
        cells = A.segments(g["qi"], g["b"], g["di"], f.terminal_di).fields()
        assert A.segments_volume(cells) / 1000.0 == pytest.approx(f.stats[pct], rel=2e-3)
