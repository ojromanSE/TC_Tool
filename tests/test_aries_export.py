"""Tests for aries_export.py — ARIES economic lines."""
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


def test_economic_lines_follow_the_lookup_template():
    groups = {"P50": dict(qi=400.0, b=1.1, di=0.12)}
    text = A.economic_lines({"Oil": groups, "Gas": groups}, {"Oil": D_LIM, "Gas": D_LIM}, "P50")
    lines = [l.split("\t") for l in text.split("\n")]
    # one cell per ARIES grid column, so a paste fills Keyword .. Qualifier
    assert A.ECONOMIC_COLUMNS[-1] == "Qualifier"
    assert len(lines) == 4 and all(len(l) == len(A.ECONOMIC_COLUMNS) == 9 for l in lines)
    assert all(c == c.strip() and c for l in lines for c in l)
    oil, tail_line, gas, _ = lines
    cells = A.segments(400.0, 1.1, 0.12, D_LIM).fields()
    assert oil == ["OIL", cells[0], "X", "B/M", cells[1], "MOS", "B/1.1", cells[3], "SE"]
    assert tail_line == ['"', "X", "X", "B/M", cells[4], "MOS", "EXP", cells[5], "SE"]
    assert gas[0] == "GAS" and gas[3] == "M/M"
    assert float(oil[4]) + float(tail_line[4]) == pytest.approx(600.0)


def test_economic_lines_qualifier_and_missing_phases():
    groups = {"P50": dict(qi=400.0, b=1.1, di=0.12)}
    rows = A.economic_cells({"Oil": groups, "Water": groups}, {"Oil": D_LIM, "Water": D_LIM}, "P50",
                            qualifier="PUD")
    assert [r[0] for r in rows] == ["OIL", '"', "WTR", '"']                  # no gas forecast: no GAS lines
    assert all(r[-1] == "PUD" for r in rows) and rows[2][3] == "B/M"
    assert A.economic_cells({"Oil": groups}, {"Oil": D_LIM}, "P10") == []    # no such type well


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
