"""ARIES export of the type curves.

Two copy-ready forms:

* **Lookup rows** for the SE ARIES lookup ``SE_TYPECURVES`` — one row per type
  curve, tab-separated in the lookup's column order::

      TC_ID  LL_EFF  OIL_QI_BPM  OIL_SWITCH_MOS  OIL_METHOD  OIL_DI_SEC_PCT  OIL_TAIL_MOS  OIL_DMIN_PCT
                     GAS_QI_MCFPM  GAS_SWITCH_MOS  GAS_METHOD  GAS_DI_SEC_PCT  GAS_TAIL_MOS  GAS_DMIN_PCT

  The lookup's economic-line template turns each row into

      OIL  QI  X  B/M  SWITCH  MOS  B/b  DI_SEC     (hyperbolic until the decline reaches DMIN)
      "    X   X  B/M  TAIL    MOS  EXP  DMIN       (exponential tail)
      GAS  ...  M/M ...

* **Economic lines** in that same form, to paste into a case directly — one
  tab-separated cell per column of the ARIES grid (Keyword, Initial Value,
  Final Value, Units, Forecast Until, Forecast Units, Rate Method, Rate Value).

Conventions (checked against the SE lookup: B/1.2 with DI_SEC 78.47 and
DMIN 4.88 switches after 197.6 months):

* rates are per month (B/M, M/M), the instantaneous rate at the start of the
  forecast;
* ``DI_SEC_PCT`` is the initial decline as a secant-effective annual percent;
* ``DMIN_PCT`` is the terminal decline as a tangent-effective annual percent
  (the exponential tail);
* ``SWITCH_MOS`` is when the hyperbolic decline reaches DMIN and
  ``SWITCH_MOS + TAIL_MOS`` is the 50-year (600-month) forecast horizon.

The parameters are the type wells the app plots (median qi, b and Di of
each EUR group, qi scaled so the curve's 50-year volume equals the EUR
percentile), so the ARIES volumes match the app's EURs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

HORIZON_MONTHS = 600
DAYS_PER_MONTH = 365.25 / 12.0

PHASE_KEYS = {"Oil": ("OIL", "BPM", "B/M"), "Gas": ("GAS", "MCFPM", "M/M"), "Water": ("WTR", "BPM", "B/M")}


def phase_columns(phase: str) -> List[str]:
    kw, qi_unit, _ = PHASE_KEYS[phase]
    return [f"{kw}_QI_{qi_unit}", f"{kw}_SWITCH_MOS", f"{kw}_METHOD", f"{kw}_DI_SEC_PCT",
            f"{kw}_TAIL_MOS", f"{kw}_DMIN_PCT"]


LOOKUP_COLUMNS = ["TC_ID", "LL_EFF"] + phase_columns("Oil") + phase_columns("Gas")


def _nominal_to_effective(d_month: float, b: float = 0.0) -> float:
    """Nominal monthly decline -> effective annual fraction (secant for b > 0,
    tangent for b = 0)."""
    D = float(d_month) * 12.0
    if b <= 1e-10:
        return float(-math.expm1(-D))
    return float(-math.expm1(-math.log1p(b * D) / b))


def _effective_to_nominal(de: float, b: float = 0.0) -> float:
    """Effective annual fraction -> nominal monthly decline (inverse of the above)."""
    de = min(max(float(de), 0.0), 0.999999)
    if b <= 1e-10:
        return float(-math.log1p(-de) / 12.0)
    return float(math.expm1(-b * math.log1p(-de)) / b / 12.0)


def fmt_b(b: float) -> str:
    """b to four significant figures, as the lookup writes it (1.2, 1.127, 0.9329)."""
    return f"{float(b):.4g}"


@dataclass
class AriesSegments:
    qi_pm: float            # initial rate per month (B/M or M/M)
    switch_mos: float       # months of the first segment
    method: str             # 'B/1.2' (hyperbolic) or 'EXP'
    di_sec_pct: float       # first-segment decline, % per year (secant effective; tangent for EXP)
    tail_mos: float         # months of the exponential tail
    dmin_pct: float         # tail decline, % per year (tangent effective)

    def fields(self) -> List[str]:
        """The six lookup cells, rounded as the SE lookup shows them."""
        return [f"{self.qi_pm:.1f}", f"{self.switch_mos:.1f}", self.method, f"{self.di_sec_pct:.2f}",
                f"{self.tail_mos:.1f}", f"{self.dmin_pct:.2f}"]


def segments(qi_per_day: float, b: float, di: float, d_lim: float,
             horizon: int = HORIZON_MONTHS) -> AriesSegments:
    """ARIES segments of a Modified Arps type well.

    qi_per_day : instantaneous initial rate per day; b : Arps exponent;
    di : initial nominal decline per month; d_lim : terminal nominal decline
    per month (the hyperbolic decline switches to an exponential at d_lim)."""
    qi_pm = float(qi_per_day) * DAYS_PER_MONTH
    dmin_pct = _nominal_to_effective(d_lim) * 100.0
    b, di, d_lim = float(b), float(di), float(d_lim)
    if di <= d_lim or not np.isfinite(di):
        # Already at or below the terminal decline: exponential at DMIN for life.
        return AriesSegments(qi_pm, float(horizon), "EXP", dmin_pct, 0.0, dmin_pct)
    if b <= 1e-6:
        # Exponential well (b = 0): its own decline for life, never slower than DMIN.
        return AriesSegments(qi_pm, float(horizon), "EXP", _nominal_to_effective(di) * 100.0, 0.0, dmin_pct)
    t_sw = (di / d_lim - 1.0) / (b * di)                      # months until D(t) = d_lim
    switch = round(min(t_sw, float(horizon)), 1)
    return AriesSegments(qi_pm, switch, f"B/{fmt_b(b)}", _nominal_to_effective(di, b) * 100.0,
                         round(float(horizon) - switch, 1), dmin_pct)


def segments_volume(cells: Sequence[str]) -> float:
    """Total volume (bbl or Mcf) ARIES computes from the six lookup cells as
    written (rounded), integrating the hyperbolic and exponential segments.
    Independent of the fitting code; used to check the export against the EUR."""
    qi, switch, method, di_pct, tail, dmin_pct = (float(cells[0]), float(cells[1]), cells[2],
                                                 float(cells[3]), float(cells[4]), float(cells[5]))
    if method.upper() == "EXP":
        d = _effective_to_nominal(di_pct / 100.0)
        n1 = qi / d * -math.expm1(-d * switch) if d > 0 else qi * switch
        q_sw = qi * math.exp(-d * switch)
    else:
        b = float(method.split("/", 1)[1])
        d = _effective_to_nominal(di_pct / 100.0, b)
        if abs(b - 1.0) < 1e-9:
            n1 = qi / d * math.log1p(d * switch)
        else:
            n1 = qi / ((1.0 - b) * d) * (1.0 - (1.0 + b * d * switch) ** (1.0 - 1.0 / b))
        q_sw = qi * (1.0 + b * d * switch) ** (-1.0 / b)
    dt = _effective_to_nominal(dmin_pct / 100.0)
    n2 = q_sw / dt * -math.expm1(-dt * tail) if dt > 0 else q_sw * tail
    return n1 + n2


def lookup_rows(tc_id: str, ll_eff: float, phases: Dict[str, Dict[str, Dict[str, float]]],
                d_lims: Dict[str, float], percentiles: Sequence[str], water: bool = False
                ) -> List[List[str]]:
    """One SE_TYPECURVES row per percentile.

    phases : {'Oil': groups, 'Gas': groups, 'Water': groups}, where groups maps
    'P10' / 'P50' / 'P90' / 'Mean' to {qi (per day), b, di (per month)} — the
    type wells of se_report.fluid_report.  A phase that was not forecast gets
    empty cells.  With several percentiles the TC_ID gets a -P10 / -P50 / ...
    suffix."""
    rows = []
    cols = ["Oil", "Gas"] + (["Water"] if water else [])
    for pct in percentiles:
        tid = tc_id if len(percentiles) == 1 else f"{tc_id}-{pct.upper()}"
        row = [tid, f"{int(round(ll_eff))}"]
        for ph in cols:
            g = (phases.get(ph) or {}).get(pct)
            if g is None:
                row += [""] * 6
            else:
                row += segments(g["qi"], g["b"], g["di"], d_lims[ph]).fields()
        rows.append(row)
    return rows


def header(water: bool = False) -> List[str]:
    return LOOKUP_COLUMNS + (phase_columns("Water") if water else [])


def to_tsv(rows: Iterable[Sequence[str]], with_header: bool = False, water: bool = False) -> str:
    """Tab-separated text that pastes into the ARIES lookup grid cell by cell."""
    lines = (["\t".join(header(water))] if with_header else []) + ["\t".join(map(str, r)) for r in rows]
    return "\n".join(lines)


# The columns of the ARIES economic-data grid, from the Keyword column on.
ECONOMIC_COLUMNS = ["Keyword", "Initial Value", "Final Value", "Units", "Forecast Until",
                    "Forecast Units", "Rate Method", "Rate Value"]


def economic_cells(phases: Dict[str, Dict[str, Dict[str, float]]], d_lims: Dict[str, float], pct: str,
                   order: Sequence[str] = ("Oil", "Gas", "Water")) -> List[List[str]]:
    """The economic lines of one type well, one cell per ARIES grid column
    (ECONOMIC_COLUMNS), as the SE lookup template expands them:

        OIL  29664.4  X  B/M  197.6  MOS  B/1.2  78.47
        "    X        X  B/M  402.4  MOS  EXP    4.88
    """
    out = []
    for ph in order:
        g = (phases.get(ph) or {}).get(pct)
        if g is None:
            continue
        kw, _, unit = PHASE_KEYS[ph]
        qi, sw, method, di, tail, dmin = segments(g["qi"], g["b"], g["di"], d_lims[ph]).fields()
        out.append([kw, qi, "X", unit, sw, "MOS", method, di])
        if float(tail) > 0:
            out.append(['"', "X", "X", unit, tail, "MOS", "EXP", dmin])
    return out


def economic_lines(phases: Dict[str, Dict[str, Dict[str, float]]], d_lims: Dict[str, float], pct: str,
                   order: Sequence[str] = ("Oil", "Gas", "Water")) -> str:
    """economic_cells as tab-separated text: pasted on the Keyword cell of an
    ARIES economic-data grid, every value lands in its own column."""
    return "\n".join("\t".join(r) for r in economic_cells(phases, d_lims, pct, order))
