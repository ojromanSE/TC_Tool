"""ARIES export of the type curves.

Economic lines ready to paste into a case: one tab-separated cell per column
of the ARIES economic-data grid (ECONOMIC_COLUMNS), so a paste on the
Keyword cell fills each line::

    OIL  29664.4  X  B/M  197.6  MOS  B/1.2  78.47  SE    (hyperbolic until the decline reaches DMIN)
    "    X        X  B/M  402.4  MOS  EXP    4.88   SE    (exponential tail)
    GAS  ...         M/M  ...

Conventions (checked against the SE ARIES lookup: B/1.2 with an initial
decline of 78.47 and DMIN 4.88 switches after 197.6 months):

* rates are per month (B/M, M/M), the instantaneous rate at the start of the
  forecast;
* the hyperbolic segment's decline is the initial decline as a
  secant-effective annual percent;
* the tail's decline (DMIN) is the terminal decline as a tangent-effective
  annual percent;
* the hyperbolic segment runs until its decline reaches DMIN, and the two
  segments together cover the 50-year (600-month) forecast horizon.

The parameters are the type wells the app plots (median qi, b and Di of
each EUR group, qi scaled so the curve's 50-year volume equals the EUR
percentile), so the ARIES volumes match the app's EURs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

HORIZON_MONTHS = 600
DAYS_PER_MONTH = 365.25 / 12.0

PHASE_KEYS = {"Oil": ("OIL", "B/M"), "Gas": ("GAS", "M/M"), "Water": ("WTR", "B/M")}

# The columns of the ARIES economic-data grid, from the Keyword column on.
ECONOMIC_COLUMNS = ["Keyword", "Initial Value", "Final Value", "Units", "Forecast Until",
                    "Forecast Units", "Rate Method", "Rate Value", "Qualifier"]
DEFAULT_QUALIFIER = "SE"


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
    """b to four significant figures, as the SE lookup writes it (1.2, 1.127, 0.9329)."""
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
        """The six segment values, rounded as ARIES shows them."""
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
    """Total volume (bbl or Mcf) ARIES computes from the six segment values as
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


def economic_cells(phases: Dict[str, Dict[str, Dict[str, float]]], d_lims: Dict[str, float], pct: str,
                   qualifier: str = DEFAULT_QUALIFIER,
                   order: Sequence[str] = ("Oil", "Gas", "Water")) -> List[List[str]]:
    """The economic lines of one type well, one cell per ARIES grid column
    (ECONOMIC_COLUMNS):

        OIL  29664.4  X  B/M  197.6  MOS  B/1.2  78.47  SE
        "    X        X  B/M  402.4  MOS  EXP    4.88   SE

    phases : {'Oil': groups, 'Gas': groups, 'Water': groups}, where groups maps
    'P10' / 'P50' / 'P90' / 'Mean' to {qi (per day), b, di (per month)} — the
    type wells of se_report.fluid_report.  A phase that was not forecast has
    no lines."""
    out = []
    for ph in order:
        g = (phases.get(ph) or {}).get(pct)
        if g is None:
            continue
        kw, unit = PHASE_KEYS[ph]
        qi, sw, method, di, tail, dmin = segments(g["qi"], g["b"], g["di"], d_lims[ph]).fields()
        out.append([kw, qi, "X", unit, sw, "MOS", method, di, qualifier])
        if float(tail) > 0:
            out.append(['"', "X", "X", unit, tail, "MOS", "EXP", dmin, qualifier])
    return out


def economic_lines(phases: Dict[str, Dict[str, Dict[str, float]]], d_lims: Dict[str, float], pct: str,
                   qualifier: str = DEFAULT_QUALIFIER,
                   order: Sequence[str] = ("Oil", "Gas", "Water")) -> str:
    """economic_cells as tab-separated text: pasted on the Keyword cell of an
    ARIES economic-data grid, every value lands in its own column."""
    return "\n".join("\t".join(r) for r in economic_cells(phases, d_lims, pct, qualifier, order))
