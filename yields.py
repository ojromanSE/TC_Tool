"""Yield (GOR / WOR) model for forecasting a secondary phase off the primary.

Secondary-phase volumes (gas or water) are forecast as a time-dependent ratio
to the oil forecast rather than with an independent decline:

    secondary(t) = yield(t) * oil(t)

This keeps the well's GOR / WOR trend (rising GOR below bubble point, a
maturing water cut) consistent with its oil forecast, which independent
per-phase decline fits do not guarantee.

Yield model — two-slope power law (the petbox-dca "PLYield" form, re-derived
here from the equations only):

    yield(t) = c * (t / t0)^m0        for t <  t0
             = c * (t / t0)^m         for t >= t0

so the yield is continuous at t0 with value c, follows slope m0 before t0 and
slope m afterwards, and is optionally clamped to [y_min, y_max].  In log-log
space this is a two-segment line that is linear in (log c, m0, m) for a given
t0, so the fit is an ordinary least-squares problem solved on a grid of t0.

Time is "months since first production" (mid-month values, k + 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class YieldParams:
    """Fitted two-slope power-law yield."""
    c: float                        # yield at t0 (secondary / primary volume ratio)
    m0: float                       # log-log slope before t0
    m: float                        # log-log slope after t0
    t0: float                       # breakpoint (months since first production)
    y_min: Optional[float] = None   # lower clamp on the yield (None = no clamp)
    y_max: Optional[float] = None   # upper clamp on the yield (None = no clamp)
    n_points: int = 0               # number of observed ratios used in the fit
    r2: float = float('nan')        # R^2 of the fit in log space
    mode: str = 'two-slope'         # 'two-slope' or 'single-slope'


def plyield(c: float, m0: float, m: float, t0: float, t,
            y_min: Optional[float] = None, y_max: Optional[float] = None) -> np.ndarray:
    """Two-slope power-law yield evaluated at times t (> 0).

    yield = c (t/t0)^m0 for t < t0 and c (t/t0)^m for t >= t0, clamped to
    [y_min, y_max] when given.  Non-positive times are evaluated as a small
    positive time so the result stays finite.
    """
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    t0 = max(float(t0), 1e-9)
    lr = np.log(t / t0)
    slope = np.where(t < t0, float(m0), float(m))
    y = float(c) * np.exp(slope * lr)
    if y_min is not None:
        y = np.maximum(y, float(y_min))
    if y_max is not None:
        y = np.minimum(y, float(y_max))
    return y


def yield_curve(params: YieldParams, t) -> np.ndarray:
    """Evaluate a fitted yield at times t."""
    return plyield(params.c, params.m0, params.m, params.t0, t,
                   y_min=params.y_min, y_max=params.y_max)


def observed_ratio(primary_vol, secondary_vol) -> np.ndarray:
    """Observed yield secondary / primary per month; NaN where the primary is
    zero, missing or non-positive (nothing to form a ratio against)."""
    p = np.asarray(primary_vol, dtype=float)
    s = np.asarray(secondary_vol, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(np.isfinite(p) & (p > 0) & np.isfinite(s), s / p, np.nan)
    return np.where(r < 0, np.nan, r)


def _ols_single(lt: np.ndarray, ly: np.ndarray) -> tuple[float, float, float]:
    """log y = a + m log t.  Returns (a, m, sse)."""
    A = np.column_stack([np.ones_like(lt), lt])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    return float(coef[0]), float(coef[1]), float(resid @ resid)


def _ols_two_slope(lt: np.ndarray, ly: np.ndarray, lt0: float,
                   m0_fixed: Optional[float] = None, m_fixed: Optional[float] = None
                   ) -> tuple[float, float, float, float]:
    """log y = log c + m0 (log t - lt0) [t < t0] + m (log t - lt0) [t >= t0].

    Either slope may be held fixed (e.g. at its bound), in which case only the
    remaining parameters are solved for.  Returns (log c, m0, m, sse)."""
    x = lt - lt0
    xb = np.where(x < 0, x, 0.0)
    xa = np.where(x < 0, 0.0, x)
    y = ly.copy()
    cols = [np.ones_like(x)]
    if m0_fixed is None:
        cols.append(xb)
    else:
        y = y - m0_fixed * xb
    if m_fixed is None:
        cols.append(xa)
    else:
        y = y - m_fixed * xa
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    k = 1
    lc = float(coef[0])
    if m0_fixed is None:
        m0 = float(coef[k]); k += 1
    else:
        m0 = float(m0_fixed)
    m = float(coef[k]) if m_fixed is None else float(m_fixed)
    return lc, m0, m, float(resid @ resid)


def fit_yield(t, ratio, min_points: int = 6, t0_grid: Optional[Sequence[float]] = None,
              cap_quantile: float = 0.95, cap_factor: float = 1.5,
              m0_bounds: tuple = (-10.0, 10.0), m_bounds: tuple = (-1.0, 1.0)
              ) -> Optional[YieldParams]:
    """Fit a two-slope power-law yield to observed ratios.

    Parameters
    ----------
    t, ratio : observed times (months since first production, > 0) and yields.
        Non-finite or non-positive ratios are ignored.
    min_points : minimum usable points; fewer returns None.
    t0_grid : candidate breakpoints; default = the observed times that leave at
        least three points on each side.  With fewer than 10 points (or no
        valid breakpoint) a single-slope power law is fitted instead.
    cap_quantile, cap_factor : the forecast yield is clamped to
        [quantile(1 - cap_quantile) / cap_factor, cap_factor x quantile(cap_quantile)]
        of the observed ratios, so an extrapolated rising or falling GOR / WOR
        follows its trend for a while but cannot run away in either direction.
    m0_bounds, m_bounds : slope bounds (the late slope |m| <= 1 keeps the
        secondary EUR finite relative to oil).  When the unconstrained fit lands
        outside a bound, the fit is redone with that slope held at the bound so
        the breakpoint and level are re-optimised for it.

    Returns YieldParams or None when the data cannot support a fit.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(ratio, dtype=float)
    ok = np.isfinite(t) & np.isfinite(r) & (t > 0) & (r > 0)
    t, r = t[ok], r[ok]
    n = int(t.size)
    if n < max(int(min_points), 2) or np.unique(t).size < 2:
        return None
    lt, lr = np.log(t), np.log(r)
    sst = float(np.sum((lr - lr.mean()) ** 2))
    constant = float(np.ptp(lr)) < 1e-9          # exactly constant ratio (allocated gas)

    def _r2(sse: float) -> float:
        if constant or sst <= 1e-18 * n:
            return 1.0 if sse <= 1e-9 * max(n, 1) else 0.0
        return float(1.0 - sse / sst)

    y_min = float(np.quantile(r, 1.0 - cap_quantile) / cap_factor)
    y_max = float(cap_factor * np.quantile(r, cap_quantile))
    y_max = max(y_max, y_min)

    def _grid_search(m0_fixed=None, m_fixed=None):
        best = None
        for t0 in grid:
            lc, m0, m, sse = _ols_two_slope(lt, lr, np.log(t0), m0_fixed, m_fixed)
            if not np.all(np.isfinite([lc, m0, m, sse])):
                continue
            if best is None or sse < best[-1]:
                best = (lc, m0, m, float(t0), sse)
        return best

    best = None
    if n >= 10:
        if t0_grid is None:
            order = np.sort(t)
            t0_grid = order[3:n - 2]          # >= 3 points strictly before, >= 3 at/after
        grid = [float(t0) for t0 in np.unique(np.asarray(t0_grid, dtype=float))
                if t0 > 0 and np.sum(t < t0) >= 3 and np.sum(t >= t0) >= 3]
        if grid:
            best = _grid_search()
            if best is not None:
                # A slope outside its bound: hold it at the bound and re-optimise
                # the breakpoint and the other parameters for that constraint,
                # so the segment that drives the forecast is anchored to the data.
                m0_fix = float(np.clip(best[1], *m0_bounds)) if not (m0_bounds[0] <= best[1] <= m0_bounds[1]) else None
                m_fix = float(np.clip(best[2], *m_bounds)) if not (m_bounds[0] <= best[2] <= m_bounds[1]) else None
                if m0_fix is not None or m_fix is not None:
                    best = _grid_search(m0_fix, m_fix) or best
                    if best is not None and (not (m0_bounds[0] <= best[1] <= m0_bounds[1])
                                             or not (m_bounds[0] <= best[2] <= m_bounds[1])):
                        # The other slope moved out of range too: fix both.
                        best = _grid_search(float(np.clip(best[1], *m0_bounds)),
                                            float(np.clip(best[2], *m_bounds))) or best

    if best is not None:
        lc, m0, m, t0, sse = best
        return YieldParams(c=float(np.exp(lc)), m0=m0, m=m, t0=t0, y_min=y_min, y_max=y_max,
                           n_points=n, r2=_r2(sse), mode='two-slope')

    # Single-slope fallback: y = c (t / t0)^m with t0 at the median time.
    a, m, sse = _ols_single(lt, lr)
    if not np.all(np.isfinite([a, m, sse])):
        return None
    m = float(np.clip(m, *m_bounds))
    t0 = float(np.median(t))
    pred = m * (lt - np.log(t0))
    lc = float(np.mean(lr - pred))                  # least-squares level for the (clipped) slope
    sse = float(np.sum((lr - lc - pred) ** 2))
    return YieldParams(c=float(np.exp(lc)), m0=m, m=m, t0=t0, y_min=y_min, y_max=y_max,
                       n_points=n, r2=_r2(sse), mode='single-slope')


def forecast_secondary(params: YieldParams, t_index, primary_vol) -> np.ndarray:
    """Secondary-phase monthly volumes = yield(t + 0.5) x primary monthly volume.

    t_index are integer month indices since first production (the yield is
    evaluated at mid-month); months with a non-positive or missing primary
    volume get 0.
    """
    t_mid = np.asarray(t_index, dtype=float) + 0.5
    p = np.asarray(primary_vol, dtype=float)
    y = yield_curve(params, t_mid)
    out = np.where(np.isfinite(p) & (p > 0), y * np.where(np.isfinite(p), p, 0.0), 0.0)
    return out
