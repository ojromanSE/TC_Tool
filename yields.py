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


def _ols_two_slope(lt: np.ndarray, ly: np.ndarray, lt0: float) -> tuple[float, float, float, float]:
    """log y = log c + m0 (log t - lt0) [t < t0] + m (log t - lt0) [t >= t0].
    Returns (log c, m0, m, sse)."""
    x = lt - lt0
    before = x < 0
    A = np.column_stack([np.ones_like(x), np.where(before, x, 0.0), np.where(before, 0.0, x)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), float(resid @ resid)


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
    cap_quantile, cap_factor : the forecast yield is clamped above at
        cap_factor x the cap_quantile of the observed ratios, so an extrapolated
        rising GOR / WOR cannot run away; the lower clamp is the observed minimum.
    m0_bounds, m_bounds : slopes are clipped to these ranges after the fit
        (the late slope |m| <= 1 keeps the secondary EUR finite relative to oil).

    Returns YieldParams or None when the data cannot support a fit.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(ratio, dtype=float)
    ok = np.isfinite(t) & np.isfinite(r) & (t > 0) & (r > 0)
    t, r = t[ok], r[ok]
    n = int(t.size)
    if n < max(int(min_points), 2):
        return None
    lt, lr = np.log(t), np.log(r)
    sst = float(np.sum((lr - lr.mean()) ** 2))

    def _r2(sse: float) -> float:
        return float(1.0 - sse / sst) if sst > 0 else (1.0 if sse <= 1e-12 else 0.0)

    y_min = float(np.min(r))
    y_max = float(cap_factor * np.quantile(r, cap_quantile))
    y_max = max(y_max, y_min)

    best = None
    if n >= 10:
        if t0_grid is None:
            order = np.sort(t)
            t0_grid = order[3:n - 2]          # >= 3 points strictly before, >= 3 at/after
        for t0 in np.unique(np.asarray(t0_grid, dtype=float)):
            if not (t0 > 0) or np.sum(t < t0) < 3 or np.sum(t >= t0) < 3:
                continue
            lc, m0, m, sse = _ols_two_slope(lt, lr, np.log(t0))
            if not np.all(np.isfinite([lc, m0, m, sse])):
                continue
            if best is None or sse < best[-1]:
                best = (lc, m0, m, float(t0), sse)

    if best is not None:
        lc, m0, m, t0, sse = best
        m0 = float(np.clip(m0, *m0_bounds))
        m = float(np.clip(m, *m_bounds))
        if m0 != best[1] or m != best[2]:
            # Slopes were clipped: refit the level for the clipped slopes.
            x = lt - np.log(t0)
            pred = np.where(x < 0, m0 * x, m * x)
            lc = float(np.mean(lr - pred))
            sse = float(np.sum((lr - lc - pred) ** 2))
        return YieldParams(c=float(np.exp(lc)), m0=m0, m=m, t0=t0, y_min=y_min, y_max=y_max,
                           n_points=n, r2=_r2(sse), mode='two-slope')

    # Single-slope fallback: y = c (t / t0)^m with t0 at the median time.
    a, m, sse = _ols_single(lt, lr)
    if not np.all(np.isfinite([a, m, sse])):
        return None
    m = float(np.clip(m, *m_bounds))
    t0 = float(np.median(t))
    pred = m * (lt - np.log(t0))
    lc = float(np.mean(lr - pred))
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
