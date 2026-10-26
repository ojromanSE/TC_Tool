"""Flow-regime diagnostics for decline-curve analysis.

SPEE Monograph 5 asks the analyst to identify the flow regime before trusting a
b factor: transient linear flow in multi-fractured horizontals shows a half
slope on log-log rate-time (Arps b ~ 2), and boundary-dominated flow shows a
flattening nominal decline with b < 1.  This module estimates from the data

    D(t)   = -d ln q / dt          nominal decline
    b(t)   = d(1/D) / dt           Arps b (constant for a pure hyperbolic)
    slope  = d ln q / d ln t       log-log slope (-0.5 linear flow, -1 BDF-ish)

using the Bourdet derivative (Bourdet, Ayoub & Pirard, SPE-12777, Eq. 8), the
pressure-transient industry's standard smoothed log derivative, and evaluates
the same quantities analytically for the fitted model so the two can be
overlaid.

Everything is written from the published equations; nothing is imported from
third-party DCA packages.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

LN10 = float(np.log(10.0))


def bourdet(y, x, L: float = 0.0, xlog: bool = True, ylog: bool = False) -> np.ndarray:
    """Bourdet derivative of y with respect to ln x (SPE-12777 Eq. 8).

    For each point the left and right secant slopes are taken to the nearest
    points at least L log10-cycles away (L = 0: adjacent points) and combined
    with weights equal to the *opposite* interval length:

        d = (sL * dxR + sR * dxL) / (dxL + dxR)

    Parameters
    ----------
    y, x : samples; x must be strictly increasing and positive.  NaN / inf
        values of y are skipped (dropped before differencing; NaN returned at
        those positions).
    L : smoothing window in log10 cycles of x (the window is always chosen in
        log x, so it widens with x as the data thin out on a log axis).
    xlog : True returns dy/d ln x with the secants taken in ln x (the
        classic Bourdet form).  False returns dy/dx with the secants taken in
        x itself, which is exact for y linear in x: an exponential's ln q, or
        1/D = 1/Di + b t of a hyperbolic, so b(t) comes back exactly.
    ylog : True differentiates ln y instead of y (so with xlog=False and
        ylog=True the result is d ln y / dx = -D for a rate).

    The first and last points use one-sided differences and are less reliable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be 1-D arrays of the same length")
    if np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("x must be positive, finite and strictly increasing")
    L = float(L)
    if L < 0:
        raise ValueError("L must be >= 0")

    out = np.full(x.shape, np.nan)
    yy = np.log(y) if ylog else y
    keep = np.isfinite(yy)
    if ylog:
        keep &= np.isfinite(y) & (y > 0)
    idx = np.flatnonzero(keep)
    n = idx.size
    if n < 2:
        return out
    lx = np.log10(x[idx])
    yv = yy[idx]
    pts = np.arange(n)
    if L == 0.0:
        iL = np.clip(pts - 1, 0, n - 1)
        iR = np.clip(pts + 1, 0, n - 1)
    else:
        # nearest points at distance >= L on each side (ties handled symmetrically)
        iL = np.clip(np.searchsorted(lx, lx - L, side='right') - 1, 0, n - 1)
        iR = np.clip(np.searchsorted(lx, lx + L, side='left'), 0, n - 1)
    iL = np.minimum(iL, np.maximum(pts - 1, 0))     # at least one point away
    iR = np.maximum(iR, np.minimum(pts + 1, n - 1))
    if xlog:
        dxL = (lx[pts] - lx[iL]) * LN10
        dxR = (lx[iR] - lx[pts]) * LN10
    else:
        xv = x[idx]
        dxL = xv[pts] - xv[iL]
        dxR = xv[iR] - xv[pts]
    dyL = yv[pts] - yv[iL]
    dyR = yv[iR] - yv[pts]
    der = np.zeros(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        sL = np.where(dxL > 0, dyL / dxL, 0.0)
        sR = np.where(dxR > 0, dyR / dxR, 0.0)
        both = (dxL > 0) & (dxR > 0)
        lo = (dxL > 0) & ~(dxR > 0)
        ro = ~(dxL > 0) & (dxR > 0)
        der[both] = (sL[both] * dxR[both] + sR[both] * dxL[both]) / (dxL + dxR)[both]
        der[lo] = sL[lo]
        der[ro] = sR[ro]
    out[idx] = der
    return out


def local_slope(x, y, half_width: float, min_points: int = 3) -> np.ndarray:
    """Derivative dy/dx by centred moving-window least squares.

    For each point the slope of the least-squares line through the samples
    within +/- half_width of it is returned (NaN samples are skipped).  Where
    fewer than min_points fall in the window it is widened to the nearest
    min_points.  Exact for y linear in x whatever the window, and far less
    noise-sensitive than two-point differences on monthly data; windows at the
    start and end of the series are truncated (one-sided) and less reliable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    out = np.full(n, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    xi, yi = x[ok], y[ok]
    m = xi.size
    if m < 2:
        return out
    order = np.argsort(xi)
    xi, yi = xi[order], yi[order]
    pos = np.flatnonzero(ok)[order]
    k_need = min(max(int(min_points), 2), m)
    for j in range(m):
        lo = np.searchsorted(xi, xi[j] - half_width, side='left')
        hi = np.searchsorted(xi, xi[j] + half_width, side='right')
        while hi - lo < k_need:                    # widen to the nearest points
            if lo > 0 and (hi >= m or xi[j] - xi[lo - 1] <= xi[hi] - xi[j]):
                lo -= 1
            elif hi < m:
                hi += 1
            else:
                lo -= 1
        xs, ys = xi[lo:hi], yi[lo:hi]
        xm = xs.mean()
        sxx = float(np.sum((xs - xm) ** 2))
        if sxx > 0:
            out[pos[j]] = float(np.sum((xs - xm) * (ys - ys.mean())) / sxx)
    return out


def data_diagnostics(t, q, window_D: float = 6.0, window_b: float = 9.0, L_slope: float = 0.2,
                     min_rate_frac: float = 0.2, shutin_window: float = 6.0) -> Dict[str, np.ndarray]:
    """Data-derived flow-regime diagnostics for one well.

    Parameters
    ----------
    t : times since peak (months, > 0; mid-month values k + 0.5 are the natural
        choice for monthly volumes).
    q : monthly volumes (or rates).  Zero, negative and missing values are
        dropped, as are shut-in / curtailed months: a month producing less than
        min_rate_frac x the median of its neighbours within +/- shutin_window
        months.  The test is relative to the local level, not to the peak, so
        the legitimate late-time tail of a steep decline is kept.
    window_D, window_b : half-widths (months) of the centred least-squares
        windows for D = -d ln q/dt and b = d(1/D)/dt.  b is the derivative of a
        derivative, so it is noisier and gets the wider window.  The defaults
        (13- and 19-month windows) resolve a transient-to-boundary-dominated
        transition within about a year while keeping 5 % monthly noise to
        roughly +/- 0.5 in b at late time.
    L_slope : Bourdet window (log10 cycles) for the log-log slope d ln q/d ln t,
        which lives naturally on the log time axis.

    Returns dict with arrays on the retained points: t, q, D (nominal decline
    per month), beta (= 1/D), b (= d beta / dt), loglog_slope (d ln q / d ln t),
    plus n_dropped (number of shut-in / curtailed months removed).
    The first and last points of each derivative use truncated, one-sided
    windows and are less reliable.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if t.shape != q.shape:
        raise ValueError("t and q must have the same shape")
    keep = np.isfinite(t) & np.isfinite(q) & (t > 0) & (q > 0)
    t, q = t[keep], q[keep]
    order = np.argsort(t)
    t, q = t[order], q[order]
    n_dropped = 0
    if t.size >= 3 and min_rate_frac > 0:
        # Shut-in / curtailment test against the local level (median of the
        # other producing months within +/- shutin_window).
        local = np.empty(t.size)
        for i in range(t.size):
            near = (np.abs(t - t[i]) <= shutin_window)
            near[i] = False
            local[i] = np.median(q[near]) if near.any() else q[i]
        good = q >= min_rate_frac * local
        n_dropped = int((~good).sum())
        t, q = t[good], q[good]
    # Collapse duplicate times (Bourdet needs strictly increasing x).
    if t.size and np.any(np.diff(t) <= 0):
        tu, inv = np.unique(t, return_inverse=True)
        qu = np.array([q[inv == i].mean() for i in range(tu.size)])
        t, q = tu, qu
    n = t.size
    if n < 3:
        nan = np.full(n, np.nan)
        return dict(t=t, q=q, D=nan, beta=nan.copy(), b=nan.copy(), loglog_slope=nan.copy(),
                    n_dropped=n_dropped)

    D = -local_slope(t, np.log(q), window_D)                   # -d ln q / dt
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.where(D > 0, 1.0 / D, np.nan)               # 1/D; negative D (incline) -> NaN
    b = local_slope(t, beta, window_b)                         # d(1/D)/dt
    slope = bourdet(np.log(q), t, L=L_slope, xlog=True, ylog=False)   # d ln q / d ln t
    return dict(t=t, q=q, D=D, beta=beta, b=b, loglog_slope=slope, n_dropped=n_dropped)


def _arps_D_b(di: float, b: float, t: np.ndarray):
    """Nominal decline and b of a hyperbolic/exponential Arps segment."""
    if b <= 1e-10:
        return np.full(t.shape, float(di)), np.zeros(t.shape)
    return di / (1.0 + b * di * t), np.full(t.shape, float(b))


def model_diagnostics(model: str, params: Dict[str, float], t, d_lim: float) -> Dict[str, np.ndarray]:
    """Analytic D(t), beta(t) = 1/D and b(t) = d(1/D)/dt of a fitted model,
    honouring the terminal-decline floor d_lim (D is clamped at d_lim and b = 0
    once the floor is reached, matching core's switch logic).

    model / params (as stored in a core forecast dict, plus 'qi'):
        'Modified Arps' : qi, b, di
        'Exponential'   : qi, di
        'SEPD'          : qi, tau, n
        'Duong'         : qi, a, m           (1-based time t' = t + 1)
    """
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    d_lim = float(d_lim) if d_lim is not None else 0.0

    if model == 'Modified Arps':
        D, b = _arps_D_b(float(params['di']), float(params['b']), t)
    elif model == 'Exponential':
        D, b = _arps_D_b(float(params['di']), 0.0, t)
    elif model == 'SEPD':
        tau, n = float(params['tau']), float(params['n'])
        tt = np.maximum(t, 1e-9)
        # q = qi exp(-(t/tau)^n):  D = n t^(n-1) / tau^n ;  1/D = tau^n t^(1-n) / n
        D = n * np.power(tt, n - 1.0) / tau ** n
        b = (1.0 - n) * tau ** n * np.power(tt, -n) / n          # d(1/D)/dt
    elif model == 'Duong':
        a, m = float(params['a']), float(params['m'])
        tp = t + 1.0
        # q = qi t'^(-m) exp(a/(1-m)(t'^(1-m) - 1)):  D = m/t' - a t'^(-m)
        D = m / tp - a * np.power(tp, -m)
        dD = -m / tp ** 2 + a * m * np.power(tp, -m - 1.0)      # dD/dt'
        with np.errstate(divide='ignore', invalid='ignore'):
            b = np.where(D != 0, -dD / D ** 2, np.nan)            # d(1/D)/dt = -D'/D^2
    else:
        raise ValueError(f"unknown model {model!r}")

    if d_lim > 0:
        on_floor = D <= d_lim
        if model in ('Duong',):
            # Duong's D rises then falls; the floor applies only after the final
            # crossing.  Mark everything after the last point above the floor.
            above = np.flatnonzero(D > d_lim)
            on_floor = np.zeros(t.shape, dtype=bool)
            if above.size:
                on_floor[above[-1] + 1:] = True
            else:
                on_floor[:] = True
        D = np.where(on_floor, d_lim, D)
        b = np.where(on_floor, 0.0, b)
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.where(D > 0, 1.0 / D, np.nan)
    return dict(D=D, beta=beta, b=b)


def reference_slopes() -> Dict[str, float]:
    """Log-log rate-time slopes of the classic flow regimes (d ln q / d ln t)."""
    return {
        'bilinear flow': -0.25,
        'linear flow': -0.5,
        'boundary-dominated (harmonic, b = 1)': -1.0,
    }
