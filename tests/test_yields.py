"""Tests for yields.py — two-slope power-law GOR / WOR yield model."""
from __future__ import annotations

import numpy as np
import pytest

import yields as Y


# ---------------------------------------------------------------- plyield
def test_plyield_reference_values():
    # c = 1.2, m0 = 0, m = 0.6, t0 = 180: flat before t0, 0.6-slope after.
    y = Y.plyield(1.2, 0.0, 0.6, 180.0, [90, 180, 365, 1095, 3650])
    np.testing.assert_allclose(y, [1.200, 1.200, 1.834, 3.545, 7.301], atol=5e-4)


def test_plyield_is_continuous_at_t0_and_follows_both_slopes():
    c, m0, m, t0 = 3.0, -0.3, 0.7, 24.0
    eps = 1e-9
    assert Y.plyield(c, m0, m, t0, t0 - eps) == pytest.approx(c, rel=1e-6)
    assert Y.plyield(c, m0, m, t0, t0 + eps) == pytest.approx(c, rel=1e-6)
    assert Y.plyield(c, m0, m, t0, t0 / 2) == pytest.approx(c * 0.5 ** m0)
    assert Y.plyield(c, m0, m, t0, 2 * t0) == pytest.approx(c * 2.0 ** m)


def test_plyield_clamps_and_handles_nonpositive_time():
    y = Y.plyield(1.0, 1.0, 1.0, 10.0, [0.0, 1.0, 10.0, 1000.0], y_min=0.5, y_max=20.0)
    assert np.all(np.isfinite(y))
    assert y[0] == pytest.approx(0.5)       # t <= 0 evaluated at tiny t (y -> 0), clamped below
    assert y[1] == pytest.approx(0.5)       # 0.1 clamped below
    assert y[2] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(20.0)     # 100 clamped above
    # negative early slope at t <= 0 would be +inf without the time guard
    yn = Y.plyield(1.0, -0.3, 0.5, 10.0, [0.0, -1.0, 1.0])
    assert np.all(np.isfinite(yn)) and yn[0] == yn[1] > yn[2]
    assert np.all(np.isfinite(Y.plyield(1.0, 0.2, 0.5, 0.0, [0.5, 1.0])))


# ---------------------------------------------------------------- observed_ratio
def test_observed_ratio_masks_bad_months():
    r = Y.observed_ratio([100.0, 0.0, np.nan, 50.0, 20.0], [200.0, 10.0, 5.0, np.nan, -1.0])
    assert r[0] == pytest.approx(2.0)
    assert np.isnan(r[1]) and np.isnan(r[2]) and np.isnan(r[3]) and np.isnan(r[4])


# ---------------------------------------------------------------- fit_yield
def test_fit_yield_recovers_two_slope_truth():
    rng = np.random.default_rng(7)
    t = np.arange(72) + 0.5
    truth = Y.plyield(2.0, 0.1, 0.5, 18.0, t)
    obs = truth * np.exp(rng.normal(0.0, 0.04, t.size))
    p = Y.fit_yield(t, obs)
    assert p is not None and p.mode == 'two-slope'
    assert p.c == pytest.approx(2.0, rel=0.08)
    assert p.m0 == pytest.approx(0.1, abs=0.08)
    assert p.m == pytest.approx(0.5, abs=0.05)
    assert 12.0 <= p.t0 <= 26.0
    assert p.n_points == 72
    # r2 is the log-space coefficient of determination of the returned curve
    lr = np.log(obs)
    pred = np.log(Y.plyield(p.c, p.m0, p.m, p.t0, t))
    r2_ind = 1 - np.sum((lr - pred) ** 2) / np.sum((lr - lr.mean()) ** 2)
    assert 0.9 < p.r2 < 1.0 and p.r2 == pytest.approx(r2_ind, abs=1e-12)
    # the implied curve reproduces the truth over the fitted window
    np.testing.assert_allclose(Y.yield_curve(p, t), truth, rtol=0.12)
    # exact two-slope data: near-perfect r2 (t0 restricted to observed times)
    pe = Y.fit_yield(t, truth)
    assert pe.r2 > 0.999


def test_fit_yield_constant_ratio_is_exact():
    """Allocated gas (gas = oil x fixed GOR) gives an exactly constant ratio:
    r2 must be 1.0, not rounding noise, and the curve flat at that ratio."""
    t = np.arange(20) + 0.5
    for v in (1234.5678, 2.5, 0.037, 1.0):
        p = Y.fit_yield(t, np.full(20, v))
        assert p is not None and p.r2 == 1.0
        np.testing.assert_allclose(Y.yield_curve(p, t), v, rtol=1e-9)
        assert abs(p.m0) < 1e-9 and abs(p.m) < 1e-9


def test_fit_yield_single_slope_fallback_for_short_history():
    t = np.arange(8) + 0.5
    obs = Y.plyield(1.5, 0.2, 0.2, 4.0, t)
    p = Y.fit_yield(t, obs)
    assert p is not None and p.mode == 'single-slope'
    assert p.m0 == p.m == pytest.approx(0.2, abs=1e-6)
    assert p.r2 == pytest.approx(1.0)
    np.testing.assert_allclose(Y.yield_curve(p, t), obs, rtol=1e-6)


def test_fit_yield_returns_none_when_too_few_points():
    t = np.arange(5) + 0.5
    assert Y.fit_yield(t, np.ones(5) * 2.0) is None
    # everything masked -> None
    assert Y.fit_yield(np.arange(20) + 0.5, np.full(20, np.nan)) is None
    assert Y.fit_yield(np.arange(20) + 0.5, np.zeros(20)) is None
    # a single distinct time cannot identify a slope
    assert Y.fit_yield(np.full(12, 5.5), np.linspace(1, 2, 12)) is None


def test_fit_yield_handles_duplicated_months():
    rng = np.random.default_rng(11)
    t = np.concatenate([np.arange(36) + 0.5, np.array([3.5, 7.5, 12.5, 20.5, 30.5])])
    obs = Y.plyield(2.0, 0.1, 0.5, 12.0, t) * np.exp(rng.normal(0, 0.03, t.size))
    p = Y.fit_yield(t, obs)
    assert p is not None and p.n_points == 41 and p.mode == 'two-slope'
    assert p.m == pytest.approx(0.5, abs=0.06) and p.c == pytest.approx(2.0, rel=0.1)


def test_fit_yield_ignores_nonfinite_and_nonpositive_ratios():
    rng = np.random.default_rng(3)
    t = np.arange(40) + 0.5
    obs = Y.plyield(1.0, 0.0, 0.3, 12.0, t) * np.exp(rng.normal(0, 0.03, 40))
    obs[[3, 11, 25]] = np.nan
    obs[[7, 30]] = 0.0
    p = Y.fit_yield(t, obs)
    assert p is not None
    assert p.n_points == 35
    assert p.m == pytest.approx(0.3, abs=0.06)


def test_fit_yield_caps_runaway_extrapolation():
    t = np.arange(30) + 0.5
    obs = Y.plyield(1.0, 0.0, 0.9, 6.0, t)          # strongly rising ratio
    p = Y.fit_yield(t, obs)
    assert p is not None
    assert p.y_max == pytest.approx(1.5 * np.quantile(obs, 0.95))
    assert p.y_min == pytest.approx(np.quantile(obs, 0.05) / 1.5)
    far = Y.yield_curve(p, np.array([600.5]))[0]
    assert far == pytest.approx(p.y_max)        # clamped, not 1.0 * 100^0.9


def test_fit_yield_slope_bounds_are_enforced_with_refit():
    """When the late slope exceeds its bound the fit is redone with the slope
    held at the bound: the level is the least-squares level for that slope and
    the breakpoint is re-chosen, so the curve tracks the recent data instead
    of keeping the level fitted for the unconstrained slope."""
    t = np.arange(40) + 0.5
    obs = Y.plyield(1.0, 0.0, 1.8, 10.0, t)          # late slope beyond the bound
    p = Y.fit_yield(t, obs, m_bounds=(-1.0, 1.0))
    assert p is not None and p.m == 1.0 and -10 <= p.m0 <= 10
    resid = np.log(obs) - np.log(Y.plyield(p.c, p.m0, p.m, p.t0, t))
    assert abs(resid.mean()) < 1e-9                  # least-squares level for the clipped slopes
    sst = np.sum((np.log(obs) - np.log(obs).mean()) ** 2)
    assert p.r2 == pytest.approx(1 - resid @ resid / sst, abs=1e-12)
    # realistic tight-oil GOR: flat 1.5 for a year then x4 over two years (true m ~ 1.26)
    rng = np.random.default_rng(2)
    t = np.arange(36) + 0.5
    truth = np.where(t < 12, 1.5, 1.5 * (t / 12.0) ** 1.26)
    obs = truth * np.exp(rng.normal(0, 0.06, 36))
    p = Y.fit_yield(t, obs)
    assert p.m == 1.0 and p.t0 < 12.0                  # breakpoint re-chosen for the bounded slope
    fit = Y.yield_curve(p, t)
    # a slope-1 line cannot follow a 1.26 power exactly, but the forecast must
    # start near the recent level rather than step down (the unconstrained
    # breakpoint with a clipped slope started ~9-24 % low in this scenario)
    assert 0.85 < np.mean(fit[-6:] / obs[-6:]) < 1.15
    assert Y.yield_curve(p, np.array([36.5]))[0] > 0.93 * obs[-6:].mean()


def test_fit_yield_declining_ratio_keeps_declining_in_the_forecast():
    """Water-cut-like declining WOR is handled (negative slopes), and the
    fitted trend is honoured beyond the history down to the symmetric floor
    rather than frozen at the last observed value."""
    t = np.arange(48) + 0.5
    obs = Y.plyield(5.0, -0.2, -0.6, 12.0, t)
    p = Y.fit_yield(t, obs)
    assert p is not None
    assert p.m0 == pytest.approx(-0.2, abs=0.05)
    assert p.m == pytest.approx(-0.6, abs=0.05)
    assert p.y_min == pytest.approx(np.quantile(obs, 0.05) / 1.5) and p.y_min < obs.min()
    fut = Y.yield_curve(p, np.array([60.5, 84.5, 120.5, 600.5]))
    assert fut[0] < obs.min() and fut[1] < fut[0] and fut[2] < fut[1]      # keeps declining
    assert fut[0] == pytest.approx(Y.plyield(5.0, -0.2, -0.6, 12.0, 60.5), rel=0.03)
    assert fut[-1] == pytest.approx(p.y_min)                               # ... down to the floor


# ---------------------------------------------------------------- forecast_secondary
def test_forecast_secondary_uses_midmonth_yield_and_zeroes_bad_primary():
    p = Y.YieldParams(c=2.0, m0=0.0, m=0.5, t0=10.0)
    t_index = np.array([10, 11, 12, 13, 14])
    prim = np.array([100.0, 0.0, np.nan, -5.0, 40.0])
    out = Y.forecast_secondary(p, t_index, prim)
    expect0 = Y.plyield(2.0, 0.0, 0.5, 10.0, 10.5) * 100.0
    assert out[0] == pytest.approx(expect0)
    assert out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0
    assert out[4] == pytest.approx(Y.plyield(2.0, 0.0, 0.5, 10.0, 14.5) * 40.0)
    assert np.all(np.isfinite(out))


def test_forecast_secondary_total_is_yield_weighted_primary():
    p = Y.YieldParams(c=1.0, m0=0.0, m=0.0, t0=5.0)   # constant yield 1 -> secondary == primary
    prim = np.linspace(100, 10, 50)
    out = Y.forecast_secondary(p, np.arange(50), prim)
    np.testing.assert_allclose(out, prim)
