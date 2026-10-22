"""Tests for diagnostics.py — Bourdet derivative and flow-regime diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

import core
import diagnostics as DG

DPY = 365.25


def _mh_days(t, qi=1000.0, Di_sec=0.8, b=1.5, Dterm_tan=0.05):
    """Modified-hyperbolic rate in day units with the petbox-dca conventions
    (secant-effective Di, tangent-effective terminal), used only to reproduce
    its published reference derivative values."""
    Dnom = math.expm1(b * -math.log1p(-Di_sec)) / b / DPY
    Dt = -math.log1p(-Dterm_tan) / DPY
    t_term = (1.0 / Dt - 1.0 / Dnom) / b
    t = np.asarray(t, dtype=float)
    q_h = qi * np.exp(-np.log1p(Dnom * b * t) / b)
    q_t = qi * np.exp(-np.log1p(Dnom * b * t_term) / b)
    return np.where(t < t_term, q_h, q_t * np.exp(-Dt * (t - t_term)))


# ---------------------------------------------------------------- bourdet
def test_bourdet_reproduces_published_reference_values():
    t = np.logspace(0, 4, 200)
    d = DG.bourdet(_mh_days(t), t, L=0.5)
    assert d[0] == pytest.approx(-31.884, abs=2e-3)
    assert d[100] == pytest.approx(-189.354, abs=2e-3)


def test_bourdet_power_law_gives_constant_log_slope():
    t = np.logspace(0, 3, 300)
    y = 50.0 * t ** -0.5
    d = DG.bourdet(np.log(y), t, L=0.2)                 # d ln y / d ln t
    np.testing.assert_allclose(d, -0.5, atol=1e-9)
    d0 = DG.bourdet(np.log(y), t, L=0.0)
    np.testing.assert_allclose(d0, -0.5, atol=1e-9)


def test_bourdet_exponential_gives_constant_D_with_xlog_false():
    t = np.linspace(0.5, 120.5, 241)
    D = 0.02
    q = 100.0 * np.exp(-D * t)
    der = DG.bourdet(q, t, L=0.0, xlog=False, ylog=True)   # d ln q / dt = -D
    np.testing.assert_allclose(der, -D, atol=1e-9)


def test_bourdet_skips_nan_and_keeps_positions():
    t = np.linspace(1, 50, 50)
    y = 3.0 * t
    y[[4, 20]] = np.nan
    d = DG.bourdet(y, t, L=0.0, xlog=False)              # dy/dx = 3
    assert np.isnan(d[4]) and np.isnan(d[20])
    good = np.isfinite(d)
    np.testing.assert_allclose(d[good], 3.0, atol=1e-9)
    assert good.sum() == 48


def test_bourdet_rejects_non_increasing_x_and_bad_shapes():
    with pytest.raises(ValueError):
        DG.bourdet([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        DG.bourdet([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])      # non-positive x
    with pytest.raises(ValueError):
        DG.bourdet([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        DG.bourdet([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], L=-0.1)


def test_bourdet_too_few_points_returns_nan():
    assert np.all(np.isnan(DG.bourdet([1.0], [1.0])))
    assert np.all(np.isnan(DG.bourdet([np.nan, 2.0, np.nan], [1.0, 2.0, 3.0])))


# ---------------------------------------------------------------- local_slope
def test_local_slope_exact_for_linear_and_skips_nan():
    x = np.arange(50) + 0.5
    y = 4.0 - 0.3 * x
    s = DG.local_slope(x, y, half_width=6.0)
    np.testing.assert_allclose(s, -0.3, atol=1e-12)
    y[[3, 20, 21]] = np.nan
    s = DG.local_slope(x, y, half_width=6.0)
    assert np.isnan(s[3]) and np.isnan(s[20]) and np.isnan(s[21])
    np.testing.assert_allclose(s[np.isfinite(s)], -0.3, atol=1e-12)


def test_local_slope_widens_window_to_min_points():
    x = np.array([0.5, 1.5, 2.5, 30.5, 31.5, 32.5])
    y = 2.0 * x
    s = DG.local_slope(x, y, half_width=0.1, min_points=3)   # window holds only the point itself
    np.testing.assert_allclose(s, 2.0, atol=1e-12)
    assert np.all(np.isnan(DG.local_slope([1.0], [1.0], 5.0)))
    assert np.all(np.isnan(DG.local_slope([1.0, 2.0], [np.nan, 1.0], 5.0)))


def test_local_slope_averages_noise():
    rng = np.random.default_rng(0)
    x = np.arange(200) + 0.5
    y = 0.5 * x + rng.normal(0, 1.0, 200)
    s_narrow = DG.local_slope(x, y, half_width=1.0)
    s_wide = DG.local_slope(x, y, half_width=12.0)
    assert np.std(s_wide[20:-20]) < 0.25 * np.std(s_narrow[20:-20])
    assert abs(np.mean(s_wide[20:-20]) - 0.5) < 0.02


# ---------------------------------------------------------------- model_diagnostics
@pytest.mark.parametrize("model,params,rate_fn", [
    ('Modified Arps', {'qi': 5000.0, 'b': 1.1, 'di': 0.15},
     lambda p, x, dl: core.modified_arps(p['qi'], p['b'], p['di'], dl, x)),
    ('Exponential', {'qi': 5000.0, 'di': 0.03},
     lambda p, x, dl: core.exponential(p['qi'], p['di'], x)),
    ('SEPD', {'qi': 5000.0, 'tau': 8.0, 'n': 0.5},
     lambda p, x, dl: core.sepd(p['qi'], p['tau'], p['n'], dl, x)),
    ('Duong', {'qi': 5000.0, 'a': 1.2, 'm': 1.25},
     lambda p, x, dl: core.duong(p['qi'], p['a'], p['m'], dl, x)),
])
def test_model_diagnostics_matches_core_rate_functions(model, params, rate_fn):
    d_lim = core.D_LIM_DEFAULT
    t = np.linspace(0.5, 300.5, 601)
    md = DG.model_diagnostics(model, params, t, d_lim)
    h = 1e-4
    D_fd = -(np.log(rate_fn(params, t + h, d_lim)) - np.log(rate_fn(params, t - h, d_lim))) / (2 * h)
    # Core switches to the floor exactly where D reaches d_lim; skip the one
    # grid point that may straddle the switch.
    straddle = np.abs(D_fd - d_lim) < 1e-3 * d_lim
    ok = ~straddle & np.isfinite(D_fd) & (D_fd > 0)
    np.testing.assert_allclose(md['D'][ok], D_fd[ok], rtol=1e-6)
    np.testing.assert_allclose(md['beta'][ok], 1.0 / D_fd[ok], rtol=1e-6)
    # b = d(1/D)/dt by central differences on the analytic D
    md_p = DG.model_diagnostics(model, params, t + h, d_lim)
    md_m = DG.model_diagnostics(model, params, t - h, d_lim)
    b_fd = (md_p['beta'] - md_m['beta']) / (2 * h)
    off_floor = ok & (md['D'] > d_lim * (1 + 1e-9))
    np.testing.assert_allclose(md['b'][off_floor], b_fd[off_floor], atol=1e-4, rtol=1e-4)
    on_floor = md['D'] <= d_lim * (1 + 1e-12)
    assert np.all(md['b'][on_floor] == 0.0)
    assert np.all(md['D'] >= d_lim * (1 - 1e-12))


def test_model_diagnostics_arps_b_is_constant_before_floor():
    t = np.arange(0.5, 100.5)
    md = DG.model_diagnostics('Modified Arps', {'qi': 1.0, 'b': 1.3, 'di': 0.2}, t, 0.004)
    before = md['D'] > 0.004
    assert before.sum() > 10
    np.testing.assert_allclose(md['b'][before], 1.3)
    assert np.all(np.diff(md['D']) <= 1e-15)        # monotone non-increasing


def test_model_diagnostics_exponential_without_floor():
    t = np.arange(0.5, 60.5)
    md = DG.model_diagnostics('Exponential', {'qi': 1.0, 'di': 0.05}, t, 0.0)
    np.testing.assert_allclose(md['D'], 0.05)
    np.testing.assert_allclose(md['b'], 0.0)


def test_model_diagnostics_sepd_b_is_decreasing_power_law():
    t = np.array([1.0, 4.0, 16.0])
    md = DG.model_diagnostics('SEPD', {'qi': 1.0, 'tau': 10.0, 'n': 0.5}, t, 0.0)
    # b = (1-n) tau^n t^-n / n -> halves when t quadruples (n = 0.5)
    assert md['b'][1] == pytest.approx(md['b'][0] / 2.0)
    assert md['b'][2] == pytest.approx(md['b'][1] / 2.0)


def test_model_diagnostics_duong_floor_applies_after_last_crossing():
    # Duong D starts low (t' = 1: D = m - a), rises, then falls below d_lim.
    t = np.arange(0.5, 400.5)
    md = DG.model_diagnostics('Duong', {'qi': 1.0, 'a': 1.2, 'm': 1.25}, t, core.D_LIM_DEFAULT)
    sw = core._duong_switch(1.2, 1.25, core.D_LIM_DEFAULT)
    assert np.isfinite(sw) and sw > 1
    on_floor = md['D'] <= core.D_LIM_DEFAULT * (1 + 1e-12)
    assert np.all(on_floor[t > sw + 1.0])
    assert not np.any(on_floor[(t > 1.5) & (t < sw - 1.0)])


def test_model_diagnostics_unknown_model_raises():
    with pytest.raises(ValueError):
        DG.model_diagnostics('Nope', {'qi': 1.0}, np.arange(1.0, 5.0), 0.004)


# ---------------------------------------------------------------- data_diagnostics
def test_data_diagnostics_recovers_hyperbolic_D_and_b():
    b, di = 1.2, 0.15
    t = np.arange(120) + 0.5
    q = core.arps(5000.0, b, di, t)                        # instantaneous rates, no floor
    dd = DG.data_diagnostics(t, q)
    assert set(dd) == {'t', 'q', 'D', 'beta', 'b', 'loglog_slope'}
    assert dd['t'].size == 120
    D_true = di / (1.0 + b * di * dd['t'])
    # D changes fastest in the first year, where a 13-month window averages
    # over it; beyond that the regression derivative is within a few percent.
    np.testing.assert_allclose(dd['D'][12:-1], D_true[12:-1], rtol=0.03)
    np.testing.assert_allclose(dd['D'][1:12], D_true[1:12], rtol=0.25)
    assert abs(np.nanmedian(dd['b'][10:-10]) - b) < 0.1
    assert np.nanmin(dd['b'][12:-12]) > b - 0.1 and np.nanmax(dd['b'][12:-12]) < b + 0.1
    # late-time log-log slope tends to -1/b for a hyperbolic
    assert dd['loglog_slope'][100] == pytest.approx(-1.0 / b * (b * di * t[100]) / (1 + b * di * t[100]), abs=0.05)


def test_data_diagnostics_two_regime_b_transition():
    """b = 2 transient (18 mo) then b = 0.8 BDF: the data b must fall below 1
    within a few months of the switch."""
    t = np.arange(60) + 0.5
    b1, b2, di = 2.0, 0.8, 0.25
    t_sw = 18.0
    q1 = core.arps(5000.0, b1, di, t)
    D_sw = di / (1 + b1 * di * t_sw)
    q_sw = core.arps(5000.0, b1, di, t_sw)
    q = np.where(t < t_sw, q1, core.arps(q_sw, b2, D_sw, t - t_sw))
    dd = DG.data_diagnostics(t, q)
    early = dd['b'][3:15]
    late = dd['b'][30:-6]
    assert np.nanmedian(early) == pytest.approx(2.0, abs=0.35)
    assert np.nanmedian(late) == pytest.approx(0.8, abs=0.1)
    assert np.nanmedian(early) > 1.5 > np.nanmedian(late)
    # b must drop below 1 within a year of the regime change
    first_below = np.flatnonzero(dd['b'][18:] < 1.0)
    assert first_below.size and first_below[0] <= 12
    # ... and the data D must have flattened relative to a b = 2 extrapolation
    D_b2 = di / (1 + b1 * di * dd['t'])
    assert np.all(dd['D'][30:-6] > 1.3 * D_b2[30:-6])


def test_data_diagnostics_noisy_hyperbolic_is_readable():
    """With 5 % lognormal noise the early-time b (transient, b = 2) is still
    identifiable from the median of the first 15 months."""
    t = np.arange(48) + 0.5
    q = core.arps(5000.0, 2.0, 0.25, t)
    med = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        dd = DG.data_diagnostics(t, q * np.exp(rng.normal(0, 0.05, t.size)))
        med.append(np.nanmedian(dd['b'][3:15]))
    assert np.median(med) == pytest.approx(2.0, abs=0.5)
    assert min(med) > 1.0


def test_data_diagnostics_drops_shutins_and_garbage():
    t = np.arange(40) + 0.5
    q = core.arps(3000.0, 1.0, 0.1, t)
    q[[5, 17]] = 0.0            # shut-in months
    q[23] = np.nan
    q[30] = 10.0                # curtailed month (< 2 % of max)
    dd = DG.data_diagnostics(t, q)
    assert dd['t'].size == 36
    assert not np.any(np.isin(dd['t'], t[[5, 17, 23, 30]]))
    assert np.all(np.isfinite(dd['D'][1:-1]))


def test_data_diagnostics_handles_tiny_inputs():
    out = DG.data_diagnostics(np.array([0.5, 1.5]), np.array([10.0, 8.0]))
    assert out['t'].size == 2 and np.all(np.isnan(out['D']))
    out0 = DG.data_diagnostics(np.array([]), np.array([]))
    assert out0['t'].size == 0 and out0['b'].size == 0
    with pytest.raises(ValueError):
        DG.data_diagnostics(np.array([1.0, 2.0]), np.array([1.0]))


def test_data_diagnostics_incline_gives_nan_beta_not_crash():
    t = np.arange(30) + 0.5
    q = 100.0 + 5.0 * t                                   # inclining production
    dd = DG.data_diagnostics(t, q)
    assert np.all(dd['D'][1:-1] < 0)
    assert np.all(np.isnan(dd['beta'][1:-1]))


def test_reference_slopes():
    rs = DG.reference_slopes()
    assert rs['linear flow'] == -0.5
    assert all(v < 0 for v in rs.values())
