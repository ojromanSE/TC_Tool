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


def test_bourdet_weighting_is_exact_for_quadratics_on_nonuniform_grid():
    """SPE-12777 Eq. 8 weights each secant by the OPPOSITE interval, which makes
    the 3-point estimate exact for a quadratic on any grid; swapped weights are
    not (errors 0.5-3.8 on this grid)."""
    x = np.array([1, 1.3, 2.1, 2.4, 4, 4.3, 7, 7.1, 11, 15.0])
    d = DG.bourdet(x ** 2, x, L=0.0, xlog=False)              # dy/dx = 2x
    np.testing.assert_allclose(d[1:-1], 2 * x[1:-1], atol=1e-10)
    lx = np.log(x)
    d2 = DG.bourdet(lx ** 2, x, L=0.0, xlog=True)             # dy/d ln x = 2 ln x
    np.testing.assert_allclose(d2[1:-1], 2 * lx[1:-1], atol=1e-10)


def test_bourdet_window_ties_are_symmetric():
    """A sample exactly L cycles away is 'at least L away' on both sides."""
    x = 10.0 ** np.arange(12)
    y = np.log(x) ** 3
    d = DG.bourdet(y, x, L=1.0)
    # symmetric +/- one-decade window -> exact centred 3-point value
    lx = np.log(x)
    i = 5
    sL = (y[i] - y[i - 1]) / (lx[i] - lx[i - 1]); sR = (y[i + 1] - y[i]) / (lx[i + 1] - lx[i])
    assert d[i] == pytest.approx((sL + sR) / 2, rel=1e-12)


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


def test_local_slope_widens_window_to_nearest_points():
    x = np.array([0.5, 1.5, 2.5, 30.5, 31.5, 32.5])
    y = x ** 2                                     # slope depends on which points are used
    s = DG.local_slope(x, y, half_width=0.1, min_points=3)   # window holds only the point itself
    # nearest three points: {0.5,1.5,2.5} for the first cluster (LS slope 3.0),
    # {30.5,31.5,32.5} for the second (LS slope 63.0); never across the gap
    np.testing.assert_allclose(s[:3], 3.0, atol=1e-9)
    np.testing.assert_allclose(s[3:], 63.0, atol=1e-9)
    assert np.all(np.isnan(DG.local_slope([1.0], [1.0], 5.0)))
    assert np.all(np.isnan(DG.local_slope([1.0, 2.0], [np.nan, 1.0], 5.0)))


def test_local_slope_unsorted_input_matches_sorted():
    rng = np.random.default_rng(5)
    x = np.arange(40) + 0.5
    y = x ** 2
    perm = rng.permutation(40)
    s_sorted = DG.local_slope(x, y, 6.0)
    s_perm = DG.local_slope(x[perm], y[perm], 6.0)
    np.testing.assert_allclose(s_perm, s_sorted[perm], atol=1e-9)
    np.testing.assert_allclose(s_sorted[6:-6], 2 * x[6:-6], atol=1e-9)   # exact for quadratics on symmetric windows


def test_local_slope_averages_noise():
    rng = np.random.default_rng(0)
    x = np.arange(200) + 0.5
    y = 0.5 * x + rng.normal(0, 1.0, 200)
    s_narrow = DG.local_slope(x, y, half_width=1.0)
    s_wide = DG.local_slope(x, y, half_width=12.0)
    assert np.std(s_wide[20:-20]) < 0.25 * np.std(s_narrow[20:-20])
    assert abs(np.mean(s_wide[20:-20]) - 0.5) < 0.02


# ---------------------------------------------------------------- model_diagnostics
def _core_switch(model, params, d_lim):
    if model == 'Modified Arps' and params['b'] > 0:
        return core._arps_switch(params['b'], params['di'], d_lim)
    if model == 'SEPD':
        return core._sepd_switch(params['tau'], params['n'], d_lim)
    if model == 'Duong':
        return core._duong_switch(params['a'], params['m'], d_lim)
    return 0.0 if params['di'] <= d_lim else np.inf          # Exponential: floor or never


@pytest.mark.parametrize("model,params,rate_fn", [
    ('Modified Arps', {'qi': 5000.0, 'b': 1.1, 'di': 0.15},
     lambda p, x, dl: core.modified_arps(p['qi'], p['b'], p['di'], dl, x)),
    ('Modified Arps', {'qi': 5000.0, 'b': 0.0, 'di': 0.03},
     lambda p, x, dl: core.modified_arps(p['qi'], p['b'], p['di'], dl, x)),
    ('Exponential', {'qi': 5000.0, 'di': 0.03},
     lambda p, x, dl: core.exponential(p['qi'], p['di'], x)),
    ('SEPD', {'qi': 5000.0, 'tau': 8.0, 'n': 0.5},
     lambda p, x, dl: core.sepd(p['qi'], p['tau'], p['n'], dl, x)),
    ('SEPD', {'qi': 5000.0, 'tau': 30.0, 'n': 0.3},            # switch inside the grid
     lambda p, x, dl: core.sepd(p['qi'], p['tau'], p['n'], dl, x)),
    ('Duong', {'qi': 5000.0, 'a': 1.2, 'm': 1.25},
     lambda p, x, dl: core.duong(p['qi'], p['a'], p['m'], dl, x)),
])
@pytest.mark.parametrize("d_lim", [core.D_LIM_DEFAULT, 0.0])
def test_model_diagnostics_matches_core_rate_functions(model, params, rate_fn, d_lim):
    t = np.linspace(0.5, 300.5, 601)
    step = t[1] - t[0]
    md = DG.model_diagnostics(model, params, t, d_lim)
    h = 1e-4
    D_fd = -(np.log(rate_fn(params, t + h, d_lim)) - np.log(rate_fn(params, t - h, d_lim))) / (2 * h)
    t_sw = _core_switch(model, params, d_lim) if d_lim > 0 else np.inf
    near_switch = np.abs(t - t_sw) <= step if np.isfinite(t_sw) else np.zeros(t.shape, bool)
    ok = ~near_switch & np.isfinite(D_fd) & (D_fd > 0)
    assert ok.sum() > 500
    # D and 1/D agree with core everywhere, including on the floor
    np.testing.assert_allclose(md['D'][ok], D_fd[ok], rtol=1e-6)
    np.testing.assert_allclose(md['beta'][ok], 1.0 / D_fd[ok], rtol=1e-6)
    # b = d(1/D)/dt by central differences on the analytic D
    md_p = DG.model_diagnostics(model, params, t + h, d_lim)
    md_m = DG.model_diagnostics(model, params, t - h, d_lim)
    b_fd = (md_p['beta'] - md_m['beta']) / (2 * h)
    np.testing.assert_allclose(md['b'][ok], b_fd[ok], atol=1e-4, rtol=1e-4)
    if d_lim > 0:
        floor = t > t_sw + step
        if t_sw < t[-1] - 20 * step:
            assert floor.sum() > 10                     # these cases do reach the floor in-grid
        assert floor.sum() > 10 or model == 'Exponential' or params.get('b') == 0.0 or params.get('tau') == 8.0
        np.testing.assert_allclose(md['D'][floor], d_lim, rtol=1e-9)
        np.testing.assert_allclose(D_fd[floor], d_lim, rtol=1e-6)
        assert np.all(md['b'][floor] == 0.0)
        assert np.all(md['D'] >= d_lim * (1 - 1e-12))
    else:
        assert np.all(md['b'][ok] != 0.0) or model == 'Exponential' or params.get('b') == 0.0


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
    # With a = m Duong's D is exactly 0 at t = 0 (t' = 1: D = m - a), rises
    # above d_lim, then falls below it for good: the early dip must NOT be
    # treated as the floor (a naive D <= d_lim rule would floor it).
    d_lim = core.D_LIM_DEFAULT
    t = np.arange(0.0, 400.0)
    params = {'qi': 1.0, 'a': 1.2, 'm': 1.2}
    md = DG.model_diagnostics('Duong', params, t, d_lim)
    raw_D0 = params['m'] / 1.0 - params['a'] * 1.0 ** (-params['m'])
    assert raw_D0 == pytest.approx(0.0)
    assert md['D'][0] == pytest.approx(0.0, abs=1e-12) and np.isnan(md['beta'][0])
    assert md['D'][1] > d_lim and md['b'][1] != 0.0
    sw = core._duong_switch(1.2, 1.2, d_lim)
    assert np.isfinite(sw) and sw > 1
    on_floor = md['D'] <= d_lim * (1 + 1e-12)
    assert np.all(on_floor[t > sw + 1.0])
    assert not np.any(on_floor[(t >= 1.0) & (t < sw - 1.0)])
    # interior matches core's finite-difference D
    h = 1e-4
    D_fd = -(np.log(core.duong(1.0, 1.2, 1.2, d_lim, t[1:] + h)) - np.log(core.duong(1.0, 1.2, 1.2, d_lim, t[1:] - h))) / (2 * h)
    inner = np.abs(t[1:] - sw) > 1.0
    np.testing.assert_allclose(md['D'][1:][inner], D_fd[inner], rtol=1e-6)


def test_model_diagnostics_unknown_model_raises():
    with pytest.raises(ValueError):
        DG.model_diagnostics('Nope', {'qi': 1.0}, np.arange(1.0, 5.0), 0.004)


# ---------------------------------------------------------------- data_diagnostics
def test_data_diagnostics_recovers_hyperbolic_D_and_b():
    b, di = 1.2, 0.15
    t = np.arange(120) + 0.5
    q = core.arps(5000.0, b, di, t)                        # instantaneous rates, no floor
    dd = DG.data_diagnostics(t, q)
    assert set(dd) == {'t', 'q', 'D', 'beta', 'b', 'loglog_slope', 'n_dropped'}
    assert dd['t'].size == 120 and dd['n_dropped'] == 0
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
    q[30] = 10.0                # curtailed month (~1 % of its neighbours)
    dd = DG.data_diagnostics(t, q)
    assert dd['t'].size == 36 and dd['n_dropped'] == 1
    assert not np.any(np.isin(dd['t'], t[[5, 17, 23, 30]]))
    assert np.all(np.isfinite(dd['D'][1:-1]))


def test_data_diagnostics_keeps_legitimate_late_tail_of_steep_declines():
    """The shut-in test is relative to the local level, not the peak: on a
    steep decline the late months are a tiny fraction of the peak yet on trend
    and must be kept (they are the boundary-dominated period the panel shows)."""
    t = np.arange(120) + 0.5
    for V in (core.model_volumes(core.modified_arps_cum, (0.5, 0.30, core.D_LIM_DEFAULT), np.arange(120), 1e4)[0],
              core.model_volumes(core.sepd_cum, (2.0, 0.5, core.D_LIM_DEFAULT), np.arange(120), 1e4)[0],
              core.exponential(1e4, 0.05, t)):
        assert V[-1] / V.max() < 0.02
        dd = DG.data_diagnostics(t, V)
        assert dd['t'].size == 120 and dd['n_dropped'] == 0
    # ... while a single curtailed month inside that tail is still removed
    V = core.exponential(1e4, 0.05, t)
    V[100] *= 0.05
    dd = DG.data_diagnostics(t, V)
    assert dd['t'].size == 119 and dd['n_dropped'] == 1 and 100.5 not in dd['t']


def test_data_diagnostics_collapses_duplicate_times():
    t = np.array([0.5, 1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    q = core.arps(1000.0, 1.0, 0.3, t)
    q[2] *= 1.2                                      # second sample of month 1.5
    dd = DG.data_diagnostics(t, q)
    assert dd['t'].size == 8
    assert dd['q'][1] == pytest.approx((q[1] + q[2]) / 2)
    assert np.all(np.isfinite(dd['D'])) and np.all(np.isfinite(dd['loglog_slope']))
    # a duplicate that is a zero is filtered first and does not average in
    q2 = q.copy(); q2[2] = 0.0
    dd2 = DG.data_diagnostics(t, q2)
    assert dd2['t'].size == 8 and dd2['q'][1] == pytest.approx(q[1])


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
