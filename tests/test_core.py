"""Property and synthetic-recovery tests for TC_Tool's decline models and fitting
(core.py).  Run with:  cd TC_Tool && python -m pytest tests -q

Every test derives its expectations from closed forms or planted truth; no
external data is needed.  Tolerances are deliberately loose where noise and
model-selection ambiguity are inherent (see the SEPD/Arps confusion notes).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid

import core

D = core.D_LIM_DEFAULT


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _trap_cum(rate_fn, T, n=200_001):
    g = np.linspace(0.0, T, n)
    return float(trapezoid(rate_fn(g), g))


def _param_sets(seed=0, n=12):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(('Modified Arps', (rng.uniform(0.0, 2.0), rng.uniform(D, 0.6), D)))
        out.append(('SEPD', (rng.uniform(0.5, 300.0), rng.uniform(0.05, 1.0), D)))
        m = rng.uniform(1.01, 3.0)
        out.append(('Duong', (rng.uniform(0.001, m), m, D)))
        out.append(('Exponential', (rng.uniform(D, 0.5),)))
    return out


def synthetic_well(kind, params, n_months, noise, seed, qi=6000.0, ramp=0.55):
    """Monthly volumes from exact model cumulatives, one ramp-up month, lognormal noise.
    Returns (wd DataFrame, true EUR over history + 600 forecast months)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months, dtype=float)
    _, cum, _ = core.MODEL_REGISTRY[kind]
    N = cum(qi, *params, np.concatenate([t, t + 1.0]))
    V = N[n_months:] - N[:n_months]
    V0 = np.concatenate([[V[0] * ramp], V])
    remaining = float(cum(qi, *params, np.array([n_months + 600.0]))[0]
                      - cum(qi, *params, np.array([float(n_months)]))[0])
    rows = [dict(WellID='W', MonthYear=pd.Period('2018-01', 'M') + k,
                 NormOil=v * rng.lognormal(0.0, noise), NormGas=0.0, NormWater=0.0)
            for k, v in enumerate(V0)]
    return pd.DataFrame(rows), float(V0.sum()) + remaining


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
def test_terminal_decline_default_is_five_percent_effective():
    assert core.D_LIM_DEFAULT == pytest.approx(-np.log(0.95) / 12.0, rel=1e-12)
    assert core.effective_annual_from_nominal_monthly(core.D_LIM_DEFAULT) == pytest.approx(0.05, rel=1e-12)


@pytest.mark.parametrize("b", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("d_eff", [0.02, 0.05, 0.3, 0.8])
def test_decline_conversions_round_trip(b, d_eff):
    nom = core.nominal_monthly_from_effective_annual(d_eff, b)
    assert core.effective_annual_from_nominal_monthly(nom, b) == pytest.approx(d_eff, rel=1e-10)


# ---------------------------------------------------------------------------
# model functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _param_sets())
def test_closed_form_cumulative_matches_integral(name, params):
    rate, cum, _ = core.MODEL_REGISTRY[name]
    for T in (12.0, 240.0, 600.0):
        N = float(cum(1000.0, *params, np.array([T]))[0])
        I = _trap_cum(lambda g: rate(1000.0, *params, g), T)
        assert N == pytest.approx(I, rel=2e-5), (name, params, T)


@pytest.mark.parametrize("name,params", _param_sets(seed=1))
def test_monthly_volumes_pinned_positive_and_non_increasing(name, params):
    _, cum, _ = core.MODEL_REGISTRY[name]
    t = np.arange(0.0, 601.0)
    V, qi = core.model_volumes(cum, params, t, 5000.0)
    assert np.isfinite(V).all() and np.isfinite(qi) and qi > 0
    assert V[0] == pytest.approx(5000.0, rel=1e-9)
    assert np.all(np.diff(V) <= 1e-9 * V[:-1] + 1e-9)


@pytest.mark.parametrize("name,params", _param_sets(seed=2))
def test_terminal_floor_applies_to_every_model(name, params):
    """Late in life the nominal decline of every model is >= d_lim (never a slower tail)."""
    rate, _, _ = core.MODEL_REGISTRY[name]
    t = np.arange(0.0, 601.0)
    q = rate(1000.0, *params, t)
    with np.errstate(divide='ignore', invalid='ignore'):
        Dn = -np.diff(np.log(q))
    assert Dn[-1] >= D * 0.999, (name, params, Dn[-1])


@pytest.mark.parametrize("name,params,switch", [
    ('Modified Arps', (1.0, 0.1, D), core._arps_switch(1.0, 0.1, D)),
    ('SEPD', (10.0, 0.4, D), core._sepd_switch(10.0, 0.4, D)),
    ('Duong', (1.0, 1.3, D), core._duong_switch(1.0, 1.3, D)),
])
def test_rate_is_continuous_at_the_terminal_switch(name, params, switch):
    rate = core.MODEL_REGISTRY[name][0]
    assert np.isfinite(switch) and switch > 0
    eps = 1e-6
    q1 = rate(1000.0, *params, np.array([switch - eps]))[0]
    q2 = rate(1000.0, *params, np.array([switch + eps]))[0]
    assert abs(q1 - q2) / q1 < 1e-5


def test_small_b_does_not_overflow_and_tends_to_exponential():
    t = np.arange(0.0, 600.0)
    q = core.modified_arps(5000.0, 1e-6, 0.1, D, t[:30])
    assert np.isfinite(q).all()
    assert np.allclose(q, 5000.0 * np.exp(-0.1 * t[:30]), rtol=1e-4)
    assert np.allclose(core.modified_arps(5000.0, 0.0, 0.1, D, t), 5000.0 * np.exp(-0.1 * t))


def test_b_zero_honours_terminal_floor():
    t = np.arange(0.0, 120.0)
    assert np.allclose(core.modified_arps(5000.0, 0.0, 0.001, D, t), 5000.0 * np.exp(-D * t))


def test_sepd_cumulative_has_gamma_factor():
    """Without Gamma(1/n) the EUR would be wrong by +33 % at n = 0.4 (dca changelog)."""
    from scipy.special import gamma
    tau, n = 20.0, 0.4
    N_inf = float(core._sepd_raw_cum(1000.0, tau, n, np.array([1e9]))[0])
    assert N_inf == pytest.approx(1000.0 * tau / n * gamma(1.0 / n), rel=1e-6)


def test_sepd_switch_near_n_one_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert core._sepd_switch(50.0, 0.9999, D) == np.inf or core._sepd_switch(50.0, 0.9999, D) > 0


# ---------------------------------------------------------------------------
# fitting / selection on planted truth
# ---------------------------------------------------------------------------
def test_peak_is_snapped_to_the_raw_maximum():
    """A centred 3-month smoother peaks a month late on steep declines; the peak
    index must be the raw maximum (month 1 here, after a 55 % ramp-up month)."""
    hits = 0
    for s in range(20):
        wd, _ = synthetic_well('Modified Arps', (1.0, 0.25, D), 36, 0.05, s)
        fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
        hits += fc['peak_idx'] == 1
    assert hits >= 18


def test_recovers_hyperbolic_truth_at_60_months():
    errs, bs = [], []
    for s in range(12):
        wd, te = synthetic_well('Modified Arps', (1.0, 0.12, D), 60, 0.05, s)
        fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
        errs.append(fc['EUR_total'] / te - 1.0)
        bs.append(fc['b'])
    assert abs(np.median(errs)) < 0.03
    assert np.percentile(np.abs(errs), 90) < 0.10
    assert abs(np.median(bs) - 1.0) < 0.1


def test_recovers_exponential_truth_in_vertical_mode():
    errs = []
    for s in range(12):
        wd, te = synthetic_well('Exponential', (0.03,), 48, 0.05, s)
        fc = core.forecast_one_well(wd, 'oil', 0.0, 0.5, 600, d_lim=D,
                                    allow_exponential=True, exclude_duong=True)
        errs.append(fc['EUR_total'] / te - 1.0)
        assert fc['b'] < 0.3, "exponential truth must come out near-exponential"
        assert 'Duong' not in fc['bics']
    assert abs(np.median(errs)) < 0.05


def test_recovers_sepd_truth_without_terminal_bias():
    """SEPD truths used to be forecast 30-36 % high by Modified Arps / Duong."""
    errs = []
    for s in range(12):
        wd, te = synthetic_well('SEPD', (40.0, 0.3, D), 36, 0.05, s)
        fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
        errs.append(fc['EUR_total'] / te - 1.0)
    assert abs(np.median(errs)) < 0.08
    assert np.percentile(np.abs(errs), 90) < 0.25


def test_slow_decliner_keeps_its_fitted_arps_di():
    """The former post-fit override rewrote any di < 3*d_lim to max(di0, 0.03),
    disqualifying Modified Arps on every well declining slower than ~14 %/yr."""
    wd, _ = synthetic_well('Exponential', (0.008,), 60, 0.03, 1)
    fc = core.forecast_one_well(wd, 'oil', 0.0, 0.5, 600, d_lim=D,
                                allow_exponential=True, exclude_duong=True)
    assert fc['di_arps'] < 0.012
    assert fc['di'] == pytest.approx(0.008, abs=0.003)


def test_duong_is_gated_on_short_histories():
    wd, _ = synthetic_well('Modified Arps', (1.0, 0.12, D), 12, 0.05, 2)
    fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
    assert 'Duong' not in fc['bics']
    wd, _ = synthetic_well('Modified Arps', (1.0, 0.12, D), 36, 0.05, 2)
    fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D, duong_min_months=18)
    assert 'Duong' in fc['bics']


def test_qi_cap_is_optional_and_honoured():
    wd, _ = synthetic_well('Modified Arps', (1.0, 0.12, D), 24, 0.0, 3, qi=300_000.0)
    fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
    assert fc['v_peak_obs'] > 152_000.0, "no implicit 152,000 cap any more"
    fc2 = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D, qi_cap=100_000.0)
    assert fc2['v_peak_obs'] == pytest.approx(100_000.0)
    assert core.PEAK_SCALE_LO <= fc2['peak_scale'] <= core.PEAK_SCALE_HI


def test_soft_peak_pin_absorbs_peak_month_noise():
    """A hard pin to a +12 % peak-month outlier forced the Arps fit to a bound (or
    handed the well to SEPD); the soft pin lets the scale absorb the outlier so the
    hyperbolic shape and EUR are recovered from the rest of the decline."""
    wd, te = synthetic_well('Modified Arps', (1.0, 0.12, D), 48, 0.0, 5)
    wd = wd.copy()
    wd.loc[wd.index[1], 'NormOil'] *= 1.12          # the peak month (index 1 after the ramp)
    fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
    assert fc['model'] == 'Modified Arps'
    assert fc['b'] == pytest.approx(1.0, abs=0.1)
    assert fc['peak_scale'] < 1.0                   # part of the outlier absorbed by the scale
    assert fc['EUR_total'] == pytest.approx(te, rel=0.05)


def test_invalid_configuration_raises():
    wd, _ = synthetic_well('Modified Arps', (1.0, 0.12, D), 24, 0.05, 6)
    with pytest.raises(ValueError):
        core.forecast_one_well(wd, 'oil', 1.2, 0.8, 600, d_lim=D)
    with pytest.raises(ValueError):
        core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=0.0)


def test_hindcast_rejects_training_windows_without_decline():
    """A hold-out that leaves only the ramp-up in training must not report seed
    parameters as a validated forecast."""
    rng = np.random.default_rng(8)
    ramp = np.linspace(500, 8000, 8)
    t = np.arange(22, dtype=float)
    N = core.modified_arps_cum(8000.0, 1.0, 0.1, D, np.concatenate([t, t + 1]))
    V = np.concatenate([ramp, (N[22:] - N[:22])[1:]])
    wd = pd.DataFrame([dict(WellID='W', MonthYear=pd.Period('2018-01', 'M') + k, NormOil=v,
                            NormGas=0.0, NormWater=0.0) for k, v in enumerate(V)])
    assert core.hindcast_one_well(wd, 'oil', 0.8, 1.2, holdout_months=22, d_lim=D) is None
    hc = core.hindcast_one_well(wd, 'oil', 0.8, 1.2, holdout_months=12, d_lim=D)
    assert hc is not None and hc['n_fit_train'] >= 4


def test_forecast_dict_is_finite_and_consistent():
    wd, _ = synthetic_well('Modified Arps', (0.9, 0.15, D), 30, 0.08, 4)
    fc = core.forecast_one_well(wd, 'oil', 0.8, 1.2, 600, d_lim=D)
    for k in ('qi', 'b', 'di', 'EUR_total', 'EUR_fcst', 'decline_1yr_pct', 'fit_r2_log', 'fit_mape_pct'):
        assert np.isfinite(fc[k]), k
    assert fc['EUR_fcst'] == pytest.approx(float(np.sum(fc['f_vals'])), rel=1e-6)
    assert len(fc['fit_hist']) == fc['n_months']
    assert fc['f_months'][0] == fc['n_post']
    assert 0.0 < fc['decline_1yr_pct'] < 100.0
    assert set(fc['params']) == {'b', 'di'} if fc['model'] == 'Modified Arps' else True


# ---------------------------------------------------------------------------
# population functions
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def merged_population():
    rows = []
    rng = np.random.default_rng(7)
    for w in range(12):
        qi, di = rng.uniform(2000, 8000), rng.uniform(0.08, 0.2)
        wd, _ = synthetic_well('Modified Arps', (1.0, di, D), 36, 0.05, w, qi=qi)
        wd = wd.copy()
        wd['WellID'] = f'W{w}'
        wd['WellName'] = f'Well {w}'
        wd['API10'] = f'42{w:08d}'
        wd['LateralLength'] = 9000.0
        rows.append(wd)
    return pd.concat(rows, ignore_index=True)


def test_forecast_all_outputs(merged_population):
    cfg = core.ForecastConfig('oil', 0.8, 1.2)
    one, mon, fcs = core.forecast_all(merged_population, cfg)
    assert len(one) == 12 and len(fcs) == 12
    for c in ('Model', 'Model Params', 'b at bound', 'Terminal Decline (%/yr eff)',
              'Fit MAPE (%)', 'Fit R² (log)', 'EUR (Mbbl)', 'qi (per day)'):
        assert c in one.columns, c
    assert one['Terminal Decline (%/yr eff)'].iloc[0] == pytest.approx(5.0, abs=0.01)
    assert set(mon['Segment']) == {'Historical', 'Forecast'}
    assert (mon.groupby('WellID')['Segment'].apply(lambda s: (s == 'Forecast').sum()) <= 600).all()


def test_hindcast_all_returns_metrics(merged_population):
    cfg = core.ForecastConfig('oil', 0.8, 1.2)
    hc = core.hindcast_all(merged_population, cfg, holdout_months=12)
    assert len(hc) == 12
    assert np.isfinite(hc['Cum error (%)']).all() and np.isfinite(hc['MAPE (%)']).all()
    assert hc['Test months'].eq(12).all()
    assert hc['Cum error (%)'].abs().median() < 15.0


def test_hindcast_fraction_mode_and_too_short(merged_population):
    wd = merged_population[merged_population['WellID'] == 'W0']
    hc = core.hindcast_one_well(wd, 'oil', 0.8, 1.2, train_frac=0.5, d_lim=D)
    assert hc is not None and hc['n_train'] == 18 and hc['n_test'] == 19
    assert core.hindcast_one_well(wd.iloc[:7], 'oil', 0.8, 1.2, d_lim=D) is None
