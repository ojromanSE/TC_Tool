from __future__ import annotations
import matplotlib
# Fix for headless/cloud environments
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from scipy.optimize import minimize, brentq
from scipy.special import gamma, gammainc
from scipy.integrate import cumulative_trapezoid
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy import stats

# ---------------- Provider column maps ----------------
# Added 'WellNumber' to requirements and maps
REQUIRED_HEADER_COLUMNS = [
    'WellName','WellNumber','County','LateralLength','PrimaryFormation',
    'CompletionDate','FirstProdDate','State','API10'
]
REQUIRED_PROD_COLUMNS = [
    'WellName','WellNumber','ReportDate','TotalOil','TotalGas','TotalWater','API10'
]

HEADER_COLUMN_MAPS = [
    # WDB-style
    {'WellName':'WellName','WellNumber':'WellNumber','County':'County','LateralLength':'LateralLength',
     'PrimaryFormation':'PrimaryFormation','CompletionDate':'CompletionDate',
     'FirstProdDate':'FirstProdDate','State':'State','API10':'API'},
    # DI / IHS-style
    {'WellName':'Well Name','WellNumber':'Well Number','County':'County/Parish','LateralLength':'DI Lateral Length',
     'PrimaryFormation':'Producing Reservoir','CompletionDate':'Completion Date',
     'FirstProdDate':'First Prod Date','State':'State','API10':'API10'}
]
PROD_COLUMN_MAPS = [
    # WDB-style
    {'WellName':'WellName','WellNumber':'WellNumber','ReportDate':'ReportDate','TotalOil':'TotalOil',
     'TotalGas':'TotalGas','TotalWater':'TotalWater','API10':'API'},
    # DI / IHS-style
    {'WellName':'Well Name','WellNumber':'Well Number','ReportDate':'Monthly Production Date','TotalOil':'Monthly Oil',
     'TotalGas':'Monthly Gas','TotalWater':'Monthly Water','API10':'API10'}
]

def _normalize_columns_before_map(df: pd.DataFrame) -> pd.DataFrame:
    # 1. API Logic (still useful for display/QC)
    if 'API10' not in df.columns:
        for alias in ['API14', 'API12', 'API/UWI', 'API']:
            if alias in df.columns:
                df['API10'] = df[alias].astype(str).str.replace(r'\.0$', '', regex=True).str[:10]
                break

    # 2. Well Number Aliases
    if 'Well Number' not in df.columns and 'WellNumber' not in df.columns:
        if 'WellNumber' in df.columns:
            df['Well Number'] = df['WellNumber']

    # 3. Lateral Length Aliases
    if 'DI Lateral Length' not in df.columns and 'LateralLength' not in df.columns:
        for alias in ['HorizontalLength', 'Horizontal Length', 'Lateral Length']:
            if alias in df.columns:
                df['LateralLength'] = df[alias]
                break

    # 4. Create NaN placeholder if completely missing
    if 'LateralLength' not in df.columns and 'DI Lateral Length' not in df.columns:
        df['LateralLength'] = np.nan

    # 5. Date Aliases
    if 'Completion Date' not in df.columns and 'CompletionDate' not in df.columns:
        for alias in ['SpudDate', 'Spud Date']:
            if alias in df.columns:
                df['Completion Date'] = df[alias]
                break
    if 'First Prod Date' not in df.columns and 'FirstProdDate' not in df.columns:
        for alias in ['FirstProductionDate']:
            if alias in df.columns:
                df['First Prod Date'] = df[alias]
                break
    return df

def _translate(df: pd.DataFrame, maps, required) -> pd.DataFrame:
    for m in maps:
        if all(c in df.columns for c in m.values()):
            new_data = {}
            for target_name, source_name in m.items():
                new_data[target_name] = df[source_name]

            subset = pd.DataFrame(new_data)

            missing = [c for c in required if c not in subset.columns]
            if 'LateralLength' in missing and 'LateralLength' in df.columns:
                 subset['LateralLength'] = df['LateralLength']

            final_miss = [c for c in required if c not in subset.columns]
            if not final_miss:
                return subset[required].copy()

    raise ValueError(f"Could not map columns. Found in file: {list(df.columns)}")

def load_header(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    df = _normalize_columns_before_map(df)
    return _translate(df, HEADER_COLUMN_MAPS, REQUIRED_HEADER_COLUMNS)

def load_production(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    df = _normalize_columns_before_map(df)
    out = _translate(df, PROD_COLUMN_MAPS, REQUIRED_PROD_COLUMNS)
    out['ReportDate'] = pd.to_datetime(out['ReportDate'])
    return out

# ---------------- Header QC ----------------
def fill_lateral_by_geo(
    df: pd.DataFrame,
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    lateral_col: str = 'LateralLength',
    decimals: int = 2
) -> pd.DataFrame:
    df = df.copy()
    if lat_col not in df.columns or lon_col not in df.columns:
        df['LateralImputed'] = False
        df['LateralImputeNote'] = 'No lat/lon columns found'
        return df

    df[lateral_col] = pd.to_numeric(df[lateral_col], errors='coerce')
    miss = df[lateral_col].isna() | (df[lateral_col] == 0)

    df['lat_bin'] = df[lat_col].round(decimals)
    df['lon_bin'] = df[lon_col].round(decimals)
    valid = df[~miss & df[lateral_col].notna()]

    if valid.empty:
        df['LateralImputed'] = False
        df['LateralImputeNote'] = 'No valid laterals to train from'
        return df.drop(columns=['lat_bin','lon_bin'], errors='ignore')

    bin_means = (valid.groupby(['lat_bin','lon_bin'])[lateral_col]
                      .mean().rename('bin_mean'))
    df = df.merge(bin_means, left_on=['lat_bin','lon_bin'], right_index=True, how='left')
    df['LateralImputed'] = False
    fill_mask = miss & df['bin_mean'].notna()
    df.loc[fill_mask, lateral_col] = df.loc[fill_mask, 'bin_mean']
    df.loc[fill_mask, 'LateralImputed'] = True
    df['LateralImputeNote'] = np.where(df['LateralImputed'], 'Filled by geo bin', 'As provided')
    return df.drop(columns=['lat_bin','lon_bin','bin_mean'], errors='ignore')

# ---------------- Preprocess ----------------
@dataclass
class PreprocessConfig:
    normalization_length: int = 10_000
    use_normalization: bool = True
    # Lateral-length scaling exponent a in (production ~ L^a).
    # 1.0 == legacy linear scaling (EUR proportional to length).
    # SPEE Monograph 5 ("Scaling") notes EUR/ft DECREASES with increasing
    # lateral length, i.e. production scales sub-linearly (a < 1).
    normalization_exponent: float = 1.0


def estimate_lateral_exponent(header_df: pd.DataFrame, prod_df: pd.DataFrame,
                              lo: float = 0.5, hi: float = 1.0) -> float | None:
    """Estimate the lateral-length scaling exponent a where production ~ L^a.

    SPEE Monograph 5 observes that EUR per foot decreases with increasing
    lateral length, so cumulative production scales sub-linearly with length
    (a < 1).  We approximate a by regressing log(cumulative BOE) on
    log(lateral length) across wells, using total historical cum as an EUR
    proxy (BOE = oil + gas/6).

    Returns the slope clipped to [lo, hi], or None when the data are too sparse
    or show no usable length variation (caller should fall back to 1.0/linear).
    """
    try:
        hd = header_df.copy()
        pr = prod_df.copy()
        hd['WellID'] = _make_well_id(hd)
        pr['WellID'] = _make_well_id(pr)

        for c in ['TotalOil', 'TotalGas', 'TotalWater']:
            pr[c] = pd.to_numeric(pr.get(c), errors='coerce').fillna(0.0)
        pr['_boe'] = pr['TotalOil'] + pr['TotalGas'] / 6.0
        cum = pr.groupby('WellID')['_boe'].sum().rename('cum')

        hd['LateralLength'] = pd.to_numeric(hd['LateralLength'], errors='coerce')
        lat = hd.groupby('WellID')['LateralLength'].first().rename('lat')

        df = pd.concat([cum, lat], axis=1).dropna()
        df = df[(df['cum'] > 0) & (df['lat'] > 0)]
        if len(df) < 8 or df['lat'].nunique() < 4:
            return None

        slope, _intercept, r, _p, _se = stats.linregress(
            np.log(df['lat'].values), np.log(df['cum'].values)
        )
        if not np.isfinite(slope):
            return None
        return float(np.clip(slope, lo, hi))
    except Exception:
        return None

def _make_well_id(df: pd.DataFrame) -> pd.Series:
    """Creates composite ID from WellName + WellNumber"""
    wn = df['WellName'].astype(str).str.strip().str.upper()
    # Remove .0 from float-like strings (e.g. "1.0" -> "1")
    num = df['WellNumber'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return wn + "_" + num

def preprocess(header_df: pd.DataFrame, prod_df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    hd = header_df.copy(); pr = prod_df.copy()

    # Create WellID for merging
    hd['WellID'] = _make_well_id(hd)
    pr['WellID'] = _make_well_id(pr)

    pr['MonthYear'] = pr['ReportDate'].dt.to_period('M')

    # Fill missing phase production with 0 — a well with no gas should report
    # 0 bbl, not be dropped. Only remove rows where every phase is NaN
    # (i.e. the row carries no production information at all).
    for c in ['TotalOil', 'TotalGas', 'TotalWater']:
        pr[c] = pd.to_numeric(pr[c], errors='coerce').fillna(0)
    pr = pr[~((pr['TotalOil'] == 0) & (pr['TotalGas'] == 0) & (pr['TotalWater'] == 0))]

    # Sort and Deduplicate
    pr = pr.sort_values(['WellID','MonthYear','TotalOil','TotalGas','TotalWater'],
                        ascending=[True,True,False,False,False])
    pr = pr.drop_duplicates(subset=['WellID','MonthYear'], keep='first')

    keep_hdr = ['WellID','API10','WellName','WellNumber','State','County','PrimaryFormation',
                'LateralLength','CompletionDate','FirstProdDate']
    keep_hdr = [c for c in keep_hdr if c in hd.columns]

    # MERGE ON WellID instead of API10
    merged = pr.merge(hd[keep_hdr], on='WellID', how='inner', suffixes=('', '_hdr'))

    # Clean up name/number collisions if any
    if 'WellName_hdr' in merged.columns:
        merged['WellName'] = merged['WellName_hdr']
        merged = merged.drop(columns=['WellName_hdr'])
    if 'WellNumber_hdr' in merged.columns:
        merged['WellNumber'] = merged['WellNumber_hdr']
        merged = merged.drop(columns=['WellNumber_hdr'])
    if 'API10_hdr' in merged.columns:
        merged['API10'] = merged['API10_hdr']
        merged = merged.drop(columns=['API10_hdr'])

    if cfg.use_normalization:
        merged['LateralLength'] = pd.to_numeric(merged['LateralLength'], errors='coerce').replace(0, np.nan)

        # Impute missing laterals with the dataset median so wells without a
        # lateral length are not silently dropped from the merged dataset.
        lat_median = merged['LateralLength'].median()
        fallback = lat_median if pd.notna(lat_median) else float(cfg.normalization_length)
        merged['LateralLength'] = merged['LateralLength'].fillna(fallback)

        # Sub-linear scaling per SPEE Monograph 5: production ~ L^a, so the
        # normalization factor is (target/L)^a.  a = 1.0 reproduces the legacy
        # linear behaviour; a < 1 reflects EUR/ft declining with length.
        a = float(getattr(cfg, 'normalization_exponent', 1.0) or 1.0)
        s = (cfg.normalization_length / merged['LateralLength']) ** a

        for c in ['TotalOil','TotalGas','TotalWater']:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0)

        merged['NormOil']   = merged['TotalOil']*s
        merged['NormGas']   = merged['TotalGas']*s
        merged['NormWater'] = merged['TotalWater']*s
    else:
        merged['NormOil']   = pd.to_numeric(merged['TotalOil'],   errors='coerce').fillna(0)
        merged['NormGas']   = pd.to_numeric(merged['TotalGas'],   errors='coerce').fillna(0)
        merged['NormWater'] = pd.to_numeric(merged['TotalWater'],  errors='coerce').fillna(0)

    merged = merged.replace([np.inf, -np.inf], 0)
    return merged

# ---------------- Models ----------------
# All model functions work in TC_Tool's month-index time: t = 0 is the peak
# month and q(t) is the instantaneous rate in volume / month.  Every model has
# a rate function, a closed-form cumulative N(t) with N(0) = 0, and honours the
# SAME terminal-decline floor d_lim: once the nominal decline D(t) = -d ln q/dt
# falls to d_lim the curve continues as an exponential at d_lim (Robertson 1988
# / SPE-78695), so no candidate can carry a multi-decade tail declining slower
# than the analyst's terminal rate.  Monthly volumes are exact interval
# integrals N(k+1) - N(k), never point rates.

DAYS_PER_MONTH: float = 365.25 / 12.0          # 30.4375


def nominal_monthly_from_effective_annual(d_eff: float, b: float = 0.0) -> float:
    """Effective annual decline (fraction, e.g. 0.05 = 5 %/yr) -> nominal per-month.

    b = 0 gives the tangent-effective form D = -ln(1 - d_eff); b > 0 the
    secant-effective (first-year) form D = ((1 - d_eff)^(-b) - 1) / b.  Reserves
    software quotes declines as effective annual percentages; the models here
    use nominal rates.
    """
    d_eff = float(min(max(d_eff, 0.0), 0.999999))
    if b <= 1e-10:
        return float(-np.log1p(-d_eff) / 12.0)
    return float(np.expm1(-b * np.log1p(-d_eff)) / b / 12.0)


def effective_annual_from_nominal_monthly(d_nom: float, b: float = 0.0) -> float:
    """Inverse of nominal_monthly_from_effective_annual (returns a fraction)."""
    D = float(d_nom) * 12.0
    if b <= 1e-10:
        return float(-np.expm1(-D))
    return float(-np.expm1(-np.log1p(b * D) / b))


# Terminal decline: 5 %/yr effective (tangent) = 0.004274 / month nominal.
# (The former constant 0.00417 was 5 %/yr *nominal*, i.e. 4.88 %/yr effective.)
D_LIM_DEFAULT: float = nominal_monthly_from_effective_annual(0.05)

# BIC gap below which a more complex model is not preferred over the
# 1-parameter Exponential model (Kass & Raftery 1995: ΔBIC < 2 is
# "not worth more than a bare mention").
EXP_BIC_TOLERANCE: float = 2.0

# Modified Arps is the industry reference model (the SPEE M5 baseline).  A
# same-complexity alternative (SEPD, Duong) must beat it by at least this BIC
# margin to be selected.  Kass & Raftery call ΔBIC 2-6 "positive evidence";
# 4 sits mid-band: on planted hyperbolic truth with 5 % noise, SEPD still
# edges Arps by ΔBIC 2-3 on ~12 % of wells and then under-forecasts ~30 %,
# while a margin of 6 starts suppressing genuine SEPD behaviour.
ARPS_BIC_PREFERENCE: float = 4.0

# Soft peak pin: the fitted month-0 volume may deviate from the observed
# peak-month volume by this factor range (the observed peak is a single,
# max-selected noisy sample; a hard pin biases every model's shape).
PEAK_SCALE_LO: float = 0.75
PEAK_SCALE_HI: float = 1.25


# ---- Arps segment primitives (shared by every hyperbolic-family model) ----
def _seg_rate(q0: float, b: float, D0: float, dt) -> np.ndarray:
    """Rate along one Arps segment dt months after its start (dt clamped >= 0).

    Evaluated in log space: for small b the exponent 1/b is huge and
    (1 + b D0 dt)^(1/b) overflows (as a Python scalar it raises OverflowError,
    which used to drop near-exponential wells); exp(log q0 - log1p(b D0 dt)/b)
    is stable and tends to the exponential limit as b -> 0.
    """
    dt = np.maximum(np.asarray(dt, dtype=float), 0.0)
    if b <= 1e-10:
        return q0 * np.exp(-D0 * dt)
    return np.exp(np.log(max(float(q0), 1e-300)) - np.log1p(b * D0 * dt) / b)


def _seg_cum(q0: float, b: float, D0: float, dt) -> np.ndarray:
    """Closed-form cumulative of one Arps segment over dt months (dt clamped >= 0)."""
    dt = np.maximum(np.asarray(dt, dtype=float), 0.0)
    if D0 <= 0:
        return q0 * dt
    if b <= 1e-10:
        return q0 / D0 * (-np.expm1(-D0 * dt))
    if abs(b - 1.0) < 1e-9:
        return q0 / D0 * np.log1p(D0 * dt)
    # expm1 form avoids cancellation in 1 - exp(...) as b -> 1
    return -q0 / ((1.0 - b) * D0) * np.expm1((1.0 - 1.0 / b) * np.log1p(b * D0 * dt))


def _trapz_cum(rate_fn, t: np.ndarray, n_grid: int = 20000) -> np.ndarray:
    """Numerical cumulative (fallback only) on a log-spaced grid (dense near t = 0,
    where decline rates change fastest) joined with the requested times."""
    t = np.asarray(t, dtype=float)
    tmax = max(float(np.nanmax(t)) if t.size else 0.0, 1.0)
    grid = np.unique(np.concatenate([[0.0], np.geomspace(1e-9, tmax, n_grid), t]))
    N = np.concatenate([[0.0], cumulative_trapezoid(rate_fn(grid), grid)])
    return np.interp(t, grid, N)


# ---- Exponential (Arps b = 0) ----
def exponential(qi: float, di: float, t) -> np.ndarray:
    """Exponential decline: q(t) = qi exp(-di t) — boundary-dominated flow in
    conventional / vertical wells (SPEE M5: BDF has b < 1, b -> 0 for
    volumetric depletion)."""
    return _seg_rate(qi, 0.0, di, t)


def exponential_cum(qi: float, di: float, t) -> np.ndarray:
    return _seg_cum(qi, 0.0, di, t)


def arps(qi: float, b: float, di: float, t) -> np.ndarray:
    """Classic Arps decline without a terminal floor (legacy helper)."""
    t = np.asarray(t, dtype=float)
    if b <= 0:
        return exponential(qi, di, t)
    if b == 1:
        return qi / (1.0 + di * t)
    return _seg_rate(qi, b, di, t)


# ---- Modified Arps (hyperbolic -> terminal exponential) ----
def _arps_switch(b: float, di: float, d_lim: float) -> float:
    """Month at which the hyperbolic decline D(t) = di/(1 + b di t) reaches d_lim
    (inf when there is no floor, 0 when the curve starts at or below it)."""
    if d_lim <= 0:
        return np.inf
    if b <= 0 or di <= d_lim:
        return 0.0
    return (di / d_lim - 1.0) / (b * di)


def modified_arps(qi: float, b: float, di: float, d_lim: float, t) -> np.ndarray:
    """Modified Arps rate (Robertson 1988 / SPE-78695): hyperbolic until the
    nominal decline falls to d_lim, exponential at d_lim afterwards.  b <= 0 is
    exponential decline (never slower than d_lim)."""
    t = np.asarray(t, dtype=float)
    if b <= 0:
        return exponential(qi, max(di, d_lim), t)
    t_sw = _arps_switch(b, di, d_lim)
    if not np.isfinite(t_sw):
        return _seg_rate(qi, b, di, t)
    if t_sw <= 0:
        return exponential(qi, d_lim, t)
    q_sw = float(_seg_rate(qi, b, di, t_sw))
    return np.where(t <= t_sw, _seg_rate(qi, b, di, t), _seg_rate(q_sw, 0.0, d_lim, t - t_sw))


def modified_arps_cum(qi: float, b: float, di: float, d_lim: float, t) -> np.ndarray:
    """Closed-form cumulative of modified_arps from t = 0."""
    t = np.asarray(t, dtype=float)
    if b <= 0:
        return _seg_cum(qi, 0.0, max(di, d_lim), t)
    t_sw = _arps_switch(b, di, d_lim)
    if not np.isfinite(t_sw):
        return _seg_cum(qi, b, di, t)
    if t_sw <= 0:
        return _seg_cum(qi, 0.0, d_lim, t)
    q_sw = float(_seg_rate(qi, b, di, t_sw))
    return _seg_cum(qi, b, di, np.minimum(t, t_sw)) + _seg_cum(q_sw, 0.0, d_lim, t - t_sw)


# ---- SEPD (stretched exponential, Valkó 2009 SPE-119369) ----
def _sepd_raw(qi: float, tau: float, n: float, t) -> np.ndarray:
    return qi * np.exp(-np.power(np.maximum(np.asarray(t, dtype=float), 0.0) / tau, n))


def _sepd_raw_cum(qi: float, tau: float, n: float, t) -> np.ndarray:
    """N(t) = qi tau/n Gamma(1/n) P(1/n, (t/tau)^n) — the Gamma(1/n) factor is
    essential (without it the EUR is wrong by up to +33 % at n = 0.4)."""
    tt = np.maximum(np.asarray(t, dtype=float), 0.0)
    coef = qi * tau / n * gamma(1.0 / n)
    if np.isfinite(coef):
        return coef * gammainc(1.0 / n, np.power(tt / tau, n))
    # Gamma(1/n) overflows only for n < ~0.006, outside the fitting bounds.
    return _trapz_cum(lambda x: _sepd_raw(qi, tau, n, x), tt)


def _sepd_switch(tau: float, n: float, d_lim: float) -> float:
    """Month at which SEPD's nominal decline n t^(n-1)/tau^n falls to d_lim
    (0 if it is already below from the start, inf if it never falls to it)."""
    if d_lim <= 0:
        return np.inf
    if n >= 1.0 - 1e-12:
        return 0.0 if (1.0 / tau) <= d_lim else np.inf
    lg = np.log(d_lim * tau ** n / n) / (n - 1.0)      # log space: 1/(n-1) is large near n = 1
    return np.inf if lg > 700.0 else float(np.exp(lg))


def sepd(qi: float, tau: float, n: float, d_lim: float, t) -> np.ndarray:
    """SEPD rate q = qi exp(-(t/tau)^n) with the terminal-decline floor applied."""
    t = np.asarray(t, dtype=float)
    q = _sepd_raw(qi, tau, n, t)
    t_sw = _sepd_switch(tau, n, d_lim)
    if not np.isfinite(t_sw):
        return q
    if t_sw <= 0:
        return exponential(qi, d_lim, t)
    q_sw = float(_sepd_raw(qi, tau, n, t_sw))
    return np.where(t <= t_sw, q, _seg_rate(q_sw, 0.0, d_lim, t - t_sw))


def sepd_cum(qi: float, tau: float, n: float, d_lim: float, t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t_sw = _sepd_switch(tau, n, d_lim)
    if not np.isfinite(t_sw):
        return _sepd_raw_cum(qi, tau, n, t)
    if t_sw <= 0:
        return _seg_cum(qi, 0.0, d_lim, t)
    q_sw = float(_sepd_raw(qi, tau, n, t_sw))
    return _sepd_raw_cum(qi, tau, n, np.minimum(t, t_sw)) + _seg_cum(q_sw, 0.0, d_lim, t - t_sw)


# ---- Duong (fracture-dominated flow, SPE-137748); 1-based time t' = t + 1 ----
def _duong_raw(qi: float, a: float, m: float, t) -> np.ndarray:
    tp = np.maximum(np.asarray(t, dtype=float), 0.0) + 1.0
    return qi * np.power(tp, -m) * np.exp(a / (1.0 - m) * (np.power(tp, 1.0 - m) - 1.0))


def _duong_raw_cum(qi: float, a: float, m: float, t) -> np.ndarray:
    """N(t) = qi/a [exp(a/(1-m)((t+1)^(1-m) - 1)) - 1], from Duong's q/N = a t'^(-m)."""
    tp = np.maximum(np.asarray(t, dtype=float), 0.0) + 1.0
    return qi / a * np.expm1(a / (1.0 - m) * (np.power(tp, 1.0 - m) - 1.0))


def _duong_D(a: float, m: float, tp) -> np.ndarray:
    return m / tp - a * np.power(tp, -m)


def _duong_switch(a: float, m: float, d_lim: float) -> float:
    """0-based month after which Duong's nominal decline m/t' - a t'^(-m) stays
    below d_lim.  D peaks at t' = a^(1/(m-1)) and decreases afterwards, so the
    last crossing is bracketed from there (0 if D never exceeds d_lim)."""
    if d_lim <= 0:
        return np.inf
    lp = np.log(a) / (m - 1.0) if a > 0 else -np.inf
    tp_peak = 1.0 if lp <= 0 else float(np.exp(min(lp, 60.0)))
    if _duong_D(a, m, tp_peak) <= d_lim:
        return 0.0
    hi = 2.0 * tp_peak
    while _duong_D(a, m, hi) > d_lim:
        hi *= 2.0
        if hi > 1e9:
            return np.inf
    tp_sw = brentq(lambda x: _duong_D(a, m, x) - d_lim, tp_peak, hi, xtol=1e-6)
    return float(tp_sw - 1.0)


def duong(qi: float, a: float, m: float, d_lim: float, t) -> np.ndarray:
    """Duong rate q = qi t'^(-m) exp(a/(1-m)(t'^(1-m) - 1)) with the terminal floor."""
    t = np.asarray(t, dtype=float)
    q = _duong_raw(qi, a, m, t)
    t_sw = _duong_switch(a, m, d_lim)
    if not np.isfinite(t_sw):
        return q
    if t_sw <= 0:
        return exponential(qi, d_lim, t)
    q_sw = float(_duong_raw(qi, a, m, t_sw))
    return np.where(t <= t_sw, q, _seg_rate(q_sw, 0.0, d_lim, t - t_sw))


def duong_cum(qi: float, a: float, m: float, d_lim: float, t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t_sw = _duong_switch(a, m, d_lim)
    if not np.isfinite(t_sw):
        return _duong_raw_cum(qi, a, m, t)
    if t_sw <= 0:
        return _seg_cum(qi, 0.0, d_lim, t)
    q_sw = float(_duong_raw(qi, a, m, t_sw))
    return _duong_raw_cum(qi, a, m, np.minimum(t, t_sw)) + _seg_cum(q_sw, 0.0, d_lim, t - t_sw)


# Model registry: name -> (rate_fn, cum_fn, number of free parameters with qi pinned)
MODEL_REGISTRY = {
    'Modified Arps': (modified_arps, modified_arps_cum, 2),
    'SEPD':          (sepd,          sepd_cum,          2),
    'Duong':         (duong,         duong_cum,         2),
    'Exponential':   (exponential,   exponential_cum,   1),
}


def model_volumes(cum_fn, params: tuple, t, v_peak: float) -> Tuple[np.ndarray, float]:
    """Monthly volumes V(k) = N(k+1) - N(k) of a model whose cumulative is
    cum_fn(qi, *params, t), with the rate scale qi chosen so that the month-0
    volume equals the pinned peak-month volume v_peak.  Returns (V, qi).  qi is
    the instantaneous rate at the peak month in volume / month."""
    t = np.asarray(t, dtype=float)
    unit_first = float(cum_fn(1.0, *params, np.array([1.0]))[0])
    if not np.isfinite(unit_first) or unit_first <= 0:
        return np.full(t.shape, np.nan), np.nan
    qi = v_peak / unit_first
    N = cum_fn(qi, *params, np.concatenate([t, t + 1.0]))
    k = len(t)
    return N[k:] - N[:k], qi


def _log_mse(y_pred: np.ndarray, y_obs: np.ndarray) -> float:
    """Log-space MSE — appropriate for multiplicative DCA errors."""
    return float(np.mean(
        (np.log(np.clip(y_pred, 1e-6, None)) - np.log(np.clip(y_obs, 1e-6, None))) ** 2
    ))


def _bic(y_obs: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Bayesian Information Criterion in log space.  Lower = better fit."""
    n = len(y_obs)
    rss = np.sum(
        (np.log(np.clip(y_obs, 1e-6, None)) - np.log(np.clip(y_pred, 1e-6, None))) ** 2
    )
    sigma2 = max(rss / n, 1e-30)
    return n * np.log(sigma2) + k * np.log(max(n, 2))


def _fit_model(loss_fn, x0_list: list, bounds: list, args: tuple = ()) -> Tuple[np.ndarray, float]:
    """Multi-start L-BFGS-B: tries each starting point, returns (best_params, best_loss)."""
    best_x, best_f = None, np.inf
    for x0 in x0_list:
        try:
            r = minimize(loss_fn, x0=x0, args=args, bounds=bounds, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
            if np.isfinite(r.fun) and r.fun < best_f:
                best_f, best_x = r.fun, r.x.copy()
        except Exception:
            pass
    return best_x, best_f


def _smooth(x, w=3):
    s = pd.Series(x, dtype=float)
    return s.rolling(window=w, center=True, min_periods=1).mean().values


def _fit_quality(y_obs: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Goodness-of-fit of model monthly volumes to observed volumes on the fit
    window ("history match" metrics): log-space RMSE and R², and MAPE."""
    m = np.isfinite(y_obs) & np.isfinite(y_pred) & (y_obs > 0) & (y_pred > 0)
    if m.sum() < 2:
        return dict(fit_log_rmse=np.nan, fit_r2_log=np.nan, fit_mape_pct=np.nan, n_fit=int(m.sum()))
    lo, lp = np.log(y_obs[m]), np.log(y_pred[m])
    sse = float(np.sum((lp - lo) ** 2))
    sst = float(np.sum((lo - lo.mean()) ** 2))
    return dict(
        fit_log_rmse=float(np.sqrt(sse / m.sum())),
        fit_r2_log=float(1.0 - sse / sst) if sst > 0 else np.nan,
        fit_mape_pct=float(np.mean(np.abs(y_pred[m] - y_obs[m]) / y_obs[m]) * 100.0),
        n_fit=int(m.sum()),
    )


def forecast_one_well(wd: pd.DataFrame, commodity: str, b_low: float, b_high: float,
                      max_months: int,
                      d_lim: float = D_LIM_DEFAULT,
                      allow_exponential: bool = False,
                      exclude_duong: bool = False,
                      duong_min_months: int = 18,
                      qi_cap: float | None = None) -> Dict[str, object]:
    """Fit the candidate decline models to one well's monthly volumes and forecast.

    Workflow (SPEE Monograph 5 aligned):
      * The peak month is located on a 3-month smoothed series and then snapped
        to the raw maximum within +/-1 month (centered smoothing alone peaks a
        month late on steep declines); t = 0 is that month and every model is
        pinned so its month-0 VOLUME equals the observed peak-month volume.
      * Models are fitted to RAW post-peak monthly volumes (shut-in runs of >= 3
        months below 2 % of peak excluded) by multi-start L-BFGS-B on log-space
        MSE, comparing exact model monthly volumes N(k+1) - N(k) with the data.
      * Candidates: Modified Arps, SEPD, Duong (only with >= duong_min_months
        post-peak points — its power-law tail is uncontrolled on short windows),
        and Exponential when allow_exponential.  All share the terminal-decline
        floor d_lim.  Selection by BIC with a Kass-Raftery tie-break towards the
        1-parameter Exponential.
      * Forecast monthly volumes for max_months; EUR = history + analytic
        N(end) - N(now) of the winner.
    """
    commodity = commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[commodity]
    if not (0.0 < d_lim < 1.0):
        raise ValueError(f"d_lim must be in (0, 1) per month, got {d_lim}")
    if b_low > b_high:
        raise ValueError(f"b_low ({b_low}) exceeds b_high ({b_high})")

    wd = wd.sort_values('MonthYear').copy()
    t_hist = np.arange(len(wd), dtype=float)
    y_hist = wd[col].values.astype(float)
    y_hist = np.where(np.isfinite(y_hist), y_hist, 0.0)
    if len(y_hist) == 0 or not (y_hist > 0).any():
        raise ValueError("no positive monthly volumes to fit")
    y_s    = _smooth(y_hist, 3)
    n = len(y_hist)

    # ---- Peak: smoothed candidate, snapped to the raw maximum within +/-1 month ----
    cand = int(np.nanargmax(y_s)) if n else 0
    lo, hi = max(cand - 1, 0), min(cand + 1, n - 1)
    peak_idx = lo + int(np.nanargmax(y_hist[lo:hi + 1]))
    v_peak_obs = float(max(y_hist[peak_idx], 1.0))
    if qi_cap is not None and qi_cap > 0:
        v_peak_obs = min(v_peak_obs, float(qi_cap))

    # ---- Post-peak decline data (raw monthly volumes; t = 0 at the peak month) ----
    y_decline  = y_hist[peak_idx:]
    ys_decline = y_s[peak_idx:]
    t_decline  = np.arange(len(y_decline), dtype=float)

    # Shut-in filtering: contiguous runs of >= 3 months at < 2 % of peak are excluded.
    shutin_thresh = max(0.02 * v_peak_obs, 1.0)
    low = y_decline <= shutin_thresh
    in_shutin = np.zeros(len(y_decline), dtype=bool)
    run = 0
    for i, is_low in enumerate(low):
        run = (run + 1) if is_low else 0
        if run >= 3:
            in_shutin[max(0, i - run + 1):i + 1] = True
    active = (~in_shutin) & (y_decline > 0)
    if active.sum() < 4:
        active = y_decline > 0
    t_fit = t_decline[active]
    y_fit = y_decline[active]
    if len(y_fit) == 0:                       # degenerate: single positive peak only
        t_fit, y_fit = np.array([0.0]), np.array([v_peak_obs])
    n_fit_pts = len(y_fit)

    # ---- Seeds from the smoothed series ----
    # b0 from the slope of 1/D vs t (Arps: d(1/D)/dt = b)
    if len(ys_decline) >= 5:
        log_q = np.log(np.clip(ys_decline, 1e-9, None))
        d_nom = np.clip(-np.gradient(log_q), 1e-9, None)
        inv_d = _smooth(1.0 / d_nom, 3)
        b_slope, *_ = stats.linregress(np.arange(len(inv_d), dtype=float), inv_d)
        b0 = float(np.clip(b_slope, b_low, b_high))
    else:
        b0 = 0.5 * (b_low + b_high)
    # di0 from the LATEST decline trend (anchor to current behaviour, SPE-78695)
    if len(ys_decline) >= 4:
        tail_n = max(6, len(ys_decline) // 3)
        tail = ys_decline[-tail_n:] if len(ys_decline) > tail_n else ys_decline
        pos = tail > 0
        if pos.sum() >= 3:
            slope, *_ = stats.linregress(np.arange(len(tail), dtype=float)[pos], np.log(tail[pos]))
            di0 = float(np.clip(-slope, d_lim, 1.0))
        else:
            di0 = 0.05
    else:
        di0 = 0.05
    di0 = max(di0, d_lim)
    tau0 = max(1.0, 1.0 / max(di0, 1e-6))

    # ---- Candidate fits: model monthly volumes vs observed volumes ----
    # The month-0 volume is pinned SOFTLY: a common scale s in [PEAK_SCALE_LO,
    # PEAK_SCALE_HI] multiplies the observed peak-month volume, and the peak
    # month is an ordinary fit point.  A hard pin to the single, max-selected
    # peak sample biases the anchor up by its noise and forces every model to
    # absorb that error in its shape (Arps pushed to a bound, SEPD winning on a
    # thin tail); s is the same for every model so the BIC comparison stays fair.
    def _vol(cum_fn, params, t, s=1.0):
        return model_volumes(cum_fn, params, t, v_peak_obs * s)[0]

    s_bounds = (PEAK_SCALE_LO, PEAK_SCALE_HI)
    fits: Dict[str, tuple] = {}          # name -> (s, params...)

    def _arps_obj(p):
        return _log_mse(_vol(modified_arps_cum, (p[1], p[2], d_lim), t_fit, p[0]), y_fit)
    arps_starts = [
        [1.0, b0, di0], [1.0, b_low, di0], [1.0, b_high, min(di0 * 2.0, 0.5)],
        [1.0, b0, max(di0 * 0.5, d_lim)], [1.0, 0.5 * (b_low + b_high), max(di0 * 1.5, d_lim)],
    ]
    p, _ = _fit_model(_arps_obj, arps_starts, [s_bounds, (b_low, b_high), (d_lim, 1.0)])
    if p is None:
        p = np.array([1.0, np.clip(b0, b_low, b_high), max(di0, d_lim)])
    s_a, b_a, di_a = float(p[0]), float(p[1]), float(p[2])
    fits['Modified Arps'] = (s_a, b_a, di_a, d_lim)

    def _sepd_obj(p):
        return _log_mse(_vol(sepd_cum, (p[1], p[2], d_lim), t_fit, p[0]), y_fit)
    sepd_starts = [[1.0, tau0, 0.5], [1.0, tau0 * 2, 0.3], [1.0, tau0 * 0.5, 0.7],
                   [1.0, tau0, 0.4], [1.0, tau0, 0.9]]
    p, _ = _fit_model(_sepd_obj, sepd_starts, [s_bounds, (0.5, 500.0), (0.05, 1.0)])
    if p is None:
        p = np.array([1.0, tau0, 0.5])
    fits['SEPD'] = (float(p[0]), float(p[1]), float(p[2]), d_lim)

    if (not exclude_duong) and n_fit_pts >= duong_min_months:
        def _duong_obj(p):
            if p[1] > p[2]:        # a > m: the rate would RISE from the peak
                return 1e6 + 1e3 * (p[1] - p[2])
            return _log_mse(_vol(duong_cum, (p[1], p[2], d_lim), t_fit, p[0]), y_fit)
        duong_starts = [[1.0, 1.0, 1.5], [1.0, 0.5, 1.2], [1.0, 2.0, 2.2], [1.0, 1.0, 2.0], [1.0, 0.8, 1.1]]
        p, _ = _fit_model(_duong_obj, duong_starts, [s_bounds, (0.001, 10.0), (1.01, 3.0)])
        if p is not None and p[1] <= p[2]:
            fits['Duong'] = (float(p[0]), float(p[1]), float(p[2]), d_lim)

    if allow_exponential:
        def _exp_obj(p):
            return _log_mse(_vol(exponential_cum, (p[1],), t_fit, p[0]), y_fit)
        exp_starts = [[1.0, di0], [1.0, min(di0 * 2.0, 1.0)], [1.0, max(di0 * 0.5, d_lim)]]
        p, _ = _fit_model(_exp_obj, exp_starts, [s_bounds, (d_lim, 1.0)])
        fits['Exponential'] = (float(p[0]), float(p[1])) if p is not None else (1.0, float(di0))

    # ---- Model selection by BIC (k = model params + the shared peak scale) ----
    preds = {name: _vol(MODEL_REGISTRY[name][1], f[1:], t_fit, f[0]) for name, f in fits.items()}
    bics = {name: _bic(y_fit, preds[name], MODEL_REGISTRY[name][2] + 1) for name in fits}
    bic_min = min(bics.values())
    best_model = min(bics, key=bics.get)
    # Reference-model preference: SEPD / Duong need positive evidence (ΔBIC >= 2)
    # over Modified Arps; a smaller gap is indistinguishable from noise.
    if (best_model in ('SEPD', 'Duong') and 'Modified Arps' in bics
            and bics['Modified Arps'] - bics[best_model] < ARPS_BIC_PREFERENCE):
        best_model = 'Modified Arps'
    # Parsimony tie-break (against the true BIC minimum): a 2-parameter model
    # that beats the 1-parameter exponential by ΔBIC < 2 has no real support.
    if 'Exponential' in bics and bics['Exponential'] - bic_min < EXP_BIC_TOLERANCE:
        best_model = 'Exponential'
    rate_fn, cum_fn, _k = MODEL_REGISTRY[best_model]
    s_best, params = fits[best_model][0], fits[best_model][1:]
    v_peak = v_peak_obs * s_best

    # ---- Fit + forecast on the peak-relative axis (monthly VOLUMES) ----
    n_post = len(t_hist) - peak_idx
    t_post = np.arange(n_post, dtype=float)
    t_fcst = np.arange(n_post, n_post + max_months, dtype=float)
    post_fit, qi_rate = model_volumes(cum_fn, params, t_post, v_peak)
    f_vals_all, _     = model_volumes(cum_fn, params, t_fcst, v_peak)
    eur_fcst = float(cum_fn(qi_rate, *params, np.array([float(n_post + max_months)]))[0]
                     - cum_fn(qi_rate, *params, np.array([float(n_post)]))[0])

    fit_hist = np.concatenate([y_s[:peak_idx], post_fit])   # smoothed ramp-up + model decline
    valid = np.isfinite(f_vals_all) & (f_vals_all >= 1e-6)
    f_m   = t_fcst[valid].astype(int)
    f_v   = f_vals_all[valid]
    eur_hist = float(np.nansum(y_hist))

    quality = _fit_quality(y_fit, preds[best_model])

    # ---- Arps-equivalent (b, di) consistent with the winner, over the HISTORY
    # window + 24 months (so it describes the observed decline, not the 50-year
    # tail; it is NOT an EUR-matched equivalent for SEPD/Duong winners).
    if best_model == 'Modified Arps':
        b_out, di_out = b_a, di_a
    elif best_model == 'Exponential':
        b_out, di_out = 0.0, float(params[0])
    else:
        t_eq = np.arange(0, n_post + 24, dtype=float)
        v_eq, _ = model_volumes(cum_fn, params, t_eq, v_peak)
        meq = np.isfinite(v_eq) & (v_eq > 1e-9)
        b_out, di_out = b_a, di_a
        if meq.sum() >= 4:
            def _eq_obj(p):
                return _log_mse(model_volumes(modified_arps_cum, (p[0], p[1], d_lim), t_eq[meq], v_peak)[0],
                                v_eq[meq])
            pe, _ = _fit_model(_eq_obj, [st[1:] for st in arps_starts], [(b_low, b_high), (d_lim, 1.0)])
            if pe is not None:
                b_out, di_out = float(pe[0]), float(pe[1])
    b_at_bound = bool(best_model in ('Modified Arps', 'SEPD', 'Duong')
                      and min(abs(b_out - b_low), abs(b_out - b_high)) < 1e-3)

    # ---- First-year (secant effective) decline from the winner's own rate ----
    q0  = float(rate_fn(qi_rate, *params, np.array([0.0]))[0])
    q12 = float(rate_fn(qi_rate, *params, np.array([12.0]))[0])
    decline_1yr_pct = (1.0 - q12 / q0) * 100.0 if q0 > 0 else np.nan

    native = {
        'Modified Arps': lambda p: {'b': p[0], 'di': p[1]},
        'SEPD':          lambda p: {'tau': p[0], 'n': p[1]},
        'Duong':         lambda p: {'a': p[0], 'm': p[1]},
        'Exponential':   lambda p: {'di': p[0]},
    }[best_model](params)

    return dict(
        qi=qi_rate, v_peak=v_peak, v_peak_obs=v_peak_obs, peak_scale=s_best,
        b=b_out, di=di_out, d_lim=d_lim,
        b_arps=b_a, di_arps=di_a, b_at_bound=b_at_bound,
        model=best_model, params=native, bics=bics, fits=fits,
        n_months=int(len(t_hist)), n_post=int(n_post), peak_idx=int(peak_idx),
        t_hist=t_hist, hist=y_hist,
        fit_hist=fit_hist, f_months=f_m, f_vals=f_v,
        EUR_total=eur_hist + eur_fcst, EUR_fcst=eur_fcst,
        decline_1yr_pct=decline_1yr_pct, **quality,
    )


def hindcast_one_well(wd: pd.DataFrame, commodity: str, b_low: float, b_high: float,
                      train_frac: float = 0.6, d_lim: float = D_LIM_DEFAULT,
                      min_train: int = 6, min_test: int = 3,
                      holdout_months: int | None = None,
                      allow_exponential: bool = False,
                      exclude_duong: bool = False,
                      duong_min_months: int = 18,
                      qi_cap: float | None = None,
                      min_post_train: int = 4) -> Dict[str, object] | None:
    """Back-test one well's forecast (SPEE Monograph 5 — Validation / history match).

    Fit the full workflow on the first part of the history (either the first
    ``train_frac`` of the months, or everything but the last ``holdout_months``),
    forecast across the held-out remainder, and compare with the actual
    volumes.  Returns observed/predicted arrays plus error metrics, or None when
    the history is too short to split meaningfully or the training window has
    fewer than ``min_post_train`` usable post-peak points (a "fit" to a single
    pinned point would just return the seed parameters).

    Metrics (both computed over held-out months with actual production > 0, so
    a shut-in during the hold-out is not counted as forecast error)
    -------
    cum_err_pct : signed error of forecast cumulative vs actual cumulative
                  (positive = forecast over-predicts).
    mape_pct    : mean absolute percentage error across those months.
    """
    com = commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[com]
    wd = wd.sort_values('MonthYear').copy()
    n = len(wd)
    if holdout_months is not None and holdout_months > 0:
        n_train = n - int(holdout_months)
    else:
        n_train = int(round(n * train_frac))
    if n < (min_train + min_test) or n_train < min_train or (n - n_train) < min_test:
        return None

    train = wd.iloc[:n_train]
    test = wd.iloc[n_train:]
    y_test = test[col].values.astype(float)
    n_test = len(y_test)

    fc = forecast_one_well(train, com, b_low, b_high, max_months=n_test + 12, d_lim=d_lim,
                           allow_exponential=allow_exponential, exclude_duong=exclude_duong,
                           duong_min_months=duong_min_months, qi_cap=qi_cap)
    if int(fc.get('n_fit', 0)) < min_post_train:
        return None                            # no post-peak decline to learn from
    # fc forecast begins at the month after the end of training history.
    y_pred = np.asarray(fc['f_vals'], dtype=float)[:n_test]
    if len(y_pred) < n_test:
        y_pred = np.concatenate([y_pred, np.full(n_test - len(y_pred), np.nan)])

    valid = np.isfinite(y_pred) & np.isfinite(y_test) & (y_test > 0)
    if valid.sum() == 0:
        return None
    mape = float(np.mean(np.abs(y_pred[valid] - y_test[valid]) / y_test[valid]) * 100.0)
    cum_actual = float(np.sum(y_test[valid]))
    cum_pred = float(np.sum(y_pred[valid]))
    cum_err = float((cum_pred - cum_actual) / cum_actual * 100.0) if cum_actual > 0 else np.nan

    return dict(
        n_train=n_train, n_test=n_test,
        n_post_train=int(fc.get('n_post', 0)), n_fit_train=int(fc.get('n_fit', 0)),
        t_train=np.arange(1, n_train + 1, dtype=float),
        y_train=train[col].values.astype(float),
        t_test=np.arange(n_train + 1, n_train + 1 + n_test, dtype=float),
        y_test=y_test, y_pred=y_pred,
        cum_actual=cum_actual, cum_pred=cum_pred,
        mape_pct=mape, cum_err_pct=cum_err, model=fc.get('model'),
        fit_r2_log=fc.get('fit_r2_log'), fit_mape_pct=fc.get('fit_mape_pct'),
    )


@dataclass
class ForecastConfig:
    commodity: str          # 'oil' | 'gas' | 'water'
    b_low: float = 0.8
    b_high: float = 1.2
    max_months: int = 600
    d_lim: float = D_LIM_DEFAULT      # terminal nominal decline (per month), all models
    # SPEE Monograph 5 stresses "sufficient history" for individual forecasts.
    min_months_history: int = 6       # wells with fewer months are skipped
    # Vertical / conventional datasets: offer an explicit Exponential (b = 0)
    # candidate, and drop Duong — a fracture-dominated transient-flow model
    # that only applies to multi-fractured horizontal wells.
    allow_exponential: bool = False
    exclude_duong: bool = False
    # Duong's power-law tail is uncontrolled on short windows: require this many
    # post-peak points before it may compete.
    duong_min_months: int = 18
    # Optional cap on the pinned peak-month volume (None = no cap).
    qi_cap: float | None = None
    # Forecast gas / water as a yield (GOR / WOR) off the oil forecast.
    yield_from_oil: bool = False


def _forecast_well_job(well_id, wd: pd.DataFrame, cfg: ForecastConfig):
    """Module-level worker for joblib parallel execution."""
    try:
        fc = forecast_one_well(wd, cfg.commodity, cfg.b_low, cfg.b_high, cfg.max_months,
                               d_lim=cfg.d_lim, allow_exponential=cfg.allow_exponential,
                               exclude_duong=cfg.exclude_duong,
                               duong_min_months=cfg.duong_min_months, qi_cap=cfg.qi_cap)
        return well_id, wd, fc
    except Exception:
        return None


def _hindcast_well_job(well_id, wd: pd.DataFrame, cfg: ForecastConfig,
                       holdout_months, train_frac):
    try:
        hc = hindcast_one_well(wd, cfg.commodity, cfg.b_low, cfg.b_high, train_frac=train_frac,
                               d_lim=cfg.d_lim, holdout_months=holdout_months,
                               min_train=max(6, int(cfg.min_months_history)),
                               allow_exponential=cfg.allow_exponential,
                               exclude_duong=cfg.exclude_duong,
                               duong_min_months=cfg.duong_min_months, qi_cap=cfg.qi_cap)
        return well_id, wd, hc
    except Exception:
        return None


def _forecast_well_as_yield(wd: pd.DataFrame, com: str, pfc: dict, cfg: ForecastConfig) -> dict:
    """Forecast gas or water as a two-slope power-law yield (GOR / WOR, the dca
    PLYield form) applied to the well's OIL forecast, returning an fc dict with
    the same keys as forecast_one_well so plots, onelines and type curves work
    unchanged.  Raises when the yield cannot be fitted (caller falls back to an
    independent decline fit)."""
    from yields import fit_yield, forecast_secondary, observed_ratio, yield_curve   # optional module
    col = {'gas': 'NormGas', 'water': 'NormWater'}[com]
    wd = wd.sort_values('MonthYear').copy()
    y_sec = wd[col].values.astype(float)
    prim_hist = np.asarray(pfc['hist'], dtype=float)
    n = min(len(y_sec), len(prim_hist))
    if n < 6:
        raise ValueError("history too short for a yield fit")
    y_sec, prim_hist = y_sec[:n], prim_hist[:n]
    t_mid = np.arange(n, dtype=float) + 0.5                  # months since first production
    params = fit_yield(t_mid, observed_ratio(prim_hist, y_sec))
    if params is None:
        raise ValueError("yield fit failed")

    peak_idx = int(pfc['peak_idx'])
    n_post = n - peak_idx
    prim_clean = np.where(np.isfinite(prim_hist), np.maximum(prim_hist, 0.0), 0.0)
    fit_hist = yield_curve(params, t_mid) * prim_clean       # implied secondary history match
    f_months = np.asarray(pfc['f_months'], dtype=int)
    f_vals = forecast_secondary(params, f_months + peak_idx, np.asarray(pfc['f_vals'], dtype=float))
    valid = np.isfinite(f_vals) & (f_vals >= 1e-6)
    f_months, f_vals = f_months[valid], f_vals[valid]
    eur_fcst, eur_hist = float(np.sum(f_vals)), float(np.nansum(y_sec))

    # Arps-equivalent (b, di) of the implied secondary curve over the history
    # match + 24 months, so the oneline and the type-curve builder have a shape.
    v_peak = float(max(fit_hist[peak_idx], 1.0)) if n_post > 0 else 1.0
    trace = np.concatenate([fit_hist[peak_idx:], f_vals[:24]])
    t_eq = np.arange(len(trace), dtype=float)
    meq = np.isfinite(trace) & (trace > 1e-9)
    b_out, di_out, qi = 1.0, max(cfg.d_lim, 0.05), v_peak
    if meq.sum() >= 4:
        def _obj(p):
            V, _ = model_volumes(modified_arps_cum, (p[0], p[1], cfg.d_lim), t_eq[meq], v_peak)
            return _log_mse(V, trace[meq])
        pe, _ = _fit_model(_obj, [[1.0, 0.1], [0.5, 0.05], [1.5, 0.2], [0.0, 0.02]],
                           [(0.0, 2.0), (cfg.d_lim, 1.0)])
        if pe is not None:
            b_out, di_out = float(pe[0]), float(pe[1])
            _, qi = model_volumes(modified_arps_cum, (b_out, di_out, cfg.d_lim), np.array([0.0]), v_peak)
    q0  = float(modified_arps(qi, b_out, di_out, cfg.d_lim, np.array([0.0]))[0])
    q12 = float(modified_arps(qi, b_out, di_out, cfg.d_lim, np.array([12.0]))[0])
    quality = _fit_quality(y_sec[peak_idx:], fit_hist[peak_idx:])
    return dict(
        qi=qi, v_peak=v_peak, b=b_out, di=di_out, d_lim=cfg.d_lim, b_arps=b_out, di_arps=di_out,
        b_at_bound=False, model='Yield (GOR)' if com == 'gas' else 'Yield (WOR)',
        params={'c': params.c, 'm0': params.m0, 'm': params.m, 't0': params.t0,
                'ymax': params.y_max if params.y_max is not None else np.nan, 'r2': params.r2},
        bics={}, fits={}, n_months=int(n), n_post=int(n_post), peak_idx=peak_idx,
        t_hist=np.arange(n, dtype=float), hist=y_sec, fit_hist=fit_hist,
        f_months=f_months, f_vals=f_vals, EUR_total=eur_hist + eur_fcst, EUR_fcst=eur_fcst,
        decline_1yr_pct=(1.0 - q12 / q0) * 100.0 if q0 > 0 else np.nan,
        yield_params=params, **quality,
    )


def _params_label(params: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4g}" for k, v in params.items())


def _eur_columns(com: str, fc: dict) -> Dict[str, float]:
    unit = {'oil': 'Mbbl', 'gas': 'MMcf', 'water': 'Mbbl water'}[com]
    return {f'EUR ({unit})':       round(fc['EUR_total'] / 1_000.0, 2),
            f'Remaining ({unit})': round(fc['EUR_fcst']  / 1_000.0, 2)}


def _oneline_row(well_id, wd: pd.DataFrame, fc: dict, com: str) -> Dict[str, object]:
    hdr = wd.iloc[0]
    row = {
        'WellID':             str(well_id),
        'API10':              str(hdr.get('API10', 'N/A')),
        'WellName':           hdr.get('WellName', 'N/A'),
        'WellNumber':         str(hdr.get('WellNumber', '')),
        'State':              hdr.get('State'),
        'County':             hdr.get('County'),
        'PrimaryFormation':   hdr.get('PrimaryFormation'),
        'LateralLength':      hdr.get('LateralLength'),
        'CompletionDate':     hdr.get('CompletionDate'),
        'FirstProdDate':      hdr.get('FirstProdDate'),
        'Model':              fc.get('model', 'Modified Arps'),
        'Model Params':       _params_label(fc.get('params', {})),
        'Months History':     int(fc.get('n_months', len(wd))),
        'qi (per day)':       round(float(fc['qi']) / DAYS_PER_MONTH, 0),
        'b':                  round(float(fc['b']), 3),
        'di (per month)':     round(float(fc['di']), 4),
        'b at bound':         bool(fc.get('b_at_bound', False)),
        'First-Year Decline (%)':      round(float(fc.get('decline_1yr_pct', np.nan)), 1),
        'Terminal Decline (%/yr eff)': round(effective_annual_from_nominal_monthly(fc['d_lim']) * 100.0, 2),
        'Fit MAPE (%)':       round(float(fc.get('fit_mape_pct', np.nan)), 1),
        'Fit R² (log)':       round(float(fc.get('fit_r2_log', np.nan)), 3),
    }
    row.update(_eur_columns(com, fc))
    return row


def _monthly_rows(well_id, wd: pd.DataFrame, fc: dict, com: str, col: str, merged: pd.DataFrame) -> list:
    hdr = wd.iloc[0]
    well, api10 = hdr.get('WellName', 'N/A'), str(hdr.get('API10', 'N/A'))
    wd_sorted = wd.sort_values('MonthYear')
    hist_dates = wd_sorted['MonthYear'].dt.to_timestamp(how='start')
    hist_vals  = wd_sorted[col].values.astype(float)
    start = (hist_dates.iloc[-1] + pd.offsets.MonthBegin(1) if len(hist_dates) > 0
             else merged['MonthYear'].min().to_timestamp(how='start'))
    f_dates = pd.date_range(start=start, periods=len(fc['f_vals']), freq='MS')
    rows = [{'WellID': str(well_id), 'API10': api10, 'WellName': well, 'Date': d,
             f'Monthly_{com}_volume': float(v), 'Segment': 'Historical'}
            for d, v in zip(hist_dates, hist_vals)]
    rows += [{'WellID': str(well_id), 'API10': api10, 'WellName': well, 'Date': d,
              f'Monthly_{com}_volume': float(v), 'Segment': 'Forecast'}
             for d, v in zip(f_dates, fc['f_vals'])]
    return rows


def forecast_all(merged: pd.DataFrame, cfg: ForecastConfig,
                 primary_fcs: Dict[str, dict] | None = None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Forecast all wells in parallel.

    Parameters
    ----------
    primary_fcs : optional dict WellID -> oil forecast (as returned in
        well_forecasts).  When ``cfg.yield_from_oil`` is set and the commodity
        is gas or water, each well is forecast as a yield (GOR / WOR) applied to
        its oil forecast instead of an independent decline fit.

    Returns
    -------
    oneline_df    : one row per well with DCA parameters, fit quality and EUR
    monthly_df    : monthly historical + forecast volumes per well
    well_forecasts: dict mapping WellID -> raw fc dict (for per-well plots)
    """
    com = cfg.commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[com]

    well_data = [
        (well_id, wd.copy())
        for well_id, wd in merged.groupby('WellID')
        if wd[col].max() > 0 and wd[col].sum() > 0 and len(wd) >= cfg.min_months_history
    ]

    use_yield = bool(cfg.yield_from_oil and com in ('gas', 'water') and primary_fcs)
    if use_yield:
        raw_results = []
        for well_id, wd in well_data:
            pfc = primary_fcs.get(str(well_id))
            if pfc is None:
                r = _forecast_well_job(well_id, wd, cfg)    # no oil forecast: independent fit
            else:
                try:
                    r = (well_id, wd, _forecast_well_as_yield(wd, com, pfc, cfg))
                except Exception:
                    r = _forecast_well_job(well_id, wd, cfg)
            raw_results.append(r)
    else:
        # Threading backend is safe with Streamlit and benefits from scipy's GIL
        # release during L-BFGS-B optimisation.
        raw_results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_forecast_well_job)(well_id, wd, cfg) for well_id, wd in well_data
        )

    oneline: list = []
    monthly: list = []
    well_forecasts: Dict[str, dict] = {}
    for result in raw_results:
        if result is None:
            continue
        well_id, wd, fc = result
        well_forecasts[str(well_id)] = fc
        oneline.append(_oneline_row(well_id, wd, fc, com))
        monthly += _monthly_rows(well_id, wd, fc, com, col, merged)

    oneline_df = pd.DataFrame(oneline)
    monthly_df = (pd.DataFrame(monthly).sort_values(['WellID', 'Date', 'Segment'])
                  if monthly else pd.DataFrame(columns=['WellID', 'API10', 'WellName', 'Date',
                                                        f'Monthly_{com}_volume', 'Segment']))
    return oneline_df, monthly_df, well_forecasts


def hindcast_all(merged: pd.DataFrame, cfg: ForecastConfig,
                 holdout_months: int | None = None, train_frac: float = 0.6) -> pd.DataFrame:
    """Population-wide hindcast (SPEE Monograph 5 — Validation): every well that
    has enough history is re-forecast from a truncated history and compared with
    what it actually produced over the held-out months.  Returns one row per
    well with the error metrics; empty DataFrame if no well qualifies."""
    com = cfg.commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[com]
    well_data = [
        (well_id, wd.copy())
        for well_id, wd in merged.groupby('WellID')
        if wd[col].max() > 0 and wd[col].sum() > 0 and len(wd) >= cfg.min_months_history
    ]
    raw = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_hindcast_well_job)(well_id, wd, cfg, holdout_months, train_frac)
        for well_id, wd in well_data
    )
    rows = []
    for r in raw:
        if r is None or r[2] is None:
            continue
        well_id, wd, hc = r
        hdr = wd.iloc[0]
        rows.append({
            'WellID': str(well_id), 'WellName': hdr.get('WellName', 'N/A'),
            'API10': str(hdr.get('API10', '')),
            'Model': hc['model'], 'Train months': hc['n_train'], 'Test months': hc['n_test'],
            'Post-peak months (train)': hc['n_post_train'], 'Fit points (train)': hc['n_fit_train'],
            'Actual (test)': round(hc['cum_actual'], 1), 'Forecast (test)': round(hc['cum_pred'], 1),
            'Cum error (%)': round(hc['cum_err_pct'], 1), 'MAPE (%)': round(hc['mape_pct'], 1),
            'Fit R² (log)': round(float(hc['fit_r2_log']), 3) if hc.get('fit_r2_log') is not None else np.nan,
        })
    cols = ['WellID', 'WellName', 'API10', 'Model', 'Train months', 'Test months',
            'Post-peak months (train)', 'Fit points (train)',
            'Actual (test)', 'Forecast (test)', 'Cum error (%)', 'MAPE (%)', 'Fit R² (log)']
    return pd.DataFrame(rows, columns=cols)


def compute_eur_stats(eur_array: List[float]) -> Dict[str, float]:
    eur = np.array(eur_array, dtype=float)
    eur = eur[np.isfinite(eur)]
    if eur.size == 0:
        return {"P10": np.nan, "P50": np.nan, "P90": np.nan, "Mean": np.nan,
                "Count": 0, "P10/P90": np.nan, "TypeCurve": np.nan}
    eur_sorted = np.sort(eur)[::-1]
    p10  = np.percentile(eur_sorted, 90)
    p50  = np.percentile(eur_sorted, 50)
    p90  = np.percentile(eur_sorted, 10)
    mean = float(np.mean(eur_sorted))
    ratio = float(p10 / p90) if (p90 and np.isfinite(p90)) else np.nan
    return {"P10": float(p10), "P50": float(p50), "P90": float(p90),
            "Mean": mean, "Count": int(eur_sorted.size),
            "P10/P90": ratio, "TypeCurve": float(p50)}

def _mantissa_formatter(val, pos=None):
    s = f"{val:g}"
    return s if len(s) <= 4 else ""

def _probit_ax(ax, values: np.ndarray, unit_label: str, color):
    """Draw a single probit panel onto *ax*."""
    vals = np.sort(values)[::-1]
    ranks = np.arange(1, vals.size + 1)
    probs = (ranks - 0.5) / vals.size
    z = stats.norm.ppf(1 - probs)
    p10 = np.percentile(vals, 90)
    p90 = np.percentile(vals, 10)
    ax.scatter(vals, z, edgecolors='black', alpha=0.85, s=48, color=color)
    ax.plot([p90, p10],
            [stats.norm.ppf(1 - 0.90), stats.norm.ppf(1 - 0.10)],
            linewidth=2, color=color if color else 'black')
    ticks = [stats.norm.ppf(1 - x / 100) for x in (10, 50, 90)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(['P10', 'P50', 'P90'])
    ax.set_ylabel("Probit")
    ax.set_xlabel(f"EUR ({unit_label})")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(left=0)


def probit_plot(eurs: List[float], unit_label: str, title: str,
                color: str | None = None, norm_len: int | None = None):
    eur = np.array([x for x in eurs if np.isfinite(x) and x > 0.0], dtype=float)
    if eur.size == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.axis('off')
        return fig

    factor = 1000.0  # Mbbl→Bbl/ft, MMcf→Mcf/ft
    _per_ft_map = {"Mbbl": "Bbl/ft", "MMcf": "Mcf/ft"}
    per_ft_label = _per_ft_map.get(unit_label, f"{unit_label}/ft")

    if norm_len and norm_len > 0:
        eur_ft = eur / norm_len * factor
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        _probit_ax(ax_left,  eur,    unit_label,    color)
        _probit_ax(ax_right, eur_ft, per_ft_label,  color)
        ax_left.set_title(f"{title} — EUR")
        ax_right.set_title(f"{title} — EUR/ft")
    else:
        fig, ax_left = plt.subplots(figsize=(9, 7))
        _probit_ax(ax_left, eur, unit_label, color)
        ax_left.set_title(title)

    plt.tight_layout()
    return fig

def eur_summary_table(fluid_name: str, stats_dict: Dict[str, float], unit: str, norm_len: int) -> pd.DataFrame:
    def per_ft(x: float) -> float:
        return (x / norm_len) if (pd.notna(x) and np.isfinite(x) and norm_len) else np.nan
    rows = [
        ["P90",                  stats_dict.get("P90"),       per_ft(stats_dict.get("P90"))],
        ["P50",                  stats_dict.get("P50"),       per_ft(stats_dict.get("P50"))],
        ["P10",                  stats_dict.get("P10"),       per_ft(stats_dict.get("P10"))],
        ["P10/P90 Ratio",        stats_dict.get("P10/P90"),   np.nan],
        ["Well Count",           stats_dict.get("Count"),     np.nan],
        ["Normalization Length", norm_len,                    "ft"],
        ["Mean",                 stats_dict.get("Mean"),      per_ft(stats_dict.get("Mean"))],
        ["Type Curve EUR",       stats_dict.get("TypeCurve"), per_ft(stats_dict.get("TypeCurve"))],
    ]
    return pd.DataFrame(rows, columns=[f"{fluid_name} EURs", unit, f"{unit}/ft"])

def plot_one_well(wd: pd.DataFrame, fc: dict, commodity: str):
    com = commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[com]
    color = {'oil':'green','gas':'red','water':'blue'}[com]
    eps = 1e-6

    wd = wd.sort_values('MonthYear').copy()
    n_hist = len(wd)
    hist_t  = np.arange(1, n_hist + 1, dtype=float)
    hist_vals = np.clip(wd[col].astype(float).values, eps, None)

    fit_hist = np.clip(fc['fit_hist'], eps, None)
    f_t   = np.arange(n_hist + 1, n_hist + 1 + len(fc['f_vals']), dtype=float)
    f_vals = np.clip(fc['f_vals'], eps, None)

    unit = {"oil":"(norm bbl/mo)","gas":"(norm Mcf/mo)","water":"(norm bbl/mo)"}[com]
    well = wd.iloc[0].get('WellName','N/A')
    api  = str(wd.iloc[0].get('API10',''))
    if not api or api == 'nan':
        api = wd.iloc[0].get('WellID','N/A')

    fig, ax = plt.subplots(figsize=(10,6))
    if n_hist > 0:
        ax.scatter(hist_t, hist_vals, label="Historical", s=22, color=color)
        ax.plot(hist_t, fit_hist, label="Fit (history)", linewidth=2, color=color)
    if len(f_t) > 0:
        ax.plot(f_t, f_vals, label="Forecast", linestyle="--", linewidth=2, color=color)

    model_label = fc.get('model', 'Modified Arps')
    ax.set_title(f"{well} | ID: {api} | {commodity.capitalize()} [{model_label}]")
    ax.set_xlabel("Month of Production")
    ax.set_ylabel(f"Monthly {commodity} {unit}")
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)
    ax.grid(True, linestyle="--", alpha=0.4, which='both')
    ax.legend()
    fig.tight_layout()
    return fig
