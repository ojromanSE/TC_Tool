from __future__ import annotations
import matplotlib
# Fix for headless/cloud environments
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize
from scipy.special import huber
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy import stats

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

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

    pr = pr.dropna(subset=['TotalOil','TotalGas','TotalWater'])

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
        s = cfg.normalization_length / merged['LateralLength']

        for c in ['TotalOil','TotalGas','TotalWater']:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0)

        merged['NormOil']   = merged['TotalOil']*s
        merged['NormGas']   = merged['TotalGas']*s
        merged['NormWater'] = merged['TotalWater']*s
    else:
        merged['NormOil']   = merged['TotalOil']
        merged['NormGas']   = merged['TotalGas']
        merged['NormWater'] = merged['TotalWater']

    merged = merged.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    merged = merged.dropna(subset=['NormOil','NormGas','NormWater'])
    return merged

# ====================================================================
# DCA Models — Modified Arps, SEPD (Stretched Exponential), Duong
# ====================================================================

# ---------- 1. Arps (legacy, kept for backward compat) ----------
def arps(qi: float, b: float, di: float, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if b == 0:
        return qi * np.exp(-di * t)
    if b == 1:
        return qi / (1.0 + di * t)
    return qi / ((1.0 + b * di * t) ** (1.0 / b))

# ---------- 2. Modified Arps with terminal decline (Dmin) ----------
def modified_arps(qi: float, b: float, di: float, dmin: float,
                  t: np.ndarray) -> np.ndarray:
    """Hyperbolic decline that switches to exponential at Dmin.

    Parameters
    ----------
    qi   : initial rate (monthly volume)
    b    : Arps b-factor
    di   : initial nominal decline rate (per month)
    dmin : terminal (minimum) nominal decline rate (per month).
           Typical values: 0.004-0.008 monthly (≈5-10% annual effective).
    t    : time array in months
    """
    t = np.asarray(t, dtype=float)
    result = np.empty_like(t, dtype=float)

    if b <= 0 or di <= dmin:
        # Pure exponential at terminal rate
        return qi * np.exp(-dmin * t)

    # Time at which instantaneous decline D(t) = di/(1+b*di*t) reaches dmin
    t_switch = (1.0 / (b * di)) * ((di / dmin) - 1.0)
    q_switch = arps(qi, b, di, np.array([t_switch]))[0]

    hyp_mask = t <= t_switch
    result[hyp_mask] = arps(qi, b, di, t[hyp_mask])
    result[~hyp_mask] = q_switch * np.exp(-dmin * (t[~hyp_mask] - t_switch))
    return result

def _modified_arps_cum(qi: float, b: float, di: float, dmin: float,
                       t: np.ndarray) -> np.ndarray:
    """Cumulative production for Modified Arps (numerical via trapezoidal)."""
    rates = modified_arps(qi, b, di, dmin, t)
    return np.cumsum(rates)

# ---------- 3. Stretched Exponential Production Decline (SEPD) ----------
def sepd(qi: float, tau: float, n: float, t: np.ndarray) -> np.ndarray:
    """q(t) = qi * exp(-(t/tau)^n)

    Parameters
    ----------
    qi  : initial rate
    tau : characteristic time constant (months)
    n   : stretching exponent (0 < n <= 1 typical for unconventionals)
    """
    t = np.asarray(t, dtype=float)
    t_safe = np.maximum(t, 1e-12)
    return qi * np.exp(-((t_safe / tau) ** n))

# ---------- 4. Duong Model ----------
def duong(qi: float, a: float, m: float, t: np.ndarray) -> np.ndarray:
    """q(t) = qi * t^(-m) * exp(a/(1-m) * (t^(1-m) - 1))

    Designed for fracture-dominated flow in shale/unconventional wells.

    Parameters
    ----------
    qi : rate scalar
    a  : intercept parameter
    m  : slope parameter (typically 1.0 - 1.5)
    """
    t = np.asarray(t, dtype=float)
    t_safe = np.maximum(t, 0.1)  # avoid log(0) / 0^(-m)
    exponent = (a / (1.0 - m)) * (t_safe ** (1.0 - m) - 1.0)
    return qi * (t_safe ** (-m)) * np.exp(exponent)

# ====================================================================
# Model fitting helpers
# ====================================================================

def _smooth(x, w=3):
    s = pd.Series(x, dtype=float)
    return s.rolling(window=w, center=True, min_periods=1).mean().values

def robust_loss(params: np.ndarray, t: np.ndarray, y: np.ndarray, delta: float=1.0) -> float:
    qi, b, di = params
    return huber(delta, arps(qi, b, di, t) - y).mean()

def _sse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sum((pred - obs) ** 2))

def _aicc(n: int, k: int, sse: float) -> float:
    """Corrected Akaike Information Criterion.
    Lower is better.  Returns +inf if model is degenerate."""
    if n <= k + 1 or sse <= 0:
        return np.inf
    ll = -0.5 * n * np.log(sse / n)
    aic = 2.0 * k - 2.0 * ll
    correction = (2.0 * k * (k + 1.0)) / max(n - k - 1.0, 1.0)
    return aic + correction

# --------------- Per-model fitters ---------------

def _fit_modified_arps(t: np.ndarray, y: np.ndarray, y_s: np.ndarray,
                       b_low: float, b_high: float,
                       dmin: float) -> Dict:
    """Fit Modified Arps and return result dict."""
    peak_qi = float(np.nanmax(y_s)) if len(y_s) > 0 else 1.0
    peak_qi = min(peak_qi, 152_000.0)

    # Multi-start: try a few initial guesses
    best_sse, best_params = np.inf, (peak_qi, 1.0, 0.05)
    for qi0_frac in [1.0, 0.8, 1.2]:
        for b0 in [b_low, (b_low + b_high) / 2, b_high]:
            for di0 in [0.02, 0.05, 0.10, 0.20]:
                qi0 = min(peak_qi * qi0_frac, max(peak_qi * 0.5, 0.01))
                bounds = [(0.01, peak_qi * 1.5), (b_low, b_high), (1e-6, 1.0)]
                try:
                    def _loss(p):
                        pred = modified_arps(p[0], p[1], p[2], dmin, t)
                        return huber(1.0, pred - y_s).mean()
                    res = minimize(_loss, x0=[qi0, b0, di0],
                                   bounds=bounds, method='L-BFGS-B',
                                   options={'maxiter': 300})
                    if res.fun < best_sse:
                        best_sse = res.fun
                        best_params = tuple(map(float, res.x))
                except Exception:
                    pass

    qi, b, di = best_params
    pred = modified_arps(qi, b, di, dmin, t)
    sse = _sse(pred, y_s)
    aicc = _aicc(len(t), 3, sse)  # 3 free params (qi, b, di); dmin fixed
    return dict(model='ModifiedArps', qi=qi, b=b, di=di, dmin=dmin,
                pred=pred, sse=sse, aicc=aicc)


def _fit_sepd(t: np.ndarray, y: np.ndarray, y_s: np.ndarray) -> Dict:
    """Fit SEPD and return result dict."""
    peak_qi = float(np.nanmax(y_s)) if len(y_s) > 0 else 1.0
    peak_qi = min(peak_qi, 152_000.0)

    best_sse, best_params = np.inf, (peak_qi, 20.0, 0.5)
    for qi0_frac in [1.0, 0.8]:
        for tau0 in [10.0, 30.0, 80.0]:
            for n0 in [0.3, 0.5, 0.8]:
                qi0 = peak_qi * qi0_frac
                bounds = [(0.01, peak_qi * 1.5), (1.0, 500.0), (0.05, 1.0)]
                try:
                    def _loss(p):
                        pred = sepd(p[0], p[1], p[2], t)
                        return huber(1.0, pred - y_s).mean()
                    res = minimize(_loss, x0=[qi0, tau0, n0],
                                   bounds=bounds, method='L-BFGS-B',
                                   options={'maxiter': 300})
                    if res.fun < best_sse:
                        best_sse = res.fun
                        best_params = tuple(map(float, res.x))
                except Exception:
                    pass

    qi, tau, n = best_params
    pred = sepd(qi, tau, n, t)
    sse = _sse(pred, y_s)
    aicc = _aicc(len(t), 3, sse)
    return dict(model='SEPD', qi=qi, tau=tau, n=n,
                pred=pred, sse=sse, aicc=aicc)


def _fit_duong(t: np.ndarray, y: np.ndarray, y_s: np.ndarray) -> Dict:
    """Fit Duong model and return result dict."""
    peak_qi = float(np.nanmax(y_s)) if len(y_s) > 0 else 1.0
    peak_qi = min(peak_qi, 152_000.0)

    # Duong uses t starting at 1 (month 1, 2, ...)
    t_duong = t + 1.0  # shift so first month = 1

    best_sse, best_params = np.inf, (peak_qi, 1.0, 1.1)
    for qi0_frac in [1.0, 0.8]:
        for a0 in [0.5, 1.0, 2.0]:
            for m0 in [1.0, 1.1, 1.3]:
                qi0 = peak_qi * qi0_frac
                bounds = [(0.01, peak_qi * 2.0), (0.01, 5.0), (0.5, 2.5)]
                try:
                    def _loss(p):
                        pred = duong(p[0], p[1], p[2], t_duong)
                        return huber(1.0, pred - y_s).mean()
                    res = minimize(_loss, x0=[qi0, a0, m0],
                                   bounds=bounds, method='L-BFGS-B',
                                   options={'maxiter': 300})
                    if res.fun < best_sse:
                        best_sse = res.fun
                        best_params = tuple(map(float, res.x))
                except Exception:
                    pass

    qi, a, m = best_params
    pred = duong(qi, a, m, t_duong)
    sse = _sse(pred, y_s)
    aicc = _aicc(len(t), 3, sse)
    return dict(model='Duong', qi=qi, a=a, m=m,
                pred=pred, sse=sse, aicc=aicc)


def _select_best_model(candidates: List[Dict]) -> Dict:
    """Pick the candidate with the lowest AICc."""
    valid = [c for c in candidates if np.isfinite(c['aicc'])]
    if not valid:
        return candidates[0]
    return min(valid, key=lambda c: c['aicc'])


# ====================================================================
# XGBoost / RF training — learns REAL per-well DCA parameters
# ====================================================================

def _fit_arps_quick(t: np.ndarray, y_s: np.ndarray,
                    b_low: float, b_high: float) -> Tuple[float, float, float]:
    """Quick Arps fit for a single well (used to build training labels)."""
    peak_qi = float(np.nanmax(y_s)) if len(y_s) > 0 else 1.0
    peak_qi = min(peak_qi, 152_000.0)
    bounds = [(0.01, peak_qi * 1.5), (b_low, b_high), (1e-6, 1.0)]
    best_sse, best_params = np.inf, (peak_qi, 1.0, 0.05)
    for b0 in [(b_low + b_high) / 2]:
        for di0 in [0.05, 0.10]:
            try:
                res = minimize(robust_loss, x0=[peak_qi * 0.9, b0, di0],
                               args=(t, y_s), bounds=bounds, method='L-BFGS-B',
                               options={'maxiter': 200})
                if res.fun < best_sse:
                    best_sse = res.fun
                    best_params = tuple(map(float, res.x))
            except Exception:
                pass
    return best_params


def _train_rf(df: pd.DataFrame, target_col: str) -> Dict:
    """Train ML models on real per-well fitted Arps parameters.

    Returns a dict with keys 'qi', 'b', 'di' mapping to trained regressors.
    Backward-compatible signature (still called _train_rf).
    """
    col = target_col
    well_params = []
    for well_id, wd in df.groupby('WellID'):
        wd = wd.sort_values('MonthYear')
        y = wd[col].values.astype(float)
        if len(y) < 3 or np.nanmax(y) <= 0:
            continue
        t = np.arange(len(y), dtype=float)
        y_s = _smooth(y, 3)
        qi, b, di = _fit_arps_quick(t, y_s, 0.0, 2.5)
        well_params.append({
            'WellID': well_id,
            'NormOil': wd['NormOil'].mean(),
            'NormGas': wd['NormGas'].mean(),
            'NormWater': wd['NormWater'].mean(),
            'LateralLength': pd.to_numeric(wd['LateralLength'].iloc[0], errors='coerce'),
            'n_months': len(y),
            'qi_fit': qi,
            'b_fit': b,
            'di_fit': di,
        })

    if len(well_params) < 3:
        # Not enough data — return dummy models
        dummy = RandomForestRegressor(n_estimators=1, random_state=42)
        dummy.fit([[0, 0, 0, 0, 0]], [0])
        return {'qi': dummy, 'b': dummy, 'di': dummy}

    wp = pd.DataFrame(well_params)
    feat_cols = ['NormOil', 'NormGas', 'NormWater', 'LateralLength', 'n_months']
    wp['LateralLength'] = wp['LateralLength'].fillna(wp['LateralLength'].median())
    X = wp[feat_cols].values

    models = {}
    for target, label in [('qi_fit', 'qi'), ('b_fit', 'b'), ('di_fit', 'di')]:
        y_target = wp[target].values
        if _HAS_XGBOOST:
            mdl = XGBRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
        else:
            mdl = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        mdl.fit(X, y_target)
        models[label] = mdl

    return models


# ====================================================================
# Forecasting — single well
# ====================================================================

def forecast_one_well(wd: pd.DataFrame, commodity: str, b_low: float, b_high: float,
                      max_months: int, models: Dict,
                      dmin: float = 0.006) -> Dict[str, object]:
    """Forecast a single well using multi-model DCA with AICc selection.

    The function fits Modified Arps, SEPD, and Duong, then picks the best
    model via corrected Akaike Information Criterion (AICc).

    Parameters
    ----------
    wd         : DataFrame for one well (sorted by MonthYear)
    commodity  : 'oil' | 'gas' | 'water'
    b_low      : lower bound for Arps b-factor
    b_high     : upper bound for Arps b-factor
    max_months : forecast horizon in months
    models     : dict of trained ML models (from _train_rf)
    dmin       : terminal decline rate per month for Modified Arps
                 (default 0.006 ≈ 7% annual effective decline)
    """
    commodity = commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[commodity]

    wd = wd.sort_values('MonthYear').copy()
    t_hist = np.arange(len(wd), dtype=float)
    y_hist = wd[col].values.astype(float)
    y_s = _smooth(y_hist, 3)

    # --- Fit all three models ---
    candidates = []
    ma_result = _fit_modified_arps(t_hist, y_hist, y_s, b_low, b_high, dmin)
    candidates.append(ma_result)

    if len(t_hist) >= 4:
        sepd_result = _fit_sepd(t_hist, y_hist, y_s)
        candidates.append(sepd_result)

    if len(t_hist) >= 6:
        duong_result = _fit_duong(t_hist, y_hist, y_s)
        candidates.append(duong_result)

    best = _select_best_model(candidates)

    # --- Generate forecast using the winning model ---
    fit_hist = best['pred']

    f_m, f_v = [], []
    m_idx = len(t_hist)
    model_name = best['model']

    while m_idx < len(t_hist) + max_months:
        t_arr = np.array([m_idx], dtype=float)
        if model_name == 'ModifiedArps':
            q = float(modified_arps(best['qi'], best['b'], best['di'], best['dmin'], t_arr)[0])
        elif model_name == 'SEPD':
            q = float(sepd(best['qi'], best['tau'], best['n'], t_arr)[0])
        elif model_name == 'Duong':
            q = float(duong(best['qi'], best['a'], best['m'], t_arr + 1.0)[0])
        else:
            q = float(arps(best['qi'], best.get('b', 1.0), best.get('di', 0.05), t_arr)[0])

        if q < 1e-6:
            break
        f_m.append(m_idx)
        f_v.append(q)
        m_idx += 1

    eur_hist = float(y_hist.sum())
    eur_fcst = float(np.sum(f_v))

    # Compute Arps-equivalent params for the oneline table
    qi_out = best.get('qi', 0.0)
    b_out = best.get('b', np.nan)
    di_out = best.get('di', np.nan)

    return dict(
        qi=qi_out, b=b_out, di=di_out,
        model=model_name,
        model_params={k: v for k, v in best.items()
                      if k not in ('pred', 'sse', 'aicc', 'model')},
        aicc=best['aicc'],
        t_hist=t_hist, hist=y_hist,
        fit_hist=fit_hist,
        f_months=np.array(f_m, int), f_vals=np.array(f_v, float),
        EUR_total=eur_hist + eur_fcst, EUR_fcst=eur_fcst,
    )


# ====================================================================
# Forecast config & batch runner
# ====================================================================

@dataclass
class ForecastConfig:
    commodity: str      # 'oil'|'gas'|'water'
    b_low: float = 0.0
    b_high: float = 2.0
    max_months: int = 600
    dmin: float = 0.006  # terminal decline ~7% annual effective

def forecast_all(merged: pd.DataFrame, cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    com = cfg.commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[com]
    models = _train_rf(merged, col)

    oneline, monthly = [], []

    for well_id, wd in merged.groupby('WellID'):
        if wd[col].max() <= 0 or wd[col].sum() <= 0:
            continue
        fc = forecast_one_well(wd, com, cfg.b_low, cfg.b_high, cfg.max_months,
                               models, dmin=cfg.dmin)
        hdr = wd.iloc[0]
        well = hdr.get('WellName', 'N/A')
        api10 = hdr.get('API10', 'N/A')

        qi_day = fc['qi'] / 30.4
        # First-year decline via the winning model
        t12 = np.array([12.0], dtype=float)
        model_name = fc['model']
        mp = fc['model_params']
        if model_name == 'ModifiedArps':
            q12 = float(modified_arps(mp['qi'], mp['b'], mp['di'], mp['dmin'], t12)[0])
        elif model_name == 'SEPD':
            q12 = float(sepd(mp['qi'], mp['tau'], mp['n'], t12)[0])
        elif model_name == 'Duong':
            q12 = float(duong(mp['qi'], mp['a'], mp['m'], t12 + 1.0)[0])
        else:
            q12 = float(arps(fc['qi'], fc['b'], fc['di'], t12)[0])

        decline_yr = (1.0 - (q12 / fc['qi'])) * 100.0 if fc['qi'] > 0 else 0.0

        row = {
            'WellID': str(well_id),
            'API10': str(api10),
            'WellName': well,
            'WellNumber': str(hdr.get('WellNumber', '')),
            'State': hdr.get('State'),
            'County': hdr.get('County'),
            'PrimaryFormation': hdr.get('PrimaryFormation'),
            'LateralLength': hdr.get('LateralLength'),
            'CompletionDate': hdr.get('CompletionDate'),
            'FirstProdDate': hdr.get('FirstProdDate'),
            'qi (per day)': round(qi_day, 0),
            'b': round(fc['b'], 3) if np.isfinite(fc['b']) else '',
            'di (per month)': round(fc['di'], 4) if np.isfinite(fc['di']) else '',
            'First-Year Decline (%)': round(decline_yr, 1),
            'Best Model': fc['model'],
        }
        if com == 'oil':
            row.update({'EUR (Mbbl)': round(fc['EUR_total'] / 1_000.0, 2),
                        'Remaining (Mbbl)': round(fc['EUR_fcst'] / 1_000.0, 2)})
        elif com == 'gas':
            row.update({'EUR (MMcf)': round(fc['EUR_total'] / 1_000_000.0, 2),
                        'Remaining (MMcf)': round(fc['EUR_fcst'] / 1_000_000.0, 2)})
        else:
            row.update({'EUR (Mbbl water)': round(fc['EUR_total'] / 1_000.0, 2),
                        'Remaining (Mbbl water)': round(fc['EUR_fcst'] / 1_000.0, 2)})
        oneline.append(row)

        hist_dates = wd.sort_values('MonthYear')['MonthYear'].dt.to_timestamp(how='start')
        hist_vals = wd[col].values.astype(float)
        start = hist_dates.iloc[-1] + pd.offsets.MonthBegin(1) if len(hist_dates) > 0 \
            else merged['MonthYear'].min().to_timestamp(how='start')
        f_dates = pd.date_range(start=start, periods=len(fc['f_vals']), freq='MS')

        for d, v in zip(hist_dates, hist_vals):
            monthly.append({'WellID': str(well_id), 'API10': str(api10),
                            'WellName': well, 'Date': d,
                            f'Monthly_{com}_volume': float(v), 'Segment': 'Historical'})
        for d, v in zip(f_dates, fc['f_vals']):
            monthly.append({'WellID': str(well_id), 'API10': str(api10),
                            'WellName': well, 'Date': d,
                            f'Monthly_{com}_volume': float(v), 'Segment': 'Forecast'})

    oneline_df = pd.DataFrame(oneline)
    monthly_df = pd.DataFrame(monthly).sort_values(['WellID', 'Date', 'Segment'])
    return oneline_df, monthly_df

# ====================================================================
# Statistics & visualization (unchanged public API)
# ====================================================================

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

def probit_plot(eurs: List[float], unit_label: str, title: str, color: str | None = None):
    eur = np.array([x for x in eurs if np.isfinite(x) and x>0.0], dtype=float)
    if eur.size == 0:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.axis('off')
        return fig
    eur_sorted = np.sort(eur)[::-1]
    ranks = np.arange(1, eur_sorted.size+1)
    probs = (ranks - 0.5) / eur_sorted.size
    z = stats.norm.ppf(1 - probs)

    p10  = np.percentile(eur_sorted, 90)
    p50  = np.percentile(eur_sorted, 50)
    p90  = np.percentile(eur_sorted, 10)

    fig, ax = plt.subplots(figsize=(9,7))
    ax.scatter(eur_sorted, z, edgecolors='black', alpha=0.85, s=48, color=color)
    ax.plot([p90, p10],
            [stats.norm.ppf(1-0.90), stats.norm.ppf(1-0.10)],
            linewidth=2, color=color if color else 'black')
    ticks = [stats.norm.ppf(1-x/100) for x in (10,50,90)]
    ax.set_yticks(ticks); ax.set_yticklabels(['P10','P50','P90'])
    ax.set_ylabel("Probit"); ax.set_xlabel(f"EUR ({unit_label})")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10), numticks=50))
    ax.xaxis.set_minor_formatter(FuncFormatter(_mantissa_formatter))
    ax.set_title(title); plt.tight_layout()
    return fig

def eur_summary_table(fluid_name: str, stats_dict: Dict[str, float], unit: str, norm_len: int) -> pd.DataFrame:
    factor = 1000.0 if unit != "MMcf" else 1.0
    def per_ft(x: float) -> float:
        return (x / norm_len * factor) if (pd.notna(x) and np.isfinite(x) and norm_len) else np.nan
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
    hist_dates = wd['MonthYear'].dt.to_timestamp(how='start')
    hist_vals  = np.clip(wd[col].astype(float).values, eps, None)

    fit_hist = np.clip(fc['fit_hist'], eps, None)
    fit_dates = hist_dates

    if len(hist_dates) > 0:
        start = hist_dates.iloc[-1] + pd.offsets.MonthBegin(1)
    else:
        start = pd.Timestamp.today().normalize().replace(day=1)
    f_dates = pd.date_range(start=start, periods=len(fc['f_vals']), freq='MS')
    f_vals  = np.clip(fc['f_vals'], eps, None)

    unit = {"oil":"(norm bbl/mo)","gas":"(norm Mcf/mo)","water":"(norm bbl/mo)"}[com]
    well = wd.iloc[0].get('WellName','N/A')
    api  = str(wd.iloc[0].get('API10',''))
    if not api or api == 'nan':
        api = wd.iloc[0].get('WellID','N/A')

    model_label = fc.get('model', 'Arps')

    fig, ax = plt.subplots(figsize=(10,6))
    if len(hist_dates) > 0:
        ax.scatter(hist_dates, hist_vals, label="Historical", s=22, color=color)
        ax.plot(fit_dates, fit_hist, label=f"Fit ({model_label})", linewidth=2, color=color)
    if len(f_dates) > 0:
        ax.plot(f_dates, f_vals, label=f"Forecast ({model_label})", linestyle="--", linewidth=2, color=color)

    ax.set_title(f"{well} | ID: {api} | {commodity.capitalize()} | Model: {model_label}")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Monthly {commodity} {unit}")
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)
    ax.grid(True, linestyle="--", alpha=0.4, which='both')
    ax.legend()
    fig.tight_layout()
    return fig
