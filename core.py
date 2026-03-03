from __future__ import annotations
import matplotlib
# Fix for headless/cloud environments
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from scipy.special import huber
from sklearn.ensemble import RandomForestRegressor
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

        # Impute missing laterals with the dataset median so wells without a
        # lateral length are not silently dropped from the merged dataset.
        lat_median = merged['LateralLength'].median()
        fallback = lat_median if pd.notna(lat_median) else float(cfg.normalization_length)
        merged['LateralLength'] = merged['LateralLength'].fillna(fallback)

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

# ---------------- Models ----------------

# Terminal decline rate: ~5% effective annual / 12 months
# Industry standard for Modified Arps (Robertson 1988 / SPE-78695)
D_LIM_DEFAULT: float = 0.00417

def arps(qi: float, b: float, di: float, t: np.ndarray) -> np.ndarray:
    """Classic hyperbolic Arps decline (legacy, no terminal floor)."""
    if b == 0 or b == 1:
        return qi / (1.0 + di * t)
    return qi / ((1.0 + b * di * t) ** (1.0 / b))


def modified_arps(qi: float, b: float, di: float, d_lim: float, t: np.ndarray) -> np.ndarray:
    """
    Modified Arps with terminal exponential decline (Robertson 1988 / SPE-78695).

    When the instantaneous nominal decline rate falls to d_lim the model
    switches from hyperbolic to exponential, preventing the long tail that
    causes EUR over-estimation in unconventional wells.

    Switch time:  t_sw = (di/d_lim - 1) / (b * di)
    Switch rate:  q_sw = qi / (1 + b*di*t_sw)^(1/b)
    After switch: q(t) = q_sw * exp(-d_lim * (t - t_sw))
    """
    t = np.asarray(t, dtype=float)

    # Degenerate case — preserve legacy harmonic behaviour
    if b <= 0:
        return qi / (1.0 + di * t)

    # Initial decline already at or below terminal: pure exponential
    if di <= d_lim:
        return qi * np.exp(-d_lim * t)

    # Switch time where D_nom(t_sw) = d_lim
    t_sw = (di / d_lim - 1.0) / (b * di)
    q_sw = qi / ((1.0 + b * di * t_sw) ** (1.0 / b))

    q_hyp = qi / ((1.0 + b * di * t) ** (1.0 / b))
    q_exp = q_sw * np.exp(-d_lim * (t - t_sw))
    return np.where(t <= t_sw, q_hyp, q_exp)


def robust_loss(params: np.ndarray, t: np.ndarray, y: np.ndarray,
                d_lim: float = D_LIM_DEFAULT, delta: float = 1.0) -> float:
    qi, b, di = params
    return huber(delta, modified_arps(qi, b, di, d_lim, t) - y).mean()


def _smooth(x, w=3):
    s = pd.Series(x, dtype=float)
    return s.rolling(window=w, center=True, min_periods=1).mean().values


def _train_rf(df: pd.DataFrame, target_col: str) -> Dict[str, RandomForestRegressor]:
    """
    Train RF model for qi warm-start prediction.

    Uses per-well PEAK production as the target (a proper proxy for initial
    rate qi) instead of the mean.  Estimator count reduced to 60 and n_jobs=-1
    so training is faster and uses all available cores.
    b and di initial estimates are derived analytically per-well inside
    forecast_one_well, so stub models are returned for those keys.
    """
    agg_peak = (df.groupby('WellID')[['NormOil', 'NormGas', 'NormWater']]
                  .max().reset_index())
    agg_peak = agg_peak.dropna(subset=[target_col])

    X = agg_peak[['NormOil', 'NormGas', 'NormWater']].values
    if len(X) == 0:
        dummy = RandomForestRegressor(n_estimators=1, random_state=42)
        dummy.fit([[0, 0, 0]], [0])
        return {'qi': dummy, 'b': dummy, 'di': dummy}

    # qi: trained on peak production per well — best proxy for initial rate
    qi_model = RandomForestRegressor(
        n_estimators=60, random_state=42, n_jobs=-1
    ).fit(X, agg_peak[target_col].values)

    # b and di: single-node stubs; actual estimates computed analytically below
    stub = RandomForestRegressor(n_estimators=1, random_state=42)
    stub.fit([[0], [1]], [1.0, 1.0])
    return {'qi': qi_model, 'b': stub, 'di': stub}


def forecast_one_well(wd: pd.DataFrame, commodity: str, b_low: float, b_high: float,
                      max_months: int, models: Dict[str, RandomForestRegressor],
                      d_lim: float = D_LIM_DEFAULT) -> Dict[str, object]:
    commodity = commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[commodity]

    wd = wd.sort_values('MonthYear').copy()
    t_hist = np.arange(len(wd), dtype=float)
    y_hist = wd[col].values.astype(float)
    y_s = _smooth(y_hist, 3)

    peak_qi = float(np.nanmax(y_s)) if len(y_s) > 0 else 1.0
    peak_qi = min(peak_qi, 152000.0)

    # ---- Analytical initial parameter estimation ----
    # qi: RF prediction anchored to peak of smoothed history
    feats = np.array([[wd['NormOil'].mean(), wd['NormGas'].mean(), wd['NormWater'].mean()]],
                     dtype=float)
    try:
        qi0 = float(min(peak_qi, max(models['qi'].predict(feats)[0], 0.01)))
    except Exception:
        qi0 = peak_qi

    # di: log-linear regression on the declining portion after the peak month
    peak_idx = int(np.nanargmax(y_s)) if len(y_s) > 0 else 0
    decline_portion = y_s[peak_idx:]
    t_decline = np.arange(len(decline_portion), dtype=float)
    if len(decline_portion) >= 3:
        pos_mask = decline_portion > 0
        if pos_mask.sum() >= 3:
            slope, *_ = stats.linregress(t_decline[pos_mask],
                                         np.log(decline_portion[pos_mask]))
            di0 = float(min(max(-slope, d_lim), 1.0))
        else:
            di0 = 0.05
    else:
        di0 = 0.05

    # b: start at midpoint of allowed bounds
    b0 = 0.5 * (b_low + b_high)

    # ---- Shut-in / zero-production filtering for optimization ----
    # Months below 1% of peak (workovers, curtailments) distort the fit;
    # exclude them from the objective but keep them in EUR history totals.
    shutin_thresh = max(0.01 * peak_qi, 1.0)
    active = y_s > shutin_thresh
    t_fit = t_hist[active] if active.sum() >= 3 else t_hist
    y_fit = y_s[active]   if active.sum() >= 3 else y_s

    # ---- Bounded optimization (Modified Arps objective) ----
    # Lower bound on di = d_lim ensures optimizer stays in the physical regime
    # where the hyperbolic phase applies before the terminal switch.
    bounds = [(0.01, peak_qi * 1.5), (b_low, b_high), (d_lim, 1.0)]
    try:
        res = minimize(robust_loss, x0=[qi0, b0, di0],
                       args=(t_fit, y_fit, d_lim),
                       bounds=bounds, method='L-BFGS-B')
        qi, b, di = map(float, res.x)
    except Exception:
        qi, b, di = qi0, b0, di0

    # ---- Fit history with Modified Arps ----
    fit_hist = modified_arps(qi, b, di, d_lim, t_hist)

    # ---- Vectorised forecast (replaces per-month Python loop) ----
    t_fcst = np.arange(len(t_hist), len(t_hist) + max_months, dtype=float)
    f_vals_all = modified_arps(qi, b, di, d_lim, t_fcst)
    valid = f_vals_all >= 1e-6
    f_m = t_fcst[valid].astype(int)
    f_v = f_vals_all[valid]

    eur_hist = float(y_hist.sum())
    eur_fcst = float(np.sum(f_v))
    return dict(qi=qi, b=b, di=di, d_lim=d_lim,
                t_hist=t_hist, hist=y_hist,
                fit_hist=fit_hist, f_months=f_m, f_vals=f_v,
                EUR_total=eur_hist + eur_fcst, EUR_fcst=eur_fcst)


@dataclass
class ForecastConfig:
    commodity: str          # 'oil' | 'gas' | 'water'
    b_low: float = 0.8
    b_high: float = 1.2
    max_months: int = 600
    d_lim: float = D_LIM_DEFAULT      # terminal nominal decline rate (Modified Arps)
    min_months_history: int = 1       # wells with fewer months are skipped


def _forecast_well_job(well_id, wd: pd.DataFrame, com: str,
                       b_low: float, b_high: float, max_months: int,
                       models: Dict, d_lim: float):
    """Module-level worker for joblib parallel execution."""
    try:
        fc = forecast_one_well(wd, com, b_low, b_high, max_months, models, d_lim)
        return well_id, wd, fc
    except Exception:
        return None


def forecast_all(merged: pd.DataFrame,
                 cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Forecast all wells in parallel.

    Returns
    -------
    oneline_df    : one row per well with DCA parameters and EUR
    monthly_df    : monthly historical + forecast volumes per well
    well_forecasts: dict mapping WellID -> raw fc dict (for per-well plots)
    """
    com = cfg.commodity.lower()
    col = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}[com]

    models = _train_rf(merged, col)

    # Collect qualifying wells upfront (avoids repeated groupby inside threads)
    well_data = [
        (well_id, wd.copy())
        for well_id, wd in merged.groupby('WellID')
        if wd[col].max() > 0
        and wd[col].sum() > 0
        and len(wd) >= cfg.min_months_history
    ]

    # Parallel execution — threading backend is safe with Streamlit and
    # benefits from scipy's GIL release during L-BFGS-B optimisation.
    raw_results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_forecast_well_job)(
            well_id, wd, com,
            cfg.b_low, cfg.b_high, cfg.max_months,
            models, cfg.d_lim
        )
        for well_id, wd in well_data
    )

    oneline: list = []
    monthly: list = []
    well_forecasts: Dict[str, dict] = {}

    for result in raw_results:
        if result is None:
            continue
        well_id, wd, fc = result
        well_forecasts[str(well_id)] = fc

        hdr   = wd.iloc[0]
        well  = hdr.get('WellName', 'N/A')
        api10 = hdr.get('API10', 'N/A')

        qi_day     = fc['qi'] / 30.4
        q12        = float(modified_arps(fc['qi'], fc['b'], fc['di'], fc['d_lim'],
                                         np.array([12.0]))[0])
        decline_yr = (1.0 - (q12 / fc['qi'])) * 100.0 if fc['qi'] > 0 else 0.0

        row = {
            'WellID':             str(well_id),
            'API10':              str(api10),
            'WellName':           well,
            'WellNumber':         str(hdr.get('WellNumber', '')),
            'State':              hdr.get('State'),
            'County':             hdr.get('County'),
            'PrimaryFormation':   hdr.get('PrimaryFormation'),
            'LateralLength':      hdr.get('LateralLength'),
            'CompletionDate':     hdr.get('CompletionDate'),
            'FirstProdDate':      hdr.get('FirstProdDate'),
            'qi (per day)':       round(qi_day, 0),
            'b':                  round(fc['b'], 3),
            'di (per month)':     round(fc['di'], 4),
            'First-Year Decline (%)': round(decline_yr, 1),
        }
        if com == 'oil':
            row.update({'EUR (Mbbl)':        round(fc['EUR_total'] / 1_000.0, 2),
                        'Remaining (Mbbl)':  round(fc['EUR_fcst']  / 1_000.0, 2)})
        elif com == 'gas':
            row.update({'EUR (MMcf)':        round(fc['EUR_total'] / 1_000_000.0, 2),
                        'Remaining (MMcf)':  round(fc['EUR_fcst']  / 1_000_000.0, 2)})
        else:
            row.update({'EUR (Mbbl water)':       round(fc['EUR_total'] / 1_000.0, 2),
                        'Remaining (Mbbl water)': round(fc['EUR_fcst']  / 1_000.0, 2)})
        oneline.append(row)

        hist_dates = wd.sort_values('MonthYear')['MonthYear'].dt.to_timestamp(how='start')
        hist_vals  = wd.sort_values('MonthYear')[col].values.astype(float)
        start = (hist_dates.iloc[-1] + pd.offsets.MonthBegin(1) if len(hist_dates) > 0
                 else merged['MonthYear'].min().to_timestamp(how='start'))
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
    return oneline_df, monthly_df, well_forecasts


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
    # Use WellID for title if API is empty
    if not api or api == 'nan':
        api = wd.iloc[0].get('WellID','N/A')

    fig, ax = plt.subplots(figsize=(10,6))
    if len(hist_dates) > 0:
        ax.scatter(hist_dates, hist_vals, label="Historical", s=22, color=color)
        ax.plot(fit_dates, fit_hist, label="Fit (history)", linewidth=2, color=color)
    if len(f_dates) > 0:
        ax.plot(f_dates, f_vals, label="Forecast", linestyle="--", linewidth=2, color=color)

    ax.set_title(f"{well} | ID: {api} | {commodity.capitalize()}")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Monthly {commodity} {unit}")
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)
    ax.grid(True, linestyle="--", alpha=0.4, which='both')
    ax.legend()
    fig.tight_layout()
    return fig
