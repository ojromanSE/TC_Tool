from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from scipy.special import huber
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy import stats

# ---------------- Provider column maps (WDB & DI/IHS) ----------------
# We target these standard column names internally:
REQUIRED_HEADER_COLUMNS = [
    'WellName','County','LateralLength','PrimaryFormation',
    'CompletionDate','FirstProdDate','State','API10'
]
REQUIRED_PROD_COLUMNS = [
    'WellName','ReportDate','TotalOil','TotalGas','TotalWater','API10'
]

# Maps are {Target: Source}. We look for 'Source' in the input CSV.
HEADER_COLUMN_MAPS = [
    # WDB-style
    {'WellName':'WellName','County':'County','LateralLength':'LateralLength',
     'PrimaryFormation':'PrimaryFormation','CompletionDate':'CompletionDate',
     'FirstProdDate':'FirstProdDate','State':'State','API10':'API'},
    # DI / IHS-style
    {'WellName':'Well Name','County':'County/Parish','LateralLength':'DI Lateral Length',
     'PrimaryFormation':'Producing Reservoir','CompletionDate':'Completion Date',
     'FirstProdDate':'First Prod Date','State':'State','API10':'API10'}
]
PROD_COLUMN_MAPS = [
    # WDB-style
    {'WellName':'WellName','ReportDate':'ReportDate','TotalOil':'TotalOil',
     'TotalGas':'TotalGas','TotalWater':'TotalWater','API10':'API'},
    # DI / IHS-style
    {'WellName':'Well Name','ReportDate':'Monthly Production Date','TotalOil':'Monthly Oil',
     'TotalGas':'Monthly Gas','TotalWater':'Monthly Water','API10':'API10'}
]

def _normalize_header_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Helper: fix common missing columns before strict mapping."""
    # 1. Fix API: Ensure we have 'API' (for WDB map) and 'API10' (for DI map/internal use)
    # If we have API14 but not API10, create API10
    if 'API14' in df.columns and 'API10' not in df.columns:
        df['API10'] = df['API14'].astype(str).str[:10]
    
    # If we have API10 but not API (WDB map expects 'API'), alias it.
    if 'API10' in df.columns and 'API' not in df.columns:
        df['API'] = df['API10']
    
    # 2. Fix LateralLength: Check aliases
    if 'LateralLength' not in df.columns:
        for alias in ['HorizontalLength', 'Horizontal Length', 'Lateral Length']:
            if alias in df.columns:
                df['LateralLength'] = df[alias]
                break
    
    # 3. Fix LateralLength: If still missing, but Lat/Lon exist, create as NaN 
    # (So mapping succeeds, and fill_lateral_by_geo can fill it later)
    if 'LateralLength' not in df.columns and 'Latitude' in df.columns:
        df['LateralLength'] = np.nan

    # 4. Fix Dates: If CompletionDate missing, try aliases
    if 'CompletionDate' not in df.columns:
        for alias in ['Completion Date', 'SpudDate', 'Spud Date']:
            if alias in df.columns:
                df['CompletionDate'] = df[alias] # Fallback to spud if completion missing
                break
                
    # 5. Fix FirstProdDate: If missing, try aliases
    if 'FirstProdDate' not in df.columns:
        for alias in ['First Prod Date', 'FirstProductionDate']:
            if alias in df.columns:
                df['FirstProdDate'] = df[alias]
                break

    return df

def _translate(df: pd.DataFrame, maps, required) -> pd.DataFrame:
    for m in maps:
        # Check if all source columns (values in map) exist in df
        if all(c in df.columns for c in m.values()):
            out = df.rename(columns={v:k for k,v in m.items()})
            # Double check we have all required targets
            miss = [c for c in required if c not in out.columns]
            if miss: 
                # Should not happen if map check passed, unless map is incomplete vs required
                continue 
            return out[required].copy()
            
    # If we get here, no map matched completely.
    raise ValueError(f"Could not map columns. Found in file: {list(df.columns)}")

def load_header(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    df = _normalize_header_columns(df)
    return _translate(df, HEADER_COLUMN_MAPS, REQUIRED_HEADER_COLUMNS)

def load_production(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    
    # Normalize Prod columns
    if 'API/UWI' in df.columns and 'API10' not in df.columns:
        df['API10'] = df['API/UWI'].astype(str).str[:10]
    if 'API10' in df.columns and 'API' not in df.columns:
        df['API'] = df['API10']
        
    out = _translate(df, PROD_COLUMN_MAPS, REQUIRED_PROD_COLUMNS)
    out['ReportDate'] = pd.to_datetime(out['ReportDate'])
    return out

# ---------------- Header QC (optional helper) ----------------
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
        
    # Treat 0 or NaN as missing
    df[lateral_col] = pd.to_numeric(df[lateral_col], errors='coerce')
    miss = df[lateral_col].isna() | (df[lateral_col] == 0)
    
    df['lat_bin'] = df[lat_col].round(decimals)
    df['lon_bin'] = df[lon_col].round(decimals)
    valid = df[~miss & df[lateral_col].notna()]
    
    if valid.empty:
        # Avoid crash if no wells have valid lateral length yet
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

# ---------------- Preprocess: merge + normalization ----------------
@dataclass
class PreprocessConfig:
    normalization_length: int = 10_000
    use_normalization: bool = True

def preprocess(header_df: pd.DataFrame, prod_df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    hd = header_df.copy(); pr = prod_df.copy()
    hd['API10'] = hd['API10'].astype(str); pr['API10'] = pr['API10'].astype(str)
    pr['MonthYear'] = pr['ReportDate'].dt.to_period('M')

    # FIX: Don't filter out 0-production rows. "Use all production".
    pr = pr.dropna(subset=['TotalOil','TotalGas','TotalWater'])
    
    pr = pr.sort_values(['API10','MonthYear','TotalOil','TotalGas','TotalWater'],
                        ascending=[True,True,False,False,False])
    pr = pr.drop_duplicates(subset=['API10','MonthYear'], keep='first')

    keep_hdr = ['API10','WellName','State','County','PrimaryFormation',
                'LateralLength','CompletionDate','FirstProdDate']
    keep_hdr = [c for c in keep_hdr if c in hd.columns]
    merged = pr.merge(hd[keep_hdr], on='API10', how='inner')

    if cfg.use_normalization:
        # Avoid division by zero/NaN if lateral is still missing
        merged['LateralLength'] = merged['LateralLength'].replace(0, np.nan)
        s = cfg.normalization_length / merged['LateralLength']
        merged['NormOil']   = merged['TotalOil']*s
        merged['NormGas']   = merged['TotalGas']*s
        merged['NormWater'] = merged['TotalWater']*s
    else:
        merged['NormOil']   = merged['TotalOil']
        merged['NormGas']   = merged['TotalGas']
        merged['NormWater'] = merged['TotalWater']

    # FIX: inplace=True is deprecated
    merged = merged.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # We still drop rows where Normalized values became NaN (due to missing LateralLength)
    # but we DO NOT drop 0 production.
    merged = merged.dropna(subset=['NormOil','NormGas','NormWater'])

    return merged

# ---------------- Decline model + RF seeding ----------------
def arps(qi: float, b: float, di: float, t: np.ndarray) -> np.ndarray:
    if b==0 or b==1: return qi/(1.0+di*t)
    return qi/((1.0+b*di*t)**(1.0/b))

def robust_loss(params: np.ndarray, t: np.ndarray, y: np.ndarray, delta: float=1.0) -> float:
    qi,b,di = params
    return huber(delta, arps(qi,b,di,t) - y).mean()

def _smooth(x, w=3):
    s = pd.Series(x, dtype=float)
    return s.rolling(window=w, center=True, min_periods=1).mean().values

def _train_rf(df: pd.DataFrame, target_col: str) -> Dict[str, RandomForestRegressor]:
    # Train only on non-zero averages to avoid training on dead wells
    agg = (df.groupby('API10')[['NormOil','NormGas','NormWater']]
             .mean().reset_index())
    agg = agg.dropna(subset=[target_col])
    
    X = agg[['NormOil','NormGas','NormWater']].values
    if len(X) == 0:
        # Fallback if no training data
        dummy_rf = RandomForestRegressor(n_estimators=1, random_state=42)
        # fit on dummy data
        dummy_rf.fit([[0,0,0]], [0])
        return {'qi': dummy_rf, 'b': dummy_rf, 'di': dummy_rf}
        
    qi_model = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, agg[target_col].values)
    b_model  = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.full(len(agg), 1.0))
    di_model = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.full(len(agg), 0.05))
    return {'qi': qi_model, 'b': b_model, 'di': di_model}

def forecast_one_well(wd: pd.DataFrame, commodity: str, b_low: float, b_high: float,
                      max_months: int, models: Dict[str, RandomForestRegressor]) -> Dict[str, object]:
    commodity = commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[commodity]

    wd = wd.sort_values('MonthYear').copy()
    t_hist = np.arange(len(wd), dtype=float)
    y_hist = wd[col].values.astype(float)
    y_s = _smooth(y_hist, 3)

    feats = np.array([[wd['NormOil'].mean(), wd['NormGas'].mean(), wd['NormWater'].mean()]], dtype=float)
    
    # Safety predict
    try:
        qi_pred = float(models['qi'].predict(feats)[0])
        b_pred = float(models['b'].predict(feats)[0])
        di_pred = float(models['di'].predict(feats)[0])
    except:
        qi_pred, b_pred, di_pred = 100.0, 1.0, 0.05

    peak_qi = float(np.nanmax(y_s)) if len(y_s)>0 else 1.0
    peak_qi = min(peak_qi, 152000.0)
    qi0 = min(peak_qi, max(qi_pred, 0.01))
    b0  = b_pred
    di0 = di_pred

    bounds = [(0.01, peak_qi*1.5), (b_low, b_high), (1e-6, 1.0)]
    try:
        res = minimize(robust_loss, x0=[qi0,b0,di0], args=(t_hist, y_s),
                       bounds=bounds, method='L-BFGS-B')
        qi,b,di = map(float, res.x)
    except:
        qi,b,di = qi0, b0, di0

    fit_hist = arps(qi,b,di,t_hist)
    f_m, f_v = [], []
    m = len(t_hist)
    while m < len(t_hist)+max_months:
        q = float(arps(qi,b,di,np.array([m],float))[0])
        if q < 1e-6: break
        f_m.append(m); f_v.append(q); m += 1

    eur_hist = float(y_hist.sum())
    eur_fcst = float(np.sum(f_v))
    return dict(qi=qi, b=b, di=di, t_hist=t_hist, hist=y_hist,
                fit_hist=fit_hist, f_months=np.array(f_m,int), f_vals=np.array(f_v,float),
                EUR_total=eur_hist+eur_fcst, EUR_fcst=eur_fcst)

# ---------------- All-wells forecast ----------------
@dataclass
class ForecastConfig:
    commodity: str      # 'oil'|'gas'|'water'
    b_low: float = 0.8
    b_high: float = 1.2
    max_months: int = 600

def forecast_all(merged: pd.DataFrame, cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    com = cfg.commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[com]
    models = _train_rf(merged, col)

    oneline, monthly = [], []
    for api10, wd in merged.groupby('API10'):
        # Still skip if *all* history is <= 0 (dead well), otherwise we can't fit
        if wd[col].max()<=0 or wd[col].sum()<=0:
            continue
        fc = forecast_one_well(wd, com, cfg.b_low, cfg.b_high, cfg.max_months, models)
        hdr = wd.iloc[0]; well = hdr.get('WellName','N/A')

        qi_day = fc['qi']/30.4
        q12 = float(arps(fc['qi'], fc['b'], fc['di'], np.array([12.0]))[0])
        decline_yr = (1.0 - (q12/fc['qi']))*100.0 if fc['qi']>0 else 0.0

        row = {
            'API10': str(api10), 'WellName': well,
            'State': hdr.get('State'), 'County': hdr.get('County'),
            'PrimaryFormation': hdr.get('PrimaryFormation'),
            'LateralLength': hdr.get('LateralLength'),
            'CompletionDate': hdr.get('CompletionDate'),
            'FirstProdDate': hdr.get('FirstProdDate'),
            'qi (per day)': round(qi_day,0),
            'b': round(fc['b'],3),
            'di (per month)': round(fc['di'],4),
            'First-Year Decline (%)': round(decline_yr,1),
        }
        if com=='oil':
            row.update({'EUR (Mbbl)': round(fc['EUR_total']/1_000.0,2),
                        'Remaining (Mbbl)': round(fc['EUR_fcst']/1_000.0,2)})
        elif com=='gas':
            row.update({'EUR (MMcf)': round(fc['EUR_total']/1_000_000.0,2),
                        'Remaining (MMcf)': round(fc['EUR_fcst']/1_000_000.0,2)})
        else:
            row.update({'EUR (Mbbl water)': round(fc['EUR_total']/1_000.0,2),
                        'Remaining (Mbbl water)': round(fc['EUR_fcst']/1_000.0,2)})
        oneline.append(row)

        hist_dates = wd.sort_values('MonthYear')['MonthYear'].dt.to_timestamp(how='start')
        hist_vals  = wd[col].values.astype(float)
        start = hist_dates.iloc[-1] + pd.offsets.MonthBegin(1) if len(hist_dates)>0 \
                else merged['MonthYear'].min().to_timestamp(how='start')
        f_dates = pd.date_range(start=start, periods=len(fc['f_vals']), freq='MS')

        for d,v in zip(hist_dates, hist_vals):
            monthly.append({'API10':str(api10),'WellName':well,'Date':d,
                            f'Monthly_{com}_volume':float(v),'Segment':'Historical'})
        for d,v in zip(f_dates, fc['f_vals']):
            monthly.append({'API10':str(api10),'WellName':well,'Date':d,
                            f'Monthly_{com}_volume':float(v),'Segment':'Forecast'})

    oneline_df = pd.DataFrame(oneline)
    monthly_df = pd.DataFrame(monthly).sort_values(['API10','Date','Segment'])
    return oneline_df, monthly_df

# ---------------- EUR stats & Probit (color support) ----------------
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
    """
    Build a small summary table for EUR stats, plus a per-ft column.
    - unit: 'Mbbl' or 'MMcf' (table is unit-aware)
    - norm_len: normalization length in ft (used to compute per-ft numbers)
    """
    # For liquids we typically show per-ft in bbl/ft (Mbbl → bbl multiplier = 1000).
    # For gas the unit is already MMcf; leave factor = 1.0 to get MMcf/ft.
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

# ---------------- Single-well plot with fixed colors + LOG Y ----------------
def plot_one_well(wd: pd.DataFrame, fc: dict, commodity: str):
    """
    Plot historical monthly volumes, fitted curve over history, and forward forecast.
    Colors: oil=green, gas=red, water=blue. Points and lines both colored.
    Y-axis is logarithmic (values clamped to small epsilon for display).
    """
    com = commodity.lower()
    col_map = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}
    color_map = {'oil':'green','gas':'red','water':'blue'}
    col = col_map[com]
    color = color_map[com]
    eps = 1e-6  # for log-scale safety

    # Prepare historical series
    wd = wd.sort_values('MonthYear').copy()
    hist_dates = wd['MonthYear'].dt.to_timestamp(how='start')
    hist_vals  = np.clip(wd[col].astype(float).values, eps, None)

    # Fitted (over historical months)
    fit_hist = np.clip(fc['fit_hist'], eps, None)
    fit_dates = hist_dates  # 1:1 with history

    # Forecast series
    if len(hist_dates) > 0:
        start = hist_dates.iloc[-1] + pd.offsets.MonthBegin(1)
    else:
        start = pd.Timestamp.today().normalize().replace(day=1)
    f_dates = pd.date_range(start=start, periods=len(fc['f_vals']), freq='MS')
    f_vals  = np.clip(fc['f_vals'], eps, None)

    # Labels / units
    unit = {"oil":"(normalized bbl/month)",
            "gas":"(normalized Mcf/month)",
            "water":"(normalized bbl/month)"}[com]
    well = wd.iloc[0].get('WellName','N/A')
    api  = str(wd.iloc[0].get('API10',''))

    # Plot
    fig, ax = plt.subplots(figsize=(10,6))
    if len(hist_dates) > 0:
        ax.scatter(hist_dates, hist_vals, label="Historical", s=22, color=color)
        ax.plot(fit_dates, fit_hist, label="Fit (history)", linewidth=2, color=color)
    if len(f_dates) > 0:
        ax.plot(f_dates, f_vals, label="Forecast", linestyle="--", linewidth=2, color=color)

    ax.set_title(f"{well}  |  API10 {api}  |  {commodity.capitalize()} forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Monthly {commodity} {unit}")
    ax.set_yscale('log')
    # force the lower bound to 10^0
    ax.set_ylim(bottom=1)  
    ax.grid(True, linestyle="--", alpha=0.4, which='both')
    ax.legend()
    fig.tight_layout()
    return fig
