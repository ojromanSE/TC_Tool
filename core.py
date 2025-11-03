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
REQUIRED_HEADER_COLUMNS = [
    'WellName','County','LateralLength','PrimaryFormation',
    'CompletionDate','FirstProdDate','State','API10'
]
REQUIRED_PROD_COLUMNS = [
    'WellName','ReportDate','TotalOil','TotalGas','TotalWater','API10'
]
HEADER_COLUMN_MAPS = [
    # WDB
    {'WellName':'WellName','County':'County','LateralLength':'LateralLength',
     'PrimaryFormation':'PrimaryFormation','CompletionDate':'CompletionDate',
     'FirstProdDate':'FirstProdDate','State':'State','API10':'API'},
    # DI / IHS
    {'WellName':'Well Name','County':'County/Parish','LateralLength':'DI Lateral Length',
     'PrimaryFormation':'Producing Reservoir','CompletionDate':'Completion Date',
     'FirstProdDate':'First Prod Date','State':'State','API10':'API10'}
]

# IMPORTANT: include DI/IHS variant that has API (not API10)
PROD_COLUMN_MAPS = [
    # WDB (monthly) -> uses API
    {'WellName':'WellName','ReportDate':'ReportDate','TotalOil':'TotalOil',
     'TotalGas':'TotalGas','TotalWater':'TotalWater','API10':'API'},

    # DI / IHS (monthly) -> prefers API10
    {'WellName':'Well Name','ReportDate':'Monthly Production Date','TotalOil':'Monthly Oil',
     'TotalGas':'Monthly Gas','TotalWater':'Monthly Water','API10':'API10'},

    # DI / IHS (monthly) variant -> uses API (we normalize API10 earlier)
    {'WellName':'Well Name','ReportDate':'Monthly Production Date','TotalOil':'Monthly Oil',
     'TotalGas':'Monthly Gas','TotalWater':'Monthly Water','API10':'API'},

    # Generic daily passed through monthly loader (compatibility)
    {'WellName':'WellName','ReportDate':'Date','TotalOil':'DailyOil',
     'TotalGas':'DailyGas','TotalWater':'DailyWater','API10':'API'}
]

# --- Daily production column maps (common variants) ---
DAILY_PROD_COLUMN_MAPS = [
    # DI / IHS style
    {'WellName':'Well Name','Date':'Production Date','DailyOil':'Daily Oil',
     'DailyGas':'Daily Gas','DailyWater':'Daily Water','API':'API10'},
    # Generic
    {'WellName':'WellName','Date':'Date','DailyOil':'DailyOil',
     'DailyGas':'DailyGas','DailyWater':'DailyWater','API':'API'},
    # Another common export
    {'WellName':'Well Name','Date':'Prod Date','DailyOil':'Oil (bbl/day)',
     'DailyGas':'Gas (mcf/day)','DailyWater':'Water (bbl/day)','API':'API10'},
]

# ---------------- Translation helpers ----------------
def _translate(df: pd.DataFrame, maps, required) -> pd.DataFrame:
    for m in maps:
        if all(c in df.columns for c in m.values()):
            out = df.rename(columns={v:k for k,v in m.items()})
            miss = [c for c in required if c not in out.columns]
            if miss: raise ValueError(f"Missing required cols after rename: {miss}")
            return out[required].copy()
    raise ValueError(f"Could not map columns. Found: {df.columns.tolist()}")

def _try_translate(df: pd.DataFrame, maps_list, required) -> pd.DataFrame | None:
    for m in maps_list:
        try:
            return _translate(df, [m], required)
        except Exception:
            pass
    return None

def load_header(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    return _translate(df, HEADER_COLUMN_MAPS, REQUIRED_HEADER_COLUMNS)

def load_production(file_like) -> pd.DataFrame:
    """
    Monthly loader that also accepts a few daily-shaped files (compat path).
    It normalizes API to a 10-digit 'API10' if the source only has API or API/UWI.
    Output columns: ['WellName','ReportDate','TotalOil','TotalGas','TotalWater','API10']
    """
    df = pd.read_csv(file_like)

    # Normalize API10 early so DI/IHS mappings can match regardless of source header
    if 'API10' not in df.columns:
        if 'API' in df.columns:
            df['API10'] = df['API'].astype(str).str[:10]
        elif 'API/UWI' in df.columns:
            df['API10'] = df['API/UWI'].astype(str).str[:10]

    out = _translate(df, PROD_COLUMN_MAPS, REQUIRED_PROD_COLUMNS)
    out['ReportDate'] = pd.to_datetime(out['ReportDate'], errors='coerce')
    out = out.dropna(subset=['ReportDate'])
    return out

def load_production_daily(file_like) -> pd.DataFrame:
    """
    Load DAILY production CSVs. Returns standardized columns:
      ['WellName','Date','DailyOil','DailyGas','DailyWater','API10']
    Accepts API (10/14) or API/UWI; trims to API10.
    """
    raw = pd.read_csv(file_like)

    # If only API/UWI present, derive API
    if 'API/UWI' in raw.columns and 'API' not in raw.columns and 'API10' not in raw.columns:
        raw['API'] = raw['API/UWI']

    df = None
    # Already standardized?
    if {'WellName','Date','DailyOil','DailyGas','DailyWater','API10'}.issubset(raw.columns):
        df = raw.copy()
    elif {'WellName','Date','DailyOil','DailyGas','DailyWater','API'}.issubset(raw.columns):
        df = raw.rename(columns={'API':'API10'})
    else:
        df = _try_translate(
            raw,
            DAILY_PROD_COLUMN_MAPS,
            required=['WellName','Date','DailyOil','DailyGas','DailyWater','API']
        )
        if df is not None and 'API10' not in df.columns:
            df = df.rename(columns={'API':'API10'})

    if df is None:
        raise ValueError(f"Could not map daily columns. Found: {raw.columns.tolist()}")

    df['API10'] = df['API10'].astype(str).str[:10]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for c in ('DailyOil','DailyGas','DailyWater'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Date'])
    return df[['WellName','Date','DailyOil','DailyGas','DailyWater','API10']]

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
    miss = df[lateral_col].isna() | (df[lateral_col] == 0)
    df['lat_bin'] = df[lat_col].round(decimals)
    df['lon_bin'] = df[lon_col].round(decimals)
    valid = df[~miss & df[lateral_col].notna()]
    bin_means = (valid.groupby(['lat_bin','lon_bin'])[lateral_col]
                      .mean().rename('bin_mean'))
    df = df.merge(bin_means, left_on=['lat_bin','lon_bin'], right_index=True, how='left')
    df['LateralImputed'] = False
    fill_mask = miss & df['bin_mean'].notna()
    df.loc[fill_mask, lateral_col] = df.loc[fill_mask, 'bin_mean']
    df.loc[fill_mask, 'LateralImputed'] = True
    df['LateralImputeNote'] = np.where(df['LateralImputed'], 'Filled by geo bin', 'As provided')
    return df.drop(columns=['lat_bin','lon_bin','bin_mean'], errors='ignore')

# ---------------- Preprocess: merge + normalization (monthly) ----------------
@dataclass
class PreprocessConfig:
    normalization_length: int = 10_000
    use_normalization: bool = True


def preprocess(header_df: pd.DataFrame, prod_df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    Merge monthly production with header using WellName instead of API10.
    """
    hd = header_df.copy()
    pr = prod_df.copy()

    # Normalize WellName
    hd['WellName'] = hd['WellName'].astype(str).str.strip().str.upper()
    pr['WellName'] = pr['WellName'].astype(str).str.strip().str.upper()

    pr['MonthYear'] = pr['ReportDate'].dt.to_period('M')
    pr = pr.dropna(subset=['TotalOil','TotalGas','TotalWater'])
    pr = pr[(pr['TotalOil']>0)|(pr['TotalGas']>0)|(pr['TotalWater']>0)]

    pr = pr.sort_values(['WellName','MonthYear','TotalOil','TotalGas','TotalWater'],
                        ascending=[True,True,False,False,False])
    pr = pr.drop_duplicates(subset=['WellName','MonthYear'], keep='first')

    keep_hdr = ['WellName','State','County','PrimaryFormation','LateralLength',
                'CompletionDate','FirstProdDate']
    keep_hdr = [c for c in keep_hdr if c in hd.columns]

    merged = pr.merge(hd[keep_hdr], on='WellName', how='inner')

    if cfg.use_normalization:
        s = cfg.normalization_length / merged['LateralLength']
        merged['NormOil']   = merged['TotalOil']*s
        merged['NormGas']   = merged['TotalGas']*s
        merged['NormWater'] = merged['TotalWater']*s
    else:
        merged['NormOil']   = merged['TotalOil']
        merged['NormGas']   = merged['TotalGas']
        merged['NormWater'] = merged['TotalWater']

    merged.replace([np.inf,-np.inf], np.finfo(np.float64).max, inplace=True)
    merged = merged[(merged['NormOil']>0)|(merged['NormGas']>0)|(merged['NormWater']>0)]
    return merged


@dataclass
class PreprocessDailyConfig:
    normalization_length: int = 10_000
    use_normalization: bool = True


def preprocess_daily(header_df: pd.DataFrame, daily_df: pd.DataFrame,
                     cfg: PreprocessDailyConfig) -> pd.DataFrame:
    """
    Merge daily production with header using WellName instead of API10.
    """
    hd = header_df.copy()
    pr = daily_df.copy()

    hd['WellName'] = hd['WellName'].astype(str).str.strip().str.upper()
    pr['WellName'] = pr['WellName'].astype(str).str.strip().str.upper()

    pr = pr.dropna(subset=['DailyOil','DailyGas','DailyWater'])
    pr = pr[(pr['DailyOil']>0)|(pr['DailyGas']>0)|(pr['DailyWater']>0)]

    keep_hdr = ['WellName','State','County','PrimaryFormation','LateralLength',
                'CompletionDate','FirstProdDate']
    keep_hdr = [c for c in keep_hdr if c in hd.columns]

    merged = pr.merge(hd[keep_hdr], on='WellName', how='inner')

    if cfg.use_normalization:
        s = cfg.normalization_length / merged['LateralLength']
        merged['NormOil']   = merged['DailyOil']*s
        merged['NormGas']   = merged['DailyGas']*s
        merged['NormWater'] = merged['DailyWater']*s
    else:
        merged['NormOil']   = merged['DailyOil']
        merged['NormGas']   = merged['DailyGas']
        merged['NormWater'] = merged['DailyWater']

    merged.replace([np.inf,-np.inf], np.finfo(np.float64).max, inplace=True)
    merged = merged[(merged['NormOil']>0)|(merged['NormGas']>0)|(merged['NormWater']>0)]
    merged['Day'] = pd.to_datetime(merged['Date'], errors='coerce').dt.normalize()
    merged['MonthYear'] = merged['Day'].dt.to_period('M')
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
    """
    Small RandomForest regressors used only as seeds for the decline fit.
    Deterministic (random_state=42).
    """
    agg = (
        df.groupby('API10')[['NormOil','NormGas','NormWater']]
          .mean().reset_index().dropna(subset=[target_col])
    )
    if agg.empty:
        X = np.zeros((1, 3), dtype=float)
        qi_model = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.array([1.0]))
        b_model  = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.array([1.0]))
        di_model = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.array([0.05]))
        return {'qi': qi_model, 'b': b_model, 'di': di_model}

    X = agg[['NormOil','NormGas','NormWater']].values
    qi_model = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, agg[target_col].values)
    b_model  = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.full(len(agg), 1.0))
    di_model = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, np.full(len(agg), 0.05))
    return {'qi': qi_model, 'b': b_model, 'di': di_model}

# ---------------- Monthly forecast (existing) ----------------
@dataclass
class ForecastConfig:
    commodity: str      # 'oil'|'gas'|'water'
    b_low: float = 0.8
    b_high: float = 1.2
    max_months: int = 600

def forecast_one_well(wd: pd.DataFrame, commodity: str, b_low: float, b_high: float,
                      max_months: int, models: Dict[str, RandomForestRegressor]) -> Dict[str, object]:
    commodity = commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[commodity]

    wd = wd.sort_values('MonthYear').copy()
    t_hist = np.arange(len(wd), dtype=float)
    y_hist = wd[col].values.astype(float)
    y_s = _smooth(y_hist, 3)

    feats = np.array([[wd['NormOil'].mean(), wd['NormGas'].mean(), wd['NormWater'].mean()]], dtype=float)
    peak_qi = float(np.nanmax(y_s)) if len(y_s)>0 else 1.0
    peak_qi = min(peak_qi, 152000.0)
    qi0 = min(peak_qi, max(float(models['qi'].predict(feats)[0]), 0.01))
    b0  = float(models['b'].predict(feats)[0])
    di0 = float(models['di'].predict(feats)[0])

    bounds = [(0.01, peak_qi*1.5), (b_low, b_high), (1e-6, 1.0)]
    res = minimize(robust_loss, x0=[qi0,b0,di0], args=(t_hist, y_s),
                   bounds=bounds, method='L-BFGS-B')
    qi,b,di = map(float, res.x)

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

def forecast_all(merged: pd.DataFrame, cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    com = cfg.commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[com]
    models = _train_rf(merged, col)

    oneline, monthly = [], []
    for api10, wd in merged.groupby('API10'):
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

    # Build monthly table (hist + fcast)
    for api10, wd in merged.groupby('API10'):
        hdr = wd.iloc[0]; well = hdr.get('WellName','N/A')
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
    monthly_df = pd.DataFrame(monthly)
    if monthly_df.empty:
        monthly_df = pd.DataFrame(
            columns=['API10','WellName','Date',f'Monthly_{com}_volume','Segment']
        )
    else:
        sort_cols = [c for c in ['API10','Date','Segment'] if c in monthly_df.columns]
        if sort_cols:
            monthly_df = monthly_df.sort_values(sort_cols)
    return oneline_df, monthly_df

# ---------------- DAILY helpers & forecast ----------------
def _aggregate_daily_to_monthly(df: pd.DataFrame, com: str) -> pd.DataFrame:
    """
    Convert a daily, normalized dataframe to monthly totals.
    Robust to missing MonthYear / WellName and to Norm* not present.
    Returns columns: ['API10','WellName','Date', f'Monthly_{com}_volume', 'Segment']
    """
    out_cols = ['API10','WellName','Date', f'Monthly_{com}_volume', 'Segment']

    if df is None or df.empty:
        return pd.DataFrame(columns=out_cols)

    df = df.copy()

    # ---- Ensure API10 as string
    if 'API10' not in df.columns:
        raise KeyError("Daily data missing 'API10' column after preprocess_daily().")
    df['API10'] = df['API10'].astype(str)

    # ---- Ensure WellName present (fallback to API10)
    if 'WellName' not in df.columns:
        df['WellName'] = df['API10']
    else:
        # If fully/mostly missing, fill from API10
        if df['WellName'].isna().all():
            df['WellName'] = df['API10']
        else:
            df['WellName'] = df['WellName'].fillna(df['API10'])

    # ---- Find a usable date column and build MonthYear
    if 'MonthYear' not in df.columns:
        date_col = None
        for cand in ['Day', 'Date', 'ReportDate', 'ProductionDate']:
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            raise KeyError(
                f"Daily data missing a date column (expected one of Day/Date/ReportDate/ProductionDate). "
                f"Columns seen: {list(df.columns)}"
            )
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df['MonthYear'] = df[date_col].dt.to_period('M')

    # ---- Pick the correct volume column
    norm_map = {'oil': 'NormOil', 'gas': 'NormGas', 'water': 'NormWater'}
    daily_map = {'oil': 'DailyOil', 'gas': 'DailyGas', 'water': 'DailyWater'}
    vol_col = norm_map[com]

    if vol_col not in df.columns:
        # Fallback to daily raw if normalization somehow wasn’t applied (shouldn’t happen, but safe)
        if daily_map[com] in df.columns:
            vol_col = daily_map[com]
        else:
            raise KeyError(
                f"Neither '{norm_map[com]}' nor '{daily_map[com]}' present in daily data for commodity '{com}'. "
                f"Columns: {list(df.columns)}"
            )

    # ---- Group and output
    g = (
        df.groupby(['API10','WellName','MonthYear'], as_index=False)[vol_col]
          .sum()
          .rename(columns={vol_col: f"Monthly_{com}_volume"})
    )
    g['Date'] = g['MonthYear'].dt.to_timestamp(how='start')
    g['Segment'] = 'Historical'
    return g[['API10','WellName','Date', f'Monthly_{com}_volume', 'Segment']]

def forecast_one_well_daily(wd: pd.DataFrame, commodity: str,
                            b_low: float, b_high: float, max_days: int,
                            models: Dict[str, RandomForestRegressor]) -> Dict[str, object]:
    """
    Fit on DAILY Norm* series. Decline parameters are per-month, so convert
    days to 'months' using 30.4375 day/month. Works with Day/Date/ReportDate.
    """
    commodity = commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[commodity]

    wd = wd.copy()

    # --- pick a valid date column
    date_col = None
    for cand in ['Day', 'Date', 'ReportDate', 'ProductionDate']:
        if cand in wd.columns:
            date_col = cand
            break
    if date_col is None:
        raise KeyError("No date column found in daily data (expected Day/Date/ReportDate).")

    wd[date_col] = pd.to_datetime(wd[date_col], errors='coerce')
    wd = wd.dropna(subset=[date_col]).sort_values(date_col).copy()

    # time in months from first day
    t_days = (wd[date_col] - wd[date_col].iloc[0]).dt.days.astype(float)
    t_hist = t_days / 30.4375

    y_hist = wd[col].astype(float).values
    if y_hist.size == 0 or (np.nanmax(y_hist) if np.size(y_hist) else 0) <= 0:
        return dict(qi=np.nan, b=np.nan, di=np.nan,
                    t_hist=np.array([]), hist=np.array([]),
                    fit_hist=np.array([]),
                    f_days=np.array([]), f_vals=np.array([]),
                    EUR_total=np.nan, EUR_fcst=np.nan)

    # gentle smoothing for daily
    y_s = _smooth(y_hist, 5)

    feats = np.array([[wd['NormOil'].mean(), wd['NormGas'].mean(), wd['NormWater'].mean()]], dtype=float)
    peak_qi = float(np.nanmax(y_s)) if len(y_s) > 0 else 1.0
    peak_qi = min(peak_qi, 152000.0)
    qi0 = min(peak_qi, max(float(models['qi'].predict(feats)[0]), 0.01))
    b0  = float(models['b'].predict(feats)[0])
    di0 = float(models['di'].predict(feats)[0])

    bounds = [(0.01, peak_qi*1.5), (b_low, b_high), (1e-6, 1.0)]
    res = minimize(robust_loss, x0=[qi0,b0,di0], args=(t_hist, y_s),
                   bounds=bounds, method='L-BFGS-B')
    qi,b,di = map(float, res.x)

    fit_hist = arps(qi,b,di,t_hist)

    # forward forecast in *days*, evaluate ARPS at months = day/30.4375
    f_d, f_v = [], []
    d = int(t_days.max()) + 1 if t_days.size else 0
    end_d = d + int(max_days)
    while d < end_d:
        q = float(arps(qi,b,di, np.array([d/30.4375], float))[0])
        if q < 1e-6:
            break
        f_d.append(d); f_v.append(q); d += 1

    eur_hist = float(np.nansum(y_hist))
    eur_fcst = float(np.nansum(f_v))
    return dict(qi=qi, b=b, di=di,
                t_hist=t_hist, hist=y_hist, fit_hist=fit_hist,
                f_days=np.array(f_d, int), f_vals=np.array(f_v, float),
                EUR_total=eur_hist+eur_fcst, EUR_fcst=eur_fcst)

def forecast_all_daily(merged_daily: pd.DataFrame, cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    com = cfg.commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[com]
    models = _train_rf(merged_daily, col)

    oneline, monthly_rows = [], []
    monthly_hist = _aggregate_daily_to_monthly(merged_daily, com)

    for api10, wd in merged_daily.groupby('API10'):
        if wd[col].max()<=0 or wd[col].sum()<=0:
            continue
        fc = forecast_one_well_daily(wd, com, cfg.b_low, cfg.b_high, cfg.max_months*30, models)
        hdr = wd.iloc[0]; well = hdr.get('WellName','N/A')

        qi_day = fc['qi']/30.4 if np.isfinite(fc['qi']) else np.nan
        q12 = float(arps(fc['qi'], fc['b'], fc['di'], np.array([12.0]))[0]) if np.isfinite(fc['qi']) else np.nan
        decline_yr = (1.0 - (q12/fc['qi']))*100.0 if (np.isfinite(fc['qi']) and fc['qi']>0 and np.isfinite(q12)) else np.nan

        row = {
            'API10': str(api10), 'WellName': well,
            'State': hdr.get('State'), 'County': hdr.get('County'),
            'PrimaryFormation': hdr.get('PrimaryFormation'),
            'LateralLength': hdr.get('LateralLength'),
            'CompletionDate': hdr.get('CompletionDate'),
            'FirstProdDate': hdr.get('FirstProdDate'),
            'qi (per day)': np.round(qi_day,0) if np.isfinite(qi_day) else np.nan,
            'b': np.round(fc['b'],3) if np.isfinite(fc['b']) else np.nan,
            'di (per month)': np.round(fc['di'],4) if np.isfinite(fc['di']) else np.nan,
            'First-Year Decline (%)': np.round(decline_yr,1) if np.isfinite(decline_yr) else np.nan,
        }
        if com=='oil':
            row.update({'EUR (Mbbl)': np.round(fc['EUR_total']/1_000.0,2) if np.isfinite(fc['EUR_total']) else np.nan,
                        'Remaining (Mbbl)': np.round(fc['EUR_fcst']/1_000.0,2) if np.isfinite(fc['EUR_fcst']) else np.nan})
        elif com=='gas':
            row.update({'EUR (MMcf)': np.round(fc['EUR_total']/1_000_000.0,2) if np.isfinite(fc['EUR_total']) else np.nan,
                        'Remaining (MMcf)': np.round(fc['EUR_fcst']/1_000_000.0,2) if np.isfinite(fc['EUR_fcst']) else np.nan})
        else:
            row.update({'EUR (Mbbl water)': np.round(fc['EUR_total']/1_000.0,2) if np.isfinite(fc['EUR_total']) else np.nan,
                        'Remaining (Mbbl water)': np.round(fc['EUR_fcst']/1_000.0,2) if np.isfinite(fc['EUR_fcst']) else np.nan})
        oneline.append(row)

        # forward daily → monthly aggregate
        date_col = None
        for cand in ['Day', 'Date', 'ReportDate', 'ProductionDate']:
            if cand in wd.columns:
                date_col = cand; break
        if date_col is None:
            continue
        last_day = pd.to_datetime(wd[date_col]).max()
        f_dates = pd.date_range(start=last_day + pd.offsets.Day(1), periods=len(fc['f_vals']), freq='D')
        df_f = pd.DataFrame({
            'API10': str(api10),
            'WellName': well,
            'Day': f_dates,
            col: fc['f_vals'],
            'Segment': 'Forecast'
        })
        df_f['MonthYear'] = df_f['Day'].dt.to_period('M')
        df_f_m = (df_f.groupby(['API10','WellName','MonthYear'], as_index=False)[col].sum())
        df_f_m['Date'] = df_f_m['MonthYear'].dt.to_timestamp(how='start')
        df_f_m[f'Monthly_{com}_volume'] = df_f_m[col]
        df_f_m['Segment'] = 'Forecast'
        monthly_rows.append(df_f_m[['API10','WellName','Date',f'Monthly_{com}_volume','Segment']])

    oneline_df = pd.DataFrame(oneline)

    if monthly_rows:
        monthly_df = pd.concat([monthly_hist] + monthly_rows, ignore_index=True)
    else:
        monthly_df = monthly_hist.copy()

    if monthly_df.empty:
        monthly_df = pd.DataFrame(
            columns=['API10','WellName','Date',f'Monthly_{com}_volume','Segment']
        )
    else:
        sort_cols = [c for c in ['API10','Date','Segment'] if c in monthly_df.columns]
        if sort_cols:
            monthly_df = monthly_df.sort_values(sort_cols)
    return oneline_df, monthly_df

# ---------------- HYBRID helpers (daily preferred per well) ----------------
def _train_rf_hybrid(merged_m: pd.DataFrame | None, merged_d: pd.DataFrame | None, target_col: str) -> Dict[str, RandomForestRegressor]:
    frames = []
    if merged_m is not None and not merged_m.empty:
        frames.append(merged_m[['API10','NormOil','NormGas','NormWater']].copy())
    if merged_d is not None and not merged_d.empty:
        frames.append(merged_d[['API10','NormOil','NormGas','NormWater']].copy())
    if not frames:
        return _train_rf(pd.DataFrame(columns=['API10','NormOil','NormGas','NormWater']), target_col)
    df_all = pd.concat(frames, ignore_index=True)
    return _train_rf(df_all, target_col)

def forecast_all_hybrid(merged_m: pd.DataFrame | None,
                        merged_d: pd.DataFrame | None,
                        cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each API10: prefer DAILY if present; otherwise use MONTHLY.
    Outputs monthly tables; daily forecasts are aggregated to months.
    """
    com = cfg.commodity.lower()
    col = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}[com]

    models = _train_rf_hybrid(merged_m, merged_d, col)

    # Historical monthly (daily aggregated + monthly native)
    hist_parts = []
    if merged_d is not None and not merged_d.empty:
        hist_parts.append(_aggregate_daily_to_monthly(merged_d, com))
    if merged_m is not None and not merged_m.empty:
        mm = merged_m.sort_values(['API10','MonthYear'])
        hist_m = pd.DataFrame({
            'API10': mm['API10'].astype(str),
            'WellName': mm['WellName'],
            'Date': mm['MonthYear'].dt.to_timestamp(how='start'),
            f'Monthly_{com}_volume': mm[col].astype(float),
            'Segment': 'Historical'
        })
        hist_parts.append(hist_m)

    monthly_hist = pd.concat(hist_parts, ignore_index=True) if hist_parts else pd.DataFrame(
        columns=['API10','WellName','Date', f'Monthly_{com}_volume','Segment']
    )

    # Prefer daily wells: drop monthly hist rows for those
    daily_apis = set()
    if merged_d is not None and not merged_d.empty:
        daily_apis = set(merged_d['API10'].astype(str).unique())
        if not monthly_hist.empty:
            mask = monthly_hist['API10'].astype(str).isin(daily_apis) & (monthly_hist['Segment'] == 'Historical')
            monthly_hist = monthly_hist[~mask]

    apis = set()
    if merged_m is not None and not merged_m.empty:
        apis |= set(merged_m['API10'].astype(str).unique())
    if merged_d is not None and not merged_d.empty:
        apis |= set(merged_d['API10'].astype(str).unique())

    oneline, monthly_rows = [], []

    for api in sorted(apis):
        use_daily = api in daily_apis
        if use_daily:
            wd = merged_d[merged_d['API10'].astype(str) == api]
            if wd.empty:
                continue
            if wd[col].max() <= 0 or wd[col].sum() <= 0:
                continue
            fc = forecast_one_well_daily(wd, com, cfg.b_low, cfg.b_high, cfg.max_months*30, models)
            hdr = wd.iloc[0]; well = hdr.get('WellName','N/A')

            qi_day = fc['qi']/30.4 if np.isfinite(fc['qi']) else np.nan
            q12 = float(arps(fc['qi'], fc['b'], fc['di'], np.array([12.0]))[0]) if np.isfinite(fc['qi']) else np.nan
            decline_yr = (1.0 - (q12/fc['qi']))*100.0 if (np.isfinite(fc['qi']) and fc['qi']>0 and np.isfinite(q12)) else np.nan

            row = {
                'API10': str(api), 'WellName': well,
                'State': hdr.get('State'), 'County': hdr.get('County'),
                'PrimaryFormation': hdr.get('PrimaryFormation'),
                'LateralLength': hdr.get('LateralLength'),
                'CompletionDate': hdr.get('CompletionDate'),
                'FirstProdDate': hdr.get('FirstProdDate'),
                'qi (per day)': np.round(qi_day,0) if np.isfinite(qi_day) else np.nan,
                'b': np.round(fc['b'],3) if np.isfinite(fc['b']) else np.nan,
                'di (per month)': np.round(fc['di'],4) if np.isfinite(fc['di']) else np.nan,
                'First-Year Decline (%)': np.round(decline_yr,1) if np.isfinite(decline_yr) else np.nan,
            }
            if com=='oil':
                row.update({'EUR (Mbbl)': np.round(fc['EUR_total']/1_000.0,2) if np.isfinite(fc['EUR_total']) else np.nan,
                            'Remaining (Mbbl)': np.round(fc['EUR_fcst']/1_000.0,2) if np.isfinite(fc['EUR_fcst']) else np.nan})
            elif com=='gas':
                row.update({'EUR (MMcf)': np.round(fc['EUR_total']/1_000_000.0,2) if np.isfinite(fc['EUR_total']) else np.nan,
                            'Remaining (MMcf)': np.round(fc['EUR_fcst']/1_000_000.0,2) if np.isfinite(fc['EUR_fcst']) else np.nan})
            else:
                row.update({'EUR (Mbbl water)': np.round(fc['EUR_total']/1_000.0,2) if np.isfinite(fc['EUR_total']) else np.nan,
                            'Remaining (Mbbl water)': np.round(fc['EUR_fcst']/1_000.0,2) if np.isfinite(fc['EUR_fcst']) else np.nan})
            oneline.append(row)

            # daily forecast → monthly aggregate
            date_col = None
            for cand in ['Day', 'Date', 'ReportDate', 'ProductionDate']:
                if cand in wd.columns:
                    date_col = cand; break
            if date_col is not None:
                last_day = pd.to_datetime(wd[date_col]).max()
                f_dates = pd.date_range(start=last_day + pd.offsets.Day(1), periods=len(fc['f_vals']), freq='D')
                df_f = pd.DataFrame({
                    'API10': str(api),
                    'WellName': well,
                    'Day': f_dates,
                    col: fc['f_vals'],
                    'Segment': 'Forecast'
                })
                df_f['MonthYear'] = df_f['Day'].dt.to_period('M')
                df_f_m = (df_f.groupby(['API10','WellName','MonthYear'], as_index=False)[col].sum())
                df_f_m['Date'] = df_f_m['MonthYear'].dt.to_timestamp(how='start')
                df_f_m[f'Monthly_{com}_volume'] = df_f_m[col]
                df_f_m['Segment'] = 'Forecast'
                monthly_rows.append(df_f_m[['API10','WellName','Date',f'Monthly_{com}_volume','Segment']])

        else:
            wd = merged_m[merged_m['API10'].astype(str) == api]
            if wd.empty or wd[col].max() <= 0 or wd[col].sum() <= 0:
                continue
            fc = forecast_one_well(wd, com, cfg.b_low, cfg.b_high, cfg.max_months, models)
            hdr = wd.iloc[0]; well = hdr.get('WellName','N/A')

            qi_day = fc['qi']/30.4
            q12 = float(arps(fc['qi'], fc['b'], fc['di'], np.array([12.0]))[0])
            decline_yr = (1.0 - (q12/fc['qi']))*100.0 if fc['qi']>0 else 0.0

            row = {
                'API10': str(api), 'WellName': well,
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
            start = hist_dates.iloc[-1] + pd.offsets.MonthBegin(1) if len(hist_dates)>0 \
                    else merged_m['MonthYear'].min().to_timestamp(how='start')
            f_dates = pd.date_range(start=start, periods=len(fc['f_vals']), freq='MS')
            df_fm = pd.DataFrame({
                'API10': str(api),
                'WellName': well,
                'Date': f_dates,
                f'Monthly_{com}_volume': fc['f_vals'],
                'Segment': 'Forecast'
            })
            monthly_rows.append(df_fm[['API10','WellName','Date',f'Monthly_{com}_volume','Segment']])

    oneline_df = pd.DataFrame(oneline)

    monthly_df = monthly_hist.copy()
    if monthly_rows:
        monthly_df = pd.concat([monthly_df] + monthly_rows, ignore_index=True)

    if monthly_df.empty:
        monthly_df = pd.DataFrame(
            columns=['API10','WellName','Date',f'Monthly_{com}_volume','Segment']
        )
    else:
        sort_cols = [c for c in ['API10','Date','Segment'] if c in monthly_df.columns]
        if sort_cols:
            monthly_df = monthly_df.sort_values(sort_cols)

    return oneline_df, monthly_df

# ---------------- EUR stats & Probit ----------------
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

# ---------------- Single-well plot (monthly axis) ----------------
def plot_one_well(wd: pd.DataFrame, fc: dict, commodity: str):
    com = commodity.lower()
    col_map = {'oil':'NormOil','gas':'NormGas','water':'NormWater'}
    color_map = {'oil':'green','gas':'red','water':'blue'}
    col = col_map[com]
    color = color_map[com]
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

    unit = {"oil":"(normalized bbl/month)",
            "gas":"(normalized Mcf/month)",
            "water":"(normalized bbl/month)"}[com]
    well = wd.iloc[0].get('WellName','N/A')
    api  = str(wd.iloc[0].get('API10',''))

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
    ax.set_ylim(bottom=1)
    ax.grid(True, linestyle="--", alpha=0.4, which='both')
    ax.legend()
    fig.tight_layout()
    return fig
