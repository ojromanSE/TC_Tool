# app.py — SE Oil & Gas Autoforecasting (hybrid daily + monthly)
# Fully updated Nov-2025

import os
import tempfile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import (
    load_header, load_production, fill_lateral_by_geo,
    preprocess, PreprocessConfig, forecast_all, ForecastConfig,
    forecast_all_daily,  # your new hybrid core helper
    plot_one_well, forecast_one_well, _train_rf,
    compute_eur_stats, probit_plot, eur_summary_table
)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SE Tool", layout="wide")

LOGO_PATH = os.path.join("static", "logo.png")

cols = st.columns([0.12, 0.88])
with cols[0]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
with cols[1]:
    st.success("All Systems Working")
    st.title("SE Oil & Gas Autoforecasting")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Global Parameters")
    norm_len = st.number_input("Normalization Length (ft)", 1000, 30000, 10000, step=500)
    use_norm = st.checkbox("Apply Normalization", value=True)
    st.markdown("---")
    b_low  = st.number_input("b-factor Low", value=0.8, step=0.1, format="%.3f")
    b_high = st.number_input("b-factor High", value=1.2, step=0.1, format="%.3f")
    st.markdown("---")
    lat_col = st.text_input("Latitude column (optional QC)", value="Latitude")
    lon_col = st.text_input("Longitude column (optional QC)", value="Longitude")
    bin_decimals = st.number_input("Geo Bin Decimals", 0, 4, 2)
    st.markdown("---")
    if st.button("Reset All"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# ---------- Helpers ----------
def _fmt2(v):
    try:
        f = float(v)
        if np.isfinite(f):
            return f"{f:.2f}"
        return ""
    except Exception:
        return "" if v is None else str(v)

def format_df_2dec(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_fmt2)
    return out

def _phase_color(fluid): return {'oil':'green','gas':'red','water':'blue'}[fluid.lower()]

def _save_fig(fig, dpi=220):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ======================================================
# ============ Upload & Prepare Data ===================
# ======================================================
st.header("Upload & Prepare Data")
c1, c2 = st.columns(2)
with c1:
    header_file = st.file_uploader("Header CSV", type=["csv"], key="header_csv")
with c2:
    prod_file = st.file_uploader("Monthly Production CSV", type=["csv"], key="prod_csv")

# Optional daily upload
daily_file = st.file_uploader(
    "Daily Production CSV (optional — used automatically if present)",
    type=["csv"], key="daily_csv"
)

if st.button("Load / QC / Merge"):
    if not header_file:
        st.error("Please upload at least the Header CSV.")
    else:
        try:
            header_bytes = header_file.getvalue()
            header_df = load_header(BytesIO(header_bytes))
            raw_hdr = pd.read_csv(BytesIO(header_bytes))
            for col in [lat_col, lon_col]:
                if col in raw_hdr.columns and col not in header_df.columns:
                    header_df[col] = raw_hdr[col]
            header_qc = fill_lateral_by_geo(
                header_df, lat_col=lat_col, lon_col=lon_col,
                lateral_col='LateralLength', decimals=int(bin_decimals)
            )

            pp_cfg = PreprocessConfig(
                normalization_length=int(norm_len),
                use_normalization=bool(use_norm)
            )

            merged_m = merged_d = None
            if prod_file:
                prod_df = load_production(prod_file)
                merged_m = preprocess(header_qc, prod_df, pp_cfg)
            if daily_file:
                daily_df = load_production(daily_file)
                merged_d = preprocess(header_qc, daily_df, pp_cfg)

            st.session_state.header_qc = header_qc
            st.session_state.merged_monthly = merged_m
            st.session_state.merged_daily = merged_d

            # ---------- Combine hybrid wells ----------
            def _build_picker(df):
                if df is None or df.empty:
                    return pd.DataFrame(columns=['API10','WellName','MonthYear','NormOil','NormGas','NormWater'])
                dd = df.copy()
                if 'MonthYear' not in dd.columns:
                    if 'Date' in dd.columns:
                        dd['MonthYear'] = pd.to_datetime(dd['Date']).dt.to_period('M')
                    elif 'ReportDate' in dd.columns:
                        dd['MonthYear'] = pd.to_datetime(dd['ReportDate']).dt.to_period('M')
                    else:
                        return pd.DataFrame(columns=['API10','WellName','MonthYear','NormOil','NormGas','NormWater'])
                for c in ['WellName','NormOil','NormGas','NormWater']:
                    if c not in dd.columns:
                        dd[c] = np.nan
                return dd.groupby(['API10','WellName','MonthYear'], as_index=False)[['NormOil','NormGas','NormWater']].sum()

            daily_apis = set()
            if merged_d is not None and not merged_d.empty:
                daily_apis = set(merged_d['API10'].astype(str).unique())
                daily_picker = _build_picker(merged_d)
            else:
                daily_picker = pd.DataFrame(columns=['API10','WellName','MonthYear','NormOil','NormGas','NormWater'])
            if merged_m is not None and not merged_m.empty:
                monthly_picker = merged_m[~merged_m['API10'].astype(str).isin(daily_apis)][
                    ['API10','WellName','MonthYear','NormOil','NormGas','NormWater']
                ].copy()
            else:
                monthly_picker = pd.DataFrame(columns=['API10','WellName','MonthYear','NormOil','NormGas','NormWater'])
            picker_df = pd.concat([daily_picker, monthly_picker], ignore_index=True)
            st.session_state.merged = picker_df
            st.session_state.daily_api_set = daily_apis

            wells = len(picker_df['API10'].astype(str).unique())
            if daily_apis:
                st.success(f"Hybrid ready. {len(daily_apis)} wells will use DAILY, others MONTHLY. Total wells: {wells}.")
            else:
                st.success(f"Using MONTHLY only. Total wells: {wells}.")

        except Exception as e:
            st.exception(e)

# ======================================================
# ============ Forecasting UI per fluid ================
# ======================================================

def fluid_block(fluid_name: str, eur_col: str, norm_col_for_models: str):
    st.header(f"{fluid_name} — Run & Analyze")
    disabled = "merged" not in st.session_state
    if st.button(f"Run {fluid_name} Forecast", disabled=disabled):
        cfg = ForecastConfig(
            commodity=fluid_name.lower(),
            b_low=float(b_low), b_high=float(b_high), max_months=600
        )
        merged_d = st.session_state.get('merged_daily')
        merged_m = st.session_state.get('merged_monthly')
        if merged_d is not None and not merged_d.empty:
            oneline, monthly = forecast_all_daily(merged_d, cfg)
        elif merged_m is not None and not merged_m.empty:
            oneline, monthly = forecast_all(merged_m, cfg)
        else:
            st.warning("No data available to forecast.")
            return
        st.session_state[f"{fluid_name}_oneline"] = oneline
        st.session_state[f"{fluid_name}_monthly"] = monthly
        st.success(f"{fluid_name} forecast completed for {oneline.shape[0]} wells.")

    on_key = f"{fluid_name}_oneline"
    mo_key = f"{fluid_name}_monthly"
    if on_key in st.session_state:
        tab1, tab2, tab3, tab4 = st.tabs([
            f"Oneline ({fluid_name})",
            f"Monthly ({fluid_name})",
            f"B-Factor ({fluid_name})",
            f"Probit ({fluid_name})"
        ])
        with tab1:
            st.dataframe(format_df_2dec(st.session_state[on_key]), use_container_width=True)
        with tab2:
            st.dataframe(format_df_2dec(st.session_state[mo_key]), use_container_width=True)
        with tab3:
            oneline = st.session_state[on_key]
            cols = ['API10','WellName','qi (per day)','b','di (per month)','First-Year Decline (%)']
            cols = [c for c in cols if c in oneline.columns]
            st.dataframe(format_df_2dec(oneline[cols]), use_container_width=True)
        with tab4:
            oneline = st.session_state[on_key]
            if eur_col not in oneline.columns:
                st.info("Run the forecast to populate EURs.")
            else:
                eurs = pd.to_numeric(oneline[eur_col], errors="coerce").astype(float).tolist()
                unit = "Mbbl" if "Mbbl" in eur_col else "MMcf"
                stats = compute_eur_stats(eurs)
                st.dataframe(format_df_2dec(eur_summary_table(fluid_name, stats, unit, int(norm_len))),
                             use_container_width=True)
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Probit", color=_phase_color(fluid_name))
                st.pyplot(fig)

        # ---------- Per-well Plot (robust) ----------
        st.subheader(f"{fluid_name} — Per-well Plot")
        merged = st.session_state.merged
        base = merged[['API10']].astype({'API10': str}).copy()
        base['WellName'] = merged.get('WellName', base['API10'])
        base = base.dropna().drop_duplicates()
        base['label'] = base.apply(
            lambda r: r['WellName'] if r['WellName'] and r['WellName'] != r['API10'] else f"API {r['API10']}", axis=1
        )
        label_to_api = dict(zip(base['label'], base['API10']))
        pick_label = st.selectbox(
            f"Pick Well ({fluid_name})", sorted(base['label'].tolist()),
            key=f"{fluid_name}_plot_pick"
        )
        pick_api = label_to_api[pick_label]
        merged_m = st.session_state.get('merged_monthly')
        merged_d = st.session_state.get('merged_daily')
        wd_m = pd.DataFrame()
        wd_d = pd.DataFrame()
        if merged_m is not None:
            wd_m = merged_m[merged_m['API10'].astype(str)==str(pick_api)].copy()
        if merged_d is not None:
            wd_d = merged_d[merged_d['API10'].astype(str)==str(pick_api)].copy()
        if not wd_d.empty:
            if 'MonthYear' not in wd_d.columns:
                date_col = None
                for c in ['Date','Day','ReportDate']:
                    if c in wd_d.columns: date_col=c; break
                if date_col:
                    wd_d['MonthYear'] = pd.to_datetime(wd_d[date_col]).dt.to_period('M')
            vol_col = f"Daily{fluid_name.capitalize()}"
            if vol_col not in wd_d.columns:
                vol_col = f"Norm{fluid_name.capitalize()}"
            wd_m = (wd_d.groupby(['API10','WellName','MonthYear'], as_index=False)[vol_col]
                    .sum().rename(columns={vol_col:f"Monthly_{fluid_name.lower()}_volume"}))
        wd_final = wd_m if not wd_m.empty else wd_d if not wd_d.empty else merged[
            merged['API10'].astype(str)==str(pick_api)]
        if wd_final.empty:
            st.warning("No production data found for this well.")
        else:
            models = _train_rf(merged, norm_col_for_models)
            fc = forecast_one_well(wd_final, fluid_name.lower(), float(b_low), float(b_high), 600, models)
            fig = plot_one_well(wd_final, fc, fluid_name.lower())
            st.pyplot(fig, clear_figure=True)

# ======================================================
# ============ Fluid Sections ==========================
# ======================================================
fluid_block("Oil",   "EUR (Mbbl)",       "NormOil")
st.markdown("---")
fluid_block("Gas",   "EUR (MMcf)",       "NormGas")
st.markdown("---")
fluid_block("Water", "EUR (Mbbl water)", "NormWater")

st.caption("Hybrid daily + monthly forecasting workflow — SE Tools ©2025")
