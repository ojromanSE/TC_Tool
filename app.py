# app.py — SE Oil & Gas Autoforecasting (by fluid)
import streamlit as st
import pandas as pd
from io import BytesIO

from core import (
    load_header, load_production, fill_lateral_by_geo,
    preprocess, PreprocessConfig, forecast_all, ForecastConfig,
    plot_one_well, forecast_one_well, _train_rf,
    compute_eur_stats, probit_plot, eur_summary_table
)

# ---- smoke line so we know entrypoint is running
st.set_page_config(page_title="SE Tool", layout="wide")
st.write("✅ app.py loaded (entrypoint OK)")

st.title("SE Oil & Gas Autoforecasting — Broken Down by Fluid")

# ================= Sidebar: global params =================
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

# ================= Upload & Prepare Data =================
st.header("Upload & Prepare Data")
c1, c2 = st.columns(2)
with c1:
    header_file = st.file_uploader("Header CSV", type=["csv"], key="header_csv")
with c2:
    prod_file   = st.file_uploader("Production CSV", type=["csv"], key="prod_csv")

if st.button("Load / QC / Merge"):
    if not header_file or not prod_file:
        st.error("Please upload both Header and Production CSV files.")
    else:
        try:
            # ---- read the header file ONCE and reuse bytes for two passes
            header_bytes = header_file.getvalue()
            header_df = load_header(BytesIO(header_bytes))   # mapped columns
            raw_hdr   = pd.read_csv(BytesIO(header_bytes))   # raw columns (for lat/lon)

            # bring raw lat/lon into the mapped header if present
            for col in [lat_col, lon_col]:
                if col in raw_hdr.columns and col not in header_df.columns:
                    header_df[col] = raw_hdr[col]

            # QC/impute lateral by geo (if lat/lon provided)
            header_qc = fill_lateral_by_geo(
                header_df, lat_col=lat_col, lon_col=lon_col,
                lateral_col='LateralLength', decimals=int(bin_decimals)
            )

            # production (read once)
            prod_df = load_production(prod_file)

            # stash for later steps
            st.session_state.header_qc = header_qc
            st.session_state.prod_df = prod_df

            # merge + normalization
            pp_cfg = PreprocessConfig(
                normalization_length=int(norm_len),
                use_normalization=bool(use_norm)
            )
            merged = preprocess(header_qc, prod_df, pp_cfg)
            if merged.empty:
                st.warning("No rows after preprocessing. Check your inputs.")
            else:
                st.session_state.merged = merged
                st.success(f"Merged {len(merged['API10'].unique())} wells and {len(merged)} monthly rows.")
        except Exception as e:
            st.exception(e)

if "merged" in st.session_state:
    with st.expander("Preview: Header (QC’d)"):
        st.dataframe(st.session_state.header_qc.head(20), use_container_width=True)
    with st.expander("Preview: Production"):
        st.dataframe(st.session_state.prod_df.head(20), use_container_width=True)
    with st.expander("Preview: Merged"):
        st.dataframe(st.session_state.merged.head(20), use_container_width=True)

st.markdown("---")

# ================= Helper: per-fluid UI block =================
def fluid_block(fluid_name: str, eur_col: str, norm_col_for_models: str):
    st.header(f"{fluid_name} — Run & Analyze")

    disabled = "merged" not in st.session_state
    if st.button(f"Run {fluid_name} Forecast", disabled=disabled):
        merged = st.session_state.merged
        cfg = ForecastConfig(
            commodity=fluid_name.lower(),
            b_low=float(b_low), b_high=float(b_high), max_months=600
        )
        oneline, monthly = forecast_all(merged, cfg)
        st.session_state[f"{fluid_name}_oneline"] = oneline
        st.session_state[f"{fluid_name}_monthly"] = monthly
        st.success(f"{fluid_name} forecast completed for {oneline.shape[0]} wells.")

    on_key = f"{fluid_name}_oneline"
    mo_key = f"{fluid_name}_monthly"

    if on_key in st.session_state:
        tab1, tab2, tab3, tab4 = st.tabs([
            f"Oneline ({fluid_name})",
            f"Monthly ({fluid_name})",
            f"B-Factor Table ({fluid_name})",
            f"Probit ({fluid_name})",
        ])

        # Oneline
        with tab1:
            oneline = st.session_state[on_key]
            st.dataframe(oneline, use_container_width=True)
            st.download_button(
                f"Download Oneline — {fluid_name} (CSV)",
                oneline.to_csv(index=False).encode("utf-8"),
                file_name=f"oneline_{fluid_name.lower()}.csv",
                mime="text/csv"
            )

        # Monthly
        with tab2:
            monthly = st.session_state[mo_key]
            st.dataframe(monthly, use_container_width=True)
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
                st.session_state[on_key].to_excel(w, sheet_name="Oneline", index=False)
                monthly.to_excel(w, sheet_name="Monthly", index=False)
            st.download_button(
                f"Download Excel — {fluid_name}",
                data=buf.getvalue(),
                file_name=f"forecast_{fluid_name.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # B-Factor Table + analytics
        with tab3:
            oneline = st.session_state[on_key]
            cols = ['API10','WellName','qi (per day)','b','di (per month)','First-Year Decline (%)']
            cols = [c for c in cols if c in oneline.columns]
            btable = oneline[cols].copy().sort_values('b', ascending=False)
            st.dataframe(btable, use_container_width=True)
            st.download_button(
                f"Download B-Factor Table — {fluid_name} (CSV)",
                btable.to_csv(index=False).encode("utf-8"),
                file_name=f"bfactors_{fluid_name.lower()}.csv",
                mime="text/csv"
            )

            # ---- analytics plots
            color_map = {'oil':'green','gas':'red','water':'blue'}
            color = color_map[fluid_name.lower()]
            if {'b','qi (per day)'}.issubset(oneline.columns):
                st.subheader("B-Factor Analytics")
                # 1) histogram of b
                fig1, ax1 = plt.subplots(figsize=(7,4))
                ax1.hist(oneline['b'].dropna().values, bins=25, color=color, alpha=0.85)
                ax1.set_xlabel("b"); ax1.set_ylabel("Count"); ax1.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig1)
                # 2) scatter of b vs qi (per day)
                fig2, ax2 = plt.subplots(figsize=(7,4))
                ax2.scatter(oneline['b'], oneline['qi (per day)'], s=22, color=color, alpha=0.85)
                ax2.set_xlabel("b"); ax2.set_ylabel("qi (per day)")
                ax2.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig2)

        # Probit (colored like forecast)
        with tab4:
            oneline = st.session_state[on_key]
            if eur_col not in oneline.columns:
                st.info("Run the forecast to populate EURs.")
            else:
                eurs = pd.to_numeric(oneline[eur_col], errors="coerce").astype(float).tolist()
                unit = "Mbbl" if "Mbbl" in eur_col else "MMcf" if "MMcf" in eur_col else "Mbbl"
                stats = compute_eur_stats(eurs)
                st.dataframe(
                    eur_summary_table(fluid_name, stats, unit, int(norm_len)),
                    use_container_width=True
                )
                color_map = {'oil':'green','gas':'red','water':'blue'}
                color = color_map[fluid_name.lower()]
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Probit", color=color)
                st.pyplot(fig)

        # -------- Per-well plot  ▶ pick by WellName (robust to missing WellName) --------
        st.subheader(f"{fluid_name} — Per-well Plot")
        merged = st.session_state.merged

        if 'API10' not in merged.columns:
            st.error("Merged data is missing API10 after preprocessing.")
        else:
            base = merged[['API10']].astype({'API10': str}).copy()
            if 'WellName' in merged.columns:
                base['WellName'] = merged['WellName'].astype(str)
            else:
                base['WellName'] = base['API10']

            opts = base.dropna().drop_duplicates()
            if opts.empty:
                st.info("No wells available.")
            else:
                def make_label(r):
                    wn = (r['WellName'] or "").strip()
                    ap = r['API10']
                    return wn if wn and wn != ap else f"API {ap}"

                opts['label'] = opts.apply(make_label, axis=1)
                label_to_api = dict(zip(opts['label'], opts['API10']))

                pick_label = st.selectbox(
                    f"Pick Well ({fluid_name})",
                    options=sorted(opts['label'].tolist()),
                    key=f"{fluid_name}_plot_pick"
                )
                pick_api = label_to_api[pick_label]

                wd = merged[merged['API10'].astype(str) == str(pick_api)]
                models = _train_rf(merged, norm_col_for_models)
                fc = forecast_one_well(wd, fluid_name.lower(), float(b_low), float(b_high), 600, models)
                fig = plot_one_well(wd, fc, fluid_name.lower())
                st.pyplot(fig, clear_figure=True)

# imports for the B-Factor analytics plots
import matplotlib.pyplot as plt

# ================= Per-fluid sections =================
fluid_block("Oil",   "EUR (Mbbl)",        "NormOil")
st.markdown("---")
fluid_block("Gas",   "EUR (MMcf)",        "NormGas")
st.markdown("---")
fluid_block("Water", "EUR (Mbbl water)",  "NormWater")
st.markdown("---")

# ================= Final: Type Wells summary =================
st.header("Type Wells — Summary (End)")

def _eurs_from_oneline(df: pd.DataFrame, col: str) -> list[float]:
    if df is None or df.empty or col not in df.columns:
        return []
    return pd.to_numeric(df[col], errors="coerce").astype(float).tolist()

tw_tabs = st.tabs(["Oil", "Gas", "Water"])

with tw_tabs[0]:
    on_key = "Oil_oneline"
    if on_key not in st.session_state:
        st.info("Run Oil first.")
    else:
        eurs = _eurs_from_oneline(st.session_state[on_key], "EUR (Mbbl)")
        stats_o = compute_eur_stats(eurs)
        st.dataframe(eur_summary_table("Oil", stats_o, "Mbbl", int(norm_len)), use_container_width=True)

with tw_tabs[1]:
    on_key = "Gas_oneline"
    if on_key not in st.session_state:
        st.info("Run Gas first.")
    else:
        eurs = _eurs_from_oneline(st.session_state[on_key], "EUR (MMcf)")
        stats_g = compute_eur_stats(eurs)
        st.dataframe(eur_summary_table("Gas", stats_g, "MMcf", int(norm_len)), use_container_width=True)

with tw_tabs[2]:
    on_key = "Water_oneline"
    if on_key not in st.session_state:
        st.info("Run Water first.")
    else:
        eurs = _eurs_from_oneline(st.session_state[on_key], "EUR (Mbbl water)")
        stats_w = compute_eur_stats(eurs)
        st.dataframe(eur_summary_table("Water", stats_w, "Mbbl", int(norm_len)), use_container_width=True)

st.caption("Per-fluid workflow: forecast → B-factors & probits → end with Type Wells.")
