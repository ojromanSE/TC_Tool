# app.py — SE Oil & Gas Autoforecasting

import os
import pathlib
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import (
    load_header, load_production, fill_lateral_by_geo,
    preprocess, PreprocessConfig, forecast_all, ForecastConfig,
    plot_one_well, forecast_one_well, _train_rf,
    compute_eur_stats, probit_plot, eur_summary_table,
    modified_arps, D_LIM_DEFAULT
)

st.set_page_config(page_title="SE Tool", layout="wide")
LOGO_PATH = pathlib.Path(__file__).parent / "static" / "logo.png"

# ---------- Header ----------
cols = st.columns([0.12, 0.88])
with cols[0]:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_column_width=True)
with cols[1]:
    st.success("All Systems Working")
    st.title("SE Oil & Gas Autoforecasting")

# ================= Sidebar =================
with st.sidebar:
    st.header("Global Parameters")
    vertical_wells = st.checkbox(
        "Vertical Well Dataset",
        value=False,
        help="Disables lateral length normalization. No lateral length required — all wells are retained."
    )
    if vertical_wells:
        use_norm = False
        norm_len = 10_000   # unused placeholder
        st.info("Normalization disabled for vertical wells.")
    else:
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
        st.rerun()

def _fmt2(value) -> str:
    try:
        f = float(value)
        if np.isfinite(f): return f"{f:.2f}"
        return ""
    except: return "" if value is None else str(value)

def format_df_2dec(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_fmt2)
    return out

def _phase_color(fluid: str) -> str:
    return {'oil': 'green', 'gas': 'red', 'water': 'blue'}[fluid.lower()]

def _phase_color_rgb(fluid: str):
    """Return ReportLab color object for PDF generation"""
    from reportlab.lib import colors
    color_map = {
        'oil': colors.HexColor('#2E7D32'),    # Green
        'gas': colors.HexColor('#D32F2F'),    # Red
        'water': colors.HexColor('#1976D2')   # Blue
    }
    return color_map.get(fluid.lower(), colors.grey)

def _save_fig(fig, dpi=220):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ================= Upload =================
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
            header_bytes = header_file.getvalue()
            header_df = load_header(BytesIO(header_bytes))
            raw_hdr   = pd.read_csv(BytesIO(header_bytes))
            for col in [lat_col, lon_col]:
                if col in raw_hdr.columns and col not in header_df.columns:
                    header_df[col] = raw_hdr[col]
            header_qc = fill_lateral_by_geo(
                header_df, lat_col=lat_col, lon_col=lon_col,
                lateral_col='LateralLength', decimals=int(bin_decimals)
            )
            prod_df = load_production(prod_file)
            st.session_state.header_qc = header_qc
            st.session_state.prod_df   = prod_df
            pp_cfg = PreprocessConfig(
                normalization_length=int(norm_len),
                use_normalization=bool(use_norm)
            )
            merged = preprocess(header_qc, prod_df, pp_cfg)
            if merged.empty:
                st.warning("No rows after preprocessing. Check your inputs.")
            else:
                st.session_state.merged = merged
                st.success(f"Merged {len(merged['WellID'].unique())} wells and {len(merged)} monthly rows.")
        except Exception as e:
            st.exception(e)

if "merged" in st.session_state:
    with st.expander("Preview: Header (QC'd)"):
        st.dataframe(format_df_2dec(st.session_state.header_qc.head(20)), use_container_width=True)
    with st.expander("Preview: Production"):
        st.dataframe(format_df_2dec(st.session_state.prod_df.head(20)), use_container_width=True)
    with st.expander("Preview: Merged"):
        st.dataframe(format_df_2dec(st.session_state.merged.head(20)), use_container_width=True)

st.markdown("---")

# ---------- Analytics ----------
def bfactor_analytics_figures(oneline: pd.DataFrame, fluid: str, eur_col: str):
    color = _phase_color(fluid)
    df = oneline.copy()
    df = df[pd.to_numeric(df.get('b'), errors='coerce').notna()]
    if df.empty:
        return None, None, None, pd.DataFrame([{"message":"no b values"}])

    b = pd.to_numeric(df['b'], errors='coerce').dropna().astype(float)
    P10, P50, P90 = np.percentile(b, [10, 50, 90])
    stats_rows = [
        ["count", float(b.size)], ["mean", float(b.mean())], ["median", P50],
        ["P10", P10], ["P90", P90]
    ]
    stats_df = pd.DataFrame(stats_rows, columns=[f"{fluid} b-factor statistics", "value"])

    fig_h, axh = plt.subplots(figsize=(6.6, 3.4))
    axh.hist(b.values, bins=30, color=color, alpha=0.85, edgecolor='none')
    axh.axvline(P50, color='black', linestyle='--', label=f"Median={P50:.2f}")
    axh.legend()
    hist_png = _save_fig(fig_h)

    fig_bx, axbx = plt.subplots(figsize=(6.6, 1.1))
    axbx.boxplot(b.values, vert=False, medianprops=dict(color='black'))
    axbx.set_yticks([])
    box_png = _save_fig(fig_bx)

    scatter_png = None
    if eur_col in df.columns:
        eur = pd.to_numeric(df[eur_col], errors='coerce')
        mask = eur > 0
        if mask.any():
            x = np.log10(eur[mask].astype(float).values)
            y = b[mask].astype(float).values
            fig_s, axs = plt.subplots(figsize=(6.6, 3.2))
            axs.scatter(10**x, y, s=16, color=color, alpha=0.85)
            axs.set_xscale('log')
            scatter_png = _save_fig(fig_s)
    return hist_png, box_png, scatter_png, stats_df

def _fit_arps_to_trace(t: np.ndarray, y: np.ndarray, d_lim: float = D_LIM_DEFAULT):
    """Fit a Modified Arps curve to an empirical trace; returns (qi, b, di)."""
    from scipy.optimize import minimize
    y = np.clip(y, 1e-6, None)
    qi0, b0, di0 = float(y[0]), 1.0, 0.10
    def loss(p):
        qi, b, di = p
        pred = modified_arps(qi, b, di, d_lim, t)
        return np.mean((np.log(pred + 1e-6) - np.log(y + 1e-6)) ** 2)
    bounds = [(y.max() * 0.1, y.max() * 3.0), (0.01, 2.0), (d_lim, 1.5)]
    res = minimize(loss, [qi0, b0, di0], method='L-BFGS-B', bounds=bounds)
    return tuple(res.x) if res.success else (qi0, b0, di0)


def build_type_curves_and_lines(monthly_df: pd.DataFrame, com: str, min_wells: int = 1,
                                oneline: pd.DataFrame = None):
    vol_col = f"Monthly_{com}_volume"
    if monthly_df.empty or vol_col not in monthly_df.columns:
        return pd.DataFrame(columns=['t','P10','P50','P90']), []

    # Historical lines for background well traces
    hist = monthly_df[monthly_df['Segment'] == 'Historical'][['WellID','Date',vol_col]].dropna().copy()
    hist = hist.sort_values(['WellID','Date'])
    hist['t'] = hist.groupby('WellID').cumcount() + 1
    lines = []
    for _, g in hist.groupby('WellID'):
        t = g['t'].to_numpy()
        y = np.clip(g[vol_col].to_numpy(), 1e-6, None)
        lines.append((t, y))

    # Build smooth type curves from per-well Arps parameters sorted by EUR.
    # Grouping by EUR percentile and taking median qi/b/di per group produces
    # pure Modified Arps curves with no drops or kinks.
    eur_col = {'oil': 'EUR (Mbbl)', 'gas': 'EUR (MMcf)', 'water': 'EUR (Mbbl water)'}[com.lower()]
    if (oneline is not None and not oneline.empty
            and eur_col in oneline.columns
            and all(c in oneline.columns for c in ['qi (per day)', 'b', 'di (per month)'])):
        df = oneline.copy()
        df['_eur'] = pd.to_numeric(df[eur_col], errors='coerce')
        df['_qi']  = pd.to_numeric(df['qi (per day)'], errors='coerce') * 30.4  # monthly
        df['_b']   = pd.to_numeric(df['b'], errors='coerce')
        df['_di']  = pd.to_numeric(df['di (per month)'], errors='coerce')
        df = df.dropna(subset=['_eur', '_qi', '_b', '_di'])
        df = df.sort_values('_eur', ascending=False).reset_index(drop=True)
        n = len(df)
        if n >= min_wells:
            t_out = np.arange(1, 601, dtype=float)
            smooth = {'t': t_out}
            p50_lo = max(0, int(n * 0.45))
            p50_hi = max(p50_lo + 1, int(n * 0.55))
            groups = {
                'P10': df.iloc[:max(1, int(n * 0.1))],
                'P50': df.iloc[p50_lo:p50_hi],
                'P90': df.iloc[max(0, int(n * 0.9)):],
            }
            # Use P50's b and di as the shape for all percentiles;
            # scale qi by EUR ratio so curves are parallel and smooth.
            p50_grp  = groups['P50']
            b_shape  = float(p50_grp['_b'].median())
            di_shape = float(p50_grp['_di'].median())
            qi_p50   = float(p50_grp['_qi'].median())
            eur_p50  = float(p50_grp['_eur'].median()) or 1.0
            for pct, grp in groups.items():
                eur_ratio = float(grp['_eur'].median()) / eur_p50
                smooth[pct] = modified_arps(qi_p50 * eur_ratio, b_shape, di_shape,
                                            D_LIM_DEFAULT, t_out)
            return pd.DataFrame(smooth), lines

    # Fallback: empirical percentiles with Arps fitting (used when oneline unavailable)
    all_df = monthly_df[['WellID','Date',vol_col]].dropna().sort_values(['WellID','Date']).copy()
    all_df['t'] = all_df.groupby('WellID').cumcount() + 1
    counts = all_df.groupby('t')[vol_col].count()
    valid_t = counts[counts >= min_wells].index
    if valid_t.empty:
        return pd.DataFrame(columns=['t','P10','P50','P90']), lines

    emp = all_df[all_df['t'].isin(valid_t)].groupby('t')[vol_col].quantile([0.90, 0.50, 0.10]).unstack(level=1)
    emp.columns = ['P10', 'P50', 'P90']
    emp = emp.reset_index()
    t_emp = emp['t'].to_numpy(dtype=float)

    t_out = np.arange(1, int(t_emp.max()) + 1, dtype=float)
    smooth = {'t': t_out}
    for pct in ['P10', 'P50', 'P90']:
        qi, b, di = _fit_arps_to_trace(t_emp, emp[pct].to_numpy())
        smooth[pct] = modified_arps(qi, b, di, D_LIM_DEFAULT, t_out)

    return pd.DataFrame(smooth), lines

def plot_type_curves(curves, lines, fluid):
    color = _phase_color(fluid)
    fig, ax = plt.subplots(figsize=(9,5))
    if lines:
        for t, y in lines:
            ax.plot(t, y, color='gray', alpha=0.1, linewidth=0.5)
    if not curves.empty:
        ax.plot(curves['t'], curves['P50'], color=color, linewidth=2, label='P50')
        ax.plot(curves['t'], curves['P10'], color=color, linestyle='--', label='P10')
        ax.plot(curves['t'], curves['P90'], color=color, linestyle='--', label='P90')
    ax.set_yscale('log'); ax.set_ylim(bottom=1)
    ax.legend()
    return fig

# ================= Fluid Block =================
def fluid_block(fluid_name: str, eur_col: str, norm_col_for_models: str):
    st.header(f"{fluid_name} — Run & Analyze")
    disabled = "merged" not in st.session_state
    if st.button(f"Run {fluid_name} Forecast", disabled=disabled):
        merged = st.session_state.merged
        cfg = ForecastConfig(
            commodity=fluid_name.lower(),
            b_low=float(b_low), b_high=float(b_high), max_months=600
        )
        oneline, monthly, well_fcs = forecast_all(merged, cfg)
        st.session_state[f"{fluid_name}_oneline"] = oneline
        st.session_state[f"{fluid_name}_monthly"] = monthly
        st.session_state[f"{fluid_name}_well_fcs"] = well_fcs   # cache per-well forecasts
        # Reset analog selection so new forecast starts with all wells selected
        st.session_state.pop(f"{fluid_name}_tc_selection", None)
        st.success(f"{fluid_name} forecast completed.")

    on_key = f"{fluid_name}_oneline"
    mo_key = f"{fluid_name}_monthly"

    if on_key in st.session_state:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Oneline", "Monthly", "B-Factor", "Probit"
        ])
        with tab1: st.dataframe(format_df_2dec(st.session_state[on_key]), use_container_width=True)
        with tab2: st.dataframe(format_df_2dec(st.session_state[mo_key]), use_container_width=True)
        with tab3:
            oneline = st.session_state[on_key]
            hist_png, box_png, scatter_png, bstats = bfactor_analytics_figures(oneline, fluid_name, eur_col)
            c1, c2 = st.columns([2,1])
            with c1:
                if hist_png: st.image(hist_png)
                if scatter_png: st.image(scatter_png)
            with c2:
                if box_png: st.image(box_png)
                st.dataframe(format_df_2dec(bstats))
        with tab4:
            oneline = st.session_state[on_key]
            if eur_col in oneline.columns:
                eurs = pd.to_numeric(oneline[eur_col], errors="coerce").astype(float).tolist()
                unit = "Mbbl" if "Mbbl" in eur_col else "MMcf"
                stats = compute_eur_stats(eurs)
                st.dataframe(format_df_2dec(eur_summary_table(fluid_name, stats, unit, int(norm_len))))
                fig = probit_plot(eurs, unit, f"{fluid_name} Probit", _phase_color(fluid_name), norm_len=int(norm_len))
                st.pyplot(fig)

        st.subheader(f"{fluid_name} — Per-well Plot")
        merged = st.session_state.merged
        # Use WellID for selection
        opts = merged[['WellID']].drop_duplicates().sort_values('WellID')
        pick_id = st.selectbox(f"Pick Well ({fluid_name})", opts['WellID'], key=f"{fluid_name}_pick")
        
        wd = merged[merged['WellID'] == pick_id]
        fcs_key = f"{fluid_name}_well_fcs"
        if fcs_key in st.session_state and pick_id in st.session_state[fcs_key]:
            # Reuse cached forecast — no RF re-training or re-optimisation
            fc = st.session_state[fcs_key][pick_id]
        else:
            models = _train_rf(merged, norm_col_for_models)
            fc = forecast_one_well(wd, fluid_name.lower(), float(b_low), float(b_high), 600, models)
        fig = plot_one_well(wd, fc, fluid_name.lower())
        st.pyplot(fig)

fluid_block("Oil", "EUR (Mbbl)", "NormOil")
st.markdown("---")
fluid_block("Gas", "EUR (MMcf)", "NormGas")
st.markdown("---")
fluid_block("Water", "EUR (Mbbl water)", "NormWater")
st.markdown("---")

# ================= Type Wells =================
st.header("Type Wells — Summary")
tw_tabs = st.tabs(["Oil", "Gas", "Water"])

def _render_tw(fluid, on_key, mo_key, eur_col):
    if on_key not in st.session_state:
        st.info(f"Run {fluid} first.")
        return

    oneline_full = st.session_state[on_key]

    # --- Analog selection table ---
    sel_key = f"{fluid}_tc_selection"

    # Build display columns available in the oneline
    show_cols = [c for c in [
        'WellName', 'PrimaryFormation', 'LateralLength', eur_col
    ] if c in oneline_full.columns]

    # Initialise selection state: set of all WellIDs (all selected)
    all_ids = set(oneline_full['WellID'].astype(str).tolist())
    if sel_key not in st.session_state:
        st.session_state[sel_key] = all_ids

    # Build editable frame: "Select" checkbox + display columns
    saved_ids = st.session_state[sel_key]
    editor_df = oneline_full[show_cols].copy().reset_index(drop=True)
    editor_df.insert(0, "Select", oneline_full['WellID'].astype(str).isin(saved_ids).tolist())

    col_cfg = {"Select": st.column_config.CheckboxColumn("Select", default=True)}
    for c in show_cols:
        col_cfg[c] = st.column_config.TextColumn(c, disabled=True)

    # ── Two-column layout: table left, plot right ──
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.caption(f"**Analog Wells** — check/uncheck to include or exclude from the Type Curve ({len(oneline_full)} wells total)")
        edited = st.data_editor(
            editor_df,
            column_config=col_cfg,
            use_container_width=True,
            hide_index=True,
            key=f"{fluid}_analog_editor",
        )

    # Persist selection as a set of WellIDs (robust across reruns/index shifts)
    selected_mask = edited["Select"].values.astype(bool)
    oneline_filtered = oneline_full[selected_mask].reset_index(drop=True)
    st.session_state[sel_key] = set(oneline_filtered['WellID'].astype(str).tolist())
    n_sel = int(selected_mask.sum())
    n_tot = len(oneline_full)

    # Compute EUR stats upfront so P50 curve can be scaled to match
    eurs = []
    eur_stats = {}
    if n_sel > 0 and eur_col in oneline_filtered.columns:
        eurs = pd.to_numeric(oneline_filtered[eur_col], errors='coerce').dropna().tolist()
        eur_stats = compute_eur_stats(eurs)

    with right_col:
        if mo_key in st.session_state and n_sel > 0:
            sel_ids = set(oneline_filtered['WellID'].astype(str))
            monthly_filtered = st.session_state[mo_key][
                st.session_state[mo_key]['WellID'].astype(str).isin(sel_ids)
            ]
            curves, lines = build_type_curves_and_lines(
                monthly_filtered, fluid.lower(), oneline=oneline_filtered
            )
            # Scale P50 curve so its cumulative volume matches the statistical P50 EUR.
            # Curves are in Bbl or Mcf; EUR stats are in Mbbl or MMcf → *1000.
            if (not curves.empty and 'P50' in curves.columns
                    and eur_stats.get('P50') and np.isfinite(eur_stats['P50'])):
                p50_curve_sum = curves['P50'].sum()
                if p50_curve_sum > 0:
                    scale = (eur_stats['P50'] * 1000) / p50_curve_sum
                    curves = curves.copy()
                    curves['P50'] = curves['P50'] * scale
            st.session_state[f"{fluid}_tc_curves"] = curves
            st.pyplot(plot_type_curves(curves, lines, fluid.lower()))

    # ── EUR summary below ──
    st.caption(f"**{n_sel} / {n_tot} wells selected**")
    if n_sel > 0:
        st.dataframe(
            format_df_2dec(eur_summary_table(fluid, eur_stats, "Mbbl" if fluid != "Gas" else "MMcf", int(norm_len))),
            use_container_width=True,
        )
    else:
        st.warning("No wells selected — select at least one well to compute statistics.")

with tw_tabs[0]: _render_tw("Oil",   "Oil_oneline",   "Oil_monthly",   "EUR (Mbbl)")
with tw_tabs[1]: _render_tw("Gas",   "Gas_oneline",   "Gas_monthly",   "EUR (MMcf)")
with tw_tabs[2]: _render_tw("Water", "Water_oneline", "Water_monthly", "EUR (Mbbl water)")

# ── Multi-phase TC download ──
_tc_keys = ["Oil_tc_curves", "Gas_tc_curves", "Water_tc_curves"]
if any(k in st.session_state for k in _tc_keys):
    import io as _io
    _oil_c   = st.session_state.get("Oil_tc_curves",   pd.DataFrame())
    _gas_c   = st.session_state.get("Gas_tc_curves",   pd.DataFrame())
    _water_c = st.session_state.get("Water_tc_curves", pd.DataFrame())

    # Build combined Oil + Gas sheet (P50 columns adjacent, units converted)
    _og_parts = [pd.Series(range(1, 601), name="Month")]
    for _pct in ["P10", "P50", "P90"]:
        if not _oil_c.empty and _pct in _oil_c.columns:
            _og_parts.append((_oil_c[_pct] / 1000).round(4).rename(f"{_pct}_Oil_Mbbl_per_month"))
        if not _gas_c.empty and _pct in _gas_c.columns:
            _og_parts.append((_gas_c[_pct] / 1000).round(4).rename(f"{_pct}_Gas_MMcf_per_month"))
    _og_df = pd.concat(_og_parts, axis=1)

    # Water sheet
    _water_parts = [pd.Series(range(1, 601), name="Month")]
    for _pct in ["P10", "P50", "P90"]:
        if not _water_c.empty and _pct in _water_c.columns:
            _water_parts.append((_water_c[_pct] / 1000).round(4).rename(f"{_pct}_Water_Mbbl_per_month"))
    _water_df = pd.concat(_water_parts, axis=1)

    _xls_buf = _io.BytesIO()
    with pd.ExcelWriter(_xls_buf, engine="openpyxl") as _writer:
        _og_df.to_excel(_writer, sheet_name="Oil & Gas TC", index=False)
        if len(_water_parts) > 1:
            _water_df.to_excel(_writer, sheet_name="Water TC", index=False)
    st.download_button(
        label="⬇ Download All Phases TC Monthly Volumes (Excel)",
        data=_xls_buf.getvalue(),
        file_name="TC_Monthly_Volumes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="all_tc_download",
    )

# ================= PDF EXPORT =================
from datetime import datetime

def generate_comprehensive_pdf(tc_name: str = ""):
    """Generate a comprehensive PDF report with all sections."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#2E7D32'))
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1976D2'))
    tc_name_style = ParagraphStyle('TCName', parent=styles['Normal'], fontSize=16,
                                   textColor=colors.HexColor('#1976D2'), fontName='Helvetica-Bold')

    # Title page
    story.append(Paragraph("SE Oil & Gas Autoforecasting Report", title_style))
    story.append(Spacer(1, 12))
    if tc_name:
        name_data = [[Paragraph(f"Type Curve Name:&nbsp;&nbsp;<font color='#1976D2'>{tc_name}</font>", tc_name_style)]]
        name_box = Table(name_data, colWidths=[6.5*inch])
        name_box.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#EAF4FB')),
            ('TOPPADDING',    (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING',   (0, 0), (-1, -1), 14),
            ('LINEBEFORE',    (0, 0), (0, -1),  4, colors.HexColor('#1976D2')),
        ]))
        story.append(name_box)
        story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 24))
    
    # Parameters section
    story.append(Paragraph("Analysis Parameters", heading_style))
    story.append(Spacer(1, 12))
    params_data = [
        ["Parameter", "Value"],
        ["Normalization Length", f"{norm_len} ft"],
        ["Apply Normalization", "Yes" if use_norm else "No"],
        ["B-factor Range", f"{b_low:.3f} - {b_high:.3f}"],
        ["Geographic Bin Decimals", str(bin_decimals)]
    ]
    params_table = Table(params_data, colWidths=[3*inch, 2*inch])
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(params_table)
    story.append(PageBreak())
    
    # Function to add fluid section
    def add_fluid_section(fluid_name, oneline_key, monthly_key, eur_col):
        if oneline_key not in st.session_state:
            return

        story.append(Paragraph(f"{fluid_name} Analysis", heading_style))
        story.append(Spacer(1, 12))

        oneline_full = st.session_state[oneline_key]

        # Apply same well selection as shown in the TC section (stored as a set of WellIDs)
        sel_key = f"{fluid_name}_tc_selection"
        if sel_key in st.session_state and isinstance(st.session_state[sel_key], set):
            sel_ids = st.session_state[sel_key]
            oneline = oneline_full[oneline_full['WellID'].astype(str).isin(sel_ids)].reset_index(drop=True)
        else:
            oneline = oneline_full
            sel_ids = set(oneline_full['WellID'].astype(str).tolist())

        # Filter monthly to the same selected wells
        monthly_full = st.session_state.get(monthly_key)
        if monthly_full is not None:
            monthly_sel = monthly_full[monthly_full['WellID'].astype(str).isin(sel_ids)]
        else:
            monthly_sel = None

        # Well count
        well_count = len(oneline)
        story.append(Paragraph(f"Wells in Type Curve: {well_count} / {len(oneline_full)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # EUR Statistics
        if eur_col in oneline.columns:
            eurs = pd.to_numeric(oneline[eur_col], errors="coerce").dropna().astype(float).tolist()

            if eurs:
                stats = compute_eur_stats(eurs)
                unit = "Mbbl" if fluid_name != "Gas" else "MMcf"
                
                story.append(Paragraph(f"EUR Statistics ({unit})", styles['Heading2']))
                story.append(Spacer(1, 6))
                
                # Build EUR data table with available stats
                eur_data = [["Metric", "Value"]]
                
                # Handle different possible key formats
                count_key = next((k for k in stats.keys() if 'count' in k.lower()), None)
                if count_key:
                    eur_data.append(["Count", f"{stats[count_key]:.0f}"])
                
                mean_key = next((k for k in stats.keys() if 'mean' in k.lower()), None)
                if mean_key:
                    eur_data.append(["Mean", f"{stats[mean_key]:.2f}"])
                
                # Try P50 or median
                p50_key = next((k for k in stats.keys() if 'p50' in k.lower() or 'median' in k.lower()), None)
                if p50_key:
                    eur_data.append(["Median (P50)", f"{stats[p50_key]:.2f}"])
                
                p10_key = next((k for k in stats.keys() if 'p10' in k.lower()), None)
                if p10_key:
                    eur_data.append(["P10", f"{stats[p10_key]:.2f}"])
                
                p90_key = next((k for k in stats.keys() if 'p90' in k.lower()), None)
                if p90_key:
                    eur_data.append(["P90", f"{stats[p90_key]:.2f}"])
                
                min_key = next((k for k in stats.keys() if 'min' in k.lower()), None)
                if min_key:
                    eur_data.append(["Min", f"{stats[min_key]:.2f}"])
                
                max_key = next((k for k in stats.keys() if 'max' in k.lower()), None)
                if max_key:
                    eur_data.append(["Max", f"{stats[max_key]:.2f}"])
                
                eur_table = Table(eur_data, colWidths=[2.5*inch, 2*inch])
                eur_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), _phase_color_rgb(fluid_name)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(eur_table)
                story.append(Spacer(1, 18))
        
        # B-factor statistics
        if 'b' in oneline.columns:
            b_vals = pd.to_numeric(oneline['b'], errors='coerce').dropna()
            if not b_vals.empty:
                story.append(Paragraph("B-Factor Statistics", styles['Heading2']))
                story.append(Spacer(1, 6))
                
                b_data = [
                    ["Metric", "Value"],
                    ["Count", f"{len(b_vals):.0f}"],
                    ["Mean", f"{b_vals.mean():.3f}"],
                    ["Median", f"{b_vals.median():.3f}"],
                    ["P10", f"{np.percentile(b_vals, 10):.3f}"],
                    ["P90", f"{np.percentile(b_vals, 90):.3f}"]
                ]
                
                b_table = Table(b_data, colWidths=[2.5*inch, 2*inch])
                b_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(b_table)
                story.append(Spacer(1, 18))
        
       # Add plots
        hist_png, box_png, scatter_png, _ = bfactor_analytics_figures(oneline, fluid_name, eur_col)
        
        if hist_png:
            story.append(Paragraph("B-Factor Distribution", styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(Image(hist_png, width=5*inch, height=2.5*inch))
            story.append(Spacer(1, 12))
        
        # Create a new page for the three plots together
        story.append(PageBreak())
        story.append(Paragraph(f"{fluid_name} Analysis Charts", heading_style))
        story.append(Spacer(1, 12))
        
        # EUR vs B-Factor
        if scatter_png:
            story.append(Paragraph("EUR vs B-Factor", styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(Image(scatter_png, width=5.5*inch, height=2.6*inch))
            story.append(Spacer(1, 10))
        
        # Type curves (use filtered monthly + filtered oneline to match the UI)
        if monthly_sel is not None:
            curves, lines = build_type_curves_and_lines(
                monthly_sel, fluid_name.lower(), oneline=oneline
            )
            if not curves.empty:
                story.append(Paragraph("Type Curve", styles['Heading2']))
                story.append(Spacer(1, 6))
                fig = plot_type_curves(curves, lines, fluid_name.lower())
                tc_png = _save_fig(fig)
                story.append(Image(tc_png, width=5.5*inch, height=2.6*inch))
                story.append(Spacer(1, 8))

                # Arps parameters table for each type curve tier
                qi_col  = 'qi (per day)'
                di_col  = 'di (per month)'
                b_col   = 'b'
                dec_col = 'First-Year Decline (%)'
                if all(c in oneline.columns for c in [qi_col, b_col, di_col]):
                    qi  = pd.to_numeric(oneline[qi_col],  errors='coerce').dropna()
                    b_s = pd.to_numeric(oneline[b_col],   errors='coerce').dropna()
                    di  = pd.to_numeric(oneline[di_col],  errors='coerce').dropna()
                    dec = pd.to_numeric(oneline[dec_col], errors='coerce').dropna() if dec_col in oneline.columns else None

                    qi_unit = "Mcf/day" if fluid_name == "Gas" else "bbl/day"

                    tc_data = [["Parameter", "P90 Curve\n(Low)", "P50 Curve\n(Mid)", "P10 Curve\n(High)"]]
                    tc_data.append([
                        f"qi ({qi_unit})",
                        f"{np.percentile(qi, 10):,.0f}",
                        f"{np.percentile(qi, 50):,.0f}",
                        f"{np.percentile(qi, 90):,.0f}",
                    ])
                    tc_data.append([
                        "b-factor",
                        f"{np.percentile(b_s, 10):.3f}",
                        f"{np.percentile(b_s, 50):.3f}",
                        f"{np.percentile(b_s, 90):.3f}",
                    ])
                    tc_data.append([
                        "Di (per month)",
                        f"{np.percentile(di, 90):.4f}",
                        f"{np.percentile(di, 50):.4f}",
                        f"{np.percentile(di, 10):.4f}",
                    ])
                    if dec is not None and not dec.empty:
                        tc_data.append([
                            "1st-Year Decline (%)",
                            f"{np.percentile(dec, 90):.1f}%",
                            f"{np.percentile(dec, 50):.1f}%",
                            f"{np.percentile(dec, 10):.1f}%",
                        ])
                    tc_data.append(["Terminal Di (d_lim)", "0.00417/mo", "0.00417/mo", "0.00417/mo"])

                    tc_table = Table(tc_data, colWidths=[2.0*inch, 1.4*inch, 1.4*inch, 1.4*inch])
                    tc_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), _phase_color_rgb(fluid_name)),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                    ]))
                    story.append(tc_table)
                story.append(Spacer(1, 10))
        
        # EUR summary table + probit plot
        if eur_col in oneline.columns:
            eurs = pd.to_numeric(oneline[eur_col], errors="coerce").dropna().astype(float).tolist()
            if eurs:
                unit = "Mbbl" if fluid_name != "Gas" else "MMcf"

                # EUR summary table
                eur_stats = compute_eur_stats(eurs)
                tbl_df = eur_summary_table(fluid_name, eur_stats, unit, int(norm_len))
                story.append(Paragraph("EUR Statistics", styles['Heading2']))
                story.append(Spacer(1, 4))
                col0, col1, col2 = tbl_df.columns
                eur_tbl_data = [[col0, col1, col2]] + [
                    [str(r[col0]),
                     f"{r[col1]:.2f}" if isinstance(r[col1], float) and np.isfinite(r[col1]) else (str(r[col1]) if not isinstance(r[col1], float) else ""),
                     f"{r[col2]:.2f}" if isinstance(r[col2], float) and np.isfinite(r[col2]) else (str(r[col2]) if not isinstance(r[col2], float) else "")]
                    for _, r in tbl_df.iterrows()
                ]
                eur_tbl = Table(eur_tbl_data, colWidths=[2.2*inch, 1.4*inch, 1.4*inch])
                eur_tbl.setStyle(TableStyle([
                    ('BACKGROUND',    (0, 0), (-1, 0), _phase_color_rgb(fluid_name)),
                    ('TEXTCOLOR',     (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN',         (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE',      (0, 0), (-1, -1), 9),
                    ('TOPPADDING',    (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID',          (0, 0), (-1, -1), 0.5, colors.black),
                    ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                ]))
                story.append(eur_tbl)
                story.append(Spacer(1, 10))

                # Probit plot (EUR + EUR/ft side by side)
                story.append(Paragraph("Probit Plot", styles['Heading2']))
                story.append(Spacer(1, 6))
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Distribution",
                                  _phase_color(fluid_name), norm_len=int(norm_len))
                probit_png = _save_fig(fig)
                story.append(Image(probit_png, width=7.0*inch, height=3.0*inch))
                story.append(Spacer(1, 12))
        
        story.append(PageBreak())
    
    # Add each fluid section
    add_fluid_section("Oil", "Oil_oneline", "Oil_monthly", "EUR (Mbbl)")
    add_fluid_section("Gas", "Gas_oneline", "Gas_monthly", "EUR (MMcf)")
    add_fluid_section("Water", "Water_oneline", "Water_monthly", "EUR (Mbbl water)")
    
    # Summary section
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 12))
    
    summary_data = [["Fluid", "Wells", "Mean EUR", "P50 EUR", "P10 EUR"]]
    
    for fluid, key, eur_col in [("Oil", "Oil_oneline", "EUR (Mbbl)"), 
                                  ("Gas", "Gas_oneline", "EUR (MMcf)"), 
                                  ("Water", "Water_oneline", "EUR (Mbbl water)")]:
        if key in st.session_state:
            oneline = st.session_state[key]
            well_count = len(oneline)
            if eur_col in oneline.columns:
                eurs = pd.to_numeric(oneline[eur_col], errors="coerce").dropna().astype(float).tolist()
                if eurs:
                    stats = compute_eur_stats(eurs)
                    unit = "MMcf" if fluid == "Gas" else "Mbbl"
                    
                    # Get the actual key names from stats dict
                    mean_key = next((k for k in stats.keys() if 'mean' in k.lower()), None)
                    p50_key = next((k for k in stats.keys() if 'p50' in k.lower() or 'median' in k.lower()), None)
                    p10_key = next((k for k in stats.keys() if 'p10' in k.lower()), None)
                    
                    mean_val = f"{stats[mean_key]:.2f} {unit}" if mean_key else "N/A"
                    p50_val = f"{stats[p50_key]:.2f} {unit}" if p50_key else "N/A"
                    p10_val = f"{stats[p10_key]:.2f} {unit}" if p10_key else "N/A"
                    
                    summary_data.append([
                        fluid,
                        str(well_count),
                        mean_val,
                        p50_val,
                        p10_val
                    ])
    
    if len(summary_data) > 1:
        summary_table = Table(summary_data, colWidths=[1*inch, 1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
    
    # Build PDF
    doc.build(story)
    return tmp.name

st.markdown("---")
st.header("📄 Export Report")

if st.button("Generate PDF Report"):
    if "Oil_oneline" not in st.session_state and "Gas_oneline" not in st.session_state and "Water_oneline" not in st.session_state:
        st.error("Please run at least one forecast (Oil, Gas, or Water) before generating a PDF report.")
    else:
        st.session_state["_pdf_name_prompt"] = True

if st.session_state.get("_pdf_name_prompt"):
    tc_name = st.text_input(
        "Enter a name for this Type Curve report:",
        key="_pdf_tc_name",
        placeholder="e.g. Niobrara 2024 Q1",
    )
    if st.button("Confirm & Generate", key="_pdf_confirm"):
        try:
            with st.spinner("Generating comprehensive PDF report..."):
                pdf_path = generate_comprehensive_pdf(tc_name=tc_name.strip())
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in tc_name.strip())
            file_name = f"{safe_name}.pdf" if safe_name else f"SE_Autoforecast_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=f.read(),
                    file_name=file_name,
                    mime="application/pdf",
                    key="_pdf_dl",
                )
            st.success("PDF report generated successfully!")
            st.session_state["_pdf_name_prompt"] = False
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            st.exception(e)
