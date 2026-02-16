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
    compute_eur_stats, probit_plot, eur_summary_table
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
    norm_len = st.number_input("Normalization Length (ft)", 1000, 30000, 10000, step=500)
    use_norm = st.checkbox("Apply Normalization", value=True)
    st.markdown("---")
    st.subheader("DCA Parameters")
    b_low  = st.number_input("b-factor Low", value=0.0, step=0.1, format="%.3f",
                              help="Lower bound for Arps b-factor (0 = exponential)")
    b_high = st.number_input("b-factor High", value=2.0, step=0.1, format="%.3f",
                              help="Upper bound for Arps b-factor (>1 for unconventionals)")
    dmin   = st.number_input("Dmin (terminal decline, monthly)", value=0.006, step=0.001,
                              format="%.4f",
                              help="Terminal decline rate per month for Modified Arps. "
                                   "0.004≈5%/yr, 0.006≈7%/yr, 0.008≈10%/yr effective.")
    st.caption("Models: Modified Arps, SEPD, Duong — best selected via AICc")
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

def build_type_curves_and_lines(monthly_df: pd.DataFrame, com: str):
    vol_col = f"Monthly_{com}_volume"
    if monthly_df.empty or vol_col not in monthly_df.columns:
        return pd.DataFrame(columns=['t','P10','P50','P90']), []
    
    # Use WellID for grouping
    hist = monthly_df[monthly_df['Segment'] == 'Historical'][['WellID','Date',vol_col]].dropna().copy()
    hist = hist.sort_values(['WellID','Date'])
    hist['t'] = hist.groupby('WellID').cumcount() + 1
    lines = []
    for _, g in hist.groupby('WellID'):
        t = g['t'].to_numpy()
        y = np.clip(g[vol_col].to_numpy(), 1e-6, None)
        lines.append((t, y))
        
    all_df = monthly_df[['WellID','Date',vol_col]].dropna().sort_values(['WellID','Date']).copy()
    all_df['t'] = all_df.groupby('WellID').cumcount() + 1
    q = all_df.groupby('t')[vol_col].quantile([0.90,0.50,0.10]).unstack(level=1)
    q.columns = ['P10','P50','P90']
    q = q.reset_index()
    return q, lines

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
            b_low=float(b_low), b_high=float(b_high), max_months=600,
            dmin=float(dmin)
        )
        oneline, monthly = forecast_all(merged, cfg)
        st.session_state[f"{fluid_name}_oneline"] = oneline
        st.session_state[f"{fluid_name}_monthly"] = monthly
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
                if fluid_name == "Gas" or "MMcf" in eur_col:
                    eurs = [x * 1000 if x is not None else None for x in eurs]
                unit = "Mbbl" if "Mbbl" in eur_col else "MMcf"
                stats = compute_eur_stats(eurs)
                st.dataframe(format_df_2dec(eur_summary_table(fluid_name, stats, unit, int(norm_len))))
                fig = probit_plot(eurs, unit, f"{fluid_name} Probit", _phase_color(fluid_name))
                st.pyplot(fig)

        st.subheader(f"{fluid_name} — Per-well Plot")
        merged = st.session_state.merged
        # Use WellID for selection
        opts = merged[['WellID']].drop_duplicates().sort_values('WellID')
        pick_id = st.selectbox(f"Pick Well ({fluid_name})", opts['WellID'], key=f"{fluid_name}_pick")
        
        wd = merged[merged['WellID'] == pick_id]
        models = _train_rf(merged, norm_col_for_models)
        fc = forecast_one_well(wd, fluid_name.lower(), float(b_low), float(b_high), 600, models,
                              dmin=float(dmin))
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
    eurs = st.session_state[on_key][eur_col].astype(float).tolist()
    if fluid=="Gas": eurs = [x*1000 for x in eurs if pd.notna(x)]
    stats = compute_eur_stats(eurs)
    st.dataframe(format_df_2dec(eur_summary_table(fluid, stats, "Mbbl" if fluid!="Gas" else "MMcf", int(norm_len))))
    
    if mo_key in st.session_state:
        curves, lines = build_type_curves_and_lines(st.session_state[mo_key], fluid.lower())
        st.pyplot(plot_type_curves(curves, lines, fluid.lower()))

with tw_tabs[0]: _render_tw("Oil", "Oil_oneline", "Oil_monthly", "EUR (Mbbl)")
with tw_tabs[1]: _render_tw("Gas", "Gas_oneline", "Gas_monthly", "EUR (MMcf)")
with tw_tabs[2]: _render_tw("Water", "Water_oneline", "Water_monthly", "EUR (Mbbl water)")

# ================= PDF EXPORT =================
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime

def generate_comprehensive_pdf():
    """Generate a comprehensive PDF report with all sections."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#2E7D32'))
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1976D2'))
    
    # Title page
    story.append(Paragraph("SE Oil & Gas Autoforecasting Report", title_style))
    story.append(Spacer(1, 12))
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
        
        oneline = st.session_state[oneline_key]
        
        # Well count
        well_count = len(oneline)
        story.append(Paragraph(f"Total Wells Analyzed: {well_count}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # EUR Statistics
        if eur_col in oneline.columns:
            eurs = pd.to_numeric(oneline[eur_col], errors="coerce").dropna().astype(float).tolist()
            if fluid_name == "Gas":
                eurs = [x * 1000 for x in eurs]
            
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
        
        # Type curves
        if monthly_key in st.session_state:
            curves, lines = build_type_curves_and_lines(st.session_state[monthly_key], fluid_name.lower())
            if not curves.empty:
                story.append(Paragraph("Type Curve", styles['Heading2']))
                story.append(Spacer(1, 6))
                fig = plot_type_curves(curves, lines, fluid_name.lower())
                tc_png = _save_fig(fig)
                story.append(Image(tc_png, width=5.5*inch, height=2.6*inch))
                story.append(Spacer(1, 10))
        
        # Probit plot
        if eur_col in oneline.columns:
            eurs = pd.to_numeric(oneline[eur_col], errors="coerce").dropna().astype(float).tolist()
            if fluid_name == "Gas":
                eurs = [x * 1000 for x in eurs]
            if eurs:
                unit = "Mbbl" if fluid_name != "Gas" else "MMcf"
                story.append(Paragraph("Probit Plot", styles['Heading2']))
                story.append(Spacer(1, 6))
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Distribution", _phase_color(fluid_name))
                probit_png = _save_fig(fig)
                story.append(Image(probit_png, width=5.5*inch, height=2.6*inch))
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
                if fluid == "Gas":
                    eurs = [x * 1000 for x in eurs]
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
        try:
            with st.spinner("Generating comprehensive PDF report..."):
                pdf_path = generate_comprehensive_pdf()
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=f.read(),
                    file_name=f"SE_Autoforecast_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            st.success("PDF report generated successfully!")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            st.exception(e)
