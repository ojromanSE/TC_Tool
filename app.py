# app.py — SE Oil & Gas Autoforecasting
# Tables: force EXACTLY 2 decimals everywhere (UI + PDF)
# PDF: 3 pages per fluid (B-factor/Probit/Type-curve) with header band + logo
#      B-factor page uses a 2-column layout (plots left, stats table right)

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

# ---------- PAGE CONFIG (must be first Streamlit call) ----------
st.set_page_config(page_title="SE Tool", layout="wide")

# ---------- Paths / constants ----------
# FIX: Use pathlib for robust cross-platform/container paths
LOGO_PATH = pathlib.Path(__file__).parent / "static" / "logo.png"

# ---------- Top header with logo (left) + status/title (right) ----------
cols = st.columns([0.12, 0.88])
with cols[0]:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_column_width=True)
with cols[1]:
    st.success("All Systems Working")
    st.title("SE Oil & Gas Autoforecasting")

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
        # FIX: experimental_rerun is deprecated
        st.rerun()

# ================= Formatting helpers (2 decimals everywhere) =================
def _fmt2(value) -> str:
    """Format any numeric-like value to 2 decimals; leave text as-is; blank for NaN."""
    try:
        f = float(value)
        if np.isfinite(f):
            return f"{f:.2f}"
        return ""
    except Exception:
        return "" if value is None else str(value)

def format_df_2dec(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with EVERY cell rendered to a string w/ exactly 2 decimals for numerics."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_fmt2)
    return out

def _phase_color(fluid: str) -> str:
    return {'oil': 'green', 'gas': 'red', 'water': 'blue'}[fluid.lower()]

def _save_fig(fig, dpi=220):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

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
            header_bytes = header_file.getvalue()
            header_df = load_header(BytesIO(header_bytes))   # mapped columns
            raw_hdr   = pd.read_csv(BytesIO(header_bytes))   # raw columns (for lat/lon)
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
                st.success(f"Merged {len(merged['API10'].unique())} wells and {len(merged)} monthly rows.")
        except Exception as e:
            st.exception(e)

if "merged" in st.session_state:
    with st.expander("Preview: Header (QC’d)"):
        st.dataframe(format_df_2dec(st.session_state.header_qc.head(20)), use_container_width=True)
    with st.expander("Preview: Production"):
        st.dataframe(format_df_2dec(st.session_state.prod_df.head(20)), use_container_width=True)
    with st.expander("Preview: Merged"):
        st.dataframe(format_df_2dec(st.session_state.merged.head(20)), use_container_width=True)

st.markdown("---")

# ---------- B-factor analytics (hist + box + scatter + stats) ----------
def bfactor_analytics_figures(oneline: pd.DataFrame, fluid: str, eur_col: str):
    color = _phase_color(fluid)
    df = oneline.copy()
    df = df[pd.to_numeric(df.get('b'), errors='coerce').notna()]
    if df.empty:
        return None, None, None, pd.DataFrame([{"message":"no b values"}])

    b = pd.to_numeric(df['b'], errors='coerce').dropna().astype(float)

    # stats
    P10 = float(np.percentile(b, 10))
    P25 = float(np.percentile(b, 25))
    P50 = float(np.percentile(b, 50))
    P75 = float(np.percentile(b, 75))
    P90 = float(np.percentile(b, 90))
    P95 = float(np.percentile(b, 95))
    IQR = P75 - P25
    lo, hi = P25 - 1.5*IQR, P75 + 1.5*IQR
    outliers = int(((b < lo) | (b > hi)).sum())
    stats_rows = [
        ["count",               float(b.size)],
        ["mean",                float(b.mean())],
        ["median",              float(b.median())],
        ["Standard deviation",  float(b.std(ddof=1)) if b.size>1 else 0.0],
        ["Min",                 float(b.min())],
        ["P10",                 P10],
        ["P25 (Q1)",            P25],
        ["P50 (median)",        P50],
        ["P75 (Q3)",            P75],
        ["P90",                 P90],
        ["P95",                 P95],
        ["IQR",                 float(IQR)],
        ["Outliers (±1.5*IQR)", float(outliers)],
    ]
    stats_df = pd.DataFrame(stats_rows, columns=[f"{fluid} b-factor statistics", "value"])

    # Histogram
    fig_h, axh = plt.subplots(figsize=(6.6, 3.4))
    axh.hist(b.values, bins=30, color=color, alpha=0.85, edgecolor='none')
    axh.set_xlabel("b-factor"); axh.set_ylabel("Number of wells")
    axh.grid(True, linestyle="--", alpha=0.35)
    mean_b = float(b.mean())
    axh.axvline(mean_b,  color=color, linewidth=2, label=f"Mean = {mean_b:.3f}")
    axh.axvline(P50,     color='black', linestyle='--', linewidth=1.8, label=f"Median = {P50:.3f}")
    axh.axvline(P10,     color='purple', linestyle=':', linewidth=1.8, label=f"P10 = {P10:.3f}")
    axh.axvline(P90,     color='purple', linestyle='--', linewidth=1.8, label=f"P90 = {P90:.3f}")
    axh.legend(loc="upper right", fontsize=8, frameon=False)
    hist_png = _save_fig(fig_h)

    # Boxplot
    fig_bx, axbx = plt.subplots(figsize=(6.6, 1.1))
    axbx.boxplot(b.values, vert=False, widths=0.5,
                 boxprops=dict(color='gray'),
                 whiskerprops=dict(color='gray'),
                 capprops=dict(color='gray'),
                 medianprops=dict(color='black'))
    axbx.set_yticks([]); axbx.set_xlabel("b-factor")
    axbx.grid(True, axis='x', linestyle='--', alpha=0.35)
    box_png = _save_fig(fig_bx)

    # Scatter b vs EUR
    scatter_png = None
    if eur_col in df.columns:
        eur = pd.to_numeric(df[eur_col], errors='coerce')
        mask = eur > 0
        if mask.any():
            x = np.log10(eur[mask].astype(float).values)
            y = b[mask].astype(float).values
            a, c = np.polyfit(x, y, 1)
            yhat = a*x + c
            ss_res = np.sum((y - yhat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan

            fig_s, axs = plt.subplots(figsize=(6.6, 3.2))
            axs.scatter(10**x, y, s=16, color=color, alpha=0.85)
            x_line = np.linspace(x.min(), x.max(), 200)
            axs.plot(10**x_line, a*x_line + c, linestyle='--', color='red', linewidth=1.8,
                     label=f"Trend: b = {a:.2f}·log₁₀(EUR) + {c:.2f}\nR² = {r2:.2f}")
            axs.set_xscale('log')
            axs.set_xlabel("EUR (normalized units)")
            axs.set_ylabel("b-factor")
            axs.grid(True, which='both', linestyle='--', alpha=0.35)
            axs.legend(loc='lower right', fontsize=8, frameon=False)
            scatter_png = _save_fig(fig_s)

    return hist_png, box_png, scatter_png, stats_df

# ---------- Type curve helpers ----------
def build_type_curves_and_lines(monthly_df: pd.DataFrame, com: str):
    vol_col = f"Monthly_{com}_volume"
    if monthly_df.empty or vol_col not in monthly_df.columns:
        return pd.DataFrame(columns=['t','P10','P50','P90']), []
    hist = monthly_df[monthly_df['Segment'] == 'Historical'][['API10','Date',vol_col]].dropna().copy()
    hist = hist.sort_values(['API10','Date'])
    hist['t'] = hist.groupby('API10').cumcount() + 1
    lines = []
    for _, g in hist.groupby('API10'):
        t = g['t'].to_numpy()
        y = np.clip(g[vol_col].to_numpy(), 1e-6, None)
        lines.append((t, y))
    all_df = monthly_df[['API10','Date',vol_col]].dropna().sort_values(['API10','Date']).copy()
    all_df['t'] = all_df.groupby('API10').cumcount() + 1
    q = all_df.groupby('t')[vol_col].quantile([0.90,0.50,0.10]).unstack(level=1)
    q.columns = ['P10','P50','P90']
    q = q.reset_index()
    return q, lines

def plot_type_curves(curves: pd.DataFrame, lines, fluid: str):
    color = _phase_color(fluid)
    fig, ax = plt.subplots(figsize=(9,5))
    if lines:
        for t, y in lines:
            ax.plot(t, y, color='gray', alpha=0.15, linewidth=0.8)
    if curves.empty:
        ax.text(0.5, 0.5, "No data for type curve", ha='center', va='center'); ax.axis('off')
        return fig
    eps = 1e-6
    P50 = np.clip(curves['P50'].to_numpy(dtype=float), eps, None)
    P10 = np.clip(curves['P10'].to_numpy(dtype=float), eps, None)
    P90 = np.clip(curves['P90'].to_numpy(dtype=float), eps, None)
    ax.plot(curves['t'], P50, color=color, linewidth=2, label='P50')
    ax.plot(curves['t'], P10, color=color, linewidth=1.5, linestyle='--', label='P10')
    ax.plot(curves['t'], P90, color=color, linewidth=1.5, linestyle='--', label='P90')
    ax.set_xlabel("Months since first production")
    ax.set_ylabel(f"Monthly {fluid.lower()} (normalized units)")
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    ax.legend()
    fig.tight_layout()
    return fig

# ================= Per-fluid UI block =================
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
            f"B-Factor ({fluid_name})",
            f"Probit ({fluid_name})",
        ])

        # Oneline
        with tab1:
            st.dataframe(format_df_2dec(st.session_state[on_key]), use_container_width=True)

        # Monthly
        with tab2:
            st.dataframe(format_df_2dec(st.session_state[mo_key]), use_container_width=True)

        # B-Factor analytics + table
        with tab3:
            oneline = st.session_state[on_key]
            cols = ['API10','WellName','qi (per day)','b','di (per month)','First-Year Decline (%)']
            cols = [c for c in cols if c in oneline.columns]
            btable = oneline[cols].copy().sort_values('b', ascending=False)
            st.subheader(f"{fluid_name} — B-Factor Table")
            st.dataframe(format_df_2dec(btable), use_container_width=True)

            hist_png, box_png, scatter_png, bstats = bfactor_analytics_figures(
                oneline, fluid=fluid_name, eur_col=eur_col
            )

            st.subheader(f"{fluid_name} — B-Factor Analytics")
            c1, c2 = st.columns([2,1], vertical_alignment="top")
            with c1:
                if hist_png: st.image(hist_png, caption="Distribution of forecasted b-factors")
                if scatter_png: st.image(scatter_png, caption="b-factor vs EUR (log scale)")
            with c2:
                if box_png: st.image(box_png, caption="b-factor boxplot")
                st.dataframe(format_df_2dec(bstats), use_container_width=True)

        # Probit
        with tab4:
            oneline = st.session_state[on_key]
            if eur_col not in oneline.columns:
                st.info("Run the forecast to populate EURs.")
            else:
                eurs = pd.to_numeric(oneline[eur_col], errors="coerce").astype(float).tolist()
                # For Gas, multiply by 1000 for display but keep unit as MMcf as requested
                if fluid_name == "Gas" or eur_col == "EUR (MMcf)":
                    eurs = [x * 1000 if x is not None else None for x in eurs]
                unit = "Mbbl" if "Mbbl" in eur_col else "MMcf" if "MMcf" in eur_col else "Mbbl"
                stats = compute_eur_stats(eurs)
                st.dataframe(
                    format_df_2dec(eur_summary_table(fluid_name, stats, unit, int(norm_len))),
                    use_container_width=True
                )
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Probit", color=_phase_color(fluid_name))
                st.pyplot(fig)

        # Per-well plot
        st.subheader(f"{fluid_name} — Per-well Plot")
        merged = st.session_state.merged
        base = merged[['API10']].astype({'API10': str}).copy()
        if 'WellName' in merged.columns:
            base['WellName'] = merged['WellName'].astype(str)
        else:
            base['WellName'] = base['API10']
        opts = base.dropna().drop_duplicates()

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
        # FIX: clear_figure=True removed
        st.pyplot(fig)

# ================= Per-fluid sections =================
fluid_block("Oil",   "EUR (Mbbl)",        "NormOil")
st.markdown("---")
fluid_block("Gas",   "EUR (MMcf)",        "NormGas")
st.markdown("---")
fluid_block("Water", "EUR (Mbbl water)",  "NormWater")
st.markdown("---")

# ================= Final: Type Wells summary + P10/P50/P90 plots =================
st.header("Type Wells — Summary (End)")

def _eurs_from_oneline(df: pd.DataFrame, col: str, fluid: str = None) -> list[float]:
    if df is None or df.empty or col not in df.columns:
        return []
    eur_list = pd.to_numeric(df[col], errors="coerce").astype(float).tolist()
    # For Gas, apply *1000 for display
    if fluid and fluid.lower() == "gas":
        eur_list = [x * 1000 if x is not None else None for x in eur_list]
    return eur_list

tw_tabs = st.tabs(["Oil", "Gas", "Water"])

with tw_tabs[0]:
    on_key, mo_key = "Oil_oneline", "Oil_monthly"
    if on_key not in st.session_state:
        st.info("Run Oil first.")
    else:
        eurs = _eurs_from_oneline(st.session_state[on_key], "EUR (Mbbl)")
        stats_o = compute_eur_stats(eurs)
        st.dataframe(format_df_2dec(eur_summary_table("Oil", stats_o, "Mbbl", int(norm_len))), use_container_width=True)
        curves, lines = (build_type_curves_and_lines(st.session_state[mo_key], "oil")
                         if mo_key in st.session_state else (pd.DataFrame(), []))
        st.subheader("Oil Type Curve (P10 / P50 / P90)")
        st.pyplot(plot_type_curves(curves, lines, "oil"))

with tw_tabs[1]:
    on_key, mo_key = "Gas_oneline", "Gas_monthly"
    if on_key not in st.session_state:
        st.info("Run Gas first.")
    else:
        eurs = _eurs_from_oneline(st.session_state[on_key], "EUR (MMcf)", "Gas")
        stats_g = compute_eur_stats(eurs)
        st.dataframe(format_df_2dec(eur_summary_table("Gas", stats_g, "MMcf", int(norm_len))), use_container_width=True)
        curves, lines = (build_type_curves_and_lines(st.session_state[mo_key], "gas")
                         if mo_key in st.session_state else (pd.DataFrame(), []))
        st.subheader("Gas Type Curve (P10 / P50 / P90)")
        st.pyplot(plot_type_curves(curves, lines, "gas"))

with tw_tabs[2]:
    on_key, mo_key = "Water_oneline", "Water_monthly"
    if on_key not in st.session_state:
        st.info("Run Water first.")
    else:
        eurs = _eurs_from_oneline(st.session_state[on_key], "EUR (Mbbl water)")
        stats_w = compute_eur_stats(eurs)
        st.dataframe(format_df_2dec(eur_summary_table("Water", stats_w, "Mbbl", int(norm_len))), use_container_width=True)
        curves, lines = (build_type_curves_and_lines(st.session_state[mo_key], "water")
                         if mo_key in st.session_state else (pd.DataFrame(), []))
        st.subheader("Water Type Curve (P10 / P50 / P90)")
        st.pyplot(plot_type_curves(curves, lines, "water"))

st.caption("Per-fluid workflow: enhanced B-factors & probits → Type Wells.")

# =========================== PDF REPORT ===========================
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.utils import ImageReader

def _df_to_table(df: pd.DataFrame, title: str, font_size: int = 7):
    """Convert a DataFrame to a reportlab Table (2-dec formatting, repeated header)."""
    styles = getSampleStyleSheet()
    if df is None or df.empty:
        return [Paragraph(f"<b>{title}</b> — (no data)", styles['Heading3']), Spacer(1,6)]

    df_str = format_df_2dec(df)

    data = [df_str.columns.tolist()] + df_str.values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1), font_size),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('GRID',(0,0),(-1,-1), 0.25, colors.grey),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ('TOPPADDING',(0,0),(-1,-1),3),
    ]))
    return [Paragraph(f"<b>{title}</b>", styles['Heading3']), tbl, Spacer(1,8)]

def _logo_on_page(canvas, doc):
    """Draw logo at top-right and page number bottom-right, leaving a header band."""
    try:
        if LOGO_PATH.exists():
            img = ImageReader(str(LOGO_PATH))
            w, h = 0.9*inch, 0.9*inch
            canvas.drawImage(img, doc.pagesize[0]-w-0.4*inch, doc.pagesize[1]-h-0.35*inch,
                             width=w, height=h, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    page_num = canvas.getPageNumber()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(doc.pagesize[0]-0.4*inch, 0.4*inch, f"Page {page_num}")

def _fluid_section(story, fluid: str, on_key: str, mo_key: str, eur_col: str):
    """
    Build 3 pages per fluid:
      1) B-factor plots (left) + stats table (right)
      2) Probit plot + probit table
      3) Type-curve table (EUR stats summary) + type-curve plot
    """
    styles = getSampleStyleSheet()

    if on_key not in st.session_state:
        story.append(Paragraph(f"<b>{fluid}</b>", styles['Heading2']))
        story.append(Paragraph("(No data – run forecast first.)", styles['Normal']))
        story.append(PageBreak())
        return

    oneline = st.session_state[on_key].copy()

    # -------------------- PAGE 1: B-factor (2-column layout) --------------------
    story.append(Paragraph(f"<b>{fluid} — B-Factor Analytics</b>", styles['Heading2']))

    hist_png, box_png, scatter_png, bstats = bfactor_analytics_figures(oneline, fluid, eur_col)

    left_stack = []
    if hist_png:
        left_stack.append(Image(hist_png, width=4.2*inch, height=2.5*inch))
        left_stack.append(Spacer(1, 4))
    if box_png:
        left_stack.append(Image(box_png,  width=4.2*inch, height=0.85*inch))
        left_stack.append(Spacer(1, 4))
    if scatter_png:
        left_stack.append(Image(scatter_png, width=4.2*inch, height=2.3*inch))

    right_stack = _df_to_table(bstats, f"{fluid} — b-factor statistics", font_size=7)

    two_col = Table(
        [[left_stack, right_stack]],
        colWidths=[4.4*inch, 3.0*inch]
    )
    two_col.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(two_col)
    story.append(PageBreak())

    # -------------------- PAGE 2: Probit ----------------------
    story.append(Paragraph(f"<b>{fluid} — Probit</b>", styles['Heading2']))
    if eur_col in oneline.columns:
        eurs = pd.to_numeric(oneline[eur_col], errors="coerce").astype(float).tolist()
        # For Gas, multiply by 1000 for display but keep unit as MMcf as requested
        if fluid.lower() == "gas" or eur_col == "EUR (MMcf)":
            eurs = [x * 1000 if x is not None else None for x in eurs]
        unit = "Mbbl" if "Mbbl" in eur_col else "MMcf" if "MMcf" in eur_col else "Mbbl"
        stats = compute_eur_stats(eurs)
        figp = probit_plot(eurs, unit, f"{fluid} EUR Probit", color=_phase_color(fluid))
        story.append(Image(_save_fig(figp), width=6.5*inch, height=4.2*inch))
        story += _df_to_table(
            eur_summary_table(fluid, stats, unit, int(st.session_state.get('norm_len_pdf', 10000))),
            f"{fluid} — Probit Table",
            font_size=7
        )
    else:
        story.append(Paragraph("No EUR column found for probit.", styles['Normal']))
    story.append(PageBreak())

    # -------------------- PAGE 3: Type Curve ------------------
    story.append(Paragraph(f"<b>{fluid} — Type Curve</b>", styles['Heading2']))
    if mo_key in st.session_state and on_key in st.session_state:
        oneline_df = st.session_state[on_key]
        eurs = pd.to_numeric(oneline_df[eur_col], errors="coerce").astype(float).tolist()
        # For Gas, multiply by 1000 for display but keep unit as MMcf as requested
        if fluid.lower() == "gas" or eur_col == "EUR (MMcf)":
            eurs = [x * 1000 if x is not None else None for x in eurs]
        unit = "Mbbl" if "Mbbl" in eur_col else "MMcf" if "MMcf" in eur_col else "Mbbl"
        stats = compute_eur_stats(eurs)
        df_tc = eur_summary_table(fluid, stats, unit, int(st.session_state.get('norm_len_pdf', 10000)))
        story += _df_to_table(df_tc, f"{fluid} — Type Curve Table", font_size=7)

        curves, lines = build_type_curves_and_lines(st.session_state[mo_key], fluid.lower())
        figtc = plot_type_curves(curves, lines, fluid.lower())
        story.append(Image(_save_fig(figtc), width=6.5*inch, height=4.0*inch))
    else:
        story.append(Paragraph("No data available for type-curve.", styles['Normal']))
    story.append(PageBreak())

# ============ PDF button ============
st.session_state['norm_len_pdf'] = int(norm_len)  # pass to report builder

if any(k in st.session_state for k in ["Oil_oneline","Gas_oneline","Water_oneline"]):
    if st.button("Generate PDF Report"):
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(
            tmp_pdf.name,
            pagesize=letter,
            leftMargin=0.5*inch, rightMargin=0.5*inch,
            topMargin=1.15*inch, bottomMargin=0.6*inch
        )
        story = []

        styles = getSampleStyleSheet()
        story.append(Paragraph("<b>SE Oil & Gas Autoforecasting — Report</b>", styles['Title']))
        story.append(Spacer(1, 8))

        _fluid_section(story, "Oil",   "Oil_oneline",   "Oil_monthly",   "EUR (Mbbl)")
        _fluid_section(story, "Gas",   "Gas_oneline",   "Gas_monthly",   "EUR (MMcf)")
        _fluid_section(story, "Water", "Water_oneline", "Water_monthly", "EUR (Mbbl water)")

        doc.build(story, onFirstPage=_logo_on_page, onLaterPages=_logo_on_page)

        with open(tmp_pdf.name, "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f.read(),
                file_name="SE_autoforecast_report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Run at least one forecast (Oil/Gas/Water) to enable the PDF report.")
