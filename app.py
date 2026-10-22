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
    plot_one_well, forecast_one_well,
    compute_eur_stats, probit_plot, eur_summary_table,
    modified_arps, modified_arps_cum, model_volumes, D_LIM_DEFAULT, DAYS_PER_MONTH,
    nominal_monthly_from_effective_annual, effective_annual_from_nominal_monthly,
    estimate_lateral_exponent, hindcast_one_well, hindcast_all,
)
try:
    import diagnostics as diag          # flow-regime diagnostics panel (optional module)
except ImportError:
    diag = None

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
        norm_method = "Linear  EUR ∝ L"
        auto_fit_exp = False
        norm_exponent = 1.0
        st.info("Normalization disabled for vertical wells.")
    else:
        norm_len = st.number_input("Normalization Length (ft)", 1000, 30000, 10000, step=500)
        use_norm = st.checkbox("Apply Normalization", value=True)
        norm_method = st.radio(
            "Scaling method (SPEE M5)",
            ["Power-law  EUR ∝ Lᵃ  (recommended)", "Linear  EUR ∝ L"],
            index=0,
            help=("SPEE Monograph 5 ('Scaling'): EUR per foot DECREASES with lateral "
                  "length, so production scales sub-linearly (exponent a < 1). Linear "
                  "scaling (a = 1) is the legacy behaviour and over-corrects long laterals."),
        )
        if norm_method.startswith("Power"):
            auto_fit_exp = st.checkbox("Auto-fit exponent from data", value=True)
            _est = st.session_state.get("lat_exponent_est")
            if auto_fit_exp:
                if _est is not None:
                    st.caption(f"Auto-fit exponent a ≈ {float(_est):.2f} (from log-cum vs log-lateral).")
                else:
                    st.caption("Exponent will be auto-fit from your data on Load / QC / Merge.")
                norm_exponent = float(_est) if _est is not None else 0.85
            else:
                norm_exponent = st.number_input(
                    "Lateral scaling exponent a", 0.30, 1.00,
                    value=float(_est) if _est is not None else 0.85, step=0.05,
                )
        else:
            auto_fit_exp = False
            norm_exponent = 1.0
        st.caption("Note: M5 also recommends scaling for kH and proppant intensity — not "
                   "applied here; bin on those instead (see Analogous Binning below).")
    st.markdown("---")
    # b-factor bounds and the model set depend on the dataset type: conventional
    # vertical wells decline exponentially-to-mildly-hyperbolic (b ≈ 0–0.5,
    # SPEE M5 boundary-dominated-flow guidance); unconventional horizontals
    # b ≈ 0.8–1.2.  Each mode remembers its own values via a mode-specific key.
    _bk = "v" if vertical_wells else "h"
    _b_lo_def, _b_hi_def = (0.0, 0.5) if vertical_wells else (0.8, 1.2)
    b_low  = st.number_input("b-factor Low",  min_value=0.0, value=_b_lo_def, step=0.1,
                             format="%.3f", key=f"b_low_{_bk}")
    b_high = st.number_input("b-factor High", min_value=0.0, value=_b_hi_def, step=0.1,
                             format="%.3f", key=f"b_high_{_bk}")
    allow_exp = st.checkbox(
        "Allow exponential decline (b = 0)", value=vertical_wells, key=f"allow_exp_{_bk}",
        help=("Adds an explicit Exponential model (q = qi·e^(−di·t), one free parameter) "
              "to the BIC model selection. It wins only when a well shows no hyperbolic "
              "curvature, so near-exponential wells get an exponential forecast. "
              "On by default for vertical datasets."),
    )
    if vertical_wells:
        st.caption("Vertical dataset: Duong (fracture-dominated transient flow, horizontal "
                   "wells only) is excluded; candidates are Exponential, Modified Arps, SEPD.")
    min_hist = st.number_input(
        "Min months history per well", 1, 60, 6, step=1,
        help=("SPEE M5 stresses sufficient history for individual forecasts. Wells with "
              "fewer producing months are excluded from the forecast."),
    )
    term_dec_pct = st.number_input(
        "Terminal decline (%/yr effective)", 1.0, 30.0, 5.0, step=0.5,
        help=("Floor on the nominal decline for EVERY model (Modified Arps, SEPD, Duong, "
              "Exponential): once a curve's decline falls to this rate it continues as an "
              "exponential at this rate. Quoted as an effective annual decline, as in reserves "
              "software; 5 %/yr effective = 0.00427/month nominal."),
    )
    d_lim_cfg = nominal_monthly_from_effective_annual(float(term_dec_pct) / 100.0)
    yield_mode = st.checkbox(
        "Forecast gas & water as yields (GOR / WOR) off the oil forecast", value=False,
        help=("Instead of independent decline fits, gas and water are forecast as a two-slope "
              "power-law yield applied to each well's oil forecast, keeping GOR/WOR trends "
              "physical. Requires the Oil forecast to be run first."),
    )
    with st.expander("Advanced fitting options"):
        qi_cap_in = st.number_input(
            "Peak-month volume cap (0 = none)", min_value=0.0, value=0.0, step=1000.0,
            help=("Optional cap on the pinned peak-month volume (normalized units). The old "
                  "hard cap of 152,000 understated large gas wells by 15-25 %; leave at 0."),
        )
        duong_min = st.number_input(
            "Min post-peak months for Duong", 6, 60, 18, step=1,
            help=("Duong's power-law tail is uncontrolled on short windows and produced 2x EUR "
                  "blow-ups on 9-12-month wells; it only competes with at least this many "
                  "post-peak months."),
        )
    qi_cap_cfg = float(qi_cap_in) if qi_cap_in and qi_cap_in > 0 else None
    st.markdown("---")
    lat_col = st.text_input("Latitude column (optional QC)", value="Latitude")
    lon_col = st.text_input("Longitude column (optional QC)", value="Longitude")
    bin_decimals = st.number_input("Geo Bin Decimals", 0, 4, 2)
    st.markdown("---")
    if st.button("Reset All"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

def _decline_models_label() -> str:
    """Human-readable list of the decline models in BIC selection."""
    models = (["Exponential"] if allow_exp else []) + ["Modified Arps", "SEPD"]
    if not vertical_wells:
        models.append("Duong")
    return ", ".join(models)

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


def _log_lower_fence(eur: pd.Series) -> float:
    """Lower outlier fence computed in log space.

    EUR populations are approximately log-normal (see probit plots), so the
    linear Q1 − 1.5×IQR fence is usually negative and never flags the
    under-performers.  Computing the fence in log space yields a meaningful
    positive lower bound (SPEE M5 — 'Look for Outliers and why')."""
    eur = pd.to_numeric(eur, errors='coerce')
    pos = eur[(eur > 0) & np.isfinite(eur)]
    if len(pos) < 3:
        return 0.0
    le = np.log(pos.astype(float))
    lq1, lq3 = le.quantile(0.25), le.quantile(0.75)
    return float(np.exp(lq1 - 1.5 * (lq3 - lq1)))


def _make_outlier_fig(oneline_df, eur_col, fluid_name):
    """EUR vs Lateral Length scatter with outlier detection.
    High fence: Q3 + 1.5×IQR (linear).  Low fence: Q1 − 1.5×IQR in LOG space
    (a linear lower fence is usually negative for log-normal EURs and never
    triggers).  Returns (fig, upper_fence, lower_fence, n_active, n_removed,
    n_high, n_low, median_eur, unit) or None."""
    df = oneline_df.copy()
    df['_eur'] = pd.to_numeric(df[eur_col], errors='coerce')
    if 'LateralLength' in df.columns:
        df['_lat'] = pd.to_numeric(df['LateralLength'], errors='coerce')
    else:
        return None
    df = df.dropna(subset=['_eur', '_lat'])
    if len(df) < 3:
        return None

    q1, q3 = df['_eur'].quantile(0.25), df['_eur'].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = _log_lower_fence(df['_eur'])

    high_mask = df['_eur'] > upper_fence
    low_mask  = df['_eur'] < lower_fence
    ok_mask   = ~high_mask & ~low_mask
    unit = "MMcf" if fluid_name.lower() == "gas" else "Mbbl"

    fig, ax = plt.subplots(figsize=(7, 4))
    if ok_mask.any():
        ax.scatter(df.loc[ok_mask, '_lat'], df.loc[ok_mask, '_eur'],
                   color='#2C2C2C', s=40, zorder=3, label='Within range')
    if high_mask.any():
        ax.scatter(df.loc[high_mask, '_lat'], df.loc[high_mask, '_eur'],
                   color='#781F28', s=40, zorder=3, label='High EUR outlier (>1.5× IQR)')
    if low_mask.any():
        ax.scatter(df.loc[low_mask, '_lat'], df.loc[low_mask, '_eur'],
                   color='#4A6FA5', s=40, zorder=3, label='Low EUR outlier (log-IQR fence)')
    ax.axhline(upper_fence, color='#781F28', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(0.98, 0.97, f"Upper fence  {upper_fence:,.0f}",
            transform=ax.transAxes, ha='right', va='top', color='#781F28', fontsize=8)
    if lower_fence > 0:
        ax.axhline(lower_fence, color='#4A6FA5', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(0.98, 0.03, f"Lower fence  {lower_fence:,.0f}",
                transform=ax.transAxes, ha='right', va='bottom', color='#4A6FA5', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#E8AABB')
        spine.set_linewidth(1.5)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    ax.set_xlabel('Lateral Length (ft)', fontsize=9)
    ax.set_ylabel(f'EUR ({unit})', fontsize=9)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(0.99, 0.88))
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    n_active  = int(ok_mask.sum())
    n_removed = int((high_mask | low_mask).sum())
    n_high    = int(high_mask.sum())
    n_low     = int(low_mask.sum())
    median_eur = float(df['_eur'].median())
    return fig, upper_fence, lower_fence, n_active, n_removed, n_high, n_low, median_eur, unit

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

            # Estimate the SPEE M5 sub-linear lateral-scaling exponent from data.
            est_exp = None
            if use_norm and not vertical_wells:
                est_exp = estimate_lateral_exponent(header_qc, prod_df)
                st.session_state["lat_exponent_est"] = est_exp

            # Resolve the exponent actually applied to this merge.
            if not use_norm or vertical_wells or not norm_method.startswith("Power"):
                applied_exp = 1.0
            elif auto_fit_exp:
                applied_exp = float(est_exp) if est_exp is not None else 0.85
            else:
                applied_exp = float(norm_exponent)
            st.session_state["applied_norm_exponent"] = applied_exp

            pp_cfg = PreprocessConfig(
                normalization_length=int(norm_len),
                use_normalization=bool(use_norm),
                normalization_exponent=float(applied_exp),
            )
            merged = preprocess(header_qc, prod_df, pp_cfg)
            if merged.empty:
                st.warning("No rows after preprocessing. Check your inputs.")
            else:
                # Keep the full merged set; binning derives the working subset.
                st.session_state.merged_full = merged
                st.session_state.merged = merged
                _exp_msg = (f" Scaling exponent a = {applied_exp:.2f}."
                            if use_norm and not vertical_wells else "")
                st.success(
                    f"Merged {len(merged['WellID'].unique())} wells and "
                    f"{len(merged)} monthly rows.{_exp_msg}"
                )
        except Exception as e:
            st.exception(e)

if "merged_full" in st.session_state:
    with st.expander("Preview: Header (QC'd)"):
        st.dataframe(format_df_2dec(st.session_state.header_qc.head(20)), use_container_width=True)
    with st.expander("Preview: Production"):
        st.dataframe(format_df_2dec(st.session_state.prod_df.head(20)), use_container_width=True)
    with st.expander("Preview: Merged"):
        st.dataframe(format_df_2dec(st.session_state.merged_full.head(20)), use_container_width=True)

st.markdown("---")

# ================= Analogous Binning (SPEE Monograph 5) =================
def _well_significance_note(n: int) -> None:
    """Render an M5 sample-size adequacy note for a well population."""
    if n >= 30:
        st.caption(f"✅ {n} wells — statistically meaningful sample (M5).")
    elif n >= 10:
        st.caption(f"⚠️ {n} wells — moderate sample; treat P10/P90 spread with caution (M5).")
    else:
        st.caption(f"🚩 {n} wells — small sample; type-well percentiles are not statistically robust (M5).")

if "merged_full" in st.session_state:
    st.header("Analogous Binning (SPEE Monograph 5)")
    st.caption(
        "M5 stresses grouping wells into an analogous population — same play/formation, "
        "vintage, and completion scale — while keeping the sample statistically meaningful. "
        "Filters below define the working set used for all forecasts."
    )
    _base = st.session_state.merged_full
    _bcols = st.columns(2)
    with _bcols[0]:
        _forms = sorted([f for f in _base['PrimaryFormation'].dropna().unique()]) \
            if 'PrimaryFormation' in _base.columns else []
        _sel_forms = st.multiselect("Formation", _forms, default=_forms) if _forms else []
        _counties = sorted([c for c in _base['County'].dropna().unique()]) \
            if 'County' in _base.columns else []
        _sel_counties = st.multiselect("County", _counties, default=_counties) if _counties else []
    with _bcols[1]:
        _lat = pd.to_numeric(_base.get('LateralLength'), errors='coerce')
        if _lat.notna().any():
            _lmin, _lmax = int(np.floor(_lat.min())), int(np.ceil(_lat.max()))
            if _lmin >= _lmax:
                _lat_range = (_lmin, _lmax)
                st.caption(f"Lateral length: {_lmin:,} ft (single value)")
            else:
                _lat_range = st.slider("Lateral length (ft)", _lmin, _lmax, (_lmin, _lmax), step=100)
        else:
            _lat_range = None
        _years = pd.to_datetime(_base.get('CompletionDate'), errors='coerce').dt.year
        if _years.notna().any():
            _ymin, _ymax = int(_years.min()), int(_years.max())
            if _ymin >= _ymax:
                _yr_range = (_ymin, _ymax)
                st.caption(f"Completion vintage: {_ymin} (single year)")
            else:
                _yr_range = st.slider("Completion vintage (year)", _ymin, _ymax, (_ymin, _ymax), step=1)
        else:
            _yr_range = None

    _mask = pd.Series(True, index=_base.index)
    if _forms and _sel_forms:
        _mask &= _base['PrimaryFormation'].isin(_sel_forms)
    if _counties and _sel_counties:
        _mask &= _base['County'].isin(_sel_counties)
    if _lat_range is not None:
        # Keep unknown lateral (NaN) wells rather than silently dropping them.
        _mask &= (_lat.between(_lat_range[0], _lat_range[1]) | _lat.isna())
    if _yr_range is not None:
        _mask &= (_years.between(_yr_range[0], _yr_range[1]) | _years.isna())

    _binned = _base[_mask].copy()
    st.session_state.merged = _binned
    _n_wells = _binned['WellID'].nunique()
    _n_total = _base['WellID'].nunique()
    st.markdown(f"**Working set: {_n_wells} / {_n_total} wells** ({len(_binned)} monthly rows).")
    _well_significance_note(_n_wells)
    st.markdown("---")

# ================= Editor-state helpers =================
# A checkbox editor's edits live in the browser and are re-sent for as long as
# the widget keeps the same identity; deleting its session_state entry only
# clears the server-side copy.  The robust way to reset an editor is therefore
# to give it a NEW key: a per-fluid generation counter is bumped whenever the
# applied selection changes and is embedded in the editor keys.  (Streamlit
# only forbids *assigning* a widget's state after it is drawn; bumping this
# plain counter is allowed at any point in the run.)
_EDITOR_GROUPS = ('analog', 'outlier')

def _editor_gen(fluid: str, group: str) -> int:
    return int(st.session_state.get(f"{fluid}_{group}_editor_gen", 0))

def _request_editor_reset(fluid: str, group: str | None = None) -> None:
    for g in (_EDITOR_GROUPS if group is None else [group]):
        st.session_state[f"{fluid}_{g}_editor_gen"] = _editor_gen(fluid, g) + 1

def _editor_key(fluid: str, group: str, name: str) -> str:
    return f"{fluid}_{name}_g{_editor_gen(fluid, group)}"

def _flash(key: str, msg: str) -> None:
    """Queue a success message that survives the st.rerun() that follows."""
    st.session_state[key] = msg

def _show_flash(key: str) -> None:
    msg = st.session_state.pop(key, None)
    if msg:
        st.success(msg)

# ================= Forecast configuration =================
def _make_cfg(fluid: str) -> ForecastConfig:
    """ForecastConfig from the current sidebar settings for one fluid."""
    fl = fluid.lower()
    return ForecastConfig(
        commodity=fl, b_low=float(b_low), b_high=float(b_high), max_months=600,
        d_lim=float(d_lim_cfg), min_months_history=int(min_hist),
        allow_exponential=bool(allow_exp), exclude_duong=bool(vertical_wells),
        duong_min_months=int(duong_min), qi_cap=qi_cap_cfg,
        yield_from_oil=bool(yield_mode) and fl in ('gas', 'water'),
    )


def _primary_fcs_for(fluid: str, oil_fcs=None):
    """Oil forecasts that drive yield-mode gas / water forecasts (None otherwise)."""
    if yield_mode and fluid.lower() in ('gas', 'water'):
        return oil_fcs if oil_fcs is not None else st.session_state.get("Oil_well_fcs")
    return None


# ================= Global Re-run (update all TCs) =================
def _rerun_all_phases(keep_selections: bool = True) -> list:
    """Re-merge with the CURRENT sidebar settings (normalization length,
    lateral scaling exponent, b-factor bounds, min history) and re-run every
    phase that has already been forecast, so all onelines and type curves are
    consistent with the current parameters.

    Analog/outlier selections are preserved (intersected with the new well
    set) so exclusion decisions survive a re-run. Returns the list of phases
    that were re-run."""
    if "header_qc" not in st.session_state or "prod_df" not in st.session_state:
        return []

    # Resolve the scaling exponent from current sidebar state (mirrors Load/QC/Merge)
    if not use_norm or vertical_wells or not norm_method.startswith("Power"):
        applied_exp = 1.0
    elif auto_fit_exp:
        _est = st.session_state.get("lat_exponent_est")
        applied_exp = float(_est) if _est is not None else 0.85
    else:
        applied_exp = float(norm_exponent)
    st.session_state["applied_norm_exponent"] = applied_exp

    pp_cfg = PreprocessConfig(
        normalization_length=int(norm_len),
        use_normalization=bool(use_norm),
        normalization_exponent=float(applied_exp),
    )
    merged_full = preprocess(st.session_state.header_qc, st.session_state.prod_df, pp_cfg)
    st.session_state.merged_full = merged_full

    # Preserve the current binning: the well set is invariant to normalization
    # settings, so filtering to the currently-binned WellIDs reproduces it.
    prev = st.session_state.get("merged")
    if prev is not None and not prev.empty:
        _keep = set(prev['WellID'].astype(str))
        merged = merged_full[merged_full['WellID'].astype(str).isin(_keep)].copy()
    else:
        merged = merged_full
    st.session_state.merged = merged
    if merged.empty:
        return []

    ran = []
    oil_fcs = None
    for fl in ["Oil", "Gas", "Water"]:            # Oil first: yield mode needs its forecasts
        if f"{fl}_oneline" not in st.session_state:
            continue
        cfg = _make_cfg(fl)
        oneline, monthly, well_fcs = forecast_all(merged, cfg, primary_fcs=_primary_fcs_for(fl, oil_fcs))
        if fl == "Oil":
            oil_fcs = well_fcs
        st.session_state[f"{fl}_oneline"]  = oneline
        st.session_state[f"{fl}_monthly"]  = monthly
        st.session_state[f"{fl}_well_fcs"] = well_fcs
        sel_key = f"{fl}_tc_selection"
        new_ids = set(oneline['WellID'].astype(str))
        if keep_selections and isinstance(st.session_state.get(sel_key), set):
            kept = st.session_state[sel_key] & new_ids
            st.session_state[sel_key] = kept if kept else new_ids
        else:
            st.session_state.pop(sel_key, None)
        # Reset the checkbox editors so stale edits don't misapply if the
        # well list changed (e.g. min-history raised).
        _request_editor_reset(fl)
        ran.append(fl)
    return ran


def _global_rerun_button(key: str):
    """Render the 'update all TCs' button; on click, re-run and refresh."""
    if st.button("🔄 Re-run All Phases & Update TCs", key=key, type="primary",
                 help="Re-applies the current b-factor bounds, normalization length and "
                      "lateral scaling, re-runs every phase already forecast, and refreshes "
                      "all type curves. Analog/outlier exclusions are kept."):
        if not any(f"{fl}_oneline" in st.session_state for fl in ("Oil", "Gas", "Water")):
            st.warning("No phase has been forecast yet — run at least one phase forecast first.")
            return
        try:
            with st.spinner("Re-running all phases with current settings..."):
                ran = _rerun_all_phases()
        except Exception as e:
            st.exception(e)
            return
        if not ran:
            st.warning("Nothing to re-run — check that the binning selection is not empty.")
            return
        st.session_state["_global_rerun_msg"] = (
            f"Re-ran {', '.join(ran)} with current settings — all type curves updated "
            f"(a = {st.session_state.get('applied_norm_exponent', 1.0):.2f}, "
            f"b ∈ [{float(b_low):.3f}, {float(b_high):.3f}], "
            f"min history = {int(min_hist)} mo)."
        )
        st.rerun()


if "merged_full" in st.session_state:
    _grm = st.session_state.pop("_global_rerun_msg", None)
    if _grm:
        st.success(_grm)
        try:
            st.toast(_grm)
        except Exception:
            pass
    _global_rerun_button("rerun_all_top")
    st.caption("Changed b-factors, normalization length, or lateral scaling? One click "
               "re-runs every forecast phase and brings all type curves up to date.")
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


def _tc_monthly_volumes(qi, b, di, d_lim, t_out):
    """Exact monthly volumes N(t) - N(t-1) of a Modified Arps curve for months t_out (1-based)."""
    t_out = np.asarray(t_out, dtype=float)
    return modified_arps_cum(qi, b, di, d_lim, t_out) - modified_arps_cum(qi, b, di, d_lim, t_out - 1.0)


def build_type_curves_and_lines(monthly_df: pd.DataFrame, com: str, min_wells: int = 1,
                                oneline: pd.DataFrame = None, d_lim: float = D_LIM_DEFAULT):
    vol_col = f"Monthly_{com}_volume"
    if monthly_df.empty or vol_col not in monthly_df.columns:
        return pd.DataFrame(columns=['t','P10','P50','P90']), []

    # Historical lines for background well traces
    hist = monthly_df[monthly_df['Segment'] == 'Historical'][['WellID','Date',vol_col]].dropna().copy()
    hist = hist.sort_values(['WellID','Date'])
    hist['t'] = hist.groupby('WellID').cumcount() + 1
    lines = []
    for _wid, g in hist.groupby('WellID'):
        t = g['t'].to_numpy()
        y = np.clip(g[vol_col].to_numpy(), 1e-6, None)
        lines.append((t, y, str(_wid)))

    # Build smooth type curves from per-well Arps parameters sorted by EUR.
    # Grouping by EUR percentile and taking median qi/b/di per group produces
    # pure Modified Arps curves with no drops or kinks.
    eur_col = {'oil': 'EUR (Mbbl)', 'gas': 'EUR (MMcf)', 'water': 'EUR (Mbbl water)'}[com.lower()]
    if (oneline is not None and not oneline.empty
            and eur_col in oneline.columns
            and all(c in oneline.columns for c in ['qi (per day)', 'b', 'di (per month)'])):
        df = oneline.copy()
        df['_eur'] = pd.to_numeric(df[eur_col], errors='coerce')
        df['_qi']  = pd.to_numeric(df['qi (per day)'], errors='coerce') * DAYS_PER_MONTH  # monthly rate
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
            # Build each percentile curve from ITS OWN group's median (b, di)
            # shape (per SPEE M5 — percentiles should carry their own decline
            # character, not a single shared shape). qi is a placeholder;
            # _render_tw rescales each curve to its EUR target downstream.
            for pct, grp in groups.items():
                qi   = float(grp['_qi'].median())
                b_g  = float(grp['_b'].median())
                di_g = float(grp['_di'].median())
                smooth[pct] = _tc_monthly_volumes(qi, b_g, di_g, d_lim, t_out)
            # Mean ("average well", M5) curve from the full-population median shape.
            smooth['Mean'] = _tc_monthly_volumes(
                float(df['_qi'].median()), float(df['_b'].median()),
                float(df['_di'].median()), d_lim, t_out)
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
        qi, b, di = _fit_arps_to_trace(t_emp, emp[pct].to_numpy(), d_lim=d_lim)
        smooth[pct] = _tc_monthly_volumes(qi, b, di, d_lim, t_out)
    # No per-well params available here — use the P50 shape as the Mean proxy.
    smooth['Mean'] = smooth['P50'].copy()

    return pd.DataFrame(smooth), lines

def plot_type_curves(curves, lines, fluid):
    """Static (matplotlib) type-curve chart — used for the PDF/PPTX reports."""
    color = _phase_color(fluid)
    fig, ax = plt.subplots(figsize=(9,5))
    if lines:
        for t, y, *_ in lines:
            ax.plot(t, y, color='gray', alpha=0.3, linewidth=0.5)
    if not curves.empty:
        ax.plot(curves['t'], curves['P50'], color=color, linewidth=2, label='P50')
        ax.plot(curves['t'], curves['P10'], color=color, linestyle='--', label='P10')
        ax.plot(curves['t'], curves['P90'], color=color, linestyle='--', label='P90')
        if 'Mean' in curves.columns:
            ax.plot(curves['t'], curves['Mean'], color='black', linewidth=1.5,
                    linestyle=':', label='Mean (avg well)')
    ax.set_yscale('log'); ax.set_ylim(bottom=1)
    ax.legend()
    return fig


try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:          # chart falls back to the static matplotlib version
    go = None
    _HAS_PLOTLY = False


def _selection_fingerprint(ids) -> str:
    """Short stable digest of a well-ID set. Chart widgets are keyed by it so
    their click-selection state resets whenever the plotted well set changes
    (a stale selection would otherwise map onto the wrong curves)."""
    import hashlib
    return hashlib.md5("|".join(sorted(map(str, ids))).encode()).hexdigest()[:10]


def plot_type_curves_interactive(curves, lines, fluid, name_map, highlight_ids=()):
    """Interactive (plotly) type-curve chart for the UI.

    Background well traces are drawn in the order of *lines* (never reordered,
    so trace indices are stable), carry the well name in the hover tooltip and
    the WellID as per-point customdata so chart clicks map back to wells.  They
    use invisible markers because plotly only selects points on traces that
    have markers — a lines-only trace can be hovered but never clicked.
    Highlighted wells get an orange overlay trace appended on top.  Returns
    (fig, trace_well_ids) where trace_well_ids[i] is the WellID of background
    trace i and None for every other trace."""
    color = _phase_color(fluid)
    unit  = {"oil": "bbl/mo", "gas": "Mcf/mo", "water": "bbl/mo"}[fluid]
    highlight_ids = set(map(str, highlight_ids))
    fig, trace_ids, ymax = go.Figure(), [], 10.0
    for t, y, wid in lines:
        wid = str(wid)
        label = name_map.get(wid, wid)
        if len(y):
            ymax = max(ymax, float(np.nanmax(y)))
        fig.add_trace(go.Scatter(
            x=t, y=y, mode='lines+markers', name=label, showlegend=False,
            line=dict(color='rgba(120,120,120,0.45)', width=1),
            marker=dict(size=6, opacity=0),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
            customdata=[wid] * len(t),
            hovertemplate=f"<b>{label}</b><br>Month %{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
        ))
        trace_ids.append(wid)
    if not curves.empty:
        ymax = max(ymax, float(np.nanmax(curves[['P10', 'P50', 'P90']].to_numpy())))
        for col, dash, width in (('P10', 'dash', 2), ('P50', 'solid', 3), ('P90', 'dash', 2)):
            fig.add_trace(go.Scatter(
                x=curves['t'], y=curves[col], mode='lines', name=col,
                line=dict(color=color, dash=dash, width=width),
                hovertemplate=f"<b>{col}</b><br>Month %{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
            ))
            trace_ids.append(None)
        if 'Mean' in curves.columns:
            fig.add_trace(go.Scatter(
                x=curves['t'], y=curves['Mean'], mode='lines', name='Mean (avg well)',
                line=dict(color='black', dash='dot', width=1.5),
                hovertemplate=f"<b>Mean</b><br>Month %{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
            ))
            trace_ids.append(None)
    # Orange overlays for highlighted wells, appended last so they sit on top
    # (the grey base trace underneath keeps providing the hover tooltip).
    for t, y, wid in lines:
        if str(wid) in highlight_ids:
            fig.add_trace(go.Scatter(
                x=t, y=y, mode='lines', name=name_map.get(str(wid), str(wid)),
                showlegend=False, hoverinfo='skip',
                line=dict(color='#FF8C00', width=3),
            ))
            trace_ids.append(None)
    fig.update_layout(
        height=480, margin=dict(l=50, r=20, t=30, b=45), hovermode='closest',
        clickmode='event+select',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        xaxis_title='Month of Production',
        yaxis_title=f"Monthly {fluid} (norm {unit})",
    )
    fig.update_yaxes(type='log', range=[0, np.log10(ymax) + 0.1])
    return fig, trace_ids


def _chart_clicked_wells(plot_key: str, trace_ids) -> set:
    """WellIDs of the background curves the analyst has clicked on the chart,
    read from the chart widget's stored selection state (available before the
    chart is drawn again, so the clicked curves can be styled)."""
    state = st.session_state.get(plot_key)
    points = []
    if state is not None:
        try:
            points = state["selection"]["points"]
        except Exception:
            try:
                points = state.selection.points
            except Exception:
                points = []
    found = set()
    for p in points or []:
        get = p.get if isinstance(p, dict) else (lambda k, d=None: getattr(p, k, d))
        wid = None
        cd = get("customdata")
        if cd is not None:
            wid = cd[0] if isinstance(cd, (list, tuple)) and cd else cd
        if wid is None:
            cn = get("curve_number")
            if cn is not None and 0 <= int(cn) < len(trace_ids):
                wid = trace_ids[int(cn)]
        if wid is not None:
            found.add(str(wid))
    return found

# ================= Fluid Block =================
def fluid_block(fluid_name: str, eur_col: str, norm_col_for_models: str):
    st.header(f"{fluid_name} — Run & Analyze")
    disabled = "merged" not in st.session_state
    if st.button(f"Run {fluid_name} Forecast", disabled=disabled):
        merged = st.session_state.merged
        if merged is None or merged.empty:
            st.warning("No wells in the current binning selection — widen the Analogous "
                       "Binning filters above before running a forecast.")
            st.stop()
        cfg = _make_cfg(fluid_name)
        if cfg.yield_from_oil and "Oil_well_fcs" not in st.session_state:
            st.warning(f"Yield mode is on but Oil has not been forecast yet — {fluid_name} "
                       "will be fitted independently. Run Oil first for GOR/WOR-based forecasts.")
        oneline, monthly, well_fcs = forecast_all(merged, cfg, primary_fcs=_primary_fcs_for(fluid_name))
        st.session_state[f"{fluid_name}_oneline"] = oneline
        st.session_state[f"{fluid_name}_monthly"] = monthly
        st.session_state[f"{fluid_name}_well_fcs"] = well_fcs   # cache per-well forecasts
        # Reset analog selection so new forecast starts with all wells selected
        st.session_state.pop(f"{fluid_name}_tc_selection", None)
        st.success(f"{fluid_name} forecast completed.")

    on_key = f"{fluid_name}_oneline"
    mo_key = f"{fluid_name}_monthly"

    if on_key in st.session_state:
        # --- M5 sample-size & maturity diagnostics ---
        _ol = st.session_state[on_key]
        _nw = len(_ol)
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Wells forecast", _nw)
        if 'Months History' in _ol.columns:
            _mh = pd.to_numeric(_ol['Months History'], errors='coerce')
            _immature = int((_mh < 12).sum())
            _c2.metric("Median months history", f"{_mh.median():.0f}")
            _c3.metric("Immature (<12 mo)", f"{_immature} ({_immature/_nw*100:.0f}%)" if _nw else "0")
        if _nw < 10:
            st.warning(f"🚩 Only {_nw} wells — below SPEE M5's threshold for a statistically "
                       "meaningful type well. Percentiles are indicative only.")
        elif _nw < 30:
            st.info(f"⚠️ {_nw} wells — moderate sample size; interpret P10/P90 spread with caution (M5).")
        if 'Months History' in _ol.columns and _nw and _immature / _nw > 0.5:
            st.warning(f"⚠️ {_immature/_nw*100:.0f}% of wells have <12 months of history — the EUR "
                       "distribution is forecast-dominated and may carry forecast bias (M5).")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Oneline", "Monthly", "B-Factor", "Probit", "Validation", "Diagnostics"
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

        with tab5:
            # ---- Population hindcast / history-match validation (SPEE M5 "Validation") ----
            st.markdown(
                f"**Hindcast validation — {fluid_name}.** Every well is re-forecast from a "
                "truncated history and compared with what it actually produced over the held-out "
                "months. Positive cumulative error = over-forecast. Use it to judge how well this "
                "model set and these settings predict *this* dataset before trusting the EURs."
            )
            _hc1, _hc2 = st.columns([1, 2])
            with _hc1:
                hc_mode = st.radio("Hold-out", ["Last N months", "Fraction of history"],
                                   key=f"{fluid_name}_hc_mode")
            with _hc2:
                if hc_mode == "Last N months":
                    hc_hold = int(st.number_input("Months held out", 3, 60, 12, step=1,
                                                  key=f"{fluid_name}_hc_n"))
                    hc_frac = 0.6
                else:
                    hc_hold = None
                    hc_frac = float(st.slider("Training fraction", 0.4, 0.9, 0.6, step=0.05,
                                              key=f"{fluid_name}_hc_frac"))
            if st.button(f"Run {fluid_name} hindcast on all wells", key=f"{fluid_name}_hc_run",
                         type="primary"):
                with st.spinner("Re-forecasting every well from its truncated history…"):
                    st.session_state[f"{fluid_name}_hindcast"] = hindcast_all(
                        st.session_state.merged, _make_cfg(fluid_name),
                        holdout_months=hc_hold, train_frac=hc_frac)
            hcdf = st.session_state.get(f"{fluid_name}_hindcast")
            if hcdf is None:
                st.caption("Choose a hold-out and run the hindcast.")
            elif hcdf.empty:
                st.info("No wells have enough history for this hold-out.")
            else:
                ce = pd.to_numeric(hcdf['Cum error (%)'], errors='coerce')
                mp = pd.to_numeric(hcdf['MAPE (%)'], errors='coerce')
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Wells hindcast", len(hcdf))
                k2.metric("Median cum error", f"{ce.median():+.1f}%",
                          help="Bias: median signed cumulative error over the held-out window.")
                k3.metric("P90 |cum error|", f"{ce.abs().quantile(0.9):.1f}%")
                k4.metric("Median MAPE", f"{mp.median():.1f}%")
                k5.metric("Within ±10% / ±25%",
                          f"{(ce.abs() <= 10).mean()*100:.0f}% / {(ce.abs() <= 25).mean()*100:.0f}%")
                by_model = (hcdf.groupby('Model')
                            .agg(wells=('WellID', 'count'),
                                 median_cum_err_pct=('Cum error (%)', 'median'),
                                 median_mape_pct=('MAPE (%)', 'median'))
                            .reset_index())
                st.dataframe(format_df_2dec(by_model), use_container_width=True, hide_index=True)
                if _HAS_PLOTLY:
                    figh = go.Figure()
                    figh.add_trace(go.Scatter(
                        x=hcdf['Actual (test)'], y=hcdf['Forecast (test)'], mode='markers',
                        text=hcdf['WellName'], customdata=hcdf['Model'], name='wells',
                        marker=dict(color=_phase_color(fluid_name), size=8, opacity=0.75),
                        hovertemplate="<b>%{text}</b><br>%{customdata}<br>actual %{x:,.0f}"
                                      "<br>forecast %{y:,.0f}<extra></extra>"))
                    lo_v = max(float(min(hcdf['Actual (test)'].min(), hcdf['Forecast (test)'].min())), 1.0)
                    hi_v = max(float(max(hcdf['Actual (test)'].max(), hcdf['Forecast (test)'].max())), lo_v * 10)
                    figh.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode='lines',
                                              line=dict(color='black', dash='dash'), name='1:1'))
                    figh.update_layout(height=420, xaxis_type='log', yaxis_type='log',
                                       xaxis_title='Actual held-out volume',
                                       yaxis_title='Forecast held-out volume',
                                       margin=dict(l=50, r=20, t=30, b=45))
                    st.plotly_chart(figh, use_container_width=True, key=f"{fluid_name}_hc_plot")
                st.dataframe(format_df_2dec(hcdf), use_container_width=True, hide_index=True)
                if yield_mode and fluid_name != "Oil":
                    st.caption("Note: the hindcast fits each well independently; yield-mode "
                               "(GOR/WOR) forecasts are not hindcast here.")

        with tab6:
            # ---- Flow-regime diagnostics (SPEE M5: identify the regime before trusting b) ----
            if diag is None:
                st.info("Flow-regime diagnostics module (diagnostics.py) is not available.")
            elif not _HAS_PLOTLY:
                st.info("Install `plotly` to view the diagnostics panel.")
            else:
                st.markdown(
                    "**Flow-regime diagnostics.** Rate–time on log-log axes with reference slopes, "
                    "and the nominal decline D(t) and Arps b(t) estimated from the data "
                    "(Bourdet-smoothed derivative) against the fitted model. A half slope with "
                    "b ≈ 2 is transient linear flow; b falling below ~1 with a flattening D is "
                    "boundary-dominated flow. The last few points use one-sided differences and "
                    "are unreliable — read the interior, not the end."
                )
                _ol = st.session_state[on_key]
                _dpick = st.selectbox(f"Well ({fluid_name} diagnostics)",
                                      _ol['WellID'].astype(str).tolist(), key=f"{fluid_name}_diag_pick")
                fcd = st.session_state.get(f"{fluid_name}_well_fcs", {}).get(_dpick)
                if fcd is None:
                    st.info("No cached forecast for this well — re-run the forecast.")
                else:
                    from plotly.subplots import make_subplots
                    pk = int(fcd['peak_idx'])
                    qpost = np.asarray(fcd['hist'], dtype=float)[pk:]
                    tpost = np.arange(len(qpost), dtype=float) + 0.5
                    dd = diag.data_diagnostics(tpost, qpost)
                    mdl = diag.model_diagnostics(fcd['model'], {'qi': fcd['qi'], **fcd['params']},
                                                 dd['t'], fcd['d_lim'])
                    color = _phase_color(fluid_name)
                    figd = make_subplots(rows=2, cols=2, subplot_titles=(
                        "Rate–time (log-log)", "Nominal decline D(t) [/month]",
                        "Arps b(t) = d(1/D)/dt", "Log-log slope d ln q / d ln t"))
                    figd.add_trace(go.Scatter(x=tpost, y=qpost, mode='markers', name='data',
                                              marker=dict(color=color)), 1, 1)
                    figd.add_trace(go.Scatter(x=tpost, y=np.asarray(fcd['fit_hist'])[pk:], mode='lines',
                                              name=fcd['model'], line=dict(color=color)), 1, 1)
                    t0, q0 = tpost[0], max(qpost[0], 1e-9)
                    for slope, lab, dash in ((-0.5, 'half slope (linear flow)', 'dot'),
                                             (-1.0, 'unit slope', 'dash')):
                        figd.add_trace(go.Scatter(x=tpost, y=q0 * (tpost / t0) ** slope, mode='lines',
                                                  name=lab, line=dict(color='gray', dash=dash, width=1)), 1, 1)
                    figd.add_trace(go.Scatter(x=dd['t'], y=dd['D'], mode='markers', name='D (data)',
                                              marker=dict(color=color)), 1, 2)
                    figd.add_trace(go.Scatter(x=dd['t'], y=mdl['D'], mode='lines', name='D (model)',
                                              line=dict(color=color)), 1, 2)
                    figd.add_hline(y=fcd['d_lim'], line=dict(color='black', dash='dot'), row=1, col=2)
                    figd.add_trace(go.Scatter(x=dd['t'], y=dd['b'], mode='markers', name='b (data)',
                                              marker=dict(color=color)), 2, 1)
                    figd.add_trace(go.Scatter(x=dd['t'], y=mdl['b'], mode='lines', name='b (model)',
                                              line=dict(color=color)), 2, 1)
                    for yv in (2.0, 1.0):
                        figd.add_hline(y=yv, line=dict(color='gray', dash='dot'), row=2, col=1)
                    figd.add_trace(go.Scatter(x=dd['t'], y=dd['loglog_slope'], mode='markers',
                                              name='log-log slope', marker=dict(color=color)), 2, 2)
                    for yv in (-0.5, -1.0):
                        figd.add_hline(y=yv, line=dict(color='gray', dash='dot'), row=2, col=2)
                    figd.update_xaxes(type='log', title_text='months since peak')
                    figd.update_yaxes(type='log', row=1, col=1)
                    figd.update_yaxes(type='log', row=1, col=2)
                    figd.update_yaxes(range=[-0.5, 3.0], row=2, col=1)
                    figd.update_yaxes(range=[-2.5, 0.5], row=2, col=2)
                    figd.update_layout(height=680, legend=dict(orientation='h', y=-0.08),
                                       margin=dict(l=40, r=20, t=40, b=40))
                    st.plotly_chart(figd, use_container_width=True, key=f"{fluid_name}_diag_plot")
                    _pstr = ", ".join(f"{k}={v:.3g}" for k, v in fcd['params'].items())
                    st.caption(f"Winning model: {fcd['model']} ({_pstr}); history-match fit "
                               f"R² (log) = {fcd.get('fit_r2_log', float('nan')):.3f}, "
                               f"MAPE = {fcd.get('fit_mape_pct', float('nan')):.1f} %.")

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
            fc = forecast_one_well(wd, fluid_name.lower(), float(b_low), float(b_high), 600,
                                   d_lim=float(d_lim_cfg), allow_exponential=bool(allow_exp),
                                   exclude_duong=bool(vertical_wells),
                                   duong_min_months=int(duong_min), qi_cap=qi_cap_cfg)
        fig = plot_one_well(wd, fc, fluid_name.lower())
        st.pyplot(fig)

        # --- Hindcast validation (SPEE M5 — Validation) ---
        if st.checkbox(f"Show hindcast validation ({fluid_name})", key=f"{fluid_name}_hc_chk",
                       help="Fit on the first 60% of this well's history and forecast the "
                            "held-out remainder to gauge forecast reliability (M5)."):
            hc = hindcast_one_well(wd, fluid_name.lower(), float(b_low), float(b_high),
                                   d_lim=float(d_lim_cfg),
                                   allow_exponential=bool(allow_exp),
                                   exclude_duong=bool(vertical_wells),
                                   duong_min_months=int(duong_min), qi_cap=qi_cap_cfg)
            if hc is None:
                st.info("Not enough history to hindcast this well (need ≥ ~9 months).")
            else:
                color = _phase_color(fluid_name)
                figh, axh = plt.subplots(figsize=(10, 5))
                axh.scatter(hc['t_train'], np.clip(hc['y_train'], 1e-6, None),
                            s=22, color=color, label="Train (actual)")
                axh.scatter(hc['t_test'], np.clip(hc['y_test'], 1e-6, None),
                            s=40, facecolors='none', edgecolors=color, label="Held-out (actual)")
                axh.plot(hc['t_test'], np.clip(hc['y_pred'], 1e-6, None),
                         linestyle='--', linewidth=2, color=color, label="Forecast")
                axh.set_yscale('log'); axh.set_ylim(bottom=1)
                axh.set_xlabel("Month of Production"); axh.set_ylabel("Monthly volume (norm)")
                axh.set_title(f"Hindcast — train on {hc['n_train']} mo, test on {hc['n_test']} mo "
                              f"[{hc['model']}]")
                axh.grid(True, linestyle="--", alpha=0.4, which='both'); axh.legend()
                figh.tight_layout()
                st.pyplot(figh)
                mcol1, mcol2 = st.columns(2)
                mcol1.metric("Cumulative error", f"{hc['cum_err_pct']:+.1f}%",
                             help="Forecast cum vs actual cum over the held-out window. + = over-forecast.")
                mcol2.metric("Mean abs. % error", f"{hc['mape_pct']:.1f}%")

fluid_block("Oil", "EUR (Mbbl)", "NormOil")
st.markdown("---")
fluid_block("Gas", "EUR (MMcf)", "NormGas")
st.markdown("---")
fluid_block("Water", "EUR (Mbbl water)", "NormWater")
st.markdown("---")

# ================= Outlier Detection =================
st.header("Outlier Detection — EUR vs Lateral Length")

_out_fluid_cfg = [
    ("Oil",   "Oil_oneline",   "EUR (Mbbl)"),
    ("Gas",   "Gas_oneline",   "EUR (MMcf)"),
    ("Water", "Water_oneline", "EUR (Mbbl water)"),
]

def _outlier_section(fluid: str, on_key: str, eur_col: str):
    if on_key not in st.session_state:
        st.info(f"Run {fluid} forecast first.")
        return

    _show_flash(f"{fluid}_outlier_flash")
    oneline = st.session_state[on_key]
    result  = _make_outlier_fig(oneline, eur_col, fluid)
    if result is None:
        st.info(f"Not enough data for {fluid} outlier detection (need ≥ 3 wells with lateral length).")
        return

    fig, upper_fence, lower_fence, n_active, n_removed, n_high, n_low, median_eur, unit = result
    st.pyplot(fig)
    plt.close(fig)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Active wells",  n_active)
    m2.metric("Flagged wells", n_removed)
    m3.metric("Median EUR",    f"{median_eur:,.0f} {unit}")
    m4.metric("Upper fence",   f"{upper_fence:,.0f} {unit}")
    m5.metric("Lower fence",   f"{lower_fence:,.0f} {unit}")
    m6.metric("High / Low",    f"{n_high} ⬆ / {n_low} ⬇")

    # Recompute fences to identify individual outlier wells
    df = oneline.copy()
    df['_eur'] = pd.to_numeric(df[eur_col], errors='coerce')
    if 'LateralLength' in df.columns:
        df['_lat'] = pd.to_numeric(df['LateralLength'], errors='coerce')
        df = df.dropna(subset=['_eur', '_lat'])
    else:
        df = df.dropna(subset=['_eur'])
    q1, q3      = df['_eur'].quantile(0.25), df['_eur'].quantile(0.75)
    iqr         = q3 - q1
    upper_f     = q3 + 1.5 * iqr
    lower_f     = _log_lower_fence(df['_eur'])
    high_df     = df[df['_eur'] > upper_f].sort_values('_eur', ascending=False).copy()
    low_df      = df[df['_eur'] < lower_f].sort_values('_eur', ascending=True).copy()

    sel_key = f"{fluid}_tc_selection"
    all_ids = set(oneline['WellID'].astype(str).tolist())
    if sel_key not in st.session_state:
        st.session_state[sel_key] = all_ids.copy()

    if high_df.empty and low_df.empty:
        st.success(f"No outliers detected for {fluid}.")
        return

    st.markdown(f"**{len(high_df) + len(low_df)} outlier(s) detected.** Tick wells to exclude, "
                "or use a bulk button, then click **Apply** — nothing recomputes until you do.")

    show_cols = [c for c in ['WellName', 'API10', 'LateralLength', 'Months History', eur_col]
                 if c in df.columns]
    current_excluded = all_ids - st.session_state[sel_key]
    flagged_ids = set(high_df['WellID'].astype(str)) | set(low_df['WellID'].astype(str))
    tables = []

    # Everything inside the form is collected without triggering a script
    # rerun; only the submit buttons below do.
    with st.form(key=f"{fluid}_outlier_form", border=False):
        for _label, _flag_df, _ed_key in [
            (f"⬆ High EUR outliers — above {upper_f:,.0f} {unit} (1.5×IQR fence)",
             high_df, _editor_key(fluid, 'outlier', 'outlier_editor_high')),
            (f"⬇ Low EUR outliers — below {lower_f:,.0f} {unit} (log-space 1.5×IQR fence)",
             low_df, _editor_key(fluid, 'outlier', 'outlier_editor_low')),
        ]:
            st.markdown(f"**{_label}**")
            if _flag_df.empty:
                st.caption("None detected.")
                continue

            display = _flag_df[['WellID'] + show_cols].reset_index(drop=True)
            display.insert(0, "Exclude", display['WellID'].astype(str).isin(current_excluded).tolist())

            col_cfg = {"Exclude": st.column_config.CheckboxColumn("Exclude", default=False)}
            for c in show_cols:
                col_cfg[c] = st.column_config.TextColumn(c, disabled=True)

            edited = st.data_editor(
                display.drop(columns=['WellID']),
                column_config=col_cfg,
                hide_index=True,
                use_container_width=True,
                key=_ed_key,
            )
            tables.append((display, edited))

        b1, b2, b3, b4 = st.columns(4)
        apply_ticked = b1.form_submit_button("Apply ticked exclusions", type="primary",
                                             use_container_width=True)
        excl_high    = b2.form_submit_button(f"Exclude all {len(high_df)} high",
                                             disabled=high_df.empty, use_container_width=True)
        excl_low     = b3.form_submit_button(f"Exclude all {len(low_df)} low",
                                             disabled=low_df.empty, use_container_width=True)
        excl_all     = b4.form_submit_button(f"Exclude all {len(flagged_ids)} flagged",
                                             use_container_width=True)

    if apply_ticked or excl_high or excl_low or excl_all:
        new_excl: set = set()
        for display, edited in tables:
            _mask = edited['Exclude'].values.astype(bool)
            new_excl |= set(display.loc[_mask, 'WellID'].astype(str).tolist())
        if excl_high or excl_all:
            new_excl |= set(high_df['WellID'].astype(str))
        if excl_low or excl_all:
            new_excl |= set(low_df['WellID'].astype(str))
        # Keep exclusions made elsewhere (analog table / chart) for wells that
        # are not in these outlier tables.
        other_excl = current_excluded - flagged_ids
        st.session_state[sel_key] = all_ids - (new_excl | other_excl)
        _request_editor_reset(fluid)
        _flash(f"{fluid}_outlier_flash",
               f"{fluid}: {len(new_excl)} outlier(s) excluded from the Type Curve "
               f"({len(st.session_state[sel_key])} / {len(all_ids)} wells remain). "
               "Scroll down to see the updated TC.")
        st.rerun()

    if st.button(f"Reset {fluid} selection (include all wells)", key=f"{fluid}_reset_outlier"):
        st.session_state[sel_key] = all_ids.copy()
        _request_editor_reset(fluid)
        _flash(f"{fluid}_outlier_flash", f"{fluid} Type Curve restored to all {len(all_ids)} wells.")
        st.rerun()

_out_tabs = st.tabs(["Oil", "Gas", "Water"])
with _out_tabs[0]: _outlier_section("Oil",   "Oil_oneline",   "EUR (Mbbl)")
with _out_tabs[1]: _outlier_section("Gas",   "Gas_oneline",   "EUR (MMcf)")
with _out_tabs[2]: _outlier_section("Water", "Water_oneline", "EUR (Mbbl water)")

st.markdown("---")

# ================= Type Wells =================
st.header("Type Wells — Summary")
if "merged_full" in st.session_state:
    _global_rerun_button("rerun_all_tw")
tw_tabs = st.tabs(["Oil", "Gas", "Water"])

def _render_tw(fluid, on_key, mo_key, eur_col):
    if on_key not in st.session_state:
        st.info(f"Run {fluid} first.")
        return

    _show_flash(f"{fluid}_tw_flash")
    oneline_full = st.session_state[on_key]

    # --- Analog selection state: set of included WellIDs ---
    sel_key = f"{fluid}_tc_selection"
    all_ids = set(oneline_full['WellID'].astype(str).tolist())
    if sel_key not in st.session_state:
        st.session_state[sel_key] = all_ids.copy()
    # Drop IDs that no longer exist (e.g. after a re-forecast changed the well list)
    st.session_state[sel_key] = set(st.session_state[sel_key]) & all_ids
    saved_ids = st.session_state[sel_key]

    # WellID -> display name (disambiguate duplicate well names with the ID)
    _names = oneline_full[['WellID', 'WellName']].astype(str)
    _dup = _names['WellName'].duplicated(keep=False)
    name_map = {}
    for wid, nm, d in zip(_names['WellID'], _names['WellName'], _dup):
        nm = wid if (not nm or nm == 'nan') else nm
        name_map[wid] = f"{nm} ({wid})" if d else nm

    # Build display columns available in the oneline
    show_cols = [c for c in [
        'WellName', 'PrimaryFormation', 'LateralLength', eur_col
    ] if c in oneline_full.columns]

    editor_df = oneline_full[show_cols].copy().reset_index(drop=True)
    editor_df.insert(0, "Select", oneline_full['WellID'].astype(str).isin(saved_ids).tolist())

    col_cfg = {"Select": st.column_config.CheckboxColumn("Select", default=True)}
    for c in show_cols:
        col_cfg[c] = st.column_config.TextColumn(c, disabled=True)

    # ── Two-column layout: table left, plot right ──
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.caption(f"**Analog Wells** — tick/untick, then **Apply selection**; nothing "
                   f"recomputes until you do ({len(oneline_full)} wells total).")
        # The data editor sits inside a form so ticking boxes does not rerun the app.
        with st.form(key=f"{fluid}_analog_form", border=False):
            edited = st.data_editor(
                editor_df,
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                key=_editor_key(fluid, 'analog', 'analog_editor'),
            )
            fb1, fb2, fb3 = st.columns(3)
            apply_sel = fb1.form_submit_button("Apply selection", type="primary",
                                               use_container_width=True)
            sel_all   = fb2.form_submit_button("Select all", use_container_width=True)
            desel_all = fb3.form_submit_button("Deselect all", use_container_width=True)
        if apply_sel or sel_all or desel_all:
            if sel_all:
                new_sel = all_ids.copy()
            elif desel_all:
                new_sel = set()
            else:
                _mask = edited["Select"].values.astype(bool)
                new_sel = set(oneline_full['WellID'].astype(str).to_numpy()[_mask].tolist())
            st.session_state[sel_key] = new_sel
            _request_editor_reset(fluid, 'analog')
            _flash(f"{fluid}_tw_flash", f"{fluid}: {len(new_sel)} / {len(all_ids)} wells selected "
                                        "for the Type Curve.")
            st.rerun()

    sel_ids = set(st.session_state[sel_key])
    oneline_filtered = oneline_full[oneline_full['WellID'].astype(str).isin(sel_ids)].reset_index(drop=True)
    n_sel = len(oneline_filtered)
    n_tot = len(oneline_full)

    # Compute EUR stats upfront so P50 curve can be scaled to match
    eurs = []
    eur_stats = {}
    if n_sel > 0 and eur_col in oneline_filtered.columns:
        eurs = pd.to_numeric(oneline_filtered[eur_col], errors='coerce').dropna().tolist()
        eur_stats = compute_eur_stats(eurs)

    with right_col:
        if mo_key in st.session_state and n_sel > 0:
            monthly_filtered = st.session_state[mo_key][
                st.session_state[mo_key]['WellID'].astype(str).isin(sel_ids)
            ]
            curves, lines = build_type_curves_and_lines(
                monthly_filtered, fluid.lower(), oneline=oneline_filtered, d_lim=float(d_lim_cfg)
            )
            # Scale every TC curve so its cumulative volume exactly matches
            # the corresponding EUR percentile from the stats table.
            # Curves are in Bbl or Mcf; EUR stats are in Mbbl or MMcf → ×1000.
            if not curves.empty:
                curves = curves.copy()
                for _pct, _stat in [('P10', 'P10'), ('P50', 'P50'), ('P90', 'P90'),
                                    ('Mean', 'Mean')]:
                    if (_pct in curves.columns
                            and eur_stats.get(_stat)
                            and np.isfinite(eur_stats[_stat])):
                        _csum = curves[_pct].sum()
                        if _csum > 0:
                            curves[_pct] = curves[_pct] * (eur_stats[_stat] * 1000 / _csum)
            st.session_state[f"{fluid}_tc_curves"] = curves
            st.session_state[f"{fluid}_tc_lines"] = lines

            if _HAS_PLOTLY:
                sel_fp   = _selection_fingerprint(sel_ids)
                hl_key   = f"{fluid}_hl_{sel_fp}"       # highlight picker = the single
                plot_key = f"{fluid}_tc_plot_{sel_fp}"  # source of truth for picks
                # Both widgets are keyed by the plotted well set, so their state is
                # dropped whenever the selection changes.
                #
                # Curves clicked on the chart are reported by the chart widget on the
                # following run; read them BEFORE the picker is drawn and merge them
                # into it (assigning a widget's value is allowed before it is drawn,
                # and the assignment is what pushes the new value to the browser).
                # The figure depends only on the picker, never on the chart's own
                # selection: Streamlit hashes the figure into the chart's identity,
                # so styling from its selection would re-key the chart on every
                # click and lose the selection.
                if st.session_state.pop(f"{fluid}_hl_clear", False):
                    st.session_state[hl_key] = []
                base_trace_ids = [str(l[2]) for l in lines]
                clicked = _chart_clicked_wells(plot_key, base_trace_ids) & sel_ids
                cur_hl = set(st.session_state.get(hl_key, []))
                if clicked - cur_hl:
                    st.session_state[hl_key] = sorted(
                        cur_hl | clicked, key=lambda w: name_map.get(w, w).lower())
                hl_ids = set(st.multiselect(
                    "Highlighted wells (click a curve or pick a name)",
                    options=sorted(sel_ids, key=lambda w: name_map.get(w, w).lower()),
                    format_func=lambda w: name_map.get(w, w),
                    key=hl_key,
                    placeholder="Click curves on the chart or pick wells here…",
                ))
                fig, _ = plot_type_curves_interactive(curves, lines, fluid.lower(), name_map, hl_ids)
                st.plotly_chart(fig, use_container_width=True, key=plot_key,
                                on_select="rerun", selection_mode="points")
                st.caption("Hover a curve to see its well; click a curve to add it to the "
                           "highlighted set (remove one with the × in the picker above).")
                if hl_ids:
                    st.markdown("**Selected:** " + ", ".join(
                        name_map.get(w, w) for w in sorted(hl_ids, key=lambda w: name_map.get(w, w))))
                bx1, bx2 = st.columns([2, 1])
                if bx1.button(f"Exclude {len(hl_ids)} selected well(s) from Type Curve",
                              key=f"{fluid}_excl_picked_{sel_fp}", type="primary",
                              disabled=not hl_ids, use_container_width=True):
                    st.session_state[sel_key] = sel_ids - hl_ids
                    _request_editor_reset(fluid, 'analog')
                    _flash(f"{fluid}_tw_flash",
                           f"{fluid}: excluded {len(hl_ids)} well(s) from the Type Curve "
                           f"({len(sel_ids - hl_ids)} / {n_tot} remain).")
                    st.rerun()
                if bx2.button("Clear highlights", key=f"{fluid}_hl_clear_btn_{sel_fp}",
                              disabled=not hl_ids, use_container_width=True):
                    st.session_state[f"{fluid}_hl_clear"] = True
                    st.rerun()
            else:
                st.pyplot(plot_type_curves(curves, lines, fluid.lower()))
                st.caption("Install `plotly` for an interactive chart with hover names "
                           "and click-to-select.")

    # ── EUR summary below ──
    st.caption(f"**{n_sel} / {n_tot} wells selected**")
    if n_sel > 0:
        _well_significance_note(n_sel)
        st.dataframe(
            format_df_2dec(eur_summary_table(fluid, eur_stats, "Mbbl" if fluid != "Gas" else "MMcf", int(norm_len))),
            use_container_width=True,
        )
        _mean_v = eur_stats.get("Mean")
        _p50_v  = eur_stats.get("P50")
        if _mean_v and _p50_v and np.isfinite(_mean_v) and np.isfinite(_p50_v):
            _unit = "Mbbl" if fluid != "Gas" else "MMcf"
            st.caption(
                f"SPEE M5 'average well' = **Mean {_mean_v:,.1f} {_unit}** "
                f"({(_mean_v/_p50_v - 1)*100:+.0f}% vs P50 {_p50_v:,.1f}). "
                "Use the Mean (expected value) when aggregating reserves across wells."
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
    for _pct in ["P10", "P50", "P90", "Mean"]:
        if not _oil_c.empty and _pct in _oil_c.columns:
            _og_parts.append((_oil_c[_pct] / 1000).round(4).rename(f"{_pct}_Oil_Mbbl_per_month"))
        if not _gas_c.empty and _pct in _gas_c.columns:
            _og_parts.append((_gas_c[_pct] / 1000).round(4).rename(f"{_pct}_Gas_MMcf_per_month"))
    _og_df = pd.concat(_og_parts, axis=1)

    # Water sheet
    _water_parts = [pd.Series(range(1, 601), name="Month")]
    for _pct in ["P10", "P50", "P90", "Mean"]:
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


# ── PPTX data builder ──────────────────────────────────────────────────────────
def _build_pptx_data(tc_name: str) -> dict:
    """Extract live session-state data into the pptxgenjs data contract."""
    import json as _json
    from datetime import datetime as _dt

    D_LIM = float(d_lim_cfg)

    def _get_fluid(fluid_name, oneline_key, eur_col):
        if oneline_key not in st.session_state:
            return None
        oneline_full = st.session_state[oneline_key]
        sel_key = f"{fluid_name}_tc_selection"
        if sel_key in st.session_state and isinstance(st.session_state[sel_key], set):
            sel_ids = st.session_state[sel_key]
            oneline = oneline_full[oneline_full['WellID'].astype(str).isin(sel_ids)].reset_index(drop=True)
        else:
            oneline = oneline_full.reset_index(drop=True)

        n_tc    = len(oneline)
        n_total = len(oneline_full)
        is_gas  = fluid_name.lower() == 'gas'
        eur_unit      = "MMcf" if is_gas else "Mbbl"
        eur_per_ft_u  = "Mcf/ft" if is_gas else "Bbl/ft"
        qi_unit       = "Mcf/day" if is_gas else "bbl/day"

        eurs = pd.to_numeric(oneline[eur_col], errors='coerce').dropna().astype(float) if eur_col in oneline.columns else pd.Series([], dtype=float)
        eur_stats = compute_eur_stats(eurs.tolist()) if len(eurs) > 0 else {}
        _pk = lambda kw: next((k for k in eur_stats if kw in k.lower()), None)
        p10e  = float(eur_stats.get(_pk('p10'), 0) or 0)
        p50e  = float(eur_stats.get(_pk('p50') or _pk('median'), 0) or 0)
        p90e  = float(eur_stats.get(_pk('p90'), 0) or 0)
        mne   = float(eur_stats.get(_pk('mean'), 0) or 0)

        lat_vals = pd.to_numeric(oneline['LateralLength'], errors='coerce').dropna() if 'LateralLength' in oneline.columns else pd.Series([], dtype=float)
        avg_lat  = float(lat_vals.mean()) if len(lat_vals) > 0 else float(norm_len)
        # Normalised EURs are already scaled to norm_len, so per-foot values must divide
        # by norm_len (as the app's table and probit do), not by the analogs' mean lateral.
        _ft_basis = float(norm_len) if use_norm else avg_lat
        def _epft(e): return (e * 1000) / _ft_basis if _ft_basis > 0 else 0.0

        b_vals = pd.to_numeric(oneline['b'], errors='coerce').dropna() if 'b' in oneline.columns else pd.Series([], dtype=float)
        bc = len(b_vals)

        # Group parameters (mirror build_type_curves_and_lines grouping)
        df_tc = oneline.copy()
        df_tc['_eur'] = pd.to_numeric(df_tc[eur_col], errors='coerce') if eur_col in df_tc.columns else 0
        df_tc['_qi']  = pd.to_numeric(df_tc.get('qi (per day)', pd.Series(dtype=float)), errors='coerce')
        df_tc['_b']   = pd.to_numeric(df_tc.get('b',            pd.Series(dtype=float)), errors='coerce')
        df_tc['_di']  = pd.to_numeric(df_tc.get('di (per month)', pd.Series(dtype=float)), errors='coerce')
        df_tc['_dec'] = pd.to_numeric(df_tc.get('First-Year Decline (%)', pd.Series(dtype=float)), errors='coerce')
        df_tc = df_tc.dropna(subset=['_eur','_qi','_b','_di']).sort_values('_eur', ascending=False).reset_index(drop=True)
        nn = len(df_tc)

        def _gs(grp):
            qi  = float(grp['_qi'].median()) if len(grp) > 0 else 0.0
            b_  = float(grp['_b'].median())  if len(grp) > 0 else 1.0
            di_ = float(grp['_di'].median()) if len(grp) > 0 else D_LIM
            dec = float(grp['_dec'].dropna().median()) / 100 if (len(grp) > 0 and grp['_dec'].notna().any()) else 0.0
            return qi, b_, di_, dec

        if nn >= 1:
            p50_lo = max(0, int(nn * 0.45))
            p50_hi = max(p50_lo + 1, int(nn * 0.55))
            grps = {
                'P10': df_tc.iloc[:max(1, int(nn * 0.1))],
                'P50': df_tc.iloc[p50_lo:p50_hi],
                'P90': df_tc.iloc[max(0, int(nn * 0.9)):],
            }
            p10qi, p10b, p10di, p10dec = _gs(grps['P10'])
            p50qi, p50b, p50di, p50dec = _gs(grps['P50'])
            p90qi, p90b, p90di, p90dec = _gs(grps['P90'])
        else:
            p10qi = p50qi = p90qi = 0.0
            p10b  = p50b  = p90b  = 1.0
            p10di = p50di = p90di = D_LIM
            p10dec = p50dec = p90dec = 0.0

        analog_rows = []
        for _, row in oneline.iterrows():
            api_v  = str(row.get('API10', row.get('WellID', '')))
            wn     = str(row.get('WellName', ''))
            lat_ft = float(row['LateralLength']) if 'LateralLength' in row and pd.notna(row['LateralLength']) else avg_lat
            qi_d   = float(row['qi (per day)'])        if 'qi (per day)'        in row and pd.notna(row['qi (per day)'])        else 0.0
            bv     = float(row['b'])                   if 'b'                   in row and pd.notna(row['b'])                   else 1.0
            di_mo  = float(row['di (per month)'])      if 'di (per month)'      in row and pd.notna(row['di (per month)'])      else D_LIM
            dec1   = float(row['First-Year Decline (%)']) if 'First-Year Decline (%)' in row and pd.notna(row['First-Year Decline (%)']) else 0.0
            eur_v  = float(row[eur_col])               if eur_col               in row and pd.notna(row[eur_col])               else 0.0
            _den   = _ft_basis if use_norm else lat_ft
            epft_v = (eur_v * 1000) / _den if _den > 0 else 0.0
            analog_rows.append({
                'api': api_v, 'wellName': wn, 'latLenFt': lat_ft,
                'qi': qi_d, 'b': bv, 'diPerMo': di_mo,
                'decline1yrPct': dec1, 'eur': eur_v, 'eurPerFt': epft_v,
            })

        return {
            'fluid': fluid_name.lower(),
            'nTcWells': n_tc, 'nTotal': n_total,
            'meanEur': mne, 'p10Eur': p10e, 'p50Eur': p50e, 'p90Eur': p90e,
            'eurUnit': eur_unit, 'eurPerFtUnit': eur_per_ft_u,
            'meanEurPerFt': _epft(mne), 'p10EurPerFt': _epft(p10e),
            'p50EurPerFt': _epft(p50e), 'p90EurPerFt': _epft(p90e),
            'bMean':   float(b_vals.mean())             if bc > 0 else 0.0,
            'bMedian': float(b_vals.median())           if bc > 0 else 0.0,
            'bP10':    float(np.percentile(b_vals, 10)) if bc > 0 else 0.0,
            'bP90':    float(np.percentile(b_vals, 90)) if bc > 0 else 0.0,
            'bCount': int(bc),
            'bValues': [float(v) for v in b_vals.tolist()],
            'p90Qi': p90qi, 'p50Qi': p50qi, 'p10Qi': p10qi, 'qiUnit': qi_unit,
            'p90B':  p90b,  'p50B':  p50b,  'p10B':  p10b,
            'p90Di': p90di, 'p50Di': p50di, 'p10Di': p10di,
            'p90Decline1yr': p90dec, 'p50Decline1yr': p50dec, 'p10Decline1yr': p10dec,
            'terminalDi': D_LIM,
            'analogRows': analog_rows,
        }

    # Well meta — derive from best available oneline
    primary_key = next((k for k in ['Oil_oneline','Gas_oneline','Water_oneline'] if k in st.session_state), None)
    prim_fl     = primary_key.split('_')[0] if primary_key else 'Oil'
    prim_ol     = st.session_state.get(primary_key, pd.DataFrame())
    sel_key     = f"{prim_fl}_tc_selection"
    if sel_key in st.session_state and isinstance(st.session_state[sel_key], set):
        prim_tc = prim_ol[prim_ol['WellID'].astype(str).isin(st.session_state[sel_key])]
    else:
        prim_tc = prim_ol

    def _mode(col): return str(prim_tc[col].mode().iloc[0]) if (col in prim_tc.columns and not prim_tc[col].mode().empty) else ''
    county    = _mode('County')
    state_val = _mode('State')
    reservoir = _mode('PrimaryFormation')
    lat_vals  = pd.to_numeric(prim_tc['LateralLength'], errors='coerce').dropna() if 'LateralLength' in prim_tc.columns else pd.Series([], dtype=float)
    avg_lat   = float(lat_vals.mean()) if len(lat_vals) > 0 else float(norm_len)

    comp_dates  = pd.to_datetime(prim_tc['CompletionDate'], errors='coerce').dropna() if 'CompletionDate' in prim_tc.columns else pd.Series([], dtype='datetime64[ns]')
    comp_vintage = (f"{comp_dates.min().year} – {comp_dates.max().year}" if len(comp_dates) > 0 else '')

    unique_counties = list(prim_tc['County'].dropna().unique()) if 'County' in prim_tc.columns else [county]
    ll_range = (f"{int(lat_vals.min()):,} – {int(lat_vals.max()):,} ft" if len(lat_vals) > 0 else '')

    now = _dt.now()
    data = {
        'wellMeta': {
            'wellName':       tc_name or 'TC Report',
            'county':         county,
            'state':          state_val,
            'operator':       'SE',
            'reservoir':      reservoir,
            'afeLateralLenFt': float(norm_len),
            'tvdFt':          0.0,
            'reportDate':     now.isoformat(),
            'generatedDate':  now.strftime('%Y-%m-%d'),
            'normLengthFt':   float(norm_len),
            'bFactorRange':   f"{b_low:.3f} - {b_high:.3f}",
            'scalingExponent': float(st.session_state.get('applied_norm_exponent', 1.0)),
            'terminalDeclinePct': float(term_dec_pct),
            'declineModels':  _decline_models_label(),
        },
        'analogCriteria': {
            'nAnalogs':        len(prim_tc),
            'counties':        ', '.join(str(c) for c in unique_counties),
            'reservoir':       reservoir,
            'llRangeFt':       ll_range,
            'llAvgFt':         avg_lat,
            'compVintage':     comp_vintage,
            'avgDistanceMi':   'N/A',
        },
    }
    _blank = lambda fl, eu, epfu, qu: {
        'fluid': fl, 'nTcWells': 0, 'nTotal': 0,
        'meanEur': 0.0, 'p10Eur': 0.0, 'p50Eur': 0.0, 'p90Eur': 0.0,
        'eurUnit': eu, 'eurPerFtUnit': epfu,
        'meanEurPerFt': 0.0, 'p10EurPerFt': 0.0, 'p50EurPerFt': 0.0, 'p90EurPerFt': 0.0,
        'bMean': 0.0, 'bMedian': 0.0, 'bP10': 0.0, 'bP90': 0.0, 'bCount': 0, 'bValues': [],
        'p90Qi': 0.0, 'p50Qi': 0.0, 'p10Qi': 0.0, 'qiUnit': qu,
        'p90B': 1.0, 'p50B': 1.0, 'p10B': 1.0,
        'p90Di': D_LIM, 'p50Di': D_LIM, 'p10Di': D_LIM,
        'p90Decline1yr': 0.0, 'p50Decline1yr': 0.0, 'p10Decline1yr': 0.0,
        'terminalDi': D_LIM, 'analogRows': [],
    }
    data['oil']   = _get_fluid('Oil',   'Oil_oneline',   'EUR (Mbbl)')       or _blank('oil',   'Mbbl',  'Bbl/ft', 'bbl/day')
    data['gas']   = _get_fluid('Gas',   'Gas_oneline',   'EUR (MMcf)')       or _blank('gas',   'MMcf',  'Mcf/ft', 'Mcf/day')
    data['water'] = _get_fluid('Water', 'Water_oneline', 'EUR (Mbbl water)') or _blank('water', 'Mbbl',  'Bbl/ft', 'bbl/day')

    # Generate the exact plots shown in the Streamlit interface as PNG bytes
    def _fig_bytes(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        buf.seek(0); plt.close(fig)
        return buf.read()

    def _attach_figs(fluid_name, oneline_key, monthly_key, eur_col, fd):
        if oneline_key not in st.session_state or fd['nTcWells'] == 0:
            return
        oneline_full = st.session_state[oneline_key]
        sel_key = f"{fluid_name}_tc_selection"
        if sel_key in st.session_state and isinstance(st.session_state[sel_key], set):
            oneline = oneline_full[oneline_full['WellID'].astype(str).isin(st.session_state[sel_key])]
        else:
            oneline = oneline_full

        # B-factor histogram (same as bfactor_analytics_figures but keep figure)
        b_vals = pd.to_numeric(oneline.get('b'), errors='coerce').dropna()
        if len(b_vals) > 0:
            color = _phase_color(fluid_name)
            P50 = float(np.median(b_vals))
            fig, ax = plt.subplots(figsize=(6.6, 3.4))
            ax.hist(b_vals.values, bins=30, color=color, alpha=0.85, edgecolor='none')
            ax.axvline(P50, color='black', linestyle='--', label=f"Median={P50:.2f}")
            ax.legend(); ax.set_xlabel('b-factor'); ax.set_ylabel('Count')
            plt.tight_layout()
            fd['bfactor_hist_bytes'] = _fig_bytes(fig)

        # Type curve — use cached session-state curves so it matches the UI exactly
        curves = st.session_state.get(f"{fluid_name}_tc_curves", pd.DataFrame())
        lines  = st.session_state.get(f"{fluid_name}_tc_lines",  [])
        if not curves.empty:
            fig = plot_type_curves(curves, lines, fluid_name.lower())
            fd['typecurve_bytes'] = _fig_bytes(fig)

        # Probit plot
        if eur_col in oneline.columns:
            eurs = pd.to_numeric(oneline[eur_col], errors='coerce').dropna().astype(float).tolist()
            if eurs:
                unit = "Mbbl" if fluid_name != "Gas" else "MMcf"
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Distribution",
                                  _phase_color(fluid_name), norm_len=int(norm_len))
                fd['probit_bytes'] = _fig_bytes(fig)

    _attach_figs('Oil',   'Oil_oneline',   'Oil_monthly',   'EUR (Mbbl)',       data['oil'])
    _attach_figs('Gas',   'Gas_oneline',   'Gas_monthly',   'EUR (MMcf)',        data['gas'])
    _attach_figs('Water', 'Water_oneline', 'Water_monthly', 'EUR (Mbbl water)', data['water'])
    return data


def generate_tc_pptx(pptx_data: dict, output_dir: str) -> str | None:
    """Generate a TC PowerPoint report using python-pptx (no Node.js required)."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
    except ImportError:
        st.warning("python-pptx not installed. Run: pip install python-pptx")
        return None

    import math as _math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt
    from io import BytesIO as _BytesIO
    from datetime import datetime as _dt

    # ── Brand constants ────────────────────────────────────────────────────────
    SE_RED    = RGBColor(0x95, 0x37, 0x35)
    NAVY      = RGBColor(0x1A, 0x27, 0x44)
    WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
    ROW_ALT   = RGBColor(0xF5, 0xED, 0xED)
    BODY_GRAY = RGBColor(0x40, 0x40, 0x40)
    MARK_GRAY = RGBColor(0xAA, 0xAA, 0xAA)
    FLUID_HEX = {'oil': '#2D6E2D', 'gas': '#CC3333', 'water': '#2244AA'}
    D_LIM     = float(d_lim_cfg)
    _RECT     = 1   # MSO_AUTO_SHAPE_TYPE.RECTANGLE
    _OVAL     = 9   # MSO_AUTO_SHAPE_TYPE.OVAL
    _ALIGN    = {'left': PP_ALIGN.LEFT, 'center': PP_ALIGN.CENTER, 'right': PP_ALIGN.RIGHT}

    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(5.625)
    _blank = prs.slide_layouts[6]

    # ── Drawing primitives ─────────────────────────────────────────────────────
    def _rect(sl, x, y, w, h, fill, line=None, lw=0.5):
        sp = sl.shapes.add_shape(_RECT, Inches(x), Inches(y), Inches(w), Inches(h))
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
        if line:
            sp.line.color.rgb = line; sp.line.width = Pt(lw)
        else:
            sp.line.fill.background()
        return sp

    def _oval(sl, x, y, w, h, fill, line, lw=1.5):
        sp = sl.shapes.add_shape(_OVAL, Inches(x), Inches(y), Inches(w), Inches(h))
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
        sp.line.color.rgb = line; sp.line.width = Pt(lw)
        return sp

    def _txt(sl, text, x, y, w, h, size=10, bold=False, italic=False,
             color=None, face="Garamond", align='left', wrap=True):
        tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = wrap
        p = tf.paragraphs[0]; p.alignment = _ALIGN.get(align, PP_ALIGN.LEFT)
        r = p.add_run(); r.text = str(text)
        r.font.size = Pt(size); r.font.bold = bool(bold); r.font.italic = bool(italic)
        r.font.name = face
        if color: r.font.color.rgb = color
        return tb

    def _chrome(sl, pg):
        _rect(sl, 0, 5.3, 10, 0.325, SE_RED)
        _oval(sl, 0.15, 5.05, 0.5, 0.5, NAVY, SE_RED)
        _txt(sl, "SE",  0.15, 5.12, 0.5,  0.35, size=9,  bold=True, color=WHITE, face="Garamond", align='center')
        _txt(sl, str(pg), 9.5, 5.3,  0.4,  0.32, size=10, color=WHITE, align='right')
        _txt(sl, "CONFIDENTIAL", 0.0, 0.5, 0.5, 4.5, size=7, color=MARK_GRAY, align='center')
        _txt(sl, "CONFIDENTIAL", 9.5, 0.5, 0.5, 4.5, size=7, color=MARK_GRAY, align='center')

    def _slide_title(sl, title):
        _txt(sl, title, 0.4, 0.18, 9.2, 0.52, size=24, bold=True, color=SE_RED, face="Garamond")
        _rect(sl, 0.4, 0.73, 9.2, 0.03, SE_RED)

    def _kv_box(sl, header, rows, x, y, w, rh=0.30):
        _rect(sl, x, y, w, rh, SE_RED)
        _txt(sl, header, x, y, w, rh, size=10, bold=True, color=WHITE)
        for i, (lbl, val) in enumerate(rows):
            ry = y + rh * (i + 1)
            bg = WHITE if i % 2 == 0 else ROW_ALT
            _rect(sl, x, ry, w, rh, bg, RGBColor(0xDD, 0xDD, 0xDD))
            _txt(sl, lbl,       x,          ry, w * 0.5, rh, size=9.5, color=BODY_GRAY)
            _txt(sl, str(val),  x + w * 0.5, ry, w * 0.5, rh, size=9.5, color=BODY_GRAY, align='center')

    def _table(sl, headers, rows, x, y, cws, rh=0.28):
        cx = x
        for i, hdr in enumerate(headers):
            _rect(sl, cx, y, cws[i], rh, SE_RED)
            _txt(sl, hdr, cx, y, cws[i], rh, size=9, bold=True, color=WHITE, align='center')
            cx += cws[i]
        for ri, row in enumerate(rows):
            ry = y + rh * (ri + 1)
            bg = WHITE if ri % 2 == 0 else ROW_ALT
            cx2 = x
            for ci, cell in enumerate(row):
                _rect(sl, cx2, ry, cws[ci], rh, bg, RGBColor(0xDD, 0xDD, 0xDD))
                _txt(sl, str(cell), cx2, ry, cws[ci], rh, size=9, color=BODY_GRAY, align='center')
                cx2 += cws[ci]

    def _embed(sl, fig, x, y, w, h):
        buf = _BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#F8F8F8')
        buf.seek(0); _plt.close(fig)
        sl.shapes.add_picture(buf, Inches(x), Inches(y), Inches(w), Inches(h))

    # ── Format helpers ─────────────────────────────────────────────────────────
    def _ff(n):       return f"{int(round(n)):,} ft"
    def _fe(n, u):    return f"{n:.1f} {u}"
    def _fn(n):       return f"{int(round(n)):,}" if n >= 1000 else f"{n:.2f}"
    def _fp(v):       return f"{v * 100:.1f}%"
    def _cap(s):      return s[0].upper() + s[1:] if s else s
    def _fd(iso):
        try:    return _dt.fromisoformat(iso).strftime("%B %-d, %Y")
        except: return str(iso)

    def _arps(qi_d, b, di, term_di, months=600):
        qi = qi_d * DAYS_PER_MONTH
        t_sw = ((di - term_di) / (b * di * term_di)) if (b > 0 and di > term_di) else float('inf')
        q_sw = qi / (1 + b * di * t_sw) ** (1 / b) if t_sw < float('inf') else qi
        q = []
        for t in range(1, months + 1):
            if t >= t_sw and t_sw < float('inf'):
                q.append(q_sw * _math.exp(-term_di * (t - t_sw)))
            else:
                q.append(qi / (1 + b * di * t) ** (1 / b))
        return q

    wm  = pptx_data['wellMeta']
    ac  = pptx_data['analogCriteria']
    oil = pptx_data['oil']
    gas = pptx_data['gas']
    wtr = pptx_data['water']

    # ── Slide 1: Title ─────────────────────────────────────────────────────────
    sl = prs.slides.add_slide(_blank)
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb = NAVY
    _oval(sl, 0.3, 0.2, 1.1, 1.1, NAVY, SE_RED, lw=3)
    _txt(sl, "SE", 0.3, 0.42, 1.1, 0.5, size=22, bold=True, color=WHITE, face="Garamond", align='center')
    _txt(sl, wm['wellName'], 0.4, 1.8, 7, 0.85, size=36, bold=True, color=WHITE, face="Garamond")
    _txt(sl, f"TC Report – {_fd(wm['reportDate'])}", 0.4, 2.7, 7, 0.6,
         size=24, italic=True, color=RGBColor(0xCC, 0xCC, 0xCC), face="Garamond")
    _rect(sl, 0, 5.3, 10, 0.325, SE_RED)

    # ── Slide 2: Disclaimer ────────────────────────────────────────────────────
    sl = prs.slides.add_slide(_blank)
    _slide_title(sl, "Disclaimer"); _chrome(sl, 2)
    _txt(sl, (
        'The information contained in this confidential presentation (this “Presentation”) is provided for '
        'informational and discussion purposes only and is not, and may not be relied on in any manner as, '
        'legal, tax or investment advice or as an offer to sell or a solicitation of an offer to buy an interest '
        'in any security. The information contained in this Presentation must be kept strictly confidential and '
        'may not be reproduced or redistributed in any format without the approval of Schaper International '
        'Petroleum Consulting, LLC (“SIPC”). In considering any performance data contained in this Presentation, '
        'you should bear in mind that past or targeted performance is not indicative of future results, and there '
        'can be no assurance that any investment will achieve comparable results or that target returns will be met. '
        'In addition, there can be no assurance that any investment will achieve or be realized at the valuations '
        'shown, as actual realized returns will depend on, among other factors, future operating results, the value '
        'of assets and market conditions at the time of disposition, any related transaction costs and the timing '
        'and manner of sale, all of which may differ from the assumptions on which the valuations contained in this '
        'Presentation are based. Nothing contained in this Presentation should be deemed to be a prediction or '
        'projection of future performance of any investment. Investors should make their own investigations and '
        'evaluations of a potential investment and the information contained in this Presentation. Except where '
        'otherwise indicated in this Presentation, the information provided in this Presentation is based on matters '
        'as they exist as of the date of preparation and not as of any future date, and will not be updated or '
        'otherwise revised to reflect information that subsequently becomes available, or circumstances existing or '
        'changes occurring after the date hereof.'
    ), 0.4, 0.9, 9.2, 4.3, size=10.5, color=RGBColor(0x33, 0x33, 0x33), wrap=True)

    # ── Slide 3: Analog Selection ──────────────────────────────────────────────
    sl = prs.slides.add_slide(_blank)
    _slide_title(sl, "Analog Selection Methodology"); _chrome(sl, 3)
    _kv_box(sl, "Subject Location", [
        ["Well",      wm['wellName']],
        ["County",    wm.get('county', '')],
        ["Reservoir", wm.get('reservoir', '')],
        ["AFE LL",    _ff(wm['afeLateralLenFt'])],
    ], 0.4, 0.85, 3.5)
    _kv_box(sl, "Analog Pool Criteria", [
        ["# of Analogs",            str(ac.get('nAnalogs', ''))],
        ["County",                  str(ac.get('counties', ''))],
        ["Reservoir",               str(ac.get('reservoir', ''))],
        ["LL Range",                str(ac.get('llRangeFt', ''))],
        ["LL Avg",                  _ff(ac['llAvgFt']) if ac.get('llAvgFt') else ''],
        ["Completion Vintage",      str(ac.get('compVintage', ''))],
        ["Avg Distance to Subject", str(ac.get('avgDistanceMi', 'N/A'))],
    ], 0.4, 2.25, 3.5)
    _rect(sl, 4.2, 0.85, 5.4, 4.2, RGBColor(0xCC, 0xCC, 0xCC), RGBColor(0x99, 0x99, 0x99))
    _txt(sl, "[ Insert analog location map here ]", 4.2, 2.5, 5.4, 0.6,
         size=11, color=RGBColor(0x66, 0x66, 0x66), italic=True, align='center')

    # ── Slide 4: Summary ───────────────────────────────────────────────────────
    sl = prs.slides.add_slide(_blank)
    _slide_title(sl, f"Summary – {wm['wellName']} TC"); _chrome(sl, 4)
    _txt(sl, "Analysis Parameters", 0.4, 0.85, 9, 0.3, size=12, bold=True, color=BODY_GRAY)
    _kv_box(sl, "Parameter", [
        ["Type Curve Name",      f"{wm['wellName']} TC"],
        ["Generated",            wm['generatedDate']],
        ["Normalization Length", _ff(wm['normLengthFt'])],
        ["Scaling Exponent (a)", f"{wm.get('scalingExponent', 1.0):.2f}"],
        ["Terminal Decline",      f"{wm.get('terminalDeclinePct', 5.0):.1f} %/yr eff."],
        ["B-factor Range",       wm.get('bFactorRange', '')],
        ["Decline Models",       wm.get('declineModels', '')],
    ], 0.4, 1.18, 4.5, rh=0.24)
    _txt(sl, "EUR Summary", 0.4, 3.15, 9, 0.3, size=12, bold=True, color=BODY_GRAY)

    def _eur_row(fd):
        return [_cap(fd['fluid']),
                f"{fd['nTcWells']} / {fd['nTotal']}",
                _fe(fd['p90Eur'], fd['eurUnit']),
                _fe(fd['p50Eur'], fd['eurUnit']),
                _fe(fd['p10Eur'], fd['eurUnit']),
                _fe(fd['meanEur'], fd['eurUnit'])]

    _table(sl, ["Fluid", "TC Wells", "P90 EUR", "P50 EUR", "P10 EUR", "Mean EUR"],
           [_eur_row(oil), _eur_row(gas), _eur_row(wtr)],
           0.4, 3.45, [1.1, 0.9, 1.65, 1.65, 1.65, 1.65])

    # ── Slides 5/8/11: Fluid stats + parameters ────────────────────────────────
    for fd, pg in [(oil, 5), (gas, 8), (wtr, 11)]:
        if not fd or fd['nTcWells'] == 0:
            continue
        sl = prs.slides.add_slide(_blank)
        _slide_title(sl, f"{_cap(fd['fluid'])} Analysis"); _chrome(sl, pg)
        _txt(sl, f"{_cap(fd['fluid'])} EUR Statistics ({fd['eurUnit']})",
             0.4, 0.85, 3.5, 0.3, size=11, bold=True, color=BODY_GRAY)
        _kv_box(sl, "Metric", [
            ["Wells in TC",  f"{fd['nTcWells']} / {fd['nTotal']}"],
            ["Mean",         f"{fd['meanEur']:.2f}"],
            ["Median (P50)", f"{fd['p50Eur']:.2f}"],
            ["P10",          f"{fd['p10Eur']:.2f}"],
            ["P90",          f"{fd['p90Eur']:.2f}"],
        ], 0.4, 1.17, 3.5)
        _txt(sl, "B-Factor Statistics", 0.4, 3.0, 3.5, 0.3, size=11, bold=True, color=BODY_GRAY)
        _kv_box(sl, "Metric", [
            ["Count",    str(fd['bCount'])],
            ["Mean",     f"{fd['bMean']:.3f}"],
            ["Median",   f"{fd['bMedian']:.3f}"],
            ["P10/P90",  f"{fd['bP10']:.3f} / {fd['bP90']:.3f}"],
        ], 0.4, 3.3, 3.5)
        _txt(sl, "Type Curve Parameters", 4.1, 0.85, 5.6, 0.3, size=11, bold=True, color=BODY_GRAY)
        _table(sl, ["Parameter", "P90 (Low)", "P50 (Mid)", "P10 (High)"], [
            [f"qi ({fd['qiUnit']})", _fn(fd['p90Qi']),  _fn(fd['p50Qi']),  _fn(fd['p10Qi'])],
            ["b-factor",             f"{fd['p90B']:.3f}", f"{fd['p50B']:.3f}", f"{fd['p10B']:.3f}"],
            ["Di (per month)",       f"{fd['p90Di']:.4f}", f"{fd['p50Di']:.4f}", f"{fd['p10Di']:.4f}"],
            ["1st-Year Decline",     _fp(fd['p90Decline1yr']), _fp(fd['p50Decline1yr']), _fp(fd['p10Decline1yr'])],
            ["Terminal Di",          *[f"{fd['terminalDi']:.5f}/mo ({effective_annual_from_nominal_monthly(fd['terminalDi'])*100:.1f}%/yr)"] * 3],
            [f"EUR ({fd['eurUnit']})", f"{fd['p90Eur']:.2f}", f"{fd['p50Eur']:.2f}", f"{fd['p10Eur']:.2f}"],
        ], 4.1, 1.17, [2.2, 1.1, 1.1, 1.2])
        _txt(sl, f"{_cap(fd['fluid'])} EURs", 4.1, 3.4, 5.6, 0.3, size=11, bold=True, color=BODY_GRAY)
        _table(sl, ["Percentile", f"EUR ({fd['eurUnit']})", f"EUR ({fd['eurPerFtUnit']})"], [
            ["P90",  f"{fd['p90Eur']:.2f}",  f"{fd['p90EurPerFt']:.2f}"],
            ["P50",  f"{fd['p50Eur']:.2f}",  f"{fd['p50EurPerFt']:.2f}"],
            ["P10",  f"{fd['p10Eur']:.2f}",  f"{fd['p10EurPerFt']:.2f}"],
            ["Mean", f"{fd['meanEur']:.2f}", f"{fd['meanEurPerFt']:.2f}"],
        ], 4.1, 3.72, [1.5, 2.05, 2.05])

    # ── Slides 6/9/12: Charts — paste exact Streamlit interface plots as images ──
    def _embed_bytes(sl, data_bytes, x, y, w, h):
        sl.shapes.add_picture(_BytesIO(data_bytes), Inches(x), Inches(y), Inches(w), Inches(h))

    for fd, pg in [(oil, 6), (gas, 9), (wtr, 12)]:
        if not fd or fd['nTcWells'] == 0:
            continue
        sl = prs.slides.add_slide(_blank)
        _slide_title(sl, f"{_cap(fd['fluid'])} Analysis – Charts"); _chrome(sl, pg)

        has_hist = 'bfactor_hist_bytes' in fd
        has_tc   = 'typecurve_bytes' in fd
        has_prob = 'probit_bytes' in fd

        if has_hist and has_tc:
            # Top row: b-factor histogram (left) + type curve (right)
            _txt(sl, "B-Factor Distribution", 0.4, 0.82, 4.5, 0.25, size=10, bold=True, color=BODY_GRAY)
            _embed_bytes(sl, fd['bfactor_hist_bytes'], 0.4, 1.08, 4.5, 2.1)
            _txt(sl, "Type Curve", 5.1, 0.82, 4.5, 0.25, size=10, bold=True, color=BODY_GRAY)
            _embed_bytes(sl, fd['typecurve_bytes'], 5.1, 1.08, 4.5, 2.1)
        elif has_hist:
            _txt(sl, "B-Factor Distribution", 0.4, 0.82, 9.2, 0.25, size=10, bold=True, color=BODY_GRAY)
            _embed_bytes(sl, fd['bfactor_hist_bytes'], 0.4, 1.08, 9.2, 2.1)
        elif has_tc:
            _txt(sl, "Type Curve", 0.4, 0.82, 9.2, 0.25, size=10, bold=True, color=BODY_GRAY)
            _embed_bytes(sl, fd['typecurve_bytes'], 0.4, 1.08, 9.2, 2.1)

        if has_prob:
            _txt(sl, "EUR Probit", 0.4, 3.25, 9.2, 0.25, size=10, bold=True, color=BODY_GRAY)
            _embed_bytes(sl, fd['probit_bytes'], 0.4, 3.5, 9.2, 1.8)

    # ── Slides 7/10/13: Analog well tables ─────────────────────────────────────
    for fd, pg in [(oil, 7), (gas, 10), (wtr, 13)]:
        if not fd or fd['nTcWells'] == 0:
            continue
        sl = prs.slides.add_slide(_blank)
        _slide_title(sl, f"{_cap(fd['fluid'])} Analysis – Analogs"); _chrome(sl, pg)
        rows_data = fd.get('analogRows', [])
        n = len(rows_data)
        rh = 0.225 if n > 15 else (0.255 if n > 10 else 0.31)
        _table(sl,
               ["API/UWI", "Well Name", "Lat Len (ft)",
                f"qi ({fd['qiUnit']})", "b", "Di (/mo)", "1yr Dec (%)", f"EUR ({fd['eurUnit']})"],
               [[str(r['api']), str(r['wellName']), f"{int(round(r['latLenFt'])):,}",
                 _fn(r['qi']), f"{r['b']:.4f}", f"{r['diPerMo']:.4f}",
                 f"{r['decline1yrPct']:.2f}", f"{r['eur']:.2f}"] for r in rows_data],
               0.4, 0.88, [1.35, 1.85, 1.0, 0.85, 0.8, 0.75, 0.9, 0.7], rh)

    # ── Save ───────────────────────────────────────────────────────────────────
    try:
        safe = "".join(c if c.isalnum() or c in ' _-' else '_' for c in wm['wellName'])
        out_path = os.path.join(output_dir, f"SE_{safe or 'TC'}_TC_Report_{wm['generatedDate']}.pptx")
        prs.save(out_path)
        return out_path
    except Exception as e:
        st.warning(f"PPTX save failed: {e}")
        return None

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
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#781F28'))
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
        ["Lateral Scaling Exponent (a)", f"{st.session_state.get('applied_norm_exponent', 1.0):.2f}"],
        ["Min Months History", str(int(min_hist))],
        ["Terminal Decline (%/yr eff.)", f"{float(term_dec_pct):.1f}"],
        ["Peak Volume Cap", f"{qi_cap_cfg:,.0f}" if qi_cap_cfg else "none"],
        ["Gas/Water as Yields (GOR/WOR)", "Yes" if yield_mode else "No"],
        ["B-factor Range", f"{b_low:.3f} - {b_high:.3f}"],
        ["Decline Models", _decline_models_label()],
        ["Vertical Dataset", "Yes" if vertical_wells else "No"],
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
    story.append(Spacer(1, 20))

    # ── Executive Summary (right after parameters) ──
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 8))

    exec_data = [["Fluid", "TC Wells", "P90 EUR", "P50 EUR", "P10 EUR", "Mean EUR"]]
    for _fl, _key, _ecol in [("Oil", "Oil_oneline", "EUR (Mbbl)"),
                               ("Gas", "Gas_oneline", "EUR (MMcf)"),
                               ("Water", "Water_oneline", "EUR (Mbbl water)")]:
        if _key not in st.session_state:
            continue
        _oneline_full = st.session_state[_key]
        _sel_key = f"{_fl}_tc_selection"
        if _sel_key in st.session_state and isinstance(st.session_state[_sel_key], set):
            _sel_ids = st.session_state[_sel_key]
            _oneline_tc = _oneline_full[_oneline_full['WellID'].astype(str).isin(_sel_ids)]
        else:
            _oneline_tc = _oneline_full
        _tc_count = len(_oneline_tc)
        if _ecol in _oneline_tc.columns:
            _eurs = pd.to_numeric(_oneline_tc[_ecol], errors="coerce").dropna().astype(float).tolist()
            if _eurs:
                _st = compute_eur_stats(_eurs)
                _unit = "MMcf" if _fl == "Gas" else "Mbbl"
                _p90k = next((k for k in _st if 'p90' in k.lower()), None)
                _p50k = next((k for k in _st if 'p50' in k.lower() or 'median' in k.lower()), None)
                _p10k = next((k for k in _st if 'p10' in k.lower()), None)
                _mnk  = next((k for k in _st if 'mean' in k.lower()), None)
                exec_data.append([
                    _fl, str(_tc_count),
                    f"{_st[_p90k]:.1f} {_unit}" if _p90k else "N/A",
                    f"{_st[_p50k]:.1f} {_unit}" if _p50k else "N/A",
                    f"{_st[_p10k]:.1f} {_unit}" if _p10k else "N/A",
                    f"{_st[_mnk]:.1f} {_unit}" if _mnk else "N/A",
                ])
    if len(exec_data) > 1:
        exec_tbl = Table(exec_data, colWidths=[0.8*inch, 0.8*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
        exec_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0),  colors.HexColor('#781F28')),
            ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.whitesmoke),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0),  10),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('GRID',          (0, 0), (-1, -1), 0.5, colors.black),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#F9ECEC')]),
        ]))
        story.append(exec_tbl)
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

        # Type curves — use the same curves already computed and scaled by _render_tw
        # so the PDF plot is guaranteed to match the UI plot exactly.
        _cached_curves = st.session_state.get(f"{fluid_name}_tc_curves", pd.DataFrame())
        _cached_lines  = st.session_state.get(f"{fluid_name}_tc_lines",  [])
        if _cached_curves.empty and monthly_sel is not None:
            # Fallback: recompute if session state was cleared (e.g. page reload before PDF)
            _cached_curves, _cached_lines = build_type_curves_and_lines(
                monthly_sel, fluid_name.lower(), oneline=oneline, d_lim=float(d_lim_cfg)
            )
        if not _cached_curves.empty:
            curves = _cached_curves
            lines  = _cached_lines
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
                _tdl = f"{float(d_lim_cfg):.5f}/mo ({float(term_dec_pct):.1f}%/yr eff.)"
                tc_data.append(["Terminal Di (d_lim)", _tdl, _tdl, _tdl])

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

        # ── Analog well oneline with Arps parameters ──
        story.append(PageBreak())
        story.append(Paragraph(f"{fluid_name} — Analog Wells & Arps Parameters", heading_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"The following {len(oneline)} wells were used to build the {fluid_name} type curve.",
            styles['Normal']))
        story.append(Spacer(1, 8))

        _well_cols_wanted = ['API10', 'WellName', 'LateralLength',
                              'qi (per day)', 'b', 'di (per month)', 'First-Year Decline (%)', eur_col]
        _well_cols = [c for c in _well_cols_wanted if c in oneline.columns]
        if _well_cols:
            _qi_unit = "Mcf/d" if fluid_name == "Gas" else "bbl/d"
            _hdr_map = {
                'API10': 'API/UWI', 'WellName': 'Well Name',
                'LateralLength': 'Lat Len (ft)',
                'qi (per day)': f'qi ({_qi_unit})', 'b': 'b',
                'di (per month)': 'Di (/mo)', 'First-Year Decline (%)': '1yr Dec (%)',
                eur_col: 'EUR'
            }
            _col_w = {
                'API10': 1.2*inch, 'WellName': 1.5*inch,
                'LateralLength': 0.7*inch, 'qi (per day)': 0.8*inch, 'b': 0.5*inch,
                'di (per month)': 0.6*inch, 'First-Year Decline (%)': 0.7*inch, eur_col: 0.7*inch
            }
            _headers = [_hdr_map.get(c, c) for c in _well_cols]
            _widths  = [_col_w.get(c, 0.9*inch) for c in _well_cols]

            _well_data = [_headers]
            for _, row in oneline[_well_cols].iterrows():
                _row_vals = []
                for c in _well_cols:
                    v = row[c]
                    if isinstance(v, float) and np.isfinite(v):
                        _row_vals.append(f"{v:,.2f}" if c not in ('b', 'di (per month)') else f"{v:.4f}")
                    else:
                        _row_vals.append(str(v) if pd.notna(v) else "")
                _well_data.append(_row_vals)

            _well_tbl = Table(_well_data, colWidths=_widths, repeatRows=1)
            _well_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, 0),  _phase_color_rgb(fluid_name)),
                ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.whitesmoke),
                ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0, 0), (-1, -1), 7),
                ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN',         (0, 1), (1, -1),  'LEFT'),
                ('TOPPADDING',    (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('GRID',          (0, 0), (-1, -1), 0.4, colors.black),
                ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
            ]))
            story.append(_well_tbl)
        story.append(Spacer(1, 10))

        # ── Outlier detection: EUR vs Lateral Length ──
        _out = _make_outlier_fig(oneline, eur_col, fluid_name)
        if _out is not None:
            _of, _uf, _lf, _na, _nr, _nh, _nl, _med, _unit = _out
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"{fluid_name} — EUR Outlier Detection", styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(Image(_save_fig(_of), width=6.5*inch, height=3.2*inch))
            story.append(Spacer(1, 8))
            # Summary stats bar
            _stat_data = [[
                Paragraph("<b>Active wells</b><br/>" + str(_na),                styles['Normal']),
                Paragraph(f"<b>Flagged</b><br/>{_nh} high / {_nl} low",         styles['Normal']),
                Paragraph(f"<b>Median EUR</b><br/>{_med:,.0f} {_unit}",        styles['Normal']),
                Paragraph(f"<b>Upper fence</b><br/>{_uf:,.0f} {_unit}",        styles['Normal']),
                Paragraph(f"<b>Lower fence</b><br/>{_lf:,.0f} {_unit}",        styles['Normal']),
            ]]
            _stat_tbl = Table(_stat_data, colWidths=[1.2*inch]*5)
            _stat_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#EFEFEF')),
                ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE',      (0, 0), (-1, -1), 8),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ]))
            story.append(_stat_tbl)
            story.append(Spacer(1, 10))

        story.append(PageBreak())

    # Add each fluid section
    add_fluid_section("Oil", "Oil_oneline", "Oil_monthly", "EUR (Mbbl)")
    add_fluid_section("Gas", "Gas_oneline", "Gas_monthly", "EUR (MMcf)")
    add_fluid_section("Water", "Water_oneline", "Water_monthly", "EUR (Mbbl water)")

    # Build PDF
    doc.build(story)
    return tmp.name

st.markdown("---")
st.header("Export Report")

_has_forecast = any(k in st.session_state for k in ["Oil_oneline", "Gas_oneline", "Water_oneline"])

if st.button("Generate PDF + PPTX Report", disabled=not _has_forecast):
    if not _has_forecast:
        st.error("Please run at least one forecast before generating a report.")
    else:
        st.session_state["_pdf_name_prompt"] = True

if not _has_forecast:
    st.info("Run at least one forecast (Oil, Gas, or Water) to enable report export.")

if st.session_state.get("_pdf_name_prompt"):
    tc_name = st.text_input(
        "Enter a name for this Type Curve report:",
        key="_pdf_tc_name",
        placeholder="e.g. Niobrara 2024 Q1",
    )
    if st.button("Confirm & Generate", key="_pdf_confirm"):
        try:
            name_clean = tc_name.strip()
            with st.spinner("Generating PDF report..."):
                pdf_path = generate_comprehensive_pdf(tc_name=name_clean)
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name_clean)
            pdf_file_name = f"{safe_name}.pdf" if safe_name else f"SE_Autoforecast_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(pdf_path, "rb") as _f:
                st.session_state["_pdf_bytes"]     = _f.read()
                st.session_state["_pdf_file_name"] = pdf_file_name

            with st.spinner("Generating PPTX report..."):
                pptx_data = _build_pptx_data(name_clean)
                pptx_path = generate_tc_pptx(pptx_data, tempfile.gettempdir())
            if pptx_path and os.path.exists(pptx_path):
                with open(pptx_path, "rb") as _f:
                    st.session_state["_pptx_bytes"]     = _f.read()
                    st.session_state["_pptx_file_name"] = os.path.basename(pptx_path)
                st.success("PDF and PPTX reports ready for download.")
            else:
                st.session_state.pop("_pptx_bytes", None)
                st.success("PDF report ready for download. (PPTX export failed — see warning above)")

            st.session_state["_pdf_name_prompt"] = False
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.exception(e)

# Persistent download buttons — survive rerenders via session state
_dl_col1, _dl_col2 = st.columns(2)
with _dl_col1:
    if st.session_state.get("_pdf_bytes"):
        st.download_button(
            label="Download PDF Report",
            data=st.session_state["_pdf_bytes"],
            file_name=st.session_state.get("_pdf_file_name", "SE_Report.pdf"),
            mime="application/pdf",
            key="_pdf_dl",
            use_container_width=True,
        )
with _dl_col2:
    if st.session_state.get("_pptx_bytes"):
        st.download_button(
            label="Download PPTX Report",
            data=st.session_state["_pptx_bytes"],
            file_name=st.session_state.get("_pptx_file_name", "SE_Report.pptx"),
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            key="_pptx_dl",
            use_container_width=True,
        )
