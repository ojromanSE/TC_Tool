# app.py — SE Oil & Gas Autoforecasting

import dataclasses
import os
from functools import partial
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
    modified_arps, modified_arps_cum, model_volumes, D_LIM_DEFAULT, DAYS_PER_MONTH, MODEL_REGISTRY,
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

def _clear_forecast_state(fluids) -> None:
    """Drop every cached result of the given fluids (forecast, config, hindcast,
    type-curve selection)."""
    for fl in fluids:
        for suffix in ("_oneline", "_monthly", "_well_fcs", "_cfg", "_hindcast", "_tc_selection",
                       "_tc_undo", "_tc_curves", "_tc_lines"):
            st.session_state.pop(f"{fl}{suffix}", None)


def _phase_color(fluid: str) -> str:
    return {'oil': 'green', 'gas': 'red', 'water': 'blue'}[fluid.lower()]

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
                # Forecasts made from the previous data no longer line up with
                # these months (yield mode would silently consume stale oil
                # forecasts) — start clean.
                _clear_forecast_state(["Oil", "Gas", "Water"])
                _exp_msg = (f" Scaling exponent a = {applied_exp:.2f}."
                            if use_norm and not vertical_wells else "")
                st.success(
                    f"Merged {len(merged['WellID'].unique())} wells and "
                    f"{len(merged)} monthly rows.{_exp_msg}"
                )
        except Exception as e:
            st.exception(e)

# One-click "Run All Phases" sits right under Load / QC / Merge.  Its code is
# defined further down the script, so reserve the spot here and fill it later.
_run_all_slot = st.container()

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


# ================= Type-curve selection state =================
# The wells in a fluid's type curve are the set st.session_state[f"{fluid}_tc_selection"].
# Every change goes through _set_tc_selection, which records the previous set
# for Undo, refreshes the analog checkbox table and retires the chart's click
# state (see _tc_chart_gen) so a consumed click can never be applied twice.
_TC_UNDO_DEPTH = 25

def _tc_chart_gen(fluid: str) -> int:
    return int(st.session_state.get(f"{fluid}_tc_chart_gen", 0))

def _bump_tc_chart_gen(fluid: str) -> None:
    """New key for the type-curve chart: its stored click selection is dropped."""
    st.session_state[f"{fluid}_tc_chart_gen"] = _tc_chart_gen(fluid) + 1

def _set_tc_selection(fluid: str, new_sel, msg: str | None = None) -> bool:
    """Replace a fluid's type-curve well set, keeping the old one for Undo.
    Returns False (and records nothing) when the set does not change."""
    sel_key = f"{fluid}_tc_selection"
    prev = set(st.session_state.get(sel_key, set()))
    new_sel = set(map(str, new_sel))
    if msg:
        _flash(f"{fluid}_tw_flash", msg)
    if new_sel == prev:
        return False
    undo = st.session_state.setdefault(f"{fluid}_tc_undo", [])
    undo.append(prev)
    del undo[:-_TC_UNDO_DEPTH]
    st.session_state[sel_key] = new_sel
    _bump_tc_chart_gen(fluid)
    _request_editor_reset(fluid, 'analog')
    return True

def _undo_tc_selection(fluid: str) -> None:
    stack = st.session_state.get(f"{fluid}_tc_undo") or []
    if not stack:
        return
    prev = stack.pop()
    st.session_state[f"{fluid}_tc_selection"] = set(prev)
    _bump_tc_chart_gen(fluid)
    _request_editor_reset(fluid, 'analog')
    _flash(f"{fluid}_tw_flash", f"{fluid}: undone — {len(prev)} wells in the Type Curve.")


# ================= Figure cache =================
# The analysis figures (b-factor, probit, per-well, outlier) only change when a
# forecast is re-run or another well is picked, but Streamlit re-executes the
# whole script on every interaction; redrawing ~18 matplotlib figures took
# most of each rerun.  Results are cached per slot in session state and
# reused while the source object is the SAME object (identity, so a re-run
# forecast, which creates new frames, always invalidates) and the extra
# parameters are equal.  ``source`` may be a tuple of objects, compared
# element-wise by identity.
def _same_source(a, b) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))
    return a is b

def _memo(slot: str, source, extra, build):
    cache = st.session_state.setdefault("_fig_memo", {})
    hit = cache.get(slot)
    if hit is not None and _same_source(hit[0], source) and hit[1] == extra:
        return hit[2]
    value = build()
    cache[slot] = (source, extra, value)
    return value

def _fig_png(fig, dpi: int = 200, max_px: int = 1400) -> bytes:
    """PNG bytes of a matplotlib figure (as st.pyplot draws it), figure closed.
    The resolution is capped so the image stays under Streamlit's 1460 px
    display limit: wider images are bilinear-resized and re-encoded by
    Streamlit on every run, which is slower and blurrier than drawing at the
    right size."""
    dpi = min(float(dpi), max_px / max(float(fig.get_figwidth()), 1e-6))
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def _png_file_bytes(path) -> bytes | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        return data
    except OSError:
        return None

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


def _as_fitted_cfg(cfg: ForecastConfig, primary_fcs) -> ForecastConfig:
    """The config to store with a fluid's results: yield_from_oil records what
    actually happened (False when no oil forecasts were available and the wells
    were fitted independently), so reports and the settings-mismatch warning
    describe the forecasts, not the checkbox."""
    return dataclasses.replace(cfg, yield_from_oil=bool(cfg.yield_from_oil and primary_fcs))


def _invalidate_yield_dependents() -> None:
    """After a new Oil forecast, Gas / Water results that were derived from the
    previous oil forecasts (yield mode) are stale: clear them and say so."""
    stale = [fl for fl in ("Gas", "Water")
             if getattr(st.session_state.get(f"{fl}_cfg"), "yield_from_oil", False)]
    if stale:
        _clear_forecast_state(stale)
        st.info(f"{' and '.join(stale)} forecasts were GOR/WOR yields of the previous Oil forecast "
                "and have been cleared — re-run them to rebuild from the new Oil forecast.")


def _fitted_cfg(fluid: str | None = None):
    """The ForecastConfig a fluid's cached forecasts were actually produced with
    (None if that fluid has not been forecast).  With fluid=None, the first
    forecast fluid's config."""
    fluids = [fluid] if fluid else ["Oil", "Gas", "Water"]
    for fl in fluids:
        cfg = st.session_state.get(f"{fl}_cfg")
        if cfg is not None:
            return cfg
    return None


def _fitted_d_lim(fluid: str | None = None) -> float:
    """Terminal decline the cached forecasts were fitted with — reports and type
    curves must use this, not the live sidebar value, which may have moved."""
    cfg = _fitted_cfg(fluid)
    return float(cfg.d_lim) if cfg is not None else float(d_lim_cfg)


def _fitted_term_dec_pct(fluid: str | None = None) -> float:
    return effective_annual_from_nominal_monthly(_fitted_d_lim(fluid)) * 100.0


def _yield_mode_label() -> str:
    """What the cached Gas / Water forecasts actually are (per fluid), falling
    back to the sidebar setting when neither has been forecast."""
    fitted = {fl: st.session_state.get(f"{fl}_cfg") for fl in ("Gas", "Water")}
    fitted = {fl: cfg for fl, cfg in fitted.items() if cfg is not None}
    if not fitted:
        return "Yes" if yield_mode else "No"
    parts = [f"{fl}: {'Yes' if cfg.yield_from_oil else 'No'}" for fl, cfg in fitted.items()]
    return ", ".join(parts)


def _settings_mismatch_warning(fluid: str) -> None:
    """Warn when the sidebar no longer matches the settings behind the cached forecasts."""
    cfg = _fitted_cfg(fluid)
    if cfg is None:
        return
    cur = _make_cfg(fluid)
    diffs = [name for name, a, b in [
        ("terminal decline", cfg.d_lim, cur.d_lim), ("b-factor bounds", (cfg.b_low, cfg.b_high), (cur.b_low, cur.b_high)),
        ("min history", cfg.min_months_history, cur.min_months_history),
        ("exponential model", cfg.allow_exponential, cur.allow_exponential),
        ("Duong gate", (cfg.exclude_duong, cfg.duong_min_months), (cur.exclude_duong, cur.duong_min_months)),
        ("peak cap", cfg.qi_cap, cur.qi_cap), ("yield mode", cfg.yield_from_oil, cur.yield_from_oil),
    ] if a != b]
    if diffs:
        st.warning(f"{fluid}: the sidebar ({', '.join(diffs)}) differs from the settings these forecasts "
                   "were made with — results and reports reflect the forecast-time settings "
                   "(terminal decline "
                   f"{effective_annual_from_nominal_monthly(cfg.d_lim)*100:.1f} %/yr). "
                   "Use 'Re-run All Phases & Update TCs' to apply the new settings.")


def _hindcast_fingerprint(fluid: str, hold, frac) -> str:
    cfg = _make_cfg(fluid)
    wells = sorted(st.session_state.merged['WellID'].astype(str).unique()) if "merged" in st.session_state else []
    return _selection_fingerprint([repr(cfg), str(hold), f"{frac:.3f}"] + list(wells))


# ================= Run all phases (one click) =================
_ALL_FLUIDS = ("Oil", "Gas", "Water")

def _run_all_phases(keep_selections: bool = True, progress=None) -> tuple:
    """Re-merge with the CURRENT sidebar settings (normalization length,
    lateral scaling exponent, b-factor bounds, terminal decline, min history)
    and forecast EVERY phase — Oil, then Gas, then Water (yield mode needs the
    Oil forecasts first) — so all onelines and type curves are consistent
    with the current parameters.

    For phases that were already forecast, analog/outlier selections are
    preserved (intersected with the new well set) so exclusion decisions
    survive a re-run; newly forecast phases start with every well selected.
    A phase with no production in the working set is skipped (its old
    results are cleared).  ``progress(i, fluid)`` is called before each phase.
    Returns (phases forecast, phases skipped)."""
    if "header_qc" not in st.session_state or "prod_df" not in st.session_state:
        return [], []

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
        return [], []

    ran, skipped = [], []
    oil_fcs = None
    for i, fl in enumerate(_ALL_FLUIDS):        # Oil first: yield mode needs its forecasts
        if progress is not None:
            progress(i, fl)
        cfg = _make_cfg(fl)
        primary = _primary_fcs_for(fl, oil_fcs)
        oneline, monthly, well_fcs = forecast_all(merged, cfg, primary_fcs=primary)
        if fl == "Oil":
            oil_fcs = well_fcs
        if oneline.empty or 'WellID' not in oneline.columns:
            _clear_forecast_state([fl])           # no production of this phase
            skipped.append(fl)
            continue
        sel_key = f"{fl}_tc_selection"
        had_selection = isinstance(st.session_state.get(sel_key), set)
        st.session_state[f"{fl}_cfg"] = _as_fitted_cfg(cfg, primary)
        st.session_state.pop(f"{fl}_hindcast", None)
        st.session_state.pop(f"{fl}_tc_undo", None)      # old sets refer to the old forecast
        st.session_state[f"{fl}_oneline"]  = oneline
        st.session_state[f"{fl}_monthly"]  = monthly
        st.session_state[f"{fl}_well_fcs"] = well_fcs
        new_ids = set(oneline['WellID'].astype(str))
        if keep_selections and had_selection:
            kept = st.session_state[sel_key] & new_ids
            st.session_state[sel_key] = kept if kept else new_ids
        else:
            st.session_state.pop(sel_key, None)
        # Reset the checkbox editors and the chart so stale edits / clicks
        # don't misapply if the well list changed (e.g. min-history raised).
        _request_editor_reset(fl)
        _bump_tc_chart_gen(fl)
        ran.append(fl)
    return ran, skipped


def _run_all_button(key: str):
    """Render the one-click 'run every phase' button; on click, forecast Oil,
    Gas and Water with the current settings and refresh every type curve."""
    if st.button("▶ Run All Phases & Update TCs", key=key, type="primary",
                 help="Forecasts Oil, Gas and Water in one go with the current b-factor bounds, "
                      "terminal decline, normalization length, lateral scaling and advanced fitting "
                      "options, then rebuilds every type curve. Analog/outlier exclusions on phases "
                      "already forecast are kept."):
        bar = st.progress(0.0, text="Preparing data…")

        def _progress(i, fl):
            bar.progress(i / len(_ALL_FLUIDS),
                         text=f"Forecasting {fl} ({i + 1}/{len(_ALL_FLUIDS)})…")
        try:
            ran, skipped = _run_all_phases(progress=_progress)
        except Exception as e:
            bar.empty()
            st.exception(e)
            return
        bar.empty()
        if not ran:
            st.warning("Nothing was forecast — check that the binning selection is not empty "
                       "and that the wells have production.")
            return
        msg = (f"Forecast {', '.join(ran)} with current settings — all type curves updated "
               f"(a = {st.session_state.get('applied_norm_exponent', 1.0):.2f}, "
               f"b ∈ [{float(b_low):.3f}, {float(b_high):.3f}], "
               f"min history = {int(min_hist)} mo).")
        if skipped:
            msg += f" Skipped {', '.join(skipped)}: no production in the working set."
        st.session_state["_global_rerun_msg"] = msg
        st.rerun()


if "merged_full" in st.session_state:
    with _run_all_slot:                       # rendered under the Load / QC / Merge button
        _grm = st.session_state.pop("_global_rerun_msg", None)
        if _grm:
            st.success(_grm)
            try:
                st.toast(_grm)
            except Exception:
                pass
        _run_all_button("rerun_all_top")
        st.caption("One click forecasts Oil, Gas and Water with the current settings and builds "
                   "every type curve. Click it again after changing b-factors, terminal decline, "
                   "normalization length, lateral scaling or the binning — exclusions are kept.")

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

    All background well histories are ONE trace, separated by gaps (NaN), with
    the WellID and well name as per-point customdata: the hover tooltip shows
    the name and a click reports the WellID.  One trace instead of one per
    well keeps the figure small and fast to build and send, which matters
    because the chart is redrawn after every click.  The trace uses invisible
    markers because plotly only selects points on traces that have markers.
    Highlighted wells get an orange overlay trace appended on top.  Returns
    (fig, n_background_wells)."""
    color = _phase_color(fluid)
    unit  = {"oil": "bbl/mo", "gas": "Mcf/mo", "water": "bbl/mo"}[fluid]
    highlight_ids = set(map(str, highlight_ids))
    fig, ymax = go.Figure(), 10.0
    xs, ys, cds = [], [], []
    for t, y, wid in lines:
        wid = str(wid)
        n = len(t)
        if n == 0:
            continue
        ymax = max(ymax, float(np.nanmax(y)))
        xs.append(np.asarray(t, dtype=float)); xs.append(np.array([np.nan]))
        ys.append(np.asarray(y, dtype=float)); ys.append(np.array([np.nan]))
        cds.extend([[wid, name_map.get(wid, wid)]] * n)
        cds.append([None, None])
    if xs:
        # customdata as a numpy object array: plotly copies it in one step
        # instead of validating a nested Python list element by element
        # (~25x faster for a few thousand points).
        fig.add_trace(go.Scatter(
            x=np.concatenate(xs), y=np.concatenate(ys), customdata=np.array(cds, dtype=object),
            mode='lines+markers', name='Wells', showlegend=False, connectgaps=False,
            line=dict(color='rgba(120,120,120,0.45)', width=1),
            # Markers are needed for click-selection but are invisible; plotly
            # hovers and selects on the data, not on drawn markers, so drawing
            # just one avoids thousands of SVG nodes (the browser-side cost of
            # every redraw grows with them).
            marker=dict(size=7, opacity=0, maxdisplayed=1),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
            hovertemplate=f"<b>%{{customdata[1]}}</b><br>Month %{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
        ))
    if not curves.empty:
        ymax = max(ymax, float(np.nanmax(curves[['P10', 'P50', 'P90']].to_numpy())))
        for col, dash, width in (('P10', 'dash', 2), ('P50', 'solid', 3), ('P90', 'dash', 2)):
            fig.add_trace(go.Scatter(
                x=curves['t'], y=curves[col], mode='lines', name=col,
                line=dict(color=color, dash=dash, width=width),
                hovertemplate=f"<b>{col}</b><br>Month %{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
            ))
        if 'Mean' in curves.columns:
            fig.add_trace(go.Scatter(
                x=curves['t'], y=curves['Mean'], mode='lines', name='Mean (avg well)',
                line=dict(color='black', dash='dot', width=1.5),
                hovertemplate=f"<b>Mean</b><br>Month %{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
            ))
    # Orange overlays for highlighted wells, appended last so they sit on top
    # (the grey base trace underneath keeps providing the hover tooltip).
    for t, y, wid in lines:
        if str(wid) in highlight_ids:
            fig.add_trace(go.Scatter(
                x=t, y=y, mode='lines', name=name_map.get(str(wid), str(wid)),
                showlegend=False, hoverinfo='skip',
                line=dict(color='#FF8C00', width=3),
            ))
    fig.update_layout(
        height=480, margin=dict(l=50, r=20, t=30, b=45), hovermode='closest',
        clickmode='event+select',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        xaxis_title='Month of Production',
        yaxis_title=f"Monthly {fluid} (norm {unit})",
    )
    fig.update_yaxes(type='log', range=[0, np.log10(ymax) + 0.1])
    return fig, len(xs) // 2


def _chart_clicked_wells(plot_key: str) -> set:
    """WellIDs of the well curves clicked on the chart, read from the chart
    widget's selection state (each point's customdata is [WellID, name];
    type-curve traces carry no customdata and are ignored)."""
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
        cd = get("customdata")
        wid = cd[0] if isinstance(cd, (list, tuple)) and cd else cd
        if wid not in (None, "", "None"):
            found.add(str(wid))
    return found

# ================= Fluid Block =================
def fluid_block(fluid_name: str, eur_col: str):
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
        primary = _primary_fcs_for(fluid_name)
        oneline, monthly, well_fcs = forecast_all(merged, cfg, primary_fcs=primary)
        st.session_state[f"{fluid_name}_cfg"] = _as_fitted_cfg(cfg, primary)   # settings these results reflect
        st.session_state.pop(f"{fluid_name}_hindcast", None)      # hindcast is for the old forecast
        st.session_state[f"{fluid_name}_oneline"] = oneline
        st.session_state[f"{fluid_name}_monthly"] = monthly
        st.session_state[f"{fluid_name}_well_fcs"] = well_fcs   # cache per-well forecasts
        # Reset analog selection so new forecast starts with all wells selected
        st.session_state.pop(f"{fluid_name}_tc_selection", None)
        st.session_state.pop(f"{fluid_name}_tc_undo", None)
        _bump_tc_chart_gen(fluid_name)
        if fluid_name == "Oil":
            _invalidate_yield_dependents()
        st.success(f"{fluid_name} forecast completed.")

    on_key = f"{fluid_name}_oneline"
    mo_key = f"{fluid_name}_monthly"

    if on_key in st.session_state:
        _settings_mismatch_warning(fluid_name)
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
        with tab1:
            _o = st.session_state[on_key]
            st.dataframe(_memo(f"{fluid_name}_oneline_fmt", _o, (), lambda: format_df_2dec(_o)),
                         use_container_width=True)
        with tab2:
            _m = st.session_state[mo_key]
            st.dataframe(_memo(f"{fluid_name}_monthly_fmt", _m, (), lambda: format_df_2dec(_m)),
                         use_container_width=True)
        with tab3:
            oneline = st.session_state[on_key]

            def _bfactor_pngs():
                h, bx, sc, stats = bfactor_analytics_figures(oneline, fluid_name, eur_col)
                return _png_file_bytes(h), _png_file_bytes(bx), _png_file_bytes(sc), stats
            hist_png, box_png, scatter_png, bstats = _memo(
                f"{fluid_name}_bfactor", oneline, (eur_col,), _bfactor_pngs)
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
                probit_png = _memo(
                    f"{fluid_name}_probit", oneline, (eur_col, int(norm_len)),
                    lambda: _fig_png(probit_plot(eurs, unit, f"{fluid_name} Probit",
                                                 _phase_color(fluid_name), norm_len=int(norm_len))))
                st.image(probit_png, use_column_width=True)

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
                    st.session_state[f"{fluid_name}_hindcast_fp"] = _hindcast_fingerprint(fluid_name, hc_hold, hc_frac)
            hcdf = st.session_state.get(f"{fluid_name}_hindcast")
            if hcdf is None:
                st.caption("Choose a hold-out and run the hindcast.")
            elif hcdf.empty:
                st.info("No wells have enough history for this hold-out (each needs at least "
                        "4 post-peak months in the training window).")
            else:
                if st.session_state.get(f"{fluid_name}_hindcast_fp") != _hindcast_fingerprint(fluid_name, hc_hold, hc_frac):
                    st.warning("These hindcast results were produced with different settings, hold-out "
                               "or working set than the current ones — re-run to refresh.")
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
                    "and the nominal decline D(t) and Arps b(t) estimated from the data against the "
                    "fitted model. D and b come from centred moving-window regressions (13- and "
                    "19-month windows) of ln q and 1/D; the log-log slope is a Bourdet derivative. "
                    "A half slope with b ≈ 2 is transient linear flow; b falling below ~1 with a "
                    "flattening D is boundary-dominated flow. b is a second derivative of noisy "
                    "monthly data, so read the trend of the interior points, not single values, and "
                    "discount the first and last few points (truncated windows)."
                )
                _ol = st.session_state[on_key]
                _dpick = st.selectbox(f"Well ({fluid_name} diagnostics)",
                                      _ol['WellID'].astype(str).tolist(), key=f"{fluid_name}_diag_pick")
                fcd = st.session_state.get(f"{fluid_name}_well_fcs", {}).get(_dpick)
                if fcd is None:
                    st.info("No cached forecast for this well — re-run the forecast.")
                elif fcd.get('model') not in MODEL_REGISTRY:
                    st.info("Diagnostics are shown for decline-model fits only; this well was "
                            f"forecast as {fcd.get('model')} off the oil forecast.")
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
            # Reuse cached forecast — no re-optimisation
            fc = st.session_state[fcs_key][pick_id]
        else:
            fc = forecast_one_well(wd, fluid_name.lower(), float(b_low), float(b_high), 600,
                                   d_lim=float(d_lim_cfg), allow_exponential=bool(allow_exp),
                                   exclude_duong=bool(vertical_wells),
                                   duong_min_months=int(duong_min), qi_cap=qi_cap_cfg)
        well_png = _memo(f"{fluid_name}_well_plot", fc, (str(pick_id),),
                         lambda: _fig_png(plot_one_well(wd, fc, fluid_name.lower())))
        st.image(well_png, use_column_width=True)

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

fluid_block("Oil", "EUR (Mbbl)")
st.markdown("---")
fluid_block("Gas", "EUR (MMcf)")
st.markdown("---")
fluid_block("Water", "EUR (Mbbl water)")
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

    def _outlier_png():
        res = _make_outlier_fig(oneline, eur_col, fluid)
        return None if res is None else (_fig_png(res[0]),) + tuple(res[1:])
    result = _memo(f"{fluid}_outlier_fig", oneline, (eur_col,), _outlier_png)
    if result is None:
        st.info(f"Not enough data for {fluid} outlier detection (need ≥ 3 wells with lateral length).")
        return

    png, upper_fence, lower_fence, n_active, n_removed, n_high, n_low, median_eur, unit = result
    st.image(png, use_column_width=True)

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
        _set_tc_selection(fluid, all_ids - (new_excl | other_excl))
        _request_editor_reset(fluid)
        _flash(f"{fluid}_outlier_flash",
               f"{fluid}: {len(new_excl)} outlier(s) excluded from the Type Curve "
               f"({len(st.session_state[sel_key])} / {len(all_ids)} wells remain). "
               "Scroll down to see the updated TC.")
        st.rerun()

    if st.button(f"Reset {fluid} selection (include all wells)", key=f"{fluid}_reset_outlier"):
        _set_tc_selection(fluid, all_ids)
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
    _run_all_button("rerun_all_tw")

_TC_CLICK_REMOVE    = "Removes the well from the type curve"
_TC_CLICK_HIGHLIGHT = "Highlights the well (exclude later)"


def _well_name_map(fluid: str, oneline_full: pd.DataFrame) -> dict:
    """WellID -> display name (duplicate well names disambiguated with the ID)."""
    def _build():
        names = oneline_full[['WellID', 'WellName']].astype(str)
        dup = names['WellName'].duplicated(keep=False)
        out = {}
        for wid, nm, d in zip(names['WellID'], names['WellName'], dup):
            nm = wid if (not nm or nm == 'nan') else nm
            out[wid] = f"{nm} ({wid})" if d else nm
        return out
    return _memo(f"{fluid}_tc_names", oneline_full, (), _build)


def _tc_build(fluid: str, oneline_full: pd.DataFrame, mo_key: str, sel_ids: set, eur_col: str):
    """Type-curve inputs for the selected wells: (oneline of the selected wells,
    EUR stats, P10/P50/P90/Mean curves scaled to the EUR percentiles, well
    history lines).  Cached until the forecast, the well set or the terminal
    decline changes, so redraws that do not change the well set (highlighting,
    switching the click mode) cost nothing here."""
    monthly_full = st.session_state.get(mo_key)
    d_lim = _fitted_d_lim(fluid)

    def _build():
        ol = oneline_full[oneline_full['WellID'].astype(str).isin(sel_ids)].reset_index(drop=True)
        eur_stats = {}
        if len(ol) and eur_col in ol.columns:
            eur_stats = compute_eur_stats(pd.to_numeric(ol[eur_col], errors='coerce').dropna().tolist())
        curves, lines = pd.DataFrame(), []
        if monthly_full is not None and len(ol):
            mf = monthly_full[monthly_full['WellID'].astype(str).isin(sel_ids)]
            curves, lines = build_type_curves_and_lines(mf, fluid.lower(), oneline=ol, d_lim=d_lim)
            # Scale every TC curve so its cumulative volume exactly matches the
            # corresponding EUR percentile from the stats table (curves are in
            # bbl or Mcf, EUR stats in Mbbl or MMcf -> x 1000).
            if not curves.empty:
                curves = curves.copy()
                for pct in ('P10', 'P50', 'P90', 'Mean'):
                    if pct in curves.columns and eur_stats.get(pct) and np.isfinite(eur_stats[pct]):
                        csum = curves[pct].sum()
                        if csum > 0:
                            curves[pct] = curves[pct] * (eur_stats[pct] * 1000 / csum)
        return ol, eur_stats, curves, lines

    return _memo(f"{fluid}_tc_build", oneline_full,
                 (id(monthly_full), frozenset(sel_ids), float(d_lim), eur_col), _build)


# ---- Callbacks: they run BEFORE the rerun a click triggers, so the rerun
# ---- already draws the updated type curve (no second st.rerun needed).
def _cb_tc_chart_click(fluid: str, plot_key: str, hl_key: str, name_map: dict, n_tot: int) -> None:
    """A curve was clicked on the type-curve chart."""
    sel = set(st.session_state.get(f"{fluid}_tc_selection", set()))
    clicked = _chart_clicked_wells(plot_key) & sel
    _bump_tc_chart_gen(fluid)            # consume the click: the redrawn chart starts unselected
    if not clicked:
        return
    picked = [w for w in st.session_state.get(hl_key, []) if w in sel]
    if st.session_state.get(f"{fluid}_tc_click_mode", _TC_CLICK_REMOVE) == _TC_CLICK_REMOVE:
        remain = sel - clicked
        names = ", ".join(sorted(name_map.get(w, w) for w in clicked))
        _set_tc_selection(fluid, remain,
                          msg=f"{fluid}: removed {names} from the Type Curve ({len(remain)} / {n_tot} "
                              "wells remain). Click ↶ Undo to put it back.")
        st.session_state[hl_key] = [w for w in picked if w not in clicked]
    else:
        # Highlight mode toggles: clicking a highlighted well un-highlights it.
        st.session_state[hl_key] = [w for w in picked if w not in clicked] + sorted(
            clicked - set(picked), key=lambda w: name_map.get(w, w).lower())


def _cb_tc_exclude_highlighted(fluid: str, hl_key: str, n_tot: int) -> None:
    sel = set(st.session_state.get(f"{fluid}_tc_selection", set()))
    picked = set(st.session_state.get(hl_key, [])) & sel
    st.session_state[hl_key] = []
    if picked:
        remain = sel - picked
        _set_tc_selection(fluid, remain,
                          msg=f"{fluid}: excluded {len(picked)} well(s) from the Type Curve "
                              f"({len(remain)} / {n_tot} remain).")


def _cb_tc_clear_highlights(fluid: str, hl_key: str) -> None:
    st.session_state[hl_key] = []
    _bump_tc_chart_gen(fluid)


def _cb_tc_analog(fluid: str, action: str) -> None:
    """Apply / Select all / Deselect all in the analog-well table."""
    ed_key, ids, flags = st.session_state.get(f"{fluid}_analog_rows", (None, [], []))
    if not ids:                                   # table not drawn yet: nothing to apply
        return
    if action == "all":
        new = set(ids)
    elif action == "none":
        new = set()
    else:
        flags = list(flags)
        edits = st.session_state.get(ed_key) if ed_key else None
        rows = edits.get("edited_rows", {}) if isinstance(edits, dict) else {}
        for r, change in (rows or {}).items():
            r = int(r)
            if 0 <= r < len(flags) and isinstance(change, dict) and "Select" in change:
                flags[r] = bool(change["Select"])
        new = {w for w, f in zip(ids, flags) if f}
    _set_tc_selection(fluid, new, msg=f"{fluid}: {len(new)} / {len(ids)} wells selected for the Type Curve.")
    _request_editor_reset(fluid, 'analog')       # redraw the table from the applied set


def _render_tw(fluid, on_key, mo_key, eur_col):
    if on_key not in st.session_state:
        st.info(f"Run {fluid} first.")
        st.session_state.pop(f"{fluid}_tc_curves", None)
        st.session_state.pop(f"{fluid}_tc_lines", None)
        return

    _show_flash(f"{fluid}_tw_flash")
    _settings_mismatch_warning(fluid)
    oneline_full = st.session_state[on_key]

    # --- Analog selection state: set of included WellIDs ---
    sel_key = f"{fluid}_tc_selection"
    ids_order = oneline_full['WellID'].astype(str).tolist()
    all_ids = set(ids_order)
    if sel_key not in st.session_state:
        st.session_state[sel_key] = all_ids.copy()
    # Drop IDs that no longer exist (e.g. after a re-forecast changed the well list)
    st.session_state[sel_key] = set(st.session_state[sel_key]) & all_ids
    sel_ids = set(st.session_state[sel_key])
    n_tot = len(all_ids)
    name_map = _well_name_map(fluid, oneline_full)
    by_name = lambda w: name_map.get(w, w).lower()

    # Build display columns available in the oneline
    show_cols = [c for c in [
        'WellName', 'PrimaryFormation', 'LateralLength', eur_col
    ] if c in oneline_full.columns]

    editor_df = oneline_full[show_cols].copy().reset_index(drop=True)
    editor_df.insert(0, "Select", [w in sel_ids for w in ids_order])

    col_cfg = {"Select": st.column_config.CheckboxColumn("Select", default=True)}
    for c in show_cols:
        col_cfg[c] = st.column_config.TextColumn(c, disabled=True)

    # ── Two-column layout: table left, plot right ──
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.caption(f"**Analog Wells** — tick/untick, then **Apply selection**; nothing "
                   f"recomputes until you do ({n_tot} wells total).")
        ed_key = _editor_key(fluid, 'analog', 'analog_editor')
        # Row order / initial ticks the Apply callback maps the edits onto.
        st.session_state[f"{fluid}_analog_rows"] = (ed_key, ids_order, [w in sel_ids for w in ids_order])
        # The data editor sits inside a form so ticking boxes does not rerun the app.
        with st.form(key=f"{fluid}_analog_form", border=False):
            st.data_editor(
                editor_df,
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                key=ed_key,
            )
            fb1, fb2, fb3 = st.columns(3)
            fb1.form_submit_button("Apply selection", type="primary", use_container_width=True,
                                   on_click=_cb_tc_analog, args=(fluid, "apply"))
            fb2.form_submit_button("Select all", use_container_width=True,
                                   on_click=_cb_tc_analog, args=(fluid, "all"))
            fb3.form_submit_button("Deselect all", use_container_width=True,
                                   on_click=_cb_tc_analog, args=(fluid, "none"))

    oneline_filtered, eur_stats, curves, lines = _tc_build(fluid, oneline_full, mo_key, sel_ids, eur_col)
    n_sel = len(oneline_filtered)
    if n_sel > 0 and mo_key in st.session_state:
        st.session_state[f"{fluid}_tc_curves"] = curves
        st.session_state[f"{fluid}_tc_lines"] = lines
    else:                                   # nothing selected: no stale curves in exports
        st.session_state.pop(f"{fluid}_tc_curves", None)
        st.session_state.pop(f"{fluid}_tc_lines", None)

    with right_col:
        if mo_key in st.session_state and n_sel > 0:
            if _HAS_PLOTLY:
                hl_key   = f"{fluid}_tc_highlight"
                # The chart's key changes whenever a click has been consumed or the
                # well set changed, so a click is applied exactly once.
                plot_key = f"{fluid}_tc_plot_g{_tc_chart_gen(fluid)}"
                # Keep only highlighted wells that are still in the type curve
                # (set before the picker is drawn, which Streamlit allows).
                st.session_state[hl_key] = [w for w in st.session_state.get(hl_key, []) if w in sel_ids]
                mode = st.radio("Clicking a well curve", [_TC_CLICK_REMOVE, _TC_CLICK_HIGHLIGHT],
                                horizontal=True, key=f"{fluid}_tc_click_mode")
                hl_ids = set(st.multiselect(
                    "Highlighted wells (pick by name, or click curves in highlight mode)",
                    options=sorted(sel_ids, key=by_name),
                    format_func=lambda w: name_map.get(w, w),
                    key=hl_key,
                    placeholder="Pick wells to highlight…",
                ))
                # Reused while the curves and highlights are unchanged (e.g. for the
                # other phases' tabs when a well is clicked in this one).
                fig = _memo(f"{fluid}_tc_fig", (curves, lines, name_map), frozenset(hl_ids),
                            lambda: plot_type_curves_interactive(curves, lines, fluid.lower(),
                                                                 name_map, hl_ids)[0])
                st.plotly_chart(fig, use_container_width=True, key=plot_key,
                                on_select=partial(_cb_tc_chart_click, fluid, plot_key, hl_key,
                                                  name_map, n_tot),
                                selection_mode="points")
                if mode == _TC_CLICK_REMOVE:
                    st.caption("Hover a curve to see its well. **Click a curve to remove that well "
                               "from the type curve** — ↶ Undo puts it back.")
                else:
                    st.caption("Hover a curve to see its well. Click curves to highlight them, then "
                               "exclude the highlighted wells together.")
                if hl_ids:
                    st.markdown("**Highlighted:** " + ", ".join(
                        name_map.get(w, w) for w in sorted(hl_ids, key=by_name)))
                undo_stack = st.session_state.get(f"{fluid}_tc_undo") or []
                bx1, bx2, bx3 = st.columns([2, 1, 1])
                bx1.button(f"Exclude {len(hl_ids)} highlighted well(s) from Type Curve" if hl_ids
                           else "Exclude highlighted wells from Type Curve",
                           key=f"{fluid}_excl_highlighted", type="primary",
                           disabled=not hl_ids, use_container_width=True,
                           on_click=_cb_tc_exclude_highlighted, args=(fluid, hl_key, n_tot))
                bx2.button("Clear highlights", key=f"{fluid}_hl_clear_btn",
                           disabled=not hl_ids, use_container_width=True,
                           on_click=_cb_tc_clear_highlights, args=(fluid, hl_key))
                bx3.button("↶ Undo", key=f"{fluid}_tc_undo_btn", disabled=not undo_stack,
                           use_container_width=True, on_click=_undo_tc_selection, args=(fluid,),
                           help="Restore the type-curve well set as it was before the last change.")
            else:
                st.pyplot(plot_type_curves(curves, lines, fluid.lower()))
                st.caption("Install `plotly` for an interactive chart with hover names "
                           "and click-to-remove.")
        excluded = sorted((name_map.get(w, w) for w in all_ids - sel_ids), key=str.lower)
        if excluded:
            with st.expander(f"Excluded from the type curve ({len(excluded)})"):
                st.caption(", ".join(excluded))
                st.caption("Re-include wells by ticking them in the analog table, with ↶ Undo, "
                           "or with **Select all**.")

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


def _all_tc_excel(oil_c, gas_c, water_c) -> bytes | None:
    """Multi-phase TC monthly volumes workbook (None when nothing to export)."""
    _oil_c   = oil_c   if oil_c   is not None else pd.DataFrame()
    _gas_c   = gas_c   if gas_c   is not None else pd.DataFrame()
    _water_c = water_c if water_c is not None else pd.DataFrame()
    if _oil_c.empty and _gas_c.empty and _water_c.empty:
        return None
    import io as _io
    # Combined Oil + Gas sheet (P50 columns adjacent, units converted)
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

    # openpyxl's write-only mode: ~3x faster than pandas' ExcelWriter, which
    # matters because the workbook is rebuilt whenever a type curve changes.
    from openpyxl import Workbook
    wb = Workbook(write_only=True)
    sheets = [("Oil & Gas TC", _og_df)]
    if len(_water_parts) > 1:
        sheets.append(("Water TC", _water_df))
    for name, frame in sheets:
        ws = wb.create_sheet(name)
        ws.append([str(c) for c in frame.columns])
        for row in frame.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and not np.isfinite(v)) else
                       (v.item() if hasattr(v, "item") else v) for v in row])
    _xls_buf = _io.BytesIO()
    wb.save(_xls_buf)
    return _xls_buf.getvalue()


def _aries_phase_groups(fluid: str, eur_col: str):
    """Type wells (P10 / P50 / P90 / Mean: qi per day, b, Di) of one phase for
    the current type-curve selection — the same parameters the charts and the
    report use — plus the EUR stats, or None when the phase was not forecast."""
    import se_report
    ol = st.session_state.get(f"{fluid}_oneline")
    if ol is None or ol.empty:
        return None
    cfg = _fitted_cfg(fluid)
    b_lo, b_hi = (cfg.b_low, cfg.b_high) if cfg is not None else (float(b_low), float(b_high))
    sel = st.session_state.get(f"{fluid}_tc_selection")
    fr = se_report.fluid_report(fluid, ol, sel if isinstance(sel, set) else None, eur_col, None, None,
                                _fitted_d_lim(fluid), float(norm_len), bool(use_norm), b_lo, b_hi)
    return fr if (fr is not None and fr.n_tc > 0 and fr.groups) else None


def _render_aries_export():
    """Copy-ready ARIES economic lines of the type wells."""
    import aries_export as AX
    fluids = (("Oil", "EUR (Mbbl)"), ("Gas", "EUR (MMcf)"), ("Water", "EUR (Mbbl water)"))
    if not any(f"{fl}_oneline" in st.session_state for fl, _ in fluids):
        return
    st.markdown("**ARIES export** — the type wells as economic lines to paste into an ARIES case.")
    c1, c2, c3 = st.columns([2, 2, 1])
    tc_id = c1.text_input("Type curve name", value=(st.session_state.get("_pdf_tc_name") or "TC").strip() or "TC",
                          key="_aries_tc_id")
    pcts = c2.multiselect("Type wells", ["P10", "P50", "P90", "Mean"], default=["P50"], key="_aries_pcts",
                          help="One block of economic lines per type well.")
    qualifier = c3.text_input("Qualifier", value=AX.DEFAULT_QUALIFIER, key="_aries_qualifier",
                              help="Written in the Qualifier column of every line.").strip()
    if st.button("Export ARIES lines", type="primary", key="_aries_btn"):
        st.session_state["_aries_show"] = True
    if not st.session_state.get("_aries_show"):
        return
    if not pcts:
        st.warning("Pick at least one type well (P10, P50, P90 or Mean).")
        return
    reports = {fl: _aries_phase_groups(fl, ec) for fl, ec in fluids}
    reports = {fl: r for fl, r in reports.items() if r is not None}
    if not reports:
        st.warning("No type curve to export — select at least one well in a type curve.")
        return
    phases = {fl: r.groups for fl, r in reports.items()}
    d_lims = {fl: r.terminal_di for fl, r in reports.items()}
    prim = next(iter(reports.values()))
    if use_norm:
        basis = f"per well normalized to a {int(float(norm_len)):,} ft lateral"
    else:
        lat = pd.to_numeric(prim.wells.loc[prim.wells["in_tc"], "LateralLength"], errors="coerce").dropna()
        basis = ("per well, not normalized" +
                 (f" (the type-curve wells average a {int(round(lat.mean(), -1)):,} ft lateral)" if len(lat) else ""))
    tc_id = tc_id.strip() or "TC"

    for pct in pcts:
        st.markdown(f"**{tc_id} {pct}**")
        st.code(AX.economic_lines(phases, d_lims, pct, qualifier), language=None)
    st.caption("Each value is its own tab-separated cell, in the order of the ARIES grid: "
               f"{', '.join(AX.ECONOMIC_COLUMNS)}. Copy with the button at the top right of the box, "
               "click the Keyword cell of the first empty line in the Production section and paste. "
               f"Rates are per month (B/M, M/M), {basis}.")

    # Check: the volume ARIES computes from the exported (rounded) cells against the app's EUR.
    check = []
    for fl, r in reports.items():
        for pct in pcts:
            g = r.groups.get(pct)
            if g is None:
                continue
            cells = AX.segments(g["qi"], g["b"], g["di"], r.terminal_di).fields()
            vol = AX.segments_volume(cells) / 1000.0
            check.append({"Type well": pct, "Phase": fl, "EUR in the app": round(g["eur"], 2),
                          "EUR from the ARIES lines": round(vol, 2),
                          "Difference (%)": round((vol / g["eur"] - 1) * 100, 3) if g["eur"] else np.nan,
                          "Unit": r.unit})
    if check:
        st.dataframe(pd.DataFrame(check), hide_index=True, use_container_width=True)


@st.fragment
def _type_wells_section():
    """Everything that depends on the type-curve well sets.  As a fragment, a
    chart click, a highlight or an analog-table Apply reruns only this section
    (not the forecast blocks, outlier plots and reports above), and the
    multi-phase download below always matches the curves on screen."""
    tw_tabs = st.tabs(["Oil", "Gas", "Water"])
    with tw_tabs[0]: _render_tw("Oil",   "Oil_oneline",   "Oil_monthly",   "EUR (Mbbl)")
    with tw_tabs[1]: _render_tw("Gas",   "Gas_oneline",   "Gas_monthly",   "EUR (MMcf)")
    with tw_tabs[2]: _render_tw("Water", "Water_oneline", "Water_monthly", "EUR (Mbbl water)")

    # ── Multi-phase TC download ──
    curves = tuple(st.session_state.get(f"{fl}_tc_curves") for fl in _ALL_FLUIDS)
    xlsx = _memo("all_tc_xlsx", curves, (), lambda: _all_tc_excel(*curves))
    if xlsx is not None:
        st.download_button(
            label="⬇ Download All Phases TC Monthly Volumes (Excel)",
            data=xlsx,
            file_name="TC_Monthly_Volumes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="all_tc_download",
        )

    # ── ARIES export (same type wells as the charts above) ──
    st.markdown("---")
    _render_aries_export()


_type_wells_section()


# ================= REPORT EXPORT (SE Deck Standard) =================
# The PPTX and the PDF are the same SE-standard deck (se_report.py): the PPTX is
# built on the SE template layouts with native charts, the PDF draws the same
# slides.  This section only gathers the forecast data.
from datetime import datetime
import se_report

REPORT_DISCLAIMER = 'The information contained in this confidential presentation (this “Presentation”) is provided for informational and discussion purposes only and is not, and may not be relied on in any manner as, legal, tax or investment advice or as an offer to sell or a solicitation of an offer to buy an interest in any security. The information contained in this Presentation must be kept strictly confidential and may not be reproduced or redistributed in any format without the approval of Schaper Energy Consulting (“SE”). In considering any performance data contained in this Presentation, you should bear in mind that past or targeted performance is not indicative of future results, and there can be no assurance that any investment will achieve comparable results or that target returns will be met. In addition, there can be no assurance that any investment will achieve or be realized at the valuations shown, as actual realized returns will depend on, among other factors, future operating results, the value of assets and market conditions at the time of disposition, any related transaction costs and the timing and manner of sale, all of which may differ from the assumptions on which the valuations contained in this Presentation are based. Nothing contained in this Presentation should be deemed to be a prediction or projection of future performance of any investment. Investors should make their own investigations and evaluations of a potential investment and the information contained in this Presentation. Except where otherwise indicated in this Presentation, the information provided in this Presentation is based on matters as they exist as of the date of preparation and not as of any future date, and will not be updated or otherwise revised to reflect information that subsequently becomes available, or circumstances existing or changes occurring after the date hereof.'


def _report_input(tc_name: str, source: str) -> "se_report.ReportInput":
    """Everything the report shows, from the cached forecasts and type-curve
    selections (the same numbers the app displays)."""
    fluids = []
    for fl, eur_col in (("Oil", "EUR (Mbbl)"), ("Gas", "EUR (MMcf)"), ("Water", "EUR (Mbbl water)")):
        ol = st.session_state.get(f"{fl}_oneline")
        if ol is None or ol.empty:
            continue
        cfg = _fitted_cfg(fl)
        b_lo, b_hi = (cfg.b_low, cfg.b_high) if cfg is not None else (float(b_low), float(b_high))
        sel = st.session_state.get(f"{fl}_tc_selection")
        models = ol.get("Model", pd.Series(dtype=str)).astype(str)
        yl = next((m for m in models if m.startswith("Yield")), "")
        fr = se_report.fluid_report(
            fl, ol, sel if isinstance(sel, set) else None, eur_col,
            st.session_state.get(f"{fl}_tc_curves"), st.session_state.get(f"{fl}_tc_lines"),
            _fitted_d_lim(fl), float(norm_len), bool(use_norm), b_lo, b_hi, yield_label=yl)
        if fr is not None:
            fluids.append(fr)
    prim_ol = next((st.session_state[k] for k in ("Oil_oneline", "Gas_oneline", "Water_oneline")
                    if k in st.session_state), pd.DataFrame())
    an = se_report.analog_summary(fluids, prim_ol)
    as_of = ""
    merged_now = st.session_state.get("merged")
    if merged_now is not None and "MonthYear" in merged_now:
        last = pd.Series(merged_now["MonthYear"]).dropna()
        if len(last):
            last = last.max()
            as_of = (last.to_timestamp() if hasattr(last, "to_timestamp") else pd.Timestamp(last)).strftime("%b %Y")
    cfg0 = _fitted_cfg()
    b_lo, b_hi = (cfg0.b_low, cfg0.b_high) if cfg0 is not None else (float(b_low), float(b_high))
    exp_a = float(st.session_state.get("applied_norm_exponent", 1.0))
    settings = [
        ("Normalization length", f"{int(norm_len):,} ft" if use_norm else "Off (volumes as reported)"),
        ("Lateral scaling exponent a", f"{exp_a:.2f}"),
        ("Decline models", _decline_models_label()),
        ("b-factor range", f"{b_lo:.2f}–{b_hi:.2f}"),
        ("Terminal decline", f"{_fitted_term_dec_pct():.1f}%/yr effective"),
        ("Minimum history", f"{int(cfg0.min_months_history if cfg0 is not None else min_hist)} months"),
        ("Gas and water as yields", _yield_mode_label()),
        ("Peak volume cap", f"{qi_cap_cfg:,.0f} per month" if qi_cap_cfg else "None"),
        ("Vertical well dataset", "Yes" if vertical_wells else "No"),
        ("Forecast horizon", "50 years (600 months)"),
    ]
    return se_report.ReportInput(
        tc_name=tc_name, source=source or "Production data", as_of=as_of, generated=datetime.now().date(),
        counties=an["counties"], state=an["state"], formations=an["formations"], vintage=an["vintage"],
        lateral_min=an["lateral_min"], lateral_max=an["lateral_max"], lateral_mean=an["lateral_mean"],
        months_median=an["months_median"], norm_len=float(norm_len), use_norm=bool(use_norm), exponent=exp_a,
        settings=settings, decline_models=se_report.join_and(_decline_models_label().split(", ")),
        fluids=fluids, disclaimer=REPORT_DISCLAIMER)


def generate_reports(tc_name: str, source: str) -> dict:
    """Build the SE deck once and write it as PPTX and PDF; returns the bytes
    and file names (same stem, SE section 12)."""
    deck = se_report.build_deck(_report_input(tc_name, source))
    with tempfile.TemporaryDirectory() as td:
        pptx_path = se_report.render_pptx(deck, os.path.join(td, deck.file_stem + ".pptx"))
        pdf_path = se_report.render_pdf(deck, os.path.join(td, deck.file_stem + ".pdf"))
        with open(pptx_path, "rb") as f:
            pptx_bytes = f.read()
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    return dict(pptx=pptx_bytes, pdf=pdf_bytes, stem=deck.file_stem, n_slides=len(deck.slides))


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
    st.caption("The PowerPoint and the PDF follow the SE Deck Standard: the SE template, cover, "
               "footer with the project and the data date, native charts, and the same slides in both files.")
    tc_name = st.text_input(
        "Enter a name for this Type Curve report:",
        key="_pdf_tc_name",
        placeholder="e.g. Niobrara 2024 Q1",
    )
    data_source = st.text_input(
        "Data source for the slide footer:",
        value="Production data", key="_pdf_source",
        help="The footer shows it with the month of the latest production, e.g. "
             "'Enverus production data through Sep 2025'.",
    )
    if st.button("Confirm & Generate", key="_pdf_confirm"):
        try:
            with st.spinner("Building the SE-standard deck (PPTX and PDF)..."):
                rep = generate_reports(tc_name.strip(), data_source.strip())
            st.session_state["_pdf_bytes"] = rep["pdf"]
            st.session_state["_pdf_file_name"] = rep["stem"] + ".pdf"
            st.session_state["_pptx_bytes"] = rep["pptx"]
            st.session_state["_pptx_file_name"] = rep["stem"] + ".pptx"
            st.success(f"PDF and PPTX reports ready for download ({rep['n_slides']} slides).")
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
