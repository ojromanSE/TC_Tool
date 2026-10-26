"""SE-standard type-curve report: one deck model, two renderers.

The report follows the Schaper Energy (SE) Deck Standard (SE Deck Kit,
version 1): the SE template layouts (COVER, CONTENT, SECTION), the SE colours
and type scale (Cambria for titles and big numbers, Calibri for everything
else), the 13.33 x 7.5 in grid, the footer with the project and the data
source, KPI cards, bullet panels with bold leads, maroon-header tables and
native charts.

    deck = build_deck(report_input)          # pure data: slides and blocks
    render_pptx(deck, "out.pptx")            # PowerPoint on the SE template, native charts
    render_pdf(deck, "out.pdf")              # the same slides as a PDF

Both renderers draw the same blocks at the same positions with the same text,
so the PDF is the deck exported (SE standard: "export a PDF with the same
name").  Text sizes are fitted once, in build_deck, with font metrics of
Calibri / Cambria (or the metric-compatible Carlito / Caladea).
"""
from __future__ import annotations

import copy
import datetime as _dt
import io
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "static", "se_deck")
TEMPLATE = os.path.join(ASSET_DIR, "SE_Deck_Base.pptx")
LOGO = os.path.join(ASSET_DIR, "logo.jpg")
LOGO_COVER = os.path.join(ASSET_DIR, "logo_cover.png")
COMPANY = "Schaper Energy Consulting"

# ---------------------------------------------------------------- design tokens (SE Deck Kit)
C = dict(MAROON="5A0617", INK="2B2B2B", GREY="6E6E73", MID="9C9599", BLUE="2F6FBF", AMBER="B8860B",
         GREEN="2E8B2E", RED="C00000", CARD="F5F3F2", LINE="E6DFDE", GRID="E4E3DC", NEUTRAL="C9C7BD",
         WHITE="FFFFFF", AXIS="BFBFBF", COVER_EYEBROW="C9A7AD", COVER_SUB="E7CCD0")
TINT = dict(MAROON="EDE7E9", BLUE="E7EEF8", GREEN="E7F1E7", AMBER="F3ECDD", RED="F6E9E9")
SIZE = dict(coverTitle=44, coverEyebrow=13.5, coverSub=18, coverStats=15, coverDate=12.5,
            dividerEyebrow=14, dividerTitle=38, dividerSub=15, title=25, subtitle=12.5,
            panelHeading=15, body=11.5, bodySmall=10.5, kpi=30, kpiLabel=10, kpiDesc=10.5,
            caption=9.5, footnote=8.5, table=9.5, chartAxis=9, chartLabel=9, blockHeading=13.5)
GRID = dict(left=0.62, right=12.71, contentTop=1.55, contentBottom=6.9, footerY=7.08)
SLIDE_W, SLIDE_H = 40.0 / 3.0, 7.5          # 13 1/3 x 7.5 in (960 x 540 pt), the template size
LEADING = 1.22                 # line height / font size for Calibri and Cambria
NBSP = " "
EN = "–"                  # en dash
DOT = "   ·   "           # stats separator on the cover

# Type-curve series: the P50 we apply is maroon (SE: "the series we apply is
# maroon; history and other are neutral grey"), then the SE series order.
TC_SERIES = (("P90", "AMBER", "dash", 1.5), ("P50", "MAROON", "solid", 2.5),
             ("P10", "BLUE", "dash", 1.5), ("Mean", "INK", "dot", 1.25))


def hexrgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------- fonts and text measurement
_FONTS: Dict[str, str] = {}           # role -> reportlab font name


def _find_font_file(family: str, bold: bool = False, italic: bool = False) -> Optional[str]:
    try:
        from matplotlib import font_manager as fm
        prop = fm.FontProperties(family=family, weight="bold" if bold else "normal",
                                 style="italic" if italic else "normal")
        path = fm.findfont(prop, fallback_to_default=False)
    except Exception:
        return None
    try:
        from matplotlib import font_manager as fm
        name = fm.get_font(path).family_name
    except Exception:
        name = ""
    return path if family.lower() in str(name).lower() else None


def fonts() -> Dict[str, str]:
    """Register the deck fonts with reportlab (once) and return role -> font name.

    Calibri / Cambria when installed (Windows), else the metric-compatible
    Carlito / Caladea, else Helvetica / Times (wider, so fitted text stays
    inside its box)."""
    if _FONTS:
        return _FONTS
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping

    def _register(stem: str, families: Sequence[str]) -> bool:
        files = {}
        for fam in families:
            for b, i, sfx in ((False, False, ""), (True, False, "-Bold"), (False, True, "-Italic"),
                              (True, True, "-BoldItalic")):
                path = _find_font_file(fam, b, i)
                if path is None and sfx == "":
                    files = {}
                    break
                files[sfx] = path
            if files:
                break
        if not files:
            return False
        for sfx in ("", "-Bold", "-Italic", "-BoldItalic"):
            path = files.get(sfx) or files.get("-Bold" if "Bold" in sfx else "")
            kw = {"subfontIndex": 0} if str(path).lower().endswith(".ttc") else {}
            pdfmetrics.registerFont(TTFont(stem + sfx, path, **kw))
        for b, i, sfx in ((0, 0, ""), (1, 0, "-Bold"), (0, 1, "-Italic"), (1, 1, "-BoldItalic")):
            addMapping(stem, b, i, stem + sfx)
        return True

    if _register("SEBody", ("Calibri", "Carlito")):
        body = dict(body="SEBody", body_bold="SEBody-Bold", body_italic="SEBody-Italic",
                    body_bold_italic="SEBody-BoldItalic")
    else:
        body = dict(body="Helvetica", body_bold="Helvetica-Bold", body_italic="Helvetica-Oblique",
                    body_bold_italic="Helvetica-BoldOblique")
    if _register("SEHead", ("Cambria", "Caladea")):
        head = dict(head="SEHead", head_bold="SEHead-Bold")
    else:
        head = dict(head="Times-Roman", head_bold="Times-Bold")
    _FONTS.update(body)
    _FONTS.update(head)
    return _FONTS


def font_name(face: str = "body", bold: bool = False, italic: bool = False) -> str:
    f = fonts()
    if face == "head":
        return f["head_bold"] if bold else f["head"]
    if bold and italic:
        return f["body_bold_italic"]
    if bold:
        return f["body_bold"]
    if italic:
        return f["body_italic"]
    return f["body"]


def text_width(text: str, size: float, face: str = "body", bold: bool = False, italic: bool = False) -> float:
    """Width in inches."""
    from reportlab.pdfbase import pdfmetrics
    return pdfmetrics.stringWidth(str(text), font_name(face, bold, italic), size) / 72.0


def wrap_runs(runs: Sequence[Tuple[str, bool]], width_in: float, size: float, face: str = "body") -> int:
    """Number of lines a paragraph of (text, bold) runs needs at this width
    (greedy word wrap on ordinary spaces; non-breaking spaces stay joined)."""
    words: List[Tuple[str, bool, bool]] = []          # (word, bold, space before)
    for text, bold in runs:
        parts = str(text).split(" ")
        for k, w in enumerate(parts):
            if w == "":
                continue
            words.append((w, bold, True))
    if not words:
        return 1
    lines, cur = 1, 0.0
    space = text_width(" ", size, face)
    for w, bold, _ in words:
        ww = text_width(w, size, face, bold)
        add = ww if cur == 0 else space + ww
        if cur > 0 and cur + add > width_in:
            lines += 1
            cur = ww
        else:
            cur += add
    return lines


def fit_size(paragraphs: Sequence[Sequence[Tuple[str, bool]]], width_in: float, height_in: float,
             size: float, min_size: float, gap_pt: float = 0.0, face: str = "body",
             indent_in: float = 0.0) -> float:
    """Largest font size <= size (0.5 pt steps, >= min_size) at which the
    paragraphs fit the box."""
    s = size
    while s > min_size:
        lines = sum(wrap_runs(p, width_in - indent_in, s, face) for p in paragraphs)
        h = lines * s * LEADING / 72.0 + max(len(paragraphs) - 1, 0) * gap_pt / 72.0
        if h <= height_in:
            return s
        s -= 0.5
    return min_size


def fit_line(text: str, width_in: float, size: float, min_size: float, face: str = "body",
             bold: bool = False) -> float:
    s = size
    while s > min_size and text_width(text, s, face, bold) > width_in:
        s -= 0.5
    return max(s, min_size)


def clip_text(text: str, width_in: float, size: float, bold: bool = False) -> str:
    """Shorten text with an ellipsis so it fits one line of the given width."""
    text = str(text)
    if text_width(text, size, bold=bold) <= width_in:
        return text
    while len(text) > 1 and text_width(text + "…", size, bold=bold) > width_in:
        text = text[:-1]
    return text.rstrip() + "…"


# ---------------------------------------------------------------- number and text helpers (SE section 4)
_UNITS = r"(ft|mi|Mbbl|MMcf|bbl|Mcf|bbl/ft|Mcf/ft|bbl/d|Mcf/d|bbl/mo|Mcf/mo|wells|months|per|%/yr|years)"


def nb(text: str) -> str:
    """Keep numbers with their units (non-breaking spaces), SE section 4."""
    return re.sub(r"(\d) " + _UNITS + r"\b", lambda m: m.group(1) + NBSP + m.group(2), str(text))


def fmt_int(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{int(round(float(v))):,}"


def fmt_vol(v) -> str:
    """Volumes (Mbbl, MMcf, bbl/ft): commas, no decimals from 100; one decimal
    below that, with '.0' dropped; two below 1."""
    if v is None or not np.isfinite(v):
        return "–"
    a = abs(float(v))
    if a >= 100:
        return f"{float(v):,.0f}"
    if a >= 1:
        s = f"{float(v):,.1f}"
        return s[:-2] if s.endswith(".0") else s
    return f"{float(v):.2f}"


def fmt_pct(frac) -> str:
    return "–" if frac is None or not np.isfinite(frac) else f"{round(float(frac) * 100):.0f}%"


def fmt_ratio(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{float(v):.1f}"


def fmt_b(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{float(v):.2f}"


def fmt_di(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{float(v):.3f}"


def title_name(name: str) -> str:
    """UPPER CASE names to Title Case (SE: 'BLUE MESA 12' -> 'Blue Mesa 12')."""
    name = str(name).strip()
    if name and name == name.upper() and any(ch.isalpha() for ch in name):
        return re.sub(r"(^|[\s/(-])([a-z])", lambda m: m.group(1) + m.group(2).upper(), name.lower())
    return name


def join_and(items: Sequence[str]) -> str:
    items = [str(i) for i in items if str(i)]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- report input (pure data)
@dataclass
class FluidReport:
    name: str                                   # Oil | Gas | Water
    unit: str                                   # Mbbl | MMcf
    per_ft_unit: str                            # bbl/ft | Mcf/ft
    rate_unit: str                              # bbl/mo | Mcf/mo
    qi_unit: str                                # bbl/d | Mcf/d
    n_tc: int                                   # wells in the type curve
    n_total: int                                # wells forecast
    stats: Dict[str, float]                     # P90, P50, P10, Mean, Count, P10/P90, Min, Max
    per_ft_basis: float                         # ft used for EUR per ft
    groups: Dict[str, Dict[str, float]]         # P90/P50/P10/Mean -> qi (per day), b, di, dec1, eur, n
    terminal_di: float                          # nominal per month
    terminal_pct: float                         # effective %/yr
    b_values: List[float]
    b_low: float
    b_high: float
    n_b_at_bound: int
    curves: pd.DataFrame                        # t, P10, P50, P90, Mean (monthly volumes, per well)
    lines: List[Tuple[np.ndarray, np.ndarray, str]]
    wells: pd.DataFrame                         # all forecast wells: WellID, WellName, API10, County,
                                                # LateralLength, Months, EUR, b, in_tc
    upper_fence: float
    lower_fence: float
    yield_label: str = ""                       # 'Yield (GOR)' when forecast as a yield

    def per_ft(self, eur: float) -> float:
        return eur * 1000.0 / self.per_ft_basis if self.per_ft_basis and np.isfinite(eur) else float("nan")


@dataclass
class ReportInput:
    tc_name: str
    source: str                                 # e.g. 'Production data'
    as_of: str                                  # e.g. 'Sep 2025'
    generated: _dt.date
    counties: List[str]
    state: str
    formations: List[str]
    vintage: str                                # e.g. '2017–2021'
    lateral_min: float
    lateral_max: float
    lateral_mean: float
    months_median: float
    norm_len: float
    use_norm: bool
    exponent: float
    settings: List[Tuple[str, str]]             # (setting, value) as forecast
    decline_models: str
    fluids: List[FluidReport]
    disclaimer: str = ""


# ---------------------------------------------------------------- report input from forecast tables
_FLUID_UNITS = {"oil": ("Mbbl", "bbl/ft", "bbl/mo", "bbl/d"), "gas": ("MMcf", "Mcf/ft", "Mcf/mo", "Mcf/d"),
                "water": ("Mbbl", "bbl/ft", "bbl/mo", "bbl/d")}


def log_lower_fence(eur) -> float:
    """Lower EUR outlier fence in log space (same rule as the app's outlier screen)."""
    eur = pd.to_numeric(pd.Series(eur), errors="coerce")
    pos = eur[(eur > 0) & np.isfinite(eur)]
    if len(pos) < 3:
        return 0.0
    le = np.log(pos.astype(float))
    lq1, lq3 = le.quantile(0.25), le.quantile(0.75)
    return float(np.exp(lq1 - 1.5 * (lq3 - lq1)))


def fluid_report(name: str, oneline_full: pd.DataFrame, sel_ids, eur_col: str, curves: pd.DataFrame,
                 lines, d_lim: float, norm_len: float, use_norm: bool, b_low: float, b_high: float,
                 yield_label: str = "") -> Optional[FluidReport]:
    """Report data for one phase from its oneline (all forecast wells), the
    type-curve well set and the type curves shown in the app."""
    import core
    if oneline_full is None or oneline_full.empty or eur_col not in oneline_full.columns:
        return None
    unit, per_ft_unit, rate_unit, qi_unit = _FLUID_UNITS[name.lower()]
    ol = oneline_full.copy()
    ol["WellID"] = ol["WellID"].astype(str)
    sel_ids = set(map(str, sel_ids)) if sel_ids is not None else set(ol["WellID"])
    ol["in_tc"] = ol["WellID"].isin(sel_ids)
    ol["_eur"] = pd.to_numeric(ol[eur_col], errors="coerce")
    tc = ol[ol["in_tc"]].copy()
    eurs = tc["_eur"].dropna().astype(float)
    stats = core.compute_eur_stats(eurs.tolist())
    stats["Min"] = float(eurs.min()) if len(eurs) else float("nan")
    stats["Max"] = float(eurs.max()) if len(eurs) else float("nan")
    lat_tc = pd.to_numeric(tc.get("LateralLength"), errors="coerce").dropna() if "LateralLength" in tc else pd.Series([], dtype=float)
    basis = float(norm_len) if use_norm else (float(lat_tc.mean()) if len(lat_tc) else float(norm_len))

    # Type-well parameters: the median qi, b and Di of each EUR group (the same
    # grouping as the app's type-curve builder), with qi scaled so the curve's
    # 50-year volume equals the group's EUR target, exactly as the plotted curves.
    g = tc.copy()
    g["_qi"] = pd.to_numeric(g.get("qi (per day)"), errors="coerce")
    g["_b"] = pd.to_numeric(g.get("b"), errors="coerce")
    g["_di"] = pd.to_numeric(g.get("di (per month)"), errors="coerce")
    g = g.dropna(subset=["_eur", "_qi", "_b", "_di"]).sort_values("_eur", ascending=False).reset_index(drop=True)
    n = len(g)
    groups: Dict[str, Dict[str, float]] = {}
    if n:
        lo50 = max(0, int(n * 0.45))
        hi50 = max(lo50 + 1, int(n * 0.55))
        parts = {"P90": g.iloc[max(0, int(n * 0.9)):], "P50": g.iloc[lo50:hi50],
                 "P10": g.iloc[:max(1, int(n * 0.1))], "Mean": g}
        for key, grp in parts.items():
            if grp.empty:
                continue
            qi, b, di = float(grp["_qi"].median()), float(grp["_b"].median()), float(grp["_di"].median())
            target = float(stats.get(key, float("nan")))
            unscaled = float(core.modified_arps_cum(qi * core.DAYS_PER_MONTH, b, di, d_lim, np.array([600.0]))[0])
            scale = target * 1000.0 / unscaled if (unscaled > 0 and np.isfinite(target)) else float("nan")
            q0 = float(core.modified_arps(1.0, b, di, d_lim, np.array([0.0]))[0])
            q12 = float(core.modified_arps(1.0, b, di, d_lim, np.array([12.0]))[0])
            groups[key] = dict(qi=qi * scale, b=b, di=di, dec1=1.0 - q12 / q0 if q0 > 0 else float("nan"),
                               eur=target, n=len(grp))
    b_all = pd.to_numeric(tc.get("b"), errors="coerce").dropna() if "b" in tc else pd.Series([], dtype=float)
    at_bound = int(tc["b at bound"].astype(bool).sum()) if "b at bound" in tc else 0

    def col(c, default=np.nan):
        return ol[c] if c in ol.columns else pd.Series([default] * len(ol), index=ol.index)

    wells = pd.DataFrame({
        "WellID": ol["WellID"], "WellName": col("WellName", ""), "API10": col("API10", ""),
        "County": col("County", ""), "LateralLength": pd.to_numeric(col("LateralLength"), errors="coerce"),
        "Months": pd.to_numeric(col("Months History"), errors="coerce"), "EUR": ol["_eur"],
        "b": pd.to_numeric(col("b"), errors="coerce"), "in_tc": ol["in_tc"]}).reset_index(drop=True)
    e_all = ol["_eur"].dropna()
    q1, q3 = (e_all.quantile(0.25), e_all.quantile(0.75)) if len(e_all) else (np.nan, np.nan)
    return FluidReport(
        name=name, unit=unit, per_ft_unit=per_ft_unit, rate_unit=rate_unit, qi_unit=qi_unit,
        n_tc=int(len(tc)), n_total=int(len(ol)), stats=stats, per_ft_basis=basis, groups=groups,
        terminal_di=float(d_lim), terminal_pct=core.effective_annual_from_nominal_monthly(float(d_lim)) * 100.0,
        b_values=[float(v) for v in b_all], b_low=float(b_low), b_high=float(b_high), n_b_at_bound=at_bound,
        curves=curves if curves is not None else pd.DataFrame(), lines=list(lines or []), wells=wells,
        upper_fence=float(q3 + 1.5 * (q3 - q1)) if len(e_all) else float("nan"),
        lower_fence=log_lower_fence(e_all), yield_label=yield_label)


def analog_summary(fluids: Sequence[FluidReport], header: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Counties, formations, vintage and lateral statistics of the wells in the
    type curves (union over phases), from the onelines."""
    ids = set()
    for f in fluids:
        ids |= set(f.wells.loc[f.wells["in_tc"], "WellID"].astype(str))
    frames = [f.wells for f in fluids]
    w = pd.concat(frames).drop_duplicates("WellID") if frames else pd.DataFrame()
    w = w[w["WellID"].astype(str).isin(ids)] if len(w) else w
    out: Dict[str, Any] = dict(counties=[], state="", formations=[], vintage="", lateral_min=float("nan"),
                               lateral_max=float("nan"), lateral_mean=float("nan"), months_median=float("nan"))
    if len(w):
        lat = pd.to_numeric(w["LateralLength"], errors="coerce").dropna()
        if len(lat):
            out.update(lateral_min=float(lat.min()), lateral_max=float(lat.max()), lateral_mean=float(lat.mean()))
        mo = pd.to_numeric(w["Months"], errors="coerce").dropna()
        if len(mo):
            out["months_median"] = float(mo.median())
        c = w["County"].dropna().astype(str)
        c = c[c.str.strip() != ""]
        out["counties"] = list(c.value_counts().index)
    if header is not None and len(header):
        h = header[header["WellID"].astype(str).isin(ids)] if "WellID" in header else header
        for colname, key in (("State", "state"),):
            if colname in h and h[colname].notna().any():
                out[key] = str(h[colname].dropna().astype(str).mode().iloc[0])
        if "PrimaryFormation" in h:
            fm = h["PrimaryFormation"].dropna().astype(str)
            out["formations"] = list(fm[fm.str.strip() != ""].value_counts().index)
        for colname in ("FirstProdDate", "CompletionDate"):
            if colname in h:
                d = pd.to_datetime(h[colname], errors="coerce").dropna()
                if len(d):
                    y0, y1 = d.min().year, d.max().year
                    out["vintage"] = f"{y0}" if y0 == y1 else f"{y0}{EN}{str(y1)[-2:] if str(y0)[:2] == str(y1)[:2] else y1}"
                    break
    return out


# ---------------------------------------------------------------- deck model
@dataclass
class Block:
    kind: str                     # text | bullets | card | panel | table | chart | image | rect
    x: float
    y: float
    w: float
    h: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Slide:
    layout: str                   # COVER | CONTENT | SECTION
    section: str
    fields: Dict[str, Any] = field(default_factory=dict)    # title, sub, eyebrow, stats, date (+ sizes)
    blocks: List[Block] = field(default_factory=list)


@dataclass
class Deck:
    slides: List[Slide]
    footer_left: str
    footer_right: str
    file_stem: str
    title: str
    subject: str


# ---- block constructors -------------------------------------------------------
def text_block(x, y, w, h, paragraphs, size=SIZE["body"], color="INK", face="body", align="l",
               valign="t", italic=False, bold=False, gap=0.0, name="text") -> Block:
    """paragraphs: list of paragraphs; a paragraph is a str or a list of (text, bold) runs."""
    paras = [[(nb(p), bold)] if isinstance(p, str) else [(nb(t), b) for t, b in p] for p in paragraphs]
    return Block("text", x, y, w, h, dict(paras=paras, size=size, color=color, face=face, align=align,
                                          valign=valign, italic=italic, gap=gap, name=name))


def heading_block(x, y, w, text) -> Block:
    return text_block(x, y, w, 0.34, [text], size=SIZE["blockHeading"], color="MAROON", face="head",
                      bold=True, name="block heading")


def caption_block(x, y, w, h, text, size=SIZE["footnote"]) -> Block:
    s = fit_size([[(nb(text), False)]], w, h, size, 7.5)
    return text_block(x, y, w, h, [text], size=s, color="MID", italic=True, name="caption")


def bullets_block(x, y, w, h, heading, items, size=SIZE["body"], gap=6.0, min_size=9.5) -> Block:
    """A bullet panel: optional Cambria heading, bullets with a bold lead."""
    items = [(nb(lead), nb(txt)) for lead, txt in items]
    top = 0.46 if heading else 0.0
    paras = [[(lead + " ", True), (txt, False)] if lead else [(txt, False)] for lead, txt in items]
    s = fit_size(paras, w, h - top, size, min_size, gap_pt=gap, indent_in=0.1875)
    return Block("bullets", x, y, w, h, dict(heading=nb(heading) if heading else heading, items=items,
                                             size=s, gap=gap))


def bullets_height(b: Block) -> float:
    """Height (in) the fitted bullets of a bullets block take up."""
    d = b.data
    paras = [[(lead + " ", True), (txt, False)] if lead else [(txt, False)] for lead, txt in d["items"]]
    lines = sum(wrap_runs(p, b.w - 0.1875, d["size"]) for p in paras)
    return ((0.46 if d["heading"] else 0.0) + lines * d["size"] * LEADING / 72.0
            + max(len(paras) - 1, 0) * d["gap"] / 72.0)


def panel_with_bullets(x, y, w, h_max, heading, items, size, gap=5.0, pad=0.2) -> List[Block]:
    """Grey panel sized to its bullets (at most h_max tall)."""
    b = bullets_block(x + 0.26, y + pad, w - 0.52, h_max - 2 * pad, heading, items, size=size, gap=gap)
    h = min(h_max, bullets_height(b) + 2 * pad + 0.08)
    b.h = h - 2 * pad
    return [panel_block(x, y, w, h), b]


def card_block(x, y, w, h, value, label, desc="", foot="", color="MAROON") -> Block:
    vs = fit_line(value, w - 0.2, SIZE["kpi"], 18, face="head", bold=True)
    ds = fit_size([[(nb(desc), False)]], w - 0.2, 0.55, SIZE["kpiDesc"], 8.5) if desc else SIZE["kpiDesc"]
    fs = fit_line(foot, w - 0.12, 10, 7.5, bold=True) if foot else 10
    return Block("card", x, y, w, h, dict(value=nb(value), label=label, desc=nb(desc), foot=nb(foot),
                                          color=color, vsize=vs, dsize=ds, fsize=fs))


def panel_block(x, y, w, h) -> Block:
    return Block("panel", x, y, w, h, {})


def table_block(x, y, col_w, rows, align=None, size=SIZE["table"], row_h=0.28, bold_rows=(),
                header=True, name="table") -> Block:
    """rows[0] is the header when header=True.  Cell text is clipped to the
    column width (one line per cell)."""
    pad = 0.04 * 2
    ncol = len(col_w)
    align = list(align or (["l"] + ["r"] * (ncol - 1)))
    out = []
    for ri, row in enumerate(rows):
        bold = (header and ri == 0) or ri in bold_rows
        out.append([clip_text(nb(str(c)), col_w[ci] - pad, size, bold) for ci, c in enumerate(row)])
    return Block("table", x, y, sum(col_w), row_h * len(rows),
                 dict(rows=out, col_w=list(col_w), align=align, size=size, row_h=row_h,
                      bold_rows=set(bold_rows), header=header, name=name))


def chart_block(x, y, w, h, kind, spec) -> Block:
    return Block("chart", x, y, w, h, dict(kind=kind, spec=spec))


def image_block(x, y, w, h, figure: Callable[[float, float], Any], alt: str) -> Block:
    """figure(w_in, h_in) -> matplotlib Figure drawn at the slot size."""
    return Block("image", x, y, w, h, dict(figure=figure, alt=alt))


# ---------------------------------------------------------------- deck content
def _place(inp: ReportInput, caps: bool = False) -> str:
    st = inp.state or ""
    names = [title_name(c) for c in inp.counties if str(c).strip()]
    if not names:
        place = st
    elif len(names) == 1:
        place = f"{names[0]} County" + (f", {st}" if st else "")
    elif len(names) == 2:
        place = f"{names[0]} and {names[1]} Counties" + (f", {st}" if st else "")
    else:
        place = f"{len(names)} counties" + (f", {st}" if st else "")
    return place.upper() if caps else place


def _formation_phrase(inp: ReportInput) -> str:
    f = [title_name(x) for x in inp.formations if str(x).strip()]
    if not f:
        return ""
    if len(f) == 2:
        a, b = f[0].rsplit(" ", 1) if " " in f[0] else (f[0], ""), f[1].rsplit(" ", 1) if " " in f[1] else (f[1], "")
        if a[1] and b[1] and a[0] == b[0]:               # 'Wolfcamp A' + 'Wolfcamp B' -> 'Wolfcamp A and B'
            return f"{a[0]} {a[1]} and {b[1]}"
        return join_and(f)
    if len(f) == 1:
        return f[0]
    return f"{f[0]} and {len(f) - 1} other formations"


def _sample_sentence(n: int) -> str:
    if n < 10:
        return f"{n} wells is below the 10-well minimum of SPEE Monograph 5; treat percentiles as indicative."
    if n < 30:
        return f"{n} wells is a moderate sample under SPEE Monograph 5 (30 or more is preferred)."
    return f"{n} wells meets the SPEE Monograph 5 guide of 30 or more wells."


def _norm_phrase(inp: ReportInput) -> str:
    return f"per well at {fmt_int(inp.norm_len)} ft" if inp.use_norm else "per well, as reported"


def _file_stem(inp: ReportInput) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", inp.tc_name).strip("_") or "SE"
    return f"{base}_Type_Curve_{inp.generated.strftime('%b%Y')}_v1"


def build_deck(inp: ReportInput) -> Deck:
    """The slides of the type-curve report, as data (no rendering)."""
    fl = [f for f in inp.fluids if f.n_tc > 0]
    prim = fl[0] if fl else None
    L, W = GRID["left"], 12.09
    slides: List[Slide] = []
    norm = _norm_phrase(inp)
    phases = join_and([f.name.lower() for f in fl]).capitalize() or "Type"
    name = inp.tc_name.strip() or "Type Curve Report"
    n_an = max([f.n_tc for f in fl], default=0)
    place = _place(inp)

    # ---- 1. Cover --------------------------------------------------------------
    eyebrow = DOT.join(x for x in ("TYPE CURVE", _place(inp, caps=True), _formation_phrase(inp).upper()) if x)
    title = f"{name} Type Curve" if inp.tc_name.strip() else name
    stats = [f"P50 {f.name.lower()} {fmt_vol(f.stats['P50'])} {f.unit}" for f in fl]
    stats.append(f"{n_an} analog wells")
    if inp.use_norm:
        stats.append(f"{fmt_int(inp.norm_len)} ft lateral")
    stats_txt = nb(DOT.join(stats[:5]))
    sub = nb(f"{phases} type wells from {n_an} analog wells, {norm}")
    cover = Slide("COVER", "Overview", dict(
        eyebrow=eyebrow, title=title, sub=sub, stats=stats_txt,
        date=f"{inp.generated.strftime('%B %Y')}{DOT}{COMPANY}",
        title_size=fit_line(title, 11.4, SIZE["coverTitle"], 26, face="head", bold=True),
        eyebrow_size=fit_line(eyebrow, 10.9, SIZE["coverEyebrow"], 9, bold=True),
        sub_size=fit_line(sub, 10.5, SIZE["coverSub"], 12),
        stats_size=fit_line(stats_txt, 11.5, SIZE["coverStats"], 10, bold=True)))
    slides.append(cover)

    def content(section, title, sub, blocks):
        s = Slide("CONTENT", section, dict(
            title=title, sub=nb(sub),
            title_size=fit_line(title, 11.2, SIZE["title"], 18, face="head", bold=True),
            sub_size=fit_line(nb(sub), 11.4, SIZE["subtitle"], 10)), blocks)
        slides.append(s)
        return s

    # ---- 2. Disclaimer ---------------------------------------------------------------
    if inp.disclaimer.strip():
        ds = fit_size([[(inp.disclaimer, False)]], W, 5.1, 12, 8.5)
        content("Overview", "Disclaimer", "Read this before you use the numbers in this presentation.",
                [text_block(L, 1.65, W, 5.1, [inp.disclaimer], size=ds, color="INK", name="disclaimer")])

    # ---- 3. At a glance ----------------------------------------------------------------
    if fl:
        cards = []
        n_cards = min(len(fl), 3) + 1
        cw = (W - (n_cards - 1) * 0.25) / n_cards
        for i, f in enumerate(fl[:3]):
            s = f.stats
            cards.append(card_block(L + i * (cw + 0.25), 1.6, cw, 2.36,
                                    f"{fmt_vol(s['P50'])} {f.unit}", f"P50 {f.name.upper()} EUR",
                                    f"P90 {fmt_vol(s['P90'])} {EN} P10 {fmt_vol(s['P10'])} {f.unit}",
                                    f"Mean {fmt_vol(s['Mean'])} {f.unit}  ·  "
                                    f"{fmt_vol(f.per_ft(s['P50']))} {f.per_ft_unit}"))
        i = n_cards - 1
        cards.append(card_block(L + i * (cw + 0.25), 1.6, cw, 2.36, f"{n_an}", "ANALOG WELLS",
                                f"{place}" + (f", {_formation_phrase(inp)}" if _formation_phrase(inp) else ""),
                                f"First production {inp.vintage}" if inp.vintage else ""))
        items = [("Analogs.", f"{n_an} wells in {place}" +
                  (f", first production {inp.vintage}." if inp.vintage else "."))]
        if inp.use_norm:
            items.append(("Normalization.", f"We scale each well to {fmt_int(inp.norm_len)} ft of lateral, "
                                            f"with EUR in proportion to length to the power {inp.exponent:.2f}."))
        b_lo, b_hi = (prim.b_low, prim.b_high) if prim else (0, 0)
        term = prim.terminal_pct if prim else float("nan")
        items.append(("Decline.", f"We fit {inp.decline_models} to each well, with b from {fmt_b(b_lo)} to "
                                  f"{fmt_b(b_hi)} and a {term:.0f}%/yr terminal decline."))
        items.append(("Type wells.", "P10, P50 and P90 come from the EUR spread; each curve is scaled to its EUR."))
        if prim:
            items.append(("Screen.", f"{prim.n_tc} of {prim.n_total} {prim.name.lower()} wells stay in the type "
                                     "curve; we removed EUR outliers and non-analogs."))
        content("Overview", "Type Curve at a Glance",
                f"P50 EUR {norm}, from {n_an} analog wells in {place}",
                cards + panel_with_bullets(L, 4.2, W, 2.65, "How we built the type curve", items, size=12.5))

    # ---- 4. Analog set and method -------------------------------------------------------
    if fl:
        analog_rows = [["Item", "Value"]]
        analog_rows.append(["Wells in the type curves", "  ·  ".join(f"{f.name} {f.n_tc}" for f in fl)])
        analog_rows.append(["Wells forecast", "  ·  ".join(f"{f.name} {f.n_total}" for f in fl)])
        if inp.counties:
            analog_rows.append(["Counties", join_and([title_name(c) for c in inp.counties[:4]]) +
                                (f" and {len(inp.counties) - 4} more" if len(inp.counties) > 4 else "") +
                                (f", {inp.state}" if inp.state else "")])
        if inp.formations:
            analog_rows.append(["Formations", join_and([title_name(x) for x in inp.formations[:3]]) +
                                (f" and {len(inp.formations) - 3} more" if len(inp.formations) > 3 else "")])
        if np.isfinite(inp.lateral_min):
            analog_rows.append(["Lateral length", f"{fmt_int(inp.lateral_min)}{EN}{fmt_int(inp.lateral_max)} ft "
                                                  f"(mean {fmt_int(inp.lateral_mean)} ft)"])
        if inp.vintage:
            analog_rows.append(["First production", inp.vintage])
        if np.isfinite(inp.months_median):
            analog_rows.append(["Months of history", f"median {fmt_int(inp.months_median)}"])
        analog_rows.append(["Production data", f"{inp.source} through {inp.as_of}"])
        set_rows = [["Setting", "Value"]] + [[a, b] for a, b in inp.settings]
        cw2 = 5.9
        n_rows = max(len(analog_rows), len(set_rows))
        rh = min(0.4, max(0.3, 4.2 / n_rows))
        blocks = [heading_block(L, 1.6, cw2, "Analog wells"),
                  table_block(L, 1.98, [2.3, cw2 - 2.3], analog_rows, align=["l", "l"], row_h=rh, size=10.5),
                  heading_block(L + cw2 + 0.29, 1.6, cw2, "Settings we applied"),
                  table_block(L + cw2 + 0.29, 1.98, [2.6, cw2 - 2.6], set_rows, align=["l", "l"], row_h=rh,
                              size=10.5)]
        rows_h = n_rows * rh
        blocks.append(caption_block(L, min(1.98 + rows_h + 0.15, 6.45), W, 0.42,
                                    f"EUR is the volume of one well over 50 years{', at ' + fmt_int(inp.norm_len) + ' ft of lateral' if inp.use_norm else ''}. "
                                    "Settings are the ones the forecasts were run with."))
        content("Overview", "Analog Set and Method", "The wells behind the type curves and the settings we applied",
                blocks)

    # ---- 5. EUR summary --------------------------------------------------------------------
    if fl:
        hdr = ["Phase", "Wells", "P90", "P50", "P10", "Mean", "P50 per ft", "P10/P90"]
        rows = [hdr]
        for f in fl:
            s = f.stats
            rows.append([f.name, f"{f.n_tc} of {f.n_total}"] +
                        [f"{fmt_vol(s[k])} {f.unit}" for k in ("P90", "P50", "P10", "Mean")] +
                        [f"{fmt_vol(f.per_ft(s['P50']))} {f.per_ft_unit}", fmt_ratio(s.get("P10/P90"))])
        col_w = [1.35, 1.25, 1.6, 1.6, 1.6, 1.6, 1.7, 1.39]
        tb = table_block(L, 1.6, col_w, rows, row_h=0.34, size=10.5)
        items = []
        for f in fl:
            s = f.stats
            d = (s["Mean"] / s["P50"] - 1) if s["P50"] else float("nan")
            rel = "above" if d >= 0 else "below"
            items.append((f"{f.name}.", f"P50 {fmt_vol(s['P50'])} {f.unit} per well "
                                        f"({fmt_vol(f.per_ft(s['P50']))} {f.per_ft_unit}); the Mean is "
                                        f"{fmt_pct(abs(d))} {rel} P50."))
        items.append(("Spread.", "P10/P90 ratio of " + join_and([f"{fmt_ratio(f.stats.get('P10/P90'))} for "
                                                                f"{f.name.lower()}" for f in fl]) + "."))
        items.append(("Sample.", _sample_sentence(min(f.n_tc for f in fl))))
        items.append(("Use.", "Use the Mean when you add wells together; use P50 for one typical well."))
        top = 1.6 + 0.34 * len(rows) + 0.3
        content("Overview", "EUR Summary by Phase",
                f"P90, P50 and P10 EUR {norm}; the Mean is the value to add across wells",
                [tb] + panel_with_bullets(L, top, W, 6.85 - top, "What the numbers say", items, size=12.5))

    # ---- 6. Per phase ---------------------------------------------------------------------------
    for f in fl:
        s, g = f.stats, f.groups
        p50 = g.get("P50", {})
        # 6a. type curve
        items = [("P50.", f"{fmt_vol(s['P50'])} {f.unit} per well, or {fmt_vol(f.per_ft(s['P50']))} "
                          f"{f.per_ft_unit}."),
                 ("Range.", f"P90 {fmt_vol(s['P90'])} to P10 {fmt_vol(s['P10'])} {f.unit}; the P10/P90 ratio "
                            f"is {fmt_ratio(s.get('P10/P90'))}."),
                 ("Shape.", f"P50 b is {fmt_b(p50.get('b'))} and the initial decline "
                            f"{fmt_di(p50.get('di'))} per month ({fmt_pct(p50.get('dec1'))} in year one)."),
                 ("Terminal.", f"The decline turns exponential at {f.terminal_pct:.0f}%/yr effective."),
                 ("Wells.", f"{f.n_tc} of {f.n_total} wells; {f.n_total - f.n_tc} are out as EUR outliers "
                            "or non-analogs.")]
        if f.yield_label:
            items.append(("Yield.", f"{f.name} is forecast as a {f.yield_label.split('(')[-1].rstrip(')')} "
                                    "yield on each well's oil forecast."))
        spec = dict(curves=f.curves, lines=f.lines, unit=f.rate_unit, name=f.name,
                    y_title=f"{f.name}, {f.rate_unit} ({norm})", n_wells=f.n_tc)
        content(f.name, f"{f.name} Type Curve",
                f"P10, P50 and P90 type wells over the normalized histories of {f.n_tc} analog wells",
                [chart_block(0.42, 1.55, 8.2, 5.3, "tc", spec),
                 bullets_block(8.85, 1.6, 3.9, 5.2, f"P50 of {fmt_vol(s['P50'])} {f.unit}", items, size=11.5)])

        # 6b. parameters + b distribution
        cols = [k for k in ("P90", "P50", "P10", "Mean") if k in g]
        prow = [["Parameter"] + cols,
                [f"qi, {f.qi_unit}"] + [fmt_int(g[k].get("qi")) for k in cols],
                ["b"] + [fmt_b(g[k].get("b")) for k in cols],
                ["Initial decline, per month"] + [fmt_di(g[k].get("di")) for k in cols],
                ["First-year decline"] + [fmt_pct(g[k].get("dec1")) for k in cols],
                ["Terminal decline, %/yr"] + [f"{f.terminal_pct:.1f}" for _ in cols],
                [f"EUR, {f.unit}"] + [fmt_vol(g[k].get("eur")) for k in cols],
                [f"EUR per ft, {f.per_ft_unit}"] + [fmt_vol(f.per_ft(g[k].get("eur", float("nan")))) for k in cols],
                ["Wells in the group"] + [fmt_int(g[k].get("n")) for k in cols]]
        bvals = np.asarray(f.b_values, dtype=float)
        bvals = bvals[np.isfinite(bvals)]
        bcap = (f"{bvals.size} wells. Median {fmt_b(np.median(bvals))}, P10 {fmt_b(np.percentile(bvals, 90))}, "
                f"P90 {fmt_b(np.percentile(bvals, 10))}. Fitting bounds {fmt_b(f.b_low)}{EN}{fmt_b(f.b_high)}; "
                f"{f.n_b_at_bound} wells sit on a bound.") if bvals.size else "No b values."
        content(f.name, f"{f.name} Type Curve Parameters",
                "Modified Arps parameters that reproduce each type well and its EUR",
                [heading_block(L, 1.6, 5.9, "Type wells"),
                 table_block(L, 1.98, [2.5] + [3.4 / len(cols)] * len(cols), prow, row_h=0.32),
                 caption_block(L, 1.98 + 0.32 * len(prow) + 0.12, 5.9, 0.75,
                               "Each type well uses the median qi, b and Di of its EUR group. "
                               "Groups: top 10% for P10, middle 10% for P50, bottom 10% for P90, all wells for "
                               "the Mean. We then scale each curve to its EUR."),
                 heading_block(L + 6.19, 1.6, 5.9, "b-factor of the analog wells"),
                 chart_block(L + 6.19, 1.98, 5.9, 3.95, "hist", dict(values=bvals.tolist())),
                 caption_block(L + 6.19, 6.05, 5.9, 0.6, bcap)])

        # 6c. EUR distribution
        erows = [["Statistic", "Value"],
                 ["Wells", fmt_int(s.get("Count", f.n_tc))],
                 ["P90", f"{fmt_vol(s['P90'])} {f.unit}"],
                 ["P50", f"{fmt_vol(s['P50'])} {f.unit}"],
                 ["P10", f"{fmt_vol(s['P10'])} {f.unit}"],
                 ["Mean", f"{fmt_vol(s['Mean'])} {f.unit}"],
                 ["P10/P90", fmt_ratio(s.get("P10/P90"))],
                 ["Minimum", f"{fmt_vol(s.get('Min'))} {f.unit}"],
                 ["Maximum", f"{fmt_vol(s.get('Max'))} {f.unit}"],
                 ["P50 per ft", f"{fmt_vol(f.per_ft(s['P50']))} {f.per_ft_unit}"]]
        eurs = f.wells.loc[f.wells["in_tc"], "EUR"].astype(float).to_numpy() if len(f.wells) else np.array([])
        content(f.name, f"{f.name} EUR Distribution",
                "EUR per well on a log-probability scale; a straight line means a lognormal spread",
                [image_block(0.42, 1.55, 8.2, 4.75, _probit_figure_factory(eurs, f), f"{f.name} EUR probit plot"),
                 caption_block(0.62, 6.38, 8.0, 0.45,
                               f"Points are wells; the line joins P90 and P10. Volumes are {norm}."),
                 heading_block(8.85, 1.6, 3.9, "EUR statistics"),
                 table_block(8.85, 1.98, [1.75, 2.15], erows, row_h=0.32)])

    # ---- 7. Appendix ---------------------------------------------------------------------------
    if fl:
        slides.append(Slide("SECTION", "Appendix", dict(
            eyebrow="APPENDIX", title="Outlier Screen and Analog Wells",
            sub="How we screened EUR outliers, and the wells behind each type curve")))
        for f in fl:
            wells = f.wells
            n_high = int((wells["EUR"] > f.upper_fence).sum()) if len(wells) else 0
            n_low = int((wells["EUR"] < f.lower_fence).sum()) if len(wells) else 0
            items = [("Upper fence.", f"Q3 + 1.5 × IQR = {fmt_vol(f.upper_fence)} {f.unit}."),
                     ("Lower fence.", f"The same test in log space: {fmt_vol(f.lower_fence)} {f.unit}."),
                     ("Flagged.", f"{n_high} wells above the upper fence and {n_low} below the lower fence."),
                     ("Excluded.", f"{f.n_total - f.n_tc} of {f.n_total} wells are not in the type curve, "
                                   "by this screen or by analog choice."),
                     ("Why log space.", "EURs are near lognormal, so a linear lower fence falls below zero.")]
            content("Appendix", f"Outlier Screen — {f.name}",
                    "EUR against lateral length for every forecast well, with the fences we screen against",
                    [chart_block(0.42, 1.55, 8.2, 5.3, "scatter",
                                 dict(wells=wells, upper=f.upper_fence, lower=f.lower_fence, unit=f.unit,
                                      name=f.name)),
                     bullets_block(8.85, 1.6, 3.9, 5.2, "How we screen", items, size=11.5)])
        slides.extend(_analog_table_slides(inp, fl, norm))

    footer_left = f"{COMPANY}{DOT}{name}" + (f" — {place}" if place else "")
    footer_right = f"{inp.source} through {inp.as_of}" if inp.as_of else inp.source
    return Deck(slides, footer_left, footer_right, _file_stem(inp), title, f"Type curve report — {place}")


def _analog_table_slides(inp: ReportInput, fl: List[FluidReport], norm: str,
                         rows_per_page: int = 20, max_pages: int = 4, row_h: float = 0.24) -> List[Slide]:
    prim = fl[0]
    ids: List[str] = []
    for f in fl:
        for wid in f.wells.loc[f.wells["in_tc"], "WellID"].astype(str):
            if wid not in ids:
                ids.append(wid)
    if not ids:
        return []
    base = prim.wells.set_index(prim.wells["WellID"].astype(str))
    for f in fl[1:]:
        extra = f.wells[~f.wells["WellID"].astype(str).isin(base.index)]
        if len(extra):
            base = pd.concat([base, extra.set_index(extra["WellID"].astype(str))])
    eur_by = {f.name: f.wells.set_index(f.wells["WellID"].astype(str)) for f in fl}
    pe = eur_by[prim.name]

    def _key(w):                     # wells in the first phase's type curve first, by its EUR
        if w in pe.index:
            r = pe.loc[w]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            e = float(r["EUR"]) if np.isfinite(r["EUR"]) else -1.0
            return (0 if bool(r["in_tc"]) else 1, -e)
        return (2, 0.0)
    order = sorted(ids, key=_key)
    hdr = ["#", "Well", "API", "County", "Lateral, ft", "Months"]
    for f in fl:
        hdr.append(f"{f.name} EUR, {f.unit}")
    hdr.append(f"{prim.name} b")
    fixed = [0.4, 2.55, 1.35, 1.3, 1.0, 0.75]
    rest = 12.09 - sum(fixed)
    col_w = fixed + [rest / (len(fl) + 1)] * (len(fl) + 1)
    body = []
    for i, wid in enumerate(order, 1):
        r = base.loc[wid] if wid in base.index else None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        row = [str(i), title_name(r.get("WellName", wid)) if r is not None else wid,
               str(r.get("API10", "")) if r is not None else "",
               title_name(r.get("County", "")) if r is not None else "",
               fmt_int(r.get("LateralLength", float("nan"))) if r is not None else "",
               fmt_int(r.get("Months", float("nan"))) if r is not None else ""]
        for f in fl:
            e = eur_by[f.name]
            if wid in e.index:
                rr = e.loc[wid]
                rr = rr.iloc[0] if isinstance(rr, pd.DataFrame) else rr
                row.append(fmt_vol(rr["EUR"]) if bool(rr["in_tc"]) else "–")
            else:
                row.append("–")
        e = eur_by[prim.name]
        if wid in e.index:
            rr = e.loc[wid]
            rr = rr.iloc[0] if isinstance(rr, pd.DataFrame) else rr
            row.append(fmt_b(rr["b"]) if bool(rr["in_tc"]) else "–")
        else:
            row.append("–")
        body.append(row)
    pages = [body[i:i + rows_per_page] for i in range(0, len(body), rows_per_page)]
    shown = pages[:max_pages]
    out = []
    for p, chunk in enumerate(shown, 1):
        title = "Analog Wells" + (f" ({p} of {len(shown)})" if len(shown) > 1 else "")
        blocks = [table_block(GRID["left"], 1.6, col_w, [hdr] + chunk,
                              align=["r", "l", "l", "l", "r", "r"] + ["r"] * (len(fl) + 1),
                              row_h=row_h, size=9)]
        note = f"– means the well is not in that type curve. EURs are {norm}."
        if p == len(shown) and len(pages) > max_pages:
            note += (f" Showing {len(shown) * rows_per_page} of {len(body)} wells; the full list is in "
                     "the oneline export.")
        cap_y = min(1.6 + row_h * (len(chunk) + 1) + 0.08, 6.62)
        blocks.append(caption_block(GRID["left"], cap_y, 12.09, 0.32, note))
        out.append(Slide("CONTENT", "Appendix", dict(
            title=title, sub=f"Wells in the type curves; those in the {prim.name.lower()} type curve first, "
                             f"by {prim.name.lower()} EUR",
            title_size=SIZE["title"], sub_size=SIZE["subtitle"]), blocks))
    return out


# ---------------------------------------------------------------- matplotlib figures (SE style)
_MPL_FAMILY: List[str] = []


def _mpl_family() -> List[str]:
    if not _MPL_FAMILY:
        found = [fam for fam in ("Calibri", "Carlito", "Lato") if _find_font_file(fam)]
        _MPL_FAMILY.extend(found + ["DejaVu Sans"])
    return _MPL_FAMILY


def _mpl_rc() -> Dict[str, Any]:
    return {"font.family": _mpl_family(), "font.size": 9, "text.color": "#" + C["INK"],
            "axes.labelcolor": "#" + C["INK"], "axes.edgecolor": "#" + C["AXIS"],
            "xtick.color": "#" + C["GREY"], "ytick.color": "#" + C["GREY"], "axes.grid": False,
            "grid.color": "#" + C["GRID"], "grid.linewidth": 0.5, "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.75,
            "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 9, "legend.fontsize": 9,
            "figure.facecolor": "white", "axes.facecolor": "white"}


def _new_fig(w: float, h: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with matplotlib.rc_context(_mpl_rc()):
        fig = plt.figure(figsize=(w, h))
    return fig


def figure_png(fig, dpi: int = 220) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _log_limits(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0], dtype=float)
    if v.size == 0:
        return 1.0, 10.0
    lo = 10 ** math.floor(math.log10(max(v.min(), 1e-3)))
    hi = 10 ** math.ceil(math.log10(v.max()))
    return lo, max(hi, lo * 10)


def _tc_limits(spec) -> Tuple[float, float, float]:
    curves = spec["curves"]
    vals: List[float] = []
    for col in ("P10", "P50", "P90", "Mean"):
        if col in curves:
            vals += list(np.asarray(curves[col], dtype=float))
    for _, y, _ in spec["lines"]:
        vals += list(np.asarray(y, dtype=float))
    lo, hi = _log_limits(vals)
    p90 = np.asarray(curves["P90"], dtype=float) if "P90" in curves else np.array([])
    if p90.size and np.isfinite(p90).any():                 # keep the lower axis near the P90 tail
        lo = max(lo, 10 ** math.floor(math.log10(max(np.nanmin(p90[p90 > 0]), 1e-3))))
    xmax = float(np.nanmax(curves["t"])) if "t" in curves and len(curves) else 600.0
    return lo, hi, xmax


def tc_figure(spec, w: float, h: float):
    import matplotlib
    with matplotlib.rc_context(_mpl_rc()):
        fig = _new_fig(w, h)
        ax = fig.add_axes([0.1, 0.2, 0.87, 0.76])
        first = True
        for t, y, _ in spec["lines"]:
            ax.plot(t, np.clip(np.asarray(y, dtype=float), 1e-6, None), color="#" + C["NEUTRAL"], lw=0.6,
                    label=f"Analog wells ({spec['n_wells']})" if first else None, zorder=1)
            first = False
        curves = spec["curves"]
        styles = {"solid": "-", "dash": (0, (5, 2.5)), "dot": (0, (1, 1.6))}
        for col, colour, dash, lw in TC_SERIES:
            if col in curves:
                ax.plot(curves["t"], curves[col], color="#" + C[colour], lw=lw, ls=styles[dash],
                        label="Mean (average well)" if col == "Mean" else col, zorder=3 if col == "P50" else 2)
        lo, hi, xmax = _tc_limits(spec)
        ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_xlim(0, xmax)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xticks(np.arange(0, xmax + 1, 60))
        ax.grid(True, axis="y", which="major")
        ax.set_xlabel("Months on production")
        ax.set_ylabel(spec["y_title"])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, handlelength=2.6)
    return fig


def hist_bins(values: Sequence[float]) -> Tuple[List[str], List[int]]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [], []
    lo, hi = math.floor(v.min() * 20) / 20, math.ceil(v.max() * 20) / 20
    step = 0.05 if (hi - lo) <= 1.0 else 0.1
    if hi - lo < step:
        hi = lo + step
    edges = np.round(np.arange(lo, hi + step / 2, step), 3)
    if edges.size < 2:
        edges = np.array([lo, lo + step])
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    labels = [f"{edges[i]:.2f}" for i in range(len(counts))]
    return labels, [int(c) for c in counts]


def hist_figure(spec, w: float, h: float):
    import matplotlib
    labels, counts = hist_bins(spec["values"])
    with matplotlib.rc_context(_mpl_rc()):
        fig = _new_fig(w, h)
        ax = fig.add_axes([0.04, 0.2, 0.94, 0.74])
        x = np.arange(len(counts))
        ax.bar(x, counts, width=0.72, color="#" + C["BLUE"])
        for xi, c in zip(x, counts):
            if c:
                ax.text(xi, c, f"{c}", ha="center", va="bottom", fontsize=9, color="#" + C["INK"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 12 else 90)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlabel("b-factor (bin start)")
        ax.set_ylim(0, max(counts + [1]) * 1.18)
    return fig


def scatter_figure(spec, w: float, h: float):
    import matplotlib
    wells = spec["wells"]
    with matplotlib.rc_context(_mpl_rc()):
        fig = _new_fig(w, h)
        ax = fig.add_axes([0.1, 0.2, 0.87, 0.76])
        inn = wells[wells["in_tc"]]
        out = wells[~wells["in_tc"]]
        ax.scatter(inn["LateralLength"], inn["EUR"], s=26, color="#" + C["MAROON"], label="In type curve", zorder=3)
        ax.scatter(out["LateralLength"], out["EUR"], s=26, color="#" + C["MID"], label="Not in type curve", zorder=2)
        x0, x1 = _scatter_xrange(wells)
        ax.plot([x0, x1], [spec["upper"]] * 2, color="#" + C["RED"], ls=(0, (5, 2.5)), lw=1.25,
                label=f"Upper fence {fmt_vol(spec['upper'])}")
        if spec["lower"] > 0:
            ax.plot([x0, x1], [spec["lower"]] * 2, color="#" + C["RED"], ls=(0, (1, 1.6)), lw=1.25,
                    label=f"Lower fence {fmt_vol(spec['lower'])}")
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, max(float(wells["EUR"].max()) if len(wells) else 1, spec["upper"]) * 1.08)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(True, axis="y")
        ax.set_xlabel("Lateral length, ft")
        ax.set_ylabel(f"EUR, {spec['unit']}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    return fig


def _scatter_xrange(wells) -> Tuple[float, float]:
    lat = pd.to_numeric(wells["LateralLength"], errors="coerce").dropna() if len(wells) else pd.Series([], dtype=float)
    if lat.empty:
        return 0.0, 1.0
    lo, hi = float(lat.min()), float(lat.max())
    pad = max((hi - lo) * 0.05, 100.0)
    return math.floor((lo - pad) / 500) * 500, math.ceil((hi + pad) / 500) * 500


def _nice_log_ticks(lo: float, hi: float, target: int = 6) -> List[float]:
    """1-2-5 ticks (finer when the range is narrow) between lo and hi on a log axis."""
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.01)
    for subs in ((1,), (1, 2, 5), (1, 1.5, 2, 3, 5, 7), (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)):
        ticks = [m * 10 ** e for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1)
                 for m in subs if lo <= m * 10 ** e <= hi]
        if len(ticks) >= 3:
            if len(ticks) > target + 2:
                ticks = ticks[::int(math.ceil(len(ticks) / target))]
            return ticks
    return [lo, hi]


def _probit_figure_factory(eurs: np.ndarray, f: FluidReport):
    def build(w: float, h: float):
        return probit_figure(eurs, f, w, h)
    return build


def probit_figure(eurs: np.ndarray, f: FluidReport, w: float, h: float):
    """EUR and EUR per ft on log-probability axes (P10 at the top)."""
    import matplotlib
    from scipy import stats as _st
    v = np.asarray([x for x in eurs if np.isfinite(x) and x > 0], dtype=float)
    with matplotlib.rc_context(_mpl_rc()):
        fig = _new_fig(w, h)
        panels = [(v, f"EUR, {f.unit}"), (np.array([f.per_ft(x) for x in v]), f"EUR per ft, {f.per_ft_unit}")]
        for k, (vals, label) in enumerate(panels):
            ax = fig.add_axes([0.08 + k * 0.49, 0.14, 0.4, 0.8])
            if vals.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            vs = np.sort(vals)[::-1]
            z = _st.norm.ppf(1 - (np.arange(1, vs.size + 1) - 0.5) / vs.size)
            p10, p90 = np.percentile(vs, 90), np.percentile(vs, 10)
            ax.plot([p90, p10], [_st.norm.ppf(0.10), _st.norm.ppf(0.90)], color="#" + C["MAROON"], lw=1.6, zorder=2)
            ax.scatter(vs, z, s=24, color="#" + C["MAROON"], edgecolor="white", lw=0.6, zorder=3)
            ax.set_xscale("log")
            ax.set_yticks([_st.norm.ppf(x) for x in (0.9, 0.5, 0.1)])
            ax.set_yticklabels(["P10", "P50", "P90"])
            ax.set_ylim(min(z.min(), _st.norm.ppf(0.05)) - 0.2, max(z.max(), _st.norm.ppf(0.95)) + 0.2)
            lo, hi = float(vs.min()), float(vs.max())
            ax.set_xlim(lo / 1.15, hi * 1.15)
            ticks = _nice_log_ticks(lo / 1.15, hi * 1.15)
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: fmt_vol(x)))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(True, axis="both", which="major")
            ax.set_xlabel(label)
    return fig


CHART_FIGURES = {"tc": tc_figure, "hist": hist_figure, "scatter": scatter_figure}


# ---------------------------------------------------------------- PPTX renderer
def render_pptx(deck: Deck, out_path: str, template: str = TEMPLATE) -> str:
    """Write the deck as a PowerPoint file built on the SE template layouts."""
    from pptx import Presentation

    prs = Presentation(template)
    layouts = {layout.name: layout for layout in prs.slide_layouts}
    _set_layout_footer(layouts["CONTENT"], deck.footer_left, deck.footer_right)
    sections: List[Tuple[str, List[int]]] = []
    for n, sd in enumerate(deck.slides, 1):
        slide = prs.slides.add_slide(layouts[sd.layout])
        _fill_placeholders(slide, sd)
        if sd.layout == "CONTENT":
            _add_slide_number(slide, layouts["CONTENT"], n)
        for b in sd.blocks:
            _PPTX_BLOCK[b.kind](slide, b)
        if not sections or sections[-1][0] != sd.section:
            sections.append((sd.section, []))
        sections[-1][1].append(slide.slide_id)
    _write_sections(prs, sections)
    cp = prs.core_properties
    cp.title, cp.author, cp.subject = deck.title, COMPANY, deck.subject
    cp.last_modified_by = COMPANY
    prs.save(out_path)
    return out_path


def _rgb(key: str):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(C.get(key, key))


def _set_layout_footer(layout, left: str, right: str) -> None:
    """The footer lives on the CONTENT layout (SE: never place it by hand)."""
    from pptx.util import Emu
    boxes = sorted([s for s in layout.shapes if not s.is_placeholder and s.has_text_frame
                    and abs(Emu(s.top).inches - GRID["footerY"]) < 0.05], key=lambda s: s.left)
    for shape, text in zip(boxes, (left, right)):
        runs = [r for p in shape.text_frame.paragraphs for r in p.runs]
        if runs:
            runs[0].text = text
            for r in runs[1:]:
                r.text = ""


def _style_run(run, size, color="INK", bold=False, italic=False, face="body", spc=None):
    from pptx.util import Pt
    run.font.size = Pt(size)
    run.font.bold = bool(bold)
    run.font.italic = bool(italic)
    run.font.name = "Cambria" if face == "head" else "Calibri"
    run.font.color.rgb = _rgb(color)
    if spc:
        run._r.get_or_add_rPr().set("spc", str(int(spc * 100)))


def _zero_insets(tf, anchor="t") -> None:
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]


def _set_text(tf, paragraphs, size, color="INK", bold=False, italic=False, face="body", align="l",
              gap=0.0, spc=None):
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Pt
    al = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT, "j": PP_ALIGN.JUSTIFY}[align]
    for i, runs in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = al
        if gap and i < len(paragraphs) - 1:
            p.space_after = Pt(gap)
        for text, b in runs:
            r = p.add_run()
            r.text = text
            _style_run(r, size, color, bold or b, italic, face, spc)


def _fill_placeholders(slide, sd: Slide) -> None:
    f = sd.fields
    spec = {
        "COVER": [(101, "eyebrow", f.get("eyebrow_size", SIZE["coverEyebrow"]), "COVER_EYEBROW", True, False, "body", 2),
                  (102, "title", f.get("title_size", SIZE["coverTitle"]), "WHITE", True, False, "head", None),
                  (103, "sub", f.get("sub_size", SIZE["coverSub"]), "COVER_SUB", False, False, "body", None),
                  (104, "stats", f.get("stats_size", SIZE["coverStats"]), "WHITE", True, False, "body", None),
                  (105, "date", SIZE["coverDate"], "COVER_EYEBROW", False, False, "body", None)],
        "CONTENT": [(101, "title", f.get("title_size", SIZE["title"]), "MAROON", True, False, "head", None),
                    (102, "sub", f.get("sub_size", SIZE["subtitle"]), "GREY", False, True, "body", None)],
        "SECTION": [(101, "eyebrow", SIZE["dividerEyebrow"], "COVER_EYEBROW", True, False, "body", 2),
                    (102, "title", SIZE["dividerTitle"], "WHITE", True, False, "head", None),
                    (103, "sub", SIZE["dividerSub"], "COVER_SUB", False, False, "body", None)],
    }[sd.layout]
    for idx, key, size, color, bold, italic, face, spc in spec:
        try:
            ph = slide.placeholders[idx]
        except KeyError:
            continue
        text = f.get(key, "")
        if not text:
            ph._element.getparent().remove(ph._element)       # no empty prompt text on the slide
            continue
        tf = ph.text_frame
        _zero_insets(tf, "m" if (sd.layout == "CONTENT") else "t")
        _set_text(tf, [[(text, False)]], size, color, bold, italic, face, spc=spc)


def _add_slide_number(slide, layout, n: int) -> None:
    """Copy the layout's slide-number placeholder onto the slide (the template
    does the same), with the number filled in."""
    from pptx.oxml.ns import qn
    src = next((s for s in layout.placeholders if s.placeholder_format.type is not None
                and "SLIDE_NUMBER" in str(s.placeholder_format.type)), None)
    if src is None:
        return
    el = copy.deepcopy(src._element)
    for t in el.iter(qn("a:t")):
        t.text = str(n)
    ids = [int(c.get("id")) for c in slide.shapes._spTree.iter(qn("p:cNvPr")) if str(c.get("id", "")).isdigit()]
    for c in el.iter(qn("p:cNvPr")):
        c.set("id", str(max(ids + [1]) + 1))          # unique shape id on the slide
    slide.shapes._spTree.append(el)


def _write_sections(prs, sections) -> None:
    """PowerPoint sections that match the deck flow (SE section 6)."""
    from lxml import etree
    import uuid
    p_ns = "http://schemas.openxmlformats.org/presentationml/2006/main"
    p14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
    pres = prs.part._element
    ext_lst = pres.find(f"{{{p_ns}}}extLst")
    if ext_lst is None:
        ext_lst = etree.SubElement(pres, f"{{{p_ns}}}extLst")
    for ext in list(ext_lst):
        if ext.get("uri") == "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}":
            ext_lst.remove(ext)
    ext = etree.Element(f"{{{p_ns}}}ext", uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    sl = etree.SubElement(ext, f"{{{p14}}}sectionLst", nsmap={"p14": p14})
    for name, ids in sections:
        sec = etree.SubElement(sl, f"{{{p14}}}section", name=name, id="{" + str(uuid.uuid4()).upper() + "}")
        lst = etree.SubElement(sec, f"{{{p14}}}sldIdLst")
        for sid in ids:
            etree.SubElement(lst, f"{{{p14}}}sldId", id=str(sid))
    ext_lst.insert(0, ext)


def _strip_style(shape) -> None:
    """Drop the theme style reference python-pptx adds to autoshapes (no
    shadows or theme effects: SE 'no shadows, gradients or 3-D')."""
    from pptx.oxml.ns import qn
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _round_rect(slide, x, y, w, h, fill, line, lw=0.75, name="card"):
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    _strip_style(shp)
    shp.adjustments[0] = min(0.5, 0.08 / max(min(w, h), 1e-6))
    shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(fill)
    shp.line.color.rgb = _rgb(line); shp.line.width = Pt(lw)
    shp.name = name
    return shp


def _textbox(slide, x, y, w, h, name="text"):
    from pptx.util import Inches
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    return tb


def _pptx_text(slide, b: Block) -> None:
    d = b.data
    tb = _textbox(slide, b.x, b.y, b.w, b.h, d["name"])
    _zero_insets(tb.text_frame, d["valign"])
    _set_text(tb.text_frame, d["paras"], d["size"], d["color"], False, d["italic"], d["face"], d["align"], d["gap"])


def _pptx_bullets(slide, b: Block) -> None:
    from pptx.oxml.ns import qn
    from lxml import etree
    d = b.data
    top = b.y
    if d["heading"]:
        hb = _textbox(slide, b.x, b.y, b.w, 0.38, "panel heading")
        _zero_insets(hb.text_frame)
        _set_text(hb.text_frame, [[(d["heading"], True)]], SIZE["panelHeading"], "MAROON", face="head")
        top = b.y + 0.46
    tb = _textbox(slide, b.x, top, b.w, b.h - (top - b.y), f"panel bullets g{int(d['gap'])}")
    _zero_insets(tb.text_frame)
    paras = [[(lead + " ", True), (txt, False)] if lead else [(txt, False)] for lead, txt in d["items"]]
    _set_text(tb.text_frame, paras, d["size"], "INK", gap=d["gap"])
    for p in tb.text_frame.paragraphs:                        # maroon round bullets, hanging indent
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", "171450"); pPr.set("indent", "-171450")
        for tag in ("a:buClr", "a:buFont", "a:buChar"):
            el = pPr.find(qn(tag))
            if el is not None:
                pPr.remove(el)
        buClr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(buClr, qn("a:srgbClr"), val=C["MAROON"])
        etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
        etree.SubElement(pPr, qn("a:buChar"), char="•")


def _pptx_card(slide, b: Block) -> None:
    d = b.data
    _round_rect(slide, b.x, b.y, b.w, b.h, "WHITE", "LINE", name="card")
    for (yy, hh, text, size, color, bold, face, name, spc, pad) in (
            (b.y + 0.22, 0.8, d["value"], d["vsize"], d["color"], True, "head", "card value", None, 0.0),
            (b.y + 1.04, 0.32, d["label"], SIZE["kpiLabel"], d["color"], True, "body", "card label", 1, 0.0),
            (b.y + 1.38, 0.55, d["desc"], d["dsize"], "GREY", False, "body", "card description", None, 0.1),
            (b.y + 1.96, 0.28, d["foot"], d["fsize"], "MID", True, "body", "card footnote", None, 0.06)):
        if not text:
            continue
        tb = _textbox(slide, b.x + pad, yy, b.w - 2 * pad, hh, name)
        _zero_insets(tb.text_frame, "m" if name == "card value" else "t")
        _set_text(tb.text_frame, [[(text, False)]], size, color, bold, face=face, align="c", spc=spc)


def _pptx_panel(slide, b: Block) -> None:
    _round_rect(slide, b.x, b.y, b.w, b.h, "CARD", "LINE", name="panel")


def _cell_border(tcPr, color: str, w_emu: int = 6350) -> None:
    from pptx.oxml.ns import qn
    from lxml import etree
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        old = tcPr.find(qn(tag))
        if old is not None:
            tcPr.remove(old)
    for k, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        ln = etree.Element(qn(tag), w=str(w_emu), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr"), val=color)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")
        tcPr.insert(k, ln)


def _pptx_table(slide, b: Block) -> None:
    from pptx.util import Inches
    from pptx.oxml.ns import qn
    d = b.data
    rows, col_w, rh = d["rows"], d["col_w"], d["row_h"]
    gf = slide.shapes.add_table(len(rows), len(col_w), Inches(b.x), Inches(b.y), Inches(sum(col_w)),
                                Inches(rh * len(rows)))
    gf.name = d["name"]
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        tblPr.remove(sid)
    for ci, wv in enumerate(col_w):
        tbl.columns[ci].width = Inches(wv)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(rh)
        is_head = d["header"] and ri == 0
        body_i = ri - (1 if d["header"] else 0)
        fill = "MAROON" if is_head else ("WHITE" if body_i % 2 else "CARD")
        for ci, text in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = cell.margin_top = cell.margin_bottom = Inches(0.04)
            from pptx.enum.text import MSO_ANCHOR
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid(); cell.fill.fore_color.rgb = _rgb(fill)
            _cell_border(cell._tc.get_or_add_tcPr(), C["LINE"])
            tf = cell.text_frame
            tf.word_wrap = False
            al = d["align"][ci] if not is_head else ("l" if d["align"][ci] == "l" else d["align"][ci])
            _set_text(tf, [[(text, False)]], d["size"], "WHITE" if is_head else "INK",
                      bold=is_head or ri in d["bold_rows"], align=al)


def _chart_common(chart, legend=True) -> None:
    from pptx.util import Pt
    from pptx.enum.chart import XL_LEGEND_POSITION
    chart.font.name = "Calibri"
    chart.font.size = Pt(SIZE["chartAxis"])
    chart.font.color.rgb = _rgb("INK")
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(9)
    _no_rounded_corners(chart)


def _no_rounded_corners(chart) -> None:
    from pptx.oxml.ns import qn
    from lxml import etree
    cs = chart._chartSpace
    rc = cs.find(qn("c:roundedCorners"))
    if rc is None:
        rc = etree.Element(qn("c:roundedCorners"))
        # schema order: date1904?, lang?, roundedCorners?, ... chart
        pos = 0
        for i, child in enumerate(cs):
            if child.tag in (qn("c:date1904"), qn("c:lang")):
                pos = i + 1
        cs.insert(pos, rc)
    rc.set("val", "0")


def _axis_style(axis, title: Optional[str], numfmt: Optional[str] = None, gridlines: bool = False) -> None:
    from pptx.util import Pt
    from pptx.enum.chart import XL_TICK_MARK
    axis.format.line.color.rgb = _rgb("AXIS")
    axis.format.line.width = Pt(0.75)
    axis.tick_labels.font.size = Pt(SIZE["chartAxis"])
    axis.tick_labels.font.color.rgb = _rgb("GREY")
    axis.major_tick_mark = XL_TICK_MARK.OUTSIDE
    axis.minor_tick_mark = XL_TICK_MARK.NONE
    if numfmt:
        axis.tick_labels.number_format = numfmt
        axis.tick_labels.number_format_is_linked = False
    axis.has_major_gridlines = gridlines
    if gridlines:
        axis.major_gridlines.format.line.color.rgb = _rgb("GRID")
        axis.major_gridlines.format.line.width = Pt(0.5)
    if title:
        axis.has_title = True
        tf = axis.axis_title.text_frame
        tf.text = title
        r = tf.paragraphs[0].runs[0]
        r.font.size = Pt(SIZE["chartAxis"]); r.font.bold = False; r.font.color.rgb = _rgb("INK")
        r.font.name = "Calibri"


def _log_axis(axis, lo: float, hi: float) -> None:
    from pptx.oxml.ns import qn
    from lxml import etree
    sc = axis._element.find(qn("c:scaling"))
    lb = sc.find(qn("c:logBase"))
    if lb is None:
        lb = etree.Element(qn("c:logBase"))
        sc.insert(0, lb)
    lb.set("val", "10")
    axis.minimum_scale, axis.maximum_scale = lo, hi


def _series_line(ser, colour: str, width: float, dash: str = "solid") -> None:
    from pptx.util import Pt
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    from pptx.enum.chart import XL_MARKER_STYLE
    ser.format.line.color.rgb = _rgb(colour)
    ser.format.line.width = Pt(width)
    ser.format.line.dash_style = {"solid": MSO_LINE_DASH_STYLE.SOLID, "dash": MSO_LINE_DASH_STYLE.DASH,
                                  "dot": MSO_LINE_DASH_STYLE.ROUND_DOT}[dash]
    ser.marker.style = XL_MARKER_STYLE.NONE
    ser.smooth = False


def _pptx_chart(slide, b: Block) -> None:
    kind, spec = b.data["kind"], b.data["spec"]
    {"tc": _pptx_chart_tc, "hist": _pptx_chart_hist, "scatter": _pptx_chart_scatter}[kind](slide, b, spec)


def _pptx_chart_tc(slide, b: Block, spec) -> None:
    from pptx.chart.data import XyChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches
    cd = XyChartData()
    hist = cd.add_series(f"Analog wells ({spec['n_wells']})")
    for t, y, _ in spec["lines"]:
        for tt, yy in zip(np.asarray(t, dtype=float), np.asarray(y, dtype=float)):
            if np.isfinite(yy) and yy > 0:
                hist.add_data_point(float(tt), round(float(yy), 1))
        hist.add_data_point(float(np.asarray(t, dtype=float)[-1]) + 0.5 if len(t) else 0.0, None)   # gap
    curves = spec["curves"]
    order = []
    for col, colour, dash, lw in TC_SERIES:
        if col not in curves:
            continue
        s = cd.add_series("Mean (average well)" if col == "Mean" else col)
        for tt, yy in zip(curves["t"], curves[col]):
            if np.isfinite(yy) and yy > 0:
                s.add_data_point(float(tt), round(float(yy), 1))
        order.append((colour, dash, lw))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(b.x), Inches(b.y),
                                Inches(b.w), Inches(b.h), cd)
    gf.name = "type curve chart"
    ch = gf.chart
    _chart_common(ch)
    plot = ch.plots[0]
    _series_line(plot.series[0], "NEUTRAL", 0.75)
    for ser, (colour, dash, lw) in zip(list(plot.series)[1:], order):
        _series_line(ser, colour, lw, dash)
    lo, hi, xmax = _tc_limits(spec)
    va, xa = ch.value_axis, ch.category_axis
    _axis_style(va, spec["y_title"], "#,##0", gridlines=True)
    _log_axis(va, lo, hi)
    _axis_style(xa, "Months on production", "0")
    xa.minimum_scale, xa.maximum_scale, xa.major_unit = 0, xmax, 60


def _pptx_chart_hist(slide, b: Block, spec) -> None:
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
    from pptx.util import Inches, Pt
    labels, counts = hist_bins(spec["values"])
    if not counts:
        return
    cd = CategoryChartData()
    cd.categories = labels
    cd.add_series("Wells", counts)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(b.x), Inches(b.y), Inches(b.w),
                                Inches(b.h), cd)
    gf.name = "b-factor chart"
    ch = gf.chart
    _chart_common(ch, legend=False)
    plot = ch.plots[0]
    plot.gap_width = 40
    ser = plot.series[0]
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = _rgb("BLUE")
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(SIZE["chartLabel"]); dl.font.color.rgb = _rgb("INK")
    dl.number_format = '[=0]"";0'; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.maximum_scale = max(counts) * 1.18
    _axis_style(ch.category_axis, "b-factor (bin start)")


def _pptx_chart_scatter(slide, b: Block, spec) -> None:
    from pptx.chart.data import XyChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_MARKER_STYLE
    from pptx.util import Inches, Pt
    wells = spec["wells"]
    cd = XyChartData()
    inn = cd.add_series("In type curve")
    for x, y in zip(wells.loc[wells["in_tc"], "LateralLength"], wells.loc[wells["in_tc"], "EUR"]):
        if np.isfinite(x) and np.isfinite(y):
            inn.add_data_point(float(x), round(float(y), 2))
    out = cd.add_series("Not in type curve")
    for x, y in zip(wells.loc[~wells["in_tc"], "LateralLength"], wells.loc[~wells["in_tc"], "EUR"]):
        if np.isfinite(x) and np.isfinite(y):
            out.add_data_point(float(x), round(float(y), 2))
    x0, x1 = _scatter_xrange(wells)
    up = cd.add_series(f"Upper fence {fmt_vol(spec['upper'])}")
    up.add_data_point(x0, round(spec["upper"], 2)); up.add_data_point(x1, round(spec["upper"], 2))
    has_low = spec["lower"] > 0
    if has_low:
        lw_ = cd.add_series(f"Lower fence {fmt_vol(spec['lower'])}")
        lw_.add_data_point(x0, round(spec["lower"], 2)); lw_.add_data_point(x1, round(spec["lower"], 2))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(b.x), Inches(b.y), Inches(b.w),
                                Inches(b.h), cd)
    gf.name = "outlier chart"
    ch = gf.chart
    _chart_common(ch)
    sers = list(ch.plots[0].series)
    for ser, fill, line in ((sers[0], "MAROON", "MAROON"), (sers[1], "MID", "MID")):
        ser.format.line.fill.background()
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 6
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = _rgb(fill)
        ser.marker.format.line.color.rgb = _rgb(line); ser.marker.format.line.width = Pt(1)
    _series_line(sers[2], "RED", 1.25, "dash")
    if has_low:
        _series_line(sers[3], "RED", 1.25, "dot")
    va, xa = ch.value_axis, ch.category_axis
    _axis_style(va, f"EUR, {spec['unit']}", "#,##0", gridlines=True)
    va.minimum_scale = 0
    _axis_style(xa, "Lateral length, ft", "#,##0")
    xa.minimum_scale, xa.maximum_scale = x0, x1


def _pptx_image(slide, b: Block) -> None:
    from pptx.util import Inches
    fig = b.data["figure"](b.w, b.h)
    pic = slide.shapes.add_picture(io.BytesIO(figure_png(fig)), Inches(b.x), Inches(b.y), Inches(b.w), Inches(b.h))
    pic.name = "figure"
    pic._element.nvPicPr.cNvPr.set("descr", b.data.get("alt", ""))


_PPTX_BLOCK = {"text": _pptx_text, "bullets": _pptx_bullets, "card": _pptx_card, "panel": _pptx_panel,
               "table": _pptx_table, "chart": _pptx_chart, "image": _pptx_image}


# ---------------------------------------------------------------- PDF renderer
def render_pdf(deck: Deck, out_path: str) -> str:
    """Write the same slides as a PDF (13.33 x 7.5 in pages)."""
    from reportlab.pdfgen import canvas as _canvas
    fonts()
    W, H = SLIDE_W * 72, SLIDE_H * 72
    cv = _canvas.Canvas(out_path, pagesize=(W, H))
    cv.setTitle(deck.title); cv.setAuthor(COMPANY); cv.setSubject(deck.subject)
    for n, sd in enumerate(deck.slides, 1):
        _pdf_page(cv, sd, n, deck)
        cv.showPage()
    cv.save()
    return out_path


def _col(cv, key: str, fill=True):
    from reportlab.lib.colors import HexColor
    col = HexColor("#" + C.get(key, key))
    (cv.setFillColor if fill else cv.setStrokeColor)(col)


def _pdf_rect(cv, x, y, w, h, fill_key=None, stroke_key=None, lw=0.75, radius=0.0):
    H = SLIDE_H * 72
    if fill_key:
        _col(cv, fill_key)
    if stroke_key:
        _col(cv, stroke_key, fill=False)
        cv.setLineWidth(lw)
    args = (x * 72, H - (y + h) * 72, w * 72, h * 72)
    if radius:
        cv.roundRect(*args, radius * 72, stroke=1 if stroke_key else 0, fill=1 if fill_key else 0)
    else:
        cv.rect(*args, stroke=1 if stroke_key else 0, fill=1 if fill_key else 0)


def _pdf_para_style(size, color, face="body", bold=False, italic=False, align="l", gap=0.0, indent=0.0):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    return ParagraphStyle(
        "s", fontName=font_name(face, bold, italic), fontSize=size, leading=size * LEADING,
        textColor=HexColor("#" + C.get(color, color)), spaceAfter=gap,
        alignment={"l": TA_LEFT, "c": TA_CENTER, "r": TA_RIGHT, "j": TA_JUSTIFY}[align],
        leftIndent=indent, firstLineIndent=0, splitLongWords=False)


def _markup(runs, face="body") -> str:
    from xml.sax.saxutils import escape
    out = []
    for text, bold in runs:
        t = escape(text)
        out.append(f"<b>{t}</b>" if bold and face == "body" else t)
    return "".join(out)


def _pdf_paras(cv, x, y, w, h, paras, size, color="INK", face="body", bold=False, italic=False, align="l",
               valign="t", gap=0.0, bullet=False, spc=0.0):
    """Draw paragraphs into a box (inches, top-left origin)."""
    from reportlab.platypus import Paragraph
    H = SLIDE_H * 72
    st = _pdf_para_style(size, color, face, bold, italic, align, gap, indent=13.5 if bullet else 0)
    flow = []
    for runs in paras:
        txt = _markup(runs, face)
        if bullet:
            from reportlab.lib.colors import HexColor
            p = Paragraph(txt, st, bulletText="•")
            p.style = copy.copy(st)
            p.style.bulletFontName = "Helvetica"
            p.style.bulletFontSize = size
            p.style.bulletColor = HexColor("#" + C["MAROON"])
            p.style.bulletIndent = 0
        else:
            p = Paragraph(txt, st)
        flow.append(p)
    heights = []
    for p in flow:
        _, ph = p.wrap(w * 72, h * 72)
        heights.append(ph)
    total = sum(heights) + gap * max(len(flow) - 1, 0)
    top = H - y * 72
    if valign == "m":
        top = H - (y + h / 2) * 72 + total / 2
    elif valign == "b":
        top = H - (y + h) * 72 + total
    cur = top
    for p, ph in zip(flow, heights):
        p.drawOn(cv, x * 72, cur - ph)
        cur -= ph + gap


def _pdf_line(cv, text, x, y, w, h, size, color, face="body", bold=False, italic=False, align="l",
              valign="m", spc=0.0):
    """One line of text in a box (charSpace for the caps eyebrows)."""
    H = SLIDE_H * 72
    fn = font_name(face, bold, italic)
    _col(cv, color)
    from reportlab.pdfbase import pdfmetrics
    tw = pdfmetrics.stringWidth(text, fn, size) + spc * max(len(text) - 1, 0)
    asc = pdfmetrics.getAscent(fn, size) if hasattr(pdfmetrics, "getAscent") else size * 0.75
    desc = -pdfmetrics.getDescent(fn, size) if hasattr(pdfmetrics, "getDescent") else size * 0.25
    if valign == "m":
        base = H - (y + h / 2) * 72 - (asc - desc) / 2
    elif valign == "t":
        base = H - y * 72 - asc
    else:
        base = H - (y + h) * 72 + desc
    xx = x * 72 if align == "l" else ((x + w) * 72 - tw if align == "r" else (x + w / 2) * 72 - tw / 2)
    t = cv.beginText(xx, base)
    t.setFont(fn, size)
    t.setCharSpace(spc)
    t.textOut(text)
    cv.drawText(t)


def _pdf_image_file(cv, path, x, y, w, h):
    H = SLIDE_H * 72
    if os.path.exists(path):
        cv.drawImage(path, x * 72, H - (y + h) * 72, w * 72, h * 72, mask="auto", preserveAspectRatio=True)


def _pdf_page(cv, sd: Slide, n: int, deck: Deck) -> None:
    f = sd.fields
    if sd.layout in ("COVER", "SECTION"):
        _pdf_rect(cv, 0, 0, SLIDE_W, SLIDE_H, fill_key="MAROON")
    if sd.layout == "COVER":
        _pdf_image_file(cv, LOGO_COVER, 11.5, 0.62, 1.15, 1.15)
        _pdf_line(cv, f.get("eyebrow", ""), 0.92, 2.28, 11.0, 0.4, f.get("eyebrow_size", SIZE["coverEyebrow"]),
                  "COVER_EYEBROW", bold=True, valign="t", spc=2)
        _pdf_line(cv, f.get("title", ""), 0.9, 2.72, 11.5, 1.0, f.get("title_size", SIZE["coverTitle"]), "WHITE",
                  face="head", bold=True, valign="t")
        _pdf_line(cv, f.get("sub", ""), 0.92, 3.86, 10.6, 0.6, f.get("sub_size", SIZE["coverSub"]), "COVER_SUB", valign="t")
        _pdf_line(cv, f.get("stats", ""), 0.92, 4.78, 11.6, 0.4, f.get("stats_size", SIZE["coverStats"]), "WHITE",
                  bold=True, valign="t")
        _pdf_line(cv, f.get("date", ""), 0.92, 6.5, 11.0, 0.4, SIZE["coverDate"], "COVER_EYEBROW", valign="t")
    elif sd.layout == "SECTION":
        _pdf_image_file(cv, LOGO_COVER, 11.35, 0.72, 1.1, 1.1)
        _pdf_line(cv, f.get("eyebrow", ""), 0.92, 2.9, 8.0, 0.4, SIZE["dividerEyebrow"], "COVER_EYEBROW", bold=True,
                  valign="t", spc=2)
        _pdf_line(cv, f.get("title", ""), 0.9, 3.32, 11.0, 0.9, SIZE["dividerTitle"], "WHITE", face="head",
                  bold=True, valign="t")
        _pdf_paras(cv, 0.92, 4.34, 11.0, 0.6, [[(f.get("sub", ""), False)]], SIZE["dividerSub"], "COVER_SUB")
    else:
        _pdf_image_file(cv, LOGO, 12.16, 0.34, 0.8, 0.8)
        _pdf_line(cv, f.get("title", ""), 0.62, 0.4, 11.2, 0.6, f.get("title_size", SIZE["title"]), "MAROON",
                  face="head", bold=True)
        if f.get("sub"):
            _pdf_line(cv, f.get("sub", ""), 0.62, 1.02, 11.4, 0.4, f.get("sub_size", SIZE["subtitle"]), "GREY",
                      italic=True)
        _pdf_line(cv, deck.footer_left, 0.62, GRID["footerY"], 7.0, 0.3, SIZE["footnote"], "GREY", bold=True, valign="t")
        _pdf_line(cv, deck.footer_right, 7.6, GRID["footerY"], 4.3, 0.3, SIZE["footnote"], "MID", align="r", valign="t")
        _pdf_line(cv, str(n), 12.01, GRID["footerY"], 0.7, 0.3, SIZE["footnote"], "MID", align="r", valign="t")
    for b in sd.blocks:
        _PDF_BLOCK[b.kind](cv, b)


def _pdf_text(cv, b: Block) -> None:
    d = b.data
    _pdf_paras(cv, b.x, b.y, b.w, b.h, d["paras"], d["size"], d["color"], d["face"],
               italic=d["italic"], align=d["align"], valign=d["valign"], gap=d["gap"])


def _pdf_bullets(cv, b: Block) -> None:
    d = b.data
    top = b.y
    if d["heading"]:
        _pdf_paras(cv, b.x, b.y, b.w, 0.38, [[(d["heading"], True)]], SIZE["panelHeading"], "MAROON", face="head",
                   bold=True)
        top = b.y + 0.46
    paras = [[(lead + " ", True), (txt, False)] if lead else [(txt, False)] for lead, txt in d["items"]]
    _pdf_paras(cv, b.x, top, b.w, b.h - (top - b.y), paras, d["size"], "INK", gap=d["gap"], bullet=True)


def _pdf_card(cv, b: Block) -> None:
    d = b.data
    _pdf_rect(cv, b.x, b.y, b.w, b.h, fill_key="WHITE", stroke_key="LINE", radius=0.08)
    _pdf_line(cv, d["value"], b.x, b.y + 0.22, b.w, 0.8, d["vsize"], d["color"], face="head", bold=True, align="c")
    _pdf_line(cv, d["label"], b.x, b.y + 1.04, b.w, 0.32, SIZE["kpiLabel"], d["color"], bold=True, align="c",
              valign="t", spc=1)
    if d["desc"]:
        _pdf_paras(cv, b.x + 0.1, b.y + 1.38, b.w - 0.2, 0.55, [[(d["desc"], False)]], d["dsize"], "GREY", align="c")
    if d["foot"]:
        _pdf_line(cv, d["foot"], b.x + 0.06, b.y + 1.96, b.w - 0.12, 0.28, d["fsize"], "MID", bold=True, align="c",
                  valign="t")


def _pdf_panel(cv, b: Block) -> None:
    _pdf_rect(cv, b.x, b.y, b.w, b.h, fill_key="CARD", stroke_key="LINE", radius=0.08)


def _pdf_table(cv, b: Block) -> None:
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.colors import HexColor
    d = b.data
    H = SLIDE_H * 72
    rows, col_w, rh = d["rows"], d["col_w"], d["row_h"]
    t = Table(rows, colWidths=[w * 72 for w in col_w], rowHeights=[rh * 72] * len(rows))
    cmds = [("FONTNAME", (0, 0), (-1, -1), font_name("body")), ("FONTSIZE", (0, 0), (-1, -1), d["size"]),
            ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#" + C["INK"])),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#" + C["LINE"])),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2.88), ("RIGHTPADDING", (0, 0), (-1, -1), 2.88),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]
    for ci, a in enumerate(d["align"]):
        cmds.append(("ALIGN", (ci, 0), (ci, -1), {"l": "LEFT", "r": "RIGHT", "c": "CENTER"}[a]))
    start = 0
    if d["header"]:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), HexColor("#" + C["MAROON"])),
                 ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#" + C["WHITE"])),
                 ("FONTNAME", (0, 0), (-1, 0), font_name("body", bold=True))]
        start = 1
    for ri in range(start, len(rows)):
        fill = C["WHITE"] if (ri - start) % 2 else C["CARD"]
        cmds.append(("BACKGROUND", (0, ri), (-1, ri), HexColor("#" + fill)))
    for ri in d["bold_rows"]:
        cmds.append(("FONTNAME", (0, ri), (-1, ri), font_name("body", bold=True)))
    t.setStyle(TableStyle(cmds))
    tw, th = t.wrap(sum(col_w) * 72, rh * 72 * len(rows))
    t.drawOn(cv, b.x * 72, H - b.y * 72 - th)


def _pdf_figure(cv, b: Block, fig) -> None:
    from reportlab.lib.utils import ImageReader
    H = SLIDE_H * 72
    img = ImageReader(io.BytesIO(figure_png(fig, dpi=220)))
    cv.drawImage(img, b.x * 72, H - (b.y + b.h) * 72, b.w * 72, b.h * 72)


def _pdf_chart(cv, b: Block) -> None:
    fig = CHART_FIGURES[b.data["kind"]](b.data["spec"], b.w, b.h)
    _pdf_figure(cv, b, fig)


def _pdf_image(cv, b: Block) -> None:
    _pdf_figure(cv, b, b.data["figure"](b.w, b.h))


_PDF_BLOCK = {"text": _pdf_text, "bullets": _pdf_bullets, "card": _pdf_card, "panel": _pdf_panel,
              "table": _pdf_table, "chart": _pdf_chart, "image": _pdf_image}
