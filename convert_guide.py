"""Generate INSTALL_GUIDE.pdf from INSTALL_GUIDE.md using ReportLab."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, Preformatted, KeepTogether
)
from reportlab.lib.enums import TA_CENTER
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "INSTALL_GUIDE.pdf")

# ── Brand colours ─────────────────────────────────────────────
NAVY    = colors.HexColor("#1A2B3C")
TEAL    = colors.HexColor("#00B496")
LIGHT   = colors.HexColor("#F4F7F9")
GREY    = colors.HexColor("#6B7A8D")
CODE_BG = colors.HexColor("#1E2A35")
DIVIDER = colors.HexColor("#D0D8E0")

base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

TITLE  = S("title",  "Title",   fontSize=24, textColor=NAVY, spaceAfter=2,  fontName="Helvetica-Bold")
SUB    = S("sub",    "Normal",  fontSize=10, textColor=TEAL, spaceAfter=10, fontName="Helvetica")
H2     = S("h2",     "Heading2",fontSize=12, textColor=NAVY, spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold")
BODY   = S("body",   "Normal",  fontSize=10, textColor=colors.HexColor("#2C3E50"), leading=15, spaceAfter=4)
NOTE   = S("note",   "Normal",  fontSize=9,  textColor=GREY, leading=13, leftIndent=10, spaceAfter=3, fontName="Helvetica-Oblique")
FOOTER = S("footer", "Normal",  fontSize=8,  textColor=GREY, alignment=TA_CENTER)
CODE   = S("code",   "Normal",  fontName="Courier", fontSize=10, textColor=colors.white, leading=15)
PRE    = S("pre",    "Normal",  fontName="Courier", fontSize=9,  textColor=colors.white, leading=14)


def code_block(text, multiline=False):
    """Dark-background code block."""
    inner = Preformatted(text, PRE) if multiline else Paragraph(text, CODE)
    t = Table([[inner]], colWidths=[14.8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CODE_BG),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))
    return t


def step_box(number, title, body_flowables):
    num_para = Paragraph(
        f'<font color="#FFFFFF"><b>{number}</b></font>',
        ParagraphStyle("snum", fontSize=12, alignment=TA_CENTER,
                       textColor=colors.white, leading=15)
    )
    num_cell = Table([[num_para]], colWidths=[0.9*cm], rowHeights=[0.65*cm])
    num_cell.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), TEAL),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))

    title_para = Paragraph(title, S(f"st{number}", "Normal",
        fontSize=11, fontName="Helvetica-Bold", textColor=NAVY, leading=14))

    header = Table([[num_cell, title_para]], colWidths=[1.1*cm, 13.7*cm])
    header.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (1,0), (1,0),   8),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))

    inner = [[header]] + [[f] for f in body_flowables]
    card  = Table(inner, colWidths=[14.8*cm])
    card.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 9),
        ("RIGHTPADDING",  (0,0), (-1,-1), 9),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, DIVIDER),
    ]))
    return card


# ── Document ──────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2.3*cm, rightMargin=2.3*cm,
    topMargin=1.6*cm,  bottomMargin=1.6*cm,
    title="SE Tool — Installation Guide",
)

story = []

# Header bar
header_tbl = Table([[
    Paragraph("SE Tool", S("hdr", "Normal",
        fontSize=20, fontName="Helvetica-Bold", textColor=colors.white)),
    Paragraph("TCT — Tube Curve Tool", S("hdr2", "Normal",
        fontSize=10, textColor=colors.HexColor("#A8C8C0"), fontName="Helvetica")),
]], colWidths=[8*cm, 7.4*cm])
header_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), NAVY),
    ("TOPPADDING",    (0,0), (-1,-1), 12),
    ("BOTTOMPADDING", (0,0), (-1,-1), 12),
    ("LEFTPADDING",   (0,0), (0,0),   14),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN",         (1,0), (1,0),   "RIGHT"),
    ("RIGHTPADDING",  (1,0), (1,0),   14),
]))
story += [header_tbl, Spacer(1, 0.2*cm)]

story += [
    Paragraph("Installation Guide", TITLE),
    Paragraph("Follow these steps to install the SE Tool on your Windows machine.", SUB),
    HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=6),
]

# Requirements
req = Table([[
    Paragraph("Requirements", S("rh", "Normal",
        fontName="Helvetica-Bold", fontSize=10, textColor=NAVY)),
    Paragraph("Windows 10 or later &nbsp;&nbsp;•&nbsp;&nbsp; No Python or additional software needed",
        S("rb", "Normal", fontSize=10, textColor=GREY)),
]], colWidths=[3.5*cm, 11.3*cm])
req.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#EAF4F1")),
    ("TOPPADDING",    (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("LINEBEFORE",    (0,0), (0,-1),  3, TEAL),
]))
story += [req, Spacer(1, 0.3*cm)]

# Steps
story += [step_box("1", "Open the shared drive folder", [
    Paragraph("Navigate to the following network path in Windows Explorer:", BODY),
    code_block("Z:\\SIPC\\Software\\SE_TC_Tool"),
])]
story.append(Spacer(1, 0.15*cm))

story += [step_box("2", "Run the installer", [
    Paragraph("Double-click <b>install.bat</b>", BODY),
    Paragraph("If Windows shows a security warning, click <b>Run anyway</b>.", NOTE),
])]
story.append(Spacer(1, 0.15*cm))

story += [step_box("3", "Choose shortcut location", [
    Paragraph("The installer will ask where to save the shortcut:", BODY),
    code_block(
        "  Where would you like to save the shortcut?\n"
        "   [1] Desktop\n"
        "   [2] Custom location\n"
        "  Enter 1 or 2:",
        multiline=True),
    Paragraph("Type <b>1</b> and press <b>Enter</b> to add it to your Desktop.", NOTE),
])]
story.append(Spacer(1, 0.15*cm))

story += [step_box("4", "Launch the app", [
    Paragraph(
        "An <b>SE Tool</b> shortcut will appear on your Desktop. "
        "Double-click it to launch — the app opens automatically in your browser.",
        BODY),
])]
story.append(Spacer(1, 0.2*cm))

# Notes + footer — all on same page
story.append(KeepTogether([
    HRFlowable(width="100%", thickness=0.5, color=DIVIDER, spaceAfter=6),
    Paragraph("Notes", H2),
    Paragraph("• The app runs locally on your machine. No internet connection required.", BODY),
    Paragraph("• To uninstall, delete the Desktop shortcut.", BODY),
    Paragraph("• If the app does not open after a few seconds, check that your browser is not blocking <i>localhost</i>.", BODY),
    Spacer(1, 0.4*cm),
    HRFlowable(width="100%", thickness=0.5, color=DIVIDER, spaceAfter=6),
    Paragraph("For issues contact: ojrom", FOOTER),
]))

doc.build(story)
print(f"Saved {OUT}")
