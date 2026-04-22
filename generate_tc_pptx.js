"use strict";
// generate_tc_pptx.js  — called by Python bridge:
//   node generate_tc_pptx.js <json_input_path> <output_pptx_path>

const pptxgen = require("pptxgenjs");
const fs      = require("fs");

const [,, jsonPath, outPath] = process.argv;
if (!jsonPath || !outPath) {
  console.error("Usage: node generate_tc_pptx.js <json_path> <out_path>");
  process.exit(1);
}
const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

// ── Brand constants (no "#" prefix) ──────────────────────────────────────────
const SE_RED    = "953735";
const NAVY      = "1A2744";
const WHITE     = "FFFFFF";
const ROW_ALT   = "F5EDED";
const BODY_GRAY = "404040";
const MARK_GRAY = "AAAAAA";
const D_LIM     = 0.00417;

// ── Format helpers ────────────────────────────────────────────────────────────
function formatFt(n)       { return Math.round(n).toLocaleString() + " ft"; }
function formatEur(n, u)   { return n.toFixed(1) + " " + u; }
function fmtNum(n)         { return n >= 1000 ? Math.round(n).toLocaleString() : n.toFixed(2); }
function fmtInt(n)         { return Math.round(n).toLocaleString(); }
function fmtPct(v)         { return (v * 100).toFixed(1) + "%"; }
function capitalize(s)     { return s.charAt(0).toUpperCase() + s.slice(1); }
function formatDate(iso) {
  const d = new Date(iso);
  const MONTHS = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"];
  return MONTHS[d.getMonth()] + " " + d.getDate() + ", " + d.getFullYear();
}
function fluidColor(f) {
  return { oil:"2D6E2D", gas:"CC3333", water:"2244AA" }[f.toLowerCase()] || SE_RED;
}

// ── Modified Arps (hyperbolic + terminal exponential) ────────────────────────
function arpsHyperbolic(qiDaily, b, diPerMo, termDi, months) {
  months = months || 600;
  const qi = qiDaily * 30.4;          // daily → monthly
  const q  = [];
  const tSwitch = b > 0 && diPerMo > termDi
    ? (diPerMo - termDi) / (b * diPerMo * termDi)
    : Infinity;
  const qSwitch = tSwitch < Infinity
    ? qi / Math.pow(1 + b * diPerMo * tSwitch, 1 / b)
    : qi;

  for (let t = 1; t <= months; t++) {
    if (t >= tSwitch && tSwitch < Infinity) {
      q.push(qSwitch * Math.exp(-termDi * (t - tSwitch)));
    } else {
      q.push(qi / Math.pow(1 + b * diPerMo * t, 1 / b));
    }
  }
  return q;
}

// ── Slide chrome (footer, logo, page number, watermarks) ─────────────────────
function addSlideChrome(slide, pageNum) {
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 5.3, w: 10, h: 0.325,
    fill: { color: SE_RED }, line: { color: SE_RED },
  });
  slide.addShape(pres.ShapeType.ellipse, {
    x: 0.15, y: 5.05, w: 0.5, h: 0.5,
    fill: { color: NAVY }, line: { color: SE_RED, width: 1.5 },
  });
  slide.addText("SE", {
    x: 0.15, y: 5.12, w: 0.5, h: 0.35,
    fontSize: 9, bold: true, color: WHITE,
    align: "center", fontFace: "Georgia", margin: 0,
  });
  slide.addText(String(pageNum), {
    x: 9.5, y: 5.3, w: 0.4, h: 0.32,
    fontSize: 10, color: WHITE,
    align: "right", fontFace: "Calibri", margin: 0,
  });
  slide.addText("CONFIDENTIAL", {
    x: -0.05, y: 2.5, w: 0.6, h: 3,
    fontSize: 7, color: MARK_GRAY,
    align: "center", fontFace: "Calibri", rotate: 270,
  });
  slide.addText("CONFIDENTIAL", {
    x: 9.4, y: 2.5, w: 0.6, h: 3,
    fontSize: 7, color: MARK_GRAY,
    align: "center", fontFace: "Calibri", rotate: 90,
  });
}

function addSlideTitle(slide, titleText) {
  slide.addText(titleText, {
    x: 0.4, y: 0.18, w: 9.2, h: 0.52,
    fontSize: 24, bold: true, color: SE_RED,
    fontFace: "Georgia", margin: 0,
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 0.4, y: 0.73, w: 9.2, h: 0.03,
    fill: { color: SE_RED }, line: { color: SE_RED },
  });
}

// ── KV box ────────────────────────────────────────────────────────────────────
function drawKVBox(slide, title, rows, x, y, w, rowH) {
  rowH = rowH || 0.30;
  slide.addShape(pres.ShapeType.rect, {
    x: x, y: y, w: w, h: rowH,
    fill: { color: SE_RED }, line: { color: SE_RED },
  });
  slide.addText(title, {
    x: x, y: y, w: w, h: rowH,
    fontSize: 10, bold: true, color: WHITE,
    align: "left", valign: "middle",
    margin: [0, 0, 0, 6], fontFace: "Calibri",
  });
  rows.forEach(function(pair, i) {
    var label = pair[0], value = pair[1];
    var ry = y + rowH * (i + 1);
    var bg = i % 2 === 0 ? WHITE : ROW_ALT;
    slide.addShape(pres.ShapeType.rect, {
      x: x, y: ry, w: w, h: rowH,
      fill: { color: bg }, line: { color: "DDDDDD", width: 0.5 },
    });
    slide.addText(label, {
      x: x, y: ry, w: w * 0.5, h: rowH,
      fontSize: 9.5, color: BODY_GRAY,
      align: "left", valign: "middle",
      margin: [0, 0, 0, 6], fontFace: "Calibri",
    });
    slide.addText(String(value), {
      x: x + w * 0.5, y: ry, w: w * 0.5, h: rowH,
      fontSize: 9.5, color: BODY_GRAY,
      align: "center", valign: "middle",
      margin: 0, fontFace: "Calibri",
    });
  });
}

// ── Data table ────────────────────────────────────────────────────────────────
function drawTable(slide, headers, rows, x, y, colWidths, rowH) {
  rowH = rowH || 0.28;
  var cx = x;
  headers.forEach(function(hdr, i) {
    slide.addShape(pres.ShapeType.rect, {
      x: cx, y: y, w: colWidths[i], h: rowH,
      fill: { color: SE_RED }, line: { color: SE_RED },
    });
    slide.addText(hdr, {
      x: cx, y: y, w: colWidths[i], h: rowH,
      fontSize: 9, bold: true, color: WHITE,
      align: "center", valign: "middle",
      fontFace: "Calibri", margin: 0,
    });
    cx += colWidths[i];
  });
  rows.forEach(function(row, ri) {
    var ry = y + rowH * (ri + 1);
    var bg = ri % 2 === 0 ? WHITE : ROW_ALT;
    var cx2 = x;
    row.forEach(function(cell, ci) {
      slide.addShape(pres.ShapeType.rect, {
        x: cx2, y: ry, w: colWidths[ci], h: rowH,
        fill: { color: bg }, line: { color: "DDDDDD", width: 0.5 },
      });
      slide.addText(String(cell), {
        x: cx2, y: ry, w: colWidths[ci], h: rowH,
        fontSize: 9, color: BODY_GRAY,
        align: "center", valign: "middle",
        fontFace: "Calibri", margin: 0,
      });
      cx2 += colWidths[ci];
    });
  });
}

function buildEurRow(fluid) {
  return [
    capitalize(fluid.fluid),
    fluid.nTcWells + " / " + fluid.nTotal,
    formatEur(fluid.p90Eur, fluid.eurUnit),
    formatEur(fluid.p50Eur, fluid.eurUnit),
    formatEur(fluid.p10Eur, fluid.eurUnit),
    formatEur(fluid.meanEur, fluid.eurUnit),
  ];
}

// ── Presentation ──────────────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

// ── Slide 1: Title ────────────────────────────────────────────────────────────
(function() {
  var slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addShape(pres.ShapeType.ellipse, {
    x: 0.3, y: 0.2, w: 1.1, h: 1.1,
    fill: { color: NAVY }, line: { color: SE_RED, width: 3 },
  });
  slide.addText("SE", {
    x: 0.3, y: 0.42, w: 1.1, h: 0.5,
    fontSize: 22, bold: true, color: WHITE,
    fontFace: "Georgia", align: "center", margin: 0,
  });
  slide.addText(data.wellMeta.wellName, {
    x: 0.4, y: 1.8, w: 7, h: 0.85,
    fontSize: 36, bold: true, color: WHITE,
    fontFace: "Georgia", margin: 0,
  });
  slide.addText("TC Report – " + formatDate(data.wellMeta.reportDate), {
    x: 0.4, y: 2.7, w: 7, h: 0.6,
    fontSize: 24, italic: true, color: "CCCCCC",
    fontFace: "Georgia", margin: 0,
  });
  // Footer only (no page number on title slide)
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 5.3, w: 10, h: 0.325,
    fill: { color: SE_RED }, line: { color: SE_RED },
  });
})();

// ── Slide 2: Disclaimer ───────────────────────────────────────────────────────
(function() {
  var slide = pres.addSlide();
  addSlideTitle(slide, "Disclaimer");
  addSlideChrome(slide, 2);

  var text = [
    'The information contained in this confidential presentation (this “Presentation”) is provided for ',
    'informational and discussion purposes only and is not, and may not be relied on in any manner as, ',
    'legal, tax or investment advice or as an offer to sell or a solicitation of an offer to buy an interest ',
    'in any security. The information contained in this Presentation must be kept strictly confidential and ',
    'may not be reproduced or redistributed in any format without the approval of Schaper International ',
    'Petroleum Consulting, LLC (“SIPC”). In considering any performance data contained in this Presentation, ',
    'you should bear in mind that past or targeted performance is not indicative of future results, and there ',
    'can be no assurance that any investment will achieve comparable results or that target returns will be met. ',
    'In addition, there can be no assurance that any investment will achieve or be realized at the valuations ',
    'shown, as actual realized returns will depend on, among other factors, future operating results, the value ',
    'of assets and market conditions at the time of disposition, any related transaction costs and the timing ',
    'and manner of sale, all of which may differ from the assumptions on which the valuations contained in this ',
    'Presentation are based. Nothing contained in this Presentation should be deemed to be a prediction or ',
    'projection of future performance of any investment. Investors should make their own investigations and ',
    'evaluations of a potential investment and the information contained in this Presentation. Except where ',
    'otherwise indicated in this Presentation, the information provided in this Presentation is based on matters ',
    'as they exist as of the date of preparation and not as of any future date, and will not be updated or ',
    'otherwise revised to reflect information that subsequently becomes available, or circumstances existing or ',
    'changes occurring after the date hereof.',
  ].join('');

  slide.addText(text, {
    x: 0.4, y: 0.9, w: 9.2, h: 4.3,
    fontSize: 10.5, color: "333333",
    fontFace: "Calibri", align: "left", valign: "top", margin: 0,
  });
})();

// ── Slide 3: Analog Selection Methodology ─────────────────────────────────────
(function() {
  var slide = pres.addSlide();
  addSlideTitle(slide, "Analog Selection Methodology");
  addSlideChrome(slide, 3);

  var wm = data.wellMeta;
  var ac = data.analogCriteria;

  drawKVBox(slide, "Subject Location", [
    ["Well",      wm.wellName],
    ["County",    wm.county],
    ["Reservoir", wm.reservoir],
    ["AFE LL",    formatFt(wm.afeLateralLenFt)],
  ], 0.4, 0.85, 3.5);

  drawKVBox(slide, "Analog Pool Criteria", [
    ["# of Analogs",            String(ac.nAnalogs)],
    ["County",                  ac.counties],
    ["Reservoir",               ac.reservoir],
    ["LL Range",                ac.llRangeFt],
    ["LL Avg",                  formatFt(ac.llAvgFt)],
    ["Completion Vintage",      ac.compVintage],
    ["Avg Distance to Subject", String(ac.avgDistanceMi)],
  ], 0.4, 2.25, 3.5);

  // Map placeholder (only allowed placeholder)
  slide.addShape(pres.ShapeType.rect, {
    x: 4.2, y: 0.85, w: 5.4, h: 4.2,
    fill: { color: "CCCCCC" }, line: { color: "999999" },
  });
  slide.addText("[ Insert analog location map here ]", {
    x: 4.2, y: 2.5, w: 5.4, h: 0.6,
    fontSize: 11, color: "666666", italic: true,
    align: "center", fontFace: "Calibri", margin: 0,
  });
})();

// ── Slide 4: Summary ──────────────────────────────────────────────────────────
(function() {
  var slide = pres.addSlide();
  addSlideTitle(slide, "Summary – " + data.wellMeta.wellName + " TC");
  addSlideChrome(slide, 4);

  slide.addText("Analysis Parameters", {
    x: 0.4, y: 0.85, w: 9, h: 0.3,
    fontSize: 12, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });

  drawKVBox(slide, "Parameter", [
    ["Type Curve Name",      data.wellMeta.wellName + " TC"],
    ["Generated",            data.wellMeta.generatedDate],
    ["Normalization Length", formatFt(data.wellMeta.normLengthFt)],
    ["B-factor Range",       data.wellMeta.bFactorRange],
  ], 0.4, 1.18, 4.5);

  slide.addText("EUR Summary", {
    x: 0.4, y: 2.7, w: 9, h: 0.3,
    fontSize: 12, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });

  drawTable(slide,
    ["Fluid", "TC Wells", "P90 EUR", "P50 EUR", "P10 EUR", "Mean EUR"],
    [buildEurRow(data.oil), buildEurRow(data.gas), buildEurRow(data.water)],
    0.4, 3.02, [1.1, 0.9, 1.65, 1.65, 1.65, 1.65]
  );
})();

// ── Slides 5/8/11: Fluid Analysis ─────────────────────────────────────────────
[[data.oil, 5], [data.gas, 8], [data.water, 11]].forEach(function(pair) {
  var fluid = pair[0], pageNum = pair[1];
  if (!fluid || fluid.nTcWells === 0) return;

  var slide = pres.addSlide();
  addSlideTitle(slide, capitalize(fluid.fluid) + " Analysis");
  addSlideChrome(slide, pageNum);

  // Left: EUR stats
  slide.addText(capitalize(fluid.fluid) + " EUR Statistics (" + fluid.eurUnit + ")", {
    x: 0.4, y: 0.85, w: 3.5, h: 0.3,
    fontSize: 11, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });
  drawKVBox(slide, "Metric", [
    ["Wells in TC",   fluid.nTcWells + " / " + fluid.nTotal],
    ["Mean",          fluid.meanEur.toFixed(2)],
    ["Median (P50)",  fluid.p50Eur.toFixed(2)],
    ["P10",           fluid.p10Eur.toFixed(2)],
    ["P90",           fluid.p90Eur.toFixed(2)],
  ], 0.4, 1.17, 3.5);

  // Left: B-factor stats
  slide.addText("B-Factor Statistics", {
    x: 0.4, y: 3.0, w: 3.5, h: 0.3,
    fontSize: 11, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });
  drawKVBox(slide, "Metric", [
    ["Count",       String(fluid.bCount)],
    ["Mean",        fluid.bMean.toFixed(3)],
    ["Median",      fluid.bMedian.toFixed(3)],
    ["P10 / P90",   fluid.bP10.toFixed(3) + " / " + fluid.bP90.toFixed(3)],
  ], 0.4, 3.3, 3.5);

  // Right: TC parameters table
  slide.addText("Type Curve Parameters", {
    x: 4.1, y: 0.85, w: 5.6, h: 0.3,
    fontSize: 11, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });
  drawTable(slide,
    ["Parameter", "P90 (Low)", "P50 (Mid)", "P10 (High)"],
    [
      ["qi (" + fluid.qiUnit + ")",
        fmtNum(fluid.p90Qi), fmtNum(fluid.p50Qi), fmtNum(fluid.p10Qi)],
      ["b-factor",
        fluid.p90B.toFixed(3), fluid.p50B.toFixed(3), fluid.p10B.toFixed(3)],
      ["Di (per month)",
        fluid.p90Di.toFixed(4), fluid.p50Di.toFixed(4), fluid.p10Di.toFixed(4)],
      ["1st-Year Decline (%)",
        fmtPct(fluid.p90Decline1yr), fmtPct(fluid.p50Decline1yr), fmtPct(fluid.p10Decline1yr)],
      ["Terminal Di",
        fluid.terminalDi.toFixed(5)+"/mo", fluid.terminalDi.toFixed(5)+"/mo", fluid.terminalDi.toFixed(5)+"/mo"],
      ["EUR (" + fluid.eurUnit + ")",
        fluid.p90Eur.toFixed(2), fluid.p50Eur.toFixed(2), fluid.p10Eur.toFixed(2)],
    ],
    4.1, 1.17, [2.2, 1.1, 1.1, 1.2]
  );

  // Right: EUR table
  slide.addText(capitalize(fluid.fluid) + " EURs", {
    x: 4.1, y: 3.4, w: 5.6, h: 0.3,
    fontSize: 11, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });
  drawTable(slide,
    ["Percentile", "EUR (" + fluid.eurUnit + ")", "EUR (" + fluid.eurPerFtUnit + ")"],
    [
      ["P90",  fluid.p90Eur.toFixed(2),  fluid.p90EurPerFt.toFixed(2)],
      ["P50",  fluid.p50Eur.toFixed(2),  fluid.p50EurPerFt.toFixed(2)],
      ["P10",  fluid.p10Eur.toFixed(2),  fluid.p10EurPerFt.toFixed(2)],
      ["Mean", fluid.meanEur.toFixed(2), fluid.meanEurPerFt.toFixed(2)],
    ],
    4.1, 3.72, [1.5, 2.05, 2.05]
  );
});

// ── Slides 6/9/12: Fluid Analysis – Charts ───────────────────────────────────
[[data.oil, 6], [data.gas, 9], [data.water, 12]].forEach(function(pair) {
  var fluid = pair[0], pageNum = pair[1];
  if (!fluid || fluid.nTcWells === 0) return;

  var slide = pres.addSlide();
  addSlideTitle(slide, capitalize(fluid.fluid) + " Analysis – Charts");
  addSlideChrome(slide, pageNum);

  var fc = fluidColor(fluid.fluid);

  // ── B-Factor histogram ──
  slide.addText("B-Factor Distribution", {
    x: 0.4, y: 0.85, w: 4.3, h: 0.28,
    fontSize: 10, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });

  var bins      = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20];
  var binLabels = bins.slice(0, -1).map(function(b) { return b.toFixed(2); });
  var counts    = bins.slice(0, -1).map(function(lo, i) {
    return (fluid.bValues || []).filter(function(v) {
      return v >= lo && v < bins[i + 1];
    }).length;
  });

  slide.addChart(pres.ChartType.bar, [{
    name: capitalize(fluid.fluid) + " B-Factor",
    labels: binLabels,
    values: counts,
  }], {
    x: 0.4, y: 1.15, w: 4.3, h: 2.0,
    barDir: "col",
    chartColors: [fc],
    chartArea: { fill: { color: "F8F8F8" } },
    catAxisLabelFontSize: 8,
    valAxisLabelFontSize: 8,
    showLegend: false,
    showValue: false,
    valGridLine: { color: "E0E0E0", size: 0.5 },
    catGridLine: { style: "none" },
  });

  slide.addText("Median = " + fluid.bMedian.toFixed(2), {
    x: 0.4, y: 1.1, w: 4.3, h: 0.2,
    fontSize: 8, color: BODY_GRAY, align: "right",
    fontFace: "Calibri", margin: 0,
  });

  // ── Type Curve line chart ──
  slide.addText("Type Curve", {
    x: 5.0, y: 0.85, w: 4.6, h: 0.28,
    fontSize: 10, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });

  var termDi    = fluid.terminalDi || D_LIM;
  var smMonths  = [];
  for (var mi = 1; mi <= 100; mi++) { smMonths.push(mi * 6); }
  var smLabels  = smMonths.map(String);

  var p50q = arpsHyperbolic(fluid.p50Qi, fluid.p50B, fluid.p50Di, termDi);
  var p10q = arpsHyperbolic(fluid.p10Qi, fluid.p10B, fluid.p10Di, termDi);
  var p90q = arpsHyperbolic(fluid.p90Qi, fluid.p90B, fluid.p90Di, termDi);

  function sample(arr) {
    return smMonths.map(function(m) { return Math.max((arr[m - 1] || 0), 0.01); });
  }

  slide.addChart(pres.ChartType.line, [
    { name: "P50", labels: smLabels, values: sample(p50q) },
    { name: "P10", labels: smLabels, values: sample(p10q) },
    { name: "P90", labels: smLabels, values: sample(p90q) },
  ], {
    x: 5.0, y: 1.15, w: 4.6, h: 2.0,
    chartColors: [fc, fc, fc],
    lineSize: 2,
    valAxisLogScaleBase: 10,
    showLegend: true,
    legendPos: "tr",
    legendFontSize: 7,
    catAxisLabelFontSize: 7,
    valAxisLabelFontSize: 7,
    catAxisMinVal: 0,
    valGridLine: { color: "E0E0E0", size: 0.5 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: "F8F8F8" } },
  });

  // ── Probit plots ──
  slide.addText("Probit Plot", {
    x: 0.4, y: 3.28, w: 9.2, h: 0.28,
    fontSize: 10, bold: true, color: BODY_GRAY, fontFace: "Calibri",
  });

  var analogRows  = fluid.analogRows || [];
  var sortedEurs  = analogRows.map(function(r) { return r.eur; }).sort(function(a,b){return a-b;});
  var n           = sortedEurs.length;

  if (n > 0) {
    var positions = sortedEurs.map(function(_, i) { return (i + 0.5) / n; });

    // Left: EUR probit
    slide.addChart(pres.ChartType.scatter, [
      { name: "X",   values: sortedEurs },
      { name: "EUR", values: positions  },
    ], {
      x: 0.4, y: 3.6, w: 4.3, h: 1.6,
      chartColors: [fc],
      showLegend: false,
      catAxisTitle: "EUR (" + fluid.eurUnit + ")",
      valAxisTitle: "Cumulative Probability",
      catAxisLabelFontSize: 7,
      valAxisLabelFontSize: 7,
      chartArea: { fill: { color: "F8F8F8" } },
      valGridLine: { color: "E0E0E0", size: 0.5 },
      catGridLine: { color: "E0E0E0", size: 0.5 },
    });

    // Right: EUR/ft probit
    var sortedPft = analogRows.map(function(r) { return r.eurPerFt; }).sort(function(a,b){return a-b;});
    var posPft    = sortedPft.map(function(_, i) { return (i + 0.5) / n; });

    slide.addChart(pres.ChartType.scatter, [
      { name: "X",      values: sortedPft },
      { name: "EUR/ft", values: posPft    },
    ], {
      x: 5.0, y: 3.6, w: 4.6, h: 1.6,
      chartColors: [fc],
      showLegend: false,
      catAxisTitle: "EUR (" + fluid.eurPerFtUnit + ")",
      valAxisTitle: "Cumulative Probability",
      catAxisLabelFontSize: 7,
      valAxisLabelFontSize: 7,
      chartArea: { fill: { color: "F8F8F8" } },
      valGridLine: { color: "E0E0E0", size: 0.5 },
      catGridLine: { color: "E0E0E0", size: 0.5 },
    });
  }
});

// ── Slides 7/10/13: Fluid Analysis – Analogs ─────────────────────────────────
[[data.oil, 7], [data.gas, 10], [data.water, 13]].forEach(function(pair) {
  var fluid = pair[0], pageNum = pair[1];
  if (!fluid || fluid.nTcWells === 0) return;

  var slide = pres.addSlide();
  addSlideTitle(slide, capitalize(fluid.fluid) + " Analysis – Analogs");
  addSlideChrome(slide, pageNum);

  var rows = fluid.analogRows || [];
  var rowH = rows.length > 15 ? 0.225 : rows.length > 10 ? 0.255 : 0.31;

  drawTable(slide,
    [
      "API/UWI", "Well Name", "Lat Len (ft)",
      "qi (" + fluid.qiUnit + ")",
      "b", "Di (/mo)", "1yr Dec (%)", "EUR (" + fluid.eurUnit + ")"
    ],
    rows.map(function(r) {
      return [
        String(r.api),
        String(r.wellName),
        fmtInt(r.latLenFt),
        fmtNum(r.qi),
        r.b.toFixed(4),
        r.diPerMo.toFixed(4),
        r.decline1yrPct.toFixed(2),
        r.eur.toFixed(2),
      ];
    }),
    0.4, 0.88,
    [1.35, 1.85, 1.0, 0.85, 0.8, 0.75, 0.9, 0.7],
    rowH
  );
});

// ── Write file ────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: outPath })
  .then(function() { console.log("PPTX written:", outPath); })
  .catch(function(err) { console.error("PPTX error:", err); process.exit(1); });
