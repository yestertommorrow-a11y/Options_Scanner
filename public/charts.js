/*
 * charts.js - dependency-free SVG charts for the Options Scanner.
 *
 * ScanCharts.market(el, data)        market-level charts (expected move, IV vs RV, smile, activity)
 * ScanCharts.contract(el, data, row) concept charts for one contract (payoff, decay, odds, IV)
 *
 * Every chart has a crosshair/hover tooltip, a legend when it has 2+ series, and a
 * caption (class "cap") that explains the concept using the chart's own numbers.
 * Colours come from CSS variables --s1 (blue) and --s2 (orange), validated for CVD.
 */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  // ---------- maths ----------
  function erf(x) {
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const N = x => 0.5 * (1 + erf(x / Math.SQRT2));
  function bs(S, K, T, r, q, s, isCall) {
    if (T <= 1e-9 || s <= 0) return Math.max(isCall ? S - K : K - S, 0);
    const sq = Math.sqrt(T), d1 = (Math.log(S / K) + (r - q + s * s / 2) * T) / (s * sq), d2 = d1 - s * sq;
    return isCall ? S * Math.exp(-q * T) * N(d1) - K * Math.exp(-r * T) * N(d2)
                  : K * Math.exp(-r * T) * N(-d2) - S * Math.exp(-q * T) * N(-d1);
  }
  // Lognormal model of the price at expiry: density and P(price < x).
  function dist(S, T, r, q, s) {
    const v = s * Math.sqrt(T), m = Math.log(S) + (r - q - s * s / 2) * T;
    return {
      v,
      pdf: x => { const z = (Math.log(x) - m) / v; return Math.exp(-z * z / 2) / (x * v * Math.sqrt(2 * Math.PI)); },
      below: x => N((Math.log(x) - m) / v),
    };
  }
  const range = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

  // ---------- formatting ----------
  const usd = (v, d = 2) => (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const sUsd = v => (v > 0 ? "+" : "") + usd(v);
  const pc = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const compact = v => Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1).replace(/\.0$/, "") + "M"
                     : Math.abs(v) >= 1e4 ? (v / 1e3).toFixed(0) + "K"
                     : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(1).replace(/\.0$/, "") + "K" : String(Math.round(v));

  // ---------- DOM helpers ----------
  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function htmlEl(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(min, max, n) {
    const span = max - min || 1, raw = span / Math.max(n, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw))), err = raw / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }

  // Redraw charts when their card changes width.
  const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(entries => {
    for (const e of entries) {
      const h = e.target, w = Math.round(e.contentRect.width);
      if (h._draw && h._w && Math.abs(h._w - w) > 4) h._draw();
    }
  }) : null;

  // ---------- maximise ("Expand") ----------
  let maxed = null, backdrop = null;
  function closeMax() {
    if (!maxed) return;
    const h = maxed; maxed = null;
    h.classList.remove("max");
    if (backdrop) { backdrop.remove(); backdrop = null; }
    document.body.classList.remove("noscroll");
    if (h._draw) h._draw();
  }
  function openMax(host) {
    closeMax();
    maxed = host;
    backdrop = htmlEl("div", "chart-backdrop", document.body);
    backdrop.addEventListener("click", closeMax);
    host.classList.add("max");
    document.body.classList.add("noscroll");
    if (host._draw) host._draw();
    const b = host.querySelector(".maxbtn"); if (b) b.focus();
  }
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeMax(); });
  const chartHeight = (host, h) => host.classList.contains("max") ? Math.max(300, Math.min(window.innerHeight - 260, 640)) : (h || 230);

  // Card skeleton: title + expand button, plot area, legend, caption.
  function card(host, o) {
    host.textContent = "";
    host.classList.add("chart");
    const head = htmlEl("div", "chart-head", host);
    htmlEl("h3", "", head, o.title);
    const isMax = host.classList.contains("max");
    const btn = htmlEl("button", "maxbtn", head, isMax ? "✕ Close" : "⤢ Expand");
    btn.type = "button";
    btn.title = isMax ? "Close the enlarged chart (Esc)" : "Show this chart full-screen";
    btn.addEventListener("click", () => isMax ? closeMax() : openMax(host));
    if (o.sub) htmlEl("div", "chart-sub", host, o.sub);
    const plot = htmlEl("div", "plot", host);
    const tip = htmlEl("div", "tip", plot);
    tip.hidden = true;
    const named = (o.series || []).filter(s => s.name && !s.noLegend);
    if (named.length >= 2) {
      const lg = htmlEl("div", "legend", host);
      for (const s of named) {
        const it = htmlEl("span", "", lg);
        const key = htmlEl("span", o.bars ? "key rect" : "key", it);
        key.style.background = s.color;
        it.appendChild(document.createTextNode(s.name));
      }
    }
    if (o.caption) htmlEl("p", "cap", host, o.caption);
    host._w = Math.round(host.getBoundingClientRect().width);
    if (ro) ro.observe(host);
    return { plot, tip, W: Math.max(260, plot.clientWidth || 520) };
  }

  function showTip(tip, plot, px, py, W, header, rows) {
    tip.textContent = "";
    htmlEl("div", "tip-h", tip, header);
    for (const r of rows) {
      const line = htmlEl("div", "tip-r", tip);
      if (r.color) { const k = htmlEl("span", "key", line); k.style.background = r.color; }
      htmlEl("b", "", line, r.value);
      if (r.label) htmlEl("span", "tip-l", line, r.label);
    }
    tip.hidden = false;
    const tw = tip.offsetWidth, pw = plot.clientWidth;
    let left = px * (pw / W) + 14;
    if (left + tw > pw) left = px * (pw / W) - tw - 14;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = Math.max(0, py - 10) + "px";
  }

  // ---------- line chart ----------
  function lineChart(host, o) {
    host._draw = () => lineChart(host, o);
    const { plot, tip, W } = card(host, o);
    const H = chartHeight(host, o.height);
    const m = { l: o.hideY ? 14 : 54, r: 16, t: 30, b: 40 };
    const all = o.series.flatMap(s => s.pts);
    let x0 = o.xDomain ? o.xDomain[0] : Math.min(...all.map(p => p[0]));
    let x1 = o.xDomain ? o.xDomain[1] : Math.max(...all.map(p => p[0]));
    const yv = all.map(p => p[1]).concat((o.hlines || []).map(h => h.y), (o.dots || []).map(d => d.y)).filter(isFinite);
    let y0 = Math.min(...yv), y1 = Math.max(...yv);
    if (o.yZero) y0 = Math.min(0, y0);
    const pad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.1 || 1;
    y1 += pad; if (!(o.yZero && y0 === 0)) y0 -= pad;
    const pw = W - m.l - m.r;
    const sx = x => m.l + (o.xReverse ? (x1 - x) : (x - x0)) / (x1 - x0) * pw;
    // Stack reference-line labels into rows so they never overlap (matters on phones).
    const vls = (o.vlines || []).filter(v => v.x >= Math.min(x0, x1) && v.x <= Math.max(x0, x1))
      .map(v => {
        const X = sx(v.x), w = v.label.length * 6.3;
        const anchor = X - w / 2 < 2 ? "start" : X + w / 2 > W - 2 ? "end" : "middle";
        const a = anchor === "start" ? X : anchor === "end" ? X - w : X - w / 2;
        return { ...v, X, anchor, a, b: a + w };
      }).sort((p, q2) => p.X - q2.X);
    const rowEnd = [];
    for (const v of vls) {
      let r = rowEnd.findIndex(e => e < v.a - 6);
      if (r < 0) { r = rowEnd.length; rowEnd.push(0); }
      rowEnd[r] = v.b; v.row = r;
    }
    m.t = Math.max(m.t, 10 + rowEnd.length * 12);
    const ph = H - m.t - m.b;
    const sy = y => m.t + (1 - (y - y0) / (y1 - y0)) * ph;
    const fx = o.fmtX || (v => v.toFixed(2)), fy = o.fmtY || (v => v.toFixed(2));

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.title }, plot);
    plot.insertBefore(svg, tip);
    for (const t of niceTicks(y0, y1, 4)) {
      if (t < y0 || t > y1) continue;
      svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grid" }, svg);
      if (!o.hideY) svgEl("text", { x: m.l - 6, y: sy(t) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = fy(t);
    }
    for (const t of niceTicks(Math.min(x0, x1), Math.max(x0, x1), Math.max(2, Math.floor(pw / 90)))) {
      svgEl("text", { x: sx(t), y: H - m.b + 16, class: "tick", "text-anchor": "middle" }, svg).textContent = fx(t);
    }
    svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    if (o.xTitle) svgEl("text", { x: m.l + pw / 2, y: H - 6, class: "axt", "text-anchor": "middle" }, svg).textContent = o.xTitle;
    if (o.yTitle) svgEl("text", { x: m.l - (o.hideY ? 0 : 46), y: m.t - 8, class: "axt" }, svg).textContent = o.yTitle;
    if (y0 < 0 && y1 > 0) svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), class: "zero" }, svg);

    const path = pts => pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("");
    const base = sy(Math.max(y0, 0));
    for (const f of o.fills || []) {
      const pts = o.series[f.series].pts.filter(p => p[0] >= f.from && p[0] <= f.to);
      if (pts.length < 2) continue;
      svgEl("path", { d: path(pts) + `L${sx(pts[pts.length - 1][0])},${base}L${sx(pts[0][0])},${base}Z`,
        fill: f.color || o.series[f.series].color, "fill-opacity": f.opacity || 0.22 }, svg);
    }
    for (const s of o.series) {
      if (s.area) svgEl("path", { d: path(s.pts) + `L${sx(s.pts[s.pts.length - 1][0])},${base}L${sx(s.pts[0][0])},${base}Z`,
        fill: s.color, "fill-opacity": 0.1 }, svg);
      svgEl("path", { d: path(s.pts), fill: "none", stroke: s.color, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    }
    for (const h of o.hlines || []) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(h.y), y2: sy(h.y), class: "ref" }, svg);
      if (h.label) svgEl("text", { x: W - m.r, y: sy(h.y) - 5, class: "rlabel", "text-anchor": "end" }, svg).textContent = h.label;
    }
    for (const v of vls) {
      svgEl("line", { x1: v.X, x2: v.X, y1: 14 + v.row * 12, y2: H - m.b, class: "ref" }, svg);
      svgEl("text", { x: v.X, y: 10 + v.row * 12, class: "rlabel", "text-anchor": v.anchor }, svg).textContent = v.label;
    }
    for (const d of o.dots || []) {
      if (!isFinite(d.x) || !isFinite(d.y) || d.x < Math.min(x0, x1) || d.x > Math.max(x0, x1)) continue;
      const X = sx(d.x), Y = sy(d.y);
      svgEl("circle", { cx: X, cy: Y, r: 5, fill: d.color, class: "dot" }, svg);
      if (d.label) {
        const right = X > m.l + pw * 0.62;
        const below = d.below ? Y + 20 < H - m.b : Y - 18 < m.t;  // flip when the label would hit an edge
        svgEl("text", { x: X + (right ? -9 : 9), y: Y + (below ? 15 : -8), class: "dlabel", "text-anchor": right ? "end" : "start" }, svg).textContent = d.label;
      }
    }

    // hover layer: crosshair snaps to the nearest x; tooltip lists every series there
    const cross = svgEl("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" }, svg);
    const marks = o.series.map(s => svgEl("circle", { r: 4, fill: s.color, class: "dot", visibility: "hidden" }, svg));
    const hit = svgEl("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
    const nearest = (pts, x) => pts.reduce((b, p) => Math.abs(p[0] - x) < Math.abs(b[0] - x) ? p : b, pts[0]);
    hit.addEventListener("pointermove", e => {
      const rect = svg.getBoundingClientRect(), px = (e.clientX - rect.left) * (W / rect.width);
      const xv = o.xReverse ? x1 - (px - m.l) / pw * (x1 - x0) : x0 + (px - m.l) / pw * (x1 - x0);
      const near = o.series.map(s => nearest(s.pts, xv));
      const X = sx(near[0][0]);
      cross.setAttribute("x1", X); cross.setAttribute("x2", X); cross.setAttribute("visibility", "visible");
      near.forEach((p, i) => { marks[i].setAttribute("cx", sx(p[0])); marks[i].setAttribute("cy", sy(p[1])); marks[i].setAttribute("visibility", "visible"); });
      const t = o.tip ? o.tip(near[0][0], near) : { header: fx(near[0][0]), rows: near.map((p, i) => ({ color: o.series[i].color, value: fy(p[1]), label: o.series[i].name })) };
      showTip(tip, plot, X, (e.clientY - rect.top), W, t.header, t.rows);
    });
    hit.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden"); marks.forEach(c => c.setAttribute("visibility", "hidden")); tip.hidden = true;
    });
  }

  // ---------- grouped bar chart ----------
  function barChart(host, o) {
    host._draw = () => barChart(host, o);
    const { plot, tip, W } = card(host, { ...o, bars: true });
    const H = chartHeight(host, o.height), m = { l: 48, r: 12, t: 30, b: 40 };
    const n = o.cats.length, k = o.series.length, pw = W - m.l - m.r, ph = H - m.t - m.b;
    const ymax = Math.max(1, ...o.series.flatMap(s => s.vals)) * 1.12;
    const sy = y => m.t + (1 - y / ymax) * ph;
    const gw = pw / n, bw = Math.max(1, Math.min(24, (gw - 2) / k - 2));
    const fy = o.fmtY || compact;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.title }, plot);
    plot.insertBefore(svg, tip);
    for (const t of niceTicks(0, ymax, 4)) {
      if (t > ymax) continue;
      svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grid" }, svg);
      svgEl("text", { x: m.l - 6, y: sy(t) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = fy(t);
    }
    svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    const every = Math.max(1, Math.ceil(n / Math.max(1, pw / 56)));
    const hov = svgEl("rect", { y: m.t, height: ph, width: gw, class: "hov", visibility: "hidden" }, svg);
    o.cats.forEach((c, i) => {
      const gx = m.l + i * gw, inner = k * (bw + 2) - 2, start = gx + (gw - inner) / 2;
      o.series.forEach((s, j) => {
        const v = s.vals[i]; if (!(v > 0)) return;
        const x = start + j * (bw + 2), y = sy(v), h = H - m.b - y, r = Math.min(4, bw / 2, h);
        svgEl("path", { d: `M${x},${H - m.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${H - m.b}Z`, fill: s.color }, svg);
      });
      if (o.valueLabels) o.series.forEach((s, j) => {
        svgEl("text", { x: start + j * (bw + 2) + bw / 2, y: sy(s.vals[i]) - 6, class: "rlabel", "text-anchor": "middle" }, svg).textContent = fy(s.vals[i]);
      });
      if (o.flags && o.flags[i]) {
        const top = Math.min(...o.series.map(s => sy(s.vals[i] || 0)));
        svgEl("text", { x: gx + gw / 2, y: top - 4, class: "flag", "text-anchor": "middle" }, svg).textContent = "▲";
      }
      if (i % every === 0) svgEl("text", { x: gx + gw / 2, y: H - m.b + 16, class: "tick", "text-anchor": "middle" }, svg).textContent = (o.fmtCat || String)(c);
    });
    if (o.marker != null) {
      const X = m.l + o.marker.pos * gw;
      svgEl("line", { x1: X, x2: X, y1: m.t - 4, y2: H - m.b, class: "ref" }, svg);
      svgEl("text", { x: X, y: 10, class: "rlabel", "text-anchor": "middle" }, svg).textContent = o.marker.label;
    }
    if (o.xTitle) svgEl("text", { x: m.l + pw / 2, y: H - 6, class: "axt", "text-anchor": "middle" }, svg).textContent = o.xTitle;
    if (o.yTitle) svgEl("text", { x: 2, y: 12, class: "axt" }, svg).textContent = o.yTitle;
    const hit = svgEl("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", e => {
      const rect = svg.getBoundingClientRect(), px = (e.clientX - rect.left) * (W / rect.width);
      const i = Math.min(n - 1, Math.max(0, Math.floor((px - m.l) / gw)));
      hov.setAttribute("x", m.l + i * gw); hov.setAttribute("visibility", "visible");
      const t = o.tip ? o.tip(i) : { header: String(o.cats[i]), rows: o.series.map(s => ({ color: s.color, value: fy(s.vals[i]), label: s.name })) };
      showTip(tip, plot, m.l + (i + 0.5) * gw, e.clientY - rect.top, W, t.header, t.rows);
    });
    hit.addEventListener("pointerleave", () => { hov.setAttribute("visibility", "hidden"); tip.hidden = true; });
  }

  const C1 = "var(--s1)", C2 = "var(--s2)";

  // =====================================================================
  // Market-level charts
  // =====================================================================
  function market(root, d) {
    root.textContent = "";
    const s = d.summary, S = s.spot, r = d.r, q = d.q, cfg = d.config;
    if (s.atm_iv == null || !d.chain.length) return;
    const grid = htmlEl("div", "charts", root);

    // 1. Expected move as a probability curve
    {
      const T = Math.max(s.dte, 0.5) / 365, D = dist(S, T, r, q, s.atm_iv), em = s.exp_move;
      const xs = range(S * Math.exp(-3.2 * D.v), S * Math.exp(3.2 * D.v), 161);
      const pts = xs.map(x => [x, D.pdf(x)]);
      const pIn = D.below(S + em) - D.below(S - em);
      const exp = (d.expiries.find(e => e.dte === s.dte) || {}).expiry || `${s.dte} days`;
      lineChart(htmlEl("div", "", grid), {
        title: "Where the price could be at expiry",
        sub: `${exp} · using ATM IV ${pc(s.atm_iv)}`,
        series: [{ name: "Likelihood", color: C1, pts }],
        fills: [{ series: 0, from: S - em, to: S + em, opacity: 0.3 }],
        vlines: [{ x: S, label: `Today ${usd(S)}` }, { x: S - em, label: `−1σ ${usd(S - em)}` }, { x: S + em, label: `+1σ ${usd(S + em)}` }],
        hideY: true, yZero: true, xTitle: "Price at expiry", fmtX: v => usd(v, 0),
        tip: x => ({ header: `Price ${usd(x)}`, rows: [
          { value: pc(D.below(x)), label: "chance of ending below" },
          { value: pc(1 - D.below(x)), label: "chance of ending above" }] }),
        caption: `The curve's height shows how likely each price is on ${exp}, if the stock moves the way its ${pc(s.atm_iv)} implied volatility says. ` +
          `The shaded band (${usd(S - em)} to ${usd(S + em)}) is the "1-sigma expected move" and holds about ${pc(pIn, 0)} of outcomes. ` +
          `Higher IV or more days left would stretch the whole curve wider, which is exactly why those options cost more.`,
      });
    }

    // 2. IV vs realised volatility
    if (s.rv20 != null) {
      barChart(htmlEl("div", "", grid), {
        title: "What options expect vs what actually happened",
        sub: `IV ÷ RV = ${s.iv_rv.toFixed(2)}`,
        cats: ["Implied (ATM IV)", "Realised (last 20 days)"],
        series: [{ name: "Volatility", color: C1, vals: [s.atm_iv * 100, s.rv20 * 100] }],
        fmtY: v => v.toFixed(1).replace(".0", "") + "%", yTitle: "Annualised volatility", valueLabels: true,
        tip: i => ({ header: i ? "Realised volatility" : "Implied volatility", rows: [{ value: pc(i ? s.rv20 : s.atm_iv), label: i ? "actual movement, annualised" : "movement priced into options" }] }),
        caption: s.iv_rv > 1
          ? `Options are priced for ${pc(s.atm_iv)} a year of movement, but the stock has actually moved ${pc(s.rv20)}. Buyers are paying for more movement than recently seen; if calm continues, time decay works for option sellers.`
          : `Options are priced for ${pc(s.atm_iv)} a year of movement, less than the ${pc(s.rv20)} the stock has actually moved lately. Options look cheap versus recent movement; if that movement continues, buyers can outrun time decay.`,
      });
    }

    // shared expiry picker for the smile and activity charts
    const ctl = htmlEl("div", "chartctl", root);
    const lab = htmlEl("label", "", ctl, "Expiry for the two charts below ");
    const sel = htmlEl("select", "", lab);
    for (const e of d.expiries) { const op = htmlEl("option", "", sel, `${e.expiry} (${e.dte}d)`); op.value = e.expiry; }
    sel.value = (d.expiries.find(e => e.dte === s.dte) || d.expiries[0]).expiry;
    const lab2 = htmlEl("label", "", ctl, " Activity for ");
    const typ = htmlEl("select", "", lab2);
    [["c", "Calls"], ["p", "Puts"], ["b", "Calls + puts"]].forEach(([v, t]) => { const op = htmlEl("option", "", typ, t); op.value = v; });
    typ.value = "b";
    const grid2 = htmlEl("div", "charts", root);
    const smileHost = htmlEl("div", "", grid2), actHost = htmlEl("div", "", grid2);

    function drawExpiry() {
      const ex = sel.value, e = d.expiries.find(x => x.expiry === ex);
      const T = Math.max(e.dte, 0.5) / 365, band = Math.max(3 * s.atm_iv * Math.sqrt(T), 0.05);
      const lo = S * Math.exp(-band), hi = S * Math.exp(band);
      const rows = d.chain.filter(c => c[0] === ex && c[2] >= lo && c[2] <= hi);
      const lo2 = S * Math.exp(-band * 2 / 3), hi2 = S * Math.exp(band * 2 / 3);

      // 3. Volatility smile (out-of-the-money side of each type)
      const calls = rows.filter(c => c[1] === "c" && c[2] >= S * 0.98).sort((a, b) => a[2] - b[2]).map(c => [c[2], c[3] * 100]);
      const puts = rows.filter(c => c[1] === "p" && c[2] <= S * 1.02).sort((a, b) => a[2] - b[2]).map(c => [c[2], c[3] * 100]);
      if (calls.length > 1 && puts.length > 1) {
        const dots = [];
        if (e.dte === s.dte && s.p25_iv != null) {
          dots.push({ x: s.p25_strike, y: s.p25_iv * 100, color: C2, label: "25Δ put" });
          dots.push({ x: s.c25_strike, y: s.c25_iv * 100, color: C1, label: "25Δ call" });
        }
        const pAvg = puts.slice(0, Math.ceil(puts.length / 3)).reduce((a, p) => a + p[1], 0) / Math.ceil(puts.length / 3);
        const cAvg = calls.slice(-Math.ceil(calls.length / 3)).reduce((a, p) => a + p[1], 0) / Math.ceil(calls.length / 3);
        lineChart(smileHost, {
          title: "Volatility smile: IV at each strike",
          sub: `${ex} · out-of-the-money side of each type`,
          series: [{ name: "Calls (above today's price)", color: C1, pts: calls }, { name: "Puts (below today's price)", color: C2, pts: puts }],
          vlines: [{ x: S, label: `Today ${usd(S)}` }], dots,
          xTitle: "Strike", yTitle: "IV %", fmtX: v => usd(v, 0), fmtY: v => v.toFixed(0) + "%",
          tip: (x, near) => ({ header: `Strike near ${usd(x, 0)}`, rows: near.map((p, i) => ({ color: i ? C2 : C1, value: p[1].toFixed(1) + "%", label: `${i ? "put" : "call"} IV at ${usd(p[0], 0)}` })) }),
          caption: `Each line is the volatility the market charges at each strike. Far puts average about ${pAvg.toFixed(0)}% IV versus ${cAvg.toFixed(0)}% for far calls` +
            (pAvg > cAvg ? ": investors pay extra for crash protection, so the curve tilts up to the left. That tilt is skew." : ": calls are unusually expensive, so the market is leaning toward big upside moves.") +
            (dots.length ? ` The 25-delta skew card is the height gap between the two dots (${(s.p25_iv * 100).toFixed(1)}% − ${(s.c25_iv * 100).toFixed(1)}%).` : "") +
            " It also explains Rich/Cheap: the reference IV is the middle of this curve, so far puts often look Rich.",
        });
      } else smileHost.textContent = "";

      // 4. Volume vs open interest by strike
      const want = typ.value;
      const by = new Map();
      for (const c of rows) {
        if ((want !== "b" && c[1] !== want) || c[2] < lo2 || c[2] > hi2) continue;
        const a = by.get(c[2]) || [0, 0, false]; a[0] += c[4]; a[1] += c[5];
        // ▲ uses the scanner's own per-contract UOA rule, not the strike total
        if (c[4] >= cfg.min_volume && c[5] >= cfg.min_oi && c[5] > 0 && c[4] / c[5] >= cfg.uoa_ratio) a[2] = true;
        by.set(c[2], a);
      }
      const strikes = [...by.keys()].sort((a, b) => a - b);
      if (strikes.length) {
        const vol = strikes.map(k => by.get(k)[0]), oi = strikes.map(k => by.get(k)[1]);
        const flags = strikes.map(k => by.get(k)[2]);
        const pos = strikes.findIndex(k => k >= S);
        const top = strikes.map((k, i) => [k, vol[i]]).sort((a, b) => b[1] - a[1])[0];
        barChart(actHost, {
          title: "Activity by strike: traded today vs open positions",
          sub: `${ex} · ${typ.options[typ.selectedIndex].text}`,
          cats: strikes, series: [{ name: "Volume (traded today)", color: C1, vals: vol }, { name: "Open interest (open at last close)", color: C2, vals: oi }],
          flags, fmtCat: v => v.toFixed(0), xTitle: "Strike", yTitle: "Contracts",
          marker: pos > 0 ? { pos, label: `Today ${usd(S, 0)}` } : null,
          tip: i => ({ header: `Strike ${usd(strikes[i])}`, rows: [
            { color: C1, value: vol[i].toLocaleString(), label: "volume today" },
            { color: C2, value: oi[i].toLocaleString(), label: "open interest" },
            { value: oi[i] ? (vol[i] / oi[i]).toFixed(2) + "×" : "n/a", label: flags[i] ? "Vol/OI at strike · unusual contract here ▲" : "Vol/OI at strike" }] }),
          caption: `Blue towering over orange means more contracts changed hands today than were open yesterday: fresh positioning, the idea behind UOA. ` +
            `▲ marks strikes with at least one contract passing your UOA settings (volume ≥ ${cfg.min_volume}, OI ≥ ${cfg.min_oi}, Vol/OI ≥ ${cfg.uoa_ratio}). The busiest strike today is ${usd(top[0], 0)}. ` +
            "Open interest tends to pile up at round-number strikes, where many traders hold positions.",
        });
      } else actHost.textContent = "";
    }
    sel.addEventListener("change", drawExpiry);
    typ.addEventListener("change", drawExpiry);
    drawExpiry();
  }

  // =====================================================================
  // Concept charts for one contract
  // =====================================================================
  function contract(root, d, row) {
    root.textContent = "";
    const S = d.summary.spot, r = d.r, q = d.q, K = row.strike, isCall = row.type === "call";
    const iv = row.iv, mid = row.mid, T = Math.max(row.dte, 0.5) / 365;
    if (!(isFinite(mid) && mid > 0 && isFinite(iv) && iv > 0)) {
      htmlEl("p", "note", root, "Charts need a positive price and IV; this contract has no usable quote.");
      return;
    }
    const intr = Math.max(isCall ? S - K : K - S, 0);
    const grid = htmlEl("div", "charts", root);
    const word = isCall ? "above" : "below";

    // 1. Payoff at expiry vs value today
    {
      const v = Math.max(iv, 0.05) * Math.sqrt(T), spread = Math.max(3 * v, 0.06);
      const lo = Math.min(S * Math.exp(-spread), K * 0.97), hi = Math.max(S * Math.exp(spread), K * 1.03);
      const xs = range(lo, hi, 161);
      const atExp = xs.map(x => [x, Math.max(isCall ? x - K : K - x, 0) - mid]);
      const today = xs.map(x => [x, bs(x, K, T, r, q, iv, isCall) - mid]);
      const be = isCall ? K + mid : K - mid;
      const tvNow = bs(S, K, T, r, q, iv, isCall) - intr;
      lineChart(htmlEl("div", "", grid), {
        title: "Profit or loss if you buy one at the mid",
        sub: `Cost ${usd(mid)} per share = ${usd(mid * 100)} per contract`,
        series: [{ name: "On expiry day", color: C1, pts: atExp }, { name: "Today (model value)", color: C2, pts: today }],
        hlines: [{ y: 0, label: "" }],
        vlines: [{ x: K, label: `Strike ${usd(K, 0)}` }, { x: S, label: `Today ${usd(S, 0)}` }],
        dots: [{ x: be, y: 0, color: C1, label: `Break-even ${usd(be)}`, below: true }],
        xTitle: "Stock price", yTitle: "Profit / loss per share", fmtX: v => usd(v, 0), fmtY: v => sUsd(v).replace(".00", ""),
        tip: (x, near) => ({ header: `If the stock is ${usd(x)}`, rows: [
          { color: C1, value: sUsd(near[0][1]), label: `at expiry (${sUsd(near[0][1] * 100)} per contract)` },
          { color: C2, value: sUsd(near[1][1]), label: "today" },
          { value: usd(Math.max(near[1][1] - near[0][1], 0)), label: "gap = time value" }] }),
        caption: `Blue is your result on expiry day: the most you can lose is what you paid (${usd(mid * 100)} per contract), and you need the stock ${word} ${usd(be)} to profit. ` +
          `Orange is what the option is worth today at each price. The gap between the lines is time value (${usd(Math.max(tvNow, 0))} at today's price). ` +
          "Every day, orange sinks toward blue: that sinking is time decay.",
      });
    }

    // 2. Time decay curve
    {
      const D = Math.max(row.dte, 0.5), ts = range(D, 0, 121);
      const val = t => bs(S, K, t / 365, r, q, iv, isCall);
      const pts = ts.map(t => [t, val(t)]);
      const tv0 = val(D) - intr, tvHalf = val(D / 2) - intr;
      const first = tv0 > 1e-6 ? (tv0 - tvHalf) / tv0 : 0;
      lineChart(htmlEl("div", "", grid), {
        title: "Time decay: value as expiry approaches",
        sub: `Price held at ${usd(S)}, IV held at ${pc(iv)}`,
        series: [{ name: "Option value", color: C1, pts, area: true }, { name: "Intrinsic value (what's left at expiry)", color: C2, pts: ts.map(t => [t, intr]) }],
        dots: [{ x: D, y: val(D), color: C1, label: `Today ${usd(val(D))}` }],
        xReverse: true, yZero: true, xTitle: "Days left until expiry", yTitle: "Value per share",
        fmtX: v => Number.isInteger(v) ? String(v) : v.toFixed(1), fmtY: v => usd(v),
        tip: t => ({ header: `${t.toFixed(t < 2 ? 1 : 0)} days left`, rows: [
          { color: C1, value: usd(val(t)), label: "option value" },
          { value: usd(val(t) - intr), label: "time value" },
          { value: sUsd(val(Math.max(t - 1, 0)) - val(t)), label: "change over the next day (theta)" }] }),
        caption: tv0 <= 1e-6 ? "This option has almost no time value left, so there is little to decay."
          : `With nothing else changing, the value slides from ${usd(val(D))} to ${usd(intr)} at expiry. ` +
            `${pc(first, 0)} of the time value goes in the first half of the remaining days and ${pc(1 - first, 0)} in the second half` +
            (first < 0.5 ? ": decay speeds up near expiry, which is why short-dated options lose value fastest." :
                           ": for an option this far from the money, most value goes early, because the chance of reaching the strike fades quickly.") +
            ` Today's slope is theta: ${usd(row.theta, 3)} per day.`,
      });
    }

    // 3. Chance of finishing in the money
    {
      const D = dist(S, T, r, q, Math.max(iv, 0.01));
      const lo = Math.min(S * Math.exp(-3.5 * D.v), K * 0.98), hi = Math.max(S * Math.exp(3.5 * D.v), K * 1.02);
      const pts = range(lo, hi, 181).map(x => [x, D.pdf(x)]);
      const p = isCall ? 1 - D.below(K) : D.below(K);
      lineChart(htmlEl("div", "", grid), {
        title: "Chance of finishing in the money",
        sub: `Model odds using this contract's IV (${pc(iv)})`,
        series: [{ name: "Likelihood", color: C1, pts }],
        fills: [{ series: 0, from: isCall ? K : lo, to: isCall ? hi : K, opacity: 0.3 }],
        vlines: [{ x: K, label: `Strike ${usd(K, 0)}` }, { x: S, label: `Today ${usd(S, 0)}` }],
        hideY: true, yZero: true, xTitle: "Price at expiry", fmtX: v => usd(v, 0),
        tip: x => ({ header: `Price ${usd(x)}`, rows: [
          { value: pc(D.below(x)), label: "chance of ending below" }, { value: pc(1 - D.below(x)), label: "chance of ending above" }] }),
        caption: `The shaded area is the chance the stock ends ${word} the ${usd(K, 0)} strike: ${pc(p, 0)}. ` +
          `This option's delta is ${Math.abs(row.delta).toFixed(2)}, close to that figure, which is why traders read delta as rough odds. ` +
          `That is how the buckets work: ${row.bucket} means |delta| ${row.bucket === "Lotto" ? "below 0.15, so long odds" : row.bucket === "Sensible" ? "0.15–0.35" : row.bucket === "Directional" ? "0.35–0.65, close to a coin flip" : "above 0.65, likely to finish in the money"}. ` +
          "Higher IV or more time flattens the curve and spreads the shading, raising the odds of far strikes.",
      });
    }

    // 4. Price vs IV: vega and edge
    {
      const ref = row.ref_iv, top = Math.max(iv, ref || iv);
      const lo = Math.max(0.01, 0.3 * Math.min(iv, ref || iv)), hi = Math.max(1.8 * top, lo + 0.1);
      const pts = range(lo, hi, 121).map(s => [s * 100, bs(S, K, T, r, q, s, isCall)]);
      const dots = [{ x: iv * 100, y: bs(S, K, T, r, q, iv, isCall), color: C1, label: "This contract's IV" }];
      if (ref != null) dots.push({ x: ref * 100, y: row.theo, color: C2, label: "Expiry's normal IV", below: true });
      lineChart(htmlEl("div", "", grid), {
        title: "How implied volatility sets the price",
        sub: ref != null ? `Reference IV ${pc(ref)} · this contract ${pc(iv)}` : `This contract ${pc(iv)} · no reference IV for this expiry`,
        series: [{ name: "Model price", color: C1, pts }],
        hlines: [{ y: mid, label: `Market mid ${usd(mid)}` }], dots, yZero: true,
        xTitle: "Implied volatility", yTitle: "Option price per share", fmtX: v => v.toFixed(0) + "%", fmtY: v => usd(v),
        tip: (x, near) => ({ header: `IV ${x.toFixed(1)}%`, rows: [{ color: C1, value: usd(near[0][1]), label: "model price" }] }),
        caption: `Higher IV means more expected movement, so the price curve rises. Its slope is vega: about ${usd(row.vega, 3)} per IV point here. ` +
          (ref == null ? "There is no reference IV for this expiry, so no Rich/Cheap verdict." :
            `The orange dot prices the option at this expiry's normal IV: ${usd(row.theo)}. The market mid is ${usd(mid)}, so edge = ${row.edge_pct > 0 ? "+" : ""}${row.edge_pct.toFixed(1)}%: ` +
            (row.edge_pct > 0 ? "the market charges less than normal vol implies (Cheap)." : "the market charges more than normal vol implies (Rich)."))
          + " The horizontal distance between the dots is the IV gap that edge % expresses in dollars.",
      });
    }
  }

  window.ScanCharts = { market, contract };
})();
