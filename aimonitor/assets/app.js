(function () {
"use strict";

function el(id) { return document.getElementById(id); }
function fInt(v) { var n = Number(v); return isFinite(n) ? Math.round(n).toLocaleString("en-US") : "-"; }
function fFix(v, d) { var n = Number(v); return isFinite(n) ? n.toFixed(d) : "-"; }
function fSci(v, sig) {
  var n = Number(v);
  if (!isFinite(n)) return "-";
  if (n === 0) return "0";
  var e = Math.floor(Math.log(Math.abs(n)) / Math.LN10);
  return (n / Math.pow(10, e)).toFixed((sig || 3) - 1) + "e" + e;
}
function fSmart(v) {
  var n = Number(v);
  if (!isFinite(n)) return "-";
  if (n === 0) return "0";
  var a = Math.abs(n);
  if (a < 1e-3) return fSci(n, 3);
  if (a >= 1e5) return fInt(n);
  if (a >= 1e3) return n.toLocaleString("en-US", { maximumFractionDigits: 1 });
  return String(+n.toFixed(5));
}
function fPct(v) { var n = Number(v); return isFinite(n) ? (n * 100).toFixed(2) + "%" : "-"; }
function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function fmt(kind, v) {
  if (v === null || v === undefined) return "-";
  switch (kind) {
    case "int": return fInt(v);
    case "f1": return fFix(v, 1);
    case "f2": return fFix(v, 2);
    case "f3": return fFix(v, 3);
    case "f4": return fFix(v, 4);
    case "f6": return fFix(v, 6);
    case "pct": return fPct(v);
    case "sci": return fSci(v, 3);
    default: return fSmart(v);
  }
}
function setT(id, txt) { var e = el(id); if (e) e.textContent = txt; }
function hexA(hex, a) {
  if (!hex || hex.charAt(0) !== "#" || hex.length < 7) return "rgba(126,170,216," + a + ")";
  var r = parseInt(hex.substr(1, 2), 16), g = parseInt(hex.substr(3, 2), 16), b = parseInt(hex.substr(5, 2), 16);
  return "rgba(" + r + "," + g + "," + b + "," + a + ")";
}

var PALETTE = ["#7f9fd0", "#3ddc84", "#ff8f5a", "#8fb9ff", "#ffd36a", "#9a8cc4", "#c39a5c", "#67b0a3", "#c88f8f", "#86a878"];
var colorOf = {};
function pickColor(key, i) {
  if (!colorOf[key]) colorOf[key] = PALETTE[i % PALETTE.length];
  return colorOf[key];
}

var STATE = null, GRAPH = null, GRAPH_VER = -1, CTL = { busy: false };

function renderHeader(d) {
  setT("brand", d.title || "AI 训练监控");
  var info = d.info || {}, parts = [], k;
  for (k in info) if (Object.prototype.hasOwnProperty.call(info, k)) parts.push(info[k]);
  var strip = el("metricsStrip"), h = "";
  for (var i = 0; i < d.metrics.length; i++) {
    var m = d.metrics[i];
    h += '<span class="st"><em>' + esc(m.label) + '</em><b>' + fmt(m.fmt, m.value) + '</b></span>';
  }
  strip.innerHTML = h;
  var on = !!d.training;
  var st = el("hState");
  if (st) { st.className = "chip " + (on ? "on" : "off"); st.textContent = on ? "训练中" : "已停止"; }
  var bs = el("h-start"), bt = el("h-stop");
  if (bs) bs.disabled = on;
  if (bt) bt.disabled = !on;
  setT("cardSub", parts.length ? parts.join(" · ") : "实时指标");
}

function renderCards(d) {
  var h = "";
  for (var i = 0; i < d.metrics.length; i++) {
    var m = d.metrics[i];
    h += '<div class="cd"><div class="ck">' + esc(m.label) + '</div><div class="cv">' + fmt(m.fmt, m.value) + '</div></div>';
  }
  el("cards").innerHTML = h || '<div class="cd"><div class="ck">暂无指标</div><div class="cv">-</div></div>';
}

var CHART = { win: 60, smooth: false, data: [], hover: null };

function chartSeries(d) {
  var series = d.series || {}, out = [], i;
  var now = Date.now() / 1000;
  for (i = 0; i < d.metrics.length; i++) {
    var m = d.metrics[i];
    if (!m.chart) continue;
    var raw = series[m.key] || [], win = [], j;
    for (j = 0; j < raw.length; j++) {
      var v = Number(raw[j][1]), t = Number(raw[j][0]);
      if (!isFinite(v)) continue;
      if (CHART.win > 0 && now - t > CHART.win) continue;
      win.push([t, v]);
    }
    if (CHART.smooth && win.length > 2) {
      var a = 0.25, out2 = [win[0]], prev = win[0][1];
      for (j = 1; j < win.length; j++) {
        prev = prev + a * (win[j][1] - prev);
        out2.push([win[j][0], prev]);
      }
      win = out2;
    }
    out.push({ m: m, pts: win, color: m.color || colorOf[m.key] || pickColor(m.key, i) });
  }
  return out;
}

function renderCharts(d) {
  var host = el("charts"), h = "", i;
  var list = chartSeries(d);
  for (i = 0; i < list.length; i++) {
    h += '<div class="ch"><div class="cht"><span class="chn">' + esc(list[i].m.label) + '</span><b class="chv">-</b></div>'
      + '<canvas class="chc"></canvas>'
      + '<div class="chf"><span>最小 <b class="f0">-</b></span><span>最大 <b class="f1">-</b></span><span>平均 <b class="f2">-</b></span></div></div>';
  }
  if (host.getAttribute("data-n") !== String(list.length)) {
    host.innerHTML = h;
    host.setAttribute("data-n", String(list.length));
  }
  var nodes = host.querySelectorAll(".ch");
  var n = 0;
  CHART.data = list;
  for (i = 0; i < nodes.length && i < list.length; i++) {
    drawChart(nodes[i], list[i].pts, list[i].color, list[i].m.fmt,
      CHART.hover && CHART.hover.i === i ? CHART.hover.x : null);
    n = Math.max(n, list[i].pts.length);
  }
  var span = CHART.win > 0 ? (CHART.win >= 60 ? (CHART.win / 60) + " 分钟" : CHART.win + " 秒") : "全部历史";
  setT("chartSub", "窗口 " + span + " · 每 " + fFix(d.interval, 1) + " 秒采样 · 当前 " + fInt(n) + " 点"
    + (CHART.smooth ? " · 已平滑" : "") + " · 悬停看单点数值");
  return nodes;
}

function drawChart(node, win, color, kind, hoverX) {
  var cv = node.querySelector(".chc"), ctx = cv.getContext("2d");
  var w = cv.clientWidth | 0, h = cv.clientHeight | 0;
  if (w < 24 || h < 24) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var nw = Math.round(w * dpr), nh = Math.round(h * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var cnt = win.length;
  var fv = node.querySelector(".chv"), f0 = node.querySelector(".f0"), f1 = node.querySelector(".f1"), f2 = node.querySelector(".f2");
  if (!cnt) {
    fv.textContent = "-"; f0.textContent = "-"; f1.textContent = "-"; f2.textContent = "-";
    return;
  }
  var i, ymin = Infinity, ymax = -Infinity, xmin = win[0][0], xmax = win[cnt - 1][0], sum = 0;
  for (i = 0; i < cnt; i++) {
    var v = win[i][1];
    if (v < ymin) ymin = v;
    if (v > ymax) ymax = v;
    sum += v;
  }
  fv.textContent = fmt(kind, win[cnt - 1][1]);
  f0.textContent = fmt(kind, ymin);
  f1.textContent = fmt(kind, ymax);
  f2.textContent = fmt(kind, sum / cnt);
  if (ymin === ymax) { var e = Math.abs(ymin) * 0.05 + 0.5; ymin -= e; ymax += e; }
  var pad = (ymax - ymin) * 0.12;
  ymin -= pad; ymax += pad;
  if (xmax === xmin) xmax = xmin + 1;
  var L = 2, R = w - 2, T = 6, B = h - 12;
  function px(t) { return L + (t - xmin) / (xmax - xmin) * (R - L); }
  function py(v) { return B - (v - ymin) / (ymax - ymin) * (B - T); }
  ctx.strokeStyle = "rgba(255,255,255,.045)";
  ctx.lineWidth = 1;
  for (i = 0; i <= 3; i++) {
    var gy = T + (B - T) * i / 3;
    ctx.beginPath(); ctx.moveTo(L, Math.round(gy) + .5); ctx.lineTo(R, Math.round(gy) + .5); ctx.stroke();
  }
  ctx.beginPath();
  for (i = 0; i < cnt; i++) {
    var X = px(win[i][0]), Y = py(win[i][1]);
    if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  var lx = px(win[cnt - 1][0]), ly = py(win[cnt - 1][1]);
  var grad = ctx.createLinearGradient(0, T, 0, B);
  grad.addColorStop(0, hexA(color, 0.18));
  grad.addColorStop(1, hexA(color, 0));
  ctx.lineTo(lx, B); ctx.lineTo(px(win[0][0]), B); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  for (i = 0; i < cnt; i++) {
    var X2 = px(win[i][0]), Y2 = py(win[i][1]);
    if (i === 0) ctx.moveTo(X2, Y2); else ctx.lineTo(X2, Y2);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();
  ctx.beginPath(); ctx.arc(lx, ly, 2.6, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  node.__chart = { win: win, xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax, L: L, R: R, T: T, B: B, color: color, kind: kind };
  if (hoverX !== null && hoverX !== undefined) chartCrosshair(ctx, node.__chart, hoverX, w, h);
}

function chartCrosshair(ctx, c, mx, w, h) {
  var i, best = 0, bd = Infinity;
  for (i = 0; i < c.win.length; i++) {
    var d = Math.abs(c.L + (c.win[i][0] - c.xmin) / (c.xmax - c.xmin) * (c.R - c.L) - mx);
    if (d < bd) { bd = d; best = i; }
  }
  var p = c.win[best];
  var X = c.L + (p[0] - c.xmin) / (c.xmax - c.xmin) * (c.R - c.L);
  var Y = c.B - (p[1] - c.ymin) / (c.ymax - c.ymin) * (c.B - c.T);
  ctx.strokeStyle = "rgba(200,220,240,.35)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(Math.round(X) + .5, c.T); ctx.lineTo(Math.round(X) + .5, c.B); ctx.stroke();
  ctx.beginPath(); ctx.arc(X, Y, 3, 0, Math.PI * 2);
  ctx.fillStyle = c.color; ctx.fill();
  var d2 = new Date(p[0] * 1000);
  function p2(x) { return (x < 10 ? "0" : "") + x; }
  var txt = p2(d2.getHours()) + ":" + p2(d2.getMinutes()) + ":" + p2(d2.getSeconds()) + "  " + fmt(c.kind, p[1]);
  ctx.font = '10px "Microsoft YaHei UI","Segoe UI",sans-serif';
  var tw = ctx.measureText(txt).width + 10;
  var bx = Math.max(1, Math.min(w - tw - 1, X + 6));
  ctx.fillStyle = "rgba(11,15,20,.92)";
  ctx.fillRect(bx, c.T, tw, 15);
  ctx.strokeStyle = "rgba(90,110,135,.6)";
  ctx.strokeRect(bx + .5, c.T + .5, tw - 1, 14);
  ctx.fillStyle = "#dfe7ef";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(txt, bx + 5, c.T + 8);
}

function bindCharts() {
  var host = el("charts");
  if (!host || host.__bound) return;
  host.__bound = true;
  host.addEventListener("mousemove", function (ev) {
    var cv = ev.target;
    if (!cv || cv.tagName !== "CANVAS") return;
    var node = cv.parentNode;
    var boxes = host.querySelectorAll(".ch");
    var i = -1;
    for (var k = 0; k < boxes.length; k++) if (boxes[k] === node) i = k;
    if (i < 0) return;
    var r = cv.getBoundingClientRect();
    CHART.hover = { i: i, x: ev.clientX - r.left };
    drawChart(node, CHART.data[i].pts, CHART.data[i].color, CHART.data[i].m.fmt, CHART.hover.x);
  });
  host.addEventListener("mouseleave", function () {
    CHART.hover = null;
    var boxes = host.querySelectorAll(".ch");
    for (var k = 0; k < boxes.length && k < CHART.data.length; k++) {
      drawChart(boxes[k], CHART.data[k].pts, CHART.data[k].color, CHART.data[k].m.fmt, null);
    }
  });
}

function exportCSV() {
  if (!STATE) return;
  function pad2(x) { return (x < 10 ? "0" : "") + x; }
  var series = STATE.series || {}, keys = [], i;
  for (i = 0; i < STATE.metrics.length; i++) {
    if (!STATE.metrics[i].chart) continue;
    if (series[STATE.metrics[i].key]) keys.push(STATE.metrics[i].key);
  }
  var times = [], seen = {}, j;
  for (i = 0; i < keys.length; i++) {
    var arr = series[keys[i]];
    for (j = 0; j < arr.length; j++) {
      var t = Number(arr[j][0]);
      if (!seen[t]) { seen[t] = 1; times.push(t); }
    }
  }
  times.sort(function (a, b) { return a - b; });
  var rows = [["时间"].concat(keys)];
  var table = [];
  for (i = 0; i < keys.length; i++) table.push({ v: {}, k: keys[i] });
  for (i = 0; i < keys.length; i++) {
    var a2 = series[keys[i]];
    for (j = 0; j < a2.length; j++) table[i].v[Number(a2[j][0])] = a2[j][1];
  }
  for (i = 0; i < times.length; i++) {
    var d = new Date(times[i] * 1000);
    var line = [d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()) + " "
      + pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds())];
    for (j = 0; j < keys.length; j++) {
      var v = table[j].v[times[i]];
      line.push(v === undefined ? "" : v);
    }
    rows.push(line);
  }
  var csv = "\ufeff" + rows.map(function (r) { return r.join(","); }).join("\r\n");
  var blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "aimonitor-" + Date.now() + ".csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
}

function bindChartCtl() {
  var head = el("secCharts") ? el("secCharts").querySelector(".cctl") : null;
  if (!head || head.__bound) return;
  head.__bound = true;
  head.addEventListener("click", function (ev) {
    var b = ev.target;
    if (!b || b.tagName !== "BUTTON") return;
    if (b.id === "smoothBtn") {
      CHART.smooth = !CHART.smooth;
      b.className = CHART.smooth ? "vbtn on" : "vbtn";
    } else if (b.id === "exportBtn") {
      exportCSV();
      return;
    } else {
      CHART.win = Number(b.getAttribute("data-win")) || 0;
      var btns = head.querySelectorAll("button[data-win]");
      for (var i = 0; i < btns.length; i++) {
        btns[i].className = btns[i] === b ? "vbtn on" : "vbtn";
      }
    }
    if (STATE) renderCharts(STATE);
  });
}

function renderAlerts(d) {
  var as = d.alerts || [], bar = el("alertBar"), h = "";
  for (var i = 0; i < as.length; i++) {
    h += '<span class="al ' + esc(as[i].level || "warn") + '"><b>' + esc(as[i].label) + '</b>'
      + esc(as[i].msg) + '</span>';
  }
  bar.innerHTML = h;
  bar.className = h ? "alertbar on" : "alertbar";
}

var NAV_GROUPS = [
  ["训练概览", ["secCards", "secCharts", "secHealth"]],
  ["模型内部", ["secFeature", "secImages", "secHist", "secNet", "secSurface"]],
  ["诊断分析", ["secTable", "secMatrix", "secScatter", "secBar", "secLog"]],
  ["配置", ["secParams"]]
];
var navSig = "";

function buildNav() {
  var host = el("nav"), h = "", i, j, k;
  var map = {};
  var secs = document.querySelectorAll("section.card");
  for (i = 0; i < secs.length; i++) map[secs[i].id] = secs[i];
  for (i = 0; i < NAV_GROUPS.length; i++) {
    var items = [];
    for (j = 0; j < NAV_GROUPS[i][1].length; j++) {
      var s = map[NAV_GROUPS[i][1][j]];
      if (!s || s.className.indexOf("off") >= 0) continue;
      var t = s.querySelector(".ctitle");
      items.push('<a href="#' + s.id + '" data-t="' + s.id + '">' + esc(t ? t.textContent : s.id) + '</a>');
    }
    if (!items.length) continue;
    if (h) h += '<span class="gp">|</span>';
    h += items.join("");
  }
  if (h === navSig) return;
  navSig = h;
  host.innerHTML = h;
  var as = host.querySelectorAll("a");
  for (k = 0; k < as.length; k++) {
    as[k].addEventListener("click", function (ev) {
      ev.preventDefault();
      var t = document.getElementById(this.getAttribute("data-t"));
      if (t) window.scrollTo({ top: t.offsetTop - 74, behavior: "smooth" });
    });
  }
}

function markNav() {
  var host = el("nav");
  if (!host) return;
  var as = host.querySelectorAll("a"), best = null, bd = Infinity;
  for (var i = 0; i < as.length; i++) {
    var t = document.getElementById(as[i].getAttribute("data-t"));
    if (!t) continue;
    var d = Math.abs(t.getBoundingClientRect().top - 90);
    if (d < bd) { bd = d; best = as[i]; }
  }
  for (i = 0; i < as.length; i++) as[i].className = as[i] === best ? "on" : "";
}

function initFold() {
  var secs = document.querySelectorAll("section.card");
  for (var i = 0; i < secs.length; i++) {
    (function (s) {
      var head = s.querySelector(".chead");
      if (!head) return;
      var saved = null;
      try { saved = localStorage.getItem("fold:" + s.id); } catch (e) {}
      if (saved === "1") s.className = s.className + " folded";
      head.addEventListener("click", function (ev) {
        if (ev.target.tagName === "BUTTON") return;
        if (s.className.indexOf("folded") >= 0) {
          s.className = s.className.replace(" folded", "");
        } else {
          s.className = s.className + " folded";
        }
        try {
          localStorage.setItem("fold:" + s.id, s.className.indexOf("folded") >= 0 ? "1" : "0");
        } catch (e) {}
        requestNet();
        requestGD();
      });
    })(secs[i]);
  }
}

function renderHealth(d) {
  var hs = d.health || [], h = "";
  for (var i = 0; i < hs.length; i++) {
    var x = hs[i];
    h += '<div class="cd"><div class="ck">' + esc(x.label) + '</div><div class="cv">'
      + fmt(x.fmt, x.value) + (x.unit ? '<span class="cu">' + esc(x.unit) + '</span>' : '') + '</div></div>';
  }
  el("health").innerHTML = h || '<div class="cd"><div class="ck">暂无健康度指标</div><div class="cv">-</div></div>';
  setT("healthSub", fInt(hs.length) + " 项");
}

function renderImages(d) {
  var groups = { "特征图": [], "图片": [] };
  var imgs = d.images || [], i;
  for (i = 0; i < imgs.length; i++) {
    var g = imgs[i].group || "图片";
    if (!groups[g]) groups[g] = [];
    groups[g].push(imgs[i]);
  }
  function block(list) {
    var h = "";
    for (var k = 0; k < list.length; k++) {
      var im = list[k];
      h += '<div class="imgbox"><div class="it"><span>' + esc(im.label) + '</span>'
        + '<em>真实 ' + im.w + '×' + im.h + ' · 放大 ' + fFix(im.scale, 0) + '×'
        + (im.lo !== null && im.lo !== undefined ? ' · 值域 ' + fSmart(im.lo) + '~' + fSmart(im.hi) : '')
        + '</em></div><img src="' + im.png + '" alt="' + esc(im.label) + '"></div>';
    }
    return h;
  }
  el("featureMaps").innerHTML = block(groups["特征图"] || []) || '<div class="imgbox"><div class="it"><em>无特征图数据</em></div></div>';
  el("images").innerHTML = block(groups["图片"] || []) || '<div class="imgbox"><div class="it"><em>无图片数据</em></div></div>';
  setT("featSub", fInt((groups["特征图"] || []).length) + " 张特征图拼图");
  setT("imgSub", fInt((groups["图片"] || []).length) + " 张图片");
  setSec("secFeature", (groups["特征图"] || []).length > 0);
  setSec("secImages", (groups["图片"] || []).length > 0);
}

function renderHists(d) {
  var hs = d.histograms || [], host = el("hists"), h = "", i;
  for (i = 0; i < hs.length; i++) {
    h += '<div class="hist" data-i="' + i + '"><div class="ht">' + esc(hs[i].label) + '</div>'
      + '<canvas></canvas><div class="hs"><span>数量 <b>' + fInt(hs[i].n) + '</b></span>'
      + '<span>最小 <b>' + fSmart(hs[i].min) + '</b></span><span>最大 <b>' + fSmart(hs[i].max) + '</b></span>'
      + '<span>均值 <b>' + fSmart(hs[i].mean) + '</b></span><span>标准差 <b>' + fSmart(hs[i].std) + '</b></span></div></div>';
  }
  host.innerHTML = h || '<div class="hist"><div class="ht">暂无直方图数据</div><canvas></canvas></div>';
  var boxes = host.querySelectorAll(".hist");
  for (i = 0; i < boxes.length && i < hs.length; i++) {
    drawHist(boxes[i].querySelector("canvas"), hs[i]);
  }
  setT("histSub", fInt(hs.length) + " 个张量");
  setSec("secHist", hs.length > 0);
}

function drawHist(cv, h) {
  if (!cv) return;
  var ctx = cv.getContext("2d");
  var w = cv.clientWidth | 0, hh = cv.clientHeight | 0;
  if (w < 20 || hh < 20) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var nw = Math.round(w * dpr), nh = Math.round(hh * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, hh);
  var cnt = h.counts || [], m = cnt.length;
  if (!m) return;
  var mx = 0, i;
  for (i = 0; i < m; i++) if (cnt[i] > mx) mx = cnt[i];
  if (mx <= 0) mx = 1;
  var bw = w / m, B = hh - 2, T = 4;
  var neg = (h.min < 0 && h.max > 0);
  if (neg) {
    var zero = w * (0 - h.min) / ((h.max - h.min) || 1);
    ctx.strokeStyle = "rgba(200,220,240,.22)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(zero, T); ctx.lineTo(zero, B); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (i = 0; i < m; i++) {
    var bh = (B - T) * (cnt[i] / mx);
    if (bh < 0.5 && cnt[i] > 0) bh = 0.5;
    var c = (h.edges[i] + h.edges[i + 1]) / 2;
    var col = neg ? (c < 0 ? "#5b8cd6" : "#c9924f") : "#7f9fd0";
    ctx.fillStyle = col;
    ctx.fillRect(i * bw + 0.3, B - bh, Math.max(0.8, bw - 0.6), bh);
  }
  ctx.fillStyle = "#5b6672";
  ctx.font = '9px "Microsoft YaHei UI","Segoe UI",sans-serif';
  ctx.textAlign = "left";
  ctx.fillText(fSmart(h.min), 2, hh - 2);
  ctx.textAlign = "right";
  ctx.fillText(fSmart(h.max), w - 2, hh - 2);
}

function setSec(id, on) {
  var e = el(id);
  if (!e) return;
  var folded = e.className.indexOf("folded") >= 0;
  e.className = (on ? "card" : "card off") + (folded ? " folded" : "");
}

function tsShort(t) {
  var n = Number(t);
  if (!isFinite(n) || n <= 0) return "";
  var d = new Date(n * 1000);
  function p(x) { return (x < 10 ? "0" : "") + x; }
  return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}

function renderTables(d) {
  var ts = d.tables || [], h = "", i;
  for (i = 0; i < ts.length; i++) {
    var t = ts[i], cols = t.columns || [], rows = t.rows || [];
    var show = rows.slice(0, 300);
    h += '<div class="tbx"><div class="tbt">' + esc(t.label) + '<em>' + fInt(t.n) + ' 行</em></div><div class="tbw"><table><thead><tr>';
    for (var c = 0; c < cols.length; c++) h += '<th>' + esc(cols[c]) + '</th>';
    h += '</tr></thead><tbody>';
    for (var r = 0; r < show.length; r++) {
      h += '<tr>';
      for (var c2 = 0; c2 < show[r].length; c2++) {
        var v = show[r][c2];
        if (typeof v === "number") h += '<td class="num">' + fSmart(v) + '</td>';
        else h += '<td>' + esc(v) + '</td>';
      }
      h += '</tr>';
    }
    h += '</tbody></table></div>';
    if (rows.length > show.length) h += '<div class="tbf">仅显示前 ' + fInt(show.length) + ' 行，共 ' + fInt(t.n) + ' 行</div>';
    h += '</div>';
  }
  el("tables").innerHTML = h;
  setT("tableSub", ts.length ? fInt(ts.length) + " 张表" : "未注册表格");
  setSec("secTable", ts.length > 0);
}

var CMAPS = {
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma: [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  blue: [[16, 30, 42], [47, 111, 151], [140, 224, 250]],
  red: [[20, 8, 8], [140, 40, 30], [255, 210, 120]],
  diverging: [[40, 90, 200], [30, 32, 40], [220, 70, 60]],
  gray: [[0, 0, 0], [255, 255, 255]]
};

function cmapRGB(name, t) {
  var pts = CMAPS[name] || CMAPS.viridis;
  t = Math.max(0, Math.min(1, t));
  var x = t * (pts.length - 1);
  var i = Math.min(Math.floor(x), pts.length - 2);
  var f = x - i, a = pts[i], b = pts[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f)
  ];
}

function renderMatrices(d) {
  var ms = d.matrices || [], host = el("matrices"), h = "", i;
  setT("matrixSub", ms.length ? fInt(ms.length) + " 个矩阵" : "未注册矩阵");
  setSec("secMatrix", ms.length > 0);
  for (i = 0; i < ms.length; i++) {
    h += '<div class="mat"><div class="matt">' + esc(ms[i].label) + '<em>' + ms[i].h + ' × ' + ms[i].w
      + ' · 值域 ' + fSmart(ms[i].min) + ' ~ ' + fSmart(ms[i].max) + '</em></div>'
      + '<canvas class="matcv" data-i="' + i + '"></canvas>'
      + '<div class="mhint" data-i="' + i + '">悬停查看单格数值</div></div>';
  }
  host.innerHTML = h;
  var cvs = host.querySelectorAll(".matcv");
  for (i = 0; i < cvs.length && i < ms.length; i++) drawMatrix(cvs[i], ms[i]);
}

function drawMatrix(cv, m) {
  var data = m.data || [];
  if (!data.length) return;
  var rows = data.length, cols = data[0].length;
  var yl = m.ylabels || null, xl = m.xlabels || null;
  var yw = 0, i, F = 9;
  if (yl) {
    for (i = 0; i < yl.length; i++) yw = Math.max(yw, textW(yl[i], F));
    yw = Math.min(yw + 6, 92);
  }
  if (cv.style.width !== "100%") cv.style.width = "100%";
  var w = cv.clientWidth | 0;
  if (w < 60) return;
  var avail = w - yw - 2;
  var cell = Math.max(3, Math.min(48, avail / cols));
  var cw = cell * cols;
  var showX = (xl && cell >= 13);
  var xh = showX ? 22 : 4;
  var hh = cell * rows + xh + 2;
  var vw = yw + cw + 2;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var nw = Math.round(vw * dpr), nh = Math.round(hh * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  if (cv.style.width !== vw + "px") cv.style.width = vw + "px";
  if (cv.style.height !== hh + "px") cv.style.height = hh + "px";
  var ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '9px "Microsoft YaHei UI","Segoe UI",sans-serif';
  ctx.clearRect(0, 0, yw + cw + 2, hh);
  var lo = m.min, hi = m.max;
  var div = (m.cmap === "diverging") || (!m.cmap && lo < 0 && hi > 0);
  if (div) {
    var a = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    lo = -a; hi = a;
  }
  var name = m.cmap || (div ? "diverging" : "viridis");
  var span = (hi - lo) || 1;
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      var v = Number(data[r][c]);
      if (!isFinite(v)) v = 0;
      var t = (v - lo) / span;
      var rgb = cmapRGB(name, t);
      ctx.fillStyle = "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
      ctx.fillRect(yw + c * cell, r * cell, Math.ceil(cell) + 0.4, Math.ceil(cell) + 0.4);
      if (cell >= 23) {
        var lum = (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114) / 255;
        ctx.fillStyle = lum > 0.6 ? "rgba(10,14,18,.85)" : "rgba(240,246,252,.88)";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(fmt(m.fmt || "f3", v), yw + c * cell + cell / 2, r * cell + cell / 2 + 0.5);
      }
    }
  }
  ctx.textBaseline = "middle";
  if (yw > 0) {
    ctx.fillStyle = "#8b95a3";
    ctx.textAlign = "right";
    for (r = 0; r < rows && r < yl.length; r++) {
      if (cell >= 11) ctx.fillText(clipText(yl[r], yw - 6, F), yw - 4, r * cell + cell / 2 + 0.5);
    }
  }
  if (showX) {
    ctx.fillStyle = "#8b95a3";
    ctx.textAlign = "center";
    for (c = 0; c < cols && c < xl.length; c++) {
      var tx = yw + c * cell + cell / 2;
      if (cell >= 20 && xl.length <= 24) {
        ctx.fillText(clipText(xl[c], cell + 8, F), tx, cell * rows + 11);
      } else {
        ctx.save();
        ctx.translate(tx, cell * rows + 3);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "right";
        ctx.fillText(clipText(xl[c], 60, F), 0, 0);
        ctx.restore();
      }
    }
  }
  cv.__mat = { m: m, yw: yw, cell: cell, rows: rows, cols: cols };
}

function textW(s, px) {
  return String(s).length * px * 0.56;
}

function clipText(s, maxw, px) {
  var n = Math.max(1, Math.floor(maxw / (px * 0.56)));
  s = String(s);
  return s.length > n ? s.slice(0, Math.max(1, n - 1)) + "…" : s;
}

function bindMatrixHover() {
  var host = el("matrices");
  if (!host || host.__bound) return;
  host.__bound = true;
  host.addEventListener("mousemove", function (ev) {
    var cv = ev.target;
    if (!cv || !cv.__mat) return;
    var m = cv.__mat, rect = cv.getBoundingClientRect();
    var x = ev.clientX - rect.left - m.yw, y = ev.clientY - rect.top;
    var hint = host.querySelector('.mhint[data-i="' + cv.getAttribute("data-i") + '"]');
    if (!hint) return;
    var c = Math.floor(x / m.cell), r = Math.floor(y / m.cell);
    if (x < 0 || y < 0 || c < 0 || r < 0 || c >= m.cols || r >= m.rows) {
      hint.textContent = "悬停查看单格数值";
      return;
    }
    var mt = m.m, rl = (mt.ylabels && mt.ylabels[r]) || ("行 " + r);
    var cl = (mt.xlabels && mt.xlabels[c]) || ("列 " + c);
    hint.innerHTML = "行 <b>" + esc(rl) + "</b> · 列 <b>" + esc(cl) + "</b> = <b>"
      + fmt(mt.fmt || "f3", mt.data[r][c]) + "</b>";
  });
  host.addEventListener("mouseleave", function () {
    var hs = host.querySelectorAll(".mhint");
    for (var i = 0; i < hs.length; i++) hs[i].textContent = "悬停查看单格数值";
  });
}

function renderScatters(d) {
  var ss = d.scatters || [], host = el("scatters"), h = "", i;
  setT("scatterSub", ss.length ? fInt(ss.length) + " 组投影" : "未注册散点");
  setSec("secScatter", ss.length > 0);
  for (i = 0; i < ss.length; i++) {
    var s = ss[i], leg = "";
    for (var g = 0; g < s.groups.length; g++) {
      leg += '<span><i style="background:' + pickColor(s.key + ":" + g, g) + '"></i>' + esc(s.groups[g]) + '</span>';
    }
    h += '<div class="scx"><div class="sct">' + esc(s.label) + '<em>' + fInt(s.n) + ' 点 · '
      + esc(s.xlabel) + ' / ' + esc(s.ylabel) + '</em></div>'
      + '<canvas class="sccv" data-i="' + i + '"></canvas>'
      + '<div class="sclegend">' + leg + '</div></div>';
  }
  host.innerHTML = h;
  var cvs = host.querySelectorAll(".sccv");
  for (i = 0; i < cvs.length && i < ss.length; i++) drawScatter(cvs[i], ss[i]);
}

function drawScatter(cv, s) {
  var pts = s.points || [];
  if (!pts.length) return;
  var w = cv.clientWidth | 0, h = cv.clientHeight | 0;
  if (w < 40 || h < 40) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var nw = Math.round(w * dpr), nh = Math.round(h * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  var ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var L = 34, R = w - 8, T = 8, B = h - 18, i;
  var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (i = 0; i < pts.length; i++) {
    var p = pts[i];
    if (p[0] < x0) x0 = p[0];
    if (p[0] > x1) x1 = p[0];
    if (p[1] < y0) y0 = p[1];
    if (p[1] > y1) y1 = p[1];
  }
  if (x1 - x0 < 1e-12) { x0 -= 0.5; x1 += 0.5; }
  if (y1 - y0 < 1e-12) { y0 -= 0.5; y1 += 0.5; }
  var px = (v) => L + (v - x0) / (x1 - x0) * (R - L);
  var py = (v) => B - (v - y0) / (y1 - y0) * (B - T);
  ctx.strokeStyle = "rgba(255,255,255,.055)";
  ctx.lineWidth = 1;
  for (i = 0; i <= 4; i++) {
    var gy = T + (B - T) * i / 4, gx = L + (R - L) * i / 4;
    ctx.beginPath(); ctx.moveTo(L, Math.round(gy) + .5); ctx.lineTo(R, Math.round(gy) + .5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(Math.round(gx) + .5, T); ctx.lineTo(Math.round(gx) + .5, B); ctx.stroke();
  }
  ctx.fillStyle = "#66707d";
  ctx.font = '9px "Microsoft YaHei UI","Segoe UI",sans-serif';
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(fSmart(y1), L - 4, T + 4);
  ctx.fillText(fSmart(y0), L - 4, B - 4);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(fSmart(x0), L, B + 4);
  ctx.fillText(fSmart(x1), R, B + 4);
  var stride = pts.length > 20000 ? Math.ceil(pts.length / 20000) : 1;
  for (i = 0; i < pts.length; i += stride) {
    var q = pts[i];
    ctx.beginPath();
    ctx.arc(px(q[0]), py(q[1]), 1.8, 0, Math.PI * 2);
    ctx.fillStyle = pickColor(s.key + ":" + q[2], q[2]);
    ctx.globalAlpha = stride > 1 ? 0.45 : 0.8;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function renderBars(d) {
  var bs = d.bars || [], h = "", i, j;
  for (i = 0; i < bs.length; i++) {
    var b = bs[i], list = b.bars.slice();
    list.sort(function (x, y) { return Math.abs(y.value) - Math.abs(x.value); });
    var mx = 0;
    for (j = 0; j < list.length; j++) mx = Math.max(mx, Math.abs(list[j].value));
    mx = mx || 1;
    h += '<div class="barx"><div class="bart">' + esc(b.label) + '<em>' + fInt(b.n) + ' 项 · 最大 '
      + fSmart(mx) + '</em></div>';
    for (j = 0; j < list.length; j++) {
      var v = Number(list[j].value), pct = Math.abs(v) / mx * 50;
      var neg = v < 0;
      h += '<div class="brow"><span class="brl" title="' + esc(list[j].label) + '">' + esc(list[j].label) + '</span>'
        + '<span class="brt"><i style="width:' + pct.toFixed(2) + '%;' + (neg ? 'left:' + (50 - pct).toFixed(2) + '%;' : '')
        + 'background:' + (neg ? "#c9924f" : "#5f9fd6") + '"></i></span>'
        + '<span class="brv">' + fSmart(v) + '</span></div>';
    }
    h += '</div>';
  }
  el("bars").innerHTML = h;
  setT("barSub", bs.length ? fInt(bs.length) + " 组条图" : "未注册条图");
  setSec("secBar", bs.length > 0);
}

var LOGSCROLL = {};

function renderLogs(d) {
  var ls = d.logs || [], host = el("logs"), h = "", i, j;
  var wraps = host.querySelectorAll(".lgw");
  for (i = 0; i < wraps.length; i++) {
    LOGSCROLL[i] = {
      top: wraps[i].scrollTop,
      follow: wraps[i].scrollHeight - wraps[i].clientHeight - wraps[i].scrollTop < 24
    };
  }
  for (i = 0; i < ls.length; i++) {
    var g = ls[i];
    h += '<div class="lgx"><div class="lgt">' + esc(g.label) + '<em>最新 ' + fInt(g.n) + ' 行</em></div><div class="lgw">';
    for (j = 0; j < g.lines.length; j++) {
      var ln = g.lines[j], txt = ln.text || "";
      var cls = /error|错误|异常|traceback|failed/i.test(txt) ? "er"
        : (/warn|警告|warnning/i.test(txt) ? "wn" : "");
      h += '<div class="lgl ' + cls + '"><span class="lgi">' + fInt(ln.i) + '</span>'
        + '<span class="lgtm">' + tsShort(ln.t) + '</span>'
        + '<span class="lgtx">' + esc(txt) + '</span></div>';
    }
    h += '</div></div>';
  }
  host.innerHTML = h;
  wraps = host.querySelectorAll(".lgw");
  for (i = 0; i < wraps.length; i++) {
    var st = LOGSCROLL[i];
    if (!st || st.follow) wraps[i].scrollTop = wraps[i].scrollHeight;
    else wraps[i].scrollTop = st.top;
    LOGSCROLL[i] = { top: wraps[i].scrollTop, follow: !st || st.follow };
  }
  setT("logSub", ls.length ? fInt(ls.length) + " 路日志流" : "未注册日志");
  setSec("secLog", ls.length > 0);
}

function renderParams(d) {
  var p = d.params;
  setSec("secParams", !!p);
  if (!p) { el("panels").innerHTML = '<div class="pnl"><div class="pnlh">参数</div><div class="row"><span class="rk">状态</span><span class="rv">暂无</span></div></div>'; return; }
  var h = "";
  for (var i = 0; i < p.length; i++) {
    var blk = p[i];
    h += '<div class="pnl"><div class="pnlh">' + esc(blk.title || "") + '</div>';
    if (blk.layers) {
      h += '<div class="lyrs"><div class="lyr hd"><span>层</span><span>结构</span><span>参数量</span><span>真实值数</span></div>';
      var tot = 0, rtot = 0;
      for (var j = 0; j < blk.layers.length; j++) {
        var l = blk.layers[j];
        tot += Number(l.params) || 0;
        rtot += Number(l.realCount) || 0;
        h += '<div class="lyr"><span class="lnm" title="' + esc(l.name) + '">' + esc(l.name) + '</span>'
          + '<span class="lsh" title="' + esc(l.detail) + '">' + esc(l.detail) + '</span>'
          + '<span class="lpr">' + fInt(l.params) + '</span>'
          + '<span class="lot">' + fInt(l.realCount) + '</span></div>';
      }
      h += '<div class="lyr tot"><span class="lnm">合计</span><span class="lsh">' + fInt(blk.layers.length) + ' 层</span>'
        + '<span class="lpr">' + fInt(tot) + '</span><span class="lot">' + fInt(rtot) + '</span></div></div>';
    }
    if (blk.rows) {
      for (var k = 0; k < blk.rows.length; k++) {
        h += '<div class="row"><span class="rk">' + esc(blk.rows[k][0]) + '</span><span class="rv">' + blk.rows[k][1] + '</span></div>';
      }
    }
    h += '</div>';
  }
  el("panels").innerHTML = h;
  setT("paramSub", "全部为当前模型的真实参数");
}

var GD = { yaw: -0.72, pitch: 0.62, dist: 3.1, drag: false, lx: 0, ly: 0, cv: null, ctx: null, land: null, MESH: 34 };

function gdProj(c, x, y, z) {
  var vx = x - c.ox, vy = y - c.oy, vz = z - c.oz;
  var cz = vx * c.fx + vy * c.fy + vz * c.fz;
  if (cz < 0.08) cz = 0.08;
  return [(vx * c.rx + vy * c.ry + vz * c.rz) / cz, (vx * c.ux + vy * c.uy + vz * c.uz) / cz, cz];
}
function gdCam() {
  var cy = Math.cos(GD.yaw), sy = Math.sin(GD.yaw), cp = Math.cos(GD.pitch), sp = Math.sin(GD.pitch);
  var ex = cp * cy, ey = cp * sy, ez = sp, d = GD.dist;
  return { ox: ex * d, oy: ey * d, oz: ez * d, fx: -ex, fy: -ey, fz: -ez, rx: -sy, ry: cy, rz: 0, ux: -cy * sp, uy: -sy * sp, uz: cp };
}
function gdColor(t) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  var r, g, b;
  if (t < 0.35) { var u = t / 0.35; r = 20 + 22 * u; g = 40 + 60 * u; b = 66 + 84 * u; }
  else if (t < 0.7) { var u2 = (t - 0.35) / 0.35; r = 42 + 44 * u2; g = 100 + 70 * u2; b = 150 + 60 * u2; }
  else { var u3 = (t - 0.7) / 0.3; r = 86 + 104 * u3; g = 170 + 60 * u3; b = 210 + 42 * u3; }
  return [r | 0, g | 0, b | 0];
}
function gdSample(L, ti, tj) {
  var res = L.res, fi = ti * (res - 1), fj = tj * (res - 1);
  var i0 = Math.min(res - 1, Math.max(0, Math.floor(fi))), j0 = Math.min(res - 1, Math.max(0, Math.floor(fj)));
  var i1 = Math.min(res - 1, i0 + 1), j1 = Math.min(res - 1, j0 + 1), tx = fi - i0, ty = fj - j0;
  function g(i, j) { var v = Number(L.values[i * res + j]); return isFinite(v) ? v : L.zmin; }
  var a = g(i0, j0), b = g(i1, j0), cc = g(i1, j1), dd = g(i0, j1);
  return (a * (1 - tx) + b * tx) * (1 - ty) + (dd * (1 - tx) + cc * tx) * ty;
}
function gdEmpty(ctx, w, h, msg, sub) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = "rgba(120,140,168,.24)";
  ctx.lineWidth = 1;
  ctx.strokeRect(15.5, 15.5, w - 31, h - 31);
  ctx.restore();
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = '13px "Microsoft YaHei UI","Segoe UI",sans-serif'; ctx.fillStyle = "#5f6a78";
  ctx.fillText(msg, w * 0.5, h * 0.5 - 11);
  ctx.font = '11px "Microsoft YaHei UI","Segoe UI",sans-serif'; ctx.fillStyle = "#4a545f";
  ctx.fillText(sub, w * 0.5, h * 0.5 + 11);
  ctx.textBaseline = "alphabetic";
}
function drawGD() {
  var cv = GD.cv, ctx = GD.ctx;
  if (!cv || !ctx) return;
  var w = cv.clientWidth | 0, h = cv.clientHeight | 0;
  if (w < 40 || h < 40) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2), nw = Math.round(w * dpr), nh = Math.round(h * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var L = GD.land;
  if (!L) { gdEmpty(ctx, w, h, "正在测量损失曲面…", "等待采样完成"); return; }
  var res = L.res, ext = Number(L.ext) || 1, zmin = Number(L.zmin), zmax = Number(L.zmax);
  var range = (zmax - zmin) || 1, zh = 0.34;
  var c = gdCam(), FOCAL = Math.min(w, h) * 1.02, ox = w * 0.5, oy = h * 0.56;
  var M = GD.MESH, i, j, tx, ty;
  var sx = new Float64Array(M * M), sy = new Float64Array(M * M), sd = new Float64Array(M * M),
    zn = new Float64Array(M * M), wx = new Float64Array(M * M), wy = new Float64Array(M * M), wz = new Float64Array(M * M);
  for (i = 0; i < M; i++) {
    tx = i / (M - 1);
    var uu = -1 + 2 * tx;
    for (j = 0; j < M; j++) {
      ty = j / (M - 1);
      var vv = -1 + 2 * ty;
      var loss = gdSample(L, tx, ty);
      var nt = (loss - zmin) / range;
      if (nt < 0) nt = 0; if (nt > 1) nt = 1;
      var z = nt * zh - zh * 0.5;
      var p = gdProj(c, uu, vv, z), idx = i * M + j;
      sx[idx] = ox + p[0] * FOCAL; sy[idx] = oy - p[1] * FOCAL; sd[idx] = p[2];
      zn[idx] = nt; wx[idx] = uu; wy[idx] = vv; wz[idx] = z;
    }
  }
  ctx.strokeStyle = "rgba(158,190,220,.30)";
  ctx.lineWidth = 1;
  var bp = [];
  for (i = 0; i < 8; i++) bp.push(gdProj(c, -1 + 2 * i / 7, -1, -zh * 0.5));
  for (i = 1; i < 8; i++) bp.push(gdProj(c, -1, -1 + 2 * i / 7, -zh * 0.5));
  bp.push(gdProj(c, -1, -1, -zh * 0.5 - 0.12));
  ctx.beginPath();
  ctx.moveTo(ox + bp[0][0] * FOCAL, oy - bp[0][1] * FOCAL);
  for (i = 1; i < 8; i++) ctx.lineTo(ox + bp[i][0] * FOCAL, oy - bp[i][1] * FOCAL);
  for (i = 7; i < 14; i++) ctx.lineTo(ox + bp[i][0] * FOCAL, oy - bp[i][1] * FOCAL);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(ox + bp[0][0] * FOCAL, oy - bp[0][1] * FOCAL);
  ctx.lineTo(ox + bp[14][0] * FOCAL, oy - bp[14][1] * FOCAL);
  ctx.stroke();
  var quads = [];
  for (i = 0; i < M - 1; i++) for (j = 0; j < M - 1; j++) {
    var a0 = i * M + j, b0 = (i + 1) * M + j, cc = (i + 1) * M + j + 1, dd = i * M + j + 1;
    quads.push([a0, b0, cc, dd, (sd[a0] + sd[b0] + sd[cc] + sd[dd]) * 0.25, (zn[a0] + zn[b0] + zn[cc] + zn[dd]) * 0.25]);
  }
  quads.sort(function (p, q) { return q[4] - p[4]; });
  var light = [0.45, 0.25, 0.86];
  for (var k = 0; k < quads.length; k++) {
    var q = quads[k], a = q[0], b = q[1], cc2 = q[2], dd = q[3];
    var e1x = wx[b] - wx[a], e1y = wy[b] - wy[a], e1z = wz[b] - wz[a];
    var e2x = wx[dd] - wx[a], e2y = wy[dd] - wy[a], e2z = wz[dd] - wz[a];
    var nxv = e1y * e2z - e1z * e2y, nyv = e1z * e2x - e1x * e2z, nzv = e1x * e2y - e1y * e2x;
    var nl = Math.sqrt(nxv * nxv + nyv * nyv + nzv * nzv) || 1;
    var dot = (nxv * light[0] + nyv * light[1] + nzv * light[2]) / nl;
    if (dot < 0) dot = -dot;
    var sh = 0.70 + 0.42 * dot, rgb = gdColor(q[5]);
    ctx.beginPath();
    ctx.moveTo(sx[a], sy[a]); ctx.lineTo(sx[b], sy[b]); ctx.lineTo(sx[cc2], sy[cc2]); ctx.lineTo(sx[dd], sy[dd]);
    ctx.closePath();
    ctx.fillStyle = "rgb(" + Math.min(255, rgb[0] * sh | 0) + "," + Math.min(255, rgb[1] * sh | 0) + "," + Math.min(255, rgb[2] * sh | 0) + ")";
    ctx.fill();
    ctx.strokeStyle = "rgba(200,226,250,.06)";
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }
  var path = L.path || [], pn = path.length;
  var spanU = 0, spanV = 0;
  if (pn > 0) {
    var last = path[pn - 1], ca = Number(last.a) || 0, cb = Number(last.b) || 0;
    for (i = 0; i < pn; i++) {
      var da = Math.abs((Number(path[i].a) || 0) - ca), db = Math.abs((Number(path[i].b) || 0) - cb);
      if (da > spanU) spanU = da;
      if (db > spanV) spanV = db;
    }
    var scl = 0.86 / Math.max(spanU, spanV, 1e-12);
    function toWorld(pp) {
      var du = ((Number(pp.a) || 0) - ca) * scl, dv = ((Number(pp.b) || 0) - cb) * scl;
      du = du < -0.98 ? -0.98 : (du > 0.98 ? 0.98 : du);
      dv = dv < -0.98 ? -0.98 : (dv > 0.98 ? 0.98 : dv);
      var nt2 = (Number(pp.z) - zmin) / range;
      if (nt2 < 0) nt2 = 0; if (nt2 > 1) nt2 = 1;
      return [du, dv, nt2 * zh - zh * 0.5 + 0.02];
    }
    var pts = [];
    for (i = 0; i < pn; i++) {
      var wpt = toWorld(path[i]), sp = gdProj(c, wpt[0], wpt[1], wpt[2]);
      pts.push({ x: ox + sp[0] * FOCAL, y: oy - sp[1] * FOCAL });
    }
    ctx.save();
    ctx.strokeStyle = "#ffb020";
    ctx.lineWidth = 2.2;
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(255,176,32,.7)";
    ctx.shadowBlur = 9;
    ctx.beginPath();
    for (i = 0; i < pts.length; i++) { if (i === 0) ctx.moveTo(pts[i].x, pts[i].y); else ctx.lineTo(pts[i].x, pts[i].y); }
    ctx.stroke();
    ctx.restore();
    var every = Math.max(1, Math.floor(pts.length / 40));
    for (i = 0; i < pts.length - 1; i++) {
      if (i % every !== 0) continue;
      ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 2.6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,211,106,.85)"; ctx.fill();
    }
    var lp = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(lp.x, lp.y, 7, 0, Math.PI * 2);
    ctx.strokeStyle = "#3ddc84"; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); ctx.arc(lp.x, lp.y, 3.4, 0, Math.PI * 2);
    ctx.fillStyle = "#3ddc84"; ctx.fill();
    ctx.font = '10px "Microsoft YaHei UI","Segoe UI",sans-serif';
    ctx.textAlign = "left"; ctx.fillStyle = "#8fe0b2";
    ctx.fillText("当前权重", lp.x + 11, lp.y + 10);
    ctx.textAlign = "right"; ctx.fillStyle = "#ffd58a";
    ctx.fillText("损失 " + fSmart(Number(last.z)), w - 16, 16);
    ctx.fillStyle = "rgba(120,140,168,.75)";
    ctx.fillText("轨迹跨度 u " + fSmart(spanU) + " · v " + fSmart(spanV) + "（已等比缩放填满窗口）", w - 16, 30);
  }
  ctx.textAlign = "left";
  ctx.font = '10px "Microsoft YaHei UI","Segoe UI",sans-serif';
  ctx.fillStyle = "rgba(170,208,240,.75)";
  ctx.fillText("真实测量损失曲面 · 网格 " + res + "×" + res + " · 窗口 ±" + fFix(ext, 2)
    + " · 轨迹 " + fInt(pn) + " 点", 8, 14);
  ctx.fillStyle = "rgba(120,140,168,.65)";
  ctx.fillText("拖动旋转 · 滚轮缩放 · 高度=固定评估批上的真实损失", 8, h - 8);
  var cbx = w - 14, cby = 34, cbh = 84, cbw = 8;
  for (i = 0; i < 32; i++) {
    var t2 = 1 - i / 31, rgb2 = gdColor(t2);
    ctx.fillStyle = "rgb(" + rgb2[0] + "," + rgb2[1] + "," + rgb2[2] + ")";
    ctx.fillRect(cbx, cby + cbh * i / 32, cbw, cbh / 32 + 1);
  }
  ctx.strokeStyle = "rgba(220,235,250,.2)";
  ctx.lineWidth = 1;
  ctx.strokeRect(cbx + .5, cby + .5, cbw, cbh);
  ctx.textAlign = "right";
  ctx.fillStyle = "#8794a3";
  ctx.fillText(fSmart(zmax), cbx - 5, cby + 8);
  ctx.fillText(fSmart(zmin), cbx - 5, cby + cbh);
  ctx.fillStyle = "#5f6a78";
  ctx.fillText("损失", cbx - 5, cby - 8);
}
function bindGD() {
  var cv = GD.cv;
  cv.addEventListener("pointerdown", function (e) {
    GD.drag = true; GD.lx = e.clientX; GD.ly = e.clientY; cv.style.cursor = "grabbing";
    if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", function (e) {
    if (!GD.drag) return;
    GD.yaw -= (e.clientX - GD.lx) * 0.011;
    GD.pitch = Math.max(0.08, Math.min(1.45, GD.pitch + (e.clientY - GD.ly) * 0.009));
    GD.lx = e.clientX; GD.ly = e.clientY;
    requestGD();
  });
  function stop() { if (!GD.drag) return; GD.drag = false; cv.style.cursor = "grab"; }
  cv.addEventListener("pointerup", stop);
  cv.addEventListener("pointercancel", stop);
  cv.addEventListener("pointerleave", stop);
  cv.addEventListener("wheel", function (e) {
    e.preventDefault();
    GD.dist = Math.max(1.7, Math.min(11, GD.dist * Math.exp(e.deltaY * 0.0012)));
    requestGD();
  }, { passive: false });
}
var gdRaf = 0;
function requestGD() {
  if (gdRaf) return;
  gdRaf = requestAnimationFrame(function () { gdRaf = 0; drawGD(); });
}

var NET = {
  ready: false, layers: [], edges: [], act: [], pick: [], maxS: 1, totalEdges: 0,
  hover: null, hoverKey: "", view: null, mode: "3d", budget: 1200,
  pairs: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
  cam: { yaw: -1.05, pitch: 0.30, dist: 40.0, drag: false, lx: 0, ly: 0 },
  cv: null, ctx: null
};
var NET_FONT = '10px "Microsoft YaHei UI","Segoe UI",sans-serif';

function netColorRange(ly) {
  var cnt = Math.max(1, ly.nodeCount | 0), raw = ly.nodes || [], lo = Infinity, hi = -Infinity, j, v;
  for (j = 0; j < cnt; j++) {
    v = Number(raw[j]);
    if (!isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return isFinite(lo) && isFinite(hi) ? [lo, hi] : [0, 0];
}
function netTotalNodes() {
  var t = 0, i;
  for (i = 0; i < NET.layers.length; i++) t += NET.layers[i].nodeCount | 0;
  return t;
}
function netColor(t) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  var r, g, b;
  if (t < 0.5) { var u = t / 0.5; r = 16 + 31 * u; g = 30 + 81 * u; b = 42 + 109 * u; }
  else { var v = (t - 0.5) / 0.5; r = 47 + 93 * v; g = 111 + 113 * v; b = 151 + 99 * v; }
  return "rgb(" + (r | 0) + "," + (g | 0) + "," + (b | 0) + ")";
}
function netNodeAct(ly) {
  var cnt = Math.max(1, ly.nodeCount | 0), raw = ly.nodes || [], sc = netColorRange(ly);
  var span = sc[1] - sc[0], out = [], j, v;
  for (j = 0; j < cnt; j++) {
    v = Number(raw[j]);
    var t = isFinite(v) ? (span > 1e-12 ? (v - sc[0]) / span : 0.5) : 0.5;
    out.push(t < 0 ? 0 : (t > 1 ? 1 : t));
  }
  return out;
}
function buildNetPick() {
  var L = NET.layers, E = NET.edges, n = L.length, i, e, j, k;
  NET.act = [];
  for (i = 0; i < n; i++) NET.act.push(netNodeAct(L[i]));
  var pairs = Math.max(1, NET.pairs.length);
  var perPair = Math.max(24, Math.round(NET.budget / pairs));
  var all = [], total = 0;
  for (e = 0; e < pairs; e++) {
    var pr = NET.pairs[e] || [e, e + 1];
    if (!L[pr[0]] || !L[pr[1]]) continue;
    var src = NET.act[pr[0]], sN = Math.max(1, L[pr[0]].nodeCount | 0), tN = Math.max(1, L[pr[1]].nodeCount | 0);
    var M = E[e] || [];
    total += sN * tN;
    var kPer = Math.max(1, Math.round(perPair / tN));
    for (k = 0; k < tN; k++) {
      var row = M[k] || [], tmp = [];
      for (j = 0; j < sN; j++) {
        var w = Number(row[j]);
        if (!isFinite(w) || w <= 0) continue;
        tmp.push({ e: e, s: pr[0], t: pr[1], j: j, i: k, w: w, a: src[j], v: w * (0.28 + 0.72 * src[j]) });
      }
      tmp.sort(function (p, q) { return q.v - p.v; });
      var lim = Math.min(kPer, tmp.length);
      for (j = 0; j < lim; j++) all.push(tmp[j]);
    }
  }
  all.sort(function (p, q) { return p.v - q.v; });
  NET.pick = all;
  NET.totalEdges = total;
  NET.strong = all.length ? all[all.length - 1] : null;
  NET.maxS = 1e-9;
  for (i = 0; i < all.length; i++) if (all[i].v > NET.maxS) NET.maxS = all[i].v;
}
function netStats(nodes) {
  var lo = Infinity, hi = -Infinity, s = 0, n = 0, i, v;
  for (i = 0; i < nodes.length; i++) {
    v = Number(nodes[i]);
    if (!isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    s += v; n++;
  }
  return n ? { min: lo, max: hi, mean: s / n } : { min: 0, max: 0, mean: 0 };
}
function prepNet() {
  NET.ready = false;
  if (!GRAPH || !STATE) return;
  var layers = GRAPH.layers || [], acts = STATE.acts || [];
  if (layers.length < 2) return;
  var merged = [];
  for (var i = 0; i < layers.length; i++) {
    var l = layers[i], nodes = acts[i] || [], st = netStats(nodes);
    merged.push({
      name: l.name, detail: l.detail, nodeCount: l.nodeCount | 0,
      params: l.params, realCount: l.realCount, out: l.out,
      nodes: nodes, actMin: st.min, actMax: st.max, actMean: st.mean
    });
  }
  NET.layers = merged;
  NET.edges = GRAPH.edges || [];
  NET.ready = true;
  buildNetPick();
}
function netGrid(count) {
  if (count <= 1) return [1, 1];
  var cols = Math.max(1, Math.ceil(Math.sqrt(count * 1.9)));
  return [cols, Math.ceil(count / cols)];
}
function netCam(yaw, pitch, dist) {
  var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  var ex = cp * cy, ey = cp * sy, ez = sp;
  return { ox: ex * dist, oy: ey * dist, oz: ez * dist, fx: -ex, fy: -ey, fz: -ez, rx: -sy, ry: cy, rz: 0, ux: -cy * sp, uy: -sy * sp, uz: cp };
}
function netProj(c, x, y, z) {
  var vx = x - c.ox, vy = y - c.oy, vz = z - c.oz;
  var cz = vx * c.fx + vy * c.fy + vz * c.fz;
  if (cz < 0.15) cz = 0.15;
  return [(vx * c.rx + vy * c.ry + vz * c.rz) / cz, (vx * c.ux + vy * c.uy + vz * c.uz) / cz, cz];
}
function netLayout() {
  var L = NET.layers, n = L.length, i, j, gap = 2.6, pitch = 0.40, zSpan = (n - 1) * gap, boxes = [];
  for (i = 0; i < n; i++) {
    var cnt = Math.max(1, L[i].nodeCount | 0);
    var cr = netGrid(cnt), cols = cr[0], rows = cr[1];
    var x = i * gap - zSpan * 0.5, pts = [];
    for (j = 0; j < cnt; j++) {
      var g = Math.floor(j / cols), c = j % cols;
      pts.push([(c - (cols - 1) * 0.5) * pitch, ((rows - 1) * 0.5 - g) * pitch]);
    }
    boxes.push({ x: x, cols: cols, rows: rows, pts: pts, cnt: cnt, halfY: (cols - 1) * 0.5 * pitch, halfZ: (rows - 1) * 0.5 * pitch });
  }
  return { boxes: boxes, gap: gap, zSpan: zSpan };
}
function netFitFocal(lay, w, h) {
  var c = netCam(NET.cam.yaw, NET.cam.pitch, 40.0);
  var mx = 1e-4, my = 1e-4, i, k;
  for (i = 0; i < lay.boxes.length; i++) {
    var b = lay.boxes[i], ey = b.halfY + 0.30, ez = b.halfZ + 0.30;
    var cand = [[b.x, ey, ez], [b.x, -ey, ez], [b.x, ey, -ez], [b.x, -ey, -ez]];
    for (k = 0; k < 4; k++) {
      var p = netProj(c, cand[k][0], cand[k][1], cand[k][2]);
      if (Math.abs(p[0]) > mx) mx = Math.abs(p[0]);
      if (Math.abs(p[1]) > my) my = Math.abs(p[1]);
    }
  }
  return Math.min(w * 0.46 / mx, h * 0.42 / my);
}
function netEmptyFrame(ctx, w, h) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = "rgba(120,140,168,.24)";
  ctx.lineWidth = 1;
  ctx.strokeRect(15.5, 15.5, w - 31, h - 31);
  ctx.restore();
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = '13px "Microsoft YaHei UI","Segoe UI",sans-serif'; ctx.fillStyle = "#5f6a78";
  ctx.fillText("正在采集网络激活…", w * 0.5, h * 0.5 - 11);
  ctx.font = '11px "Microsoft YaHei UI","Segoe UI",sans-serif'; ctx.fillStyle = "#4a545f";
  ctx.fillText("等待训练循环产生前向快照", w * 0.5, h * 0.5 + 11);
  ctx.textBaseline = "alphabetic";
}
function drawNet3D() {
  var cv = NET.cv, ctx = NET.ctx;
  if (!cv || !ctx) return;
  var w = cv.clientWidth | 0, h = cv.clientHeight | 0;
  if (w < 40 || h < 40) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2), nw = Math.round(w * dpr), nh = Math.round(h * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!NET.ready) { NET.view = null; netEmptyFrame(ctx, w, h); return; }
  var L = NET.layers, n = L.length, i, j, lay = netLayout();
  var FOCAL = netFitFocal(lay, w, h), c = netCam(NET.cam.yaw, NET.cam.pitch, NET.cam.dist);
  var ox = w * 0.5, oy = h * 0.52, act = NET.act || [], scr = [], flat = [];
  for (i = 0; i < n; i++) {
    var b = lay.boxes[i], A = act[i] || netNodeAct(L[i]), col = [];
    for (j = 0; j < b.cnt; j++) {
      var pt = b.pts[j], p = netProj(c, b.x, pt[0], pt[1]);
      var rr = 0.055 * FOCAL / p[2];
      var nd = {
        x: ox + p[0] * FOCAL, y: oy - p[1] * FOCAL, z: p[2],
        r: rr < 0.45 ? 0.45 : (rr > 6.5 ? 6.5 : rr), li: i, bi: j, a: A[j]
      };
      col.push(nd); flat.push(nd);
    }
    scr.push(col);
  }
  NET.view = { nodes: flat };
  var zmin = Infinity, zmax = -Infinity;
  for (i = 0; i < flat.length; i++) { if (flat[i].z < zmin) zmin = flat[i].z; if (flat[i].z > zmax) zmax = flat[i].z; }
  var zspan = (zmax - zmin) || 1;
  function depthF(z) { var q = (z - zmin) / zspan; return q < 0 ? 0 : (q > 1 ? 1 : q); }
  var pick = NET.pick, edges = [];
  for (i = 0; i < pick.length; i++) {
    var q = pick[i], A2 = scr[q.s] && scr[q.s][q.j], B2 = scr[q.t] && scr[q.t][q.i];
    if (!A2 || !B2) continue;
    var st = q.v / (NET.maxS || 1);
    if (st < 0.03) continue;
    edges.push({ A: A2, B: B2, st: st, z: (A2.z + B2.z) * 0.5 });
  }
  edges.sort(function (p, q2) { return q2.z - p.z; });
  ctx.lineCap = "round";
  for (i = 0; i < edges.length; i++) {
    var ed = edges[i], fade = 1 - 0.55 * depthF(ed.z);
    ctx.strokeStyle = "rgba(150,205,246," + ((0.06 + 0.68 * ed.st) * fade).toFixed(3) + ")";
    ctx.lineWidth = 0.5 + 1.7 * ed.st;
    ctx.beginPath(); ctx.moveTo(ed.A.x, ed.A.y); ctx.lineTo(ed.B.x, ed.B.y); ctx.stroke();
  }
  ctx.lineCap = "butt";
  var order = [];
  for (i = 0; i < flat.length; i++) order.push(i);
  order.sort(function (p, q2) { return flat[q2].z - flat[p].z; });
  for (var k = 0; k < order.length; k++) {
    var nd2 = flat[order[k]], fade2 = 1 - 0.45 * depthF(nd2.z);
    if (nd2.a > 0.62) {
      ctx.beginPath(); ctx.arc(nd2.x, nd2.y, nd2.r + 0.6 + 1.8 * nd2.a, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(150,214,255," + (0.05 + 0.26 * nd2.a).toFixed(3) + ")";
      ctx.fill();
    }
    ctx.beginPath(); ctx.arc(nd2.x, nd2.y, nd2.r, 0, Math.PI * 2);
    ctx.fillStyle = netColor(nd2.a); ctx.fill();
  }
  var st2 = NET.strong;
  if (st2) {
    var SA = scr[st2.s] && scr[st2.s][st2.j], SB = scr[st2.t] && scr[st2.t][st2.i];
    if (SA && SB) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,176,32,.85)";
      ctx.lineWidth = 1.8;
      ctx.shadowColor = "rgba(255,176,32,.8)";
      ctx.shadowBlur = 9;
      ctx.beginPath(); ctx.moveTo(SA.x, SA.y); ctx.lineTo(SB.x, SB.y); ctx.stroke();
      ctx.restore();
    }
  }
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  for (i = 0; i < n; i++) {
    var bb = lay.boxes[i], lyr = L[i], lp = netProj(c, bb.x, 0, bb.halfZ + 0.42);
    var lx = ox + lp[0] * FOCAL, ly = oy - lp[1] * FOCAL, al2 = 0.55 + 0.45 * (1 - depthF(lp[2]));
    ctx.font = '700 12px "Microsoft YaHei UI","Segoe UI",sans-serif';
    ctx.fillStyle = "rgba(223,231,239," + al2.toFixed(2) + ")";
    ctx.fillText(String(lyr.name || ""), lx, ly);
    ctx.font = NET_FONT;
    ctx.fillStyle = "rgba(139,149,163," + al2.toFixed(2) + ")";
    ctx.fillText(String(lyr.detail || ""), lx, ly + 13);
    ctx.fillStyle = "rgba(102,112,125," + al2.toFixed(2) + ")";
    ctx.fillText(lyr.nodeCount + " 节点 · 均值 " + fSmart(Number(lyr.actMean)), lx, ly + 25);
  }
  var hv = NET.hover;
  if (hv && scr[hv.li] && scr[hv.li][hv.bi]) {
    var hn = scr[hv.li][hv.bi];
    ctx.strokeStyle = "#e8eef5"; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(hn.x, hn.y, hn.r + 3.4, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.font = NET_FONT; ctx.textAlign = "left";
  ctx.fillStyle = "rgba(170,208,240,.8)";
  ctx.fillText("3D 立体 · 连线亮度 = 权重 × 源激活 · 每秒刷新 · 拖动旋转 · 滚轮缩放", 8, 14);
}
function drawNet2D() {
  var cv = NET.cv, ctx = NET.ctx;
  if (!cv || !ctx) return;
  var w = cv.clientWidth | 0, h = cv.clientHeight | 0;
  if (w < 40 || h < 40) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2), nw = Math.round(w * dpr), nh = Math.round(h * dpr);
  if (cv.width !== nw || cv.height !== nh) { cv.width = nw; cv.height = nh; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!NET.ready) { NET.view = null; netEmptyFrame(ctx, w, h); return; }
  var L = NET.layers, n = L.length, i, j;
  var padX = Math.max(34, Math.min(74, w * 0.05));
  var step = n > 1 ? (w - padX * 2) / (n - 1) : 0;
  var yTop = 60, yBot = h - 80;
  if (yBot - yTop < 100) { yTop = 48; yBot = h - 64; }
  var cols = [], nodes = [];
  for (i = 0; i < n; i++) {
    var ly = L[i], cnt = Math.max(1, ly.nodeCount | 0), x = padX + step * i;
    var sp = cnt > 1 ? (yBot - yTop) / (cnt - 1) : 0;
    var rad = cnt > 1 ? Math.max(0.45, Math.min(5.2, sp * 0.42)) : 6.5;
    var ys = [];
    for (j = 0; j < cnt; j++) ys.push(cnt > 1 ? yTop + sp * j : (yTop + yBot) * 0.5);
    var col = { x: x, ys: ys, r: rad, act: NET.act[i] || netNodeAct(ly), sc: netColorRange(ly), ly: ly };
    cols.push(col);
    for (j = 0; j < ys.length; j++) nodes.push({ x: x, y: ys[j], r: rad, li: i, bi: j, a: col.act[j] });
  }
  NET.view = { nodes: nodes, cols: cols };
  var pick = NET.pick, maxS = NET.maxS || 1;
  ctx.lineCap = "round";
  for (i = 0; i < pick.length; i++) {
    var q = pick[i], qa = cols[q.s], qb = cols[q.t];
    if (!qa || !qb) continue;
    var al = q.v / maxS;
    if (al < 0.03) continue;
    ctx.strokeStyle = "rgba(150,205,246," + (0.06 + 0.70 * al).toFixed(3) + ")";
    ctx.lineWidth = 0.5 + 1.7 * al;
    ctx.beginPath(); ctx.moveTo(qa.x + qa.r, qa.ys[q.j]); ctx.lineTo(qb.x - qb.r, qb.ys[q.i]); ctx.stroke();
  }
  ctx.lineCap = "butt";
  for (i = 0; i < n; i++) {
    var cc = cols[i];
    for (j = 0; j < cc.ys.length; j++) {
      var a = cc.act[j];
      if (a > 0.62) {
        ctx.beginPath(); ctx.arc(cc.x, cc.ys[j], cc.r + 0.6 + 1.8 * a, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(150,214,255," + (0.05 + 0.26 * a).toFixed(3) + ")";
        ctx.fill();
      }
      ctx.beginPath(); ctx.arc(cc.x, cc.ys[j], cc.r, 0, Math.PI * 2);
      ctx.fillStyle = netColor(a); ctx.fill();
    }
  }
  var st = NET.strong;
  if (st) {
    var sa = cols[st.s], sb = cols[st.t];
    var x0 = sa.x + sa.r, y0 = sa.ys[st.j], x1 = sb.x - sb.r, y1 = sb.ys[st.i];
    ctx.save();
    ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(255,176,32,.85)"; ctx.shadowBlur = 11;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = NET_FONT;
    ctx.fillStyle = "#ffd58a";
    ctx.fillText("最强通路 强度 " + st.v.toFixed(3), 6, h - 8);
    ctx.textBaseline = "alphabetic";
  }
  ctx.textAlign = "center";
  for (i = 0; i < n; i++) {
    var cl = cols[i], lyr = cl.ly;
    ctx.font = '700 12px "Microsoft YaHei UI","Segoe UI",sans-serif';
    ctx.fillStyle = "#dfe7ef";
    ctx.fillText(String(lyr.name || ""), cl.x, 16);
    ctx.font = NET_FONT; ctx.fillStyle = "#8b95a3";
    ctx.fillText(String(lyr.detail || ""), cl.x, 30);
    ctx.fillStyle = "#66707d"; ctx.textAlign = "center";
    ctx.fillText(cl.ys.length + " 个节点 · 共 " + fInt(lyr.realCount) + " 个值", cl.x, 43);
    var bw2 = Math.max(40, Math.min(150, step > 0 ? step * 0.62 : 90));
    var gx = cl.x - bw2 * 0.5, gy = h - 62;
    for (j = 0; j < 24; j++) {
      ctx.fillStyle = netColor(j / 23);
      ctx.fillRect(gx + bw2 * j / 24, gy, bw2 / 24 + 1, 7);
    }
    ctx.strokeStyle = "rgba(220,235,250,.2)";
    ctx.lineWidth = 1;
    ctx.strokeRect(gx + .5, gy + .5, bw2, 7);
    ctx.font = NET_FONT; ctx.fillStyle = "#8794a3";
    ctx.textAlign = "left"; ctx.fillText(fSmart(cl.sc[0]), gx, gy + 21);
    ctx.textAlign = "right"; ctx.fillText(fSmart(cl.sc[1]), gx + bw2, gy + 21);
    ctx.textAlign = "center"; ctx.fillStyle = "#79828f";
    ctx.fillText("均值 " + fSmart(Number(lyr.actMean)), cl.x, gy + 36);
    ctx.fillStyle = "#66707d";
    ctx.fillText("全层 " + fSmart(Number(lyr.actMin)) + " ~ " + fSmart(Number(lyr.actMax)), cl.x, gy + 48);
  }
  if (NET.hover && cols[NET.hover.li]) {
    var hc = cols[NET.hover.li], hy = hc.ys[NET.hover.bi];
    if (hy !== undefined) {
      ctx.strokeStyle = "#e8eef5"; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(hc.x, hy, hc.r + 3.6, 0, Math.PI * 2); ctx.stroke();
    }
  }
  ctx.font = NET_FONT; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "rgba(170,208,240,.8)";
  ctx.fillText("2D 平面 · 连线亮度 = 权重 × 源激活 · 每秒刷新", 8, 14);
}
function drawNet() {
  if (NET.mode === "3d") drawNet3D(); else drawNet2D();
}
var netRaf = 0;
function requestNet() {
  if (netRaf) return;
  netRaf = requestAnimationFrame(function () { netRaf = 0; drawNet(); });
}
function netHit(mx, my) {
  var v = NET.view;
  if (!v) return null;
  var best = null, bd = Infinity, i, nd, d, rr;
  for (i = 0; i < v.nodes.length; i++) {
    nd = v.nodes[i];
    d = (nd.x - mx) * (nd.x - mx) + (nd.y - my) * (nd.y - my);
    rr = nd.r + 4;
    if (d <= rr * rr && d < bd) { bd = d; best = nd; }
  }
  return best;
}
function netTipHTML(hit) {
  var L = NET.layers, ly = L[hit.li], j = hit.bi;
  var cnt = Math.max(1, ly.nodeCount | 0);
  var h = '<div class="th">' + esc(ly.name) + ' · 第 ' + (j + 1) + ' / ' + cnt + ' 个</div>';
  h += '<div class="tr"><span>激活值</span><i>' + fSmart(Number((ly.nodes || [])[j])) + '</i></div>';
  h += '<div class="tr"><span>本层真实值总数</span><i>' + fInt(ly.realCount) + '</i></div>';
  if (ly.name === "输入") h += '<div class="tr"><span>含义</span><i>第 ' + (j + 1) + ' 帧 · ' + fInt((ly.realCount || 0) / cnt) + ' 像素</i></div>';
  else if (ly.name === "FC" || ly.name === "输出") h += '<div class="tr"><span>含义</span><i>第 ' + (j + 1) + ' 个神经元</i></div>';
  else h += '<div class="tr"><span>含义</span><i>第 ' + (j + 1) + ' 个通道</i></div>';
  h += '<div class="tr"><span>本层范围</span><i>' + fSmart(ly.actMin) + ' ~ ' + fSmart(ly.actMax) + '</i></div>';
  var gi = -1, kk;
  for (kk = 0; kk < NET.pairs.length; kk++) if (NET.pairs[kk][0] === hit.li) { gi = kk; break; }
  if (gi >= 0) {
    var tgt = NET.pairs[gi][1], M = NET.edges[gi] || [], tN = Math.max(1, L[tgt].nodeCount | 0), arr = [], k;
    for (k = 0; k < tN; k++) {
      var row = M[k] || [], wv = Number(row[j]);
      if (isFinite(wv)) arr.push([k, wv]);
    }
    arr.sort(function (p, q) { return q[1] - p[1]; });
    h += '<div class="th sub">输出连接最强 5 条 → ' + esc(L[tgt].name) + '（实际共 ' + fInt(tN) + ' 条）</div>';
    for (k = 0; k < Math.min(5, arr.length); k++) {
      h += '<div class="tr"><span>' + esc(L[tgt].name) + ' 第 ' + (arr[k][0] + 1) + ' 个</span><i>' + fFix(arr[k][1], 4) + '</i></div>';
    }
  } else {
    h += '<div class="tr"><span>输出层</span><i>无后续连接</i></div>';
  }
  return h;
}
function netTipShow(hit, cx, cy) {
  var tip = el("nettip"), wrap = el("netwrap");
  if (!tip || !wrap) return;
  tip.innerHTML = netTipHTML(hit);
  tip.className = "nettip on";
  var ww = wrap.clientWidth, wh = wrap.clientHeight, tw = tip.offsetWidth, th = tip.offsetHeight;
  var x = cx + 16, y = cy + 14;
  if (x + tw > ww - 6) x = cx - tw - 16;
  if (x < 6) x = 6;
  if (y + th > wh - 6) y = cy - th - 14;
  if (y < 6) y = 6;
  tip.style.left = Math.round(x) + "px";
  tip.style.top = Math.round(y) + "px";
}
function setNetMode(v) {
  NET.mode = (v === "2d") ? "2d" : "3d";
  var a = el("v-3d"), b = el("v-2d");
  if (a) a.className = "vbtn" + (NET.mode === "3d" ? " on" : "");
  if (b) b.className = "vbtn" + (NET.mode === "2d" ? " on" : "");
  if (NET.cv) NET.cv.style.cursor = (NET.mode === "3d") ? "grab" : "default";
  requestNet();
}
function bindNet() {
  var cv = NET.cv, tip = el("nettip");
  if (!cv) return;
  cv.addEventListener("pointerdown", function (e) {
    if (NET.mode !== "3d") return;
    NET.cam.drag = true; NET.cam.lx = e.clientX; NET.cam.ly = e.clientY;
    cv.style.cursor = "grabbing";
    if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", function (e) {
    if (NET.mode !== "3d" || !NET.cam.drag) return;
    NET.cam.yaw -= (e.clientX - NET.cam.lx) * 0.010;
    NET.cam.pitch = Math.max(-1.35, Math.min(1.35, NET.cam.pitch + (e.clientY - NET.cam.ly) * 0.008));
    NET.cam.lx = e.clientX; NET.cam.ly = e.clientY;
    requestNet();
  });
  function stop() {
    if (!NET.cam.drag) return;
    NET.cam.drag = false;
    cv.style.cursor = (NET.mode === "3d") ? "grab" : "default";
  }
  cv.addEventListener("pointerup", stop);
  cv.addEventListener("pointercancel", stop);
  cv.addEventListener("pointerleave", stop);
  cv.addEventListener("wheel", function (e) {
    if (NET.mode !== "3d") return;
    e.preventDefault();
    NET.cam.dist = Math.max(12, Math.min(90, NET.cam.dist * Math.exp(e.deltaY * 0.0011)));
    requestNet();
  }, { passive: false });
  cv.addEventListener("mousemove", function (ev) {
    if (NET.cam.drag) return;
    var rect = cv.getBoundingClientRect();
    var mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    var hit = netHit(mx, my);
    var key = hit ? (hit.li + ":" + hit.bi) : "";
    if (key === NET.hoverKey) { if (hit) netTipShow(hit, mx, my); return; }
    NET.hoverKey = key;
    NET.hover = hit;
    if (hit) netTipShow(hit, mx, my);
    else if (tip) { tip.className = "nettip"; tip.innerHTML = ""; }
    requestNet();
  });
  cv.addEventListener("mouseleave", function () {
    if (!NET.hoverKey) return;
    NET.hoverKey = ""; NET.hover = null;
    if (tip) { tip.className = "nettip"; tip.innerHTML = ""; }
    requestNet();
  });
  var list = document.querySelectorAll("[data-view]"), i;
  for (i = 0; i < list.length; i++) {
    list[i].addEventListener("click", function () { setNetMode(this.getAttribute("data-view")); });
  }
}

function renderSurface(d) {
  var s = d.surface;
  GD.land = (s && Number(s.ready) === 1 && s.values && s.values.length >= 4) ? s : null;
  if (GD.land) {
    setT("gdSub", "网格 " + GD.land.res + "×" + GD.land.res + " · 窗口 ±" + fFix(GD.land.ext, 2)
      + " · 轨迹 " + fInt((GD.land.path || []).length) + " 点");
  } else {
    setT("gdSub", "等待首次测量…");
  }
  setSec("secSurface", !!GD.land);
  requestGD();
}

function setConn(ok) {
  var e = el("conn"), t = el("connTxt");
  if (!ok) { e.className = "conn bad"; t.textContent = "连接断开"; return; }
  e.className = "conn ok"; t.textContent = "已连接";
}

function ctlSend(act) {
  if (CTL.busy) return;
  CTL.busy = true;
  var msg = el("hMsg"), bs = el("h-start"), bt = el("h-stop");
  if (msg) msg.textContent = act === "start" ? "正在启动…" : "正在结束并保存…";
  if (bs) bs.disabled = true;
  if (bt) bt.disabled = true;
  fetch("/api/control/" + act, { method: "POST", cache: "no-store" })
    .then(function (r) { return r.json(); })
    .then(function (j) {
      if (j.ok) {
        if (msg) msg.textContent = act === "start" ? "已启动训练" : "已结束训练，存档已保存";
        if (STATE) STATE.training = j.training;
        if (STATE) renderHeader(STATE);
      } else {
        if (msg) msg.textContent = "失败：" + (j.msg || "未知错误");
        if (STATE) renderHeader(STATE);
      }
    })
    .catch(function () { if (msg) msg.textContent = "请求失败，服务可能未运行"; })
    .then(function () { CTL.busy = false; });
}

var inflight = false;
function poll() {
  if (inflight) { setTimeout(poll, 200); return; }
  inflight = true;
  fetch("/api/state", { cache: "no-store" })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      STATE = d;
      setConn(true);
      renderHeader(d);
      renderCards(d);
      renderCharts(d);
      renderHealth(d);
      renderImages(d);
      renderHists(d);
      renderTables(d);
      renderMatrices(d);
      renderScatters(d);
      renderBars(d);
      renderLogs(d);
      renderParams(d);
      renderSurface(d);
      renderAlerts(d);
      buildNav();
      var gv = d.graph ? d.graph.version : 0;
      if (gv && gv !== GRAPH_VER) {
        return fetch("/api/graph", { cache: "no-store" })
          .then(function (r) { return r.json(); })
          .then(function (g) { GRAPH = g; GRAPH_VER = gv; });
      }
      return null;
    })
    .then(function () {
      prepNet();
      setSec("secNet", !!(GRAPH && (GRAPH.layers || []).length >= 2));
      if (GRAPH) {
        setT("netSub", "神经元 " + fInt(GRAPH.totalNodes) + " 个（全部显示）· 连线已绘 "
          + fInt(NET.pick.length) + " / 共 " + fInt(GRAPH.totalEdges) + " 条");
      }
      requestNet();
    })
    .catch(function () { setConn(false); })
    .then(function () {
      inflight = false;
      setTimeout(poll, Math.max(300, (STATE && STATE.interval ? STATE.interval : 1) * 1000));
    });
}

function init() {
  NET.cv = el("netcv");
  NET.ctx = NET.cv.getContext("2d");
  GD.cv = el("gdcv");
  GD.ctx = GD.cv.getContext("2d");
  bindNet();
  bindGD();
  bindMatrixHover();
  bindCharts();
  bindChartCtl();
  initFold();
  setNetMode("3d");
  var bs = el("h-start"), bt = el("h-stop");
  if (bs) bs.addEventListener("click", function () { ctlSend("start"); });
  if (bt) bt.addEventListener("click", function () { ctlSend("stop"); });
  window.addEventListener("resize", function () {
    requestNet(); requestGD();
    if (STATE) { renderMatrices(STATE); renderScatters(STATE); renderCharts(STATE); }
  });
  var scTick = false;
  window.addEventListener("scroll", function () {
    if (scTick) return;
    scTick = true;
    requestAnimationFrame(function () { scTick = false; markNav(); });
  });
  drawGD();
  poll();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();

})();
