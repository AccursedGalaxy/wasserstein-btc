// wbtc viewer — vanilla ES module, ECharts for visualisations.

const PALETTE_WGEO = ["#7cf2d6", "#5dd6a8", "#a5f5b8", "#5fe5c6", "#7adbb0", "#3fc2a5"];
const PALETTE_BASELINE = ["#7c83f2", "#9aa0f5", "#6068d8", "#aab0fa", "#8088ee", "#5560cd"];
const PALETTE_EXT = ["#f2c97c", "#f0a64a", "#e8d088", "#d8a04a", "#f5d99a", "#d68f3a"];

const FAMILY_BY_METHOD = new Map();

const METHOD_DESC = {
  // baselines
  "Static": ["baseline", "Unconditional empirical quantiles of the training window — the constant predictor that's surprisingly hard to beat."],
  "RW-Drift": ["baseline", "Random walk with drift: same empirical quantiles, shifted by the windowed mean return."],
  "HS-Bootstrap": ["baseline", "Historical simulation via stationary bootstrap on the training tail."],
  "GARCH-N": ["baseline", "GARCH(1,1) with Gaussian innovations — the workhorse volatility model."],
  "GARCH-t": ["baseline", "GARCH(1,1) with Student-t innovations, fit per step."],
  "GJR-GARCH-t": ["baseline", "Asymmetric GARCH (Glosten-Jagannathan-Runkle) with Student-t innovations."],
  // wgeo family
  "WGeo": ["wgeo", "The base method: tangent-space regression of empirical quantile functions along a 2-Wasserstein geodesic."],
  "WGeo-Gated": ["wgeo", "Adds a curvature gate at h=1 — falls back to the baseline when the trajectory is locally noisy."],
  "WGeo-TheilSen": ["wgeo", "Robust Theil-Sen slope for the geodesic tangent — resistant to outlier days."],
  "WGeo-EWMA": ["wgeo", "Recency-weighted slope: exponentially down-weights older training points."],
  "WGeo-Hetero": ["wgeo", "Dispersion conditioned on a GARCH-style scale estimate."],
  "WGeo-GARCH-Ens": ["wgeo", "Regime-aware mixture of WGeo and GARCH-t. The v0.3 headline forecaster."],
  // extended comparators
  "HAR-RV": ["ext", "Heterogeneous Autoregressive model on realized volatility (Corsi 2009)."],
  "CAViaR-SAV": ["ext", "Conditional Autoregressive Value-at-Risk, symmetric-absolute-value (Engle-Manganelli 2004)."],
  "MS-Normal-2": ["ext", "Two-state Markov-switching Normal (Hamilton 1989)."],
  "FIGARCH(1,d,0)": ["ext", "Fractionally integrated GARCH with long-memory (Baillie-Bollerslev-Mikkelsen 1996)."],
  "SV-AR1": ["ext", "AR(1) stochastic-volatility model (Taylor 1982 / Harvey-Ruiz-Shephard 1994), Kalman-QML."],
  "BVAR-GARCH(BTC,ETH)": ["ext", "Bivariate VAR + GARCH using BTC and ETH jointly."],
};

for (const [name, [fam]] of Object.entries(METHOD_DESC)) {
  FAMILY_BY_METHOD.set(name, fam);
}

function colorFor(name) {
  const fam = FAMILY_BY_METHOD.get(name) || "baseline";
  const list = fam === "wgeo" ? PALETTE_WGEO : fam === "ext" ? PALETTE_EXT : PALETTE_BASELINE;
  // stable hash by method-name → index
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return list[Math.abs(h) % list.length];
}

function fmtPct(x, signed = true) {
  if (!Number.isFinite(x)) return "—";
  const sign = signed && x > 0 ? "+" : "";
  return `${sign}${(x * 100).toFixed(1)}%`;
}
function fmtNum(x, d = 4) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(d);
}
function fmtP(p) {
  if (!Number.isFinite(p)) return "—";
  if (p < 0.001) return "< .001";
  if (p < 0.01) return p.toFixed(3);
  return p.toFixed(2);
}

let DATA = null;
const charts = new Map(); // dom id → echarts instance

window.addEventListener("resize", () => {
  for (const inst of charts.values()) inst.resize();
});

function getChart(id) {
  let inst = charts.get(id);
  const dom = document.getElementById(id);
  if (!dom) return null;
  if (!inst) {
    inst = echarts.init(dom, null, { renderer: "canvas" });
    charts.set(id, inst);
  }
  return inst;
}

function disposeChart(id) {
  const inst = charts.get(id);
  if (inst) { inst.dispose(); charts.delete(id); }
}

// =====================================================
// Theme helpers
// =====================================================
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function baseChartTheme() {
  const fg = cssVar("--fg");
  const mute = cssVar("--fg-mute");
  const dim = cssVar("--fg-dim");
  const grid = cssVar("--gridline");
  const cardBg = cssVar("--bg-card");
  const border = cssVar("--border");
  return {
    textStyle: { color: fg, fontFamily: cssVar("--font-sans") },
    axisLine: { lineStyle: { color: border } },
    axisLabel: { color: dim, fontSize: 11 },
    splitLine: { lineStyle: { color: grid } },
    tooltipBg: cardBg,
    tooltipBorder: border,
    tooltipFg: fg,
    mute,
    dim,
  };
}

// =====================================================
// 1. Load data
// =====================================================
async function load() {
  const res = await fetch("./data.json");
  if (!res.ok) throw new Error(`failed to load data.json: ${res.status}`);
  return res.json();
}

// =====================================================
// 2. Render: hero stats
// =====================================================
function renderHero() {
  document.getElementById("version-chip").textContent = `v${DATA.version}`;
  const head = DATA.provenance?.headline || [];
  const wins = head.filter((h) => parseFloat(h.improvement) < 0).length;
  const total = head.length;
  const meanImp = head.length
    ? head.reduce((s, h) => s + parseFloat(h.improvement), 0) / head.length
    : 0;
  const sigCount = head.filter((h) => Number.isFinite(h.dm_p) && h.dm_p < 0.1).length;
  const totalN = head.reduce((s, h) => s + (h.n_test || 0), 0);

  const stats = [
    {
      label: "Walk-forward wins",
      val: `${wins} / ${total}`,
      sub: "WGeo family vs best baseline",
      cls: wins > total / 2 ? "good" : "",
    },
    {
      label: "Mean Δ CRPS",
      val: `${meanImp.toFixed(2)}%`,
      sub: "averaged across (asset × horizon)",
      cls: meanImp < 0 ? "good" : "bad",
    },
    {
      label: "DM significant (p<.1)",
      val: `${sigCount}`,
      sub: `of ${total} (asset × horizon) cells`,
    },
    {
      label: "Total test steps",
      val: totalN.toLocaleString(),
      sub: "≈ 6.75y × 4 assets, daily",
    },
  ];

  const host = document.getElementById("hero-stats");
  host.innerHTML = stats
    .map(
      (s) => `
      <div class="stat">
        <div class="label">${s.label}</div>
        <div class="val ${s.cls || ""}">${s.val}</div>
        <div class="sub">${s.sub}</div>
      </div>`,
    )
    .join("");
}

// =====================================================
// 3. Render: method grid
// =====================================================
function renderMethodGrid() {
  const all = [
    ...DATA.baseline_methods,
    ...DATA.wgeo_methods,
    ...DATA.extended_methods,
  ];
  const host = document.getElementById("method-grid");
  host.innerHTML = all
    .map((m) => {
      const [fam, desc] = METHOD_DESC[m] || ["baseline", ""];
      const famLabel = fam === "wgeo" ? "WGeo family" : fam === "ext" ? "Extended" : "Baseline";
      return `
        <div class="m-card fam-${fam}">
          <div class="head">
            <div class="name">${m}</div>
            <div class="fam">${famLabel}</div>
          </div>
          <div class="desc">${desc}</div>
        </div>`;
    })
    .join("");
}

// =====================================================
// 4. Render: headline table
// =====================================================
function renderHeadlineTable() {
  const head = DATA.provenance?.headline || [];
  const body = document.querySelector("#headline-table tbody");
  body.innerHTML = head
    .map((h) => {
      const imp = parseFloat(h.improvement);
      const sign = imp < 0 ? "good" : imp > 0 ? "bad" : "";
      const sig = Number.isFinite(h.dm_p) && h.dm_p < 0.1 ? "sig-strong" : "sig-weak";
      return `
        <tr>
          <td><span class="mono">${h.symbol}</span></td>
          <td class="num">${h.h}</td>
          <td class="num">${(h.n_test || 0).toLocaleString()}</td>
          <td><span class="tag wgeo">${h.best_wgeo}</span></td>
          <td class="num">${fmtNum(h.wgeo_crps)}</td>
          <td><span class="tag base">${h.best_baseline}</span></td>
          <td class="num">${fmtNum(h.baseline_crps)}</td>
          <td class="num delta ${sign}">${h.improvement}</td>
          <td class="num">${Number.isFinite(h.dm_stat) ? h.dm_stat.toFixed(2) : "—"}</td>
          <td class="num ${sig}">${fmtP(h.dm_p)}</td>
        </tr>`;
    })
    .join("");
}

// =====================================================
// 5. Cumulative CRPS chart
// =====================================================
const cumState = {
  symbol: null,
  h: null,
  activeMethods: new Set(),
  view: "cum", // "cum" or "delta"
};

function initCumulativeControls() {
  const symSeg = document.getElementById("cum-symbol-seg");
  symSeg.innerHTML = DATA.symbols
    .map((s, i) => `<button data-s="${s}" class="${i === 0 ? "active" : ""}">${s.replace("/USDT", "")}</button>`)
    .join("");
  cumState.symbol = DATA.symbols[0];

  const hSeg = document.getElementById("cum-h-seg");
  hSeg.innerHTML = DATA.horizons
    .map((h, i) => `<button data-h="${h}" class="${i === 0 ? "active" : ""}">h = ${h}</button>`)
    .join("");
  cumState.h = DATA.horizons[0];

  symSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    symSeg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    cumState.symbol = b.dataset.s;
    populateCumMethodChips();
    drawCumulative();
    drawMethodBar();
    drawDMChart();
  });

  hSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    hSeg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    cumState.h = parseInt(b.dataset.h, 10);
    populateCumMethodChips();
    drawCumulative();
    drawMethodBar();
    drawDMChart();
  });

  const viewSeg = document.getElementById("cum-view-seg");
  viewSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    viewSeg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    cumState.view = b.dataset.v;
    drawCumulative();
  });
}

function populateCumMethodChips() {
  const section = DATA.long[cumState.symbol]?.[String(cumState.h)];
  const chipsHost = document.getElementById("cum-methods");
  if (!section) {
    chipsHost.innerHTML = "";
    return;
  }
  const allMethods = Object.keys(section.methods);
  // default activation: all WGeo + Static + GARCH-N
  cumState.activeMethods = new Set(
    allMethods.filter(
      (m) => FAMILY_BY_METHOD.get(m) === "wgeo" || m === "Static" || m === "GARCH-N",
    ),
  );

  chipsHost.innerHTML = allMethods
    .map((m) => {
      const fam = FAMILY_BY_METHOD.get(m) || "baseline";
      const active = cumState.activeMethods.has(m) ? "active" : "";
      const c = colorFor(m);
      return `<button class="chip-toggle fam-${fam} ${active}" style="--c:${c}" data-m="${m}">
        <span class="dot"></span>${m}
      </button>`;
    })
    .join("");

  chipsHost.querySelectorAll(".chip-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const m = btn.dataset.m;
      if (cumState.activeMethods.has(m)) {
        cumState.activeMethods.delete(m);
        btn.classList.remove("active");
      } else {
        cumState.activeMethods.add(m);
        btn.classList.add("active");
      }
      drawCumulative();
    });
  });
}

function drawCumulative() {
  const theme = baseChartTheme();
  const inst = getChart("cum-chart");
  if (!inst) return;
  const section = DATA.long[cumState.symbol]?.[String(cumState.h)];
  if (!section) {
    inst.setOption({ title: { text: "no data" } }, true);
    return;
  }
  const dates = section.dates_ds || section.methods[Object.keys(section.methods)[0]].cum_ds.map((_, i) => i);
  const activeMethods = [...cumState.activeMethods].filter((m) => section.methods[m]);
  // for delta mode we plot (cum_method - cum_static)
  const staticCum = section.methods["Static"]?.cum_ds || [];

  const isDelta = cumState.view === "delta" && staticCum.length;
  const series = activeMethods.map((m) => {
    const arr = section.methods[m].cum_ds;
    let data;
    let dashed = false;
    if (isDelta) {
      if (m === "Static") {
        // baseline plotted as flat zero, dashed
        data = arr.map(() => 0);
        dashed = true;
      } else if (staticCum.length === arr.length) {
        data = arr.map((v, i) => v - staticCum[i]);
      } else {
        data = arr;
      }
    } else {
      data = arr;
    }
    return {
      name: m,
      type: "line",
      showSymbol: false,
      smooth: false,
      lineStyle: {
        width: 1.6,
        color: colorFor(m),
        type: dashed ? "dashed" : "solid",
      },
      itemStyle: { color: colorFor(m) },
      data,
      emphasis: { lineStyle: { width: 2.4 } },
    };
  });

  inst.setOption(
    {
      animation: false,
      grid: { left: 56, right: 18, top: 32, bottom: 56 },
      legend: { show: false },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontSize: 12, fontFamily: cssVar("--font-mono") },
        valueFormatter: (v) => (typeof v === "number" ? v.toFixed(4) : v),
        axisPointer: { lineStyle: { color: theme.mute, type: "dashed" } },
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, hideOverlap: true, showMaxLabel: true },
        boundaryGap: false,
      },
      yAxis: {
        type: "value",
        name: cumState.view === "delta" ? "cum CRPS − cum CRPS(Static)" : "Cumulative CRPS",
        nameLocation: "middle",
        nameGap: 42,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
        axisLine: theme.axisLine,
        axisLabel: theme.axisLabel,
        splitLine: theme.splitLine,
        scale: true,
      },
      dataZoom: [
        { type: "inside", start: 0, end: 100 },
        { type: "slider", height: 18, bottom: 18, borderColor: theme.tooltipBorder, fillerColor: "rgba(124,242,214,0.08)", textStyle: { color: theme.dim, fontSize: 10 } },
      ],
      series,
    },
    true,
  );
}

// =====================================================
// 6. Per-method bar chart
// =====================================================
function drawMethodBar() {
  const theme = baseChartTheme();
  const inst = getChart("bar-chart");
  if (!inst) return;
  const section = DATA.long[cumState.symbol]?.[String(cumState.h)];
  if (!section) return;
  const methods = [...DATA.baseline_methods, ...DATA.wgeo_methods].filter((m) => section.methods[m]);
  const data = methods.map((m) => ({
    value: section.methods[m].mean,
    itemStyle: { color: colorFor(m) },
  }));
  inst.setOption(
    {
      animation: false,
      grid: { left: 124, right: 24, top: 18, bottom: 40 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        valueFormatter: (v) => v.toFixed(5),
      },
      xAxis: {
        type: "value",
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, formatter: (v) => v.toFixed(3) },
        splitLine: theme.splitLine,
        name: "mean CRPS",
        nameLocation: "middle",
        nameGap: 26,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: methods,
        axisLine: theme.axisLine,
        axisLabel: { color: theme.dim, fontFamily: cssVar("--font-mono"), fontSize: 11 },
      },
      series: [
        {
          type: "bar",
          data,
          barWidth: 14,
          label: { show: true, position: "right", color: theme.mute, fontFamily: cssVar("--font-mono"), fontSize: 10, formatter: (p) => p.value.toFixed(4) },
        },
      ],
    },
    true,
  );
}

// =====================================================
// 7. DM significance chart
// =====================================================
function drawDMChart() {
  const theme = baseChartTheme();
  const inst = getChart("dm-chart");
  if (!inst) return;
  const section = DATA.long[cumState.symbol]?.[String(cumState.h)];
  if (!section) return;
  const wgeo = DATA.wgeo_methods.filter((m) => section.methods[m]?.dm_vs_static);
  const stats = wgeo.map((m) => section.methods[m].dm_vs_static.stat);
  inst.setOption(
    {
      animation: false,
      grid: { left: 124, right: 24, top: 18, bottom: 40 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        formatter: (params) => {
          const p = params[0];
          const m = wgeo[p.dataIndex];
          const meta = section.methods[m].dm_vs_static;
          return `<b>${m}</b><br/>DM stat: ${meta.stat.toFixed(3)}<br/>p-value: ${fmtP(meta.p)}`;
        },
      },
      xAxis: {
        type: "value",
        axisLine: theme.axisLine,
        axisLabel: theme.axisLabel,
        splitLine: theme.splitLine,
        name: "DM statistic (←  WGeo better)",
        nameLocation: "middle",
        nameGap: 26,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: wgeo,
        axisLine: theme.axisLine,
        axisLabel: { color: theme.dim, fontFamily: cssVar("--font-mono"), fontSize: 11 },
      },
      series: [
        {
          type: "bar",
          data: stats.map((v) => ({
            value: v,
            itemStyle: { color: v < 0 ? cssVar("--good") : cssVar("--bad") },
          })),
          barWidth: 14,
          markLine: {
            symbol: "none",
            label: { show: false },
            lineStyle: { color: theme.mute, type: "dashed", width: 1 },
            data: [{ xAxis: 0 }, { xAxis: -1.645 }, { xAxis: 1.645 }],
          },
        },
      ],
    },
    true,
  );
}

// =====================================================
// 8. Extended panel
// =====================================================
const extState = { h: 1 };

function initExtControls() {
  const seg = document.getElementById("ext-h-seg");
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    seg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    extState.h = parseInt(b.dataset.h, 10);
    drawExtended();
  });
}

function drawExtended() {
  const theme = baseChartTheme();
  const section = DATA.extended?.[String(extState.h)];
  const bar = getChart("ext-bar");
  const cum = getChart("ext-cum");
  if (!section || !bar || !cum) return;

  const methods = ["WGeo-GARCH-Ens", "GARCH-t", ...DATA.extended_methods].filter(
    (m) => section.methods[m],
  );

  // bar — mean CRPS
  bar.setOption(
    {
      animation: false,
      grid: { left: 156, right: 24, top: 18, bottom: 40 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        valueFormatter: (v) => v.toFixed(5),
      },
      xAxis: {
        type: "value",
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, formatter: (v) => v.toFixed(3) },
        splitLine: theme.splitLine,
        name: "mean CRPS",
        nameLocation: "middle",
        nameGap: 26,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: methods,
        axisLine: theme.axisLine,
        axisLabel: { color: theme.dim, fontFamily: cssVar("--font-mono"), fontSize: 11 },
      },
      series: [
        {
          type: "bar",
          barWidth: 14,
          data: methods.map((m) => ({
            value: section.methods[m].mean,
            itemStyle: { color: colorFor(m) },
          })),
          label: { show: true, position: "right", color: theme.mute, fontFamily: cssVar("--font-mono"), fontSize: 10, formatter: (p) => p.value.toFixed(4) },
        },
      ],
    },
    true,
  );

  // cumulative curve
  const dates = section.dates_ds;
  const series = methods.map((m) => ({
    name: m,
    type: "line",
    showSymbol: false,
    lineStyle: { width: 1.6, color: colorFor(m) },
    itemStyle: { color: colorFor(m) },
    data: section.methods[m].cum_ds,
    emphasis: { lineStyle: { width: 2.4 } },
  }));
  cum.setOption(
    {
      animation: false,
      grid: { left: 56, right: 18, top: 24, bottom: 56 },
      legend: { show: false },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        valueFormatter: (v) => v.toFixed(4),
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, hideOverlap: true, showMaxLabel: true },
        boundaryGap: false,
      },
      yAxis: {
        type: "value",
        name: "Cumulative CRPS",
        nameLocation: "middle",
        nameGap: 42,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
        axisLine: theme.axisLine,
        axisLabel: theme.axisLabel,
        splitLine: theme.splitLine,
        scale: true,
      },
      dataZoom: [
        { type: "inside", start: 0, end: 100 },
        { type: "slider", height: 16, bottom: 18, borderColor: theme.tooltipBorder, fillerColor: "rgba(242,201,124,0.10)", textStyle: { color: theme.dim, fontSize: 10 } },
      ],
      series,
    },
    true,
  );
}

// =====================================================
// 9. Sweep heatmap
// =====================================================
function drawSweep() {
  const theme = baseChartTheme();
  const rows = DATA.sweep;
  if (!rows || !rows.length) return;
  const windows = [...new Set(rows.map((r) => r.window))].sort((a, b) => a - b);
  const lookbacks = [...new Set(rows.map((r) => r.lookback))].sort((a, b) => a - b);

  function build(key) {
    const data = [];
    for (const r of rows) {
      const xi = windows.indexOf(r.window);
      const yi = lookbacks.indexOf(r.lookback);
      data.push([xi, yi, r[key]]);
    }
    const vals = rows.map((r) => r[key]);
    return { data, min: Math.min(...vals), max: Math.max(...vals) };
  }

  for (const [id, key] of [
    ["sweep-early", "crps_early"],
    ["sweep-late", "crps_late"],
  ]) {
    const inst = getChart(id);
    const { data, min, max } = build(key);
    inst.setOption(
      {
        animation: false,
        grid: { left: 70, right: 84, top: 16, bottom: 60 },
        tooltip: {
          backgroundColor: theme.tooltipBg,
          borderColor: theme.tooltipBorder,
          textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
          formatter: (p) =>
            `window=${windows[p.data[0]]}  lookback=${lookbacks[p.data[1]]}<br/>CRPS = ${p.data[2].toFixed(5)}`,
        },
        xAxis: {
          type: "category",
          data: windows,
          name: "window (days)",
          nameLocation: "middle",
          nameGap: 32,
          nameTextStyle: { color: theme.mute, fontSize: 11 },
          axisLine: theme.axisLine,
          axisLabel: { color: theme.dim, fontFamily: cssVar("--font-mono"), fontSize: 11 },
          splitArea: { show: true, areaStyle: { color: ["transparent"] } },
        },
        yAxis: {
          type: "category",
          data: lookbacks,
          name: "lookback",
          nameLocation: "middle",
          nameGap: 44,
          nameTextStyle: { color: theme.mute, fontSize: 11 },
          axisLine: theme.axisLine,
          axisLabel: { color: theme.dim, fontFamily: cssVar("--font-mono"), fontSize: 11 },
        },
        visualMap: {
          min,
          max,
          calculable: true,
          orient: "vertical",
          right: 6,
          top: "center",
          textStyle: { color: theme.mute, fontFamily: cssVar("--font-mono"), fontSize: 10 },
          inRange: { color: ["#7cf2d6", "#5d8dd6", "#7c83f2", "#a366f2", "#f26a8d"] },
        },
        series: [
          {
            type: "heatmap",
            data,
            label: {
              show: true,
              formatter: (p) => p.data[2].toFixed(4),
              color: "#0b0d12",
              fontFamily: cssVar("--font-mono"),
              fontSize: 10,
            },
            itemStyle: { borderWidth: 1, borderColor: cssVar("--bg") },
          },
        ],
      },
      true,
    );
  }
}

// =====================================================
// 10. Market price + returns
// =====================================================
const mktState = { symbol: null };

function initMktControls() {
  const seg = document.getElementById("mkt-symbol-seg");
  const symbols = Object.keys(DATA.prices);
  seg.innerHTML = symbols
    .map((s, i) => `<button data-s="${s}" class="${i === 0 ? "active" : ""}">${s.replace("/USDT", "")}</button>`)
    .join("");
  mktState.symbol = symbols[0];
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    seg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    mktState.symbol = b.dataset.s;
    drawMarket();
  });
}

function drawMarket() {
  const theme = baseChartTheme();
  const price = getChart("price-chart");
  const ret = getChart("return-chart");
  const candles = DATA.prices[mktState.symbol] || [];
  const returns = DATA.returns[mktState.symbol];

  const dates = candles.map((c) => c.t);
  // ECharts candlestick expects [open, close, low, high]
  const ohlc = candles.map((c) => [c.o, c.c, c.l, c.h]);

  price.setOption(
    {
      animation: false,
      grid: { left: 64, right: 20, top: 18, bottom: 58 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, hideOverlap: true },
        boundaryGap: true,
      },
      yAxis: {
        type: "value",
        scale: true,
        name: `${mktState.symbol}  ·  USD`,
        nameLocation: "middle",
        nameGap: 52,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, formatter: (v) => v.toLocaleString() },
        splitLine: theme.splitLine,
      },
      dataZoom: [
        { type: "inside", start: 0, end: 100 },
        { type: "slider", height: 18, bottom: 18, borderColor: theme.tooltipBorder, fillerColor: "rgba(124,242,214,0.08)", textStyle: { color: theme.dim, fontSize: 10 } },
      ],
      series: [
        {
          type: "candlestick",
          data: ohlc,
          // omit barWidth so ECharts auto-sizes to fill the slot for the current zoom
          itemStyle: {
            color: cssVar("--good"),
            color0: cssVar("--bad"),
            borderColor: cssVar("--good"),
            borderColor0: cssVar("--bad"),
          },
        },
      ],
    },
    true,
  );

  if (!returns) {
    ret.clear();
    return;
  }
  ret.setOption(
    {
      animation: false,
      grid: { left: 64, right: 20, top: 8, bottom: 26 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        valueFormatter: (v) => (typeof v === "number" ? v.toFixed(4) : v),
      },
      xAxis: {
        type: "category",
        data: returns.dates,
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, hideOverlap: true },
        boundaryGap: false,
      },
      yAxis: {
        type: "value",
        name: "log-return",
        nameLocation: "middle",
        nameGap: 48,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
        axisLine: theme.axisLine,
        axisLabel: theme.axisLabel,
        splitLine: theme.splitLine,
      },
      series: [
        {
          type: "bar",
          data: returns.r.map((v) => ({
            value: v,
            itemStyle: { color: v >= 0 ? cssVar("--good") : cssVar("--bad"), opacity: 0.65 },
          })),
          barWidth: 1,
        },
      ],
    },
    true,
  );
}

// =====================================================
// 11. Provenance
// =====================================================
function renderProvenance() {
  const prov = DATA.provenance;
  if (!prov) return;
  const run = prov.latest_run || {};
  const pkg = run.packages || {};
  const data = run.data_sha256 || {};
  const runHost = document.getElementById("prov-run");
  runHost.innerHTML = `
    <h3>Latest run</h3>
    <dl class="kv">
      <dt>Entry</dt><dd>${run.entry_point || "—"}</dd>
      <dt>Timestamp</dt><dd>${run.timestamp || "—"}</dd>
      <dt>Git SHA</dt><dd>${(run.git_sha || "—").slice(0, 12)}${run.git_dirty ? " <span class='muted'>(dirty)</span>" : ""}</dd>
      <dt>Python</dt><dd>${run.python || "—"}</dd>
      <dt>wbtc</dt><dd>${pkg.wbtc || "—"}</dd>
      <dt>numpy / scipy</dt><dd>${pkg.numpy || "—"} · ${pkg.scipy || "—"}</dd>
      <dt>pandas / arch</dt><dd>${pkg.pandas || "—"} · ${pkg.arch || "—"}</dd>
      <dt>matplotlib</dt><dd>${pkg.matplotlib || "—"}</dd>
    </dl>`;

  const dataHost = document.getElementById("prov-data");
  const rows = Object.entries(data)
    .map(([k, v]) => `<dt>${k}</dt><dd>${v.slice(0, 24)}…</dd>`)
    .join("");
  dataHost.innerHTML = `
    <h3>Data SHA-256</h3>
    <dl class="kv">${rows || "<dt>—</dt><dd>no data hashes</dd>"}</dl>`;
}

// =====================================================
// Theme toggle
// =====================================================
function initThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  const root = document.documentElement;
  btn.addEventListener("click", () => {
    const cur = root.getAttribute("data-theme") || "dark";
    root.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
    // re-draw all charts to pick up new theme tokens
    drawCumulative();
    drawMethodBar();
    drawDMChart();
    drawExtended();
    drawSweep();
    drawMarket();
  });
}

// =====================================================
// Boot
// =====================================================
(async function main() {
  try {
    DATA = await load();
  } catch (err) {
    document.getElementById("loading").innerHTML = `<div style="color:var(--bad)">Could not load data.json — run <code>uv run python viewer/build_data.py</code> first.</div>`;
    console.error(err);
    return;
  }

  renderHero();
  renderMethodGrid();
  renderHeadlineTable();
  renderProvenance();

  initCumulativeControls();
  populateCumMethodChips();
  drawCumulative();
  drawMethodBar();
  drawDMChart();

  initExtControls();
  drawExtended();

  drawSweep();

  initMktControls();
  drawMarket();

  initThemeToggle();

  document.getElementById("loading").classList.add("hidden");
})();
