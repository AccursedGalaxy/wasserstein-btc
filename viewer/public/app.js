// wbtc viewer — vanilla ES module, ECharts for visualisations.

// 8 distinct hues per family so position-indexed lookup never collides across
// methods. The viewer has 8 WGeo + 6 baseline + 6 extended methods.
const PALETTE_WGEO = [
  "#7cf2d6", "#5dd6a8", "#3fc2a5", "#7adbb0",
  "#a5f5b8", "#36b08e", "#5fe5c6", "#84e0c8",
];
const PALETTE_BASELINE = [
  "#7c83f2", "#9aa0f5", "#6068d8", "#aab0fa",
  "#8088ee", "#5560cd",
];
const PALETTE_EXT = [
  "#f2c97c", "#f0a64a", "#e8d088", "#d8a04a",
  "#f5d99a", "#d68f3a",
];
// Dash patterns rotated by within-family index — gives ~8 visually distinct
// strokes per family so a line picked out of a tooltip can be located on the
// chart by sight, even when the colors are close.
const DASH_PATTERNS = [
  "solid",
  [6, 4],
  [2, 3],
  [10, 4, 2, 4],
  [4, 2, 2, 2],
  [12, 4],
  [1, 3],
  [8, 2, 2, 2, 2, 2],
];

// Stable position of each method inside its family — used for deterministic
// color and dash assignment. Populated below alongside METHOD_DESC.
const POS_IN_FAMILY = new Map();

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
  "WGeo-Adaptive": ["wgeo", "v0.4 — recency-weighted empirical base quantile (EW with λ_q≈0.97); the forecast tracks the current vol regime through the *shape* axis, not just the slope."],
  "WGeo-Ensemble": ["wgeo", "v0.4 headline — W₂ barycentre (equal-weight quantile-space mean) of TheilSen + EWMA + Gated. Convex CRPS makes the ensemble weakly dominate the component average."],
  // extended comparators
  "HAR-RV": ["ext", "Heterogeneous Autoregressive model on realized volatility (Corsi 2009)."],
  "CAViaR-SAV": ["ext", "Conditional Autoregressive Value-at-Risk, symmetric-absolute-value (Engle-Manganelli 2004)."],
  "MS-Normal-2": ["ext", "Two-state Markov-switching Normal (Hamilton 1989)."],
  "FIGARCH(1,d,0)": ["ext", "Fractionally integrated GARCH with long-memory (Baillie-Bollerslev-Mikkelsen 1996)."],
  "SV-AR1": ["ext", "AR(1) stochastic-volatility model (Taylor 1982 / Harvey-Ruiz-Shephard 1994), Kalman-QML."],
  "BVAR-GARCH(BTC,ETH)": ["ext", "Bivariate VAR + GARCH using BTC and ETH jointly."],
};

{
  // Walk METHOD_DESC in declaration order: that's the canonical ordering
  // every chart sorts methods by, and the index within a family becomes the
  // (color, dash) slot. Two methods in the same family never collide.
  const counters = { baseline: 0, wgeo: 0, ext: 0 };
  for (const [name, [fam]] of Object.entries(METHOD_DESC)) {
    FAMILY_BY_METHOD.set(name, fam);
    POS_IN_FAMILY.set(name, counters[fam] ?? 0);
    counters[fam] = (counters[fam] ?? 0) + 1;
  }
}

function paletteFor(fam) {
  return fam === "wgeo" ? PALETTE_WGEO : fam === "ext" ? PALETTE_EXT : PALETTE_BASELINE;
}

function colorFor(name) {
  const fam = FAMILY_BY_METHOD.get(name) || "baseline";
  const list = paletteFor(fam);
  const idx = POS_IN_FAMILY.get(name) ?? 0;
  return list[idx % list.length];
}

// ECharts accepts lineStyle.type = "solid" | "dashed" | "dotted" | number[]
function dashFor(name) {
  const idx = POS_IN_FAMILY.get(name) ?? 0;
  return DASH_PATTERNS[idx % DASH_PATTERNS.length];
}

// Family label shown in tooltips
function famLabelFor(name) {
  const fam = FAMILY_BY_METHOD.get(name);
  return fam === "wgeo" ? "WGeo" : fam === "ext" ? "Extended" : "Baseline";
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
function headlineRows() {
  // Prefer the freshly-computed headline (v0.4: includes residualised DM)
  // and fall back to the MANIFEST snapshot only if build_data.py couldn't
  // derive one (e.g., missing parquet).
  if (Array.isArray(DATA.headline) && DATA.headline.length) return DATA.headline;
  return DATA.provenance?.headline || [];
}

function renderHero() {
  document.getElementById("version-chip").textContent = `v${DATA.version}`;
  const head = headlineRows();
  const wins = head.filter((h) => parseFloat(h.improvement) < 0).length;
  const total = head.length;
  const meanImp = head.length
    ? head.reduce((s, h) => s + parseFloat(h.improvement), 0) / head.length
    : 0;
  // Use residualised DM when available (the v0.4 headline test).
  const pKey = head.some((h) => Number.isFinite(h.dm_p_r)) ? "dm_p_r" : "dm_p";
  const sigCount = head.filter(
    (h) => Number.isFinite(h[pKey]) && h[pKey] < 0.05,
  ).length;
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
      label:
        pKey === "dm_p_r"
          ? "DM significant (p_r < .05)"
          : "DM significant (p < .05)",
      val: `${sigCount}`,
      sub:
        pKey === "dm_p_r"
          ? `of ${total} cells · residualised DM`
          : `of ${total} (asset × horizon) cells`,
      cls: sigCount * 2 >= total ? "good" : "",
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
  const head = headlineRows();
  const body = document.querySelector("#headline-table tbody");
  body.innerHTML = head
    .map((h) => {
      const imp = parseFloat(h.improvement);
      const sign = imp < 0 ? "good" : imp > 0 ? "bad" : "";
      const sig = Number.isFinite(h.dm_p) && h.dm_p < 0.05 ? "sig-strong" : "sig-weak";
      const hasResid = Number.isFinite(h.dm_p_r);
      const sigR = hasResid && h.dm_p_r < 0.05 ? "sig-strong" : "sig-weak";
      const residCells = hasResid
        ? `
          <td class="num">${h.dm_stat_r.toFixed(2)}</td>
          <td class="num ${sigR}">${fmtP(h.dm_p_r)}</td>`
        : `
          <td class="num muted">—</td>
          <td class="num muted">—</td>`;
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
          <td class="num ${sig}">${fmtP(h.dm_p)}</td>${residCells}
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

const CUM_PRESETS = {
  headline: (allMethods) =>
    allMethods.filter(
      (m) => FAMILY_BY_METHOD.get(m) === "wgeo" || m === "Static" || m === "GARCH-N",
    ),
  wgeo: (allMethods) => allMethods.filter((m) => FAMILY_BY_METHOD.get(m) === "wgeo"),
  baselines: (allMethods) =>
    allMethods.filter((m) => FAMILY_BY_METHOD.get(m) === "baseline"),
  all: (allMethods) => allMethods.slice(),
  clear: () => [],
};

function applyCumPreset(name, allMethods) {
  const fn = CUM_PRESETS[name] || CUM_PRESETS.headline;
  cumState.activeMethods = new Set(fn(allMethods));
  // sync chip visuals
  document.querySelectorAll("#cum-methods .chip-toggle").forEach((btn) => {
    btn.classList.toggle("active", cumState.activeMethods.has(btn.dataset.m));
  });
  document.querySelectorAll("#cum-method-presets .preset").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.p === name);
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
  // default activation: headline preset (Static + GARCH-N + all WGeo)
  cumState.activeMethods = new Set(CUM_PRESETS.headline(allMethods));

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
      // chip click drops the preset highlight — selection is now custom
      document.querySelectorAll("#cum-method-presets .preset").forEach((b) => {
        b.classList.toggle("active", b.dataset.p === "clear" ? false : false);
      });
      drawCumulative();
    });
  });

  // Wire up the preset row (idempotent — safe to call on every populate)
  const presets = document.getElementById("cum-method-presets");
  if (presets && !presets.dataset.wired) {
    presets.addEventListener("click", (e) => {
      const b = e.target.closest("button.preset");
      if (!b) return;
      const section2 = DATA.long[cumState.symbol]?.[String(cumState.h)];
      if (!section2) return;
      applyCumPreset(b.dataset.p, Object.keys(section2.methods));
      drawCumulative();
    });
    presets.dataset.wired = "1";
  }
}

// Build a tooltip formatter that sorts series by current value and ranks them
// best→worst. "Best" depends on mode: in absolute / cum CRPS view, lower is
// better. In delta-vs-Static view, more-negative is better (further below 0).
function makeCumTooltipFormatter({ isDelta, dates, valueFmt = 4 }) {
  return (params) => {
    if (!params || !params.length) return "";
    const x = params[0].axisValue;
    const dateLabel = typeof x === "string" ? x : (dates ? dates[x] : x);
    // ECharts puts the params in series-iteration order; sort by value asc
    // (lower-is-better in both modes).
    const sorted = [...params].sort((a, b) => (a.value ?? Infinity) - (b.value ?? Infinity));
    const rows = sorted.map((p, i) => {
      const v = typeof p.value === "number" ? p.value.toFixed(valueFmt) : p.value;
      const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:999px;background:${p.color};margin-right:6px;vertical-align:middle"></span>`;
      const rank = `<span style="opacity:0.55;width:18px;display:inline-block;text-align:right">${i + 1}.</span>`;
      const fam = `<span style="opacity:0.5;font-size:10px;margin-left:6px">${famLabelFor(p.seriesName)}</span>`;
      return `<div style="display:flex;justify-content:space-between;gap:14px;line-height:1.55">
        <span>${rank} ${dot}${p.seriesName}${fam}</span>
        <span style="font-variant-numeric:tabular-nums">${v}</span>
      </div>`;
    });
    const hint = isDelta
      ? `<div style="opacity:0.6;font-size:10px;margin-bottom:4px">Δ vs Static · negative = WGeo better</div>`
      : `<div style="opacity:0.6;font-size:10px;margin-bottom:4px">cumulative CRPS · lower = better</div>`;
    return `<div style="font-weight:600;margin-bottom:4px">${dateLabel}</div>${hint}${rows.join("")}`;
  };
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
  const firstMethodName = Object.keys(section.methods)[0];
  const fallbackDates = section.methods[firstMethodName].cum_ds.map((_, i) => i);
  const dates = section.dates_ds || fallbackDates;
  const activeMethods = [...cumState.activeMethods].filter((m) => section.methods[m]);
  // for delta mode we plot (cum_method - cum_static)
  const staticCum = section.methods["Static"]?.cum_ds || [];
  const isDelta = cumState.view === "delta" && staticCum.length > 0;

  // Build per-method series. In delta mode, Static is plotted as a flat dashed
  // zero so it sits visibly on the reference line. To rank lines at the right
  // edge (for endLabels) we compute each method's final-step value.
  const finals = new Map();
  const series = activeMethods.map((m) => {
    const arr = section.methods[m].cum_ds;
    let data;
    let dashed = false;
    if (isDelta) {
      if (m === "Static") {
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
    finals.set(m, data[data.length - 1] ?? 0);
    const dashSlot = dashFor(m);
    return {
      name: m,
      type: "line",
      showSymbol: false,
      smooth: false,
      lineStyle: {
        width: 1.6,
        color: colorFor(m),
        type: dashed ? "dashed" : dashSlot,
      },
      itemStyle: { color: colorFor(m) },
      data,
      emphasis: { focus: "series", lineStyle: { width: 2.8 } },
      // dim non-hovered lines so picked series pops
      blur: { lineStyle: { opacity: 0.18 } },
    };
  });

  // Mark the winner (lowest final value) and worst (highest) with an endLabel
  // so the user can locate them on the chart without hovering.
  if (activeMethods.length >= 2) {
    const ranked = [...finals.entries()].sort((a, b) => a[1] - b[1]);
    const best = ranked[0]?.[0];
    const worst = ranked[ranked.length - 1]?.[0];
    for (const s of series) {
      if (s.name === best) {
        s.endLabel = {
          show: true,
          formatter: `▼ ${s.name}`,
          color: colorFor(s.name),
          fontFamily: cssVar("--font-mono"),
          fontSize: 10,
          distance: 8,
          padding: [1, 4],
        };
      } else if (s.name === worst) {
        s.endLabel = {
          show: true,
          formatter: `▲ ${s.name}`,
          color: colorFor(s.name),
          fontFamily: cssVar("--font-mono"),
          fontSize: 10,
          distance: 8,
          padding: [1, 4],
        };
      }
    }
  }

  // In delta mode: shade above (red) and below (green) the zero line, and put
  // a labeled "Static" reference at y = 0. This is the single biggest
  // interpretability win — readers see "below = WGeo wins" without thinking.
  const goodC = cssVar("--good");
  const badC = cssVar("--bad");
  const refSeries = isDelta
    ? [
        {
          type: "line",
          name: "__ref",
          data: [],
          silent: true,
          tooltip: { show: false },
          markArea: {
            silent: true,
            label: { show: false },
            data: [
              [{ yAxis: 0, itemStyle: { color: `${badC}10` } }, { yAxis: "max" }],
              [{ yAxis: "min", itemStyle: { color: `${goodC}14` } }, { yAxis: 0 }],
            ],
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: theme.mute, type: "dashed", width: 1 },
            label: {
              formatter: "Static",
              color: theme.mute,
              fontFamily: cssVar("--font-mono"),
              fontSize: 10,
              position: "insideEndTop",
            },
            data: [{ yAxis: 0 }],
          },
        },
      ]
    : [];

  // Corner overlays in delta mode — predictable positioning unlike markArea labels.
  const graphicOverlays = isDelta
    ? [
        {
          type: "text",
          left: 64,
          top: 40,
          silent: true,
          style: {
            text: "WGeo worse than Static ▲",
            fill: badC,
            opacity: 0.95,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
          },
        },
        {
          type: "text",
          left: 64,
          bottom: 50,
          silent: true,
          style: {
            text: "WGeo better than Static ▼",
            fill: goodC,
            opacity: 0.95,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
          },
        },
      ]
    : [];

  inst.setOption(
    {
      animation: false,
      graphic: graphicOverlays,
      grid: { left: 56, right: 110, top: 32, bottom: 60 },
      legend: { show: false },
      tooltip: {
        trigger: "axis",
        confine: true,
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontSize: 12, fontFamily: cssVar("--font-mono") },
        axisPointer: { lineStyle: { color: theme.mute, type: "dashed" } },
        formatter: makeCumTooltipFormatter({ isDelta, dates }),
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
        name: isDelta ? "Δ cum CRPS (vs Static)" : "Cumulative CRPS",
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
      series: [...series, ...refSeries],
    },
    true,
  );
}

// =====================================================
// 6. Per-method bar chart
// =====================================================
const barState = { view: "abs" }; // "abs" or "delta"

function initBarControls() {
  const seg = document.getElementById("bar-view-seg");
  if (!seg || seg.dataset.wired) return;
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    seg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    barState.view = b.dataset.v;
    drawMethodBar();
  });
  seg.dataset.wired = "1";
}

function drawMethodBar() {
  const theme = baseChartTheme();
  const inst = getChart("bar-chart");
  if (!inst) return;
  const section = DATA.long[cumState.symbol]?.[String(cumState.h)];
  if (!section) return;
  const methods = [...DATA.baseline_methods, ...DATA.wgeo_methods].filter(
    (m) => section.methods[m],
  );
  const isDelta = barState.view === "delta" && section.methods["Static"];
  const staticMean = section.methods["Static"]?.mean ?? 0;

  // Compute values per method (absolute mean or pct-Δ vs Static).
  const valueOf = (m) => {
    const mu = section.methods[m].mean;
    if (!isDelta) return mu;
    return staticMean ? ((mu - staticMean) / staticMean) * 100 : 0;
  };

  // Sort ascending — best (lowest) sits at the top of a category axis (yAxis
  // categories render bottom→top; inverting via `data` order puts best at top).
  const sorted = [...methods].sort((a, b) => valueOf(a) - valueOf(b));
  // Reverse so ECharts shows lowest-value at the top.
  const yCats = sorted.slice().reverse();

  const bestM = sorted[0];
  const accentGood = cssVar("--good");
  const accentBad = cssVar("--bad");
  const data = yCats.map((m) => {
    const v = valueOf(m);
    const c = colorFor(m);
    const isBest = m === bestM;
    let color = c;
    if (isDelta) {
      color = v < 0 ? accentGood : v > 0 ? accentBad : c;
    }
    return {
      value: v,
      itemStyle: {
        color,
        opacity: isBest ? 1 : 0.78,
        borderColor: isBest ? cssVar("--fg") : "transparent",
        borderWidth: isBest ? 1.2 : 0,
      },
    };
  });

  const fmtVal = (v) => (isDelta ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%` : v.toFixed(4));
  const refLine = isDelta
    ? [
        {
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: theme.mute, type: "dashed", width: 1 },
            label: { show: false },
            data: [{ xAxis: 0 }],
          },
        },
      ]
    : [];

  inst.setOption(
    {
      animation: false,
      grid: { left: 130, right: 56, top: 18, bottom: 44 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        formatter: (params) => {
          const p = params[0];
          const m = p.name;
          const mu = section.methods[m].mean;
          const dPct = staticMean ? ((mu - staticMean) / staticMean) * 100 : 0;
          const fam = famLabelFor(m);
          return `<div style="font-weight:600;margin-bottom:4px">${m} <span style="opacity:0.6;font-weight:400;font-size:11px">· ${fam}</span></div>
            <div>mean CRPS: <b>${mu.toFixed(5)}</b></div>
            <div style="opacity:0.75">Δ vs Static: ${dPct >= 0 ? "+" : ""}${dPct.toFixed(2)}%</div>`;
        },
      },
      xAxis: {
        type: "value",
        axisLine: theme.axisLine,
        axisLabel: {
          ...theme.axisLabel,
          formatter: (v) => (isDelta ? `${v}%` : v.toFixed(3)),
        },
        splitLine: theme.splitLine,
        name: isDelta ? "Δ vs Static  (← better)" : "mean CRPS  (← better)",
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: yCats,
        axisLine: theme.axisLine,
        axisLabel: {
          color: theme.dim,
          fontFamily: cssVar("--font-mono"),
          fontSize: 11,
          formatter: (m) => (m === bestM ? `★ ${m}` : m),
        },
      },
      series: [
        {
          type: "bar",
          data,
          barWidth: 13,
          label: {
            show: true,
            position: "right",
            color: theme.mute,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
            formatter: (p) => fmtVal(p.value),
          },
          ...refLine[0],
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
  // Sort WGeo variants by DM stat ascending so the most-strongly-negative
  // (most-significantly better) sits at the top of the chart.
  const wgeoSorted = DATA.wgeo_methods
    .filter((m) => section.methods[m]?.dm_vs_static)
    .slice()
    .sort(
      (a, b) =>
        section.methods[a].dm_vs_static.stat -
        section.methods[b].dm_vs_static.stat,
    );
  // ECharts category axis renders bottom→top — reverse so best is at top.
  const yCats = wgeoSorted.slice().reverse();
  const good = cssVar("--good");
  const bad = cssVar("--bad");

  const data = yCats.map((m) => {
    const meta = section.methods[m].dm_vs_static;
    const s = meta.stat;
    // significance buckets at the conventional |z| critical values
    let color = theme.mute;
    let opacity = 0.6;
    if (s <= -1.96) { color = good; opacity = 1.0; }
    else if (s <= -1.645) { color = good; opacity = 0.78; }
    else if (s >= 1.96) { color = bad; opacity = 1.0; }
    else if (s >= 1.645) { color = bad; opacity = 0.78; }
    return {
      value: s,
      itemStyle: { color, opacity, borderRadius: 2 },
    };
  });

  inst.setOption(
    {
      animation: false,
      grid: { left: 130, right: 28, top: 26, bottom: 50 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        formatter: (params) => {
          const p = params[0];
          const m = p.name;
          const meta = section.methods[m].dm_vs_static;
          const verdict =
            meta.stat <= -1.96
              ? `<span style="color:${good}">significantly better (p < .05)</span>`
              : meta.stat <= -1.645
              ? `<span style="color:${good}">marginally better (p < .10)</span>`
              : meta.stat >= 1.96
              ? `<span style="color:${bad}">significantly worse (p < .05)</span>`
              : meta.stat >= 1.645
              ? `<span style="color:${bad}">marginally worse (p < .10)</span>`
              : `<span style="opacity:0.7">tie (not significant)</span>`;
          return `<div style="font-weight:600;margin-bottom:4px">${m} <span style="opacity:0.6;font-weight:400">vs Static</span></div>
            <div>DM stat: <b>${meta.stat.toFixed(3)}</b></div>
            <div>p-value: <b>${fmtP(meta.p)}</b></div>
            <div style="margin-top:2px">${verdict}</div>`;
        },
      },
      xAxis: {
        type: "value",
        axisLine: theme.axisLine,
        axisLabel: theme.axisLabel,
        splitLine: theme.splitLine,
        name: "DM statistic  (← WGeo better · WGeo worse →)",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: yCats,
        axisLine: theme.axisLine,
        axisLabel: { color: theme.dim, fontFamily: cssVar("--font-mono"), fontSize: 11 },
      },
      series: [
        {
          type: "bar",
          data,
          barWidth: 14,
          label: {
            show: true,
            position: "right",
            color: theme.mute,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
            formatter: (p) => p.value.toFixed(2),
          },
          // Significance zones (tinted bands) + labeled critical-value lines.
          markArea: {
            silent: true,
            data: [
              [
                { xAxis: -Infinity, itemStyle: { color: `${good}10` } },
                { xAxis: -1.96 },
              ],
              [
                { xAxis: 1.96, itemStyle: { color: `${bad}10` } },
                { xAxis: Infinity },
              ],
            ],
          },
          markLine: {
            symbol: "none",
            silent: true,
            lineStyle: { color: theme.mute, type: "dashed", width: 1 },
            label: {
              show: true,
              fontFamily: cssVar("--font-mono"),
              fontSize: 9,
              color: theme.dim,
              position: "insideEndTop",
            },
            data: [
              { xAxis: 0, label: { formatter: "tie" } },
              { xAxis: -1.96, label: { formatter: "p=.05" } },
              { xAxis: 1.96, label: { formatter: "p=.05" } },
              { xAxis: -1.645, label: { formatter: "p=.10", color: theme.dim } },
              { xAxis: 1.645, label: { formatter: "p=.10", color: theme.dim } },
            ],
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
const extState = { h: 1, view: "delta" }; // "cum" | "delta" — default to delta
                                            // because absolute curves are
                                            // visually indistinguishable.

const EXT_BENCH = "WGeo-GARCH-Ens";

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
  const view = document.getElementById("ext-view-seg");
  if (view && !view.dataset.wired) {
    view.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      view.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      extState.view = b.dataset.v;
      drawExtended();
    });
    view.dataset.wired = "1";
  }
}

function drawExtended() {
  const theme = baseChartTheme();
  const section = DATA.extended?.[String(extState.h)];
  const bar = getChart("ext-bar");
  const cum = getChart("ext-cum");
  if (!section || !bar || !cum) return;

  const methods = [EXT_BENCH, "GARCH-t", ...DATA.extended_methods].filter(
    (m) => section.methods[m],
  );

  // ---- bar: sort ascending, highlight winner ----
  const sorted = methods
    .slice()
    .sort((a, b) => section.methods[a].mean - section.methods[b].mean);
  const yCats = sorted.slice().reverse();
  const bestM = sorted[0];

  bar.setOption(
    {
      animation: false,
      grid: { left: 156, right: 56, top: 18, bottom: 44 },
      tooltip: {
        trigger: "axis",
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        formatter: (params) => {
          const p = params[0];
          const m = p.name;
          const mu = section.methods[m].mean;
          const bench = section.methods[EXT_BENCH]?.mean;
          const dPct = bench ? ((mu - bench) / bench) * 100 : null;
          return `<div style="font-weight:600;margin-bottom:4px">${m} <span style="opacity:0.6;font-weight:400;font-size:11px">· ${famLabelFor(m)}</span></div>
            <div>mean CRPS: <b>${mu.toFixed(5)}</b></div>
            ${dPct === null ? "" : `<div style="opacity:0.75">Δ vs ${EXT_BENCH}: ${dPct >= 0 ? "+" : ""}${dPct.toFixed(2)}%</div>`}`;
        },
      },
      xAxis: {
        type: "value",
        axisLine: theme.axisLine,
        axisLabel: { ...theme.axisLabel, formatter: (v) => v.toFixed(3) },
        splitLine: theme.splitLine,
        name: "mean CRPS  (← better)",
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: { color: theme.mute, fontSize: 11 },
      },
      yAxis: {
        type: "category",
        data: yCats,
        axisLine: theme.axisLine,
        axisLabel: {
          color: theme.dim,
          fontFamily: cssVar("--font-mono"),
          fontSize: 11,
          formatter: (m) => (m === bestM ? `★ ${m}` : m),
        },
      },
      series: [
        {
          type: "bar",
          barWidth: 13,
          data: yCats.map((m) => {
            const isBest = m === bestM;
            return {
              value: section.methods[m].mean,
              itemStyle: {
                color: colorFor(m),
                opacity: isBest ? 1 : 0.78,
                borderColor: isBest ? cssVar("--fg") : "transparent",
                borderWidth: isBest ? 1.2 : 0,
              },
            };
          }),
          label: {
            show: true,
            position: "right",
            color: theme.mute,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
            formatter: (p) => p.value.toFixed(4),
          },
        },
      ],
    },
    true,
  );

  // ---- cumulative: absolute or Δ vs WGeo-GARCH-Ens ----
  const dates = section.dates_ds || section.methods[methods[0]].cum_ds.map((_, i) => i);
  const benchCum = section.methods[EXT_BENCH]?.cum_ds || [];
  const isDelta = extState.view === "delta" && benchCum.length > 0;

  const finals = new Map();
  const series = methods.map((m) => {
    const arr = section.methods[m].cum_ds;
    let data;
    let dashed = false;
    if (isDelta) {
      if (m === EXT_BENCH) {
        data = arr.map(() => 0);
        dashed = true;
      } else if (benchCum.length === arr.length) {
        data = arr.map((v, i) => v - benchCum[i]);
      } else {
        data = arr;
      }
    } else {
      data = arr;
    }
    finals.set(m, data[data.length - 1] ?? 0);
    return {
      name: m,
      type: "line",
      showSymbol: false,
      lineStyle: {
        width: 1.6,
        color: colorFor(m),
        type: dashed ? "dashed" : dashFor(m),
      },
      itemStyle: { color: colorFor(m) },
      data,
      emphasis: { focus: "series", lineStyle: { width: 2.8 } },
      blur: { lineStyle: { opacity: 0.2 } },
    };
  });

  // endLabels for winner / worst (by final value)
  if (methods.length >= 2) {
    const ranked = [...finals.entries()].sort((a, b) => a[1] - b[1]);
    const best = ranked[0][0];
    const worst = ranked[ranked.length - 1][0];
    for (const s of series) {
      if (s.name === best) {
        s.endLabel = {
          show: true, formatter: `▼ ${s.name}`, color: colorFor(s.name),
          fontFamily: cssVar("--font-mono"), fontSize: 10, distance: 8, padding: [1, 4],
        };
      } else if (s.name === worst) {
        s.endLabel = {
          show: true, formatter: `▲ ${s.name}`, color: colorFor(s.name),
          fontFamily: cssVar("--font-mono"), fontSize: 10, distance: 8, padding: [1, 4],
        };
      }
    }
  }

  // win/lose shading + benchmark reference (delta mode only)
  const goodC = cssVar("--good");
  const badC = cssVar("--bad");
  const refSeries = isDelta
    ? [
        {
          type: "line",
          name: "__ref",
          data: [],
          silent: true,
          tooltip: { show: false },
          markArea: {
            silent: true,
            label: { show: false },
            data: [
              [{ yAxis: 0, itemStyle: { color: `${badC}10` } }, { yAxis: "max" }],
              [{ yAxis: "min", itemStyle: { color: `${goodC}14` } }, { yAxis: 0 }],
            ],
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: theme.mute, type: "dashed", width: 1 },
            label: {
              formatter: EXT_BENCH,
              color: theme.mute,
              fontFamily: cssVar("--font-mono"),
              fontSize: 10,
              position: "insideEndTop",
            },
            data: [{ yAxis: 0 }],
          },
        },
      ]
    : [];

  const extGraphicOverlays = isDelta
    ? [
        {
          type: "text",
          left: 64,
          top: 36,
          silent: true,
          style: {
            text: `worse than ${EXT_BENCH} ▲`,
            fill: badC,
            opacity: 0.95,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
          },
        },
        {
          type: "text",
          left: 64,
          bottom: 50,
          silent: true,
          style: {
            text: `better than ${EXT_BENCH} ▼`,
            fill: goodC,
            opacity: 0.95,
            fontFamily: cssVar("--font-mono"),
            fontSize: 10,
          },
        },
      ]
    : [];

  cum.setOption(
    {
      animation: false,
      graphic: extGraphicOverlays,
      grid: { left: 56, right: 110, top: 24, bottom: 60 },
      legend: { show: false },
      tooltip: {
        trigger: "axis",
        confine: true,
        backgroundColor: theme.tooltipBg,
        borderColor: theme.tooltipBorder,
        textStyle: { color: theme.tooltipFg, fontFamily: cssVar("--font-mono"), fontSize: 12 },
        formatter: makeCumTooltipFormatter({ isDelta, dates }),
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
        name: isDelta ? `Δ cum CRPS (vs ${EXT_BENCH})` : "Cumulative CRPS",
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
      series: [...series, ...refSeries],
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
  initBarControls();
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
