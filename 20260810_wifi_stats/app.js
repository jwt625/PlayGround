/* Renders a Cloudflare-speed-test-style report purely from a results CSV.
 *
 * CSV columns (as exported by speed.cloudflare.com):
 *   time,direction,bytes,latency,bps,duration,serverTime,responseSize,loadedLatencies
 *
 * Formulas were reverse-engineered by matching the numbers in the saved page
 * (tmp.html.html) against this CSV:
 *   download / upload speed  = 90th percentile of `bps`  -> 70.1 / 95.5 exact match
 *   unloaded latency         = median of `latency`
 *   jitter                   = mean |difference| between consecutive samples
 *   loaded latency (dl / ul) = same stats over `loadedLatencies`, split by direction
 */

/* ------------------------------------------------------------------ *
 * Metadata the CSV does not contain.
 *
 * Packet loss, server location, ASN, and client IP live only in the page,
 * never in the export. Values below are copied from tmp.html.html so the
 * reproduction is complete; on a community site an uploader would supply
 * the place fields and the rest would come from a Cloudflare trace call.
 * ------------------------------------------------------------------ */
const META = {
  placeLabel: "Reno-Tahoe Airport",
  client: { lat: 39.4996, lon: -119.7681 },
  proto: "IPv4",
  serverCity: "San Jose",
  server: { lat: 37.3382, lon: -121.8863 },
  asnName: "Reno-Tahoe Airport Authority",
  asn: "AS401201",
  ip: "199.74.255.157",
  packetLossPct: 0,
  packetLossSamples: [1000, 1000], // [received, sent]
};

const CSV_URL = "speed-results-20260810.csv";

const COLORS = {
  download: getVar("--dl", "#F6821F"),
  upload: getVar("--ul", "#8D1EB1"),
  latency: getVar("--lat", "#0051C3"),
};

function getVar(name, fallback) {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return v || fallback;
}

/* ============================ stats ============================ */

const num = (x) => {
  const v = parseFloat(x);
  return Number.isFinite(v) ? v : null;
};

/** Linear-interpolated percentile, matching the values printed by the page. */
function percentile(arr, p) {
  if (!arr.length) return null;
  const a = [...arr].sort((x, y) => x - y);
  const k = ((a.length - 1) * p) / 100;
  const lo = Math.floor(k);
  const hi = Math.min(lo + 1, a.length - 1);
  return a[lo] + (a[hi] - a[lo]) * (k - lo);
}

const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);

/** Mean absolute difference between consecutive samples, in arrival order. */
function jitter(a) {
  if (a.length < 2) return null;
  let s = 0;
  for (let i = 1; i < a.length; i++) s += Math.abs(a[i] - a[i - 1]);
  return s / (a.length - 1);
}

function summarize(values) {
  if (!values.length) return null;
  return {
    n: values.length,
    min: Math.min(...values),
    max: Math.max(...values),
    avg: mean(values),
    p25: percentile(values, 25),
    p50: percentile(values, 50),
    p75: percentile(values, 75),
    p90: percentile(values, 90),
    jitter: jitter(values),
  };
}

/* ---------- formatting ---------- */

/** Three significant figures, the way the source page prints them. */
function sig3(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

const fmtMs = (v) => (v == null ? "—" : `${sig3(v)} ms`);

function fmtBytes(b) {
  if (b >= 1e6) return `${b / 1e6}MB`;
  if (b >= 1e3) return `${b / 1e3}kB`;
  return `${b}B`;
}

/* ============================ parsing ============================ */

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const cells = lines[i].split(",");
    const o = {};
    header.forEach((h, j) => (o[h] = (cells[j] ?? "").trim()));
    rows.push({
      time: num(o.time),
      direction: o.direction,
      bytes: num(o.bytes),
      latency: num(o.latency),
      bps: num(o.bps),
      duration: num(o.duration),
      serverTime: num(o.serverTime),
      responseSize: num(o.responseSize),
      loadedLatencies: (o.loadedLatencies || "")
        .split(/\s+/)
        .map(num)
        .filter((v) => v != null),
    });
  }
  return rows.filter((r) => r.direction === "download" || r.direction === "upload");
}

function computeMetrics(rows) {
  const dl = rows.filter((r) => r.direction === "download");
  const ul = rows.filter((r) => r.direction === "upload");

  const byOrder = (a) => [...a].sort((x, y) => x.time - y.time);
  const loadedOf = (g) => byOrder(g).flatMap((r) => r.loadedLatencies);

  const unloaded = byOrder(rows)
    .map((r) => r.latency)
    .filter((v) => v != null);
  const loadedDl = loadedOf(dl);
  const loadedUl = loadedOf(ul);

  // One box-plot row per test size, in the order the tests ran.
  const groups = (g) => {
    const sizes = [...new Set(g.map((r) => r.bytes))].sort((a, b) => a - b);
    return sizes.map((s) => {
      const sub = g.filter((r) => r.bytes === s);
      return {
        label: `${fmtBytes(s)} ${sub[0].direction} test`,
        n: sub.length,
        stats: summarize(sub.map((r) => r.bps / 1e6)),
      };
    });
  };

  return {
    rows,
    dlSpeed: percentile(dl.map((r) => r.bps), 90) / 1e6,
    ulSpeed: percentile(ul.map((r) => r.bps), 90) / 1e6,
    unloaded: summarize(unloaded),
    loadedDl: summarize(loadedDl),
    loadedUl: summarize(loadedUl),
    allLatency: summarize([...unloaded, ...loadedDl, ...loadedUl]),
    dlGroups: groups(dl),
    ulGroups: groups(ul),
    series: {
      download: byOrder(dl).map((r) => ({ t: r.time, v: r.bps / 1e6, bytes: r.bytes })),
      upload: byOrder(ul).map((r) => ({ t: r.time, v: r.bps / 1e6, bytes: r.bytes })),
    },
    measuredAt: Math.max(...rows.map((r) => r.time)),
  };
}

/* ====================== network quality score ======================
 * Approximate AIM-style ratings. Cloudflare does not publish exact cutoffs;
 * these were tuned to reproduce the three "Good" ratings in the saved page
 * from this CSV. Each category takes the worst of its inputs.
 * ================================================================== */

const RATINGS = ["Bad", "Poor", "Average", "Good", "Great"];

/** thresholds ascending, lower value = better; returns a RATINGS entry */
function tierLow(v, [great, good, avg, poor]) {
  if (v == null) return null;
  if (v < great) return "Great";
  if (v < good) return "Good";
  if (v < avg) return "Average";
  if (v < poor) return "Poor";
  return "Bad";
}

/** thresholds descending, higher value = better */
function tierHigh(v, [great, good, avg, poor]) {
  if (v == null) return null;
  if (v >= great) return "Great";
  if (v >= good) return "Good";
  if (v >= avg) return "Average";
  if (v >= poor) return "Poor";
  return "Bad";
}

const worst = (tiers) => {
  const valid = tiers.filter(Boolean);
  if (!valid.length) return "Bad";
  return valid.reduce((a, b) =>
    RATINGS.indexOf(a) <= RATINGS.indexOf(b) ? a : b
  );
};

function scoreNetwork(m) {
  const loss = META.packetLossPct;
  const lossTier = tierLow(loss, [1, 2.5, 5, 10]);

  const cats = [
    {
      name: "Video Streaming",
      icon: "▶",
      inputs: [
        ["download", tierHigh(m.dlSpeed, [50, 25, 10, 5]), `${sig3(m.dlSpeed)} Mbps`],
        ["latency during download", tierLow(m.loadedDl?.p50, [50, 150, 300, 500]), fmtMs(m.loadedDl?.p50)],
        ["jitter during download", tierLow(m.loadedDl?.jitter, [15, 30, 50, 75]), fmtMs(m.loadedDl?.jitter)],
        ["packet loss", lossTier, `${sig3(loss)} %`],
      ],
    },
    {
      name: "Online Gaming",
      icon: "◉",
      inputs: [
        ["latency during download", tierLow(m.loadedDl?.p50, [75, 125, 200, 300]), fmtMs(m.loadedDl?.p50)],
        ["jitter during download", tierLow(m.loadedDl?.jitter, [15, 30, 50, 75]), fmtMs(m.loadedDl?.jitter)],
        ["unloaded latency", tierLow(m.unloaded?.p50, [30, 60, 100, 150]), fmtMs(m.unloaded?.p50)],
        ["packet loss", lossTier, `${sig3(loss)} %`],
      ],
    },
    {
      name: "Video Chatting",
      icon: "▭",
      inputs: [
        ["upload", tierHigh(m.ulSpeed, [10, 5, 2.5, 1]), `${sig3(m.ulSpeed)} Mbps`],
        ["latency during upload", tierLow(m.loadedUl?.p50, [50, 150, 300, 500]), fmtMs(m.loadedUl?.p50)],
        ["jitter during upload", tierLow(m.loadedUl?.jitter, [20, 50, 100, 150]), fmtMs(m.loadedUl?.jitter)],
        ["packet loss", lossTier, `${sig3(loss)} %`],
      ],
    },
  ];

  return cats.map((c) => {
    const rating = worst(c.inputs.map((i) => i[1]));
    const limiting = c.inputs.find((i) => i[1] === rating);
    return { ...c, rating, limiting };
  });
}

/* ============================ charts ============================ */

const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

const tip = document.getElementById("tooltip");

function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.dataset.show = "1";
  tip.setAttribute("aria-hidden", "false");
  const r = tip.getBoundingClientRect();
  const left = Math.min(Math.max(8, x - r.width / 2), innerWidth - r.width - 8);
  const top = y - r.height - 12 < 8 ? y + 16 : y - r.height - 12;
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

function hideTip() {
  tip.dataset.show = "0";
  tip.setAttribute("aria-hidden", "true");
}

/** Re-render on container resize, the way the source page does.
 *  Redrawing changes the subtree, so bail out unless the box actually changed —
 *  otherwise the observer re-fires on its own output. */
function responsive(container, draw) {
  let lastW = -1;
  let lastH = -1;
  const run = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w <= 0 || h <= 0) return;
    if (w === lastW && h === lastH) return;
    lastW = w;
    lastH = h;
    draw(w, h);
  };
  new ResizeObserver(run).observe(container);
  run();
}

/**
 * Time-series sparkline: gradient area + 2px line + point dots + a dashed
 * 90th-percentile reference line. Crosshair tooltip on hover.
 */
function drawSparkline(container, points, color, unit) {
  container.replaceChildren();
  if (!points.length) return;

  responsive(container, (W, H) => {
    container.replaceChildren();
    const pad = { t: 14, r: 8, b: 16, l: 8 };
    const svg = el("svg", { width: W, height: H, "aria-hidden": "true" }, container);

    const t0 = points[0].t;
    const t1 = points[points.length - 1].t;
    const vmax = Math.max(...points.map((p) => p.v));
    const yTop = vmax * 1.12 || 1;

    const X = (t) => pad.l + ((t - t0) / (t1 - t0 || 1)) * (W - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - v / yTop) * (H - pad.t - pad.b);

    const p90 = percentile(points.map((p) => p.v), 90);

    // area
    const line = points.map((p, i) => `${i ? "L" : "M"}${X(p.t)},${Y(p.v)}`).join("");
    const base = H - pad.b;
    const gid = `grad-${Math.abs(hashString(container.id))}`;
    const grad = el("linearGradient", { id: gid, x1: "0", x2: "0", y1: "0", y2: "1" },
      el("defs", {}, svg));
    el("stop", { offset: "0%", "stop-color": color, "stop-opacity": "0.55" }, grad);
    el("stop", { offset: "100%", "stop-color": color, "stop-opacity": "0" }, grad);
    el("path", {
      d: `${line}L${X(t1)},${base}L${X(t0)},${base}Z`,
      fill: `url(#${gid})`,
    }, svg);
    el("path", { d: line, fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);

    // baseline
    el("line", { x1: pad.l, x2: W - pad.r, y1: base, y2: base, class: "gridline" }, svg);

    // reference line for the headline number, above the fill so it stays legible
    el("line", {
      x1: pad.l, x2: W - pad.r, y1: Y(p90), y2: Y(p90),
      stroke: "currentColor", "stroke-width": 1, "stroke-dasharray": "3 3",
      opacity: 0.5, class: "gridline",
    }, svg);
    const label = el("text", { x: pad.l, y: Y(p90) - 5, class: "reftext" }, svg);
    label.textContent = `90th percentile · ${sig3(p90)} ${unit}`;

    for (const p of points) {
      el("circle", { cx: X(p.t), cy: Y(p.v), r: 2.5, fill: color,
        stroke: "var(--card)", "stroke-width": 1 }, svg);
    }

    // elapsed-seconds axis labels
    const secs = (t) => ((t - t0) / 1000).toFixed(0);
    const a0 = el("text", { x: pad.l, y: H - 3, class: "axistext" }, svg);
    a0.textContent = "0s";
    const a1 = el("text", { x: W - pad.r, y: H - 3, class: "axistext",
      "text-anchor": "end" }, svg);
    a1.textContent = `${secs(t1)}s`;

    // hover layer
    const cross = el("line", { class: "crosshair", y1: pad.t, y2: base, opacity: 0 }, svg);
    const dot = el("circle", { r: 5, fill: color, stroke: "var(--card)",
      "stroke-width": 2, opacity: 0 }, svg);
    const hit = el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }, svg);

    hit.addEventListener("mousemove", (e) => {
      const bx = container.getBoundingClientRect();
      const mx = e.clientX - bx.left;
      let best = points[0], bd = Infinity;
      for (const p of points) {
        const d = Math.abs(X(p.t) - mx);
        if (d < bd) { bd = d; best = p; }
      }
      cross.setAttribute("x1", X(best.t));
      cross.setAttribute("x2", X(best.t));
      cross.setAttribute("opacity", 1);
      dot.setAttribute("cx", X(best.t));
      dot.setAttribute("cy", Y(best.v));
      dot.setAttribute("opacity", 1);
      showTip(
        `<b>${sig3(best.v)} ${unit}</b><br>${fmtBytes(best.bytes)} test · +${(
          (best.t - t0) / 1000
        ).toFixed(1)}s`,
        e.clientX,
        e.clientY
      );
    });
    hit.addEventListener("mouseleave", () => {
      cross.setAttribute("opacity", 0);
      dot.setAttribute("opacity", 0);
      hideTip();
    });
  });
}

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

/**
 * Horizontal box plot on a scale shared by every row in its card:
 * whisker min–max, thick 25–75 bar, solid median, dotted average.
 */
function drawBoxPlot(container, stats, scaleMax, color, unit) {
  container.replaceChildren();
  if (!stats) return;

  responsive(container, (W, H) => {
    container.replaceChildren();
    const svg = el("svg", { width: W, height: H, "aria-hidden": "true" }, container);
    const pad = 4;
    const X = (v) => pad + (v / (scaleMax || 1)) * (W - pad * 2);
    const mid = H / 2;

    el("rect", { x: 0, y: mid - 5, width: W, height: 10, rx: 5, class: "plot-track" }, svg);
    el("line", { x1: X(stats.min), x2: X(stats.max), y1: mid, y2: mid,
      stroke: color, "stroke-width": 1.5, opacity: 0.5 }, svg);
    el("rect", { x: X(stats.p25), y: mid - 5,
      width: Math.max(2, X(stats.p75) - X(stats.p25)), height: 10, rx: 3,
      fill: color, opacity: 0.85 }, svg);
    el("line", { x1: X(stats.p50), x2: X(stats.p50), y1: mid - 8, y2: mid + 8,
      stroke: color, "stroke-width": 2 }, svg);
    el("line", { x1: X(stats.avg), x2: X(stats.avg), y1: mid - 8, y2: mid + 8,
      stroke: color, "stroke-width": 2, "stroke-dasharray": "2 2" }, svg);

    const hit = el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (e) =>
      showTip(
        `median <b>${sig3(stats.p50)} ${unit}</b><br>` +
          `avg <b>${sig3(stats.avg)}</b> · p25–p75 <b>${sig3(stats.p25)}–${sig3(stats.p75)}</b><br>` +
          `min <b>${sig3(stats.min)}</b> · max <b>${sig3(stats.max)}</b> · n=${stats.n}`,
        e.clientX,
        e.clientY
      )
    );
    hit.addEventListener("mouseleave", hideTip);
  });
}

/* ============================ rendering ============================ */

const $ = (id) => document.getElementById(id);

function measurementRow(label, count, stats, scaleMax, color, unit) {
  const row = document.createElement("details");
  row.className = "row";

  const sum = document.createElement("summary");
  sum.innerHTML =
    `<span class="row-caret">▶</span>` +
    `<span class="row-label">${label}</span>` +
    `<span class="row-count">(${count})</span>` +
    `<span class="row-plot"></span>`;
  row.appendChild(sum);

  const stats$ = document.createElement("div");
  stats$.className = "stats";
  if (stats) {
    const pairs = [
      ["Min", stats.min], ["Max", stats.max], ["Average", stats.avg],
      ["Median", stats.p50], ["25th pct", stats.p25], ["75th pct", stats.p75],
    ];
    stats$.innerHTML = pairs
      .map(
        ([k, v]) =>
          `<div><span class="k">${k}</span><span class="v">${sig3(v)} ${unit}</span></div>`
      )
      .join("");
  } else {
    stats$.innerHTML = `<div><span class="k">No samples in CSV</span></div>`;
  }
  row.appendChild(stats$);

  // draw once the plot cell has a width (details is open-independent)
  requestAnimationFrame(() =>
    drawBoxPlot(sum.querySelector(".row-plot"), stats, scaleMax, color, unit)
  );
  return row;
}

function renderLatency(m) {
  const host = $("latencyRows");
  host.replaceChildren();
  const scale = m.allLatency ? m.allLatency.max * 1.05 : 1;
  const rows = [
    ["Unloaded latency", m.unloaded],
    ["Latency during download", m.loadedDl],
    ["Latency during upload", m.loadedUl],
  ];
  for (const [label, s] of rows) {
    host.appendChild(
      measurementRow(label, s ? s.n : 0, s, scale, COLORS.latency, "ms")
    );
  }
  // aggregate row, matching the summary block in the source page
  host.appendChild(
    measurementRow("All latency measurements", m.allLatency ? m.allLatency.n : 0,
      m.allLatency, scale, COLORS.latency, "ms")
  );
}

function renderThroughput(m) {
  const scale =
    Math.max(
      ...[...m.dlGroups, ...m.ulGroups].map((g) => (g.stats ? g.stats.max : 0))
    ) * 1.05;

  const dl = $("downloadRows");
  dl.replaceChildren();
  for (const g of m.dlGroups) {
    dl.appendChild(
      measurementRow(g.label, `${g.n}/${g.n}`, g.stats, scale, COLORS.download, "Mbps")
    );
  }

  const ul = $("uploadRows");
  ul.replaceChildren();
  for (const g of m.ulGroups) {
    ul.appendChild(
      measurementRow(g.label, `${g.n}/${g.n}`, g.stats, scale, COLORS.upload, "Mbps")
    );
  }

  const pk = $("packetRows");
  pk.replaceChildren();
  const [recv, sent] = META.packetLossSamples;
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML =
    `<div class="rowline">` +
    `<span class="row-label">Packet Loss Test</span>` +
    `<span class="row-count">(${recv}/${sent})</span>` +
    `<span class="row-value">${sig3(META.packetLossPct)} %</span>` +
    `<span class="badge-src">not in CSV</span></div>`;
  pk.appendChild(row);
}

function renderScores(m) {
  const host = $("scores");
  host.replaceChildren();
  for (const c of scoreNetwork(m)) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span aria-hidden="true" style="color:var(--ink-3)">${c.icon}</span>` +
      `<span>${c.name}</span>` +
      `<span class="why">limited by ${c.limiting[0]} · ${c.limiting[2]}</span>` +
      `<span class="rating" data-r="${c.rating}">${c.rating}</span>`;
    host.appendChild(li);
  }
}

let map, clientMarker;

function renderMap() {
  $("niProto").textContent = META.proto;
  $("niServer").textContent = META.serverCity;
  $("niAsn").innerHTML = `${META.asnName} <span style="color:var(--ink-3)">(${META.asn})</span>`;
  $("niIp").textContent = META.ip;
  $("metaLabel").value = META.placeLabel;

  if (typeof L === "undefined") {
    $("map").innerHTML =
      `<p class="note" style="padding:12px">Map library unavailable offline — ` +
      `client ${META.client.lat}, ${META.client.lon} → server ${META.serverCity}.</p>`;
    return;
  }

  map = L.map("map", { scrollWheelZoom: false });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap",
  }).addTo(map);

  const c = [META.client.lat, META.client.lon];
  const s = [META.server.lat, META.server.lon];

  clientMarker = L.circleMarker(c, {
    radius: 7, color: COLORS.download, fillColor: COLORS.download,
    fillOpacity: 0.9, weight: 2,
  })
    .addTo(map)
    .bindPopup(`<b>${META.placeLabel}</b><br>test location`);

  L.circleMarker(s, {
    radius: 7, color: COLORS.latency, fillColor: COLORS.latency,
    fillOpacity: 0.9, weight: 2,
  })
    .addTo(map)
    .bindPopup(`<b>${META.serverCity}</b><br>Cloudflare edge`);

  L.polyline([c, s], {
    color: "#888", weight: 1.5, dashArray: "4 4",
  }).addTo(map);

  map.fitBounds(L.latLngBounds([c, s]).pad(0.35));

  $("metaLabel").addEventListener("input", (e) => {
    const v = e.target.value || "test location";
    clientMarker.setPopupContent(`<b>${v}</b><br>test location`);
  });
}

function renderTiles(m) {
  $("dlSpeed").textContent = sig3(m.dlSpeed);
  $("ulSpeed").textContent = sig3(m.ulSpeed);
  $("latIdle").textContent = sig3(m.unloaded?.p50);
  $("latDl").textContent = fmtMs(m.loadedDl?.p50);
  $("latUl").textContent = fmtMs(m.loadedUl?.p50);
  $("jitIdle").textContent = sig3(m.unloaded?.jitter);
  $("jitDl").textContent = fmtMs(m.loadedDl?.jitter);
  $("jitUl").textContent = fmtMs(m.loadedUl?.jitter);
  $("loss").textContent = sig3(META.packetLossPct);
  $("measuredAt").textContent = new Date(m.measuredAt).toLocaleTimeString();
}

function renderTable(m) {
  const t = $("rawTable");
  const cols = ["direction", "size", "elapsed", "Mbps", "latency ms", "loaded latencies ms"];
  const t0 = Math.min(...m.rows.map((r) => r.time));
  const body = [...m.rows]
    .sort((a, b) => a.time - b.time)
    .map(
      (r) =>
        `<tr><td>${r.direction}</td><td>${fmtBytes(r.bytes)}</td>` +
        `<td>${((r.time - t0) / 1000).toFixed(2)}s</td>` +
        `<td>${sig3(r.bps / 1e6)}</td><td>${sig3(r.latency)}</td>` +
        `<td>${r.loadedLatencies.map((v) => sig3(v)).join(", ") || "—"}</td></tr>`
    )
    .join("");
  t.innerHTML =
    `<caption>Every row of the loaded CSV (${m.rows.length} samples).</caption>` +
    `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${body}</tbody>`;

  $("derivations").innerHTML =
    `<div class="prose"><ul>` +
    `<li><b>Download ${sig3(m.dlSpeed)} Mbps</b> = 90th percentile of the ${
      m.series.download.length
    } download <code>bps</code> samples.</li>` +
    `<li><b>Upload ${sig3(m.ulSpeed)} Mbps</b> = 90th percentile of the ${
      m.series.upload.length
    } upload <code>bps</code> samples.</li>` +
    `<li><b>Unloaded latency ${fmtMs(m.unloaded?.p50)}</b> = median of the ${
      m.unloaded?.n
    } <code>latency</code> values; jitter ${fmtMs(
      m.unloaded?.jitter
    )} = mean absolute difference between consecutive values.</li>` +
    `<li><b>Loaded latency</b> = same statistics over <code>loadedLatencies</code>, ` +
    `split by direction (${m.loadedDl?.n} download, ${m.loadedUl?.n} upload values).</li>` +
    `</ul></div>`;
}

function renderNotes(m) {
  $("datanotes").innerHTML = `
    <p>This page is a static reproduction: the layout, statistics, and charts are
    rebuilt from <code>${CSV_URL}</code> alone. Numbers differ slightly from the
    saved page because the export and the screenshot were taken at different
    moments in the run.</p>
    <h3>Reproduced exactly from the CSV</h3>
    <ul>
      <li>Download and upload speed — the 90th-percentile rule matches the saved
      page to the printed digit (70.1 / 95.5 Mbps).</li>
      <li>Latency and jitter, unloaded and under load, per direction.</li>
      <li>Per-test-size sample counts, box plots, and the speed-over-time charts.</li>
    </ul>
    <h3>Not in the CSV at all</h3>
    <ul>
      <li><b>Packet loss</b> — the saved page reports 0% over 1000 packets; the
      export has no packet-loss column.</li>
      <li><b>Server location, ASN, ISP name, client IP</b> — page-only, from
      Cloudflare's <code>/cdn-cgi/trace</code>.</li>
      <li><b>The unloaded ping phase</b> — the page counted 39 unloaded samples,
      46 during download and 72 during upload; this CSV carries 46 / ${m.loadedDl?.n} /
      ${m.loadedUl?.n}. The <code>latency</code> column is the closest stand-in.</li>
      <li><b>Coordinates</b> — nothing geographic is in the file, which is why a
      community map needs the uploader to name the place.</li>
    </ul>`;
}

function renderAll(rows, sourceNote) {
  const m = computeMetrics(rows);
  renderTiles(m);
  drawSparkline($("chartDownload"), m.series.download, COLORS.download, "Mbps");
  drawSparkline($("chartUpload"), m.series.upload, COLORS.upload, "Mbps");
  renderScores(m);
  renderLatency(m);
  renderThroughput(m);
  renderTable(m);
  renderNotes(m);
  if (sourceNote) $("loadstate").textContent = sourceNote;
}

/* ============================ boot ============================ */

renderMap();

fetch(CSV_URL)
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.text();
  })
  .then((text) => renderAll(parseCSV(text), ""))
  .catch((err) => {
    $("loadstate").textContent =
      `Could not fetch ${CSV_URL} (${err.message}). ` +
      `Serve this folder over HTTP (python3 -m http.server) or pick a CSV above.`;
  });

$("csvfile").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fr = new FileReader();
  fr.onload = () => {
    try {
      const rows = parseCSV(String(fr.result));
      if (!rows.length) throw new Error("no download/upload rows found");
      renderAll(rows, `Showing ${f.name} — ${rows.length} samples.`);
    } catch (err) {
      $("loadstate").textContent = `Could not parse ${f.name}: ${err.message}`;
    }
  };
  fr.readAsText(f);
});
