#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIDC Reliability Map</title>
  <script src="https://cdn.plot.ly/plotly-3.3.1.min.js" crossorigin="anonymous"></script>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\(', '\\)'], ['$', '$']], displayMath: [['\\[', '\\]']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: rgba(255, 252, 246, 0.9);
      --ink: #182028;
      --muted: #5c646c;
      --accent: #0e6b5c;
      --border: rgba(24, 32, 40, 0.12);
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at 15% 20%, rgba(229, 181, 107, 0.25), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(49, 113, 102, 0.18), transparent 24%),
        linear-gradient(180deg, #f5f1e8 0%, #ece4d4 100%);
      color: var(--ink);
    }
    .shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }
    .header {
      margin-bottom: 16px;
    }
    .header h1 {
      margin: 0 0 6px 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .header p {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      align-items: start;
    }
    .panel, .map-panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 16px 50px rgba(50, 40, 20, 0.08);
      backdrop-filter: blur(8px);
    }
    .panel {
      padding: 18px 18px 14px;
    }
    .map-panel {
      padding: 14px;
    }
    .method-panel {
      margin-top: 18px;
      padding: 18px 20px;
    }
    .control {
      margin-bottom: 18px;
    }
    .control label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
      margin-bottom: 8px;
    }
    .control strong {
      font-weight: 700;
    }
    input[type=range] {
      width: 100%;
      accent-color: var(--accent);
    }
    .stats {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .stat {
      padding: 12px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(24,32,40,0.08);
    }
    .stat .k {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .stat .v {
      font-size: 22px;
      margin-top: 4px;
      font-weight: 700;
    }
    #map {
      width: 100%;
      height: 760px;
    }
    .footnote {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .method-panel h2 {
      margin: 0 0 10px 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }
    .method-panel p {
      margin: 0 0 12px 0;
      color: var(--ink);
      font-size: 15px;
      line-height: 1.6;
    }
    .method-panel ul {
      margin: 0 0 14px 20px;
      padding: 0;
      color: var(--ink);
      font-size: 15px;
      line-height: 1.6;
    }
    .equation-block {
      margin: 14px 0;
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(24,32,40,0.08);
      overflow-x: auto;
    }
    .equation-block p {
      margin: 0;
    }
    .method-subhead {
      margin: 22px 0 10px 0;
      font-size: 18px;
      letter-spacing: -0.01em;
    }
    .method-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      align-items: start;
    }
    .method-main {
      min-width: 0;
    }
    .method-side {
      display: grid;
      gap: 14px;
    }
    .assumption-card {
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(255,255,255,0.58);
      border: 1px solid rgba(24,32,40,0.08);
    }
    .assumption-card h3 {
      margin: 0 0 8px 0;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    .assumption-card p {
      margin: 0;
      font-size: 14px;
      color: var(--ink);
      line-height: 1.6;
    }
    .assumption-list {
      display: grid;
      gap: 10px;
    }
    .assumption-item {
      display: grid;
      gap: 2px;
      padding-top: 8px;
      border-top: 1px solid rgba(24,32,40,0.08);
    }
    .assumption-item:first-child {
      padding-top: 0;
      border-top: 0;
    }
    .assumption-term {
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
    }
    .assumption-meta {
      font-size: 12px;
      color: var(--muted);
    }
    .assumption-desc {
      font-size: 13px;
      color: var(--ink);
      line-height: 1.5;
    }
    .map-panel {
      position: relative;
    }
    .fullscreen-btn {
      position: absolute;
      top: 22px;
      right: 22px;
      z-index: 10;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--muted);
      transition: background 0.15s, color 0.15s;
      box-shadow: 0 2px 8px rgba(50,40,20,0.10);
    }
    .fullscreen-btn:hover {
      background: var(--accent);
      color: #fff;
    }
    .map-panel.is-fullscreen {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      z-index: 9999;
      border-radius: 0;
      margin: 0;
      padding: 0;
    }
    .map-panel.is-fullscreen #map {
      height: 100vh !important;
    }
    .overlay-section {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
    }
    .overlay-section label.section-label {
      display: block;
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .overlay-toggles {
      display: grid;
      gap: 8px;
    }
    .overlay-toggles label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      cursor: pointer;
      padding: 6px 8px;
      border-radius: 8px;
      transition: background 0.15s;
    }
    .overlay-toggles label:hover {
      background: rgba(14, 107, 92, 0.06);
    }
    .overlay-toggles input[type=checkbox] {
      accent-color: var(--accent);
      width: 16px;
      height: 16px;
    }
    .overlay-legend {
      display: inline-block;
      width: 14px;
      height: 3px;
      border-radius: 2px;
      vertical-align: middle;
    }
    .overlay-legend.dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .overlay-legend.area {
      width: 14px;
      height: 10px;
      border-radius: 3px;
      opacity: 0.6;
    }
    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
      }
      #map {
        height: 620px;
      }
      .method-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <h1>AIDC Reliability Map</h1>
      <p>Reliability for a fixed 1 GW workload. Sliders select a cached deterministic slice. BESS is treated as energy only in this V1 map.</p>
    </div>
    <div class="layout">
      <div class="panel">
        <div class="control">
          <label for="solar"><strong>Solar Capacity</strong><span id="solarValue"></span></label>
          <input id="solar" type="range" min="0" max="__SOLAR_SLIDER_MAX__" step="__SOLAR_SLIDER_STEP__" value="0">
        </div>
        <div class="control">
          <label for="wind"><strong>Wind Capacity</strong><span id="windValue"></span></label>
          <input id="wind" type="range" min="0" max="__WIND_SLIDER_MAX__" step="__WIND_SLIDER_STEP__" value="0">
        </div>
        <div class="control">
          <label for="bess"><strong>BESS Energy</strong><span id="bessValue"></span></label>
          <input id="bess" type="range" min="0" max="__BESS_SLIDER_MAX__" step="__BESS_SLIDER_STEP__" value="0">
        </div>
      <div class="control">
          <label for="colorMode"><strong>Color Scale</strong><span id="colorModeValue"></span></label>
          <select id="colorMode" style="width:100%; padding:10px 12px; border-radius:12px; border:1px solid rgba(24,32,40,0.12); background:rgba(255,255,255,0.8); color:var(--ink);">
            <option value="fixed">Fixed 0-100%</option>
            <option value="auto">Auto visible range</option>
          </select>
        </div>
        <div class="stats">
          <div class="stat">
            <div class="k">Mean Reliability</div>
            <div class="v" id="meanReliability">0.0%</div>
          </div>
          <div class="stat">
            <div class="k">Best Site</div>
            <div class="v" id="bestReliability">0.0%</div>
          </div>
          <div class="stat">
            <div class="k">Site Count</div>
            <div class="v" id="siteCount">0</div>
          </div>
        </div>

        <div class="overlay-section" id="overlaySection" style="display:none">
          <label class="section-label">Map Overlays</label>
          <div class="overlay-toggles" id="overlayToggles"></div>
        </div>

        <div class="footnote">
          Cache model: __CACHE_MODEL__. The UI does indexed lookup only; Monte Carlo is intentionally not run in-browser.
        </div>
      </div>
      <div class="map-panel" id="mapPanel">
        <button class="fullscreen-btn" id="fullscreenBtn" title="Toggle fullscreen">&#x26F6; Fullscreen</button>
        <div id="map"></div>
      </div>
    </div>
    <div class="panel method-panel">
      <h2>Method</h2>
      <div class="method-grid">
        <div class="method-main">
          <p>
            The map value is the percentage of hourly time steps in a year for which a fixed
            \(1\,\text{GW}\) workload can be served by solar, wind, and battery storage.
          </p>
          <div class="equation-block">
            <p>\[
              G_t = S_t \, C_{\mathrm{solar}} + W_t \, C_{\mathrm{wind}}
            \]</p>
          </div>
          <p>
            Here \(S_t\) and \(W_t\) are location-dependent per-MW generation traces, while
            \(C_{\mathrm{solar}}\) and \(C_{\mathrm{wind}}\) are the installed capacities selected by the sliders.
          </p>
          <div class="equation-block">
            <p>\[
              \text{surplus}_t = \max(G_t - L, 0), \qquad
              \text{deficit}_t = \max(L - G_t, 0)
            \]</p>
          </div>
          <p>
            The battery charges from renewable surplus and discharges into deficit. Its state of charge evolves as:
          </p>
          <div class="equation-block">
            <p>\[
              \text{SOC}_{t+1}
              =
              \text{SOC}_t
              +
              \text{charge}_t \, \eta_c
              -
              \frac{\text{discharge}_t}{\eta_d}
            \]</p>
          </div>
          <p>
            with charge and discharge constrained by available energy and usable depth of discharge:
          </p>
          <div class="equation-block">
            <p>\[
              \text{charge}_t
              =
              \min\!\left(
                \text{surplus}_t,
                \frac{E_{\max} - \text{SOC}_t}{\eta_c}
              \right)
            \]</p>
            <p>\[
              \text{discharge}_t
              =
              \min\!\left(
                \text{deficit}_t,
                (\text{SOC}_t - \text{SOC}_{\min}) \eta_d
              \right)
            \]</p>
          </div>
          <p>
            A time step is counted as served when the remaining deficit is zero or negative:
          </p>
          <div class="equation-block">
            <p>\[
              \text{remaining\_deficit}_t
              =
              L
              -
              \min(G_t, L)
              -
              \text{discharge}_t
            \]</p>
          </div>
          <div class="equation-block">
            <p>\[
              \text{served}_t =
              \begin{cases}
                1, & \text{if } \text{remaining\_deficit}_t \le 0 \\
                0, & \text{otherwise}
              \end{cases}
            \]</p>
          </div>
          <p>
            Reliability is then:
          </p>
          <div class="equation-block">
            <p>\[
              \text{Reliability}(\%)
              =
              100 \times
              \frac{\sum_{t=1}^{8760} \text{served}_t}{8760}
            \]</p>
          </div>
          <h3 class="method-subhead">Real Data Pipeline</h3>
          <p>
            The cached map uses real hourly weather/resource traces where available. For each site, one NSRDB solar file
            and one WIND Toolkit wind file are cached, converted into per-MW generation profiles, and then reused during
            the reliability sweep across all slider combinations.
          </p>
          <ul>
            <li>Solar raw data: <a href="https://nsrdb.nrel.gov/" target="_blank">NREL NSRDB</a> (<a href="https://nsrdb.nrel.gov/data-viewer" target="_blank">data viewer</a>) hourly CSV for solar year <strong>__SOLAR_YEAR__</strong> with \(GHI\), \(DNI\), \(DHI\), temperature, and wind speed.</li>
            <li>Wind raw data: <a href="https://www.nrel.gov/grid/wind-toolkit.html" target="_blank">NREL WIND Toolkit</a> (<a href="https://wrdb.nlr.gov/data-viewer" target="_blank">data viewer</a>) hourly CSV for wind year <strong>__WIND_YEAR__</strong> with \(windspeed_{100m}\), wind direction, temperature, and pressure.</li>
            <li>Solar conversion: hourly PV output fraction per MW from \(GHI / 1000\), fixed losses, and a temperature derate.</li>
            <li>Wind conversion: hourly wind output fraction per MW from \(windspeed_{100m}\), a generic turbine power curve, and fixed losses.</li>
            <li>Per-site derived cache: one compressed profile file storing hourly solar, hourly wind, and source metadata for each successfully cached site.</li>
            <li>Reliability sweep: for each site and each \((C_{\mathrm{solar}}, C_{\mathrm{wind}}, E_{\max})\) tuple, the simulator computes \(G_t\), dispatches BESS hour by hour, and records the percentage of served hours.</li>
            <li>Current real-data coverage in this build: <strong>__REAL_SITE_COUNT__</strong> sites use real derived profiles; any remaining uncovered sites fall back to the older synthetic cache in the merged frontend artifact.</li>
          </ul>
          <h3 class="method-subhead">Overlay Data Sources</h3>
          <ul>
            <li>Natural gas pipelines: <a href="https://catalog.data.gov/dataset/natural-gas-pipelines" target="_blank">HIFLD Natural Gas Pipelines</a> (<a href="https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/Natural_Gas_Pipelines___Copy_shp/FeatureServer/0" target="_blank">ArcGIS FeatureServer</a>). 33,806 polyline features covering interstate and intrastate transmission pipelines. Simplified with Douglas-Peucker algorithm for browser rendering.</li>
            <li>EPA nonattainment zones: <a href="https://www.epa.gov/green-book" target="_blank">EPA Green Book</a> via <a href="https://gispub.epa.gov/arcgis/rest/services/OAR_OAQPS/NonattainmentAreas/MapServer" target="_blank">ArcGIS MapServer</a>. Covers 8-hour Ozone (2015 NAAQS) and PM2.5 Annual (2012 NAAQS) nonattainment area boundaries. Relevant for backup generator permitting thresholds.</li>
            <li>Data center / IXP facilities: <a href="https://www.peeringdb.com/" target="_blank">PeeringDB</a> US facility database. 1,347 CONUS data center and Internet exchange point locations with operator, connected network count, and IX count metadata.</li>
            <li>Fiber routes: real carrier fiber networks from <a href="https://services.arcgis.com/QO7AhQX53d0m9M2s/arcgis/rest/services/Uniti_Wholesale_Map_Demo_4_WFL1/FeatureServer" target="_blank">Uniti/Windstream</a> national core (26K segments, CONUS backbone), <a href="https://services2.arcgis.com/idZndEogp4LoOmcv/arcgis/rest/services/SegraFiberNetwork_Consolidated/FeatureServer" target="_blank">Segra</a> consolidated network (70K segments, East Coast GA-NY corridor), and <a href="https://services7.arcgis.com/6h8O0NpKpUyHiCti/arcgis/rest/services/AGOL_Publish4/FeatureServer" target="_blank">GeoTel</a> multi-carrier dataset (22K segments, PA+OH, 18 carriers including CenturyLink, Zayo, Crown Castle). Submarine cable routes from <a href="https://www.submarinecablemap.com/" target="_blank">TeleGeography</a> (108 US-touching cables with owner, length, and landing point metadata). Each carrier network is color-coded: blue (Uniti), green (Segra), purple (GeoTel), dotted blue (submarine). <em>Note: coverage is densest in the eastern US and along major backbone corridors. Western US fiber route data is underrepresented because no free public carrier datasets with that footprint were found; comprehensive national fiber route data is held commercially by GeoTel/Ookla and Infrapedia.</em></li>
          </ul>
        </div>
        <div class="method-side">
          __METHOD_SIDE_HTML__
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = __DATA_JSON__;

    const solarAxis = DATA.axes.solar_mw;
    const windAxis = DATA.axes.wind_mw;
    const bessAxis = DATA.axes.bess_mwh;
    const sites = DATA.sites;

    const lat = sites.map(s => s.lat);
    const lon = sites.map(s => s.lon);
    const resourceYears = DATA.meta.resource_years || {};
    function formatPopulation(value) {
      const population = Number(value || 0);
      if (population <= 0) {
        return "unknown";
      }
      return population.toLocaleString();
    }

    function formatHover(site, reliability) {
      const title = site.name ? `${site.name}${site.state ? `, ${site.state}` : ""}` : `Site ${site.site_id}`;
      const location = `Lat ${Number(site.lat).toFixed(3)}, Lon ${Number(site.lon).toFixed(3)}`;
      const resource = site.resource_source_label || site.resource_source || "Unknown";
      const population = `Population ${formatPopulation(site.population)}`;
      const years = site.resource_source === "real"
        ? `Data years solar ${resourceYears.solar_year ?? "n/a"}, wind ${resourceYears.wind_year ?? "n/a"}`
        : "Data source synthetic fallback";
      return `${title}<br>${population}<br>${location}<br>Resource ${resource}<br>${years}<br>Reliability ${reliability.toFixed(3)}%`;
    }

    const solarInput = document.getElementById("solar");
    const windInput = document.getElementById("wind");
    const bessInput = document.getElementById("bess");

    const solarValue = document.getElementById("solarValue");
    const windValue = document.getElementById("windValue");
    const bessValue = document.getElementById("bessValue");
    const meanReliability = document.getElementById("meanReliability");
    const bestReliability = document.getElementById("bestReliability");
    const siteCount = document.getElementById("siteCount");
    const colorMode = document.getElementById("colorMode");
    const colorModeValue = document.getElementById("colorModeValue");

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function bracket(axis, target) {
      if (target <= axis[0]) {
        return { lo: 0, hi: 0, t: 0 };
      }
      if (target >= axis[axis.length - 1]) {
        const last = axis.length - 1;
        return { lo: last, hi: last, t: 0 };
      }
      for (let i = 0; i < axis.length - 1; i++) {
        const a = axis[i];
        const b = axis[i + 1];
        if (target >= a && target <= b) {
          return { lo: i, hi: i + 1, t: (target - a) / Math.max(b - a, 1e-9) };
        }
      }
      const last = axis.length - 1;
      return { lo: last, hi: last, t: 0 };
    }

    function comboIndex(si, wi, bi) {
      return ((si * windAxis.length) + wi) * bessAxis.length + bi;
    }

    function currentValues() {
      const solar = Number(solarInput.value);
      const wind = Number(windInput.value);
      const bess = Number(bessInput.value);
      return { solar, wind, bess };
    }

    function reliabilitySlice(si, wi, bi) {
      return DATA.values_by_combo[comboIndex(si, wi, bi)];
    }

    function interpolatedSlice(solarTarget, windTarget, bessTarget) {
      const sb = bracket(solarAxis, solarTarget);
      const wb = bracket(windAxis, windTarget);
      const bb = bracket(bessAxis, bessTarget);
      const out = new Array(sites.length).fill(0);

      for (let sBit = 0; sBit < 2; sBit++) {
        const si = sBit === 0 ? sb.lo : sb.hi;
        const sw = (sb.lo === sb.hi) ? (sBit === 0 ? 1 : 0) : (sBit === 0 ? 1 - sb.t : sb.t);
        if (sw === 0) continue;
        for (let wBit = 0; wBit < 2; wBit++) {
          const wi = wBit === 0 ? wb.lo : wb.hi;
          const ww = (wb.lo === wb.hi) ? (wBit === 0 ? 1 : 0) : (wBit === 0 ? 1 - wb.t : wb.t);
          if (ww === 0) continue;
          for (let bBit = 0; bBit < 2; bBit++) {
            const bi = bBit === 0 ? bb.lo : bb.hi;
            const bw = (bb.lo === bb.hi) ? (bBit === 0 ? 1 : 0) : (bBit === 0 ? 1 - bb.t : bb.t);
            if (bw === 0) continue;
            const slice = reliabilitySlice(si, wi, bi);
            const weight = sw * ww * bw;
            for (let idx = 0; idx < slice.length; idx++) {
              out[idx] += slice[idx] * weight;
            }
          }
        }
      }
      return out;
    }

    const recommended = DATA.meta.recommended_defaults || {solar_mw: 6000, wind_mw: 0, bess_mwh: 40000};
    solarInput.value = String(clamp(recommended.solar_mw, 0, Number(solarInput.max)));
    windInput.value = String(clamp(recommended.wind_mw, 0, Number(windInput.max)));
    bessInput.value = String(clamp(recommended.bess_mwh, 0, Number(bessInput.max)));

    const initial = currentValues();
    const initialValues = interpolatedSlice(initial.solar, initial.wind, initial.bess);
    const hoverText = initialValues.map((value, idx) => formatHover(sites[idx], value));

    const trace = {
      type: "scattergeo",
      lat,
      lon,
      mode: "markers",
      text: hoverText,
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        symbol: "square",
        size: 11,
        opacity: 0.88,
        color: initialValues,
        cmin: 0,
        cmax: 100,
        colorscale: [
          [0.00, "#8f1d1d"],
          [0.20, "#c14d28"],
          [0.40, "#d9a441"],
          [0.60, "#b0bf4d"],
          [0.80, "#4e9d61"],
          [1.00, "#0f6b58"]
        ],
        colorbar: {
          title: "Reliability %",
          thickness: 18,
          len: 0.78
        },
        line: {
          color: "rgba(255,255,255,0.4)",
          width: 0.4
        }
      }
    };

    const layout = {
      margin: {l: 0, r: 0, t: 0, b: 0},
      geo: {
        scope: "usa",
        projection: {type: "albers usa"},
        showland: true,
        landcolor: "#fbf6ee",
        showlakes: true,
        lakecolor: "#e6efe9",
        bgcolor: "rgba(0,0,0,0)",
        subunitcolor: "rgba(0,0,0,0.15)",
        countrycolor: "rgba(0,0,0,0.18)"
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)"
    };

    Plotly.newPlot("map", [trace], layout, {responsive: true, displayModeBar: false});

    function updateStats(values) {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const best = Math.max(...values);
      meanReliability.textContent = `${mean.toFixed(1)}%`;
      bestReliability.textContent = `${best.toFixed(1)}%`;
    }

    function render() {
      const { solar, wind, bess } = currentValues();
      solarValue.textContent = `${solar.toLocaleString()} MW`;
      windValue.textContent = `${wind.toLocaleString()} MW`;
      bessValue.textContent = `${bess.toLocaleString()} MWh`;
      colorModeValue.textContent = colorMode.value === "fixed" ? "0-100%" : "Dynamic";

      const values = interpolatedSlice(solar, wind, bess);
      const text = values.map((value, idx) => formatHover(sites[idx], value));
      siteCount.textContent = String(lat.length);
      let cmin = 0;
      let cmax = 100;
      if (colorMode.value === "auto") {
        cmin = Math.min(...values);
        cmax = Math.max(...values);
        if (Math.abs(cmax - cmin) < 1e-6) {
          cmin = Math.max(0, cmin - 1);
          cmax = Math.min(100, cmax + 1);
        }
      }
      Plotly.restyle("map", {
        "lat": [lat],
        "lon": [lon],
        "marker.color": [values],
        "marker.cmin": [cmin],
        "marker.cmax": [cmax],
        "text": [text]
      }, [0]);
      updateStats(values);
    }

    solarInput.addEventListener("input", render);
    windInput.addEventListener("input", render);
    bessInput.addEventListener("input", render);
    colorMode.addEventListener("change", render);
    render();

    // ---- Fullscreen toggle ----
    const mapPanel = document.getElementById("mapPanel");
    const fullscreenBtn = document.getElementById("fullscreenBtn");
    fullscreenBtn.addEventListener("click", function() {
      mapPanel.classList.toggle("is-fullscreen");
      const isFS = mapPanel.classList.contains("is-fullscreen");
      fullscreenBtn.innerHTML = isFS ? "&#x2716; Exit" : "&#x26F6; Fullscreen";
      setTimeout(function() { Plotly.Plots.resize("map"); }, 50);
    });
    document.addEventListener("keydown", function(e) {
      if (e.key === "Escape" && mapPanel.classList.contains("is-fullscreen")) {
        mapPanel.classList.remove("is-fullscreen");
        fullscreenBtn.innerHTML = "&#x26F6; Fullscreen";
        setTimeout(function() { Plotly.Plots.resize("map"); }, 50);
      }
    });

    // ---- Overlay layers ----
    const OVERLAYS = __OVERLAY_JSON__;

    const overlayConfig = [
      {
        key: "pipelines",
        label: "Natural Gas Pipelines",
        legendClass: "overlay-legend",
        legendColor: "#b07020",
        build: function(data) {
          const traces = [];
          const segs = data.segments;
          // Batch segments into chunks to reduce trace count
          const BATCH = 500;
          for (let b = 0; b < segs.length; b += BATCH) {
            const batchLat = [];
            const batchLon = [];
            const batchText = [];
            const end = Math.min(b + BATCH, segs.length);
            for (let i = b; i < end; i++) {
              const s = segs[i];
              const hover = (s.operator || "Unknown operator")
                + (s.name ? "<br>" + s.name : "")
                + (s.type ? "<br>Type: " + s.type : "");
              for (let j = 0; j < s.lat.length; j++) {
                batchLat.push(s.lat[j]);
                batchLon.push(s.lon[j]);
                batchText.push(hover);
              }
              batchLat.push(null);
              batchLon.push(null);
              batchText.push("");
            }
            traces.push({
              type: "scattergeo",
              lat: batchLat,
              lon: batchLon,
              text: batchText,
              hovertemplate: "%{text}<extra>Pipeline</extra>",
              mode: "lines",
              line: { color: "rgba(176, 112, 32, 0.5)", width: 1.2 },
              visible: true,
              showlegend: false,
            });
          }
          return traces;
        },
      },
      {
        key: "epa_zones",
        label: "EPA Nonattainment",
        legendClass: "overlay-legend area",
        legendColor: "#c04040",
        build: function(data) {
          const traces = [];
          const ozoneLat = [], ozoneLon = [], ozoneText = [];
          const pmLat = [], pmLon = [], pmText = [];
          for (const zone of data.zones) {
            const tLat = zone.pollutant.startsWith("ozone") ? ozoneLat : pmLat;
            const tLon = zone.pollutant.startsWith("ozone") ? ozoneLon : pmLon;
            const tText = zone.pollutant.startsWith("ozone") ? ozoneText : pmText;
            for (const ring of zone.rings) {
              for (let j = 0; j < ring.lat.length; j++) {
                tLat.push(ring.lat[j]);
                tLon.push(ring.lon[j]);
                tText.push((zone.name || "Nonattainment") + " (" + zone.pollutant.replace(/_/g, " ") + (zone.classification ? ", " + zone.classification : "") + ")");
              }
              tLat.push(ring.lat[0]);
              tLon.push(ring.lon[0]);
              tText.push("");
              tLat.push(null);
              tLon.push(null);
              tText.push("");
            }
          }
          if (ozoneLat.length > 0) {
            traces.push({
              type: "scattergeo",
              lat: ozoneLat, lon: ozoneLon, text: ozoneText,
              hovertemplate: "%{text}<extra></extra>",
              mode: "lines",
              line: { color: "rgba(192, 64, 64, 0.6)", width: 1.5 },
              visible: true, showlegend: false,
            });
          }
          if (pmLat.length > 0) {
            traces.push({
              type: "scattergeo",
              lat: pmLat, lon: pmLon, text: pmText,
              hovertemplate: "%{text}<extra></extra>",
              mode: "lines",
              line: { color: "rgba(160, 80, 160, 0.6)", width: 1.5 },
              visible: true, showlegend: false,
            });
          }
          return traces;
        },
      },
      {
        key: "fiber",
        label: "Data Center / IXP Facilities",
        legendClass: "overlay-legend dot",
        legendColor: "#2070c0",
        build: function(data) {
          const pts = data.points;
          return [{
            type: "scattergeo",
            lat: pts.map(p => p.lat),
            lon: pts.map(p => p.lon),
            text: pts.map(p => {
              let h = "<b>" + (p.name || "Facility") + "</b>";
              h += "<br>" + (p.city || "") + (p.state ? ", " + p.state : "");
              if (p.org) h += "<br>Operator: " + p.org;
              if (p.net_count) h += "<br>Connected networks: " + p.net_count;
              if (p.ix_count) h += "<br>Internet exchanges: " + p.ix_count;
              return h;
            }),
            hovertemplate: "%{text}<extra>DC/IXP</extra>",
            mode: "markers",
            marker: {
              symbol: "diamond",
              size: 5,
              color: "rgba(32, 112, 192, 0.7)",
              line: { color: "rgba(255,255,255,0.5)", width: 0.5 },
            },
            visible: true,
            showlegend: false,
          }];
        },
      },
      {
        key: "fiber_routes",
        label: "Fiber Backbone Routes",
        legendClass: "overlay-legend",
        legendColor: "#1a8ccc",
        build: function(data) {
          const traces = [];
          const sourceStyles = {
            uniti_core: { color: "rgba(26, 140, 204, 0.5)", width: 1.5, label: "Uniti/Windstream" },
            segra:      { color: "rgba(80, 160, 50, 0.5)", width: 1.5, label: "Segra" },
            geotel:     { color: "rgba(180, 100, 200, 0.5)", width: 1.5, label: "GeoTel Multi-Carrier" },
            submarine:  { color: "rgba(26, 140, 204, 0.4)", width: 1.3, dash: "dot", label: "Submarine" },
          };
          const buckets = {};
          for (const seg of data.segments) {
            const src = seg.source || "unknown";
            if (!buckets[src]) buckets[src] = { lat: [], lon: [], text: [] };
            const b = buckets[src];
            let hover;
            if (src === "submarine") {
              hover = "<b>" + (seg.name || "Submarine cable") + "</b>";
              if (seg.owners) hover += "<br>Owners: " + seg.owners;
              if (seg.length) hover += "<br>Length: " + seg.length;
              if (seg.rfs) hover += "<br>In service: " + seg.rfs;
              if (seg.suppliers) hover += "<br>Supplier: " + seg.suppliers;
              if (seg.landing_points && seg.landing_points.length > 0) {
                const lps = seg.landing_points.slice(0, 4).join(", ");
                hover += "<br>Landings: " + lps + (seg.landing_points.length > 4 ? " +" + (seg.landing_points.length - 4) + " more" : "");
              }
            } else {
              const srcLabel = (sourceStyles[src] || {}).label || src;
              hover = "<b>" + (seg.carrier || seg.name || srcLabel + " fiber") + "</b>";
              if (seg.carrier && seg.name) hover = "<b>" + seg.carrier + "</b><br>" + seg.name;
              if (seg.type) hover += "<br>Type: " + seg.type;
              if (seg.placement) hover += "<br>Placement: " + seg.placement;
              if (seg.fiber_count) hover += "<br>Fiber count: " + seg.fiber_count;
              if (seg.ownership) hover += "<br>Ownership: " + seg.ownership;
              hover += "<br><i>" + srcLabel + "</i>";
            }
            for (let j = 0; j < seg.lat.length; j++) {
              b.lat.push(seg.lat[j]);
              b.lon.push(seg.lon[j]);
              b.text.push(hover);
            }
            b.lat.push(null);
            b.lon.push(null);
            b.text.push("");
          }
          for (const [src, b] of Object.entries(buckets)) {
            if (b.lat.length === 0) continue;
            const style = sourceStyles[src] || { color: "rgba(26, 140, 204, 0.5)", width: 1.5 };
            const traceObj = {
              type: "scattergeo",
              lat: b.lat, lon: b.lon, text: b.text,
              hovertemplate: "%{text}<extra></extra>",
              mode: "lines",
              line: { color: style.color, width: style.width },
              visible: true, showlegend: false,
            };
            if (style.dash) traceObj.line.dash = style.dash;
            traces.push(traceObj);
          }
          return traces;
        },
      },
    ];

    // Map overlay keys to their data payloads
    const overlayDataMap = {};
    if (OVERLAYS.pipelines) overlayDataMap["pipelines"] = OVERLAYS.pipelines;
    if (OVERLAYS.epa_zones) overlayDataMap["epa_zones"] = OVERLAYS.epa_zones;
    if (OVERLAYS.fiber) overlayDataMap["fiber"] = OVERLAYS.fiber;
    if (OVERLAYS.fiber_routes) overlayDataMap["fiber_routes"] = OVERLAYS.fiber_routes;

    // Build overlay traces and register toggles
    const overlayTraceIndices = {}; // key → [traceIndex, ...]
    let nextTraceIndex = 1; // trace 0 is reliability

    const toggleContainer = document.getElementById("overlayToggles");
    const overlaySection = document.getElementById("overlaySection");
    overlaySection.style.display = "block";

    // Reliability layer toggle (trace 0)
    {
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.id = "overlay_reliability";
      const legend = document.createElement("span");
      legend.className = "overlay-legend dot";
      legend.style.background = "#0f6b58";
      const text = document.createTextNode(" Reliability Sites");
      label.appendChild(cb);
      label.appendChild(legend);
      label.appendChild(text);
      toggleContainer.appendChild(label);
      cb.addEventListener("change", function() {
        Plotly.restyle("map", { visible: cb.checked }, [0]);
      });
    }

    const additionalTraces = [];
    let hasAnyOverlay = false;

    for (const cfg of overlayConfig) {
      const data = overlayDataMap[cfg.key];
      if (!data) continue;
      hasAnyOverlay = true;
      const traces = cfg.build(data);
      const indices = [];
      for (const t of traces) {
        additionalTraces.push(t);
        indices.push(nextTraceIndex++);
      }
      overlayTraceIndices[cfg.key] = indices;

      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.id = "overlay_" + cfg.key;
      const legend = document.createElement("span");
      legend.className = cfg.legendClass;
      legend.style.background = cfg.legendColor;
      const text = document.createTextNode(" " + cfg.label);
      label.appendChild(cb);
      label.appendChild(legend);
      label.appendChild(text);
      toggleContainer.appendChild(label);

      cb.addEventListener("change", function() {
        for (const idx of indices) {
          Plotly.restyle("map", { visible: cb.checked }, [idx]);
        }
      });
    }

    if (hasAnyOverlay) {
      Plotly.addTraces("map", additionalTraces);
    }
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interactive U.S. reliability map HTML.")
    parser.add_argument("--cache", type=Path, default=Path("outputs/us_reliability_map_cache.json"))
    parser.add_argument("--fallback-cache", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/assumptions_2026_us_ai_datacenter.yaml"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("outputs/overlays"))
    parser.add_argument("--out", type=Path, default=Path("outputs/interactive_reliability_map.html"))
    return parser.parse_args()


def mean_value(node: dict) -> float:
    value = node.get("value", node)
    return float(value["mean"])


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_signature(payload: dict) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    axes = payload["axes"]
    return (
        tuple(float(v) for v in axes["solar_mw"]),
        tuple(float(v) for v in axes["wind_mw"]),
        tuple(float(v) for v in axes["bess_mwh"]),
    )


def merge_payloads(primary: dict, fallback: dict) -> dict:
    if _axis_signature(primary) != _axis_signature(fallback):
        raise ValueError("Primary and fallback cache axes do not match.")
    fallback_sites = fallback["sites"]
    primary_sites_by_id = {int(site["site_id"]): site for site in primary["sites"]}
    fallback_site_ids = [int(site["site_id"]) for site in fallback_sites]
    missing_primary_ids = set(primary_sites_by_id) - set(fallback_site_ids)
    if missing_primary_ids:
        raise ValueError(f"Primary cache contains site_ids absent from fallback cache: {sorted(missing_primary_ids)[:10]}")

    merged_sites = []
    for site_id, fallback_site in zip(fallback_site_ids, fallback_sites):
        if site_id in primary_sites_by_id:
            site = dict(primary_sites_by_id[site_id])
            site["resource_source"] = "real"
            site["resource_source_label"] = "Real NSRDB/WTK"
        else:
            site = dict(fallback_site)
            site["resource_source"] = "fallback"
            site["resource_source_label"] = "Synthetic fallback"
        merged_sites.append(site)
    fallback_index_by_id = {site_id: idx for idx, site_id in enumerate(fallback_site_ids)}
    primary_index_by_id = {int(site["site_id"]): idx for idx, site in enumerate(primary["sites"])}

    merged_values_by_combo: list[list[float]] = []
    for combo_index, fallback_slice in enumerate(fallback["values_by_combo"]):
        primary_slice = primary["values_by_combo"][combo_index]
        merged_slice = list(fallback_slice)
        for site_id, primary_site_index in primary_index_by_id.items():
            merged_slice[fallback_index_by_id[site_id]] = primary_slice[primary_site_index]
        merged_values_by_combo.append(merged_slice)

    merged_meta = dict(fallback["meta"])
    merged_meta.update(
        {
            "model": f"{primary['meta'].get('model', 'primary')}+fallback",
            "resource_mode": f"{primary['meta'].get('resource_mode', 'primary')}+fallback",
            "site_count": len(merged_sites),
            "resource_years": primary["meta"].get("resource_years", fallback["meta"].get("resource_years", {})),
            "real_site_count": len(primary["sites"]),
            "primary_cache": primary["meta"].get("site_source", ""),
            "fallback_cache": fallback["meta"].get("site_source", ""),
            "primary_site_count": len(primary["sites"]),
            "fallback_site_count": len(fallback["sites"]),
        }
    )
    return {
        "meta": merged_meta,
        "axes": fallback["axes"],
        "sites": merged_sites,
        "values_by_combo": merged_values_by_combo,
    }


def annotate_site_metadata(payload: dict) -> dict:
    if any("resource_source" in site for site in payload["sites"]):
        return payload
    resource_mode = str(payload["meta"].get("resource_mode", "synthetic"))
    if resource_mode == "real":
        source = ("real", "Real NSRDB/WTK")
    else:
        source = ("synthetic", "Synthetic")
    annotated_sites = []
    for site in payload["sites"]:
        annotated = dict(site)
        annotated["resource_source"] = source[0]
        annotated["resource_source_label"] = source[1]
        annotated_sites.append(annotated)
    payload = dict(payload)
    payload["sites"] = annotated_sites
    return payload


def build_method_side_html(payload: dict, config: dict) -> str:
    meta = payload["meta"]
    axes = payload["axes"]
    bess_cfg = config["technology_parameters"]["bess_li_ion"]
    rte = mean_value(bess_cfg["round_trip_efficiency"])
    dod = mean_value(bess_cfg["usable_depth_of_discharge"])
    eta = math.sqrt(rte)

    symbols = [
        (r"\(G_t\)", r"Renewable generation at hour \(t\)."),
        (r"\(S_t\)", r"Solar output trace per installed MW at hour \(t\)."),
        (r"\(W_t\)", r"Wind output trace per installed MW at hour \(t\)."),
        (r"\(C_{\mathrm{solar}}\)", r"Installed solar capacity selected by the slider."),
        (r"\(C_{\mathrm{wind}}\)", r"Installed wind capacity selected by the slider."),
        (r"\(L\)", r"Fixed AIDC load. In this map, \(L = 1{,}000\,\mathrm{MW}\)."),
        (r"\(\mathrm{SOC}_t\)", r"Battery state of charge at hour \(t\)."),
        (r"\(E_{\max}\)", r"Installed BESS energy capacity selected by the slider."),
        (r"\(\eta_c, \eta_d\)", r"Charge and discharge efficiencies, both set from the battery round-trip efficiency."),
        (r"\(\mathrm{served}_t\)", r"Indicator equal to 1 if the full load is served in hour \(t\), else 0."),
    ]

    assumptions = [
        ("Workload", f"{meta['workload_mw']:,.0f} MW constant load", "Fixed for all sites and all hours in the map product."),
        ("Time resolution", f"{int(meta['hours'])} hourly steps", "Reliability is computed over one synthetic 8760-hour year."),
        ("Solar range", f"{int(min(axes['solar_mw'])):,} to {int(max(axes['solar_mw'])):,} MW", "Cached every 600 MW and interpolated in-browser."),
        ("Wind range", f"{int(min(axes['wind_mw'])):,} to {int(max(axes['wind_mw'])):,} MW", "Cached every 600 MW and interpolated in-browser."),
        ("BESS range", f"{int(min(axes['bess_mwh'])):,} to {int(max(axes['bess_mwh'])):,} MWh", "Cached every 20,000 MWh and interpolated in-browser."),
        ("Round-trip efficiency", f"{rte:.2f}", f"Implies \\(\\eta_c = \\eta_d \\approx {eta:.3f}\\)."),
        ("Usable depth of discharge", f"{dod:.2f}", f"Sets \\(\\mathrm{{SOC}}_{{\\min}} = (1 - {dod:.2f}) E_{{\\max}}\\)."),
        ("Battery interpretation", str(meta["battery_interpretation"]).replace("_", " "), "BESS slider controls energy only in this version."),
        ("Battery power assumption", str(meta["battery_power_assumption"]).replace("_", " "), "Battery power is assumed non-binding up to workload."),
        ("Site set", f"{meta['site_count']} deduped town points", "Built from GeoNames populated places for the lower 48 + DC."),
        ("Resource model", str(meta["model"]).replace("_", " "), "Real cached site profiles override the fallback cache where available."),
    ]

    def render_card(title: str, items: list[tuple[str, str, str]] | list[tuple[str, str]]) -> str:
        blocks = []
        for item in items:
            if len(item) == 2:
                term, desc = item
                meta_line = ""
            else:
                term, meta_line, desc = item
            blocks.append(
                f"""
                <div class="assumption-item">
                  <div class="assumption-term">{term}</div>
                  {f'<div class="assumption-meta">{meta_line}</div>' if meta_line else ''}
                  <div class="assumption-desc">{desc}</div>
                </div>
                """
            )
        return f"""
        <div class="assumption-card">
          <h3>{title}</h3>
          <div class="assumption-list">
            {''.join(blocks)}
          </div>
        </div>
        """

    return render_card("Symbols", symbols) + render_card("Assumptions", assumptions)

def load_overlays(overlay_dir: Path) -> dict:
    """Load all overlay JSON files from the overlay directory."""
    overlays: dict = {}
    overlay_files = {
        "pipelines": "natural_gas_pipelines.json",
        "epa_zones": "epa_nonattainment_zones.json",
        "fiber": "fiber_infrastructure.json",
        "fiber_routes": "fiber_routes.json",
    }
    for key, filename in overlay_files.items():
        path = overlay_dir / filename
        if path.exists():
            overlays[key] = json.loads(path.read_text(encoding="utf-8"))
            size_mb = path.stat().st_size / 1e6
            print(f"  overlay: {key} ({size_mb:.1f} MB)")
        else:
            print(f"  overlay: {key} — not found, skipping")
    return overlays


def main() -> None:
    args = parse_args()
    payload = load_payload(args.cache)
    if args.fallback_cache is not None:
        payload = merge_payloads(payload, load_payload(args.fallback_cache))
    payload = annotate_site_metadata(payload)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    payload["meta"]["recommended_defaults"] = {
        "solar_mw": 6000.0,
        "wind_mw": 0.0,
        "bess_mwh": 40000.0,
    }

    print("Loading overlays …")
    overlays = load_overlays(args.overlay_dir)

    html = HTML_TEMPLATE
    html = html.replace("__DATA_JSON__", json.dumps(payload))
    html = html.replace("__OVERLAY_JSON__", json.dumps(overlays))
    html = html.replace("__METHOD_SIDE_HTML__", build_method_side_html(payload, config))
    html = html.replace("__CACHE_MODEL__", str(payload["meta"].get("model", "deterministic cache")).replace("_", " "))
    resource_years = payload["meta"].get("resource_years", {})
    html = html.replace("__SOLAR_YEAR__", str(resource_years.get("solar_year", "n/a")))
    html = html.replace("__WIND_YEAR__", str(resource_years.get("wind_year", "n/a")))
    html = html.replace("__REAL_SITE_COUNT__", str(int(payload["meta"].get("real_site_count", payload["meta"].get("site_count", 0)))))
    workload_mw = float(payload["meta"]["workload_mw"])
    step_1pct = max(1, int(round(workload_mw * 0.01)))
    html = html.replace("__SOLAR_SLIDER_MAX__", str(max(10000, int(max(payload["axes"]["solar_mw"])))))
    html = html.replace("__WIND_SLIDER_MAX__", str(int(max(payload["axes"]["wind_mw"]))))
    html = html.replace("__BESS_SLIDER_MAX__", str(int(max(payload["axes"]["bess_mwh"]))))
    html = html.replace("__SOLAR_SLIDER_STEP__", str(step_1pct))
    html = html.replace("__WIND_SLIDER_STEP__", str(step_1pct))
    html = html.replace("__BESS_SLIDER_STEP__", str(step_1pct))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"Output: {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
