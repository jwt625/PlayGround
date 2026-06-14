import { parseSimpleYaml } from "./core/config.js";
import { parseElectrodes } from "./core/geometry.js";
import { parseMaterials } from "./core/materials.js";
import { getModePlotValues, solveOpticalModeConfig } from "./core/mode_solver.js";
import { quantityInfo } from "./core/quantities.js";
import { getPlotValues, solveConfig } from "./core/solver.js";
import { isOpticalMode, validateConfig } from "./core/validation.js";

const workspace = document.querySelector(".workspace");
const configEditor = document.querySelector("#config-editor");
const results = document.querySelector("#results");
const statusLine = document.querySelector("#status");
const canvas = document.querySelector("#field-canvas");
const ctx = canvas.getContext("2d");
const colorbarCanvas = document.querySelector("#colorbar-canvas");
const colorbarCtx = colorbarCanvas.getContext("2d");
const colorbarMax = document.querySelector("#colorbar-max");
const colorbarMid = document.querySelector("#colorbar-mid");
const colorbarMin = document.querySelector("#colorbar-min");
const quantitySelect = document.querySelector("#quantity-select");
const scaleSelect = document.querySelector("#scale-select");
const meshToggle = document.querySelector("#mesh-toggle");
const meshNxInput = document.querySelector("#mesh-nx-input");
const meshNyInput = document.querySelector("#mesh-ny-input");
const panelResizer = document.querySelector("#panel-resizer");
const resetViewButton = document.querySelector("#reset-view-button");
const solveProgress = document.querySelector("#solve-progress");
const plotTooltip = document.querySelector("#plot-tooltip");
const logWindow = document.querySelector("#log-window");

const examplePaths = {
  parallel: "../examples/parallel_plate.yaml",
  cylinders: "../examples/two_cylinders.yaml",
  stack: "../examples/material_stack.yaml",
  tfln: "../examples/tfln_partial_etched_mzm.yaml",
  bto: "../examples/bto_on_sin_plasmonic.yaml",
  siMode: "../examples/si_strip_mode.yaml",
};

let solveTimer = null;
let lastConfig = null;
let lastResult = null;
let lastDomainKey = null;
let view = null;
let activeScale = null;
let isPanning = false;
let panStart = null;
let solveRunId = 0;

document.querySelector("#solve-button").addEventListener("click", runSolve);
document.querySelector("#parallel-button").addEventListener("click", () => loadExample("parallel"));
document.querySelector("#cylinders-button").addEventListener("click", () => loadExample("cylinders"));
document.querySelector("#stack-button").addEventListener("click", () => loadExample("stack"));
document.querySelector("#tfln-button").addEventListener("click", () => loadExample("tfln"));
document.querySelector("#bto-button").addEventListener("click", () => loadExample("bto"));
document.querySelector("#si-mode-button").addEventListener("click", () => loadExample("siMode"));
quantitySelect.addEventListener("change", redrawLastResult);
scaleSelect.addEventListener("change", redrawLastResult);
meshToggle.addEventListener("change", redrawLastResult);
meshNxInput.addEventListener("change", () => updateMeshValue("mesh_nx", meshNxInput.value));
meshNyInput.addEventListener("change", () => updateMeshValue("mesh_ny", meshNyInput.value));
panelResizer.addEventListener("pointerdown", startPanelResize);
resetViewButton.addEventListener("click", () => {
  if (lastResult) {
    resetViewport(lastResult.mesh);
    redrawLastResult();
  }
});
canvas.addEventListener("wheel", zoomCanvas, { passive: false });
canvas.addEventListener("pointerdown", startCanvasPan);
canvas.addEventListener("pointermove", handleCanvasPointerMove);
canvas.addEventListener("pointerup", endCanvasPan);
canvas.addEventListener("pointercancel", endCanvasPan);
canvas.addEventListener("pointerleave", () => {
  if (!isPanning) plotTooltip.hidden = true;
});
canvas.addEventListener("dblclick", () => {
  if (lastResult) {
    resetViewport(lastResult.mesh);
    redrawLastResult();
  }
});
configEditor.addEventListener("input", () => {
  syncMeshInputsFromEditor();
  clearTimeout(solveTimer);
  solveTimer = setTimeout(runSolve, 500);
});

new ResizeObserver(() => {
  resizeCanvases();
  redrawLastResult();
}).observe(canvas);

await loadExample("parallel");

async function loadExample(name) {
  appendLog("info", `Loading example: ${name}`);
  const response = await fetch(`${examplePaths[name]}?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) {
    appendLog("error", `Failed to load example ${name}: HTTP ${response.status}`);
    return;
  }
  configEditor.value = await response.text();
  syncMeshInputsFromEditor();
  runSolve();
}

function runSolve() {
  const runId = (solveRunId += 1);
  const t0 = performance.now();
  setSolverStatus("Validating");
  appendLog("info", `Run ${runId}: validation started`);
  solveProgress.classList.add("is-active");
  solveProgress.removeAttribute("value");
  requestAnimationFrame(() => {
    let config;
    try {
      if (runId !== solveRunId) return;
      config = parseSimpleYaml(configEditor.value);
      validateConfig(config);
      appendLog("success", `Run ${runId}: validation passed`);
      setSolverStatus("Solving");
      appendLog(
        "info",
        `Run ${runId}: solve started (${config.Simulation?.mesh_nx ?? 81} x ${
          config.Simulation?.mesh_ny ?? 61
        } mesh)`,
      );
      renderResultsPending(config, "Solving");
      drawWorkInProgress(`Solving ${config.Simulation?.mesh_nx ?? 81} x ${config.Simulation?.mesh_ny ?? 61} mesh...`);
      runAfterPaint(() => finishSolveRun(runId, t0, config));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSolverStatus("Validation failed");
      appendLog("error", `Run ${runId}: ${message}`);
      renderResultsMessage("Validation failed", message);
      clearCanvas();
      solveProgress.classList.remove("is-active");
    }
  });
}

function finishSolveRun(runId, t0, config) {
  if (runId !== solveRunId) return;
  try {
      const result = isOpticalMode(config) ? solveOpticalModeConfig(config) : solveConfig(config);
    const elapsed = performance.now() - t0;
    lastConfig = config;
    lastResult = result;
    const domainKey = makeDomainKey(result.mesh.domain);
    if (!view || domainKey !== lastDomainKey) resetViewport(result.mesh);
    lastDomainKey = domainKey;
    syncMeshInputs(config);
    renderResults(result, elapsed);
    renderPlot(config, result);
    setSolverStatus(`Solved in ${elapsed.toFixed(0)} ms`);
    appendLog(
      "success",
      `Run ${runId}: solved in ${elapsed.toFixed(0)} ms; ${formatRunSummary(result)}`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setSolverStatus("Error");
    appendLog("error", `Run ${runId}: ${message}`);
    renderResultsMessage("Solver error", message);
    clearCanvas();
  } finally {
    solveProgress.classList.remove("is-active");
  }
}

function runAfterPaint(callback) {
  requestAnimationFrame(() => {
    setTimeout(callback, 0);
  });
}

function setSolverStatus(message) {
  statusLine.textContent = message;
}

function appendLog(level, message) {
  const row = document.createElement("div");
  row.className = `log-row ${level}`;
  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = timestamp();
  const levelCell = document.createElement("span");
  levelCell.className = "log-level";
  levelCell.textContent = level.toUpperCase();
  const messageCell = document.createElement("span");
  messageCell.className = "log-message";
  messageCell.textContent = message;
  row.append(time, levelCell, messageCell);
  logWindow.append(row);
  while (logWindow.children.length > 120) logWindow.firstElementChild.remove();
  logWindow.scrollTop = logWindow.scrollHeight;
}

function timestamp() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}

function updateMeshValue(key, rawValue) {
  const value = Math.max(3, Math.min(401, Math.round(Number(rawValue))));
  if (!Number.isFinite(value)) return;
  configEditor.value = setYamlScalar(configEditor.value, key, value);
  syncMeshInputsFromEditor();
  runSolve();
}

function setYamlScalar(text, key, value) {
  const lines = text.split(/\r?\n/);
  let inSimulation = false;
  let simulationIndent = -1;
  let insertAt = -1;
  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const trimmed = raw.trim();
    const indent = raw.length - raw.trimStart().length;
    if (trimmed === "Simulation:") {
      inSimulation = true;
      simulationIndent = indent;
      insertAt = i + 1;
      continue;
    }
    if (inSimulation && trimmed && indent <= simulationIndent) {
      lines.splice(insertAt, 0, `  ${key}: ${value}`);
      return lines.join("\n");
    }
    if (inSimulation) {
      insertAt = i + 1;
      if (trimmed.startsWith(`${key}:`)) {
        lines[i] = `${" ".repeat(indent)}${key}: ${value}`;
        return lines.join("\n");
      }
    }
  }
  if (inSimulation) {
    lines.splice(insertAt, 0, `  ${key}: ${value}`);
    return lines.join("\n");
  }
  return `Simulation:\n  ${key}: ${value}\n${text}`;
}

function syncMeshInputsFromEditor() {
  try {
    syncMeshInputs(parseSimpleYaml(configEditor.value));
  } catch {
    return;
  }
}

function syncMeshInputs(config) {
  meshNxInput.value = String(config.Simulation?.mesh_nx ?? 81);
  meshNyInput.value = String(config.Simulation?.mesh_ny ?? 61);
}

function startPanelResize(event) {
  event.preventDefault();
  panelResizer.setPointerCapture(event.pointerId);
  const rect = workspace.getBoundingClientRect();
  const onMove = (moveEvent) => {
    const fraction = clamp((moveEvent.clientX - rect.left) / rect.width, 0.28, 0.72);
    workspace.style.setProperty("--editor-width", `${(100 * fraction).toFixed(1)}%`);
  };
  const onUp = () => {
    panelResizer.removeEventListener("pointermove", onMove);
    panelResizer.removeEventListener("pointerup", onUp);
    panelResizer.removeEventListener("pointercancel", onUp);
  };
  panelResizer.addEventListener("pointermove", onMove);
  panelResizer.addEventListener("pointerup", onUp);
  panelResizer.addEventListener("pointercancel", onUp);
}

function renderResults(result, elapsed) {
  if (result.physics === "optical_mode") {
    renderOpticalResults(result, elapsed);
    return;
  }
  const lines = [
    ["C energy", `${result.capacitanceEnergy.toExponential(6)} F/m`],
    ["C energy", `${result.units.fF_per_mm.toFixed(4)} fF/mm`],
    ["C energy", `${result.units.pF_per_cm.toFixed(4)} pF/cm`],
    ["C charge", `${result.capacitanceCharge.toExponential(6)} F/m`],
    ["Epsilon", result.permittivityModel],
    ["Mesh", formatMeshSummary(result.mesh)],
    ["dx", `${formatValue(result.mesh.stats.minDx)} to ${formatValue(result.mesh.stats.maxDx)} m`],
    ["dy", `${formatValue(result.mesh.stats.minDy)} to ${formatValue(result.mesh.stats.maxDy)} m`],
    ["CG", `${result.iterations} iter, residual ${result.residual.toExponential(3)}`],
    ["Runtime", `${elapsed.toFixed(0)} ms`],
  ];
  if (result.reference) {
    const err =
      (result.capacitanceEnergy - result.reference.capacitance) / result.reference.capacitance;
    lines.push(
      ["Reference", `${result.reference.capacitance.toExponential(6)} F/m`],
      ["Error", `${(100 * err).toFixed(2)}%`],
    );
  }
  renderResultLines(lines);
}

function renderOpticalResults(result, elapsed) {
  const mode = result.mode;
  renderResultLines([
    ["n_eff", mode.nEff.toFixed(6)],
    ["beta", `${mode.beta.toExponential(6)} 1/m`],
    ["lambda", `${formatValue(result.wavelength)} m`],
    ["Confinement", `${(100 * mode.confinement).toFixed(2)}%`],
    ["Mode area", `${mode.modeArea.toExponential(6)} m^2`],
    ["Model", result.scalarModel],
    ["Mesh", formatMeshSummary(result.mesh)],
    ["Eigen solve", `${mode.iterations} iter, residual ${mode.residual.toExponential(3)}`],
    ["Runtime", `${elapsed.toFixed(0)} ms`],
  ]);
}

function renderResultsPending(config, state) {
  const nx = config.Simulation?.mesh_nx ?? 81;
  const ny = config.Simulation?.mesh_ny ?? 61;
  if (isOpticalMode(config)) {
    renderResultLines([
      ["n_eff", "..."],
      ["beta", "..."],
      ["lambda", `${formatValue(config.Simulation?.wavelength ?? 1.55e-6)} m`],
      ["Confinement", "..."],
      ["Mode area", "..."],
      ["Model", "scalar optical mode"],
      ["Mesh", `${nx} x ${ny}`],
      ["Eigen solve", "..."],
      ["Runtime", "..."],
      ["Status", state],
    ]);
    return;
  }
  renderResultLines([
    ["C energy", "..."],
    ["C energy", "... fF/mm"],
    ["C energy", "... pF/cm"],
    ["C charge", "..."],
    ["Epsilon", "..."],
    ["Mesh", `${nx} x ${ny}`],
    ["CG", "..."],
    ["Runtime", "..."],
    ["Status", state],
  ]);
}

function renderResultsMessage(title, message) {
  renderResultLines([
    ["C energy", "-"],
    ["C energy", "-"],
    ["C energy", "-"],
    ["C charge", "-"],
    ["Epsilon", "-"],
    ["Mesh", "-"],
    ["CG", "-"],
    ["Runtime", "-"],
    [title, message],
  ]);
}

function renderResultLines(lines) {
  results.innerHTML = lines
    .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function redrawLastResult() {
  if (lastConfig && lastResult) renderPlot(lastConfig, lastResult);
}

function renderPlot(config, result) {
  syncQuantityOptions(result);
  resizeCanvases();
  if (!view) resetViewport(result.mesh);
  const { nx, ny } = result.mesh;
  const values =
    result.physics === "optical_mode"
      ? getModePlotValues(result, quantitySelect.value)
      : getPlotValues(result, quantitySelect.value);
  const transform = makeScale(values, scaleSelect.value);
  activeScale = transform;
  const image = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      const source = j * nx + i;
      const target = ((ny - 1 - j) * nx + i) * 4;
      const color = divergingColor(transform.map(values[source]));
      image.data[target] = color[0];
      image.data[target + 1] = color[1];
      image.data[target + 2] = color[2];
      image.data[target + 3] = 255;
    }
  }
  const offscreen = document.createElement("canvas");
  offscreen.width = nx;
  offscreen.height = ny;
  offscreen.getContext("2d").putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const p0 = toCanvas(result.mesh, [result.mesh.domain.xMin, result.mesh.domain.yMin]);
  const p1 = toCanvas(result.mesh, [result.mesh.domain.xMax, result.mesh.domain.yMax]);
  ctx.drawImage(offscreen, p0[0], p1[1], p1[0] - p0[0], p0[1] - p1[1]);
  drawMaterialBoundaries(config, result);
  if (result.physics !== "optical_mode") drawElectrodes(config, result);
  if (meshToggle.checked) drawMesh(result.mesh);
  renderColorbar(transform);
  updateQuantityDescription();
}

function syncQuantityOptions(result) {
  const optical = result.physics === "optical_mode";
  const desired = optical
    ? [
        ["mode", "mode - scalar modal field"],
        ["mode_abs", "|mode| - modal field magnitude"],
        ["mode_intensity", "I - modal intensity"],
        ["n", "n - optical refractive index"],
        ["eps_r", "epsilon_r - relative permittivity"],
      ]
    : [
        ["phi", "phi - electrostatic potential"],
        ["Ex", "Ex - x electric-field component"],
        ["Ey", "Ey - y electric-field component"],
        ["normE", "|E| - electric-field magnitude"],
        ["eps_r", "epsilon_r - relative permittivity"],
        ["eps_r_xx", "epsilon_r_xx - permittivity tensor xx"],
        ["eps_r_yy", "epsilon_r_yy - permittivity tensor yy"],
        ["eps_r_xy", "epsilon_r_xy - permittivity tensor xy"],
        ["r13", "r13 - EO tensor coefficient"],
        ["r33", "r33 - EO tensor coefficient"],
        ["r22", "r22 - EO tensor coefficient"],
        ["r_eff", "r_eff - effective EO coefficient"],
      ];
  const currentValues = Array.from(quantitySelect.options).map((option) => option.value).join("|");
  const nextValues = desired.map(([value]) => value).join("|");
  if (currentValues === nextValues) return;
  const previous = quantitySelect.value;
  quantitySelect.innerHTML = "";
  for (const [value, label] of desired) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    quantitySelect.append(option);
  }
  quantitySelect.value = desired.some(([value]) => value === previous) ? previous : desired[0][0];
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  colorbarCtx.clearRect(0, 0, colorbarCanvas.width, colorbarCanvas.height);
  colorbarMax.textContent = "";
  colorbarMid.textContent = "";
  colorbarMin.textContent = "";
  plotTooltip.hidden = true;
}

function drawWorkInProgress(message) {
  resizeCanvases();
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0f141b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#2a3441";
  ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
  ctx.fillStyle = "#9aa7b5";
  ctx.font = `${14 * (window.devicePixelRatio || 1)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
  ctx.restore();
  colorbarCtx.clearRect(0, 0, colorbarCanvas.width, colorbarCanvas.height);
  colorbarMax.textContent = "";
  colorbarMid.textContent = "WIP";
  colorbarMin.textContent = "";
  plotTooltip.hidden = true;
}

function makeScale(values, mode) {
  const finite = Array.from(values).filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return { mode, min: 0, mid: 0, max: 1, map: () => 0.5 };
  }
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (mode === "symmetric") {
    const limit = Math.max(Math.abs(min), Math.abs(max)) || 1;
    min = -limit;
    max = limit;
    return {
      mode,
      min,
      mid: 0,
      max,
      map: (value) => 0.5 + 0.5 * clamp(value / limit, -1, 1),
    };
  }
  if (mode === "log") {
    const magnitudes = finite.map((value) => Math.abs(value)).filter((value) => value > 0);
    const logMin = magnitudes.length > 0 ? Math.log10(Math.min(...magnitudes)) : -30;
    const logMax = magnitudes.length > 0 ? Math.log10(Math.max(...magnitudes)) : 0;
    const span = logMax - logMin || 1;
    return {
      mode,
      min: 10 ** logMin,
      mid: 10 ** ((logMin + logMax) / 2),
      max: 10 ** logMax,
      map: (value) => clamp((Math.log10(Math.max(Math.abs(value), 10 ** logMin)) - logMin) / span, 0, 1),
    };
  }
  const span = max - min || 1;
  return {
    mode,
    min,
    mid: (min + max) / 2,
    max,
    map: (value) => clamp((value - min) / span, 0, 1),
  };
}

function renderColorbar(scale) {
  const image = colorbarCtx.createImageData(colorbarCanvas.width, colorbarCanvas.height);
  for (let y = 0; y < colorbarCanvas.height; y += 1) {
    const t = 1 - y / (colorbarCanvas.height - 1);
    const color = divergingColor(t);
    for (let x = 0; x < colorbarCanvas.width; x += 1) {
      const idx = (y * colorbarCanvas.width + x) * 4;
      image.data[idx] = color[0];
      image.data[idx + 1] = color[1];
      image.data[idx + 2] = color[2];
      image.data[idx + 3] = 255;
    }
  }
  colorbarCtx.putImageData(image, 0, 0);
  colorbarMax.textContent = formatValue(scale.max);
  colorbarMid.textContent = formatValue(scale.mid);
  colorbarMin.textContent = formatValue(scale.min);
}

function drawMesh(mesh) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.lineWidth = 0.55;
  for (const tri of mesh.triangles) {
    ctx.beginPath();
    for (let k = 0; k < tri.length; k += 1) {
      const [x, y] = toCanvas(mesh, mesh.nodes[tri[k]]);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

function drawMaterialBoundaries(config, result) {
  const materials = parseMaterials(config).filter((material) => material.shape !== "background");
  ctx.save();
  ctx.lineJoin = "miter";
  ctx.setLineDash([6, 4]);
  for (const material of materials) {
    drawShapePath(result.mesh, material.shape, material.params);
    ctx.strokeStyle = "rgba(0,0,0,0.82)";
    ctx.lineWidth = 3.0;
    ctx.stroke();
    drawShapePath(result.mesh, material.shape, material.params);
    ctx.strokeStyle = "rgba(255,255,255,0.88)";
    ctx.lineWidth = 1.35;
    ctx.stroke();
  }
  ctx.restore();
}

function drawElectrodes(config, result) {
  const electrodes = parseElectrodes(config);
  ctx.save();
  for (const electrode of electrodes) {
    ctx.fillStyle =
      electrode.potential > 0 ? "rgba(255,255,255,0.72)" : "rgba(0,0,0,0.58)";
    ctx.strokeStyle = electrode.potential > 0 ? "rgba(20,20,20,0.9)" : "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1.25;
    drawShapePath(result.mesh, electrode.shape, electrode.params);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function formatRunSummary(result) {
  if (result.physics === "optical_mode") {
    return `${result.scalarModel}; ${formatMeshSummary(result.mesh)}; n_eff ${result.mode.nEff.toFixed(6)}; eig ${result.mode.iterations} iter; residual ${result.mode.residual.toExponential(3)}`;
  }
  return `${result.permittivityModel}; ${formatMeshSummary(result.mesh)}; CG ${result.iterations} iter; residual ${result.residual.toExponential(3)}`;
}

function drawShapePath(mesh, shape, params) {
  const p = params;
  ctx.beginPath();
  if (shape === "rectangle") {
    const p0 = toCanvas(mesh, [p.x_min, p.y_min]);
    const p1 = toCanvas(mesh, [p.x_max, p.y_max]);
    ctx.rect(p0[0], p1[1], p1[0] - p0[0], p0[1] - p1[1]);
  } else if (shape === "circle") {
    const center = toCanvas(mesh, [p.x, p.y]);
    const edge = toCanvas(mesh, [p.x + p.radius, p.y]);
    ctx.arc(center[0], center[1], Math.abs(edge[0] - center[0]), 0, 2 * Math.PI);
  }
}

function toCanvas(mesh, point) {
  const currentView = view ?? mesh.domain;
  const [x, y] = point;
  const px = ((x - currentView.xMin) / (currentView.xMax - currentView.xMin)) * canvas.width;
  const py =
    canvas.height -
    ((y - currentView.yMin) / (currentView.yMax - currentView.yMin)) * canvas.height;
  return [px, py];
}

function canvasToWorld(mesh, px, py) {
  const currentView = view ?? mesh.domain;
  const x = currentView.xMin + (px / canvas.width) * (currentView.xMax - currentView.xMin);
  const y =
    currentView.yMax - (py / canvas.height) * (currentView.yMax - currentView.yMin);
  return [x, y];
}

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.round(rect.width * dpr));
  const height = Math.max(220, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const colorbarRect = colorbarCanvas.getBoundingClientRect();
  const cbWidth = Math.max(16, Math.round(colorbarRect.width * dpr));
  const cbHeight = Math.max(24, Math.round(colorbarRect.height * dpr));
  if (colorbarCanvas.width !== cbWidth || colorbarCanvas.height !== cbHeight) {
    colorbarCanvas.width = cbWidth;
    colorbarCanvas.height = cbHeight;
  }
}

function resetViewport(mesh) {
  view = { ...mesh.domain };
}

function makeDomainKey(domain) {
  return `${domain.xMin},${domain.xMax},${domain.yMin},${domain.yMax}`;
}

function zoomCanvas(event) {
  if (!lastResult || !view) return;
  event.preventDefault();
  const point = eventPoint(event);
  const [wx, wy] = canvasToWorld(lastResult.mesh, point.x, point.y);
  const factor = event.deltaY < 0 ? 0.82 : 1.22;
  const oldWidth = view.xMax - view.xMin;
  const oldHeight = view.yMax - view.yMin;
  const newWidth = oldWidth * factor;
  const newHeight = oldHeight * factor;
  const fx = point.x / canvas.width;
  const fy = 1 - point.y / canvas.height;
  view = {
    xMin: wx - fx * newWidth,
    xMax: wx + (1 - fx) * newWidth,
    yMin: wy - fy * newHeight,
    yMax: wy + (1 - fy) * newHeight,
  };
  clampViewport(lastResult.mesh);
  redrawLastResult();
  updateTooltip(event);
}

function startCanvasPan(event) {
  if (!lastResult || event.button !== 0) return;
  isPanning = true;
  canvas.classList.add("is-panning");
  canvas.setPointerCapture(event.pointerId);
  panStart = { point: eventPoint(event), view: { ...view } };
}

function handleCanvasPointerMove(event) {
  if (isPanning && panStart && lastResult) {
    const point = eventPoint(event);
    const dxWorld =
      ((point.x - panStart.point.x) / canvas.width) * (panStart.view.xMax - panStart.view.xMin);
    const dyWorld =
      ((point.y - panStart.point.y) / canvas.height) * (panStart.view.yMax - panStart.view.yMin);
    view = {
      xMin: panStart.view.xMin - dxWorld,
      xMax: panStart.view.xMax - dxWorld,
      yMin: panStart.view.yMin + dyWorld,
      yMax: panStart.view.yMax + dyWorld,
    };
    clampViewport(lastResult.mesh);
    redrawLastResult();
  }
  updateTooltip(event);
}

function endCanvasPan(event) {
  isPanning = false;
  canvas.classList.remove("is-panning");
  if (event.pointerId !== undefined) {
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch {
      // The pointer may already be released by the browser.
    }
  }
  panStart = null;
}

function clampViewport(mesh) {
  const domain = mesh.domain;
  const domainWidth = domain.xMax - domain.xMin;
  const domainHeight = domain.yMax - domain.yMin;
  const minWidth = domainWidth / 100;
  const minHeight = domainHeight / 100;
  let width = Math.max(view.xMax - view.xMin, minWidth);
  let height = Math.max(view.yMax - view.yMin, minHeight);
  width = Math.min(width, domainWidth * 3);
  height = Math.min(height, domainHeight * 3);
  let cx = (view.xMin + view.xMax) / 2;
  let cy = (view.yMin + view.yMax) / 2;
  const marginX = domainWidth;
  const marginY = domainHeight;
  cx = clamp(cx, domain.xMin - marginX, domain.xMax + marginX);
  cy = clamp(cy, domain.yMin - marginY, domain.yMax + marginY);
  view = {
    xMin: cx - width / 2,
    xMax: cx + width / 2,
    yMin: cy - height / 2,
    yMax: cy + height / 2,
  };
}

function updateTooltip(event) {
  if (!lastResult) return;
  const point = eventPoint(event);
  const [x, y] = canvasToWorld(lastResult.mesh, point.x, point.y);
  const mesh = lastResult.mesh;
  const i = nearestCoordinateIndex(mesh.xCoords, x);
  const j = nearestCoordinateIndex(mesh.yCoords, y);
  if (i < 0 || i >= mesh.nx || j < 0 || j >= mesh.ny) {
    plotTooltip.hidden = true;
    return;
  }
  const idx = j * mesh.nx + i;
  const values = getPlotValues(lastResult, quantitySelect.value);
  const info = quantityInfo(quantitySelect.value);
  plotTooltip.innerHTML = [
    `x = ${formatValue(x)} m`,
    `y = ${formatValue(y)} m`,
    `${info.label} = ${formatValue(values[idx])}`,
    `expr: ${info.expression} (${info.description})`,
  ].join("<br />");
  plotTooltip.hidden = false;
  const wrap = canvas.parentElement.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const cssX = canvasRect.left - wrap.left + point.cssX + 14;
  const cssY = canvasRect.top - wrap.top + point.cssY + 14;
  plotTooltip.style.left = `${Math.min(cssX, wrap.width - 230)}px`;
  plotTooltip.style.top = `${Math.min(cssY, wrap.height - 96)}px`;
}

function formatMeshSummary(mesh) {
  return `${mesh.stats.type}, ${mesh.nx} x ${mesh.ny}, ${mesh.triangles.length} tris`;
}

function nearestCoordinateIndex(coords, value) {
  if (value < coords[0] || value > coords[coords.length - 1]) return -1;
  let lo = 0;
  let hi = coords.length - 1;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (coords[mid] < value) lo = mid;
    else hi = mid;
  }
  return Math.abs(coords[lo] - value) <= Math.abs(coords[hi] - value) ? lo : hi;
}

function eventPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const cssX = event.clientX - rect.left;
  const cssY = event.clientY - rect.top;
  return {
    cssX,
    cssY,
    x: (cssX / rect.width) * canvas.width,
    y: (cssY / rect.height) * canvas.height,
  };
}

function updateQuantityDescription() {
  const info = quantityInfo(quantitySelect.value);
  quantitySelect.title = `${info.label}: ${info.description}; expr: ${info.expression}`;
}

function divergingColor(t) {
  const stops = [
    [49, 54, 149],
    [69, 117, 180],
    [224, 243, 248],
    [255, 255, 191],
    [253, 174, 97],
    [215, 48, 39],
  ];
  return interpolateColor(stops, t);
}

function plasmaColor(t) {
  const stops = [
    [13, 8, 135],
    [126, 3, 168],
    [203, 71, 119],
    [248, 149, 64],
    [240, 249, 33],
  ];
  return interpolateColor(stops, t);
}

function interpolateColor(stops, t) {
  const scaled = t * (stops.length - 1);
  const idx = Math.min(Math.floor(scaled), stops.length - 2);
  const f = scaled - idx;
  return [0, 1, 2].map((k) => Math.round(stops[idx][k] * (1 - f) + stops[idx + 1][k] * f));
}

function formatValue(value) {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e4 || abs < 1e-2) return value.toExponential(2);
  return value.toPrecision(4);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}
