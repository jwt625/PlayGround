import { parseSimpleYaml } from "./core/config.js";
import { parseElectrodes } from "./core/geometry.js";
import { parseMaterials } from "./core/materials.js";
import { getModePlotValues, solveOpticalModeConfig } from "./core/mode_solver.js";
import { quantityInfo } from "./core/quantities.js";
import { getPlotValues, solveConfig } from "./core/solver.js";
import { isOpticalMode, validateConfig } from "./core/validation.js";
import { getVectorModePlotValues, solveVectorModeConfig } from "./core/vector_mode_solver.js";

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
const viewModeSelect = document.querySelector("#view-mode-select");
const quantitySelect = document.querySelector("#quantity-select");
const modeSelectLabel = document.querySelector("#mode-select-label");
const modeSelect = document.querySelector("#mode-select");
const scaleSelect = document.querySelector("#scale-select");
const meshToggle = document.querySelector("#mesh-toggle");
const meshNxInput = document.querySelector("#mesh-nx-input");
const meshNyInput = document.querySelector("#mesh-ny-input");
const physicsSelect = document.querySelector("#physics-select");
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
let lastElapsed = 0;
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
document.querySelector("#tfln-button").addEventListener("click", () => loadExample("tfln", "electrostatic"));
document.querySelector("#bto-button").addEventListener("click", () => loadExample("bto", "electrostatic"));
document.querySelector("#si-mode-button").addEventListener("click", () => loadExample("siMode", "vector_mode"));
document.querySelector("#tfln-mode-button").addEventListener("click", () => loadExample("tfln", "vector_mode"));
document.querySelector("#bto-mode-button").addEventListener("click", () => loadExample("bto", "vector_mode"));
viewModeSelect.addEventListener("change", redrawLastResult);
quantitySelect.addEventListener("change", redrawLastResult);
modeSelect.addEventListener("change", () => {
  if (!lastResult) return;
  setSelectedModeIndex(lastResult, Number(modeSelect.value));
  renderResults(lastResult, lastElapsed);
  redrawLastResult();
});
scaleSelect.addEventListener("change", redrawLastResult);
meshToggle.addEventListener("change", redrawLastResult);
meshNxInput.addEventListener("change", () => updateMeshValue("mesh_nx", meshNxInput.value));
meshNyInput.addEventListener("change", () => updateMeshValue("mesh_ny", meshNyInput.value));
physicsSelect.addEventListener("change", runSolve);
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

async function loadExample(name, physicsOverride = "config") {
  appendLog("info", `Loading example: ${name}`);
  const response = await fetch(`${examplePaths[name]}?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) {
    appendLog("error", `Failed to load example ${name}: HTTP ${response.status}`);
    return;
  }
  configEditor.value = await response.text();
  physicsSelect.value = physicsOverride;
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
      config = effectiveConfigForSolve(parseSimpleYaml(configEditor.value));
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
    const result =
      String(config.Simulation?.physics ?? "").toLowerCase() === "vector_mode"
        ? solveVectorModeConfig(config)
        : isOpticalMode(config)
          ? solveOpticalModeConfig(config)
          : solveConfig(config);
    const elapsed = performance.now() - t0;
    lastConfig = config;
    lastResult = result;
    lastElapsed = elapsed;
    syncModeOptions(result);
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

function effectiveConfigForSolve(config) {
  const clone = structuredClone(config);
  clone.Simulation = clone.Simulation ?? {};
  if (physicsSelect.value === "electrostatic") {
    delete clone.Simulation.physics;
  } else if (physicsSelect.value === "optical_mode") {
    clone.Simulation.physics = "optical_mode";
    clone.Simulation.wavelength = clone.Simulation.wavelength ?? 1.55e-6;
    clone.Simulation.mode_polarization = clone.Simulation.mode_polarization ?? "Ex";
  } else if (physicsSelect.value === "vector_mode") {
    clone.Simulation.physics = "vector_mode";
    clone.Simulation.wavelength = clone.Simulation.wavelength ?? 1.55e-6;
  }
  return clone;
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
  if (isModeResult(result)) {
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
  const mode = selectedMode(result);
  const lines = [
    ["Mode", `${selectedModeIndex(result)} of ${result.modes.length - 1}`],
    ["n_eff", mode.nEff.toFixed(6)],
    ["beta", `${mode.beta.toExponential(6)} 1/m`],
    ["lambda", `${formatValue(result.wavelength)} m`],
    ["Target n_eff", result.targetNeff.toFixed(6)],
  ];
  if (result.physics === "vector_mode") {
    lines.push(
      ["TE fraction", `${(100 * mode.teFraction).toFixed(2)}%`],
      ["TM fraction", `${(100 * mode.tmFraction).toFixed(2)}%`],
      ["Boundary", result.boundaryCondition],
    );
  } else {
    lines.push(["Polarization", result.polarization]);
  }
  lines.push(
    ["Confinement", `${(100 * mode.confinement).toFixed(2)}%`],
    ["Mode-region overlap", mode.targetOverlap === null ? "-" : `${(100 * mode.targetOverlap).toFixed(2)}%`],
    ["Mode area", `${mode.modeArea.toExponential(6)} m^2`],
    ["Model", result.vectorModel ?? result.scalarModel],
    ["Mesh", formatMeshSummary(result.mesh)],
    ["Eigen solve", `${mode.iterations} iter, residual ${mode.residual.toExponential(3)}`],
    ["Runtime", `${elapsed.toFixed(0)} ms`],
  );
  renderResultLines(lines);
}

function renderResultsPending(config, state) {
  const nx = config.Simulation?.mesh_nx ?? 81;
  const ny = config.Simulation?.mesh_ny ?? 61;
  if (isOpticalMode(config)) {
    modeSelectLabel.hidden = false;
    modeSelect.disabled = true;
    renderResultLines([
      ["n_eff", "..."],
      ["beta", "..."],
      ["lambda", `${formatValue(config.Simulation?.wavelength ?? 1.55e-6)} m`],
      ["Target n_eff", `${config.Simulation?.target_neff ?? "auto"}`],
      ["Confinement", "..."],
      ["Mode area", "..."],
      ["Model", String(config.Simulation?.physics).toLowerCase() === "vector_mode" ? "vector optical mode" : "scalar optical mode"],
      ["Mesh", `${nx} x ${ny}`],
      ["Eigen solve", "..."],
      ["Runtime", "..."],
      ["Status", state],
    ]);
    return;
  }
  modeSelectLabel.hidden = true;
  modeSelect.disabled = true;
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
  results.textContent = lines.map(([key, value]) => `${key}: ${value}`).join("\n");
}

function redrawLastResult() {
  if (lastConfig && lastResult) renderPlot(lastConfig, lastResult);
}

function renderPlot(config, result) {
  syncQuantityOptions(result);
  syncModeOptions(result);
  syncViewModeControls();
  resizeCanvases();
  if (!view) resetViewport(result.mesh);
  if (viewModeSelect.value === "geometry") {
    renderGeometryView(config, result);
    return;
  }
  if (viewModeSelect.value === "mesh") {
    renderMeshView(config, result);
    return;
  }
  const { nx, ny } = result.mesh;
  const values = getActivePlotValues(result, quantitySelect.value);
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
  if (!isModeResult(result)) drawElectrodes(config, result);
  if (meshToggle.checked) drawMesh(result.mesh);
  renderColorbar(transform);
  updateQuantityDescription();
}

function syncViewModeControls() {
  const isResults = viewModeSelect.value === "results";
  quantitySelect.disabled = !isResults;
  scaleSelect.disabled = !isResults;
  meshToggle.disabled = viewModeSelect.value === "geometry";
  if (!isResults) {
    quantitySelect.title = viewModeSelect.value === "geometry" ? "Geometry view" : "Mesh view";
  }
}

function renderGeometryView(config, result) {
  clearPlotFrame("#0f141b");
  drawDomainFrame(result.mesh);
  drawMaterialRegions(config, result);
  drawElectrodeBoundaries(config, result);
  drawGeometryVertices(config, result);
  clearColorbar(viewModeSelect.value);
  renderModeBadge("Geometry");
}

function renderMeshView(config, result) {
  clearPlotFrame("#101720");
  drawMeshCells(result.mesh, { fill: true });
  drawMaterialBoundaries(config, result);
  drawElectrodeBoundaries(config, result);
  clearColorbar(viewModeSelect.value);
  renderModeBadge("Mesh");
}

function clearPlotFrame(fillStyle) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = fillStyle;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#2a3441";
  ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
  ctx.restore();
  plotTooltip.hidden = true;
}

function drawDomainFrame(mesh) {
  ctx.save();
  ctx.strokeStyle = "rgba(232,237,242,0.82)";
  ctx.lineWidth = 1.4;
  const p0 = toCanvas(mesh, [mesh.domain.xMin, mesh.domain.yMin]);
  const p1 = toCanvas(mesh, [mesh.domain.xMax, mesh.domain.yMax]);
  ctx.strokeRect(p0[0], p1[1], p1[0] - p0[0], p0[1] - p1[1]);
  drawTextLabel(mesh, "domain", [(mesh.domain.xMin + mesh.domain.xMax) / 2, mesh.domain.yMax], "#e8edf2");
  ctx.restore();
}

function drawMaterialRegions(config, result) {
  const materials = parseMaterials(config).filter((material) => material.shape !== "background");
  ctx.save();
  for (const material of materials) {
    drawShapePath(result.mesh, material.shape, material.params);
    ctx.fillStyle = materialFill(material.name);
    ctx.strokeStyle = "rgba(232,237,242,0.8)";
    ctx.lineWidth = 1.25;
    ctx.fill();
    ctx.stroke();
    drawTextLabel(result.mesh, material.name, shapeCenter(material.shape, material.params), "#f3f7f7");
  }
  ctx.restore();
}

function drawElectrodeBoundaries(config, result) {
  const electrodes = parseElectrodes(config);
  ctx.save();
  for (const electrode of electrodes) {
    drawShapePath(result.mesh, electrode.shape, electrode.params);
    ctx.fillStyle = electrode.potential > 0 ? "rgba(248,249,250,0.38)" : "rgba(32,38,46,0.62)";
    ctx.strokeStyle = electrode.potential > 0 ? "rgba(255,255,255,0.98)" : "rgba(154,167,181,0.98)";
    ctx.lineWidth = 1.6;
    ctx.fill();
    ctx.stroke();
    const label = `${electrode.name} ${formatValue(electrode.potential)} V`;
    drawTextLabel(result.mesh, label, shapeCenter(electrode.shape, electrode.params), "#ffffff");
  }
  ctx.restore();
}

function drawGeometryVertices(config, result) {
  const points = [];
  points.push([result.mesh.domain.xMin, result.mesh.domain.yMin]);
  points.push([result.mesh.domain.xMax, result.mesh.domain.yMin]);
  points.push([result.mesh.domain.xMax, result.mesh.domain.yMax]);
  points.push([result.mesh.domain.xMin, result.mesh.domain.yMax]);
  for (const material of parseMaterials(config)) collectShapeVertices(material.shape, material.params, points);
  for (const electrode of parseElectrodes(config)) collectShapeVertices(electrode.shape, electrode.params, points);
  ctx.save();
  ctx.fillStyle = "#1aa39a";
  ctx.strokeStyle = "#071015";
  ctx.lineWidth = 1;
  for (const point of points) {
    const [x, y] = toCanvas(result.mesh, point);
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawMeshCells(mesh, options = {}) {
  ctx.save();
  ctx.lineWidth = 0.55;
  for (const tri of mesh.triangles) {
    ctx.beginPath();
    for (let k = 0; k < tri.length; k += 1) {
      const [x, y] = toCanvas(mesh, mesh.nodes[tri[k]]);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    if (options.fill) {
      ctx.fillStyle = triangleFill(mesh, tri);
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(232,237,242,0.24)";
    ctx.stroke();
  }
  ctx.restore();
}

function renderModeBadge(label) {
  ctx.save();
  ctx.fillStyle = "rgba(8,12,17,0.82)";
  ctx.strokeStyle = "rgba(42,52,65,0.95)";
  ctx.lineWidth = 1;
  const dpr = window.devicePixelRatio || 1;
  ctx.font = `${12 * dpr}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  const text = `${label} view`;
  const width = ctx.measureText(text).width + 18 * dpr;
  ctx.fillRect(12 * dpr, 12 * dpr, width, 26 * dpr);
  ctx.strokeRect(12 * dpr + 0.5, 12 * dpr + 0.5, width, 26 * dpr);
  ctx.fillStyle = "#e8edf2";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 21 * dpr, 25 * dpr);
  ctx.restore();
}

function clearColorbar(label = "") {
  colorbarCtx.clearRect(0, 0, colorbarCanvas.width, colorbarCanvas.height);
  colorbarMax.textContent = "";
  colorbarMid.textContent = label;
  colorbarMin.textContent = "";
}

function syncQuantityOptions(result) {
  const optical = isModeResult(result);
  const groups = optical
    ? [
        {
          label: "EM Mode Fields",
          options:
            result.physics === "vector_mode"
              ? [
                  ["mode_Ex", "Ex - electric field"],
                  ["mode_Ey", "Ey - electric field"],
                  ["mode_Ez", "Ez - longitudinal electric field"],
                  ["mode_Hx", "Hx - magnetic field"],
                  ["mode_Hy", "Hy - magnetic field"],
                  ["mode_Hz", "Hz - longitudinal magnetic field"],
                  ["mode_normE", "|E| - electric-field magnitude"],
                  ["mode_normH", "|H| - magnetic-field magnitude"],
                  ["mode_intensity", "I - modal intensity"],
                ]
              : [
                  ["mode_Ex", "Ex - scalar modal x component"],
                  ["mode_Ey", "Ey - scalar modal y component"],
                  ["mode_Ez", "Ez - scalar modal z component"],
                  ["mode_normE", "|E| - scalar modal magnitude"],
                  ["mode_intensity", "I - modal intensity"],
                ],
        },
        {
          label: "Optical Material Properties",
          options: [
            ["n", "n - selected optical refractive index"],
            ["n_xx", "n_xx - optical tensor xx"],
            ["n_yy", "n_yy - optical tensor yy"],
            ["n_zz", "n_zz - optical tensor zz"],
          ],
        },
      ]
    : [
        {
          label: "ES Fields",
          options: [
            ["phi", "phi - electrostatic potential"],
            ["Ex", "Ex - x electric-field component"],
            ["Ey", "Ey - y electric-field component"],
            ["normE", "|E| - electric-field magnitude"],
          ],
        },
        {
          label: "RF Material Properties",
          options: [
            ["eps_r", "epsilon_r - relative permittivity"],
            ["eps_r_xx", "epsilon_r_xx - permittivity tensor xx"],
            ["eps_r_yy", "epsilon_r_yy - permittivity tensor yy"],
            ["eps_r_xy", "epsilon_r_xy - permittivity tensor xy"],
          ],
        },
        {
          label: "EO Material Properties",
          options: [
            ["r13", "r13 - EO tensor coefficient"],
            ["r33", "r33 - EO tensor coefficient"],
            ["r22", "r22 - EO tensor coefficient"],
            ["r_eff", "r_eff - effective EO coefficient"],
          ],
        },
      ];
  const desired = groups.flatMap((group) => group.options);
  const currentValues = Array.from(quantitySelect.options).map((option) => option.value).join("|");
  const nextValues = desired.map(([value]) => value).join("|");
  if (currentValues === nextValues) return;
  const previous = quantitySelect.value;
  quantitySelect.innerHTML = "";
  for (const group of groups) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group.label;
    for (const [value, label] of group.options) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      optgroup.append(option);
    }
    quantitySelect.append(optgroup);
  }
  quantitySelect.value = desired.some(([value]) => value === previous) ? previous : desired[0][0];
}

function getActivePlotValues(result, quantity) {
  if (result.physics === "vector_mode") return getVectorModePlotValues(result, quantity);
  if (result.physics === "optical_mode") return getModePlotValues(result, quantity);
  return getPlotValues(result, quantity);
}

function syncModeOptions(result) {
  const optical = isModeResult(result);
  modeSelectLabel.hidden = !optical;
  modeSelect.disabled = !optical;
  if (!optical) {
    modeSelect.innerHTML = "";
    return;
  }
  const previous = Number(modeSelect.value);
  const modes = result.modes ?? [];
  const nextValues = modes.map((mode, index) => modeOptionLabel(mode, index)).join("|");
  const currentValues = Array.from(modeSelect.options).map((option) => option.textContent).join("|");
  if (nextValues !== currentValues) {
    modeSelect.innerHTML = "";
    modes.forEach((mode, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = modeOptionLabel(mode, index);
      modeSelect.append(option);
    });
  }
  const index = Number.isInteger(previous) && previous >= 0 && previous < modes.length ? previous : 0;
  modeSelect.value = String(index);
  setSelectedModeIndex(result, index);
}

function modeOptionLabel(mode, index) {
  const overlap = mode.targetOverlap === null ? "" : `, overlap ${(100 * mode.targetOverlap).toFixed(1)}%`;
  const vector = mode.teFraction === undefined ? "" : `, TE ${(100 * mode.teFraction).toFixed(0)}%`;
  return `${index}: n_eff ${mode.nEff.toFixed(4)}${vector}${overlap}`;
}

function selectedModeIndex(result) {
  const modes = result.modes ?? [];
  const index = Number(result.selectedModeIndex ?? 0);
  return Number.isInteger(index) && index >= 0 && index < modes.length ? index : 0;
}

function selectedMode(result) {
  return result.modes?.[selectedModeIndex(result)] ?? result.mode;
}

function setSelectedModeIndex(result, index) {
  const modes = result.modes ?? [];
  const next = Number.isInteger(index) && index >= 0 && index < modes.length ? index : 0;
  result.selectedModeIndex = next;
  result.mode = modes[next] ?? result.mode;
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
  drawMeshCells(mesh);
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
  if (result.physics === "vector_mode") {
    return `${result.vectorModel}; ${formatMeshSummary(result.mesh)}; n_eff ${result.mode.nEff.toFixed(6)}; TE ${(100 * result.mode.teFraction).toFixed(1)}%; eig ${result.mode.iterations} iter; residual ${result.mode.residual.toExponential(3)}`;
  }
  if (result.physics === "optical_mode") {
    return `${result.scalarModel}; ${formatMeshSummary(result.mesh)}; ${result.polarization}; n_eff ${result.mode.nEff.toFixed(6)}; eig ${result.mode.iterations} iter; residual ${result.mode.residual.toExponential(3)}`;
  }
  return `${result.permittivityModel}; ${formatMeshSummary(result.mesh)}; CG ${result.iterations} iter; residual ${result.residual.toExponential(3)}`;
}

function isModeResult(result) {
  return result.physics === "optical_mode" || result.physics === "vector_mode";
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

function collectShapeVertices(shape, params, points) {
  if (shape === "rectangle") {
    points.push([params.x_min, params.y_min]);
    points.push([params.x_max, params.y_min]);
    points.push([params.x_max, params.y_max]);
    points.push([params.x_min, params.y_max]);
  } else if (shape === "circle") {
    points.push([params.x + params.radius, params.y]);
    points.push([params.x, params.y + params.radius]);
    points.push([params.x - params.radius, params.y]);
    points.push([params.x, params.y - params.radius]);
  }
}

function shapeCenter(shape, params) {
  if (shape === "rectangle") {
    return [(params.x_min + params.x_max) / 2, (params.y_min + params.y_max) / 2];
  }
  if (shape === "circle") return [params.x, params.y];
  return [0, 0];
}

function pointInShape(shape, params, x, y) {
  if (shape === "rectangle") {
    return params.x_min <= x && x <= params.x_max && params.y_min <= y && y <= params.y_max;
  }
  if (shape === "circle") {
    const dx = x - params.x;
    const dy = y - params.y;
    return dx * dx + dy * dy <= params.radius * params.radius;
  }
  return false;
}

function materialFill(name) {
  const colors = [
    "rgba(26,163,154,0.34)",
    "rgba(225,175,73,0.34)",
    "rgba(116,155,226,0.34)",
    "rgba(209,103,136,0.34)",
    "rgba(151,118,206,0.34)",
  ];
  return colors[hashString(name) % colors.length];
}

function triangleFill(mesh, tri) {
  const [a, b, c] = tri.map((index) => mesh.nodes[index]);
  const y = (a[1] + b[1] + c[1]) / 3;
  const t = clamp((y - mesh.domain.yMin) / (mesh.domain.yMax - mesh.domain.yMin), 0, 1);
  const color = plasmaColor(0.12 + 0.56 * t);
  return `rgba(${color[0]},${color[1]},${color[2]},0.32)`;
}

function drawTextLabel(mesh, text, point, color) {
  const [x, y] = toCanvas(mesh, point);
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.font = `${11 * dpr}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const metrics = ctx.measureText(text);
  const width = metrics.width + 8 * dpr;
  const height = 17 * dpr;
  ctx.fillStyle = "rgba(8,12,17,0.72)";
  ctx.fillRect(x - width / 2, y - height / 2, width, height);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
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
  if (viewModeSelect.value === "geometry") {
    updateGeometryTooltip(point, x, y);
    return;
  }
  if (viewModeSelect.value === "mesh") {
    updateMeshTooltip(point, x, y);
    return;
  }
  const mesh = lastResult.mesh;
  const i = nearestCoordinateIndex(mesh.xCoords, x);
  const j = nearestCoordinateIndex(mesh.yCoords, y);
  if (i < 0 || i >= mesh.nx || j < 0 || j >= mesh.ny) {
    plotTooltip.hidden = true;
    return;
  }
  const idx = j * mesh.nx + i;
  const values = getActivePlotValues(lastResult, quantitySelect.value);
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

function updateGeometryTooltip(point, x, y) {
  const entities = [];
  for (const material of parseMaterials(lastConfig).filter((item) => item.shape !== "background")) {
    if (pointInShape(material.shape, material.params, x, y)) entities.push(`domain: ${material.name}`);
  }
  for (const electrode of parseElectrodes(lastConfig)) {
    if (pointInShape(electrode.shape, electrode.params, x, y)) {
      entities.push(`boundary: ${electrode.name} (${formatValue(electrode.potential)} V)`);
    }
  }
  showTooltip(
    point,
    [
      `x = ${formatValue(x)} m`,
      `y = ${formatValue(y)} m`,
      entities.length > 0 ? entities.join("<br />") : "domain: background",
      "view: geometry",
    ],
    240,
    112,
  );
}

function updateMeshTooltip(point, x, y) {
  const nearest = nearestMeshNode(lastResult.mesh, x, y);
  const lines = [`x = ${formatValue(x)} m`, `y = ${formatValue(y)} m`, "view: mesh"];
  if (nearest) {
    lines.push(`nearest node: ${nearest.index}`, `distance: ${formatValue(nearest.distance)} m`);
  }
  lines.push(`triangles: ${lastResult.mesh.triangles.length}`);
  showTooltip(point, lines, 230, 116);
}

function showTooltip(point, lines, maxWidth, maxHeight) {
  plotTooltip.innerHTML = lines.join("<br />");
  plotTooltip.hidden = false;
  const wrap = canvas.parentElement.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const cssX = canvasRect.left - wrap.left + point.cssX + 14;
  const cssY = canvasRect.top - wrap.top + point.cssY + 14;
  plotTooltip.style.left = `${Math.min(cssX, wrap.width - maxWidth)}px`;
  plotTooltip.style.top = `${Math.min(cssY, wrap.height - maxHeight)}px`;
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

function nearestMeshNode(mesh, x, y) {
  let best = null;
  const viewWidth = view.xMax - view.xMin;
  const viewHeight = view.yMax - view.yMin;
  const maxDistance = Math.hypot(viewWidth, viewHeight) * 0.03;
  for (let index = 0; index < mesh.nodes.length; index += 1) {
    const node = mesh.nodes[index];
    const distance = Math.hypot(node[0] - x, node[1] - y);
    if (!best || distance < best.distance) best = { index, distance };
  }
  return best && best.distance <= maxDistance ? best : null;
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

function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}
