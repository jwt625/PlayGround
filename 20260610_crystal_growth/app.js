"use strict";

const NX = 108;
const NZ = 144;
const SILICON_MELT_C = 1414;
const BASE_DT = 0.015;
const TEMP_MIN = 200;
const TEMP_MAX = 1800;

const REGION = {
  AMBIENT: 0,
  MELT: 1,
  CRYSTAL: 2,
  CRUCIBLE: 3,
  HEATER: 4,
  INSULATION: 5
};

const REGION_COLORS = {
  [REGION.AMBIENT]: "rgba(255,255,255,0.08)",
  [REGION.MELT]: "rgba(103,185,210,0.42)",
  [REGION.CRYSTAL]: "rgba(220,236,240,0.38)",
  [REGION.CRUCIBLE]: "rgba(214,205,180,0.34)",
  [REGION.HEATER]: "rgba(245,148,71,0.36)",
  [REGION.INSULATION]: "rgba(156,154,145,0.22)"
};

const PRESETS = {
  hotSilicon: {
    name: "Hot silicon baseline",
    ambient: { rho: 1.0, cp: 1000, k: 0.08 },
    melt: { rho: 2520, cp: 900, k: 55 },
    crystal: { rho: 2330, cp: 760, k: 80 },
    crucible: { rho: 2200, cp: 740, k: 1.38 },
    heater: { rho: 1800, cp: 1200, k: 80 },
    insulation: { rho: 550, cp: 1000, k: 0.45 }
  },
  lowK: {
    name: "Lower-k melt sensitivity",
    ambient: { rho: 1.0, cp: 1000, k: 0.08 },
    melt: { rho: 2520, cp: 950, k: 25 },
    crystal: { rho: 2330, cp: 780, k: 45 },
    crucible: { rho: 2200, cp: 740, k: 1.38 },
    heater: { rho: 1800, cp: 1200, k: 70 },
    insulation: { rho: 550, cp: 1000, k: 0.35 }
  },
  highK: {
    name: "Higher-k silicon sensitivity",
    ambient: { rho: 1.0, cp: 1000, k: 0.08 },
    melt: { rho: 2520, cp: 850, k: 80 },
    crystal: { rho: 2330, cp: 720, k: 130 },
    crucible: { rho: 2200, cp: 740, k: 1.38 },
    heater: { rho: 1800, cp: 1100, k: 110 },
    insulation: { rho: 550, cp: 1000, k: 0.5 }
  }
};

const GEOMETRIES = {
  lab: {
    widthM: 0.72,
    heightM: 0.96,
    meltRadius: 0.33,
    meltTop: 0.55,
    meltBottom: 0.82,
    crystalRadius: 0.105,
    crucibleOuterRadius: 0.38,
    crucibleBottom: 0.87,
    heaterInner: 0.42,
    heaterOuter: 0.48
  },
  pv: {
    widthM: 1.18,
    heightM: 1.28,
    meltRadius: 0.42,
    meltTop: 0.52,
    meltBottom: 0.82,
    crystalRadius: 0.16,
    crucibleOuterRadius: 0.47,
    crucibleBottom: 0.88,
    heaterInner: 0.52,
    heaterOuter: 0.60
  },
  compact: {
    widthM: 0.42,
    heightM: 0.60,
    meltRadius: 0.29,
    meltTop: 0.56,
    meltBottom: 0.80,
    crystalRadius: 0.085,
    crucibleOuterRadius: 0.35,
    crucibleBottom: 0.86,
    heaterInner: 0.40,
    heaterOuter: 0.48
  }
};

const canvas = document.getElementById("heatCanvas");
const ctx = canvas.getContext("2d");
const legendCanvas = document.getElementById("legendCanvas");
const legendCtx = legendCanvas.getContext("2d");

const controls = {
  toggleRun: document.getElementById("toggleRun"),
  reset: document.getElementById("reset"),
  step: document.getElementById("step"),
  heaterTemp: document.getElementById("heaterTemp"),
  crystalTemp: document.getElementById("crystalTemp"),
  ambientTemp: document.getElementById("ambientTemp"),
  simSpeed: document.getElementById("simSpeed"),
  materialPreset: document.getElementById("materialPreset"),
  geometryPreset: document.getElementById("geometryPreset"),
  axisymmetric: document.getElementById("axisymmetric")
};

const outputs = {
  heater: document.getElementById("heaterOut"),
  crystal: document.getElementById("crystalOut"),
  ambient: document.getElementById("ambientOut"),
  speed: document.getElementById("speedOut"),
  time: document.getElementById("timeReadout"),
  maxTemp: document.getElementById("maxTempReadout"),
  melt: document.getElementById("meltReadout"),
  stability: document.getElementById("stabilityReadout")
};

let temp = new Float64Array(NX * NZ);
let nextTemp = new Float64Array(NX * NZ);
let alpha = new Float64Array(NX * NZ);
let region = new Uint8Array(NX * NZ);
let running = false;
let simTime = 0;
let lastFrame = 0;
let geometry = GEOMETRIES.lab;

function idx(x, z) {
  return z * NX + x;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getSettings() {
  return {
    heaterTemp: Number(controls.heaterTemp.value),
    crystalTemp: Number(controls.crystalTemp.value),
    ambientTemp: Number(controls.ambientTemp.value),
    simSpeed: Number(controls.simSpeed.value),
    axisymmetric: controls.axisymmetric.checked
  };
}

function updateLabels() {
  const s = getSettings();
  outputs.heater.value = `${s.heaterTemp} C`;
  outputs.crystal.value = `${s.crystalTemp} C`;
  outputs.ambient.value = `${s.ambientTemp} C`;
  outputs.speed.value = `${s.simSpeed}x`;
}

function alphaOf(material) {
  return material.k / (material.rho * material.cp);
}

function classifyCell(x, z) {
  const xn = x / (NX - 1);
  const zn = z / (NZ - 1);
  const r = Math.abs(xn - 0.5);
  const g = geometry;

  if (r < g.crystalRadius && zn < g.meltTop + 0.03) return REGION.CRYSTAL;
  if (r < g.meltRadius && zn >= g.meltTop && zn <= g.meltBottom) return REGION.MELT;
  if (r < g.crucibleOuterRadius && zn > g.meltBottom && zn <= g.crucibleBottom) return REGION.CRUCIBLE;
  if (r >= g.meltRadius && r < g.crucibleOuterRadius && zn >= g.meltTop - 0.02 && zn <= g.crucibleBottom) return REGION.CRUCIBLE;
  if (r >= g.heaterInner && r <= g.heaterOuter && zn >= g.meltTop - 0.08 && zn <= g.crucibleBottom + 0.02) return REGION.HEATER;
  if (r > 0.43 || zn > 0.88 || zn < 0.08) return REGION.INSULATION;
  return REGION.AMBIENT;
}

function rebuildGeometry() {
  geometry = GEOMETRIES[controls.geometryPreset.value];
  const preset = PRESETS[controls.materialPreset.value];
  const materialByRegion = {
    [REGION.AMBIENT]: preset.ambient,
    [REGION.MELT]: preset.melt,
    [REGION.CRYSTAL]: preset.crystal,
    [REGION.CRUCIBLE]: preset.crucible,
    [REGION.HEATER]: preset.heater,
    [REGION.INSULATION]: preset.insulation
  };

  for (let z = 0; z < NZ; z += 1) {
    for (let x = 0; x < NX; x += 1) {
      const i = idx(x, z);
      region[i] = classifyCell(x, z);
      alpha[i] = alphaOf(materialByRegion[region[i]]);
    }
  }
}

function resetSimulation() {
  rebuildGeometry();
  const s = getSettings();
  for (let z = 0; z < NZ; z += 1) {
    for (let x = 0; x < NX; x += 1) {
      const i = idx(x, z);
      const verticalBias = 1 - z / (NZ - 1);
      let initial = s.ambientTemp + 80 * verticalBias;
      if (region[i] === REGION.MELT) initial = 1360;
      if (region[i] === REGION.CRYSTAL) initial = s.crystalTemp;
      if (region[i] === REGION.HEATER) initial = s.heaterTemp;
      if (region[i] === REGION.CRUCIBLE) initial = 1250;
      temp[i] = initial;
      nextTemp[i] = initial;
    }
  }
  simTime = 0;
  draw();
}

function applyBoundaryConditions(field) {
  const s = getSettings();
  for (let z = 0; z < NZ; z += 1) {
    field[idx(0, z)] = s.ambientTemp;
    field[idx(NX - 1, z)] = s.ambientTemp;
  }
  for (let x = 0; x < NX; x += 1) {
    field[idx(x, 0)] = s.ambientTemp;
    field[idx(x, NZ - 1)] = s.ambientTemp;
  }

  for (let z = 0; z < NZ; z += 1) {
    for (let x = 0; x < NX; x += 1) {
      const i = idx(x, z);
      if (region[i] === REGION.HEATER) {
        field[i] = s.heaterTemp;
      } else if (region[i] === REGION.CRYSTAL && z < NZ * 0.33) {
        field[i] = 0.92 * field[i] + 0.08 * s.crystalTemp;
      }
    }
  }
}

function maxStableDt() {
  let maxAlpha = 0;
  for (let i = 0; i < alpha.length; i += 1) maxAlpha = Math.max(maxAlpha, alpha[i]);
  const dx = geometry.widthM / (NX - 1);
  const dz = geometry.heightM / (NZ - 1);
  return 0.22 / (maxAlpha * (1 / (dx * dx) + 1 / (dz * dz)));
}

function stepSimulation(count) {
  const dtLimit = maxStableDt();
  const speed = getSettings().simSpeed;
  const dt = Math.min(BASE_DT * speed, dtLimit * 0.85);
  const dx = geometry.widthM / (NX - 1);
  const dz = geometry.heightM / (NZ - 1);
  const dx2 = dx * dx;
  const dz2 = dz * dz;
  const useAxisym = getSettings().axisymmetric;

  applyBoundaryConditions(temp);

  for (let n = 0; n < count; n += 1) {
    for (let z = 1; z < NZ - 1; z += 1) {
      for (let x = 1; x < NX - 1; x += 1) {
        const i = idx(x, z);
        if (region[i] === REGION.HEATER) {
          nextTemp[i] = getSettings().heaterTemp;
          continue;
        }

        const center = temp[i];
        const d2r = (temp[idx(x + 1, z)] - 2 * center + temp[idx(x - 1, z)]) / dx2;
        const d2z = (temp[idx(x, z + 1)] - 2 * center + temp[idx(x, z - 1)]) / dz2;
        let radialTerm = 0;

        if (useAxisym) {
          const r = Math.abs((x / (NX - 1) - 0.5) * geometry.widthM);
          if (r > dx) {
            const sign = x >= NX / 2 ? 1 : -1;
            radialTerm = sign * (temp[idx(x + 1, z)] - temp[idx(x - 1, z)]) / (2 * dx * r);
          }
        }

        const source = region[i] === REGION.CRYSTAL ? -0.0009 * (center - getSettings().crystalTemp) : 0;
        nextTemp[i] = clamp(center + dt * (alpha[i] * (d2r + d2z + radialTerm) + source), TEMP_MIN, TEMP_MAX + 200);
      }
    }

    [temp, nextTemp] = [nextTemp, temp];
    applyBoundaryConditions(temp);
    simTime += dt;
  }
}

function colorForTemp(t) {
  const u = clamp((t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0, 1);
  const stops = [
    [0.0, [20, 24, 39]],
    [0.18, [38, 92, 142]],
    [0.36, [64, 164, 152]],
    [0.55, [232, 202, 92]],
    [0.75, [229, 111, 65]],
    [1.0, [244, 238, 214]]
  ];

  for (let s = 0; s < stops.length - 1; s += 1) {
    const a = stops[s];
    const b = stops[s + 1];
    if (u >= a[0] && u <= b[0]) {
      const f = (u - a[0]) / (b[0] - a[0]);
      const rgb = a[1].map((v, j) => Math.round(v + f * (b[1][j] - v)));
      return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    }
  }
  return "rgb(244,238,214)";
}

function draw() {
  const w = canvas.width;
  const h = canvas.height;
  const cellW = w / NX;
  const cellH = h / NZ;

  ctx.clearRect(0, 0, w, h);
  for (let z = 0; z < NZ; z += 1) {
    for (let x = 0; x < NX; x += 1) {
      const i = idx(x, z);
      ctx.fillStyle = colorForTemp(temp[i]);
      ctx.fillRect(x * cellW, z * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  for (let z = 0; z < NZ; z += 1) {
    for (let x = 0; x < NX; x += 1) {
      const i = idx(x, z);
      if (region[i] === REGION.AMBIENT) continue;
      ctx.fillStyle = REGION_COLORS[region[i]];
      ctx.fillRect(x * cellW, z * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  drawIsotherm(cellW, cellH);
  drawCenterline();
  drawLegend();
  updateReadouts();
}

function drawIsotherm(cellW, cellH) {
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let z = 1; z < NZ - 1; z += 1) {
    for (let x = 1; x < NX - 1; x += 1) {
      const i = idx(x, z);
      if (region[i] !== REGION.MELT && region[i] !== REGION.CRYSTAL) continue;
      const here = temp[i] >= SILICON_MELT_C;
      const neighbor = temp[idx(x + 1, z)] >= SILICON_MELT_C || temp[idx(x, z + 1)] >= SILICON_MELT_C;
      if (here !== neighbor) {
        ctx.rect(x * cellW, z * cellH, cellW, cellH);
      }
    }
  }
  ctx.stroke();
}

function drawCenterline() {
  const x = canvas.width / 2;
  ctx.strokeStyle = "rgba(255,255,255,0.48)";
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 6]);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, canvas.height);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText("axis", x + 8, 22);
  ctx.fillText(`${geometry.widthM.toFixed(2)} m x ${geometry.heightM.toFixed(2)} m`, 16, canvas.height - 18);
}

function drawLegend() {
  const w = legendCanvas.width;
  const h = legendCanvas.height;
  const vertical = h > 120;
  legendCtx.clearRect(0, 0, w, h);

  if (vertical) {
    for (let y = 0; y < h; y += 1) {
      const t = TEMP_MAX - (y / h) * (TEMP_MAX - TEMP_MIN);
      legendCtx.fillStyle = colorForTemp(t);
      legendCtx.fillRect(0, y, 24, 1);
    }
    legendCtx.fillStyle = "#f2f0e8";
    legendCtx.font = "12px system-ui, sans-serif";
    legendCtx.fillText(`${TEMP_MAX}C`, 4, 14);
    legendCtx.fillText(`${SILICON_MELT_C}C`, 4, Math.round((1 - (SILICON_MELT_C - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)) * h));
    legendCtx.fillText(`${TEMP_MIN}C`, 4, h - 8);
  } else {
    for (let x = 0; x < w; x += 1) {
      const t = TEMP_MIN + (x / w) * (TEMP_MAX - TEMP_MIN);
      legendCtx.fillStyle = colorForTemp(t);
      legendCtx.fillRect(x, 0, 1, h);
    }
  }
}

function updateReadouts() {
  let maxT = -Infinity;
  let meltCells = 0;
  let siliconCells = 0;

  for (let i = 0; i < temp.length; i += 1) {
    maxT = Math.max(maxT, temp[i]);
    if (region[i] === REGION.MELT || region[i] === REGION.CRYSTAL) {
      siliconCells += 1;
      if (temp[i] >= SILICON_MELT_C) meltCells += 1;
    }
  }

  const dtLimit = maxStableDt();
  const requested = BASE_DT * getSettings().simSpeed;
  outputs.time.textContent = `${simTime.toFixed(1)} s`;
  outputs.maxTemp.textContent = `${maxT.toFixed(0)} C`;
  outputs.melt.textContent = `${Math.round((100 * meltCells) / Math.max(1, siliconCells))}% Si cells`;
  outputs.stability.textContent = requested <= dtLimit ? "stable" : `clamped dt`;
  outputs.stability.style.color = requested <= dtLimit ? "var(--accent)" : "var(--warning)";
}

function frame(now) {
  if (!lastFrame) lastFrame = now;
  const elapsed = now - lastFrame;
  lastFrame = now;

  if (running) {
    const steps = clamp(Math.round(elapsed / 16), 1, 4);
    stepSimulation(steps);
    draw();
  }

  requestAnimationFrame(frame);
}

for (const element of Object.values(controls)) {
  if (element instanceof HTMLInputElement || element instanceof HTMLSelectElement) {
    element.addEventListener("input", () => {
      updateLabels();
      if (element === controls.materialPreset || element === controls.geometryPreset) resetSimulation();
      draw();
    });
  }
}

controls.toggleRun.addEventListener("click", () => {
  running = !running;
  controls.toggleRun.textContent = running ? "Pause" : "Start";
});

controls.reset.addEventListener("click", () => {
  running = false;
  controls.toggleRun.textContent = "Start";
  resetSimulation();
});

controls.step.addEventListener("click", () => {
  stepSimulation(12);
  draw();
});

window.addEventListener("resize", draw);

updateLabels();
resetSimulation();
requestAnimationFrame(frame);
