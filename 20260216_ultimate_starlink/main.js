import * as THREE from 'https://unpkg.com/three@0.163.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.163.0/examples/jsm/controls/OrbitControls.js';

const EARTH_R_KM = 6371;
const DEG = Math.PI / 180;
const TARGET_LAT_DEG = 37.7749;
const TARGET_LON_DEG = -122.4194;

const els = {
  preset: document.getElementById('preset'),
  freq: document.getElementById('freq'),
  shell: document.getElementById('shell'),
  beamHalf: document.getElementById('beamHalf'),
  phaseJit: document.getElementById('phaseJit'),
  seed: document.getElementById('seed'),
  freqVal: document.getElementById('freqVal'),
  shellVal: document.getElementById('shellVal'),
  beamHalfVal: document.getElementById('beamHalfVal'),
  phaseJitVal: document.getElementById('phaseJitVal'),
  seedVal: document.getElementById('seedVal'),
  presetLabel: document.getElementById('presetLabel'),
  stats: document.getElementById('stats'),
};

const panel3d = document.getElementById('panel-3d');
const singleCanvas = document.getElementById('singleCanvas');
const groundCanvas = document.getElementById('groundCanvas');

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const m = i % 6;
  const arr = [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q],
  ][m];
  return { r: Math.round(arr[0] * 255), g: Math.round(arr[1] * 255), b: Math.round(arr[2] * 255) };
}

function psiMaxRad(hKm, elevDeg) {
  const r = EARTH_R_KM;
  const h = hKm;
  const el = elevDeg * DEG;
  return Math.acos((r / (r + h)) * Math.cos(el)) - el;
}

function latLonToUnit(latDeg, lonDeg) {
  const lat = latDeg * DEG;
  // This texture is mirrored in longitude relative to geodetic convention.
  const lon = -lonDeg * DEG;
  const c = Math.cos(lat);
  // Equirectangular mapping aligned with SphereGeometry UV convention:
  // u ~ lon/360 + 0.5, v ~ lat.
  return {
    x: c * Math.cos(lon),
    y: Math.sin(lat),
    z: c * Math.sin(lon),
  };
}

function localBasisFromUp(up) {
  // Pick a reference axis that is not nearly parallel to up.
  const ref = Math.abs(up.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
  let east = {
    x: ref.y * up.z - ref.z * up.y,
    y: ref.z * up.x - ref.x * up.z,
    z: ref.x * up.y - ref.y * up.x,
  };
  let en = Math.sqrt(east.x * east.x + east.y * east.y + east.z * east.z);
  if (en < 1e-8) {
    east = { x: 1, y: 0, z: 0 };
    en = 1;
  }
  east.x /= en; east.y /= en; east.z /= en;
  const north = {
    x: up.y * east.z - up.z * east.y,
    y: up.z * east.x - up.x * east.z,
    z: up.x * east.y - up.y * east.x,
  };
  return { east, north, up };
}

function localToWorld(localDir, basis) {
  return {
    x: basis.east.x * localDir.x + basis.north.x * localDir.y + basis.up.x * localDir.z,
    y: basis.east.y * localDir.x + basis.north.y * localDir.y + basis.up.y * localDir.z,
    z: basis.east.z * localDir.x + basis.north.z * localDir.y + basis.up.z * localDir.z,
  };
}

function visibleCountApprox(hKm, totalConstellation = 29988) {
  const psi = psiMaxRad(hKm, 0);
  const f = (1 - Math.cos(psi)) / 2;
  return totalConstellation * f;
}

function sampleDirectionsInCap(count, psiMax, minSep, rand) {
  if (minSep <= 0) {
    const out = [];
    for (let i = 0; i < count; i += 1) {
      const u = rand();
      const cosPsi = 1 - u * (1 - Math.cos(psiMax));
      const sinPsi = Math.sqrt(Math.max(0, 1 - cosPsi * cosPsi));
      const az = 2 * Math.PI * rand();
      out.push({
        x: sinPsi * Math.cos(az),
        y: sinPsi * Math.sin(az),
        z: cosPsi,
      });
    }
    return out;
  }

  const dirs = [];
  const maxTries = count * 400;
  let tries = 0;
  while (dirs.length < count && tries < maxTries) {
    tries += 1;
    const u = rand();
    const cosPsi = 1 - u * (1 - Math.cos(psiMax));
    const sinPsi = Math.sqrt(Math.max(0, 1 - cosPsi * cosPsi));
    const az = 2 * Math.PI * rand();
    const x = sinPsi * Math.cos(az);
    const y = sinPsi * Math.sin(az);
    const z = cosPsi;

    let ok = true;
    for (let i = 0; i < dirs.length; i += 1) {
      const d = dirs[i];
      const dot = clamp(x * d.x + y * d.y + z * d.z, -1, 1);
      if (Math.acos(dot) < minSep) {
        ok = false;
        break;
      }
    }
    if (ok) dirs.push({ x, y, z });
  }
  return dirs;
}

function computeModel(params) {
  const rand = mulberry32(params.seed);
  const lambdaM = 0.299792458 / params.freqGHz;
  const psiMax = psiMaxRad(params.shellKm, 0);
  const targetUp = latLonToUnit(TARGET_LAT_DEG, TARGET_LON_DEG);
  const localBasis = localBasisFromUp(targetUp);
  const targetPos = {
    x: targetUp.x * EARTH_R_KM,
    y: targetUp.y * EARTH_R_KM,
    z: targetUp.z * EARTH_R_KM,
  };

  const visApprox = visibleCountApprox(params.shellKm, params.totalConstellation);
  const visCap = Math.max(1, Math.floor(visApprox));

  // Use all satellites that are within view of the target point.
  const nActive = visCap;

  const rShell = EARTH_R_KM + params.shellKm;
  const minSep = 0;
  const localDirs = sampleDirectionsInCap(nActive, psiMax, minSep, rand);
  const dirs = localDirs.map((d) => localToWorld(d, localBasis));

  while (dirs.length < nActive) {
    const az = 2 * Math.PI * rand();
    const cosPsi = 1 - rand() * (1 - Math.cos(psiMax));
    const sinPsi = Math.sqrt(Math.max(0, 1 - cosPsi * cosPsi));
    dirs.push({ x: sinPsi * Math.cos(az), y: sinPsi * Math.sin(az), z: cosPsi });
  }

  const satellites = dirs.map((d) => {
    const phaseErr = gaussian(rand) * params.phaseJitDeg * DEG;
    const p = 0.8 + 0.4 * rand();
    const satPos = {
      x: d.x * rShell,
      y: d.y * rShell,
      z: d.z * rShell,
    };
    const vx = satPos.x - targetPos.x;
    const vy = satPos.y - targetPos.y;
    const vz = satPos.z - targetPos.z;
    const norm = Math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9;
    const ux = vx / norm;
    const uy = vy / norm;
    const uz = vz / norm;
    return {
      dir: d,
      satPos,
      sky: { x: ux, y: uy, z: uz },
      amp: p,
      phase: phaseErr,
      az: Math.atan2(uy, ux),
      theta: Math.acos(clamp(uz, -1, 1)),
    };
  });

  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const s of satellites) {
    sx += s.sky.x;
    sy += s.sky.y;
    sz += s.sky.z;
  }
  const nrm = Math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-9;
  const boresight = { x: sx / nrm, y: sy / nrm, z: sz / nrm };

  const ps = psiMax;
  const dEffKm = 2 * EARTH_R_KM * ps;
  const slantM = params.shellKm * 1000;
  const spotM = 1.22 * lambdaM * slantM / Math.max(1, dEffKm * 1000);
  const panelHalfRad = params.panelHalfDeg * DEG;
  const coveragePct = 100 * nActive * (1 - Math.cos(panelHalfRad));

  const sigma = params.phaseJitDeg * DEG;
  const coherenceLoss = Math.exp(-(sigma * sigma));

  return {
    lambdaM,
    psiMax,
    visCap,
    nActive,
    satellites,
    dEffKm,
    spotM,
    coherenceLoss,
    coveragePct,
    boresight,
    targetPos,
  };
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
panel3d.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1320);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
camera.position.set(0, -27, 16);
camera.lookAt(0, 0, 0);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.06;
orbit.target.set(0, 0, 0);
orbit.minDistance = 10;
orbit.maxDistance = 90;

scene.add(new THREE.AmbientLight(0x99aacc, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(15, -20, 18);
scene.add(dirLight);

const earthTexture = new THREE.TextureLoader().load('./earth_map_8k.jpg');
earthTexture.colorSpace = THREE.SRGBColorSpace;
earthTexture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
earthTexture.minFilter = THREE.LinearMipmapLinearFilter;
earthTexture.magFilter = THREE.LinearFilter;
const earth = new THREE.Mesh(
  new THREE.SphereGeometry(8, 64, 64),
  new THREE.MeshStandardMaterial({ map: earthTexture, roughness: 0.95, metalness: 0.02 })
);
scene.add(earth);

const target = new THREE.Mesh(
  new THREE.SphereGeometry(0.15, 20, 20),
  new THREE.MeshBasicMaterial({ color: 0xff5544 })
);
scene.add(target);

let satPoints = null;
let satLines = null;

function update3D(model) {
  if (satPoints) scene.remove(satPoints);
  if (satLines) scene.remove(satLines);

  const hScale = 8 / EARTH_R_KM;
  target.position.set(
    model.targetPos.x * hScale,
    model.targetPos.y * hScale,
    model.targetPos.z * hScale
  );
  const satGeom = new THREE.BufferGeometry();
  const arr = new Float32Array(model.satellites.length * 3);
  const col = new Float32Array(model.satellites.length * 3);

  const lineVerts = [];
  const targetScaled = target.position;

  for (let i = 0; i < model.satellites.length; i += 1) {
    const s = model.satellites[i];
    const p = s.satPos;
    const x = p.x * hScale;
    const y = p.y * hScale;
    const z = p.z * hScale;
    arr[3 * i] = x;
    arr[3 * i + 1] = y;
    arr[3 * i + 2] = z;

    const hue = ((s.phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (2 * Math.PI);
    const rgb = hsvToRgb(hue, 0.65, 0.95);
    col[3 * i] = rgb.r / 255;
    col[3 * i + 1] = rgb.g / 255;
    col[3 * i + 2] = rgb.b / 255;

    lineVerts.push(x, y, z, targetScaled.x, targetScaled.y, targetScaled.z);
  }

  satGeom.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  satGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  satPoints = new THREE.Points(
    satGeom,
    new THREE.PointsMaterial({ size: 0.16, vertexColors: true, sizeAttenuation: true })
  );
  scene.add(satPoints);

  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(lineVerts, 3));
  satLines = new THREE.LineSegments(
    lineGeom,
    new THREE.LineBasicMaterial({ color: 0x66ccff, transparent: true, opacity: 0.15 })
  );
  scene.add(satLines);
}

function resize3D() {
  const w = panel3d.clientWidth;
  const h = panel3d.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function drawSinglePanel(model, params) {
  const ctx = singleCanvas.getContext('2d');
  const w = singleCanvas.width = singleCanvas.clientWidth * window.devicePixelRatio;
  const h = singleCanvas.height = singleCanvas.clientHeight * window.devicePixelRatio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const leftW = Math.floor(w * 0.43);
  const cx = leftW * 0.52;
  const cy = h * 0.58;
  const R = Math.min(leftW, h) * 0.34;

  ctx.strokeStyle = 'rgba(140,170,210,0.25)';
  ctx.lineWidth = 1.3;
  for (let i = 1; i <= 4; i += 1) {
    ctx.beginPath();
    ctx.arc(cx, cy, (R * i) / 4, 0, 2 * Math.PI);
    ctx.stroke();
  }

  ctx.strokeStyle = '#6de0ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const beamHalf = params.beamHalfDeg * DEG;
  for (let t = 0; t <= Math.PI; t += Math.PI / 360) {
    const gain = Math.pow(Math.cos(Math.min(t, Math.PI / 2)), 2 / Math.max(beamHalf, 0.01));
    const rr = R * Math.max(0.02, gain);
    const x = cx + rr * Math.cos(t - Math.PI / 2);
    const y = cy + rr * Math.sin(t - Math.PI / 2);
    if (t === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#9ec8ff';
  ctx.font = `${12 * window.devicePixelRatio}px IBM Plex Sans`;
  ctx.fillText('Single-sat far-field (representative)', cx - R * 0.9, cy + R + 22 * window.devicePixelRatio);

  const rx = leftW + (w - leftW) * 0.5;
  const ry = h * 0.53;
  const PR = Math.min((w - leftW) * 0.46, h * 0.4);

  ctx.strokeStyle = 'rgba(140,170,210,0.2)';
  ctx.beginPath();
  ctx.arc(rx, ry, PR, 0, 2 * Math.PI);
  ctx.stroke();

  for (const s of model.satellites) {
    const r = (s.theta / Math.max(model.satellites[0] ? Math.max(...model.satellites.map(v => v.theta)) : 1, 1e-6)) * PR;
    const x = rx + r * Math.cos(s.az);
    const y = ry + r * Math.sin(s.az);
    const hue = ((s.phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (2 * Math.PI);
    const rgb = hsvToRgb(hue, 0.85, 1.0);
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},0.85)`;
    const rr = (1.4 + 2.6 * s.amp) * window.devicePixelRatio;
    ctx.beginPath();
    ctx.arc(x, y, rr, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.fillStyle = '#9ec8ff';
  ctx.fillText('Active satellite sky projection', rx - PR * 0.78, ry + PR + 22 * window.devicePixelRatio);
}

function drawGround(model, params) {
  const ctx = groundCanvas.getContext('2d');
  const w = groundCanvas.width = groundCanvas.clientWidth * window.devicePixelRatio;
  const h = groundCanvas.height = groundCanvas.clientHeight * window.devicePixelRatio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const nx = 170;
  const ny = 95;
  const mapW = w * 0.96;
  const mapH = h * 0.85;
  const ox = (w - mapW) / 2;
  const oy = h * 0.08;

  const lambda = model.lambdaM;
  const k = 2 * Math.PI / lambda;
  const extentM = clamp(model.spotM * 2500, 50, 150000);

  const data = new Float32Array(nx * ny);
  let pmax = 0;

  for (let iy = 0; iy < ny; iy += 1) {
    const y = ((iy / (ny - 1)) - 0.5) * extentM;
    for (let ix = 0; ix < nx; ix += 1) {
      const x = ((ix / (nx - 1)) - 0.5) * extentM;
      let er = 0;
      let ei = 0;
      for (const s of model.satellites) {
        const ph = k * (s.sky.x * x + s.sky.y * y) + s.phase;
        er += s.amp * Math.cos(ph);
        ei += s.amp * Math.sin(ph);
      }
      const p = er * er + ei * ei;
      data[iy * nx + ix] = p;
      if (p > pmax) pmax = p;
    }
  }

  const image = ctx.createImageData(nx, ny);
  for (let i = 0; i < data.length; i += 1) {
    const v = Math.pow(data[i] / (pmax + 1e-9), 0.35);
    const hue = (0.68 - 0.7 * v + 1) % 1;
    const rgb = hsvToRgb(hue, 0.88, Math.max(0.15, v));
    image.data[4 * i] = rgb.r;
    image.data[4 * i + 1] = rgb.g;
    image.data[4 * i + 2] = rgb.b;
    image.data[4 * i + 3] = 255;
  }

  const tmp = document.createElement('canvas');
  tmp.width = nx;
  tmp.height = ny;
  tmp.getContext('2d').putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, ox, oy, mapW, mapH);

  ctx.strokeStyle = 'rgba(190,220,255,0.35)';
  ctx.lineWidth = 1.2 * window.devicePixelRatio;
  ctx.strokeRect(ox, oy, mapW, mapH);

  ctx.fillStyle = '#c7e4ff';
  ctx.font = `${12 * window.devicePixelRatio}px IBM Plex Sans`;
  ctx.fillText('Centered focus target', ox + 10 * window.devicePixelRatio, oy + 18 * window.devicePixelRatio);

  const cx = ox + mapW / 2;
  const cy = oy + mapH / 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.beginPath();
  ctx.moveTo(cx - 10 * window.devicePixelRatio, cy);
  ctx.lineTo(cx + 10 * window.devicePixelRatio, cy);
  ctx.moveTo(cx, cy - 10 * window.devicePixelRatio);
  ctx.lineTo(cx, cy + 10 * window.devicePixelRatio);
  ctx.stroke();

  const kmPerPix = (extentM / 1000) / mapW;
  const barKm = Math.max(0.02, model.spotM / 1000);
  const barPx = barKm / kmPerPix;
  const bx = ox + mapW - barPx - 20 * window.devicePixelRatio;
  const by = oy + mapH - 20 * window.devicePixelRatio;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2 * window.devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(bx + barPx, by);
  ctx.stroke();
  ctx.fillStyle = '#d6ebff';
  ctx.fillText(`${barKm.toFixed(2)} km est. spot scale`, bx, by - 6 * window.devicePixelRatio);
}

function showValues(params, model) {
  els.freqVal.textContent = `${params.freqGHz.toFixed(1)}`;
  els.shellVal.textContent = `${params.shellKm.toFixed(0)}`;
  els.beamHalfVal.textContent = `${params.panelHalfDeg.toFixed(1)}`;
  els.phaseJitVal.textContent = `${params.phaseJitDeg.toFixed(1)}`;
  els.seedVal.textContent = `${params.seed}`;
  els.presetLabel.textContent = els.preset.options[els.preset.selectedIndex].text;

  const spotCm = model.spotM * 100;
  els.stats.innerHTML = [
    `<div><b>Active sats (all in view):</b> ${model.nActive}</div>`,
    `<div><b>Visibility cap (elev >= 0 deg):</b> ${model.visCap}</div>`,
    `<div><b>Target:</b> San Francisco (${TARGET_LAT_DEG.toFixed(4)}, ${TARGET_LON_DEG.toFixed(4)})</div>`,
    `<div><b>Implied sky coverage:</b> ${model.coveragePct.toFixed(2)}%</div>`,
    `<div><b>Effective footprint diameter:</b> ${(2 * EARTH_R_KM * model.psiMax).toFixed(0)} km</div>`,
    `<div><b>Estimated spot scale:</b> ${spotCm.toFixed(2)} cm</div>`,
    `<div><b>Coherence factor exp(-sigma^2):</b> ${model.coherenceLoss.toFixed(3)}</div>`,
  ].join('');
}

function readParams() {
  return {
    freqGHz: parseFloat(els.freq.value),
    shellKm: parseFloat(els.shell.value),
    panelHalfDeg: parseFloat(els.beamHalf.value),
    phaseJitDeg: parseFloat(els.phaseJit.value),
    seed: parseInt(els.seed.value, 10),
    totalConstellation: 29988,
  };
}

function applyPreset(name) {
  if (name === 'conservative') {
    els.freq.value = '12';
    els.shell.value = '550';
    els.beamHalf.value = '1.4';
    els.phaseJit.value = '20';
  } else if (name === 'aggressive') {
    els.freq.value = '20';
    els.shell.value = '550';
    els.beamHalf.value = '1.8';
    els.phaseJit.value = '10';
  } else {
    els.freq.value = '30';
    els.shell.value = '550';
    els.beamHalf.value = '2.0';
    els.phaseJit.value = '3.0';
  }
}

let currentModel = null;
function updateAll() {
  const params = readParams();
  const model = computeModel(params);
  currentModel = model;
  showValues(params, model);
  update3D(model);
  drawSinglePanel(model, params);
  drawGround(model, params);
}

const controls = [
  els.freq, els.shell,
  els.beamHalf, els.phaseJit, els.seed,
];

for (const c of controls) c.addEventListener('input', updateAll);
els.preset.addEventListener('change', () => {
  applyPreset(els.preset.value);
  updateAll();
});

window.addEventListener('resize', () => {
  resize3D();
  if (currentModel) {
    drawSinglePanel(currentModel, readParams());
    drawGround(currentModel, readParams());
  }
});

function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
}

applyPreset('meme');
resize3D();
updateAll();
animate();
