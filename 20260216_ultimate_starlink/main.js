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
  const hh = Number.isFinite(h) ? ((h % 1) + 1) % 1 : 0;
  const ss = Number.isFinite(s) ? clamp(s, 0, 1) : 0;
  const vv = Number.isFinite(v) ? clamp(v, 0, 1) : 0;
  const i = Math.floor(hh * 6);
  const f = hh * 6 - i;
  const p = vv * (1 - ss);
  const q = vv * (1 - f * ss);
  const t = vv * (1 - (1 - f) * ss);
  const m = ((i % 6) + 6) % 6;
  const arr = [
    [vv, t, p],
    [q, vv, p],
    [p, vv, t],
    [p, q, vv],
    [t, p, vv],
    [vv, p, q],
  ][m] || [0, 0, 0];
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
  const k = 2 * Math.PI / lambdaM;
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
    // Command phase to focus at the target point (modulo 2pi), plus residual jitter.
    const commandPhase = ((-k * norm) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
    return {
      dir: d,
      satPos,
      sky: { x: ux, y: uy, z: uz },
      skyLocal: {
        x: ux * localBasis.east.x + uy * localBasis.east.y + uz * localBasis.east.z,
        y: ux * localBasis.north.x + uy * localBasis.north.y + uz * localBasis.north.z,
      },
      amp: 1.0,
      commandPhase,
      phaseError: phaseErr,
      phaseActual: commandPhase + phaseErr,
      rangeM: norm * 1000,
      // Local sky coordinates around the target zenith (for centered projection).
      localAz: 0,
      localTheta: 0,
    };
  });

  for (let i = 0; i < satellites.length; i += 1) {
    const ld = localDirs[i];
    satellites[i].localAz = Math.atan2(ld.y, ld.x);
    satellites[i].localTheta = Math.acos(clamp(ld.z, -1, 1));
  }

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
    localBasis,
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

    const hue = (s.commandPhase % (2 * Math.PI)) / (2 * Math.PI);
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

  const leftW = Math.floor(w * 0.45);
  const cx = leftW * 0.5;
  const cy = h * 0.56;
  const R = Math.min(leftW, h) * 0.33;

  const beamHalf = Math.max(0.2 * DEG, params.beamHalfDeg * DEG);
  const thSteps = 24;
  const phSteps = 36;
  const yaw = 35 * DEG;
  const pitch = -24 * DEG;
  const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const p3 = [];
  for (let it = 0; it <= thSteps; it += 1) {
    const th = (it / thSteps) * (Math.PI * 0.5);
    for (let ip = 0; ip <= phSteps; ip += 1) {
      const ph = (ip / phSteps) * (2 * Math.PI);
      const gain = Math.exp(-Math.log(2) * (th * th) / (beamHalf * beamHalf));
      const rad = R * (0.08 + 0.92 * gain);
      const x0 = rad * Math.sin(th) * Math.cos(ph);
      const y0 = rad * Math.sin(th) * Math.sin(ph);
      const z0 = rad * Math.cos(th);

      const x1 = cyaw * x0 + syaw * z0;
      const z1 = -syaw * x0 + cyaw * z0;
      const y2 = cp * y0 - sp * z1;
      const z2 = sp * y0 + cp * z1;
      const u = cx + x1 * (1 + 0.18 * z2 / R);
      const v = cy + y2 * (1 + 0.18 * z2 / R);
      p3.push({ it, ip, u, v, gain });
    }
  }

  ctx.strokeStyle = 'rgba(120,190,235,0.35)';
  ctx.lineWidth = 1.1 * window.devicePixelRatio;
  for (let it = 0; it <= thSteps; it += 3) {
    ctx.beginPath();
    for (let ip = 0; ip <= phSteps; ip += 1) {
      const p = p3[it * (phSteps + 1) + ip];
      if (ip === 0) ctx.moveTo(p.u, p.v);
      else ctx.lineTo(p.u, p.v);
    }
    ctx.stroke();
  }
  for (let ip = 0; ip <= phSteps; ip += 4) {
    ctx.beginPath();
    for (let it = 0; it <= thSteps; it += 1) {
      const p = p3[it * (phSteps + 1) + ip];
      if (it === 0) ctx.moveTo(p.u, p.v);
      else ctx.lineTo(p.u, p.v);
    }
    ctx.stroke();
  }

  for (let i = 0; i < p3.length; i += 1) {
    const p = p3[i];
    const rgb = hsvToRgb(0.58 - 0.28 * p.gain, 0.82, 0.75 + 0.25 * p.gain);
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},0.55)`;
    ctx.fillRect(p.u - 0.7 * window.devicePixelRatio, p.v - 0.7 * window.devicePixelRatio, 1.4 * window.devicePixelRatio, 1.4 * window.devicePixelRatio);
  }

  ctx.fillStyle = '#9ec8ff';
  ctx.font = `${12 * window.devicePixelRatio}px IBM Plex Sans`;
  ctx.fillText('Single-sat far-field (3D lobe)', cx - R * 0.9, cy + R + 22 * window.devicePixelRatio);

  const rx = leftW + (w - leftW) * 0.5;
  const ry = h * 0.53;
  const PR = Math.min((w - leftW) * 0.46, h * 0.4);

  ctx.strokeStyle = 'rgba(140,170,210,0.2)';
  ctx.beginPath();
  ctx.arc(rx, ry, PR, 0, 2 * Math.PI);
  ctx.stroke();

  const maxTheta = Math.max(model.psiMax, 1e-6);
  for (const s of model.satellites) {
    const r = (s.localTheta / maxTheta) * PR;
    const x = rx + r * Math.cos(s.localAz);
    const y = ry + r * Math.sin(s.localAz);
    const hue = (s.commandPhase % (2 * Math.PI)) / (2 * Math.PI);
    const rgb = hsvToRgb(hue, 0.85, 1.0);
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},0.85)`;
    const rr = (1.5 + 2.2 * s.amp) * window.devicePixelRatio;
    ctx.beginPath();
    ctx.arc(x, y, rr, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.fillStyle = '#9ec8ff';
  ctx.fillText('Sky projection (color: command phase, size: power)', rx - PR * 0.98, ry + PR + 22 * window.devicePixelRatio);
}

function drawGround(model, params) {
  const ctx = groundCanvas.getContext('2d');
  const w = groundCanvas.width = groundCanvas.clientWidth * window.devicePixelRatio;
  const h = groundCanvas.height = groundCanvas.clientHeight * window.devicePixelRatio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // Odd grid ensures the exact focus point (x=0,y=0) is sampled at center pixel.
  const nx = 301;
  const ny = 301;
  const mapW = w * 0.96;
  const mapH = h * 0.85;
  const ox = (w - mapW) / 2;
  const oy = h * 0.08;

  const lambda = model.lambdaM;
  const k = 2 * Math.PI / lambda;
  // Show a zoom window around the predicted diffraction spot; otherwise the peak is sub-pixel.
  const extentM = clamp(model.spotM * 80, 0.01, 2000);

  const targetM = {
    x: model.targetPos.x * 1000,
    y: model.targetPos.y * 1000,
    z: model.targetPos.z * 1000,
  };
  const east = model.localBasis.east;
  const north = model.localBasis.north;
  const sats = model.satellites.map((s) => {
    const sx = s.satPos.x * 1000;
    const sy = s.satPos.y * 1000;
    const sz = s.satPos.z * 1000;
    const dx = sx - targetM.x;
    const dy = sy - targetM.y;
    const dz = sz - targetM.z;
    const r0 = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return {
      x: sx, y: sy, z: sz,
      r0,
      amp: s.amp,
      phaseError: s.phaseError,
    };
  });

  const data = new Float64Array(nx * ny);
  let pmax = 0;
  let pmin = Number.POSITIVE_INFINITY;
  let imax = 0;
  let jmax = 0;

  for (let iy = 0; iy < ny; iy += 1) {
    const y = ((iy / (ny - 1)) - 0.5) * extentM;
    for (let ix = 0; ix < nx; ix += 1) {
      const x = ((ix / (nx - 1)) - 0.5) * extentM;
      const px = targetM.x + east.x * x + north.x * y;
      const py = targetM.y + east.y * x + north.y * y;
      const pz = targetM.z + east.z * x + north.z * y;
      let er = 0;
      let ei = 0;
      for (const s of sats) {
        // Exact path-difference phase around the focused target:
        // ph = k*(|r_sat - r_point| - |r_sat - r_target|) + residual phase error.
        const dx = s.x - px;
        const dy = s.y - py;
        const dz = s.z - pz;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const ph = k * (r - s.r0) + s.phaseError;
        er += s.amp * Math.cos(ph);
        ei += s.amp * Math.sin(ph);
      }
      const p = er * er + ei * ei;
      data[iy * nx + ix] = p;
      if (p > pmax) {
        pmax = p;
        imax = ix;
        jmax = iy;
      }
      if (p < pmin) pmin = p;
    }
  }

  // Exact focus-point power at the target location (x=0,y=0), independent of grid sampling.
  let er0 = 0;
  let ei0 = 0;
  for (const s of sats) {
    const ph0 = s.phaseError;
    er0 += s.amp * Math.cos(ph0);
    ei0 += s.amp * Math.sin(ph0);
  }
  const pCenterExact = er0 * er0 + ei0 * ei0;
  const cx = (nx - 1) / 2;
  const cy = (ny - 1) / 2;
  const pCenterGrid = data[cy * nx + cx];

  const image = ctx.createImageData(nx, ny);
  const pFloor = Math.max(pmin, pmax * 1e-12);
  const lmax = Math.log10(pmax + 1e-20);
  const lmin = Math.log10(pFloor);
  const lspan = Math.max(1e-9, lmax - lmin);
  for (let i = 0; i < data.length; i += 1) {
    const lv = (Math.log10(Math.max(data[i], pFloor)) - lmin) / lspan;
    const v = clamp(lv, 0, 1);
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

  const centerX = ox + mapW / 2;
  const centerY = oy + mapH / 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.beginPath();
  ctx.moveTo(centerX - 10 * window.devicePixelRatio, centerY);
  ctx.lineTo(centerX + 10 * window.devicePixelRatio, centerY);
  ctx.moveTo(centerX, centerY - 10 * window.devicePixelRatio);
  ctx.lineTo(centerX, centerY + 10 * window.devicePixelRatio);
  ctx.stroke();

  // Mark global peak location.
  const peakX = ox + (imax / (nx - 1)) * mapW;
  const peakY = oy + (jmax / (ny - 1)) * mapH;
  ctx.strokeStyle = 'rgba(255,220,0,0.95)';
  ctx.lineWidth = 1.5 * window.devicePixelRatio;
  ctx.beginPath();
  ctx.arc(peakX, peakY, 6 * window.devicePixelRatio, 0, 2 * Math.PI);
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

  // Return diagnostics for stats panel.
  const dxPix = imax - cx;
  const dyPix = jmax - cy;
  return {
    pCenterExact,
    pCenterGrid,
    pMax: pmax,
    peakOffsetM: Math.sqrt(dxPix * dxPix + dyPix * dyPix) * (extentM / (nx - 1)),
    peakOffsetPix: Math.sqrt(dxPix * dxPix + dyPix * dyPix),
    extentM,
    nx,
    ny,
  };
}

function showValues(params, model, groundDiag = null) {
  els.freqVal.textContent = `${params.freqGHz.toFixed(1)}`;
  els.shellVal.textContent = `${params.shellKm.toFixed(0)}`;
  els.beamHalfVal.textContent = `${params.panelHalfDeg.toFixed(1)}`;
  els.phaseJitVal.textContent = `${params.phaseJitDeg.toFixed(1)}`;
  els.seedVal.textContent = `${params.seed}`;
  els.presetLabel.textContent = els.preset.options[els.preset.selectedIndex].text;

  const spotCm = model.spotM * 100;
  const diagLines = groundDiag ? [
    `<div><b>P(0,0) exact / Pmax:</b> ${(groundDiag.pCenterExact / (groundDiag.pMax + 1e-12)).toFixed(4)}</div>`,
    `<div><b>P(center pixel) / Pmax:</b> ${(groundDiag.pCenterGrid / (groundDiag.pMax + 1e-12)).toFixed(4)}</div>`,
    `<div><b>Peak offset from center:</b> ${groundDiag.peakOffsetM.toExponential(3)} m (${groundDiag.peakOffsetPix.toFixed(2)} px)</div>`,
  ] : [];

  els.stats.innerHTML = [
    `<div><b>Active sats (all in view):</b> ${model.nActive}</div>`,
    `<div><b>Visibility cap (elev >= 0 deg):</b> ${model.visCap}</div>`,
    `<div><b>Target:</b> San Francisco (${TARGET_LAT_DEG.toFixed(4)}, ${TARGET_LON_DEG.toFixed(4)})</div>`,
    `<div><b>Implied sky coverage:</b> ${model.coveragePct.toFixed(2)}%</div>`,
    `<div><b>Effective footprint diameter:</b> ${(2 * EARTH_R_KM * model.psiMax).toFixed(0)} km</div>`,
    `<div><b>Estimated spot scale:</b> ${spotCm.toFixed(2)} cm</div>`,
    `<div><b>Coherence factor exp(-sigma^2):</b> ${model.coherenceLoss.toFixed(3)}</div>`,
    ...diagLines,
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
    els.freq.value = '11.7';
    els.shell.value = '550';
    els.beamHalf.value = '1.4';
    els.phaseJit.value = '0';
  } else if (name === 'aggressive') {
    els.freq.value = '19.05';
    els.shell.value = '550';
    els.beamHalf.value = '1.8';
    els.phaseJit.value = '0';
  } else {
    els.freq.value = '30.0';
    els.shell.value = '550';
    els.beamHalf.value = '2.0';
    els.phaseJit.value = '0';
  }
}

let currentModel = null;
function updateAll() {
  const params = readParams();
  const model = computeModel(params);
  currentModel = model;
  const groundDiag = drawGround(model, params);
  showValues(params, model, groundDiag);
  update3D(model);
  drawSinglePanel(model, params);
}

const controls = [
  els.freq, els.shell,
  els.beamHalf, els.phaseJit, els.seed,
];

for (const c of controls) {
  c.addEventListener('input', updateAll);
  c.addEventListener('change', updateAll);
}
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
