import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const ORIGINAL_MODEL = {
  id: "original",
  order: -1,
  label: "Original",
  shortLabel: "Original",
  path: "../data/cow.obj",
  rmse: null,
  vertices: null,
  faces: null,
};

const canvas = document.getElementById("viewer-canvas");
const versionGrid = document.getElementById("version-grid");
const statusPill = document.getElementById("status-pill");
const metricName = document.getElementById("metric-name");
const metricRmse = document.getElementById("metric-rmse");
const metricVertices = document.getElementById("metric-vertices");
const metricFaces = document.getElementById("metric-faces");
const bboxLabel = document.getElementById("bbox-label");
const shellLabel = document.getElementById("shell-label");

const scene = new THREE.Scene();
scene.background = new THREE.Color("#efe7d6");
scene.fog = new THREE.Fog("#efe7d6", 7.5, 16);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(2.6, 1.5, 2.6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);
controls.screenSpacePanning = true;
controls.minDistance = 0.6;
controls.maxDistance = 14;

scene.add(new THREE.HemisphereLight("#fff8e8", "#837463", 1.2));

const keyLight = new THREE.DirectionalLight("#fffdf6", 1.35);
keyLight.position.set(4.8, 6.5, 5.2);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight("#c5d3b0", 0.5);
fillLight.position.set(-3.5, 2.5, -2.5);
scene.add(fillLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(9, 96),
  new THREE.MeshStandardMaterial({
    color: "#d8ccb4",
    transparent: true,
    opacity: 0.72,
    roughness: 0.96,
    metalness: 0.02,
  })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.62;
floor.receiveShadow = true;
scene.add(floor);

const loader = new OBJLoader();
const modelState = new Map();
let metrics = null;
let models = [];
let currentMode = "single";
let currentSelection = "original";

function setStatus(message) {
  statusPill.textContent = message;
}

async function maybeLoadMetrics() {
  try {
    const response = await fetch("../outputs/metrics.json");
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch {
    return null;
  }
}

async function discoverOutputOrders() {
  try {
    const response = await fetch("../outputs/");
    if (!response.ok) {
      return [];
    }
    const html = await response.text();
    const matches = [...html.matchAll(/cow_reconstruction_l(\d+)\.obj/g)];
    const unique = new Set(matches.map((match) => Number(match[1])));
    return [...unique].sort((a, b) => a - b);
  } catch {
    return [];
  }
}

function buildModels(metricsData, discoveredOrders) {
  const metricOrders = metricsData ? Object.keys(metricsData.orders || {}).map(Number) : [];
  const availableOrders = [...new Set([...metricOrders, ...discoveredOrders])].sort((a, b) => a - b);
  const derived = [ORIGINAL_MODEL];

  for (const order of availableOrders) {
    const info = metricsData?.orders?.[String(order)] || null;
    derived.push({
      id: `l${order}`,
      order,
      label: `Reconstruction l <= ${order}`,
      shortLabel: `l=${order}`,
      path: `../outputs/cow_reconstruction_l${order}.obj`,
      rmse: info ? info.surface_rmse : null,
      vertices: info ? info.vertex_count : null,
      faces: info ? info.face_count : null,
    });
  }

  return derived;
}

function updateTopBar(metricsData) {
  if (!metricsData) {
    bboxLabel.textContent = "Unavailable";
    shellLabel.textContent = "No metrics loaded";
    return;
  }

  const bbox = metricsData.prepared_bbox.map((value) => value.toFixed(4));
  bboxLabel.textContent = `${bbox[0]} × ${bbox[1]} × ${bbox[2]}`;
  const shell = metricsData.star_shell_diagnostics;
  shellLabel.textContent = `${(shell.single_hit_fraction * 100).toFixed(1)}% single-hit rays`;
}

function createVersionButtons() {
  versionGrid.innerHTML = "";
  for (const model of models) {
    const button = document.createElement("button");
    button.className = "version-chip";
    button.textContent = model.shortLabel;
    button.dataset.modelId = model.id;
    button.addEventListener("click", () => {
      currentSelection = model.id;
      currentMode = "single";
      syncModeButtons();
      syncVersionButtons();
      layoutScene();
    });
    versionGrid.appendChild(button);
  }
  syncVersionButtons();
}

function syncVersionButtons() {
  for (const button of versionGrid.querySelectorAll(".version-chip")) {
    button.classList.toggle("active", button.dataset.modelId === currentSelection && currentMode === "single");
  }
}

function syncModeButtons() {
  for (const button of document.querySelectorAll(".mode-chip")) {
    button.classList.toggle("active", button.dataset.mode === currentMode);
  }
}

function createMeshGroup(object, label) {
  const group = new THREE.Group();
  group.name = label;
  let vertexCount = 0;
  let faceCount = 0;

  const color = label === "Original" ? "#6e7869" : "#b9c3b0";
  const edgeColor = label === "Original" ? "#364137" : "#6f7a68";
  const baseMaterial = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.78,
    metalness: 0.02,
    side: THREE.FrontSide,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: edgeColor,
    transparent: true,
    opacity: 0.32,
  });

  object.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    child.geometry.computeVertexNormals();
    child.geometry.computeBoundingBox();
    const positions = child.geometry.getAttribute("position");
    vertexCount += positions ? positions.count : 0;
    faceCount += child.geometry.index ? child.geometry.index.count / 3 : (positions ? positions.count / 3 : 0);
    child.castShadow = true;
    child.receiveShadow = true;
    child.material = baseMaterial;
    child.add(new THREE.LineSegments(new THREE.EdgesGeometry(child.geometry, 28), edgeMaterial));
  });

  group.add(object);
  return { group, vertexCount, faceCount };
}

async function loadModels() {
  const available = [];

  for (const model of models) {
    setStatus(`Loading ${model.shortLabel}...`);
    try {
      const object = await loader.loadAsync(model.path);
      const { group, vertexCount, faceCount } = createMeshGroup(object, model.label);
      scene.add(group);
      if (model.vertices == null) {
        model.vertices = vertexCount;
      }
      if (model.faces == null) {
        model.faces = Math.round(faceCount);
      }
      modelState.set(model.id, { group, label: model.label });
      available.push(model);
    } catch (error) {
      console.warn(`Skipping ${model.path}`, error);
    }
  }

  models = available;
  if (!models.length) {
    throw new Error("No viewable models were found.");
  }
  if (!models.some((model) => model.id === currentSelection)) {
    currentSelection = models[models.length - 1].id;
  }
  setStatus("Ready");
}

function getVisibleIds() {
  if (currentMode === "grid") {
    return models.map((model) => model.id);
  }
  return [currentSelection];
}

function updateSelectionMetrics() {
  const model = models.find((entry) => entry.id === currentSelection) ?? models[0];
  metricName.textContent = model.label;
  metricRmse.textContent = model.rmse == null ? "Reference" : model.rmse.toFixed(4);
  metricVertices.textContent = model.vertices == null ? "-" : `${model.vertices}`;
  metricFaces.textContent = model.faces == null ? "-" : `${model.faces}`;
}

function computeBounds(ids) {
  const box = new THREE.Box3();
  for (const id of ids) {
    const entry = modelState.get(id);
    if (entry) {
      box.expandByObject(entry.group);
    }
  }
  return box;
}

function frameSelection(ids) {
  const box = computeBounds(ids);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxSize = Math.max(size.x, size.y, size.z, 0.5);
  const distance = maxSize / (2 * Math.tan((camera.fov * Math.PI) / 360));

  controls.target.copy(center);
  camera.position.copy(center.clone().add(new THREE.Vector3(distance * 1.15, distance * 0.65, distance * 1.1)));
  camera.near = Math.max(0.01, distance / 100);
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function addCaptionSprite(text) {
  const canvas2d = document.createElement("canvas");
  canvas2d.width = 256;
  canvas2d.height = 64;
  const ctx = canvas2d.getContext("2d");
  ctx.clearRect(0, 0, canvas2d.width, canvas2d.height);
  ctx.fillStyle = "rgba(255, 249, 240, 0.92)";
  ctx.fillRect(0, 0, canvas2d.width, canvas2d.height);
  ctx.strokeStyle = "rgba(61, 69, 59, 0.18)";
  ctx.strokeRect(1, 1, canvas2d.width - 2, canvas2d.height - 2);
  ctx.fillStyle = "#213021";
  ctx.font = "600 26px Avenir Next, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas2d.width / 2, canvas2d.height / 2);
  const texture = new THREE.CanvasTexture(canvas2d);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.7, 0.175, 1);
  return sprite;
}

function clearLabels(group) {
  group.children
    .filter((child) => child.userData.labelSprite)
    .forEach((child) => {
      group.remove(child);
      child.material.map.dispose();
      child.material.dispose();
    });
}

function layoutScene() {
  const ids = getVisibleIds();
  const total = models.length;
  const cols = total > 20 ? 5 : total > 9 ? 4 : 3;
  const rows = Math.ceil(total / cols);
  const strideX = cols > 4 ? 1.28 : cols > 3 ? 1.55 : 1.8;
  const strideZ = rows > 4 ? 1.08 : rows > 3 ? 1.28 : 1.45;

  models.forEach((model, index) => {
    const entry = modelState.get(model.id);
    if (!entry) {
      return;
    }

    const { group } = entry;
    group.visible = ids.includes(model.id);
    clearLabels(group);

    if (currentMode === "grid") {
      const col = index % cols;
      const row = Math.floor(index / cols);
      const x = (col - (cols - 1) / 2) * strideX;
      const z = (row - (rows - 1) / 2) * strideZ;
      group.position.set(x, 0, z);
      const label = addCaptionSprite(model.shortLabel);
      label.position.set(0, cols > 4 ? 0.62 : 0.72, 0);
      label.userData.labelSprite = true;
      group.add(label);
    } else {
      group.position.set(0, 0, 0);
    }
  });

  updateSelectionMetrics();
  frameSelection(ids);
  const selected = models.find((model) => model.id === currentSelection);
  setStatus(currentMode === "grid" ? "Showing all available versions" : `Showing ${selected?.label ?? currentSelection}`);
}

function resizeRenderer() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}

function animate() {
  requestAnimationFrame(animate);
  resizeRenderer();
  controls.update();
  renderer.render(scene, camera);
}

function bindUi() {
  document.querySelectorAll(".mode-chip").forEach((button) => {
    button.addEventListener("click", () => {
      currentMode = button.dataset.mode;
      syncModeButtons();
      syncVersionButtons();
      layoutScene();
    });
  });

  window.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "g") {
      currentMode = currentMode === "grid" ? "single" : "grid";
      syncModeButtons();
      syncVersionButtons();
      layoutScene();
      return;
    }

    if (event.key.toLowerCase() === "f") {
      frameSelection(getVisibleIds());
      return;
    }

    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      const delta = event.key === "ArrowRight" ? 1 : -1;
      const index = models.findIndex((model) => model.id === currentSelection);
      const next = (index + delta + models.length) % models.length;
      currentSelection = models[next].id;
      currentMode = "single";
      syncModeButtons();
      syncVersionButtons();
      layoutScene();
    }
  });

  window.addEventListener("resize", resizeRenderer);
}

async function main() {
  try {
    bindUi();
    metrics = await maybeLoadMetrics();
    updateTopBar(metrics);
    const discoveredOrders = await discoverOutputOrders();
    models = buildModels(metrics, discoveredOrders);
    currentSelection = models[models.length - 1]?.id ?? "original";
    await loadModels();
    createVersionButtons();
    syncModeButtons();
    syncVersionButtons();
    layoutScene();
    animate();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load viewer");
    metricName.textContent = "Load error";
    metricRmse.textContent = "-";
    metricVertices.textContent = "-";
    metricFaces.textContent = "-";
  }
}

main();
