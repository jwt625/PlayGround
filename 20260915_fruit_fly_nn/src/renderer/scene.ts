import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/**
 * Display scaling is presentation only. All simulation/physics values stay in
 * SI; the renderer multiplies lengths by `scale` to keep the tiny array visible.
 */
export const DISPLAY = {
  scale: 6000,
  targetDistance: 100,
  domeRadius: 22,
  sectionSize: 44,
};

export interface SceneBundle {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  onFrame: (dt_s: number) => void;
  start: () => void;
}

export function createScene(canvas: HTMLCanvasElement, onFrame: (dt: number) => void): SceneBundle {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070c);
  scene.fog = new THREE.Fog(0x05070c, 200, 700);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 3000);
  camera.position.set(70, 55, 140);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 55);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 10;
  controls.maxDistance = 600;

  scene.add(new THREE.AmbientLight(0x8899bb, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(60, 80, 120);
  scene.add(key);

  const grid = new THREE.GridHelper(400, 40, 0x1b2533, 0x101722);
  grid.position.y = -0.001;
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  let last = performance.now();
  const bundle: SceneBundle = {
    renderer,
    scene,
    camera,
    controls,
    onFrame,
    start: () => {
      const loop = () => {
        const now = performance.now();
        const dt = (now - last) / 1000;
        last = now;
        controls.update();
        onFrame(dt);
        renderer.render(scene, camera);
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    },
  };
  return bundle;
}
