// Large multi-unit-cell lattice film (x-cut) with the domain front. Schematic scale: 16 cells per domain period.
const T_FLIP_OV = 0.05; // ms of pulse time for one atom's flip
const smooth01 = (u) => { u = Math.min(1, Math.max(0, u)); return u * u * (3 - 2 * u); };

class Overview {
  // x-cut film: crystal z is in the plane, along the electrode-to-electrode direction (field). Crystal frame (x,y,z)
  // maps to world (Xw, Yw, Zw) = (S - y, Lz - z, x): +V electrode at Yw = 0 (crystal +z points toward it), surface normal = crystal x.
  constructor(container, track) {
    const { a, c, h, cart, SP } = LAT;
    this.S = 16 * a; this.D = 3 * a; this.Lz = 13 * c; this.EXAG = 2;
    const S = this.S, D = this.D, Lz = this.Lz, LG = DOM.LG;
    const W = (p) => [S - p[1], Lz - p[2], p[0]], Wd = (d) => [-d[1], -d[2], d[0]];
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(0x0e1013);
    this.camera = new THREE.PerspectiveCamera(32, 1, 5, 3000); this.camera.up.set(0, 0, 1);
    const target = new THREE.Vector3(S / 2, Lz * 0.5, D * 0.5);
    this.camera.position.set(S / 2 + 150, Lz * 0.5 - 215, D + 150);
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(target); this.controls.autoRotate = true; this.controls.autoRotateSpeed = 0.8;
    this.controls.enableDamping = true; this.controls.update();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const d1 = new THREE.DirectionalLight(0xffffff, 0.9); d1.position.set(1, -1, 2); this.scene.add(d1);
    const d2 = new THREE.DirectionalLight(0xffffff, 0.35); d2.position.set(-1, 1, 0.5); this.scene.add(d2);

    // atoms (selected in the crystal frame, stored in the world frame)
    const nRows = Math.ceil(S / h) + 3;
    const test = (fr, p) => p[0] >= 0 && p[0] < D && p[1] >= 0 && p[1] < S && p[2] >= 0 && p[2] < Lz;
    this.atoms = LAT.gen(test, [-3, Math.ceil(nRows / 2) + 6], [-1, nRows], [0, 13]);
    const tTrigOf = (xn, yn, zf) => {
      let g = DOM.g(xn, yn);
      if (!isFinite(g)) return Infinity;
      g = Math.min(1, g + 0.04 * (1 - zf));
      return WF.tOfF(Math.min(DOM.fOfG(g), 0.999));
    };
    const ttCrystal = (p) => tTrigOf((S - p[1]) / S - 0.5, (Lz - p[2]) / Lz * LG, p[0] / D);
    const geo = new THREE.SphereGeometry(1, 10, 7);
    this.meshes = {}; const counts = { Li: 0, Nb: 0, O: 0 };
    for (const at of this.atoms) {
      at.k = counts[at.sp]++; at.tt = ttCrystal(at.p); at.s = -1;
      at.p = W(at.p); at.d = Wd(at.d);
    }
    for (const sp of ['Li', 'Nb', 'O']) {
      const m = new THREE.InstancedMesh(geo, new THREE.MeshPhongMaterial({ color: SP[sp].color, shininess: 70, specular: 0x444444 }), counts[sp]);
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.frustumCulled = false;
      this.meshes[sp] = m; this.scene.add(m);
    }
    this.updateAtoms(0, true);

    // per-cell polarization arrows (crystal +z = world -Y)
    const cyl = new THREE.CylinderGeometry(0.55, 0.55, 5, 8, 1).rotateX(Math.PI / 2).translate(0, 0, -1.75);
    const cone = new THREE.ConeGeometry(1.5, 3.5, 10, 1).rotateX(Math.PI / 2).translate(0, 0, 2.5);
    const cl = [];
    for (let k = 0; k < 13; k++) for (let j = -1; j <= nRows; j++) for (let i = -3; i <= Math.ceil(nRows / 2) + 6; i++) {
      const pc = cart([i + 0.5, j + 0.5, k + 0.5]);
      if (pc[0] < 0 || pc[0] >= D || pc[1] < 0 || pc[1] >= S || pc[2] < 0 || pc[2] >= Lz) continue;
      cl.push({ p: W(pc), xc: pc[0], tt: ttCrystal(pc), s: -1, i, j, k });
    }
    this.cells = cl;
    const am = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    this.arrows = new THREE.InstancedMesh(mergeGeoms([cyl, cone]), am, cl.length);
    this.arrows.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.arrows.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cl.length * 3), 3);
    this.arrows.frustumCulled = false; this.scene.add(this.arrows);
    this.cUp = new THREE.Color(0xe0a24a); this.cDn = new THREE.Color(0x3b78e0); this.tmp = new THREE.Color();
    this.updateArrows(0, true);

    // domain overlay on the top (x) surface: same mask as the top view, stripe i = 2
    this.ovC = document.createElement('canvas'); this.ovC.width = 100; this.ovC.height = Math.round(100 * LG);
    this.ovX = this.ovC.getContext('2d'); this.ovI = this.ovX.createImageData(this.ovC.width, this.ovC.height);
    this.ovG = new Float32Array(this.ovC.width * this.ovC.height);
    for (let r = 0; r < this.ovC.height; r++) for (let q = 0; q < this.ovC.width; q++)
      this.ovG[r * this.ovC.width + q] = DOM.g((q + 0.5) / this.ovC.width - 0.5, (r + 0.5) / this.ovC.height * LG);
    this.ovTex = new THREE.CanvasTexture(this.ovC); this.ovTex.minFilter = THREE.LinearFilter; this.lastH = NaN;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(S, Lz), new THREE.MeshBasicMaterial({ map: this.ovTex,
      transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    plane.position.set(S / 2, Lz / 2, D + 0.4); plane.renderOrder = 5; this.scene.add(plane);
    // slab outline and electrodes
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S, Lz, D)),
      new THREE.LineBasicMaterial({ color: 0x4a525b }));
    box.position.set(S / 2, Lz / 2, D / 2); this.scene.add(box);
    const eg = new THREE.BoxGeometry(0.4 * S, 20, 2);
    this.eTop = new THREE.Mesh(eg, new THREE.MeshPhongMaterial({ color: 0x8a9199, shininess: 80 }));
    this.eTop.position.set(S / 2, -10, D + 1);
    this.eBot = new THREE.Mesh(eg, new THREE.MeshPhongMaterial({ color: 0x8a9199, shininess: 80 }));
    this.eBot.position.set(S / 2, Lz + 10, D + 1);
    this.scene.add(this.eTop, this.eBot);

    // tracked unit cell: snap the requested location to the nearest cell in the surface layer
    let best = null, bd = 1e9;
    for (const q of cl) if (q.xc > D * 0.6 && isFinite(q.tt)) {
      const d = Math.hypot(q.p[0] / S - 0.5 - track.x, q.p[1] / Lz * LG - track.y);
      if (d < bd) { bd = d; best = q; }
    }
    track.x = best.p[0] / S - 0.5; track.y = best.p[1] / Lz * LG; track.tt = best.tt; track.cell = best;
    const o = cart([best.i, best.j, best.k]), A = cart([1, 0, 0]), B = cart([0, 1, 0]), C = cart([0, 0, 1]);
    const corners = [];
    for (const [x, y, z] of [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) {
      const w = W([o[0] + x * A[0] + y * B[0], o[1] + x * A[1] + y * B[1], o[2] + z * C[2]]);
      corners.push(new THREE.Vector3(w[0], w[1], w[2]));
    }
    const idx = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
    const lg = new THREE.BufferGeometry().setFromPoints(idx.map((i) => corners[i]));
    this.mark = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xffffff }));
    this.scene.add(this.mark);

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }
  resize() {
    const w = this.container.clientWidth, hh = this.container.clientHeight;
    if (!w || !hh) return;
    this.renderer.setSize(w, hh, false); this.camera.aspect = w / hh; this.camera.updateProjectionMatrix();
  }
  updateAtoms(t, force) {
    const X = this.EXAG, T = T_FLIP_OV;
    let dirty = { Li: false, Nb: false, O: false };
    for (const at of this.atoms) {
      const s = smooth01((t - at.tt) / T);
      if (!force && s === at.s) continue;
      at.s = s;
      const r = LAT.SP[at.sp].r * 0.75, q = X * (s - 0.5);
      const mesh = this.meshes[at.sp], arr = mesh.instanceMatrix.array, o = at.k * 16;
      arr[o] = r; arr[o + 5] = r; arr[o + 10] = r; arr[o + 15] = 1;
      arr[o + 12] = at.p[0] + 0.5 * at.d[0] + q * at.d[0];
      arr[o + 13] = at.p[1] + 0.5 * at.d[1] + q * at.d[1];
      arr[o + 14] = at.p[2] + 0.5 * at.d[2] + q * at.d[2];
      dirty[at.sp] = true;
    }
    for (const sp in dirty) if (dirty[sp]) this.meshes[sp].instanceMatrix.needsUpdate = true;
  }
  updateArrows(t, force) {
    const arr = this.arrows.instanceMatrix.array, col = this.arrows.instanceColor.array;
    let dirty = false;
    this.cells.forEach((q, n) => {
      const s = smooth01((t - q.tt) / T_FLIP_OV);
      if (!force && s === q.s) return;
      q.s = s; dirty = true;
      const o = n * 16, sz = 1 - 2 * s;
      arr.fill(0, o, o + 16);
      // local +z (arrow axis) -> world -Y (crystal +z); local y -> world +Z
      arr[o] = 1; arr[o + 6] = 1; arr[o + 9] = -(Math.abs(sz) < 1e-3 ? 1e-3 : sz); arr[o + 15] = 1;
      arr[o + 12] = q.p[0]; arr[o + 13] = q.p[1]; arr[o + 14] = q.p[2];
      this.tmp.copy(this.cUp).lerp(this.cDn, s);
      col[n * 3] = this.tmp.r; col[n * 3 + 1] = this.tmp.g; col[n * 3 + 2] = this.tmp.b;
    });
    if (dirty) { this.arrows.instanceMatrix.needsUpdate = true; this.arrows.instanceColor.needsUpdate = true; }
  }
  updateOverlay(h) {
    if (h === this.lastH) return; this.lastH = h;
    const d = this.ovI.data, w = this.ovC.width, hh = this.ovC.height;
    for (let r = 0; r < hh; r++) for (let q = 0; q < w; q++) {
      const i = r * w + q, k = i * 4, flip = this.ovG[i] <= h, fin = isFinite(this.ovG[i]);
      // image row 0 is the top of the canvas = +y of the plane, so flip rows
      const kk = ((hh - 1 - r) * w + q) * 4;
      if (flip) { d[kk] = 59; d[kk + 1] = 120; d[kk + 2] = 224; d[kk + 3] = 150; }
      else { d[kk] = 224; d[kk + 1] = 162; d[kk + 2] = 74; d[kk + 3] = fin || true ? 38 : 0; }
    }
    this.ovX.putImageData(this.ovI, 0, 0); this.ovTex.needsUpdate = true;
  }
  draw(t) {
    this.updateOverlay(DOM.hOf(WF.F(t)));
    this.updateAtoms(t, false); this.updateArrows(t, false);
    const v = WF.V(t) / WF.P.Vmax;
    this.eTop.material.color.setRGB(0.54 + 0.4 * v, 0.57 - 0.25 * v, 0.6 - 0.35 * v);
    this.controls.update(); this.renderer.render(this.scene, this.camera);
  }
}

// Concatenate non-indexed/indexed geometries into one indexed BufferGeometry.
function mergeGeoms(list) {
  const pos = [], nor = [], idx = []; let off = 0;
  for (const g of list) {
    const p = g.attributes.position.array, n = g.attributes.normal.array;
    pos.push(...p); nor.push(...n);
    if (g.index) for (const i of g.index.array) idx.push(i + off); else for (let i = 0; i < p.length / 3; i++) idx.push(i + off);
    off += p.length / 3;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  out.setIndex(idx); return out;
}
