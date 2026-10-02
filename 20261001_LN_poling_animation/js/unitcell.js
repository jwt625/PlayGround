// Zoomed hexagonal unit cell: true-scale atomic flip, crystal / applied field visuals, double-well inset.
const T_FLIP_UC = 0.08; // ms of pulse time for the unit-cell flip
const UC_SCALE = { Li: 1.4, Nb: 1.35, O: 1.15 }; // display radii, not ionic radii

class UnitCell {
  constructor(container, wellCv, info, track) {
    const { a, c, cart, SP } = LAT;
    this.container = container; this.wellCv = wellCv; this.info = info; this.track = track;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const dom = this.renderer.domElement;
    dom.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
    container.insertBefore(dom, container.firstChild);
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(0x0e1013);
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.5, 400); this.camera.up.set(0, 0, 1);
    const ctr = cart([0.5, 0.5, 0.5]); this.ctr = new THREE.Vector3(...ctr);
    this.camera.position.set(ctr[0] + 16, ctr[1] - 24, ctr[2] + 8);
    this.controls = new THREE.OrbitControls(this.camera, dom);
    this.controls.target.copy(this.ctr); this.controls.autoRotate = true; this.controls.autoRotateSpeed = 1.4;
    this.controls.enableDamping = true; this.controls.update();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const d1 = new THREE.DirectionalLight(0xffffff, 0.95); d1.position.set(1, -1, 2); this.scene.add(d1);
    const d2 = new THREE.DirectionalLight(0xffffff, 0.35); d2.position.set(-1, 1, 0.3); this.scene.add(d2);

    // atoms: in-cell (periodic images included) plus the O neighbours needed to close the polyhedra
    const eps = 1e-3, inCell = (fr) => fr.every((v) => v >= -eps && v <= 1 + eps);
    const all = LAT.gen(() => true, [-1, 1], [-1, 1], [-1, 1]);
    const key = (at) => at.fr.map((v) => v.toFixed(4)).join(',');
    const sphere = new THREE.SphereGeometry(1, 28, 18);
    const mats = {}, ghost = {}, dim = {}, hot = {};
    for (const sp in SP) {
      mats[sp] = new THREE.MeshPhongMaterial({ color: SP[sp].color, shininess: 90, specular: 0x555555 });
      dim[sp] = new THREE.MeshPhongMaterial({ color: new THREE.Color(SP[sp].color).multiplyScalar(0.42), shininess: 40, specular: 0x222222 });
      hot[sp] = new THREE.MeshPhongMaterial({ color: SP[sp].color, emissive: new THREE.Color(SP[sp].color).multiplyScalar(0.4), shininess: 100, specular: 0x888888 });
      ghost[sp] = new THREE.MeshPhongMaterial({ color: SP[sp].color, shininess: 40, transparent: true, opacity: 0.35 });
    }
    this.entries = new Map();
    const ensure = (at, inside) => {
      const k = key(at);
      if (this.entries.has(k)) return this.entries.get(k);
      const m = new THREE.Mesh(sphere, inside ? dim[at.sp] : ghost[at.sp]);
      m.scale.setScalar(SP[at.sp].r * UC_SCALE[at.sp]); this.scene.add(m);
      const e = { at, mesh: m, inside, halo: null };
      this.entries.set(k, e); return e;
    };
    for (const at of all) if (inCell(at.fr)) ensure(at, true);
    const inList = [...this.entries.values()];

    // tracked Li / Nb: in-cell atoms nearest the cell centre
    const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    const nearest = (sp) => inList.filter((e) => e.at.sp === sp).sort((x, y) => dist(x.at.p, ctr) - dist(y.at.p, ctr))[0];
    this.tLi = nearest('Li'); this.tNb = nearest('Nb');

    // polyhedra (Nb: O6 octahedron, Li: O3 triangle it passes through)
    this.poly = [];
    // ghost (out-of-cell) O atoms are created only for the tracked Li / Nb polyhedra
    const nbrs = (e, cut, main) => all.filter((o) => o.sp === 'O' && dist(o.p, e.at.p) < cut && (main || inCell(o.fr)))
      .map((o) => ensure(o, inCell(o.fr)));
    for (const e of inList) {
      const main = e === this.tNb || e === this.tLi, cut = e.at.sp === 'Nb' ? 2.25 : 2.1, need = e.at.sp === 'Nb' ? 6 : 3;
      if (e.at.sp === 'O') continue;
      const o = nbrs(e, cut, main);
      if (o.length === need) this.poly.push({ e, o, kind: e.at.sp, main });
    }
    // hierarchy: tracked Li / Nb bright, O of their polyhedra normal, everything else dim
    for (const p of this.poly) if (p.main) for (const o of p.o) if (o.inside) o.mesh.material = mats.O;
    this.tLi.mesh.material = hot.Li; this.tNb.mesh.material = hot.Nb;
    const label = (text, color) => {
      const c2 = document.createElement('canvas'); c2.width = 128; c2.height = 40;
      const x2 = c2.getContext('2d'); x2.font = '600 26px -apple-system, Segoe UI, sans-serif'; x2.fillStyle = color;
      x2.textAlign = 'center'; x2.textBaseline = 'middle'; x2.fillText(text, 64, 22);
      const sp2 = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c2), transparent: true, depthTest: false }));
      sp2.scale.set(3.6, 1.1, 1); sp2.renderOrder = 10; this.scene.add(sp2); return sp2;
    };
    this.lbLi = label('Li+', '#e3b8ff'); this.lbNb = label('Nb5+', '#a8e6ec');
    for (const p of this.poly) {
      const n = p.o.length, pos = new Float32Array(n * 3), faces = [], edges = [];
      const d = (i, j) => dist(p.o[i].at.p, p.o[j].at.p);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (d(i, j) < 3.3) edges.push(i, j);
      if (n === 3) faces.push(0, 1, 2);
      else for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++)
        if (d(i, j) < 3.3 && d(j, k) < 3.3 && d(i, k) < 3.3) faces.push(i, j, k);
      const col = p.kind === 'Nb' ? 0x73c2c9 : 0xcc80ff;
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(faces);
      p.mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col, transparent: true, side: THREE.DoubleSide,
        opacity: p.main ? (p.kind === 'Li' ? 0.55 : 0.22) : 0.05, depthWrite: false }));
      const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edges.length * 3), 3));
      p.edgeIdx = edges;
      p.lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: p.main ? 0.9 : 0.25 }));
      this.scene.add(p.mesh, p.lines);
    }

    // glow halos on in-cell atoms (cation orange, anion blue)
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const cx = cv.getContext('2d'), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.35, 'rgba(255,255,255,.35)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
    const tex = new THREE.CanvasTexture(cv);
    for (const e of inList) {
      const cation = SP[e.at.sp].q > 0;
      e.halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: cation ? 0xff8a3d : 0x3da5ff, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.2 }));
      e.haloBase = e.at.sp === 'Nb' ? 3.4 : e.at.sp === 'Li' ? 2.0 : 2.5;
      this.scene.add(e.halo);
      // force arrow q E
      const col = cation ? 0xff8a3d : 0x3da5ff;
      e.arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, cation ? -1 : 1), new THREE.Vector3(), 1, col, 0.4, 0.28);
      this.scene.add(e.arrow);
    }

    // cell edges
    const A = cart([1, 0, 0]), B = cart([0, 1, 0]);
    const cn = [];
    for (const [x, y, z] of [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
      cn.push(new THREE.Vector3(x * A[0] + y * B[0], x * A[1] + y * B[1], z * c));
    const idx = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
    this.scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(idx.map((i) => cn[i])),
      new THREE.LineBasicMaterial({ color: 0x6b7480 })));

    // tracked-atom path markers: start / end ghosts and a dashed path
    const mk = (e, col) => {
      const s0 = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ color: col, wireframe: true, transparent: true, opacity: 0.7 }));
      const s1 = s0.clone(); s0.scale.setScalar(SP[e.at.sp].r * UC_SCALE[e.at.sp]); s1.scale.copy(s0.scale);
      s0.position.set(...e.at.p); s1.position.set(e.at.p[0] + e.at.d[0], e.at.p[1] + e.at.d[1], e.at.p[2] + e.at.d[2]);
      const g = new THREE.BufferGeometry().setFromPoints([s0.position, s1.position]);
      const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: 0.8 }));
      l.computeLineDistances(); this.scene.add(s0, s1, l);
    };
    mk(this.tLi, 0xcc80ff); mk(this.tNb, 0x73c2c9);
    this.dLi = Math.hypot(...this.tLi.at.d); this.dNb = Math.hypot(...this.tNb.at.d);

    // field streaks
    const mkStreaks = (n, color) => {
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
      const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false }));
      const data = [];
      for (let i = 0; i < n; i++) {
        const u = Math.random(), v = Math.random();
        data.push({ x: u * A[0] + v * B[0], y: u * A[1] + v * B[1], z: -4 + Math.random() * (c + 8) });
      }
      this.scene.add(lines); return { lines, data, n };
    };
    this.eS = mkStreaks(48, 0x7fe0ff); this.pS = mkStreaks(36, 0xe0a24a);
    this.pCol = [new THREE.Color(0xe0a24a), new THREE.Color(0x3b78e0)];
    // big field arrows beside the cell
    const side = new THREE.Vector3(-7, ctr[1] - 2, ctr[2]);
    this.eArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), side.clone().add(new THREE.Vector3(0, 0, 6)), 1, 0x7fe0ff, 1.4, 0.9);
    this.pArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), side.clone().add(new THREE.Vector3(-2.4, 0, -6)), 1, 0xe0a24a, 1.4, 0.9);
    this.scene.add(this.eArrow, this.pArrow);
    this.lastT = 0;
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }
  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    const dpr = window.devicePixelRatio || 1;
    this.wellCv.width = 150 * dpr; this.wellCv.height = 96 * dpr;
  }
  drawWell(q, hTilt) {
    const cv = this.wellCv, c = cv.getContext('2d'), dpr = cv.width / 150;
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, 150, 96);
    const X = (x) => 10 + (x + 1.6) / 3.2 * 130, Y = (u) => 80 - (u + 1.6) / 3.9 * 66;
    const U = (x) => (x * x - 1) ** 2 + hTilt * x;
    c.strokeStyle = '#2a2f35'; c.beginPath(); c.moveTo(X(0) + 0.5, 8); c.lineTo(X(0) + 0.5, 82); c.stroke();
    c.strokeStyle = '#c7d0d9'; c.lineWidth = 1.5; c.beginPath();
    for (let x = -1.6; x <= 1.6; x += 0.02) { const px = X(x), py = Y(U(x)); x === -1.6 ? c.moveTo(px, py) : c.lineTo(px, py); }
    c.stroke();
    c.fillStyle = '#ff8a3d'; c.strokeStyle = '#0b0d10'; c.lineWidth = 2;
    c.beginPath(); c.arc(X(q), Y(U(q)) - 5, 4.5, 0, 6.2832); c.stroke(); c.fill();
    c.fillStyle = '#7d8791'; c.font = '10px -apple-system, Segoe UI, sans-serif'; c.textAlign = 'center';
    c.fillText('+Z', X(1), 93); c.fillText('-Z', X(-1), 93); c.fillText('polar coordinate', X(0), 10);
  }
  draw(t) {
    const dt = Math.max(0, Math.min(0.1, (performance.now() - (this.tPrev || performance.now())) / 1000)); this.tPrev = performance.now();
    const s = smooth01((t - this.track.tt) / T_FLIP_UC), En = WF.V(t) / WF.P.Vmax, P = 1 - 2 * s;
    for (const e of this.entries.values()) {
      const x = e.at.p[0] + s * e.at.d[0], y = e.at.p[1] + s * e.at.d[1], z = e.at.p[2] + s * e.at.d[2];
      e.mesh.position.set(x, y, z);
      if (e.halo) {
        e.halo.position.set(x, y, z);
        const sc = e.haloBase * (1 + 0.3 * En); e.halo.scale.set(sc, sc, 1);
        e.halo.material.opacity = 0.07 + 0.3 * En * (e.at.sp === 'Nb' ? 1 : 0.6);
        const q = LAT.SP[e.at.sp].q, len = Math.abs(q) * 0.3 * En;
        e.arrow.visible = len > 0.05;
        if (e.arrow.visible) { e.arrow.position.set(x, y, z); e.arrow.setLength(len + 0.5, Math.min(0.45, 0.2 + len * 0.3), 0.3); }
      }
    }
    this.lbLi.position.set(this.tLi.mesh.position.x + 2.0, this.tLi.mesh.position.y, this.tLi.mesh.position.z + 1.0);
    this.lbNb.position.set(this.tNb.mesh.position.x - 2.2, this.tNb.mesh.position.y, this.tNb.mesh.position.z + 1.0);
    for (const p of this.poly) {
      const pa = p.mesh.geometry.attributes.position, la = p.lines.geometry.attributes.position;
      p.o.forEach((o, i) => pa.setXYZ(i, o.mesh.position.x, o.mesh.position.y, o.mesh.position.z));
      p.edgeIdx.forEach((vi, i) => la.setXYZ(i, p.o[vi].mesh.position.x, p.o[vi].mesh.position.y, p.o[vi].mesh.position.z));
      pa.needsUpdate = true; la.needsUpdate = true;
      p.mesh.geometry.computeBoundingSphere();
    }
    // field streaks: applied E (cyan, along -z, scales with V); internal polarization field (along sign(P) z)
    const c = LAT.c, upd = (S, speed, len, sign) => {
      const pos = S.lines.geometry.attributes.position;
      S.data.forEach((d, i) => {
        d.z += sign * speed * dt; if (d.z > c + 4) d.z -= c + 8; if (d.z < -4) d.z += c + 8;
        pos.setXYZ(2 * i, d.x, d.y, d.z); pos.setXYZ(2 * i + 1, d.x, d.y, d.z + sign * len);
      });
      pos.needsUpdate = true;
    };
    upd(this.eS, 14 * En, 0.4 + 2.4 * En, -1); this.eS.lines.material.opacity = 0.55 * En;
    const aP = Math.abs(P);
    this.pS.lines.material.color.copy(this.pCol[P >= 0 ? 0 : 1]);
    upd(this.pS, 3.2 * aP, 0.4 + 1.2 * aP, P >= 0 ? 1 : -1); this.pS.lines.material.opacity = 0.55 * aP;
    // big arrows
    this.eArrow.visible = En > 0.02; if (this.eArrow.visible) this.eArrow.setLength(Math.max(1.6, 11 * En), 1.4, 0.9);
    this.pArrow.visible = aP > 0.03;
    if (this.pArrow.visible) {
      this.pArrow.setDirection(new THREE.Vector3(0, 0, P >= 0 ? 1 : -1)); this.pArrow.setLength(Math.max(1.6, 11 * aP), 1.4, 0.9);
      this.pArrow.setColor(this.pCol[P >= 0 ? 0 : 1]);
      this.pArrow.position.z = this.ctr.z + (P >= 0 ? -6 : 6);
    }
    this.drawWell(P, 1.2 * En);
    this.info.innerHTML = `V <b>${WF.V(t).toFixed(0)}</b> V &nbsp; P/Ps <b>${P >= 0 ? '+' : ''}${P.toFixed(2)}</b><br>` +
      `Li shift <b>${(s * this.dLi).toFixed(2)}</b> / ${this.dLi.toFixed(2)} A<br>Nb shift <b>${(s * this.dNb).toFixed(2)}</b> / ${this.dNb.toFixed(2)} A`;
    this.controls.update(); this.renderer.render(this.scene, this.camera);
  }
}
