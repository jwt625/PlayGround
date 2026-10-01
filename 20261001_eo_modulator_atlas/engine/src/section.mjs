// Cross-section model: painter-order region classification, mesh construction with size hints.
// Units: micrometers in the plane.  License: GPL-3.0-or-later.

import { meshDomain, extractMesh } from './mesh.mjs';
import { pointInPolygon, polygonBBox, rectToPolygon } from './geometry.mjs';

/**
 * @typedef {Object} NormGeometry
 * @property {'none'|'x_mirror_even'|'x_mirror_odd'} symmetry
 * @property {number} mirror_x_um
 * @property {{x:[number,number], y:[number,number]}} domain
 * @property {{name:string, material:string, poly:number[][]}[]} regions
 * @property {{name:string, role:string, weight:number, material:string, poly:number[][]}[]} electrodes
 * @property {{x:[number,number], y:[number,number]}|null} optical_window
 * @property {any} mesh
 * @property {{left:string,right:string,top:string,bottom:string}} boundary
 */

export class CrossSection {
  /** @param {NormGeometry} geom */
  constructor(geom, name = 'geometry') {
    this.name = name;
    this.geom = geom;
    this.regions = geom.regions.map((r) => ({ ...r, bbox: polygonBBox(r.poly) }));
    this.electrodes = geom.electrodes.map((e) => ({ ...e, bbox: polygonBBox(e.poly) }));
    const d = geom.domain;
    this.fullRect = { x0: d.x[0], x1: d.x[1], y0: d.y[0], y1: d.y[1] };
    this.rect = this._halfRect();
  }

  /** Domain actually meshed (cropped to the half-plane when a mirror symmetry is declared). */
  _halfRect() {
    const g = this.geom;
    const r = { ...this.fullRect };
    if (g.symmetry === 'none') return r;
    const m = g.mirror_x_um;
    const tol = 1e-9 * Math.max(r.x1 - r.x0, 1);
    if (m <= r.x0 + tol) return r; // domain already on the + side with an edge at the plane
    if (m >= r.x1 - tol) return r;
    // domain straddles the plane: keep the side containing the optical window (or the side with larger extent)
    const w = g.optical_window;
    let keepRight;
    if (w) {
      const wc = 0.5 * (w.x[0] + w.x[1]);
      keepRight = wc > m;
      if (w.x[0] < m && w.x[1] > m) throw new Error('optical_window straddles the mirror plane');
    } else keepRight = r.x1 - m >= m - r.x0;
    if (keepRight) r.x0 = m;
    else r.x1 = m;
    return r;
  }

  classify = (x, y) => {
    let region = -1;
    for (let i = this.regions.length - 1; i >= 0; i--) {
      const r = this.regions[i];
      const b = r.bbox;
      if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
      if (pointInPolygon(x, y, r.poly)) {
        region = i;
        break;
      }
    }
    let electrode = -1;
    for (let i = this.electrodes.length - 1; i >= 0; i--) {
      const e = this.electrodes[i];
      const b = e.bbox;
      if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
      if (pointInPolygon(x, y, e.poly)) {
        electrode = i;
        break;
      }
    }
    return { region, electrode };
  };

  /** Smallest dimension of any electrode bounding box (um). */
  minElectrodeThickness() {
    let t = Infinity;
    for (const e of this.electrodes) t = Math.min(t, e.bbox.x1 - e.bbox.x0, e.bbox.y1 - e.bbox.y0);
    return t;
  }

  /**
   * Build a conforming mesh of the (cropped) domain.
   * @param {{scale?:number, window?:{x:[number,number],y:[number,number]}, windowOnly?:boolean,
   *   hints?:any, extraSources?:any[], minAngleBound?:number}} [opts]
   */
  buildMesh(opts = {}) {
    const scale = opts.scale ?? 1;
    const g = this.geom;
    const h = opts.hints ?? g.mesh ?? {};
    const rect = opts.windowOnly && opts.window ? { x0: opts.window.x[0], x1: opts.window.x[1], y0: opts.window.y[0], y1: opts.window.y[1] } : this.rect;
    const diag = Math.hypot(rect.x1 - rect.x0, rect.y1 - rect.y0);
    const thick = this.electrodes.length ? this.minElectrodeThickness() : diag / 50;
    const maxEdge = (h.max_edge_um ?? diag / 40) * scale;
    const corner = (h.electrode_edge_um ?? Math.min(maxEdge, thick / 6)) * scale;
    const face = Math.min(maxEdge, (h.electrode_face_um ?? 8 * (h.electrode_edge_um ?? thick / 6)) * scale);
    const grade = h.grade ?? 0.4;
    const gFar = h.grade_far ?? 0.6;
    const farEdge = opts.sourcesOnly ? maxEdge : Math.max(maxEdge, h.far_edge_um ?? diag / 12);
    const sources = [];
    if (!opts.sourcesOnly)
    for (const e of this.electrodes) {
      for (const p of e.poly) sources.push({ poly: [p], filled: false, h0: corner, grade: (h.grade_corner ?? 0.2) * Math.min(1, scale) });
      sources.push({ poly: e.poly, filled: true, h0: face });
      sources.push({ poly: e.poly, filled: true, h0: maxEdge, grade: gFar });
    }
    const win = opts.window ?? g.optical_window;
    if (win && !opts.windowOnly && !opts.sourcesOnly) {
      const wp = rectToPolygon(win.x, win.y);
      const we = Math.min(maxEdge, (h.window_edge_um ?? 0.05) * scale);
      sources.push({ poly: wp, filled: true, h0: we });
      sources.push({ poly: wp, filled: true, h0: maxEdge, grade: gFar });
    }
    if (h.regions && !opts.sourcesOnly) {
      for (const [rname, edge] of Object.entries(h.regions)) {
        const reg = this.regions.find((r) => r.name === rname);
        if (reg) sources.push({ poly: reg.poly, filled: true, h0: edge * scale });
      }
    }
    for (const s of opts.extraSources ?? []) sources.push(s);
    if (!sources.length && !opts.sourcesOnly) sources.push({ poly: rectToPolygon([rect.x0, rect.x1], [rect.y0, rect.y1]), filled: true, h0: maxEdge });
    const loops = [];
    this.regions.forEach((r, i) => loops.push({ poly: r.poly, kind: 1, id: i }));
    this.electrodes.forEach((e, i) => loops.push({ poly: e.poly, kind: 2, id: i }));
    const raw = meshDomain({
      rect,
      loops,
      classify: this.classify,
      maxEdge: farEdge,
      sources,
      grade,
      minAngleBound: opts.minAngleBound ?? h.min_angle_bound ?? 1.35,
      minEdge: (h.min_edge_um ?? Math.max(1e-4, corner / 40)) * 1,
      maxVertices: h.max_vertices ?? 800000,
    });
    const mesh = extractMesh(raw);
    // verify coverage
    for (let t = 0; t < mesh.nTris; t++)
      if (mesh.triRegion[t] < 0) throw new Error(`geometry does not cover the whole domain (cross-section ${this.name}); add a background region`);
    mesh.rect = rect;
    return mesh;
  }
}
