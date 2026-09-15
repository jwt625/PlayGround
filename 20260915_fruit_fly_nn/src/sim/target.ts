import { seededRandom } from "../optics/channels";

/** Target trajectories on a fixed plane at z = Z (DevLog/001: initial training plane). */
export interface TargetMotion {
  readonly kind: string;
  positionAt(t_s: number): { x: number; y: number; z: number };
}

export function staticTarget(x: number, y: number, z: number): TargetMotion {
  return { kind: "static", positionAt: () => ({ x, y, z }) };
}

export interface LissajousOptions {
  z_m: number;
  amplitudeX_m: number;
  amplitudeY_m: number;
  freqX_hz: number;
  freqY_hz: number;
  phaseX?: number;
  phaseY?: number;
  centerX_m?: number;
  centerY_m?: number;
}

export function lissajousTarget(o: LissajousOptions): TargetMotion {
  return {
    kind: "lissajous",
    positionAt: (t) => ({
      x: (o.centerX_m ?? 0) + o.amplitudeX_m * Math.sin(2 * Math.PI * o.freqX_hz * t + (o.phaseX ?? 0)),
      y: (o.centerY_m ?? 0) + o.amplitudeY_m * Math.sin(2 * Math.PI * o.freqY_hz * t + (o.phaseY ?? 0)),
      z: o.z_m,
    }),
  };
}

export interface CircleOptions {
  z_m: number;
  radius_m: number;
  freq_hz: number;
  centerX_m?: number;
  centerY_m?: number;
}

export function circleTarget(o: CircleOptions): TargetMotion {
  return {
    kind: "circle",
    positionAt: (t) => {
      const a = 2 * Math.PI * o.freq_hz * t;
      return {
        x: (o.centerX_m ?? 0) + o.radius_m * Math.cos(a),
        y: (o.centerY_m ?? 0) + o.radius_m * Math.sin(a),
        z: o.z_m,
      };
    },
  };
}

export interface FlyOptions {
  z_m: number;
  extentX_m: number;
  extentY_m: number;
  speed_mps: number;
  seed: number;
  /** Number of waypoints in the closed loop, lower = smoother. */
  waypoints?: number;
}

/**
 * Smooth random "fly-like" loop: a periodic Catmull-Rom spline through random
 * waypoints with a mild speed limit. No injury or laser effects are modelled.
 */
export function flyTarget(o: FlyOptions): TargetMotion {
  const rand = seededRandom(o.seed);
  const count = Math.max(4, o.waypoints ?? 8);
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i < count; i++) {
    points.push({ x: (rand() * 2 - 1) * o.extentX_m, y: (rand() * 2 - 1) * o.extentY_m });
  }
  const total = points.reduce((acc, p, i) => {
    const q = points[(i + 1) % count];
    return acc + Math.hypot(q.x - p.x, q.y - p.y);
  }, 0);
  const period = total / Math.max(1e-6, o.speed_mps);

  const catmull = (p0: number, p1: number, p2: number, p3: number, u: number) => {
    const u2 = u * u;
    const u3 = u2 * u;
    return (
      0.5 *
      (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
    );
  };

  return {
    kind: "fly",
    positionAt: (t) => {
      const wrapped = ((t % period) + period) % period;
      const u = (wrapped / period) * count;
      const seg = Math.floor(u) % count;
      const f = u - Math.floor(u);
      const p0 = points[(seg - 1 + count) % count];
      const p1 = points[seg];
      const p2 = points[(seg + 1) % count];
      const p3 = points[(seg + 2) % count];
      return {
        x: catmull(p0.x, p1.x, p2.x, p3.x, f),
        y: catmull(p0.y, p1.y, p2.y, p3.y, f),
        z: o.z_m,
      };
    },
  };
}
