/**
 * Shared optical contracts. See docs/CODING_TASKS.md ("Shared contracts").
 *
 * `ChannelCommand` is what a controller asks for. `ChannelActual` is what the
 * hardware actually does after hidden errors, limits, quantization, and drift.
 * Manual controls and learned controllers must use the same command path.
 */

export interface ChannelCommand {
  piston_rad: number;
  /** Relative field amplitude, >= 0. Power scales as amplitude^2. */
  amplitude: number;
  /** Normalized transverse propagation direction, dimensionless. */
  tiltX: number;
  tiltY: number;
  /** Signed convergence, 1/m. 0 is collimated, > 0 converges. */
  curvature_per_m: number;
  enabled: boolean;
}

export interface ChannelActual extends ChannelCommand {
  /** Hidden static piston error, rad. Ground-truth telemetry only. */
  hiddenPiston_rad: number;
  /** Hidden multiplicative gain error on the field amplitude. */
  hiddenGain: number;
}

export function defaultChannelCommand(): ChannelCommand {
  return {
    piston_rad: 0,
    amplitude: 1,
    tiltX: 0,
    tiltY: 0,
    curvature_per_m: 0,
    enabled: true,
  };
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export function vec3(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

export function length3(v: Vec3): number {
  return Math.hypot(v.x, v.y, v.z);
}

export function scale3(v: Vec3, s: number): Vec3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}

export function sub3(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function add3(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function dot3(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
