import { EPS0 } from "./constants.js";

export function parallelPlateCapacitance(epsR, width, gap) {
  if (width <= 0 || gap <= 0) {
    throw new Error("width and gap must be positive");
  }
  return (EPS0 * epsR * width) / gap;
}

export function twoCylinderCapacitance(epsR, radius, centerDistance) {
  if (radius <= 0) {
    throw new Error("radius must be positive");
  }
  if (centerDistance <= 2 * radius) {
    throw new Error("centerDistance must be larger than 2 * radius");
  }
  return (Math.PI * EPS0 * epsR) / Math.acosh(centerDistance / (2 * radius));
}
