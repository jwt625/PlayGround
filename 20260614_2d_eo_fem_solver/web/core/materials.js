import { electrodeContains } from "./geometry.js";

export const MATERIAL_PROPERTY_KEYS = [
  "eps_r",
  "eps_r_xx",
  "eps_r_yy",
  "eps_r_xy",
  "r13",
  "r33",
  "r22",
  "r_eff",
];

export function parseMaterials(config) {
  const materials = [];
  for (const [name, block] of Object.entries(config.Materials ?? {})) {
    const material = { name, shape: block.shape ?? "background", properties: {}, params: {} };
    for (const [key, value] of Object.entries(block)) {
      if (key === "shape") continue;
      if (MATERIAL_PROPERTY_KEYS.includes(key)) {
        material.properties[key] = Number(value);
      } else {
        material.params[key] = Number(value);
      }
    }
    materials.push(material);
  }
  if (!materials.some((material) => material.shape === "background")) {
    materials.unshift({ name: "background", shape: "background", properties: { eps_r: 1.0 }, params: {} });
  }
  return materials;
}

export function materialAt(materials, x, y) {
  let selected = materials.find((material) => material.shape === "background") ?? materials[0];
  for (const material of materials) {
    if (material.shape === "background") continue;
    if (containsMaterial(material, x, y)) selected = material;
  }
  return selected;
}

export function materialPropertyAt(materials, x, y, property) {
  const material = materialAt(materials, x, y);
  if (material.properties[property] !== undefined) return material.properties[property];
  if (property === "eps_r_xx" || property === "eps_r_yy") {
    return material.properties.eps_r ?? 1.0;
  }
  if (property === "eps_r_xy") return 0.0;
  return material.properties[property] ?? 0.0;
}

export function materialPropertyField(mesh, materials, property) {
  const values = new Float64Array(mesh.nodes.length);
  for (let i = 0; i < mesh.nodes.length; i += 1) {
    const [x, y] = mesh.nodes[i];
    values[i] = materialPropertyAt(materials, x, y, property);
  }
  return values;
}

function containsMaterial(material, x, y) {
  if (material.shape === "rectangle" || material.shape === "circle") {
    return electrodeContains({ shape: material.shape, params: material.params }, x, y);
  }
  throw new Error(`unsupported material shape: ${material.shape}`);
}
