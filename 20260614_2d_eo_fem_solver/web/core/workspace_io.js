import { parseSimpleYaml } from "./config.js";
import { parseDomain } from "./geometry.js";

const ELEMENT_NODE_COUNTS = new Map([
  [1, 2],
  [2, 3],
  [8, 3],
  [9, 6],
  [15, 1],
]);

export function loadGeometryArtifact(text, fileName = "geometry.yaml") {
  const parsed = fileName.toLowerCase().endsWith(".json") ? JSON.parse(text) : parseSimpleYaml(text);
  const config = parsed.config ?? parsed;
  if (!config.Domain) throw new Error("geometry artifact must contain a Domain block");
  return {
    kind: "geometry",
    schemaVersion: "eo-fem.workspace/v1",
    name: fileName,
    config,
    geometry: { domain: parseDomain(config) },
  };
}

export function loadMeshArtifact(text, fileName = "mesh.msh") {
  if (fileName.toLowerCase().endsWith(".msh")) {
    return {
      kind: "mesh",
      schemaVersion: "eo-fem.workspace/v1",
      name: fileName,
      mesh: parseGmsh41(text),
    };
  }
  const parsed = JSON.parse(text);
  return {
    kind: "mesh",
    schemaVersion: "eo-fem.workspace/v1",
    name: fileName,
    mesh: normalizeMesh(parsed.mesh ?? parsed),
    config: parsed.config ?? parsed.geometry?.config ?? null,
  };
}

export function loadSolutionArtifact(text, fileName = "result.json") {
  const parsed = JSON.parse(text);
  const rawResult = parsed.solution ?? parsed.result ?? parsed;
  const mesh = normalizeMesh(rawResult.mesh ?? parsed.mesh);
  const fields = collectNodalFields(rawResult, mesh.nodes.length);
  if (Object.keys(fields).length === 0) {
    throw new Error("solution artifact has no nodal field arrays matching the mesh node count");
  }
  return {
    kind: "solution",
    schemaVersion: parsed.schemaVersion ?? "eo-fem.workspace/v1",
    name: fileName,
    config: parsed.config ?? parsed.geometry?.config ?? rawResult.config ?? null,
    mesh,
    fields,
    metadata: parsed.metadata ?? rawResult.metadata ?? solutionMetadata(rawResult),
    rawResult,
  };
}

export function normalizeMesh(rawMesh) {
  if (!rawMesh || !Array.isArray(rawMesh.nodes) || !Array.isArray(rawMesh.triangles)) {
    throw new Error("mesh artifact must contain nodes and triangles arrays");
  }
  const nodes = rawMesh.nodes.map((node, index) => {
    if (!Array.isArray(node) || node.length < 2) throw new Error(`mesh node ${index} is invalid`);
    const point = [Number(node[0]), Number(node[1])];
    if (!point.every(Number.isFinite)) throw new Error(`mesh node ${index} is not finite`);
    return point;
  });
  const triangles = rawMesh.triangles.map((triangle, index) => normalizeElement(triangle, 3, nodes.length, `triangle ${index}`));
  const boundaryEdges = (rawMesh.boundaryEdges ?? []).map((edge, index) => {
    const nodeIds = Array.isArray(edge) ? edge : edge.nodes;
    return {
      ...(Array.isArray(edge) ? {} : edge),
      nodes: normalizeElement(nodeIds, 2, nodes.length, `boundary edge ${index}`),
    };
  });
  const domain = normalizeDomain(rawMesh.domain ?? domainFromNodes(nodes));
  const geometryStats = triangleGeometryStats(nodes, triangles);
  return {
    ...rawMesh,
    domain,
    nodes,
    triangles,
    boundaryEdges,
    physicalGroups: rawMesh.physicalGroups ?? [],
    stats: {
      type: rawMesh.stats?.type ?? "unstructured",
      ...rawMesh.stats,
      nodes: nodes.length,
      triangles: triangles.length,
      boundaryEdges: boundaryEdges.length,
      ...geometryStats,
    },
  };
}

export function parseGmsh41(text) {
  const format = section(text, "MeshFormat").trim().split(/\s+/);
  if (format[0] !== "4.1" || format[1] !== "0") {
    throw new Error("only ASCII Gmsh MSH 4.1 files are supported");
  }
  const physicalNames = parsePhysicalNames(optionalSection(text, "PhysicalNames"));
  const entityGroups = parseEntities(section(text, "Entities"));
  const { nodes, nodeIndex } = parseNodes(section(text, "Nodes"));
  const { triangles, boundaryEdges, triangleGroups } = parseElements(
    section(text, "Elements"),
    nodeIndex,
    entityGroups,
    physicalNames,
  );
  if (triangles.length === 0) throw new Error("Gmsh mesh contains no supported 2D triangle elements");
  return normalizeMesh({
    domain: domainFromNodes(nodes),
    nodes,
    triangles,
    boundaryEdges,
    triangleGroups,
    physicalGroups: [...physicalNames.entries()].map(([key, name]) => {
      const [dim, tag] = key.split(":").map(Number);
      return { dim, tag, name };
    }),
    stats: { type: "Gmsh 4.1 unstructured" },
  });
}

function parsePhysicalNames(body) {
  const names = new Map();
  if (!body.trim()) return names;
  const lines = nonemptyLines(body);
  const count = Number(lines.shift());
  for (const line of lines.slice(0, count)) {
    const match = line.match(/^(\d+)\s+(\d+)\s+"(.*)"$/);
    if (!match) throw new Error(`invalid Gmsh physical name: ${line}`);
    names.set(`${match[1]}:${match[2]}`, match[3]);
  }
  return names;
}

function parseEntities(body) {
  const lines = nonemptyLines(body);
  const counts = lines.shift().split(/\s+/).map(Number);
  const groups = new Map();
  let cursor = 0;
  for (let dim = 0; dim <= 3; dim += 1) {
    for (let i = 0; i < counts[dim]; i += 1) {
      const tokens = lines[cursor].trim().split(/\s+/).map(Number);
      cursor += 1;
      const tag = tokens[0];
      const physicalCountIndex = dim === 0 ? 4 : 7;
      const physicalCount = tokens[physicalCountIndex];
      groups.set(`${dim}:${tag}`, tokens.slice(physicalCountIndex + 1, physicalCountIndex + 1 + physicalCount));
    }
  }
  return groups;
}

function parseNodes(body) {
  const tokens = body.trim().split(/\s+/);
  let cursor = 0;
  const blockCount = takeNumber(tokens, cursor++);
  const expectedNodeCount = takeNumber(tokens, cursor++);
  cursor += 2;
  const taggedNodes = [];
  for (let block = 0; block < blockCount; block += 1) {
    const entityDim = takeNumber(tokens, cursor++);
    cursor += 1;
    const parametric = takeNumber(tokens, cursor++);
    const count = takeNumber(tokens, cursor++);
    const tags = tokens.slice(cursor, cursor + count).map(Number);
    cursor += count;
    for (const tag of tags) {
      const x = takeNumber(tokens, cursor++);
      const y = takeNumber(tokens, cursor++);
      cursor += 1;
      if (parametric) cursor += entityDim;
      taggedNodes.push({ tag, point: [x, y] });
    }
  }
  if (taggedNodes.length !== expectedNodeCount) throw new Error("Gmsh node count does not match header");
  const nodeIndex = new Map(taggedNodes.map((node, index) => [node.tag, index]));
  return { nodes: taggedNodes.map((node) => node.point), nodeIndex };
}

function parseElements(body, nodeIndex, entityGroups, physicalNames) {
  const tokens = body.trim().split(/\s+/);
  let cursor = 0;
  const blockCount = takeNumber(tokens, cursor++);
  cursor += 3;
  const triangles = [];
  const triangleGroups = [];
  const boundaryEdges = [];
  for (let block = 0; block < blockCount; block += 1) {
    const dim = takeNumber(tokens, cursor++);
    const entityTag = takeNumber(tokens, cursor++);
    const elementType = takeNumber(tokens, cursor++);
    const count = takeNumber(tokens, cursor++);
    const nodeCount = ELEMENT_NODE_COUNTS.get(elementType);
    if (!nodeCount) throw new Error(`unsupported Gmsh element type ${elementType}`);
    const physicalTags = entityGroups.get(`${dim}:${entityTag}`) ?? [];
    const groups = physicalTags.map((tag) => ({
      dim,
      tag,
      name: physicalNames.get(`${dim}:${tag}`) ?? `physical_${dim}_${tag}`,
    }));
    for (let i = 0; i < count; i += 1) {
      cursor += 1;
      const tags = tokens.slice(cursor, cursor + nodeCount).map(Number);
      cursor += nodeCount;
      const indices = tags.map((tag) => {
        const index = nodeIndex.get(tag);
        if (index === undefined) throw new Error(`element references unknown node tag ${tag}`);
        return index;
      });
      if (dim === 2 && (elementType === 2 || elementType === 9)) {
        triangles.push(indices.slice(0, 3));
        triangleGroups.push(groups);
      } else if (dim === 1 && (elementType === 1 || elementType === 8)) {
        boundaryEdges.push({ nodes: indices.slice(0, 2), entityTag, groups });
      }
    }
  }
  return { triangles, boundaryEdges, triangleGroups };
}

function collectNodalFields(result, nodeCount) {
  const candidates = { ...(result.fields ?? {}) };
  for (const key of ["phi", "potential", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "normE", "normH", "intensity"]) {
    if (result[key] !== undefined && candidates[key] === undefined) candidates[key] = result[key];
  }
  const fields = {};
  for (const [name, values] of Object.entries(candidates)) {
    if ((Array.isArray(values) || ArrayBuffer.isView(values)) && values.length === nodeCount) {
      fields[name === "potential" ? "phi" : name] = Array.from(values, Number);
    }
  }
  return fields;
}

function solutionMetadata(result) {
  const metadata = {};
  for (const [key, value] of Object.entries(result)) {
    if (!["mesh", "fields", "phi", "potential", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "normE", "normH", "intensity"].includes(key)) {
      metadata[key] = value;
    }
  }
  return metadata;
}

function normalizeElement(raw, expectedLength, nodeCount, label) {
  if (!Array.isArray(raw) || raw.length < expectedLength) throw new Error(`${label} is invalid`);
  const indices = raw.slice(0, expectedLength).map(Number);
  if (!indices.every((index) => Number.isInteger(index) && index >= 0 && index < nodeCount)) {
    throw new Error(`${label} contains an invalid node index`);
  }
  return indices;
}

function normalizeDomain(domain) {
  const normalized = {
    xMin: Number(domain.xMin ?? domain.x_min),
    xMax: Number(domain.xMax ?? domain.x_max),
    yMin: Number(domain.yMin ?? domain.y_min),
    yMax: Number(domain.yMax ?? domain.y_max),
  };
  if (!Object.values(normalized).every(Number.isFinite)) throw new Error("mesh domain is invalid");
  return normalized;
}

function triangleGeometryStats(nodes, triangles) {
  let minEdge = Infinity;
  let maxEdge = 0;
  let edgeSum = 0;
  let edgeCount = 0;
  let minQuality = Infinity;
  let qualitySum = 0;
  for (const triangle of triangles) {
    const [a, b, c] = triangle.map((index) => nodes[index]);
    const edges = [Math.hypot(a[0] - b[0], a[1] - b[1]), Math.hypot(b[0] - c[0], b[1] - c[1]), Math.hypot(c[0] - a[0], c[1] - a[1])];
    for (const edge of edges) {
      if (edge < minEdge) minEdge = edge;
      if (edge > maxEdge) maxEdge = edge;
      edgeSum += edge;
      edgeCount += 1;
    }
    const area2 = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    const quality = (2 * Math.sqrt(3) * area2) / edges.reduce((sum, edge) => sum + edge * edge, 0);
    if (quality < minQuality) minQuality = quality;
    qualitySum += quality;
  }
  return {
    minEdge: edgeCount ? minEdge : 0,
    maxEdge,
    meanEdge: edgeCount ? edgeSum / edgeCount : 0,
    minQuality: triangles.length ? minQuality : 0,
    meanQuality: triangles.length ? qualitySum / triangles.length : 0,
  };
}

function domainFromNodes(nodes) {
  if (nodes.length === 0) throw new Error("mesh contains no nodes");
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const [x, y] of nodes) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  return { xMin, xMax, yMin, yMax };
}

function section(text, name) {
  const body = optionalSection(text, name);
  if (!body) throw new Error(`Gmsh file is missing $${name}`);
  return body;
}

function optionalSection(text, name) {
  const start = text.indexOf(`$${name}`);
  const end = text.indexOf(`$End${name}`);
  if (start < 0 || end < 0 || end <= start) return "";
  return text.slice(start + name.length + 1, end);
}

function nonemptyLines(text) {
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function takeNumber(tokens, index) {
  const value = Number(tokens[index]);
  if (!Number.isFinite(value)) throw new Error("invalid numeric token in Gmsh file");
  return value;
}
