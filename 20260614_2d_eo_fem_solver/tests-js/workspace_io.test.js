import assert from "node:assert/strict";
import test from "node:test";

import {
  loadGeometryArtifact,
  loadMeshArtifact,
  loadSolutionArtifact,
  parseGmsh41,
} from "../web/core/workspace_io.js";

const UNIT_TRIANGLE_MSH = `$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
1 2 "ground_boundary"
2 1 "domain"
$EndPhysicalNames
$Entities
3 3 1 0
1 0 0 0 0
2 1 0 0 0
3 0 1 0 0
1 0 0 0 1 0 0 1 2 2 1 -2
2 0 0 0 1 1 0 0 2 2 3 -2
3 0 0 0 0 1 0 0 2 2 1 -3
1 0 0 0 1 1 0 1 1 3 1 2 3
$EndEntities
$Nodes
1 3 1 3
2 1 0 3
1 2 3
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
2 4 1 4
1 1 1 3
1 1 2
2 2 3
3 3 1
2 1 2 1
4 1 2 3
$EndElements
`;

test("loadGeometryArtifact creates a geometry-only workspace layer", () => {
  const artifact = loadGeometryArtifact(`Domain:\n  x_min: -1\n  x_max: 2\n  y_min: -3\n  y_max: 4\n`, "demo.yaml");
  assert.equal(artifact.kind, "geometry");
  assert.deepEqual(artifact.geometry.domain, { xMin: -1, xMax: 2, yMin: -3, yMax: 4 });
});

test("parseGmsh41 normalizes triangles, boundary edges, and physical groups", () => {
  const mesh = parseGmsh41(UNIT_TRIANGLE_MSH);
  assert.equal(mesh.nodes.length, 3);
  assert.deepEqual(mesh.triangles, [[0, 1, 2]]);
  assert.equal(mesh.boundaryEdges.length, 3);
  assert.equal(mesh.boundaryEdges[0].groups[0].name, "ground_boundary");
  assert.equal(mesh.triangleGroups[0][0].name, "domain");
  assert.equal(mesh.stats.type, "Gmsh 4.1 unstructured");
});

test("loadMeshArtifact accepts normalized JSON meshes", () => {
  const artifact = loadMeshArtifact(JSON.stringify({ nodes: [[0, 0], [1, 0], [0, 1]], triangles: [[0, 1, 2]] }), "mesh.json");
  assert.equal(artifact.kind, "mesh");
  assert.equal(artifact.mesh.stats.triangles, 1);
  assert.equal(artifact.mesh.stats.minEdge, 1);
  assert.ok(artifact.mesh.stats.meanQuality > 0.8);
});

test("loadSolutionArtifact collects nodal fields from result JSON", () => {
  const artifact = loadSolutionArtifact(JSON.stringify({
    mesh: { nodes: [[0, 0], [1, 0], [0, 1]], triangles: [[0, 1, 2]] },
    phi: [0, 1, 0.5],
    fields: { normE: [1, 2, 3] },
    residual: 1e-8,
  }));
  assert.deepEqual(Object.keys(artifact.fields).sort(), ["normE", "phi"]);
  assert.equal(artifact.metadata.residual, 1e-8);
});
