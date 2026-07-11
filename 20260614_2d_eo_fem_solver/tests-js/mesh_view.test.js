import assert from "node:assert/strict";
import test from "node:test";

import { selectTrianglesForView } from "../web/core/mesh_view.js";

const mesh = {
  nodes: [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]],
  triangles: [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]],
};

test("mesh rendering selection culls triangles outside the viewport", () => {
  const selection = selectTrianglesForView(mesh, { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  assert.equal(selection.visibleCount, 4); // edge-touching triangles remain visible
  const cropped = selectTrianglesForView(mesh, { xMin: -0.1, xMax: 0.4, yMin: -0.1, yMax: 1.1 });
  assert.equal(cropped.visibleCount, 2);
});

test("mesh rendering selection applies a deterministic triangle budget", () => {
  const selection = selectTrianglesForView(mesh, { xMin: -1, xMax: 3, yMin: -1, yMax: 2 }, 2);
  assert.equal(selection.stride, 2);
  assert.deepEqual(selection.triangles, [mesh.triangles[0], mesh.triangles[2]]);
});
