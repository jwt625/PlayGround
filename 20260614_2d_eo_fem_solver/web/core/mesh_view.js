export function selectTrianglesForView(mesh, viewport, maxTriangles = 30_000) {
  const visible = [];
  for (let index = 0; index < mesh.triangles.length; index += 1) {
    const triangle = mesh.triangles[index];
    if (triangleIntersectsView(mesh, triangle, viewport)) visible.push(index);
  }
  const stride = Math.max(1, Math.ceil(visible.length / maxTriangles));
  const selected = [];
  for (let index = 0; index < visible.length; index += stride) selected.push(mesh.triangles[visible[index]]);
  return {
    triangles: selected,
    visibleCount: visible.length,
    totalCount: mesh.triangles.length,
    stride,
  };
}

function triangleIntersectsView(mesh, triangle, viewport) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const nodeIndex of triangle) {
    const [x, y] = mesh.nodes[nodeIndex];
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  return xMax >= viewport.xMin && xMin <= viewport.xMax && yMax >= viewport.yMin && yMin <= viewport.yMax;
}
