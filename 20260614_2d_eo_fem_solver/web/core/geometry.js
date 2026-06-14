export function parseDomain(config) {
  const block = config.Domain;
  return {
    xMin: Number(block.x_min),
    xMax: Number(block.x_max),
    yMin: Number(block.y_min),
    yMax: Number(block.y_max),
  };
}

export function parseElectrodes(config) {
  return Object.entries(config.Electrodes).map(([name, block]) => {
    const params = {};
    for (const [key, value] of Object.entries(block)) {
      if (key !== "shape" && key !== "potential") {
        params[key] = Number(value);
      }
    }
    return {
      name,
      shape: String(block.shape),
      potential: Number(block.potential),
      params,
    };
  });
}

export function electrodeContains(electrode, x, y) {
  const p = electrode.params;
  if (electrode.shape === "rectangle") {
    return p.x_min <= x && x <= p.x_max && p.y_min <= y && y <= p.y_max;
  }
  if (electrode.shape === "circle") {
    const dx = x - p.x;
    const dy = y - p.y;
    return dx * dx + dy * dy <= p.radius * p.radius;
  }
  throw new Error(`unsupported electrode shape: ${electrode.shape}`);
}
