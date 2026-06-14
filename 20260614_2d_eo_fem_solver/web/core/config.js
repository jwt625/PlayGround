export function parseSimpleYaml(text) {
  const root = {};
  const stack = [{ indent: -1, value: root }];
  const lines = text.split(/\r?\n/);
  for (let lineNumber = 0; lineNumber < lines.length; lineNumber += 1) {
    const raw = lines[lineNumber];
    const line = raw.split("#", 1)[0].trimEnd();
    if (line.trim() === "") continue;
    const indent = line.length - line.trimStart().length;
    if (indent % 2 !== 0) {
      throw new Error(`line ${lineNumber + 1}: indentation must use multiples of two spaces`);
    }
    const stripped = line.trim();
    const colon = stripped.indexOf(":");
    if (colon < 0) {
      throw new Error(`line ${lineNumber + 1}: expected key: value`);
    }
    const key = stripped.slice(0, colon).trim();
    const valueText = stripped.slice(colon + 1).trim();
    while (indent <= stack[stack.length - 1].indent) {
      stack.pop();
    }
    if (stack.length === 0) {
      throw new Error(`line ${lineNumber + 1}: invalid indentation`);
    }
    const parent = stack[stack.length - 1].value;
    if (valueText === "") {
      const child = {};
      parent[key] = child;
      stack.push({ indent, value: child });
    } else {
      parent[key] = parseScalar(valueText);
    }
  }
  return root;
}

function parseScalar(value) {
  const lowered = value.toLowerCase();
  if (lowered === "true") return true;
  if (lowered === "false") return false;
  if (lowered === "none" || lowered === "null") return null;
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric) && value.trim() !== "") {
    return numeric;
  }
  return value;
}
