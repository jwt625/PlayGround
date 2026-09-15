/** Minimal dependency-free SVG plotting for headless run reports. */

export interface Series {
  label: string;
  color: string;
  points: readonly [number, number][];
}

export interface PlotOptions {
  width?: number;
  height?: number;
  title?: string;
  xLabel?: string;
  yLabel?: string;
  yMin?: number;
  yMax?: number;
}

function scale(value: number, lo: number, hi: number, outLo: number, outHi: number): number {
  if (hi === lo) return (outLo + outHi) / 2;
  return outLo + ((value - lo) / (hi - lo)) * (outHi - outLo);
}

export function linePlotSvg(series: readonly Series[], options: PlotOptions = {}): string {
  const width = options.width ?? 720;
  const height = options.height ?? 420;
  const pad = { left: 60, right: 140, top: 44, bottom: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const allPoints = series.flatMap((s) => s.points);
  if (allPoints.length === 0) throw new Error("linePlotSvg: no points");
  const xMin = Math.min(...allPoints.map((p) => p[0]));
  const xMax = Math.max(...allPoints.map((p) => p[0]));
  const yMin = options.yMin ?? Math.min(...allPoints.map((p) => p[1]));
  const yMax = options.yMax ?? Math.max(...allPoints.map((p) => p[1]));

  const parts: string[] = [];
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
  parts.push(`<rect width="${width}" height="${height}" fill="#0d1117"/>`);
  if (options.title) {
    parts.push(`<text x="${width / 2}" y="24" fill="#e6edf3" font-family="monospace" font-size="15" text-anchor="middle">${escapeXml(options.title)}</text>`);
  }
  // Axes
  parts.push(`<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + innerH}" stroke="#30363d"/>`);
  parts.push(`<line x1="${pad.left}" y1="${pad.top + innerH}" x2="${pad.left + innerW}" y2="${pad.top + innerH}" stroke="#30363d"/>`);
  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (innerH * i) / 4;
    const value = scale(i / 4, 0, 1, yMax, yMin);
    parts.push(`<line x1="${pad.left}" y1="${y}" x2="${pad.left + innerW}" y2="${y}" stroke="#21262d"/>`);
    parts.push(`<text x="${pad.left - 8}" y="${y + 4}" fill="#8b949e" font-family="monospace" font-size="11" text-anchor="end">${value.toPrecision(4)}</text>`);
  }
  parts.push(`<text x="${pad.left + innerW / 2}" y="${height - 10}" fill="#8b949e" font-family="monospace" font-size="12" text-anchor="middle">${escapeXml(options.xLabel ?? "x")}</text>`);
  parts.push(`<text x="14" y="${pad.top + innerH / 2}" fill="#8b949e" font-family="monospace" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${pad.top + innerH / 2})">${escapeXml(options.yLabel ?? "y")}</text>`);

  series.forEach((s, si) => {
    const path = s.points
      .map((p, i) => {
        const x = scale(p[0], xMin, xMax, pad.left, pad.left + innerW);
        const y = scale(p[1], yMin, yMax, pad.top + innerH, pad.top);
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
    parts.push(`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"/>`);
    const legendY = pad.top + 14 * si;
    parts.push(`<line x1="${pad.left + innerW + 12}" y1="${legendY}" x2="${pad.left + innerW + 34}" y2="${legendY}" stroke="${s.color}" stroke-width="2"/>`);
    parts.push(`<text x="${pad.left + innerW + 40}" y="${legendY + 4}" fill="#c9d1d9" font-family="monospace" font-size="11">${escapeXml(s.label)}</text>`);
  });
  parts.push("</svg>");
  return parts.join("\n");
}

function escapeXml(value: string): string {
  return value.replace(/[<>&'"]/g, (c) =>
    c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === "&" ? "&amp;" : c === "'" ? "&apos;" : "&quot;",
  );
}
