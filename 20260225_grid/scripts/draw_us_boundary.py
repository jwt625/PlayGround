#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import shapefile


STATE_ZIP_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_20m.zip"
STATE_ZIP_PATH = Path("references/raw/geo/cb_2024_us_state_20m.zip")
EXCLUDED_STATE_CODES = {"02", "15", "60", "66", "69", "72", "78"}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>U.S. Boundary</title>
  <script src="https://cdn.plot.ly/plotly-3.3.1.min.js" crossorigin="anonymous"></script>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: rgba(255, 252, 246, 0.9);
      --ink: #182028;
      --muted: #5c646c;
      --accent: #0e6b5c;
      --border: rgba(24, 32, 40, 0.12);
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at 15% 20%, rgba(229, 181, 107, 0.25), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(49, 113, 102, 0.18), transparent 24%),
        linear-gradient(180deg, #f5f1e8 0%, #ece4d4 100%);
      color: var(--ink);
    }
    .shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 16px 50px rgba(50, 40, 20, 0.08);
      backdrop-filter: blur(8px);
      padding: 18px 20px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    p {
      margin: 0 0 12px 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }
    #map {
      width: 100%;
      height: 820px;
      margin-top: 12px;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <h1>Contiguous United States Boundary</h1>
      <p>
        Source: U.S. Census cartographic boundary shapefile, 2024 state boundaries. This view keeps only the lower-48 states
        and draws each state's largest exterior polygon as a standalone boundary line.
      </p>
      <div id="map"></div>
    </div>
  </div>
  <script>
    const DATA = __DATA_JSON__;
    const traces = DATA.rings.map((ring) => ({
      type: "scattergeo",
      mode: "lines",
      lon: ring.lon,
      lat: ring.lat,
      hoverinfo: "skip",
      line: {
        color: "#0f6b58",
        width: 1.6
      }
    }));

    Plotly.newPlot("map", traces, {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      showlegend: false,
      geo: {
        scope: "usa",
        projection: { type: "albers usa" },
        showland: true,
        landcolor: "#fbf6ee",
        showlakes: true,
        lakecolor: "#e6efe9",
        showcountries: false,
        showsubunits: true,
        subunitcolor: "rgba(0,0,0,0.12)",
        bgcolor: "rgba(0,0,0,0)"
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)"
    }, {
      responsive: true,
      displayModeBar: false
    });
  </script>
</body>
</html>
"""


def ensure_state_zip() -> Path:
    STATE_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_ZIP_PATH.exists():
        urlretrieve(STATE_ZIP_URL, STATE_ZIP_PATH)
    return STATE_ZIP_PATH


def ring_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def load_lower48_state_rings() -> list[dict[str, object]]:
    reader = shapefile.Reader(str(ensure_state_zip()))
    field_names = [field[0] for field in reader.fields[1:]]
    rings: list[dict[str, object]] = []

    for shape_record in reader.iterShapeRecords():
        record = dict(zip(field_names, shape_record.record))
        if str(record["STATEFP"]) in EXCLUDED_STATE_CODES:
            continue

        points = shape_record.shape.points
        parts = list(shape_record.shape.parts) + [len(points)]

        best_part: list[tuple[float, float]] | None = None
        best_area = -1.0
        for start, end in zip(parts[:-1], parts[1:]):
            part = [(float(x), float(y)) for x, y in points[start:end]]
            if len(part) < 3:
                continue
            area = ring_area(part)
            if area > best_area:
                best_area = area
                best_part = part

        if best_part is None:
            continue

        rings.append(
            {
                "state": str(record["STUSPS"]),
                "lon": [round(x, 4) for x, _ in best_part],
                "lat": [round(y, 4) for _, y in best_part],
            }
        )

    return rings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a standalone contiguous U.S. boundary HTML.")
    parser.add_argument("--out", type=Path, default=Path("outputs/us_boundary.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"rings": load_lower48_state_rings()}
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
