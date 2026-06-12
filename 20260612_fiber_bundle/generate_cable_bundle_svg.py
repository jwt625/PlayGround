#!/usr/bin/env python3
"""Generate an accurate SVG of packed optical cable bundle cross-sections."""

from __future__ import annotations

import html
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


SVG_PATH = Path("actual_packed_cable_bundle_cross_sections.svg")
PNG_PATH = Path("actual_packed_cable_bundle_cross_sections.png")


@dataclass(frozen=True)
class CableCase:
    name: str
    fibers_per_cable: int
    count: int
    cable_od_mm: float
    color: str

    @property
    def radius_mm(self) -> float:
        return self.cable_od_mm / 2.0

    @property
    def equivalent_area_od_mm(self) -> float:
        return self.cable_od_mm * math.sqrt(self.count)


CASES = [
    CableCase("8F trunk", 8, 1134, 4.0, "#4f83c4"),
    CableCase("144F cable", 144, 63, 9.6, "#38a169"),
    CableCase("288F cable", 288, 32, 12.1, "#dd6b20"),
    CableCase("864F cable", 864, 11, 11.4, "#805ad5"),
    CableCase("3456F cable", 3456, 3, 23.5, "#d53f8c"),
    CableCase("6912F cable", 6912, 2, 29.0, "#2c7a7b"),
]


def two_cluster(r_mm: float) -> list[tuple[float, float]]:
    """Two tangent circles side by side, centered on the origin."""
    return [(-r_mm, 0.0), (r_mm, 0.0)]


def triangle_cluster(r_mm: float) -> list[tuple[float, float]]:
    """Three mutually tangent circles in an equilateral triangle."""
    h = math.sqrt(3.0) * r_mm
    return [(-r_mm, -h / 3.0), (r_mm, -h / 3.0), (0.0, 2.0 * h / 3.0)]


def eleven_cluster(r_mm: float) -> list[tuple[float, float]]:
    """Eleven tangent circles arranged as three compact hex-staggered rows."""
    dy = math.sqrt(3.0) * r_mm
    points = []
    for y in (-dy, dy):
        points.extend((x * r_mm, y) for x in (-3.0, -1.0, 1.0, 3.0))
    points.extend((x * r_mm, 0.0) for x in (-2.0, 0.0, 2.0))
    return recenter(points)


def hex_cluster(n: int, r_mm: float) -> list[tuple[float, float]]:
    """Return n nearest points from a deterministic hexagonal lattice."""
    if n <= 0:
        return []

    dx = 2.0 * r_mm
    dy = math.sqrt(3.0) * r_mm

    # This bound is intentionally conservative and cheap for the requested N.
    max_ring = int(math.ceil(math.sqrt(n))) + 8
    candidates: list[tuple[float, float, float, int, int]] = []
    for row in range(-max_ring, max_ring + 1):
        y = row * dy
        x_offset = r_mm if row % 2 else 0.0
        for col in range(-max_ring, max_ring + 1):
            x = col * dx + x_offset
            dist2 = x * x + y * y
            candidates.append((dist2, y, x, row, col))

    candidates.sort()
    points = [(x, y) for _dist2, y, x, _row, _col in candidates[:n]]
    if len(points) != n:
        raise AssertionError(f"hex_cluster generated {len(points)} points, expected {n}")
    return recenter(points)


def recenter(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Place the centroid at the origin for visually stable layouts."""
    if not points:
        return points
    cx = sum(x for x, _y in points) / len(points)
    cy = sum(y for _x, y in points) / len(points)
    return [(x - cx, y - cy) for x, y in points]


def placement_points(case: CableCase) -> list[tuple[float, float]]:
    if case.count == 2:
        return two_cluster(case.radius_mm)
    if case.count == 3:
        return triangle_cluster(case.radius_mm)
    if case.count == 11:
        return eleven_cluster(case.radius_mm)
    return hex_cluster(case.count, case.radius_mm)


def packed_od_mm(points: list[tuple[float, float]], r_mm: float) -> float:
    return 2.0 * max(math.hypot(x, y) + r_mm for x, y in points)


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int,
    weight: int = 400,
    anchor: str = "middle",
    fill: str = "#1f2937",
) -> str:
    escaped = html.escape(text)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escaped}</text>'
    )


def circle(
    cx: float,
    cy: float,
    r: float,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0.0,
    extra: str = "",
) -> str:
    stroke_attrs = ""
    if stroke != "none" and stroke_width > 0:
        stroke_attrs = f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"'
    extra_attrs = f" {extra}" if extra else ""
    return (
        f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
        f'fill="{fill}"{stroke_attrs}{extra_attrs}/>'
    )


def generate_svg() -> tuple[str, list[tuple[CableCase, float]]]:
    layouts = []
    for case in CASES:
        points = placement_points(case)
        assert len(points) == case.count, (case.name, len(points), case.count)
        packed_od = packed_od_mm(points, case.radius_mm)
        layouts.append((case, points, packed_od))

    max_packed_od = max(packed_od for _case, _points, packed_od in layouts)
    px_per_mm = 280.0 / max_packed_od

    width = 1100
    height = 900
    columns = [210, 550, 890]
    centers_y = [270, 565]
    label_offset_y = 165

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Actual packed cable bundle cross-sections leaving one rack</title>',
        '<desc id="desc">Code-generated engineering diagram comparing exact cable counts packed on a common physical scale.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style><![CDATA[',
        'text { font-family: Inter, Avenir, Helvetica, Arial, sans-serif; }',
        '.small { fill: #4b5563; }',
        '.tiny { fill: #6b7280; }',
        ']]></style>',
        svg_text(width / 2, 52, "Actual packed cable bundle cross-sections leaving one rack", 28, 800),
        svg_text(width / 2, 86, "9072 total fibers per rack, 200G per fiber; every dot is one physical cable", 18, 500, fill="#4b5563"),
        svg_text(width / 2, 119, "All six drawings use the same physical scale", 15, 600, fill="#111827"),
    ]

    computed: list[tuple[CableCase, float]] = []
    for idx, (case, points, packed_od) in enumerate(layouts):
        cx = columns[idx % 3]
        cy = centers_y[idx // 3]
        label_y = cy + label_offset_y
        cable_r_px = case.radius_mm * px_per_mm
        equiv_r_px = (case.equivalent_area_od_mm / 2.0) * px_per_mm
        packed_r_px = (packed_od / 2.0) * px_per_mm

        parts.append(f'<g id="{html.escape(case.name.lower().replace(" ", "-"))}">')
        parts.append(
            circle(
                cx,
                cy,
                equiv_r_px,
                "none",
                "#c9ced6",
                1.4,
                'opacity="0.95"',
            )
        )
        parts.append(
            circle(
                cx,
                cy,
                packed_r_px,
                "none",
                "#374151",
                1.8,
                'stroke-dasharray="7 6"',
            )
        )

        for x_mm, y_mm in points:
            parts.append(
                circle(
                    cx + x_mm * px_per_mm,
                    cy + y_mm * px_per_mm,
                    cable_r_px,
                    case.color,
                    "#ffffff",
                    0.35,
                    'opacity="0.88"',
                )
            )

        # Invisible metadata marker used by the script's correctness check.
        parts.append(f'<metadata data-case="{html.escape(case.name)}" data-cables="{case.count}"/>')
        parts.append(svg_text(cx, label_y, case.name, 20, 800))
        parts.append(svg_text(cx, label_y + 28, f"{case.count} cables  |  cable OD {case.cable_od_mm:.1f} mm", 15, 500, fill="#374151"))
        parts.append(svg_text(cx, label_y + 51, f"packed OD {packed_od:.1f} mm  |  equiv-area {case.equivalent_area_od_mm:.0f} mm", 15, 650, fill="#111827"))
        parts.append("</g>")
        computed.append((case, packed_od))

    parts.extend(
        [
            '<line x1="70" y1="825" x2="1030" y2="825" stroke="#e5e7eb" stroke-width="1"/>',
            svg_text(
                width / 2,
                855,
                "Thin gray circle = equivalent raw-area OD. Dashed circle = actual packed OD from drawn layout.",
                15,
                fill="#4b5563",
            ),
            svg_text(
                width / 2,
                882,
                "Counts are exact; excludes routing, bend radius, connectors, fanout, and slack.",
                15,
                fill="#4b5563",
            ),
            "</svg>",
        ]
    )

    return "\n".join(parts), computed


def maybe_render_png() -> bool:
    if shutil.which("cairosvg"):
        subprocess.run(["cairosvg", str(SVG_PATH), "-o", str(PNG_PATH)], check=True)
        return True

    try:
        import cairosvg  # type: ignore
    except ImportError:
        pass
    else:
        cairosvg.svg2png(url=str(SVG_PATH), write_to=str(PNG_PATH))
        return True

    for renderer in ("magick", "convert"):
        if shutil.which(renderer):
            subprocess.run([renderer, str(SVG_PATH), str(PNG_PATH)], check=True)
            return True

    return False


def main() -> None:
    svg, computed = generate_svg()
    SVG_PATH.write_text(svg, encoding="utf-8")

    # Verify the visible cable circles match the requested total across all cases.
    expected_total = sum(case.count for case in CASES)
    visible_circles = svg.count('opacity="0.88"')
    assert visible_circles == expected_total, (visible_circles, expected_total)

    print(f"Wrote {SVG_PATH}")
    print("Computed packed bundle ODs:")
    for case, packed_od in computed:
        print(
            f"- {case.name}: {case.count} cables, "
            f"equiv-area OD {case.equivalent_area_od_mm:.1f} mm, "
            f"packed OD {packed_od:.1f} mm"
        )

    if maybe_render_png():
        print(f"Wrote {PNG_PATH}")
    else:
        print("PNG preview skipped: cairosvg is not installed.")


if __name__ == "__main__":
    main()
