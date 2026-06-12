# Context summary for Codex task: draw cable bundle cross-section SVG accurately

I was analyzing a hypothetical NVL576-style inter-rack optical scale-up fabric using 200G VCSEL links, where each optical fiber carries 200 Gb/s in one direction. The goal is to visualize how physically ugly the fiber/cable egress becomes depending on cable fiber count.

## Prior bandwidth / fiber-count context

Assume an 8-rack NVL576-like system with a non-oversubscribed rack-to-rack NVLink optical fabric.

Earlier rough bandwidth estimate:
- One NVL72 rack: 72 GPUs × 3.6 TB/s bidirectional per GPU ≈ 259.2 TB/s bidirectional total endpoint NVLink bandwidth.
- In an 8-rack NVL576, uniform all-to-all traffic means most traffic is off-rack.
- Estimated off-rack bandwidth per rack: ≈ 227 TB/s bidirectional.
- At 200 Gb/s per fiber per direction, this becomes about 9072 individual fibers leaving one rack.

Use this as the visualization assumption:
- Total fibers leaving one rack: 9072 individual fibers.
- Each individual fiber direction is counted separately.
- We are comparing physical cable-body cross-sections for carrying those 9072 fibers using different cable formats.

## Cable formats and dimensions to visualize

Use these cases:

| Case | Fibers per cable | Number of cables needed for 9072 fibers | Cable OD | Equivalent-area OD |
|---|---:|---:|---:|---:|
| 8F trunk | 8 | 1134 | 4.0 mm | ~135 mm |
| 144F cable | 144 | 63 | 9.6 mm | ~76 mm |
| 288F cable | 288 | 32 | 12.1 mm | ~68 mm |
| 864F cable | 864 | 11 | 11.4 mm | ~38 mm |
| 3456F cable | 3456 | 3 | 23.5 mm | ~41 mm |
| 6912F cable | 6912 | 2 | 29.0 mm | ~41 mm |

Equivalent-area OD means:
D_equiv = sqrt(4 * N * pi * (d/2)^2 / pi) = d * sqrt(N)

These are raw cable cross-sectional areas only. They exclude:
- Bend radius
- Routing slack
- Connector/fanout hardware
- Strain relief
- Air gaps between independently routed harnesses
- Cable trays, clips, MPO fanouts, breakouts, etc.

## What I want Codex to generate

Generate a real, accurate, self-contained SVG using actual code, not a hand-drawn AI image.

The SVG should visually compare actual cable bundle cross-sections leaving one rack. It should draw the cable bundle as a head-on cross-section: circles inside a larger dashed boundary. Each small circle represents one physical cable, not one fiber.

Important: the drawing count must match the text. If the label says 63 cables, draw exactly 63 cable circles. If it says 1134 cables, draw exactly 1134 cable circles.

## Desired visual style

Create a clean engineering-slide-style SVG:
- White background.
- Title: “Actual packed cable bundle cross-sections leaving one rack”
- Subtitle: “Assumption: 9072 total fibers leaving one rack, 200G per fiber”
- Six cases arranged left-to-right.
- All diagrams drawn to a common physical scale so relative sizes are accurate.
- For each case:
  - Draw each cable as a filled circle.
  - Draw exactly the required number of circles.
  - Circle radius proportional to cable OD.
  - Arrange circles with a reasonable packing algorithm.
  - Draw an outer dashed circle showing the actual packed bundle OD from the generated layout.
  - Optionally draw a thin light-gray circle showing the equivalent-area OD for reference.
  - Under the diagram, label:
    - case name
    - count: X cables
    - cable OD: Y mm
    - equivalent-area OD: Z mm
    - packed bundle OD: computed value from layout

## Packing requirement

Use actual deterministic packing code.

For large N cases, use hexagonal packing:
- Generate a 2D hex lattice with spacing 2r horizontally and sqrt(3)r vertically.
- Select points nearest the origin until N points are selected.
- The actual packed bundle radius is max(sqrt(x_i^2 + y_i^2) + r).
- This gives a compact circular-ish cluster.

For N = 2:
- Use two tangent circles side by side.
- Packed bundle OD should be 4r = 2 × cable OD.
- For 6912F case with cable OD 29.0 mm and N=2, packed OD should be about 58 mm.

For N = 3:
- Use three tangent circles in an equilateral triangle.
- Compute packed OD from actual geometry, not equivalent area.
- For 3456F case with cable OD 23.5 mm and N=3, packed OD should be about 50.6 mm.

For N = 11:
- Hex cluster nearest origin is fine.
- Packed OD will be larger than equivalent-area OD.

Do not force all cable circles inside the equivalent-area OD circle. Equivalent-area OD is just the ideal area-equivalent diameter. The actual discrete packed OD is larger because circular cables cannot tile a circle at 100% fill factor.

## Expected approximate packed ODs

Using a hex-nearest-origin cluster, expected packed bundle ODs should be approximately:
- 8F trunk, 1134 × 4.0 mm cables: ~145 mm
- 144F, 63 × 9.6 mm cables: ~93 mm
- 288F, 32 × 12.1 mm cables: ~85 mm
- 864F, 11 × 11.4 mm cables: ~51 mm
- 3456F, 3 × 23.5 mm cables: ~51 mm
- 6912F, 2 × 29.0 mm cables: ~58 mm

The exact packed ODs may differ slightly depending on the deterministic hex-lattice selection, but they should be close and must be computed from the actual drawn layout.

## Deliverables

Please create:
1. A Python script, e.g. generate_cable_bundle_svg.py
2. A generated SVG file, e.g. actual_packed_cable_bundle_cross_sections.svg
3. Optional PNG preview rendered from the SVG, e.g. actual_packed_cable_bundle_cross_sections.png

The SVG must be standalone and viewable in a browser.

Use Python only. Prefer standard library for SVG generation. If rendering PNG, use cairosvg if available, but the SVG itself is the main deliverable.

## Implementation details

Suggested Python structure:
- Define cases as dictionaries:
  - name
  - fibers_per_cable
  - count
  - cable_od_mm
  - color
- Compute equivalent_area_od_mm = cable_od_mm * sqrt(count)
- Generate placement points in millimeters:
  - hex_cluster(n, r_mm)
  - triangle_cluster(r_mm)
  - two_cluster(r_mm)
- Compute packed_bundle_od_mm from generated points:
  - packed_od = 2 * max(hypot(x, y) + r_mm)
- Choose a global scale factor px_per_mm so the largest packed bundle is ~340 px diameter.
- Lay out six groups horizontally.
- Draw:
  - light gray equivalent-area circle
  - dashed dark gray actual packed bundle circle
  - filled cable circles
  - text labels
- Add note:
  - “Thin gray circle = equivalent raw-area OD. Dashed circle = actual packed OD from drawn layout.”
  - “Counts are exact; this excludes routing, bend radius, connectors, fanout, and slack.”

## Important correctness checks

Before writing final SVG:
- Assert that the number of circle elements generated for each case equals the count.
- Print or include computed packed ODs.
- Make sure the 6912F case draws exactly two large circles, not a stylized blob.
- Make sure the 3456F case draws exactly three large circles in a triangle.
- Make sure the 864F case draws exactly 11 circles.
- Make sure the 144F case draws exactly 63 circles.
- Make sure the 288F case draws exactly 32 circles.
- Make sure the 8F case draws exactly 1134 small circles.

## Why this matters

The previous generated image was wrong because it was an AI-rendered infographic: the number of circles in the drawing did not match the text labels, the sizes were not computed from code, and the packings were approximate. I need a code-generated technical SVG where the count and geometry are internally consistent.