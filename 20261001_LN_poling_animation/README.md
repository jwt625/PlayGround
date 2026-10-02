# LiNbO3 periodic poling animation

Animated, interactive visualization of electrical poling of x-cut LiNbO3 (z in-plane, along the field) (Q = 90 pC case), driven by one shared pulse time:

- (a) voltage and current vs time with moving dashed cursor and markers
- (b) SHG-microscope-style top view of stripe domains (x-cut film, crystal z vertical along the field) between the +V and ground electrodes
- (c) multi-unit-cell x-cut lattice film (z along the electrode-to-electrode direction) with the domain front (playback slows during switching), orbiting camera
- (d) one hexagonal unit cell at true scale: Li hops through its O3 triangle, Nb shifts in its O6 octahedron; applied-field and internal-field effects; double-well inset

## Run

Open `index.html` in a browser (no build, no server). Space = play/pause; top-bar button toggles the camera orbit of (c) and (d); the arrow button in (c) toggles the per-cell Ps arrows; drag the graph or the slider to scrub. Hover panel titles for details and sources. Test URL: `index.html?t=2.45&paused=1`.

## Current state

v1 built; see `DevLog/DevLog-000-request-and-plan.md` for decisions, model, sources and verification. Crystal data: Hsu et al., Acta Cryst. B53, 420 (1997) (COD 2101845), cached in `references/`. Regenerate lattice data: `uv run --with numpy python scripts/gen_lattice.py`.

The waveform is synthesized to match the features of the supplied figure (voltage trapezoid, charge, current peak timing); the domain geometry and the double-well inset are schematic.
