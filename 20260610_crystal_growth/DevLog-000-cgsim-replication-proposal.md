# DevLog 000: CGSim-Like Crystal Growth Simulator Proposal

Date: 2026-06-10

## Decision This Supports

Decide whether to build a simplified in-house crystal-growth simulator, and what first implementation slice gives useful learning without pretending to reproduce a commercial tool like CGSim.

## Reference Target

STR's CGSim is a commercial crystal-growth simulation suite, not a single PDE code. Its public pages describe furnace-scale heat transfer, melt flow, radiation, crystallization-front prediction, defects, stress, impurity transport, magnetic-field effects, 2D/3D flow, RANS/LES/DNS, and process-specific workflows for Cz, LEC, VCz, Bridgman, and related methods.

Sources:

- https://str-soft.com/software/cgsim/
- https://str-soft.com/software/cgsim/flow-module/

## Practical Difficulty Estimate

| Goal | Scope | Difficulty |
| --- | --- | --- |
| Visual heat-transfer demo | 2D transient conduction with simple furnace geometry and adjustable boundary temperatures | Days |
| Useful educational prototype | 2D axisymmetric conduction with region materials, approximate radiative/convection boundary conditions, and stable controls | 1-3 weeks |
| Research prototype | Coupled heat transfer + Boussinesq melt flow + species transport + enthalpy/free-boundary solidification for one process | 2-6 months |
| Process-specific engineering model | Validated model for one real hot zone, material, recipe, and sensor set | 6-18 months |
| CGSim-class product | Robust meshing, coupled global heat transfer, turbulent flow, radiation/view factors, MHD, defects/stress, UI, validation data, support | 3-5+ years with a small expert team |

## Recommended Architecture

Start with a static browser demo that solves transient heat transfer only. Keep it intentionally small:

- No third-party packages.
- Canvas-based visualization.
- Explicit finite-difference update.
- Fixed structured grid.
- Region-specific thermal diffusivity.
- Boundary knobs for heater temperature, crystal cooling, ambient temperature, and time step multiplier.

Then add complexity in this order:

1. Conduction-only fixed mesh.
2. Better boundary conditions: convective and radiative heat loss.
3. Axisymmetric radial term in the heat equation.
4. Phase map and melt-front visualization using a melting isotherm.
5. Enthalpy method for latent heat.
6. Prescribed melt circulation field for advective heat transport.
7. Solved incompressible Boussinesq flow.
8. Moving mesh / ALE only after the physics coupling is otherwise working.

## Why Moving Mesh Should Wait

Moving mesh is visually attractive but not the best first risk. The hard part is getting coupled heat transfer, boundary conditions, and material properties into a numerically stable model. For early work, a fixed-grid enthalpy method is more robust and easier to debug than ALE.

## Initial Material / Geometry Assumptions

The first demo uses representative silicon Czochralski-like geometry rather than a real tool design.

Approximate property anchors:

| Material | Density | Heat capacity | Thermal conductivity | Notes |
| --- | ---: | ---: | ---: | --- |
| Silicon near room temp | 2330 kg/m3 | ~700 J/kg/K | ~130-150 W/m/K | Strongly temperature dependent; drops at high T |
| Liquid silicon / hot silicon | ~2500 kg/m3 | ~800-1000 J/kg/K | tens of W/m/K | Use only as an approximate melt region in demo |
| Fused silica/quartz | ~2200 kg/m3 | ~740 J/kg/K | ~1.3-1.5 W/m/K | Crucible-like low conductivity region |
| Graphite | ~1700-1900 kg/m3 | ~700-2200 J/kg/K | grade-dependent, broad range | Heater/susceptor approximation |

Sources:

- Silicon thermal properties: https://www.ioffe.ru/SVA/NSM/Semicond/Si/thermal.html
- Silicon material data: https://www.matweb.com/search/datasheet_print.aspx?matguid=7d1b56e9e0c54ac5bb9cd433a0991e27
- Fused silica properties: https://accuratus.com/fused.html
- Fused silica ranges: https://www.azom.com/properties.aspx?ArticleID=1387
- Graphite data references: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782425&Mask=2
- Graphite properties overview: https://poco.entegris.com/content/dam/poco/resources/reference-materials/brochures/brochure-graphite-properties-and-characteristics-11043.pdf
- Example Cz geometry/physics framing: https://www.comsol.com/blogs/thermal-analysis-of-a-czochralski-crystal-growth-furnace
- 2D axisymmetric Cz modeling example: https://www.mdpi.com/2073-4352/12/12/1764

## First Phase Deliverable

Build a browser app that can answer qualitative questions:

- How does heater setpoint affect melt-zone temperature?
- How does crystal cooling change the axial gradient?
- How does low-conductivity crucible material shape the thermal field?
- What region is above the silicon melting point?
- Is the explicit solver stable for the chosen grid and timestep?

It should not claim quantitative predictive accuracy.

## Success Criteria

- Runs by opening `index.html` in a browser.
- No dependencies or package installation.
- Adjustable controls update the simulation live.
- Shows furnace cross-section, material regions, temperature heatmap, and melt isotherm.
- Includes clear caveats in the UI.
- Code is compact enough to modify directly.

