# DevLog 001: Heat Transfer Demo Work Plan

Date: 2026-06-10

## Scope

Implement a static web app for a simplified 2D heat-transfer-only crystal-growth furnace demo.

Out of scope for this phase:

- Navier-Stokes flow.
- Moving mesh.
- Latent heat / enthalpy method.
- Radiation view factors.
- Quantitative validation.
- Package-based plotting or simulation libraries.

## Model

Use a structured `r-z` grid and update temperature with an explicit finite-difference heat equation:

```text
dT/dt = alpha * (d2T/dr2 + d2T/dz2 + axisymmetric_factor * (1/r) * dT/dr)
```

This is conduction-only. The UI labels it as a qualitative demo.

## Geometry

Use a representative Czochralski-like cross-section:

- Central crystal rod extending downward from the top.
- Silicon melt pool in a quartz crucible.
- Graphite heater bands around the crucible.
- Ambient/insulation background.

The geometry is parameterized by domain width/height and simple normalized dimensions. The first app uses normalized coordinates but displays approximate meter-scale values.

## Controls

- Start/pause.
- Reset.
- Heater temperature.
- Crystal cooling temperature.
- Ambient/wall temperature.
- Furnace size preset.
- Material preset.
- Simulation speed.
- Axisymmetric correction toggle.

## Progress

- [x] Write CGSim-replication proposal.
- [x] Write phase-1 work plan.
- [x] Research representative geometry/material anchors.
- [x] Create static webapp scaffold.
- [x] Implement heat solver.
- [x] Add controls and visualization.
- [x] Add stability/status readout.
- [x] Run syntax checks.

## Notes From Research

- Public CGSim material confirms that real tools require coupled global heat transfer, flow, radiation, interface prediction, impurities, stress/defects, and process-specific setup.
- Public Cz modeling examples generally use 2D axisymmetric geometry and couple continuity, Navier-Stokes/Boussinesq, and energy equations.
- Silicon properties vary strongly with temperature; the demo should expose presets rather than imply one correct constant value.
- Fused silica/quartz thermal conductivity is roughly 1-1.5 W/m/K, so the crucible is a strong thermal bottleneck compared with silicon and graphite.
- Graphite is highly grade-dependent; use a broad representative heater/susceptor value.

## Next Phase Candidates

1. Add convective/radiative boundary terms.
2. Add a prescribed toroidal melt circulation field and advection term.
3. Add an enthalpy method around the silicon melting point.
4. Add CSV export of centerline/radial temperature profiles.
5. Add geometry editing for crucible radius, melt depth, and crystal radius.

