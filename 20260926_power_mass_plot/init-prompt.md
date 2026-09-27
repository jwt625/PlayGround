# Codex handoff: reproduce and continue the power-vs-mass plot

You are taking over an existing Python/matplotlib workstream. Reproduce the current plot closely enough that future edits are incremental rather than a redesign.

## Goal

Build a portrait, mobile-friendly log-log plot of **mass (kg)** versus **whole-body/system power (W)** for four categories:

- land animals
- sea animals
- compute systems
- spacecraft

Each object has an **idle/baseline** power and a **peak/maximum** power at the same x-coordinate, connected by a vertical line.

Use `power_mass_data.json` as the source of truth for:
- point values
- estimated vs sourced status
- references/provenance
- manual label positions
- plot configuration

Do not silently replace estimates. If you improve a value, update the JSON and preserve provenance.

## Visual grammar

1. Use Python + matplotlib. Do not use seaborn.
2. Both axes are logarithmic.
3. Figure size: 7.2 × 11.0 in.
4. Output: 220 dpi PNG.
5. x limits: 0.015 to 7e5 kg.
6. y limits: 0.012 to 5e7 W.
7. x-axis: `Weight (kg)`
8. y-axis: `Whole-body / system power (W)`
9. No title.

Category markers:
- land = circle
- sea = square
- compute = diamond
- space = upward triangle

For every object:
- idle = open marker
- peak = filled marker
- same x coordinate
- vertical connector between them

If `estimated == true`, make the connector dashed.

Use matplotlib default color cycle:
- land = color 0
- sea = color 1
- compute = color 2
- space = color 4

## Specific-power guide lines

Add three diagonal dashed light-grey lines:

- 1 W/kg
- 10 W/kg
- 100 W/kg

These obey:

P = (specific power) × M

Use:
- grey around `0.75`
- linewidth ~1.1
- dashed
- thinner and lighter than object connectors
- label each near the lower-left portion of the chart in darker grey

Do **not** plot a global power-law fit line. We tried one and intentionally removed it because the specific-power lines communicate the interesting comparison more directly.

## Labels

Use the exact manual positions in:

`plot_config.label_positions`

These positions were iteratively tuned by visually inspecting the rendered PNG.

Do not replace them with automatic label placement unless explicitly asked.

Important manual choices:
- Voyager 1 label is on the right side of its markers.
- Thoroughbred horse label is higher than Starlink V1.
- Starlink V1 label is lower.
- Starlink V2 Mini and Starlink V3 labels are to the left of and close to their peak markers.
- Avoid leader/callout lines.
- Favor whitespace and unambiguous association over mathematically centering each label.

If a newly added point creates overlap:
1. render,
2. inspect the PNG,
3. manually move labels,
4. repeat until clean.

## Legend

Top-left legend:

- Land idle
- Land peak
- Sea idle
- Sea peak
- Compute idle
- Compute peak
- Space idle
- Space peak
- Estimated

Grid:
- major alpha ~0.35
- minor alpha ~0.18

## Semantics

“Idle” is not perfectly uniform across categories:

- animal: resting/basal/field-low metabolic power
- compute: powered-on idle or representative low-load draw
- spacecraft: normal/baseline operating power
- “peak” spacecraft values may represent available/generated electrical power

Likewise, “peak” animals usually means whole-body metabolic power, not useful mechanical shaft output.

This is intentionally an engineering comparison plot rather than a physiology-standardized dataset.

## Current data-quality convention

There are three kinds of points:

1. directly published measurements/specifications
2. values derived from published metabolic/VO2 numbers
3. engineering estimates

Anything meaningfully estimated should have `estimated: true`.

Some objects have one strong endpoint and one estimated endpoint; the entire vertical connector is still dashed.

## Important current points

Examples include:

Animals:
- House mouse
- Rabbit
- Dog / Alaskan Husky
- Human
- Thoroughbred horse
- African elephant
- Atlantic salmon
- Harbor seal
- Bottlenose dolphin
- Killer whale
- Blue whale

Compute:
- iPhone (2007)
- iPhone 17 Pro Max
- Apple II
- Macintosh 128K
- IBM PC 5150
- DGX B300
- GB300 NVL72
- Cray-1
- Cray-2
- ENIAC
- Summit
- Frontier

Space:
- Planet Dove
- MarCO
- Voyager 1
- Starlink V1
- Starlink V2 Mini
- Starlink V3
- JWST
- Hubble
- ISS

## References already used

The JSON contains the complete source dictionary and per-point source keys.

Key references:

### Animal metabolism

Mammalian BMR scaling:
https://www.pnas.org/doi/10.1073/pnas.0436428100

Mammalian maximum metabolic scaling:
https://journals.biologists.com/jeb/article/208/9/1611/9373/Allometric-scaling-of-mammalian-metabolism

Rabbit VO2max:
https://pubmed.ncbi.nlm.nih.gov/19923995/

Alaskan Husky VO2max:
https://journals.physiology.org/doi/full/10.1152/japplphysiol.00588.2014

Atlantic salmon:
https://onlinelibrary.wiley.com/doi/10.1111/jfb.14087
https://pmc.ncbi.nlm.nih.gov/articles/PMC11200746/

Bottlenose dolphin:
https://pubmed.ncbi.nlm.nih.gov/8340731/
https://link.springer.com/article/10.1007/s002270050575

Killer whale:
https://sciences.ucf.edu/biology/PEBL/wp-content/uploads/sites/18/2013/12/2013-Worthy-et-al-MMS.pdf
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0302758

Blue whale:
https://link.springer.com/article/10.1007/s00360-025-01640-1
https://journals.biologists.com/jeb/article/214/1/131/10226/Mechanics-hydrodynamics-and-energetics-of-blue

### Compute

Original iPhone standby:
https://www.apple.com/newsroom/2007/06/18iPhone-Delivers-Up-to-Eight-Hours-of-Talk-Time/

Macintosh 128K:
https://support.apple.com/en-ca/112190

DGX B300:
https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html

GB300 NVL72:
https://www.hpe.com/ca/fr/collaterals/collateral.a50009244enw.html

Frontier:
https://www.olcf.ornl.gov/olcf-resources/compute-systems/frontier/

Frontier cabinet mass:
https://www.datacenterdynamics.com/en/news/oak-ridge-upgrades-data-center-set-to-be-home-to-the-worlds-first-exascale-supercomputer-frontier/

Summit:
https://www.hpcwire.com/2018/06/08/ornl-summit-supercomputer-is-officially-here/

### Space

ISS:
https://www.nasa.gov/international-space-station/space-station-facts-and-figures/

Hubble:
https://science.nasa.gov/mission/hubble/overview/hubble-by-the-numbers/

JWST:
https://science.nasa.gov/mission/webb/fact-sheet/

Voyager 1:
https://science.nasa.gov/mission/voyager/voyager-1/

Starlink V1:
https://discovery.ucl.ac.uk/10198161/1/SECESA_2024.pdf

Starlink V2 Mini:
https://spaceflightnow.com/2023/04/19/falcon-9-starlink-6-2-coverage/

Starlink V3:
https://starlink.com/ls/updates/starlink-version-3-satellites

MarCO:
https://www.jpl.nasa.gov/news/press_kits/insight/launch/appendix/mars-cube-one/

## Known data-quality issues worth revisiting

Do not silently change these, but flag or research them when appropriate:

- DGX B300:
  - published busbar mass = 123 kg
  - NVIDIA currently states ~14.5 kW power consumption / ~15 kW system max
  - workstream currently uses 19 kW peak as an earlier engineering estimate
  - if strict source fidelity is desired, change peak to 15 kW

- GB300 NVL72:
  - ~1500 kg is supported by HPE’s ~3300 lb fully loaded rack value
  - 155 kW peak is published
  - ~30 kW idle is estimated

- Starlink V2 Mini power is estimated from array size / architecture.

- Starlink V3:
  - SpaceX states V3 arrays generate roughly 2× V2 power
  - current 22 kW idle / 56 kW peak are engineering estimates
  - mass is treated as ~2000 kg

- Planet Dove electrical power is representative/estimated.

- Frontier and Summit total mass are estimated from rack/cabinet counts and weights.

- Historical computer idle powers are generally less well documented than nameplate/max power.

- Human 100 W / 2.2 kW is a representative comparison point rather than one specific measured individual.

## Development rules

Keep all data external to plotting code.

`power_mass_data.json` should own:
- numeric values
- category
- estimated flag
- source references
- label positions
- style/config

`plot_power_mass.py` should only:
- read JSON
- render the figure
- write PNG

When adding a new object:
1. research mass
2. research idle/baseline power
3. research peak/max power
4. distinguish published vs derived vs estimated
5. add references
6. add the data point
7. choose a label position
8. render
9. visually inspect and iterate

Do not redesign the visual language unless explicitly requested.

## Deliverables

Maintain:

- `power_mass_data.json`
- `plot_power_mass.py`
- `power_mass_plot.png`

First reproduce the current chart exactly from the JSON.

Then, if you notice data-quality inconsistencies, report them separately rather than silently modifying the dataset.