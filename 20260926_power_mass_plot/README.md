# Power versus mass

Portrait log–log comparison of 32 animals, compute systems, and spacecraft.
`power_mass_data.json` owns the values, provenance, estimated flags, exact manual
label positions, and explicit plotting styles. `plot_power_mass.py` reads it and
writes `power_mass_plot.png` beside the script, including when invoked from a
different working directory. Rendering is headless with matplotlib's Agg backend.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python plot_power_mass.py
```

Verified with Python 3.14 and the dependency versions in `requirements.txt`.
The figure is 7.2 × 11 inches and saved at 220 dpi. The original renderer's tight
bounding-box export is preserved, producing a 1548 × 2398 pixel PNG in this
environment. Set `plot_config.save_bbox_inches` to `null` for an uncropped
1584 × 2420 pixel canvas; this changes the original output margins.

## V2: transportation

`power_mass_data_v2.json` adds eight transportation examples to the original 32
records. Render it with the same script:

```sh
.venv/bin/python plot_power_mass.py --data power_mass_data_v2.json
```

This writes `power_mass_plot_v2.png`, preserving the v1 dataset and image.
`--output path.png` overrides the destination. V2 keeps the visual grammar,
extends the mass/power limits for rockets, and uses an 8.4 × 13 inch portrait
figure with manually revised labels above the markers. New road, air, and rocket
categories use distinct markers and default-cycle colors.

New transport values use battery electrical input or fuel chemical-energy input.
Read [the v2 methods and source notes](power_mass_v2_notes.md) for operating
states, loaded mass boundaries, calculations, and assumptions. The original
spacecraft still use their existing electrical consumption/generation endpoints.

The [peak-duration survey](power_mass_peak_duration_survey.md) compares operating
intervals, rated limits, and energy-budget estimates before selecting fuel
reference guides. The provisional 180-second guides are not in the active v2.
V2 includes peak electrical-output guides for the Molicel P60C battery cell
(8.8 kW/kg, 2 s pulse) and a metal-foam SOFC cell (6.56 kW/kg at 650°C), with
source data, calculations, and cell-mass boundaries stored in the JSON.
Two additional dashed fuel guides cover 5 minutes (300 s) of ideal
stoichiometric fuel-plus-oxygen combustion: H2 + O2 at 44.4 kW/kg and
CH4 + O2 at 33.3 kW/kg of total propellant mass. These are chemical input,
not electrical output, and exclude tanks, engines, and hardware.

Three red off-scale arrows add explosive references whose estimated peak
power exceeds the 10^12 W axis top: Little Boy / Fat Man (~6–9×10^19 W),
MOAB (~5×10^13 W), and Tsar Bomba (~2×10^23 W). Masses are to scale; the
arrows are schematic in power and use yield divided by an assumed
energy-release time (see the v2 notes).

## Verification

The initial refactor produced pixels identical to the supplied renderer in
the same environment. Text labels now render at z-order 5, above markers at
z-order 4, so markers cannot obscure label text. All 32 original point records,
source records, metadata, and manual positions are unchanged. Checks covered log axes and
limits, 64 open/filled endpoints, connector styles, exact manual label positions,
three specific-power guides, and the nine legend entries. There is no global fit.

Inherited label/marker intersections remain, notably Human with the harbor seal
peak and Starlink V2 Mini with the horse peak, but text now draws above those
markers. Some labels cross other objects' connectors. The supplied manual
positions are preserved.

## Data-quality review

These findings are an internal consistency review of the supplied JSON and
handoff, not a fresh verification of the external references. Values and flags
were preserved as requested.

| Item | Issue to resolve before publication |
| --- | --- |
| DGX B300 | Peak is 19 kW, while its stored source note says 14.5 kW consumption / approximately 15 kW maximum. The current point is explicitly estimated. |
| Frontier | `estimated: false` conflicts with its notes: mass is derived from cabinet counts and idle is a working lower-load value. A dashed connector would better follow the stated convention. |
| Harbor seal | `estimated: false` but `source_keys` is empty; the earlier VO2 literature cannot be recovered from this dataset. |
| Derived animal values | Metadata says derived endpoints count as estimated, yet mouse, rabbit, salmon, husky, dolphin, and killer whale are marked false despite scaling or metabolic conversions in their source notes. Decide whether direct unit conversion differs from scaling or extrapolation, and record derivations per endpoint. |
| Missing references | Human, Thoroughbred horse, iPhone 17 Pro Max, and Apple II also have empty source lists. They are already marked estimated; the iPhone note specifically requests model verification. |
| Voyager 1 | The lower endpoint is a recent-era working power value and the upper endpoint is launch-era RTG generation. The interval mixes epochs rather than representing idle/peak at one epoch. The lower endpoint has no precise date. |
| Comparison boundaries | Animal metabolism, computer electrical consumption, and spacecraft consumption/generation differ. Blue whale baseline is field metabolism. Large-computer masses use estimated cabinet boundaries. These distinctions should accompany any reuse of the chart. |

Further known estimates include Starlink V2 Mini/V3 power, Planet Dove power,
Summit mass, and historical computer idle power. Their stored provenance remains
available in the JSON.
