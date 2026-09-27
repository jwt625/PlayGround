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
