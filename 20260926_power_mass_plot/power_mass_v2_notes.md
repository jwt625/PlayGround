# V2 transportation comparison

Eight transport examples extend the original chart to 40 objects. V1 values,
flags, and provenance are retained. The new points are all marked estimated,
including calculations from published specifications, and use dashed connectors.
Research date: September 26, 2026. Exact inputs, formulas, source URLs, and manual
label positions live in `power_mass_data_v2.json`.

## Power and mass boundaries

Road vehicles use estimated electrical input at the battery. Aircraft and
rockets use fuel chemical input, `P = fuel mass flow × lower heating value`.
This excludes upstream electricity generation and fuel manufacturing. Rocket
fuel flow excludes oxygen mass when multiplying by the fuel's heating value.
Fuel-rich exhaust can retain chemical energy, so input is not identical to
released chamber heat. Neither thrust nor charging capacity is plotted as
propulsion power.

The new transport masses include the energy store and operating load: batteries
for the RC car, a 75 kg rider/driver for scooter/car, maximum takeoff mass for
aircraft, and a full fueled reference stack for rockets. Each vertical interval
uses one reference mass; it does not track fuel depletion. These are engineering
reference configurations, not equally loaded or mission-normalized vehicles.
Human metabolic input is not added to the scooter/car electrical input.

The original spacecraft points retain their electrical consumption/generation
definitions. V2 therefore broadens an engineering comparison; it does not make
every inherited endpoint a thermodynamically identical system boundary.

## Added points

Numbers below are rounded; calculations in the JSON retain precision.

| Object | Reference mass | Open marker | Filled marker | Main assumption or limitation |
| --- | ---: | ---: | ---: | --- |
| RC car: Traxxas X-Maxx | 10.1 kg | 5 W ready | 2.96 kW | Loaded mass and current assumed; 29.6 V × 100 A is an illustrative transient, not a measured limit. |
| Segway MAX G2 D + rider | 99.25 kg | 5 W ready | 1.08 kW | Published 24.25 kg + 75 kg rider; 36 V × assumed 30 A. |
| Tesla Model 3 Performance + driver | 1,914 kg | 300 W ready | 423 kW | 2024 US 510 hp reference divided by assumed 90% efficiency; ready power assumes HVAC off. |
| Boeing 737-800 | 79,010 kg | 9.72 MW ground idle | 105 MW takeoff | Two CFM56-7B26 engines; certification fuel flow × 43 MJ/kg. APU omitted. |
| Airbus A380 | 575,000 kg | 44.4 MW ground idle | 460 MW takeoff | Four Trent 972-84 engines; certification fuel flow × 43 MJ/kg. APU omitted. |
| F-16 with F100-PW-220 | 17,010 kg | 6.45 MW ground idle | 271 MW afterburner | Idle flow assumed; peak from USAF engine maximum flow. Family reference MTOW, not a tail-specific loading. |
| Falcon 9 | 549,054 kg | 23.3 GW modeled 70% flow | 33.2 GW full booster flow | Sea-level thrust, assumed Isp/mixing ratio, kerosene LHV. |
| Starship + Super Heavy | 5,800,000 kg | 190 GW modeled 70% flow | 271 GW full booster flow | Published configuration plus estimated dry/payload mass, Isp, and mixing ratio. |

The modeled 70%-flow open rocket markers and their connectors are **hidden in
the active v2 figure** via `plot_config.hidden_endpoints`; their values remain
in the JSON. They are illustrative powered states, not idle, measured minimum
throttle, cruise, or a claim that the fully fueled stack can lift off at that
setting. Zero engine-off propulsion input cannot be shown on logarithmic axes.
The visible rocket markers describe booster flow at sea level; upper-stage
thrust is not added to booster thrust.

## Sources and calculations

### Peak specific-power reference lines

Two dashed guides use reported peak electrical output per **cell mass**.
Neither requires choosing an arbitrary discharge duration to convert energy
to power. Their operating conditions remain attached to the labels. Two
further dashed guides cover a 5-minute (300 s) burn of ideal stoichiometric
fuel-plus-carried-oxygen mixtures. They are a different boundary: chemical
input per total propellant mass, not electrical output per cell mass.

| Benchmark | Peak specific power | Source and calculation |
| --- | ---: | --- |
| Molicel P60C battery cell | 8.8 kW/kg, 2 s pulse | [Manufacturer peak claim](https://www.molicel.com/cn/inr-21700-p60c/): 660 W for 2 s. Divide by 0.075 kg maximum cell mass from the [tentative datasheet](https://www.molicel.com/wp-content/uploads/Product-Data-Sheet-of-INR-21700-P60C.pdf), v0.1, October 23, 2025. |
| Metal-foam-supported SOFC laboratory cell | 6.56 kW/kg at 650°C | Directly reported by [Ma et al., Advanced Science, 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202517694), using peak electrical power from V–I characterization divided by cell mass, under humidified hydrogen. |

The SOFC peak is a valid specific-power measurement even though a sustained
hold duration at that peak is not established. The battery number is a
manufacturer pulse claim, not an independently verified continuous rating.
Pack/stack hardware, thermal-management equipment and external reactant
inventories are excluded. These are named contemporary high-performance
benchmarks, not universal world-record claims. Extrapolating the lines across
mass does not establish feasible complete systems of every size. The new
transport points use input power; these guides use electrical output.

| Fuel guide | Average chemical input | Source and calculation |
| --- | ---: | --- |
| H2 + O2, 5 min | 44.4 kW/kg | 120 MJ/kg H2 LHV / (1 + 8) = 13.33 MJ/kg mixture, divided by 300 s. |
| CH4 + O2, 5 min | 33.3 kW/kg | 50 MJ/kg CH4 LHV / (1 + 4) = 10.00 MJ/kg mixture, divided by 300 s. |

The fuel guides use ideal stoichiometric ratios and total fuel + carried
oxidizer mass, so they exclude tanks, engines, and hardware and are not
whole-system envelopes. They also differ from the fuel-rich operating
mixtures behind the plotted rocket points, and aircraft/animals draw oxygen
from air rather than carrying it. See [the duration survey](power_mass_peak_duration_survey.md).

### Off-scale arrows

Red arrows mark references whose peak power exceeds the 10^12 W top of the
axis. Their masses are plotted to scale on the x-axis, but the arrows are
schematic in power. For the explosives, peak power is not a published
specification; it uses yield divided by an assumed energy-release time.

| Reference | Mass | Energy | Assumed release time | Estimated peak power |
| --- | ---: | ---: | ---: | ---: |
| Little Boy / Fat Man | ~4,500 kg | 13–23 kt (54–96 TJ) | ~1 µs | ~6–9×10^19 W |
| MOAB | 9,850 kg | 11 t TNT (46 GJ) | ~1 ms | ~5×10^13 W |
| Tsar Bomba | 27,000 kg | 50 Mt (210 PJ) | ~1 µs | ~2×10^23 W |
| 1 kg black hole | 1 kg | 9.0×10^16 J (rest energy) | ~8.4×10^-17 s | ~3.6×10^32 W |

Explosive yields and masses are from Wikipedia ([Little Boy](https://en.wikipedia.org/wiki/Little_Boy),
[Fat Man](https://en.wikipedia.org/wiki/Fat_Man),
[Tsar Bomba](https://en.wikipedia.org/wiki/Tsar_Bomba),
[Nuclear weapon yield](https://en.wikipedia.org/wiki/Nuclear_weapon_yield),
[GBU-43/B MOAB](https://en.wikipedia.org/wiki/GBU-43/B_MOAB)). The ~1 µs nuclear
figure follows the Little Boy article's statement that the chain reaction
lasted less than 1 microsecond; the ~1 ms MOAB figure is an order-of-magnitude
detonation time. The 1 kg black hole uses the Hawking radiation power
P = hbar c^6 / (15360 pi G^2 M^2) = ~3.6×10^32 W, with an evaporation time of
about 8.4×10^-17 s (a semiclassical theoretical value). These are illustrative
scale markers, not measurements.

### Transportation calculations

Aircraft engine fuel flows come from the March 2026
[ICAO/EASA databank](https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank).
In its `Gaseous Emissions and Smoke` sheet, columns CC and BZ contain idle and
takeoff flow in kg/s per engine:

| Engine | Databank UID | Row | Idle | Takeoff |
| --- | --- | ---: | ---: | ---: |
| CFM56-7B26 | 8CM051 | 137 | 0.113 kg/s | 1.221 kg/s |
| Trent 972-84 | 01P18RR104 | 777 | 0.258 kg/s | 2.672 kg/s |

Multiply by engine count and a representative 43 MJ/kg kerosene LHV, consistent
with the JP-8 scale in [NASA/TM-2006-214328](https://ntrs.nasa.gov/api/citations/20060021979/downloads/20060021979.pdf).
These are static idle/takeoff conditions, not cruise. Masses follow
[Boeing's 737NG specifications](https://www.boeing.com/commercial/737ng) and
[Airbus's A380 factsheet](https://www.airbus.com/sites/g/files/jlcbta136/files/2021-12/EN-Airbus-A380-Facts-and-Figures-December-2021_0.pdf).

[Luke AFB](https://www.luke.af.mil/News/Photos/igphoto/2000836396/) gives F100-220
maximum fuel flow as 49,964 lb/h, which yields about 271 MW at 43 MJ/kg. The
0.15 kg/s ground-idle flow is an assumption. The
[USAF F-16 factsheet](https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104505/f-16-fighting-falcon/)
lists 37,500 lb MTOW with a rough metric conversion. This dataset explicitly
converts the pounds figure using 0.45359237 kg/lb.

For rockets:

```text
total propellant flow = sea-level thrust / (Isp × g0)
fuel flow = total propellant flow / (1 + oxygen-to-fuel mass ratio)
chemical input = fuel flow × fuel LHV
```

[SpaceX's Falcon 9 page](https://new.spacex.com/vehicles/falcon-9) supplies mass
and 7.607 MN booster thrust. The calculation assumes 282 s Isp, a 2.56 oxygen/fuel
ratio, and 43 MJ/kg RP-1 LHV. The mixing ratio is also used in this
[NASA-hosted research model](https://ntrs.nasa.gov/api/citations/20250003113/downloads/2024_JSR_LaunchVehicleAmbiguityRemediation.pdf).

The [SpaceX Starship page](https://new.spacex.com/vehicles/starship) snapshot
specifies 1,600 t ship propellant, 3,650 t booster propellant, and 8,240 metric
ton-force booster thrust. This model adds an assumed 550 t dry hardware/payload,
uses assumed 330 s Isp, and carries forward the 3.6 oxygen/methane ratio from the
[2022 FAA assessment](https://www.faa.gov/sites/faa.gov/files/2022-06/PEA_for_SpaceX_Starship_Super_Heavy_at_Boca_Chica_FINAL.pdf).
Methane LHV is 50 MJ/kg, as in [Boeing Cascade](https://docs.cascade.boeing.com/docs/energy/methane.html).
This is a defined reference configuration, not an assertion about every Starship
flight or the next design revision. A 10% change in assumed Isp changes inferred
chemical input by approximately 10% in the opposite direction.

Road references are [Traxxas's mass comparison](https://traxxas.com/articles/maxx-vs-xmaxx),
its [8S architecture description](https://traxxas.com/articles/xmaxx-monster-truck-thrill-ride),
[Segway's MAX G2 D specifications](https://se-en.segway.com/products/max-g2-d),
[Tesla's curb-mass specification](https://www.tesla.com/model3-performance), and
[Tesla's official 2024 Performance launch video](https://www.youtube.com/watch?v=krQKnhMwxn4).
Segway's 900 W motor rating does not identify the power boundary clearly enough
to treat it as a measured battery input; the 1.08 kW electrical point remains
explicitly modeled.

The weakest new inputs are RC/scooter peak currents and road ready power. Battery
voltage/current logging would directly validate them. Next priorities are a
traceable F100-220 idle fuel flow and flight/configuration-specific rocket flow
and mass data. Known v1 provenance issues remain documented in `README.md`.
