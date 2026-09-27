# Peak-duration survey before choosing fuel guides

Research date: September 26, 2026. This is a representative survey, not a
duration assignment to every plotted endpoint. The original dataset generally
does not specify peak averaging windows. No point values have been changed.
The provisional 180-second guides were removed from the active v2 JSON/PNG
pending selection of the comparison scenario.

## Existing points

| Point | Relevant duration | Evidence and interpretation |
| --- | --- | --- |
| Falcon 9 | 145–147 s from liftoff to first-stage cutoff | SpaceX's sample LEO/GTO timelines, Tables 8-4/8-3 in the [Falcon User's Guide](https://spacex.relayto.com/e/spacex-falcon-user-s-guide-oc92qkoxzo1jf). Engines start about 3 s before liftoff. These are ascent intervals with changing throttle, not constant peak holds. |
| Starship / Super Heavy | 142 s from liftoff to most-engine cutoff | [SpaceX Flight 12 timeline](https://www.spacex.com/launches/starship-flight-12), a particular flight/configuration. The existing v2 model independently gives 146 s of full-flow-equivalent booster inventory: 3,650,000 kg / 24,970 kg/s. That calculation consumes all booster propellant, ignores reserves, and is not a flight prediction. |
| Boeing 737-800 / Airbus A380 | 300 s takeoff-rating timescale | [FAA AC 33.7-1, section 7](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_33_7-1.pdf) sets the normal rated-takeoff thrust limit at 5 minutes. This is a rating limit, not observed time at maximum fuel flow; actual takeoffs can be derated and reduce thrust sooner. Aircraft/engine-specific limits still govern. |
| Human, maximal aerobic exercise | About 400 s (6.7 min) | Eight trained runners lasted 404 ± 101 s and 402 ± 113 s in repeated runs at maximal aerobic speed in [Billat et al., 1994](https://pubmed.ncbi.nlm.nih.gov/8164545/). Exercise duration includes oxygen-uptake kinetics; it is not a constant VO2max plateau. It informs the regime of the chart's representative 2.2 kW metabolic point, but does not validate that exact point. |
| F-16 / F100-PW-220 | About 8.4 min full-flow fuel-budget equivalent | Using a rounded 7,000 lb fuel inventory and the chart's 49,964 lb/h maximum fuel flow: 7,000 / 49,964 × 60. [Luke AFB fuel-capacity context](https://www.luke.af.mil/News/Photos/igphoto/2002175516/) and [engine fuel-flow source](https://www.luke.af.mil/News/Photos/igphoto/2000836396/). Not an afterburner operating limit or typical combat burst. Ignores reserves and variation of fuel flow with flight conditions. |
| Segway MAX G2 D | 30.6 min nominal-energy equivalent | [Published 551 Wh battery](https://se-en.segway.com/ekickscooter/products/max-g2-d.html) / chart's assumed 1,080 W input × 60. Does not establish that the motor/controller/battery can maintain that peak for 30 min. |
| Tesla Model 3 Performance | Exact peak hold not established | [Tesla's 2024 launch](https://www.youtube.com/watch?v=krQKnhMwxn4) gives a 2.9 s 0–60 mph event and distinguishes peak from continuous power. Acceleration time is not duration at 510 hp; the car is not at maximum power throughout launch. Do not assign a duration to the chart's inferred 423 kW input from that number. |
| RC car (X-Maxx) | Not established | The plotted 100 A is an engineering assumption. Pack capacity and a logged voltage/current/temperature trace are needed; arbitrary capacity divided by assumed current would only produce a nominal budget. |
| Compute systems | Continuous-operation regime; exact peak windows generally unspecified | [NVIDIA DGX B300 documentation](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html) describes continuous operation. This does not validate indefinite operation at the inherited 19 kW estimate, which already has a documented source mismatch. Facility-fed compute has no onboard energy-exhaustion duration comparable to a rocket. |

The remaining animal and spacecraft endpoints need source-by-source assessment.
In particular, sustained spacecraft electrical generation, launch-era RTG
output, metabolic maxima, and short feeding events should not inherit one
duration just because the plot labels all of them “peak.”

## Proposed battery and SOFC benchmarks

- **Molicel P60C: 2 s pulse.** The manufacturer's [Chinese product page](https://www.molicel.com/cn/inr-21700-p60c/)
  explicitly claims 660 W for 2 s. With the [tentative datasheet's](https://www.molicel.com/wp-content/uploads/Product-Data-Sheet-of-INR-21700-P60C.pdf)
  75 g maximum cell mass, this is 8.8 kW/kg. The 100 A discharge rating is a
  separate specification with temperature/voltage cutoffs; the pulse claim
  lacks a complete public test protocol. Nominal 21.6 Wh / 660 W = 118 s is
  **not** a permissible extension of the two-second claim.
- **Metal-foam SOFC: peak-hold duration not established.** [Ma et al., 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202517694)
  report 6.56 kW/kg at 650°C. Section 4.3 describes swept V–I characterization;
  the main article does not demonstrate a sustained hold at that peak.
  Cell mass excludes fuel, tanks, insulation and balance of plant. A
  continuous-duty system benchmark would require a different, system-level
  source and cannot inherit the laboratory cell's specific power.

These are contemporary high-performance examples, not universal SOTA claims
across every chemistry, duration and mass boundary.

## Recommendation

There is no defensible pooled “typical peak duration” across this plot. For
fuel-plus-carried-oxidizer guides, the rocket ascent regime is the closest
physical match. **Use 150 s if the chart needs one stated reference duration**,
with 300 s as a sensitivity case. The 150 s choice is a rounded scenario
anchored to the surveyed booster intervals, not an intrinsic fuel rating or
an assertion that every object sustains its plotted peak that long.

For ideal stoichiometric mixtures, using rounded atomic masses:

```text
mixture specific energy = fuel LHV / (1 + oxygen/fuel mass ratio)
average chemical power per propellant mass = mixture specific energy / duration

H2 + O2: 120 MJ/kg / (1 + 8) = 13.33 MJ/kg mixture
CH4 + O2: 50 MJ/kg / (1 + 4) = 10.00 MJ/kg mixture
```

LHV references: [NASA/TM-2006-214328](https://ntrs.nasa.gov/api/citations/20060021979/downloads/20060021979.pdf)
and [Boeing Cascade methane model](https://docs.cascade.boeing.com/docs/energy/methane.html).
These stoichiometric ratios differ from the fuel-rich operating mixtures used
to estimate existing rocket input power.

| Discharge scenario | H2 + O2 | CH4 + O2 |
| --- | ---: | ---: |
| 150 s, booster-scale reference | 88.9 kW/kg | 66.7 kW/kg |
| 300 s, five-minute sensitivity | 44.4 kW/kg | 33.3 kW/kg |
| 1,800 s, longer-duration sensitivity | 7.41 kW/kg | 5.56 kW/kg |

The active v2 figure uses the **300 s column** for its two dashed fuel guides:
H2 + O2 at 44.4 kW/kg and CH4 + O2 at 33.3 kW/kg of total propellant mass.
This is the five-minute sensitivity case rather than the recommended 150 s
booster-ascent reference, so the lines represent a longer burn than the
surveyed rocket intervals.

Suggested line label: “CH4 + LOX, chemical energy / 150 s, propellant only.”
If duration sensitivity should appear in the figure, a 150–300 s band is more
honest than an unlabeled fuel W/kg line, although four extra boundary lines
would add clutter.

The whole-system relation is `P_average / M_system = f_consumed × e_mix / t`,
where `f_consumed` is consumed propellant mass divided by system reference mass.
An unadjusted propellant line implicitly sets that fraction to one and excludes
tanks/engines/payload. Aircraft and animals obtain oxidizer from air; the
carried-oxidizer boundary does not describe their onboard energy inventory.
Chemical input and battery/SOFC electrical output also remain different power
boundaries. Keep duration and boundary labels on each benchmark independently.
