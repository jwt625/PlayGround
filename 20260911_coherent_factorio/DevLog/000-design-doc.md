# Project PHIELDWORKS

> Design status: this document establishes the broad vision. See [001 — Gameplay decisions and frontier experiment](001-gameplay-and-frontier-experiment.md) for the agreed audience, rewards, commissioning loop, initial scope, and clarifications that take precedence where the documents differ.

## 1. Concept

### 1.1 High-level premise

**Project PHIELDWORKS** is a factory-building automation game about industrializing a hostile planet whose physical environment is dominated by wave phenomena.

At the surface level, the player experiences a familiar progression:

* extract raw materials,
* refine them,
* manufacture increasingly complex components,
* build power and logistics infrastructure,
* automate production,
* defend or manage territory,
* scale toward a planet-spanning industrial project.

The game does **not** initially present itself as an optics, photonics, or RF game.

There are no early-game buildings named:

* laser,
* interferometer,
* waveguide,
* coherent receiver,
* optical isolator,
* phased array.

Instead, the player builds industrial components such as:

* copper assemblies,
* ceramic substrates,
* precision glass,
* signal modules,
* emitters,
* sensor heads,
* amplifiers,
* resonant modules,
* precision oscillators,
* patterned surfaces,
* semiconductor dies,
* control boards,
* advanced packaging.

Their behavior happens to obey wave physics.

The player gradually discovers that the factory is not merely a network of machines connected by logistics. It is also a **coupled physical field system**.

The intended conceptual progression is:

> Things move through connections.

then:

> Connections alter signals.

then:

> Signals interact through multiple paths.

then:

> The geometry of the whole factory matters.

then:

> Coherence is what makes distant parts of the factory behave as one machine.

Late in the game, the player finally gains enough scientific understanding to intentionally exploit this fact.

The endgame is the construction of a **planet-scale beam-driven lightsail launch system**, requiring both:

1. an enormous industrial and nanofabrication base capable of manufacturing an ultralight sail, and
2. a huge high-power transmitting aperture whose many emitters must behave as a controlled coherent field.

The final achievement is therefore not simply “build a rocket.”

It is:

> **Turn the industrialized planet into a single precision instrument.**

---

# 2. Design principles

## 2.1 Wave physics should be gameplay, not terminology

The player should usually encounter the **effect before learning the name**.

For example:

* a signal unexpectedly disappears after two paths recombine;
* moving a machine changes throughput elsewhere;
* a receiver works only in certain locations;
* a high-quality source unexpectedly makes a previously stable factory unstable;
* a component starts feeding energy backward into its upstream network;
* a pulse stream becomes unreliable over a long route;
* leaked energy causes native organisms to colonize previously empty terrain.

Only much later may the technology tree expose concepts such as phase coherence, modal dispersion, reflections, spectral purity, and wavefront control.

---

## 2.2 Conservation laws should remain physically legible

The game should avoid the misleading intuition that destructive interference destroys energy.

When two coherent paths destructively interfere at one output, the energy must appear elsewhere.

Depending on topology, it may be:

* redirected into another output port,
* reflected backward,
* coupled into an unwanted spatial mode,
* radiated into the environment,
* dissipated in lossy structures.

A two-port combining structure therefore behaves more like:

```text
                   useful output
                       ▲
                       │
input A ───────┐   ┌───┴───┐
               ├───┤ network│
input B ───────┘   └───┬───┘
                       │
                       ▼
                rejected / return port
```

If phase causes the useful output to fall from 100% to 5%, the remaining 95% should visibly appear in one or more other channels.

This is both more correct and more interesting.

“Bad interference” therefore creates secondary consequences:

* reflected power,
* stray fields,
* heating,
* unwanted coupling,
* ecological attraction,
* instability elsewhere.

---

## 2.3 Better technology should introduce new failure modes

Technology should not monotonically remove complexity.

An improved component might provide:

* more power,
* narrower spectrum,
* longer range,
* higher signal density,
* lower intrinsic loss,

while simultaneously causing much stronger coherent interactions.

An especially important inversion is:

> **Higher coherence can initially make the factory worse.**

The player may eventually unlock sources or architectures with deliberately reduced temporal or spatial coherence.

At that point, sections of the factory become wonderfully simple again:

* powers approximately add,
* multipath interference averages away,
* alignment tolerances loosen,
* stray paths matter less.

The ironic late-game realization is that after mastering coherence, the player sometimes chooses **incoherence** because it is operationally superior.

This produces a satisfying return toward the simplicity of original Factorio.

---

## 2.4 Every nuisance should eventually become a tool

| Early failure                      | Later understanding | Late-game exploitation  |
| ---------------------------------- | ------------------- | ----------------------- |
| path-dependent power fluctuations  | interference        | controlled combining    |
| backflow                           | reflection          | resonant enhancement    |
| unwanted resonances                | feedback paths      | filtering / oscillation |
| signal spreading                   | dispersion          | pulse shaping           |
| path-dependent wireless dead zones | multipath           | spatial diversity       |
| unwanted mode content              | modal propagation   | mode multiplexing       |
| beam spreading                     | diffraction         | focusing / beam shaping |
| thermal sensitivity                | phase drift         | precision sensing       |
| leakage                            | stray field         | distributed sensing     |
| spectral crowding                  | finite bandwidth    | wavelength multiplexing |

Physics should never feel like a sequence of arbitrary debuffs.

---

# 3. World and narrative

## 3.1 Setting

The player arrives on a resource-rich extrasolar world with the objective of developing enough local industry to launch an interstellar probe.

The original mission assumes conventional propulsion will suffice.

It eventually becomes clear that the available launch mass, propellant, and planetary gravity make a purely chemical or nuclear spacecraft impractical for the desired interstellar velocity.

The industrial objective evolves toward a beam-driven lightsail.

The world contains unusual native organisms and mineral structures that interact strongly with electromagnetic radiation.

They are not necessarily conventionally intelligent or hostile.

Their behavior responds to the fields created by the player's industrial system.

---

# 4. Native ecology as physical failure modes

Instead of generic enemies whose main relationship with the player is pollution-triggered aggression, native organisms embody different engineering hazards.

They transform abstract electromagnetic problems into visible, spatial gameplay.

## 4.1 Straylight feeders

These organisms metabolize weak diffuse radiation.

They grow preferentially around:

* leaking transport channels,
* badly matched components,
* poorly shielded production buildings,
* imperfect combining structures,
* discarded energy ports.

They are initially harmless.

But colonies spread across usable terrain and physically block construction.

A poorly designed factory therefore gradually loses real estate.

### Gameplay effect

An inefficient factory may still meet throughput targets but becomes surrounded by biological growth.

The player can:

* clear colonies mechanically,
* improve shielding,
* redirect leakage,
* deliberately create sacrificial illumination zones,
* later harvest these organisms as specialized materials.

Thus stray light becomes analogous to Factorio pollution, but spatially tied to actual field leakage.

---

## 4.2 Source feeders

A second organism is attracted to intense emitter structures.

It preferentially grows toward high-field machinery and may:

* coat emitter windows,
* change local refractive properties,
* cause thermal loading,
* scatter part of the output,
* introduce slowly varying feedback.

It behaves almost like biological fouling.

High-power infrastructure therefore requires perimeter engineering.

---

## 4.3 Detector feeders

Another species is attracted to absorbing structures or the electrical/thermal signatures produced by sensor machinery.

They colonize around:

* monitoring heads,
* receivers,
* measurement stations,
* precision metrology.

They can:

* obscure sensors,
* create additional scattering,
* introduce offsets or false readings,
* consume weak signals before detection.

This makes measurement infrastructure itself something that must be protected.

---

## 4.4 Reflective or metallic swarms

Mobile creatures contain reflective or conductive structures.

When they cross a high-frequency communication corridor, they alter the multipath environment.

The result can be:

* fading,
* transient interference,
* fluctuating link quality,
* false sensor readings.

The threat is not direct damage.

The threat is that **the propagation environment moves**.

Later, this creates demand for:

* diversity,
* adaptive routing,
* robust modulation,
* lower coherence options,
* active beam tracking.

---

## 4.5 Resonant flora

Some native structures strongly interact with narrow spectral bands.

Early in the game they appear unpredictable:

* one production line works,
* an otherwise identical line nearby performs badly.

Later the player learns that local biological structures contain narrow resonances.

Eventually these structures may be exploited as:

* environmental sensors,
* frequency references,
* biological filters,
* specialty materials.

---

# 5. Core factory model

The game has two overlapping networks.

## 5.1 Material network

Conventional Factorio-like logistics:

* belts,
* inserters,
* trains,
* pipes,
* warehouses,
* robots.

Items remain discrete conserved objects.

---

## 5.2 Signal/energy network

Many industrial machines also transmit energy or information.

Initially this looks similar to ordinary electrical connectivity.

Internally, however, connections can carry additional hidden state:

$$
\mathcal S =
\{P,\ \omega,\ \Delta\omega,\ \phi,\ \tau,\ m,\ p,\ \mathbf{k},\ldots\}
$$

Potential state includes:

* power,
* center frequency,
* bandwidth,
* phase,
* delay,
* spatial mode,
* polarization,
* direction,
* coherence state.

The simulation should only track the subset required by the currently relevant technology.

The player therefore does not confront all degrees of freedom at once.

---

# 6. Coherence mechanics

## 6.1 What coherence changes

Without useful mutual coherence, two paths approximately behave as scalar power flows:

$$
P_{\mathrm{out}}\approx P_1+P_2.
$$

With coherence, the relative field becomes important.

The practical gameplay consequence is:

> **The history of a signal matters when it meets another signal derived from the same origin.**

Two routes carrying nominally identical power are no longer interchangeable.

---

## 6.2 Conservation-aware combiners

A general two-input / two-output component should redistribute energy according to phase.

Conceptually:

```text
       Path A ────────┐
                      │
                      ├── useful route
                      │
       Path B ────────┤
                      │
                      └── return / reject route
```

For one phase relation:

```text
useful output: 98%
reject output: 2%
```

For another:

```text
useful output: 4%
reject output: 96%
```

Nothing disappears.

This allows bad phase control to cause problems somewhere else.

A player may fix one output only to discover they have accidentally dumped huge power into a return path.

---

# 7. Backreflection

Interfaces are imperfect.

Whenever a signal encounters:

* connectors,
* material boundaries,
* discontinuities,
* damaged infrastructure,
* badly tuned modules,

some fraction may propagate backward.

Backreflection can cause:

* reduced delivered power,
* standing field patterns,
* unstable source machinery,
* excess energy elsewhere,
* narrow resonances,
* unexpected interactions with earlier branches.

Early players encounter this as apparently mysterious **upstream contamination**.

Later equipment allows diagnosis through forward/return measurements.

Solutions can include:

* improved termination,
* angled interfaces,
* impedance transitions,
* directional structures,
* absorptive dumps,
* nonreciprocal components,
* topology changes.

These should have industrial names rather than optical textbook names until late game.

---

# 8. Multipath and stray propagation

## 8.1 Hidden connections

One defining feature of the game should be:

> Not every physical interaction requires an explicit player-built connection.

Signals can leak.

Leaked energy can:

* scatter from structures,
* reflect from terrain,
* couple into neighboring transport channels,
* propagate through free space,
* return through unintended paths.

This creates an invisible secondary topology layered over the explicit factory topology.

---

## 8.2 Gameplay example

A production sector is stable.

The player adds a large metal processing tower nearby.

A remote precision line begins oscillating between high and low throughput.

Nothing is visibly connected.

The tower has created an additional return path.

The player eventually uses field diagnostics to discover:

```text
primary route                  -2 dB
secondary path                -31 dB
secondary delay               18.4 ns
mutual coherence              high
```

Despite being weak, the secondary path substantially modifies local operation.

---

# 9. Temporal coherence as a design variable

Industrial sources should differ in spectral purity without initially calling this "laser linewidth."

A source specification might show:

```text
Power                    30 units
Stability                High
Spectral concentration   Very High
Temporal memory          14 km equivalent
```

As technology improves, temporal memory increases.

This produces longer-range self-interference.

A seemingly premium source can therefore create entirely new problems.

---

## 9.1 Deliberate low-coherence technology

Later, the player gains components that deliberately broaden or decorrelate the field.

Benefits:

* reflected paths cease interfering after modest delays,
* multipath averages out,
* some combining operations become power-additive,
* environmental sensitivity drops.

Costs:

* reduced ability to focus or coherently combine,
* lower spectral density,
* poorer resonant enhancement,
* reduced range or precision in some applications.

A late-game industrial factory may therefore contain both:

### Highly coherent domains

Used for:

* precision fabrication,
* spectroscopy,
* final beam combining,
* long-baseline sensing.

### Low-coherence domains

Used for:

* rugged bulk power transport,
* illumination,
* pumping,
* local communications,
* systems where stability matters more than coherent gain.

This creates a satisfying return to "ordinary Factorio" behavior in carefully chosen regions.

---

# 10. Spatial coherence

Temporal coherence determines whether delayed paths interfere.

Spatial coherence determines whether different portions of an aperture behave as one field.

This distinction becomes central to the endgame.

Bulk power sources can provide enormous energy while having poor spatial coherence.

They are useful for:

* heating,
* pumping,
* processing,
* industrial illumination.

But the final lightsail system requires a huge effective aperture.

Therefore the endgame needs a hierarchy:

```text
bulk electrical power
        ↓
rugged high-power emitters
        ↓
pump infrastructure
        ↓
precision conversion stage
        ↓
phase-controlled aperture
        ↓
lightsail
```

This gives "crappy lasers" or equivalent industrial emitters an important role without naming them explicitly during most of the progression.

High-power incoherent sources can pump much smaller quantities of highly ordered coherent output.

---

# 11. Dispersion

## 11.1 Chromatic dispersion

High-speed signals contain finite spectral width.

Different spectral components propagate differently.

At sufficiently large:

$$
\text{bandwidth}\times \text{distance},
$$

symbols begin overlapping.

The game should communicate this through performance rather than formulas.

A short connection supports:

```text
throughput: 100%
error rate: negligible
```

The same technology used over a longer route may become:

```text
throughput: 67%
retries: high
```

---

## 11.2 Mitigations

Players gradually encounter:

* lower symbol rate,
* parallel channels,
* improved transmission media,
* compensation modules,
* equalizers,
* wavelength selection,
* coherent detection,
* DSP.

Each mitigation has costs in:

* power,
* area,
* materials,
* latency,
* complexity.

---

# 12. Modal dispersion

Early high-bandwidth transport uses large-core channels because alignment is forgiving.

These support many propagation modes.

Signals launched into them divide among several paths with different delays.

This gives the player a natural trade:

### Large transport channel

* easy alignment,
* inexpensive connectors,
* rugged,
* high coupling efficiency,
* poor long-distance signal integrity.

### Restricted-mode transport

* difficult alignment,
* expensive components,
* lower tolerance,
* vastly better reach.

Later:

* graded structures,
* mode filters,
* adaptive equalization,
* deliberate mode multiplexing

allow the nuisance to become capacity.

---

# 13. Diffraction and free-space propagation

Free-space transport gradually enters the game.

Beams are never perfect geometric rays.

A finite aperture causes spreading.

This introduces a simple relationship:

> Large aperture → low divergence
> Small aperture → high divergence

No equation need initially appear.

The player learns by placing components.

Small transmitter:

```text
near receiver:  excellent
far receiver:   terrible
```

Larger transmitter aperture:

```text
near receiver:  excellent
far receiver:   good
```

Precision focusing introduces alignment sensitivity.

---

# 14. Patterned and diffractive surfaces

Advanced manufacturing allows the production of patterned surfaces.

Again, they need not initially be called diffraction gratings or metasurfaces.

Examples:

* wavelength sorting plate,
* beam shaping tile,
* focusing membrane,
* spectral separator,
* steering surface.

Their behavior can be visualized directly:

```text
mixed input
    ↓

 ////////

↙   ↓   ↘   → 
different spectral bands
```

These components provide spatially satisfying alternatives to abstract "routing boxes."

---

# 15. Industrial production tree

The material system should remain recognizably industrial.

Representative chain:

```text
ore
↓
metals
↓
copper / aluminum / steel
↓
wire + precision metalwork
↓
ceramic substrates
↓
glass and crystalline materials
↓
basic circuit boards
↓
semiconductor components
↓
precision assemblies
↓
advanced packaged modules
↓
patterned semiconductor dies
↓
high-purity crystals
↓
nanofabricated structures
↓
precision field-control systems
```

The player therefore experiences a conventional industrial revolution whose products progressively become wave-sensitive.

---

# 16. Progression

Progression should be experiential rather than divided into obvious "RF age," "laser age," etc.

## Stage 1 — Industrial survival

Main concerns:

* mining,
* smelting,
* mechanical automation,
* steam/electricity,
* belts,
* basic wiring,
* construction.

Signals behave essentially ideally.

The game feels close to Factorio.

---

## Stage 2 — Instrumented automation

Players manufacture:

* control boards,
* sensors,
* communication modules,
* precision motors,
* higher quality electrical components.

Long-distance and high-speed signals begin experiencing:

* attenuation,
* propagation delay,
* crude reflections.

Most effects remain mild.

---

## Stage 3 — High-frequency industry

Factories demand greater information throughput and tighter timing.

Players deploy increasingly advanced signal infrastructure.

Now layout starts mattering.

Symptoms appear:

* some routes cannot sustain rated throughput,
* sharp discontinuities create return energy,
* nearby infrastructure affects communication reliability,
* long cables behave differently from short ones.

The player is still thinking primarily in electrical-industrial terms.

---

## Stage 4 — High-density transport

The demand for bandwidth and precision outgrows conventional wiring.

New materials allow dense signal transport through:

* transparent media,
* crystalline channels,
* patterned substrates,
* free-space modules.

These are framed as advanced industrial interconnects.

At first they appear miraculous:

* huge bandwidth,
* low propagation loss,
* lightweight cabling.

Then new problems emerge:

* alignment,
* mode content,
* spectral dependence,
* dispersion.

---

## Stage 5 — Precision sources

Advanced semiconductor/crystal production creates sources with dramatically improved:

* spectral concentration,
* timing stability,
* directionality.

The player sees large performance gains.

Then previously irrelevant weak reflections become important.

The factory begins experiencing:

* narrow operating windows,
* unexpected resonances,
* large layout sensitivity,
* path-dependent power redistribution.

This is the game's major coherence transition.

The player should not immediately be told:

> You have unlocked coherence.

Instead they should think:

> Why did upgrading my signal source break half my factory?

---

## Stage 6 — Controlled field infrastructure

The player develops improved instrumentation.

They can now see:

* forward and backward energy,
* phase relationships,
* delay,
* spectral structure,
* mode composition.

New control devices allow:

* adjustable delays,
* tunable phase,
* adaptive matching,
* spectral filtering,
* automated calibration.

Coherence becomes something they can engineer.

---

## Stage 7 — Industrial-scale precision manufacturing

The factory now supports:

* extreme-purity materials,
* semiconductor fabs,
* lithography,
* thin-film deposition,
* precision etching,
* atomic-scale patterning.

This is required for both:

1. the final lightsail membrane,
2. the extremely dense field-control hardware needed for the transmitter.

Nanofab becomes a major production branch rather than just a research unlock.

---

# 17. Nanofabrication progression

A plausible industrial progression:

```text
machined structures
↓
printed electronics
↓
coarse semiconductor fabrication
↓
UV lithography
↓
deep-UV lithography
↓
advanced multilayer optics
↓
short-wavelength lithography
↓
extreme-precision patterning
```

The fab itself creates new wave problems.

Lithography systems require:

* wavelength stability,
* beam uniformity,
* vibration control,
* thermal control,
* contamination management,
* precision alignment.

Thus the player has to build increasingly sophisticated wave-processing systems to manufacture the devices needed to build even more sophisticated wave-processing systems.

This recursive industrial structure is highly Factorio-compatible.

---

# 18. Late-game low-coherence renaissance

After spending much of the game learning to manage coherence, the player unlocks intentionally decorrelated emitters and transport architectures.

A region previously requiring:

* calibration,
* phase stabilization,
* reflection management

may suddenly operate approximately as:

```text
input power A
+
input power B
=
output power
```

This should feel liberating.

A late-game player may deliberately redesign large portions of the factory to use these robust sources.

This mirrors real engineering:

> Don't preserve phase if you don't need phase.

Only the small fraction of infrastructure that benefits from coherence remains highly controlled.

---

# 19. Endgame architecture

The final objective is a beam-driven interstellar lightsail.

It requires two enormous industrial programs running in parallel.

---

## 19.1 Lightsail manufacturing

The sail demands:

* extreme area,
* extremely low areal density,
* high reflectivity,
* thermal stability,
* controlled spectral response,
* mechanical strength,
* precision patterning.

Production may involve:

* semiconductor-grade substrates,
* sacrificial layers,
* thin-film deposition,
* patterned dielectric stacks,
* nanostructured surfaces,
* kilometer-scale folding and deployment systems.

Manufacturing yield becomes a late-game challenge.

---

## 19.2 Beam infrastructure

The player must construct a huge transmitting aperture.

Raw input power can come from relatively crude high-power sources.

These may have:

* poor spatial coherence,
* mediocre spectral quality,
* modest efficiency.

They pump or power a hierarchy of more precise field-generating modules.

The final aperture consists of many independently controlled sectors.

Each sector must satisfy:

* adequate power,
* spectral compatibility,
* phase control,
* pointing accuracy,
* thermal stability,
* low parasitic feedback.

---

# 20. Final beam combining

The final aperture is the first point where the game explicitly reveals the physical interpretation of many earlier mechanics.

The player learns that their "precision field emitters" must operate as one coherent aperture.

For \(N\) correctly phased emitters, the total radiated power remains proportional to \(N\).

However, coherent phasing redistributes that power into a narrow angular region, causing peak intensity at the target to scale approximately as:

$$
I_{\text{target}}\propto N^2
$$

for identical elements when comparing field addition at the desired point.

The game should explicitly maintain conservation:

* the total integrated radiated power does not become \(N^2\),
* coherent control changes **where the power goes**.

Poor phase control produces:

* broadened beams,
* sidelobes,
* stray illumination,
* power incident on the wrong terrain,
* enormous ecological responses.

Thus the final challenge is not merely producing enough power.

It is directing it into the correct spatial mode.

---

# 21. Lightsail launch sequence

The final launch should exercise nearly every system.

Requirements could include:

| System                 | Requirement                             |
| ---------------------- | --------------------------------------- |
| Power generation       | sustained multi-stage energy production |
| Cooling                | stable aperture temperature             |
| Bulk emitters          | sufficient pump power                   |
| Precision sources      | spectral compatibility                  |
| Reference network      | timing and phase distribution           |
| Aperture control       | wavefront alignment                     |
| Metrology              | continuous target tracking              |
| Atmospheric correction | adaptive compensation                   |
| Straylight control     | prevent ecological overgrowth           |
| Sail fab               | sufficient yield and area               |
| Launch infrastructure  | deploy sail accurately                  |
| Automation             | maintain lock autonomously              |

During launch:

```text
T - 60 s
Reference network stable

T - 45 s
Aperture sectors synchronized

T - 30 s
Thermal gradients within tolerance

T - 20 s
Sail deployment confirmed

T - 10 s
Target lock acquired

T - 5 s
Adaptive correction converged

3
2
1

MAIN APERTURE ENABLED
```

The visible effect should be enormous.

The planet-wide factory momentarily becomes one coordinated machine.

---

# 22. UI and UX

## 22.1 Default view

The default view remains primarily industrial.

The player sees:

* machines,
* belts,
* pipes,
* cables,
* buildings,
* native organisms,
* vehicles.

Wave phenomena should not constantly overwhelm the screen.

---

# 23. Field view

A dedicated toggle reveals the hidden physical system.

This is one of the game's central UI elements.

Possible modes:

### Intensity view

Shows where energy is concentrated.

```text
░░▒▓████▓▒░
```

---

### Propagation view

Shows dominant flow directions.

```text
→ → → ↗ ↑
← ← ↙ ↓
```

---

### Return-energy view

Highlights reflected/backward propagation.

Useful for finding bad interfaces.

---

### Phase view

Shows relative phase through hue, arrows, or animated texture.

It should not imply absolute optical phase is directly meaningful.

The view is relative to a selected reference.

---

### Coherence view

Shows where two signals remain mutually coherent.

For example:

```text
bright connection    strongly coherent
faded connection     partially coherent
gray                  effectively incoherent
```

This could be one of the most educational visualizations in the game.

---

### Delay/path view

Selecting a receiver shows major contributing paths.

```text
Path 1    72%
Path 2    19%
Path 3     6%
other      3%
```

Each can be highlighted spatially.

---

### Spectrum view

Shows occupancy in different frequency bands.

Initially abstract.

Later more detailed.

---

### Mode view

Displays power in:

* desired mode,
* higher-order modes,
* radiation modes.

---

# 24. Explain consequences before equations

A tooltip should prioritize:

```text
Delivered power:       62%
Returned power:        31%
Radiated / stray:       5%
Absorbed:               2%
```

rather than:

```text
S11 = ...
S21 = ...
```

Advanced tools can later expose the formal quantities.

The player should be able to learn the physics intuitively before learning the notation.

---

# 25. Instrumentation progression

Measurement capability should itself be technological progression.

Early:

```text
Power monitor
Input: 14 units
```

Later:

```text
Directional monitor

Forward power     14.2
Return power       2.8
```

Later:

```text
Spectral monitor

Band 1            8.2
Band 2            3.1
Band 3            2.4
```

Later:

```text
Field analyzer

Amplitude
Relative phase
Delay
Mode distribution
Mutual coherence
```

The player's ability to debug grows alongside the complexity of the system.

---

# 26. Failure diagnostics

Failures should initially be symptom-based rather than naming the physical mechanism.

Example:

```text
Precision Assembly Line

Input power            nominal
Control link           unstable
Processing yield       43%
```

Possible causes:

* reflected power,
* multipath,
* dispersion,
* modal content,
* drift,
* spectral mismatch,
* biological scattering.

The player progressively gains instruments that isolate the cause.

This makes debugging itself part of gameplay.

---

# 27. Thermal coupling

Wave systems should naturally interact with the factory's thermal system.

Absorbed power produces heat.

Temperature changes:

* dimensions,
* material properties,
* resonant conditions,
* delays.

This forms feedback:

```text
more stray power
     ↓
more absorption
     ↓
higher temperature
     ↓
phase changes
     ↓
more stray power
```

Thermal runaway becomes possible.

Later stabilization systems include:

* active cooling,
* temperature control,
* phase feedback,
* thermal isolation.

---

# 28. Mechanical coupling

High-precision infrastructure should eventually become vibration-sensitive.

Nearby:

* trains,
* crushers,
* heavy mining equipment,
* rocket launches

can perturb precision systems.

The player now faces factory zoning problems.

A high-throughput industrial district may be a terrible place for precision fabrication.

This encourages geographically differentiated regions:

* heavy industry,
* nanofab,
* precision field control,
* power generation,
* final aperture.

---

# 29. Strategic factory geography

Unlike Factorio, where compact layouts are frequently desirable, this game should create tension between compactness and isolation.

Compact systems:

* use less material,
* have shorter delays,
* are easier to control,

but:

* suffer stronger crosstalk,
* stray coupling,
* thermal interaction,
* ecological attraction.

Separated systems:

* reduce interference,

but:

* require longer transport,
* accumulate dispersion,
* require synchronization,
* consume more real estate.

There should be no universally optimal topology.

---

# 30. Science progression

Research packs should remain industrial rather than explicitly optical.

Possible families:

* mechanical science,
* electrical science,
* materials science,
* precision science,
* semiconductor science,
* field science,
* fabrication science,
* planetary engineering science.

Only late research reveals terminology closely corresponding to real photonics.

The reveal should feel like:

> These weird rules governing my factory are all manifestations of one underlying wave model.

---

# 31. Implementation architecture

## 31.1 Core design requirement

Do not run full electromagnetic simulations over the entire map.

Instead represent the factory as a hierarchical network model.

---

# 32. Simulation level 0: scalar power flow

Used when signals are effectively incoherent.

Each edge carries:

```text
power
band
direction
```

Components implement simple:

* loss,
* splitting,
* combining,
* conversion.

This should cover much of the early game and large sections of the late game.

Computational cost is low.

---

# 33. Simulation level 1: complex phasor network

For coherent narrowband systems, edges carry:

$$
A e^{i\phi}.
$$

Components are represented using small scattering matrices.

For example:

$$
\mathbf b = S\mathbf a.
$$

This naturally preserves power when \(S\) is unitary for lossless components.

Loss can be represented through coupling into:

* absorptive channels,
* radiation channels,
* explicit environment channels.

This is ideal for:

* interference,
* reflections,
* resonances,
* coherent combining.

---

# 34. Simulation level 2: multiple spectral channels

For frequency-multiplexed systems:

$$
A_k e^{i\phi_k}
$$

is tracked for a small set of spectral bins.

This enables:

* wavelength-dependent routing,
* filtering,
* chromatic dispersion,
* spectral crosstalk.

The engine should avoid resolving every physical wavelength.

Only active bands need representation.

---

# 35. Simulation level 3: envelope / delay model

High-speed communications can be represented using:

* symbol envelopes,
* impulse responses,
* aggregate eye-opening metrics,
* group delay spread.

The engine need not simulate every bit.

A link can maintain effective parameters such as:

```text
signal bandwidth
path loss
group delay
delay spread
noise
interference
```

From these, the game estimates:

* throughput,
* error probability,
* retry rate.

---

# 36. Simulation level 4: modal model

Selected transport channels may carry several modes.

Represent:

$$
\mathbf a =
[a_0,a_1,a_2,\ldots].
$$

Bends, interfaces, and scatterers apply mode-coupling matrices.

This supports:

* modal dispersion,
* mode conversion,
* intentional mode multiplexing.

Again, only a small number of aggregate modes need simulation.

---

# 37. Simulation level 5: local spatial field

True spatial wave simulation should be restricted to special structures:

* free-space apertures,
* diffractive devices,
* final phased array,
* selected precision machinery.

Use approximations such as:

* Gaussian beams,
* Fourier propagation,
* antenna-array factors,
* geometric diffraction envelopes.

Do not solve FDTD over factory-sized domains.

---

# 38. Coherence groups

A crucial optimization is the concept of a **coherence group**.

Signals only require phasor-level interaction when they share sufficient mutual coherence.

Each source generates a coherence identity.

Splitting preserves identity.

Processes may:

* preserve identity,
* partially decorrelate it,
* destroy it.

If two signals are mutually incoherent, the engine can collapse their interaction to scalar powers.

This dramatically reduces computation.

It also directly matches gameplay.

---

# 39. Temporal coherence approximation

Instead of storing full stochastic phase noise histories, each path can track:

```text
source ID
delay
effective linewidth / coherence time
```

Two paths with delay difference:

$$
|\Delta\tau|
$$

much greater than coherence time can be treated as incoherent.

Intermediate regimes can use a scalar degree of coherence:

$$
0 \le |\gamma| \le 1.
$$

This gives partial interference without expensive time-domain simulation.

---

# 40. Spatial coherence approximation

For extended sources, track coherence across aperture sectors.

A crude source may consist of many independent spatial coherence cells.

A precision source may have one large coherent field.

This allows:

* incoherent illumination,
* partially coherent pumping,
* coherent final beamforming.

---

# 41. Environmental field simulation

Stray power does not require detailed ray tracing.

Each leaking component can emit into a coarse environmental propagation model.

Possible approaches:

* low-resolution field grid,
* sparse visibility graph,
* limited reflection paths,
* precomputed terrain response,
* stochastic scatter approximation.

Environmental organisms respond to integrated exposure over time rather than instantaneous field oscillations.

This keeps ecological simulation tractable.

---

# 42. Reflection/path pruning

Multipath can explode combinatorially.

Only paths above a relevance threshold should be retained.

For each receiver:

```text
keep:
largest K coherent contributors
largest K incoherent contributors

aggregate:
all weaker paths into background
```

The UI can therefore show the few dominant causes rather than millions of microscopic paths.

This is simultaneously better engineering abstraction and better gameplay.

---

# 43. Network partitioning

Factories naturally divide into weakly coupled regions.

The engine can identify:

* strongly connected coherent networks,
* ordinary scalar networks,
* environmental coupling regions.

Only coherent strongly connected components require matrix solving.

Most of the map should remain computationally cheap.

---

# 44. Steady-state solving

For narrowband networks with reflections, propagation becomes a linear network problem.

Rather than repeatedly propagating rays, solve:

$$
\mathbf x = A\mathbf x + \mathbf s
$$

or:

$$
(I-A)\mathbf x=\mathbf s.
$$

Sparse linear methods can efficiently solve large coherent subnetworks.

This supports resonances and feedback naturally.

---

# 45. Dynamic simulation

Slow variables evolve separately:

* temperature,
* mechanical position,
* ecological growth,
* control-loop states,
* component drift.

The fast electromagnetic solution is updated only when these variables change enough to matter.

This creates realistic hierarchy:

```text
wave state             fast / quasi-steady
control systems        medium
thermal state          slow
ecological growth      very slow
industrial logistics   game-time dependent
```

---

# 46. Control systems

The player should eventually be able to automate tuning.

Basic control block:

```text
monitor
  ↓
controller
  ↓
adjustable component
```

Available operations may include:

* maximize transmitted power,
* minimize return power,
* maintain a target phase,
* track resonance,
* balance multiple outputs.

Advanced systems can have:

* limited bandwidth,
* sensor noise,
* actuator range,
* control-loop instability.

But the player should not have to implement literal PID equations unless desired.

---

# 47. Player-facing abstractions

The game should distinguish between:

### Physical truth

Used internally.

and:

### Engineering abstraction

Presented to the player.

For example, the simulation may use a scattering matrix.

The player sees:

```text
Transmission        83%
Return              11%
Stray                4%
Heat                 2%
```

Only advanced instrumentation reveals detailed field quantities.

---

# 48. Development roadmap

## Prototype A — Coherent transport puzzle

Implement only:

* source,
* paths,
* splitter,
* combiner,
* reflections,
* relative phase,
* energy conservation.

Goal:

Verify that path-dependent behavior is fun.

---

## Prototype B — Field overlay

Add:

* intensity overlay,
* propagation direction,
* return-energy overlay,
* phase overlay.

Goal:

Determine whether players can diagnose failures visually.

---

## Prototype C — Coherence length

Add:

* different source coherence times,
* delayed paths,
* coherent/incoherent transition.

Goal:

Test the central inversion:

> better source → unexpectedly harder system.

---

## Prototype D — Environmental leakage

Add:

* stray radiation,
* simple scattering,
* straylight-feeding organisms.

Goal:

Connect field engineering to map control.

---

## Prototype E — High-speed transport

Add:

* finite bandwidth,
* dispersion,
* multimode delay spread,
* throughput degradation.

Goal:

Test whether transport engineering remains understandable without explicit textbook explanations.

---

## Prototype F — Industrial progression vertical slice

Build a 5–10 hour slice:

```text
ore
→ electronics
→ precision components
→ high-bandwidth transport
→ high-stability source
→ coherence problems
→ field instrumentation
```

This is probably the critical prototype.

If players naturally discover:

> the geometry and history of the network matter,

the game concept works.

---

## Prototype G — Large aperture

Implement a modest 16×16 array.

Allow:

* independently controlled sectors,
* phase error,
* beam steering,
* sidelobes,
* target tracking.

Goal:

Verify that coherent beamforming feels satisfying enough to support the final game.

---

## Prototype H — Lightsail endgame

Scale the array concept to a hierarchical aperture.

The player should not individually tune millions of elements.

Hierarchy:

```text
element
↓
tile
↓
module
↓
sector
↓
array
```

Local controllers solve local alignment.

The player manages system architecture.

This keeps the endgame Factorio-like rather than turning it into a phase-shifter spreadsheet.

---

# 49. Desired player experience

The ideal player does not finish the game thinking:

> I learned a bunch of optical terminology.

They finish thinking:

> I now intuitively understand why coherent systems are weird.

Specifically:

* energy does not simply disappear under destructive interference;
* path history matters;
* weak reflections can strongly affect coherent systems;
* more coherence is not universally better;
* high spatial coherence allows energy to be concentrated into particular modes/directions;
* high temporal coherence makes distant paths interact;
* free-space structures are connected even when no cable exists;
* dispersion converts bandwidth into temporal spreading;
* multimode transport trades coupling tolerance against propagation fidelity;
* measurement and control are integral parts of advanced physical systems;
* large coherent systems require constant stabilization;
* sometimes the best engineering solution is to deliberately destroy coherence.

The game begins with the intuition:

> **A factory is a collection of machines.**

It ends with:

> **A factory can itself be a wave system.**

And the final lightsail launch makes that literal: the entire industrial planet becomes one aperture.
