# Project: Photonic Mode-Overlap Intuition Game

Build a browser-based educational game for photonics engineers/students that develops fast quantitative intuition for optical mode-overlap integrals.

This is NOT primarily a symbolic-calculation game. The core skill being trained is:

> Given two optical modes and their physical geometry, estimate their normalized power coupling / modal overlap quickly and accurately.

The application must generate physically plausible waveguide/fiber cross sections on the fly, solve eigenmodes locally in the browser when practical, compute the correct overlap, and turn the resulting examples into short game questions.

Do NOT spend effort on visual branding or elaborate UI styling. Another specification will define UI aesthetics. Focus here on:
- physics correctness
- scenario taxonomy
- question generation
- solver architecture
- gameplay
- scoring
- progressive difficulty
- performance
- validation

---

# 1. Core learning objective

Train intuition for quantities of the form

    eta = |<mode_B | mode_A>|^2

with physically correct normalization.

For full-vector electromagnetic modes, do not blindly use an unnormalized scalar `∫E1*E2 dA`.

Implement an overlap definition appropriate to power-normalized guided modes using E/H fields and Poynting-flux normalization.

Support a simplified scalar overlap mode for introductory Gaussian exercises, but clearly label it.

Important conceptual distinction:

## Modal overlap is not always identical to actual discontinuity transmission

For:
- fiber butt coupling
- weakly reflecting mode converters
- alignment/mode matching

the forward-mode overlap is often a useful first-order coupling metric.

For:
- abrupt high-index waveguide discontinuities
- severe width changes
- strong impedance discontinuities
- multimode structures

actual transmission can differ because power may:
- reflect backward
- scatter into radiation modes
- couple to higher-order guided modes.

The game should eventually teach this explicitly.

For early questions, choose scenarios in which modal overlap itself is the intended quantity.

For advanced questions, optionally ask:
- overlap into forward fundamental
- overlap distribution among several guided modes
- "is overlap alone enough to predict insertion loss?"
- qualitative reflection/radiation warning

Do not falsely label a single overlap coefficient as exact insertion loss when it is not.

---

# 2. Flagship gameplay: 20-question speedrun

Implement a main game mode:

## Mode Overlap Speedrun — 20 questions

For each question:

1. Generate a scenario.
2. Show:
   - cross-section / geometry A
   - cross-section / geometry B
   - solved mode A
   - solved mode B
   - wavelength
   - polarization/mode labels when appropriate
   - essential geometry dimensions
3. Player predicts overlap η ∈ [0,1].
4. Reveal:
   - true η
   - equivalent mode-mismatch loss `-10 log10(η)` in dB
   - player's error
   - short explanation of the dominant effect
5. Continue immediately.

Track:
- total score
- mean absolute η error
- median η error
- RMS error
- mean dB prediction error
- calibration versus η range
- response time
- category-specific performance

At the end produce a skill profile, e.g.:

    Spot-size mismatch       93
    Lateral alignment        88
    Vertical asymmetry       71
    Polarization             96
    Higher-order parity      54
    Heterogeneous modes      63

Do not make scoring purely binary.

---

# 3. Answer modes

Implement at least three answer modes.

## A. Continuous slider

Player selects η from 0 to 1.

Use nonlinear resolution near η≈1 so values such as:
- 0.90
- 0.95
- 0.98
- 0.995

are distinguishable without making the rest of the range unusable.

Possible approach:
- ordinary η slider plus fine-adjust control
- slider internally parameterized in dB mismatch
- toggle between η and loss in dB

The user-facing result must always show both η and dB.

## B. Interval multiple choice

Useful for beginner/fast rounds.

Default bins:

    0–0.10
    0.10–0.30
    0.30–0.60
    0.60–0.80
    0.80–0.95
    0.95–1.00

Allow adaptive intervals at higher difficulty.

## C. Ranking mode

Show 3–6 coupling scenarios and ask:

    Rank these from highest to lowest mode overlap.

This is excellent for intuition because the player need not assign exact numbers initially.

---

# 4. Other game modes

Implement the underlying game engine so these can be added easily.

## Target Practice

Unlimited questions from one selected category.

Examples:
- lateral misalignment only
- SMF-28 ↔ SSC
- Si ↔ SiN
- higher-order modes

## Blitz

60 seconds.
Answer as many interval/ranking questions as possible.

## Precision Mode

No timer.
Player gives numerical η to three decimal places.

## Tolerance Engineer

Given a nominal interface:

    η_nominal = 0.97

Ask player to estimate:
- ±0.5 μm x offset
- ±0.5 μm y offset
- ±1° angle
- ±50 nm waveguide width
- etc.

Then reveal tolerance curves.

## Optimize the Coupler

Provide 1–3 adjustable physical parameters:

Examples:
- inverse-taper tip width
- taper output slab width
- fiber lateral position
- fiber MFD
- SiN width
- TFLN width
- layer separation

Goal:

    maximize η

or:

    achieve η > 0.95 using the largest fabrication tolerance

This mode bridges intuition and real design work.

## Match the Mode

Show target mode B.
Allow player to alter waveguide A geometry until its fundamental mode matches B.

Score based on:
- final η
- number of adjustments
- solve count
- elapsed time.

## Guess the Perturbation

Show:
- original mode
- perturbed mode
- resulting overlap

Ask what changed:
- +x offset
- +y offset
- width decrease
- index decrease
- polarization change
- angular tilt

## Where Did the Power Go?

For multimode output structures:

Input mode projects onto:

    TE0: ?
    TE1: ?
    TM0: ?
    other/radiation: ?

Have player estimate the distribution.

---

# 5. Scenario taxonomy

Create a scenario generator organized around physically meaningful families.

Each family must expose:
- allowed materials
- geometry parameter distributions
- wavelength distribution
- mode selection rules
- difficulty rating
- whether scalar or vector overlap is allowed
- validity checks
- likely dominant intuition
- explanation template

---

## Family 0 — Analytic Gaussian bootcamp

No numerical eigensolver required.

Generate normalized Gaussian modes with:
- equal waists
- unequal waists
- elliptical waists
- x offset
- y offset
- x+y offset
- angular tilt / linear phase ramp
- longitudinal phase curvature where useful
- rotated elliptical Gaussian
- polarization mismatch

Typical variables:

    MFD_x
    MFD_y
    dx
    dy
    theta_x
    theta_y
    wavelength
    polarization angle

Purpose:
teach first-principles overlap intuition before complicated waveguide shapes.

Use analytic formulas as validation tests for numerical integration.

---

## Family 1 — Same-platform rectangular dielectric waveguides

Examples:

### SOI strip ↔ SOI strip

Nominal starting ranges around telecom silicon photonics:

    silicon thickness: ~180–300 nm
    widths: ~300–900 nm
    wavelength: 1260–1625 nm
    SiO2 cladding/substrate

Generate:
- equal waveguides
- width step
- height step
- both width and height change
- lateral x offset
- vertical y offset
- asymmetric cladding
- strip ↔ rib
- rib slab-height change

Questions should mostly use fundamental quasi-TE initially.

Later introduce:
- quasi-TM
- TE1 when supported
- near-cutoff geometries

---

## Family 2 — Waveguide shift / packaging alignment

Same nominal guide but one mode is transformed by:

    dx
    dy

Calculate overlap.

This should become one of the most common scenario families because it directly builds packaging intuition.

Include:
- high-confinement Si
- moderate-confinement SiN
- large weakly confined SSC modes

Teach the important fact that the same absolute alignment error means very different coupling loss depending on mode size.

---

## Family 3 — Angular misalignment / kink

For fiber-like and weakly confined modes, implement angular mismatch as a transverse phase ramp.

For a small tilt θ:

    E2(x,y) = E1(x,y) exp(i k_transverse · r)

where transverse wavevector follows the appropriate refractive medium.

Use for:
- fiber ↔ fiber
- fiber ↔ SSC
- free-space Gaussian ↔ waveguide approximation

For literal high-index waveguide bends/kinks, do not pretend this simple transform exactly captures scattering.

Advanced version:
- compare naive local overlap intuition with a more complete propagation calculation.

---

# 6. Fiber families

Create fiber objects independent from chip-waveguide objects.

## SMF-28-like fiber

At 1550 nm use realistic telecom-scale mode dimensions.

Approximate fundamental mode as:
- Gaussian for fast questions
- step-index eigenmode for higher-fidelity questions

Canonical scale:
    MFD ≈ 10 μm at 1550 nm

Do not hard-code one exact value into every question.
Randomize plausible MFD variation.

## HNA / UHNA-like fiber

Use smaller MFDs in the few-micron regime.

Create a generic "HNA fiber" family rather than implying all commercial HNA fibers share one MFD.

Possible ranges:
    MFD ≈ 3–6 μm

Eventually support named presets separately if reliable material/index data are available.

## Lensed / tapered fiber

Model output approximately as a Gaussian with adjustable:
- MFD
- curvature
- working distance
- NA

This gives useful examples between ordinary SMF and chip modes.

---

# 7. Fiber ↔ chip scenario families

These should be central to the game.

## SMF-28 ↔ raw SOI waveguide

Purpose:
demonstrate enormous mode-size mismatch.

Do not make every example absurdly close to zero:
include inverse taper stages separately.

## SMF-28 ↔ Si SSC

Examples:
- narrow inverse taper in oxide/polymer-like outer guide
- large weakly confined tip mode
- realistic x/y offsets

Generate SSC endpoint cross-sections rather than needing to simulate the full taper.

Question asks endpoint mode overlap only.

## HNA ↔ Si SSC

A very important practical intermediate regime.

Randomize:
- HNA MFD
- SSC tip width
- outer waveguide size
- alignment

Include examples where HNA is:
- clearly better than SMF-28
- too small
- almost ideal

This prevents "smaller fiber is always better" memorization.

## Fiber ↔ SiN

Include:
- thin SiN
- thick SiN
- inverse taper / wide oxide-like mode
- normal confined SiN

## Fiber ↔ TFLN

Include realistic LN ridge/slab modes where supported by solver.

---

# 8. Cross-platform integrated waveguide families

These are important advanced categories.

## SOI ↔ SiN

Examples:
- highly confined Si TE0 ↔ thick SiN TE0
- Si inverse-taper endpoint ↔ SiN mode
- vertically offset hybrid modes

## SOI ↔ TFLN

Represent:
- isolated SOI mode
- isolated LN mode
- bonded/loaded hybrid mode where practical

Teach that equal geometric centers do not imply equal field centers.

## SiN ↔ TFLN

Generate:
- LN film/slab thickness
- SiN width/thickness
- vertical layer separation
- oxide spacer

Solve actual hybrid modes when possible.

## Thick SiN ↔ thin/wide SiN

Important mode-transformer intuition.

Example families:

A:
    ~700–900 nm thick, ~1–2 μm wide SiN

B:
    ~100–400 nm thin, several-μm-wide SiN

Do not use these exact numbers as universal process definitions.
They are randomized educational ranges.

Create questions showing that:
- similar effective area does not guarantee identical mode shape
- vertical confinement can dominate mismatch
- asymmetric substrate/cladding shifts the field centroid

## Thick SiN ↔ weakly confined SiN taper endpoint

Useful fiber-interface category.

---

# 9. Shape / symmetry traps

These are essential.

A learner who only compares mode size will plateau quickly.

Generate cases such as:

## Same apparent size, different shape

Examples:
- Gaussian versus flattened waveguide mode
- Gaussian versus exponential-ish tails
- rib versus strip
- slot mode versus ordinary strip
- substrate-pulled asymmetric mode

## Same intensity image, opposite phase

Show two lobes with:
- even symmetry
- odd symmetry

Their `|E|²` images can appear extremely similar while overlap is near zero.

Make this a major advanced lesson.

## TE0 ↔ TE1

Teach orthogonality.

## TE ↔ TM

Include polarization-vector overlap.

## Rotated polarization

For otherwise identical spatial modes:

    eta ∝ |e1* · e2|²

Use simple examples before full-vector waveguide examples.

---

# 10. Multimode scenarios

When output guide supports multiple modes:

Compute several eigenmodes.

Project input onto each.

Return:

    eta_TE0
    eta_TE1
    eta_TE2
    eta_TM0
    ...

Show:

    guided mode capture = Σ eta_n

Optionally define remaining power as:

    1 - Σ guided projection

but label this carefully.

It is NOT automatically a rigorous radiation-loss calculation unless the modal basis is complete and reflections are handled.

Use text such as:

    "Power not captured by the selected forward guided modes"

rather than automatically calling it radiation loss.

---

# 11. Geometry generation

Represent all geometry parametrically.

Suggested primitive model:

    MaterialRegion {
      material
      polygon
      zOrder
    }

WaveguideScenario contains:

    wavelength
    background material
    substrate
    upper cladding
    geometryA
    geometryB
    requested modes
    transforms
    metadata
    learningConcept
    difficulty

Support shapes:
- rectangle
- trapezoid
- rib
- slab
- slot
- stacked rectangles
- arbitrary polygon later

Support:
- sidewall angle
- corner rounding only if solver representation supports it robustly
- vertical stack

Random generation must be constraint-aware.

Reject geometries when:
- intended mode is unguided
- solver does not converge
- unexpected multimode behavior invalidates question
- mode identity changes ambiguously
- field reaches boundary too strongly
- discretization error is too high

---

# 12. Material database

MVP materials:

    Si
    SiO2
    Si3N4
    LiNbO3
    air

Store refractive index as functions of wavelength where practical.

For LiNbO3 support anisotropy eventually:

    ordinary index
    extraordinary index
    optical-axis orientation

For MVP, isotropic LN approximation is acceptable ONLY if prominently encoded as an approximation internally and excluded from questions explicitly testing LN polarization physics.

Version material data and include source metadata.

---

# 13. Browser solver architecture

Goal:

Everything should run client-side if practical.

Preferred MVP approach:

## Full-vector finite-difference eigenmode solver

Implement a browser-capable FDE solver.

Reference formulation:
- established full-vector finite-difference dielectric-waveguide eigensolver formulations
- compare numerical results against known external solvers

Reasons to prefer FDE initially:
- rectangular photonic cross sections map naturally to grids
- easier than arbitrary FEM meshing in-browser
- predictable memory
- easy nonuniform/refined grid
- can operate in Web Worker
- suitable for real-time game generation

Architecture:

    UI thread
        |
        v
    question generator
        |
        v
    solver worker pool
        |
        +-- geometry rasterizer
        +-- adaptive/nonuniform mesh
        +-- epsilon tensor/map
        +-- eigensolver
        +-- mode normalization
        +-- overlap calculator
        +-- quality checks
        |
        v
    cached solved question pool

Never block UI on an eigensolve if avoidable.

Precompute upcoming questions asynchronously.

Maintain perhaps:

    5–20 solved questions ahead

depending on device performance.

---

# 14. Mesh requirements

The solver must not use a fixed pixel grid that visibly changes the answer with arbitrary geometry placement.

Implement:

## Nonuniform mesh

Fine near:
- high-index boundaries
- narrow Si cores
- slots
- thin layers

Coarse in homogeneous cladding.

Typical conceptual target:

    several to >10 cells across the smallest optically important dimension

but use convergence testing rather than a single hard-coded criterion.

## Mesh convergence check

For a subset of generated questions:
- solve at base mesh
- solve at ~1.5× or 2× resolution
- compare neff and overlap

Reject or recompute if difference exceeds tolerance.

Suggested overlap target:

    Δη < 0.005

for ordinary game questions.

Higher precision modes may require tighter criteria.

## Domain boundary check

Ensure modal field is sufficiently small at simulation boundary.

If not:
- enlarge domain
- re-solve

---

# 15. Eigensolver

Need only a handful of guided modes near a target neff.

Use a sparse/matrix-free iterative eigensolver if possible.

Candidates:
- Arnoldi
- Lanczos where operator permits
- shift-invert if practical
- custom matrix-free iteration

Implementation options:
- TypeScript
- Rust compiled to WASM
- C/C++ eigensolver compiled to WASM

Keep solver behind an interface so implementation can change.

    solveModes(problem): Mode[]

Mode:

    neff
    beta
    Ex, Ey, Ez
    Hx, Hy, Hz
    powerNormalization
    TEFraction
    TMFraction
    mesh
    metadata

Use Web Workers.

Consider WebGPU later for:
- field operators
- iterative linear algebra
- large parameter sweeps

but do not make WebGPU mandatory for MVP.

---

# 16. FEM as optional later backend

Design interfaces so an FEM backend can be added later.

FEM would improve:
- slanted sidewalls
- curved boundaries
- irregular polygons
- geometric mesh efficiency

Reference external behavior against tools such as Femwell.

Do NOT make full FEM + browser mesher a blocker for first release.

---

# 17. Gaussian fast path

Many questions do not require an eigensolve.

Implement analytic Gaussian modes and analytic overlap formulas for:
- waist mismatch
- lateral shift
- angular shift
- elliptical modes
- polarization

These questions should load essentially instantly.

They also form automated regression tests for the numerical overlap engine.

---

# 18. Mode normalization and interpolation

Two solved modes may live on different meshes.

Before overlap:

1. create common integration grid OR robust quadrature scheme
2. interpolate complex vector fields
3. preserve phase
4. normalize each mode to equal forward power
5. calculate overlap

Never compute overlap from rendered image pixels.

The visualization grid and physics integration grid must be separate concepts.

---

# 19. Complex field visualization support

Store complex fields.

Visualization data should expose selectable:

    |E|²
    |E|
    Re(Ex)
    Re(Ey)
    Re(Ez)
    arg(Ex)
    arg(Ey)
    arg(Ez)
    dominant component
    Poynting flux Sz

Player should have an adjustable colormap.

Support:
- linear scale
- log scale
- normalized scale
- global/shared scale between modes
- independent scale

For signed fields use diverging colormap capability.

For intensity use sequential colormap capability.

This is required because phase/sign/parity questions depend on it.

---

# 20. Question generation philosophy

Do not generate uniformly random physical parameters.

Generate according to learning objectives.

For every candidate question:

1. sample scenario
2. solve
3. calculate η
4. classify result
5. evaluate pedagogical usefulness
6. accept/reject

Avoid a dataset where:
- 80% of examples have η≈0
- easy geometric clues perfectly predict answer
- one family always has higher overlap than another
- parameters are unrealistically extreme

Actively balance true-overlap distribution.

Example target buckets:

    0.00–0.10
    0.10–0.30
    0.30–0.60
    0.60–0.80
    0.80–0.95
    0.95–0.99
    0.99–1.00

Not necessarily equal frequency, but all should occur.

---

# 21. Difficulty system

Assign each question an estimated difficulty.

## Level 1

- Gaussian modes
- one changing parameter
- no phase tricks
- obvious mode size differences

## Level 2

- rectangular waveguides
- lateral offsets
- fiber MFD mismatch
- one geometry dimension changes

## Level 3

- asymmetric modes
- fiber ↔ SSC
- thick ↔ thin SiN
- two simultaneous perturbations

## Level 4

- heterogeneous platforms
- vector modes
- TE/TM differences
- near cutoff
- multiple competing effects

## Level 5

- odd/even parity
- similar |E|² but differing phase
- multimode projections
- anisotropic TFLN
- counterintuitive cases

Use player performance to adapt difficulty.

---

# 22. Scoring

Continuous η error alone behaves badly across the entire range.

Compute several quantities.

Primary numerical score can use mode-mismatch loss:

    L = -10 log10(max(η, epsilon))

Compare predicted and true L.

Cap pathological penalties at a reasonable maximum.

Also track raw η error.

Example score inputs:

    etaError = |etaGuess - etaTrue|
    dBError  = |LGuess - LTrue|
    time     = responseTime

Do not reward speed enough that random fast guesses outperform thoughtful accurate ones.

Possible combined score:

    accuracyScore * modestTimeMultiplier

Show transparent scoring.

---

# 23. Feedback after every answer

Reveal:

    Your guess:        0.72
    Actual overlap:    0.84
    Mode mismatch:     0.76 dB
    Error:             -0.12

Then give ONE concise physical explanation:

Examples:

    "The spot sizes are well matched, but the vertical offset removes substantial overlap."

    "The two intensity profiles look similar, but their dominant field components have opposite parity."

    "The HNA fiber is actually smaller than the SSC mode, so reducing MFD further hurts coupling."

    "Most of the mismatch is vertical confinement rather than lateral width."

Avoid verbose textbook explanations during speedrun.

Allow expanded explanation on demand.

---

# 24. Prediction-before-reveal visualization

Do not accidentally leak the answer.

Before answer:
- show both field profiles
- show geometry
- optionally show dimensions

Do NOT show:
- combined overlap field
- coupling loss
- calculated η
- excessively revealing normalized difference map

After answer, optionally show:

    conjugate-product / overlap integrand

This is pedagogically powerful.

Let player see regions contributing:
- positively
- negatively
- negligibly

For complex/vector modes allow visualization of an appropriate overlap integrand or simplified diagnostic.

---

# 25. "Why was I wrong?" analysis

Record player biases.

Examples:

    You systematically underestimate overlaps above 0.9.

    You overweight horizontal mode size and underweight vertical mismatch.

    You often miss parity cancellation.

    Your HNA-fiber estimates are strong.

Generate future questions targeting weak categories.

---

# 26. Scenario examples that MUST exist

Create deterministic seeded examples for testing:

1. identical Gaussian modes
2. Gaussian with 2× waist ratio
3. Gaussian lateral offset
4. Gaussian angular tilt
5. equal spatial mode, 90° polarization mismatch
6. Si strip width 450 nm ↔ 500 nm
7. Si strip with lateral offset
8. Si strip ↔ much wider strip
9. SMF-28-like Gaussian ↔ raw Si strip
10. SMF-28 ↔ expanded Si SSC endpoint
11. HNA fiber ↔ SSC
12. oversized SSC ↔ HNA fiber
13. SMF-28 ↔ thin/wide SiN
14. thick SiN ↔ thin/wide SiN
15. SOI ↔ SiN
16. SiN ↔ TFLN
17. TE0 ↔ TE1
18. same |E|²-like lobes but opposite phase relationship
19. multimode receiving waveguide with TE0 + TE1 projections
20. x offset + size mismatch together

---

# 27. Validation

Physics validation is mandatory.

Create automated tests against:

## Analytic solutions

Gaussian overlap:
- identical
- size mismatch
- lateral offset
- angular tilt
- polarization

## Known slab-waveguide solutions

Validate neff.

## External numerical solvers

Create a small benchmark suite compared against at least one respected external mode solver such as:
- Femwell
- a published vector FDE implementation
- another trustworthy photonic mode solver

Compare:
- neff
- field symmetry
- polarization fraction
- normalized overlap

Set explicit tolerances.

Do not tune only to screenshots.

---

# 28. Deterministic randomization

Every question gets:

    seed
    scenarioType
    parameters
    solverVersion
    materialDBVersion
    meshSettings

A question must be reproducible exactly from its seed + version metadata.

Useful for:
- debugging
- leaderboard comparisons
- sharing difficult questions
- regression testing

Implement shareable challenge IDs.

---

# 29. Caching

Hash:

    geometry
    material parameters
    wavelength
    solver settings
    mesh settings

Cache solved modes in IndexedDB.

Many random questions will share one side of an interface.

Examples:
- same SMF mode
- same nominal Si wire
- same common SSC geometry

Exploit this heavily.

---

# 30. Performance target

Target modern desktop browsers first.

Desired experience:

- analytic question generation: effectively instantaneous
- cached numerical question: instantaneous
- ordinary new eigenmode: preferably sub-second to a few seconds
- game never pauses between questions because next questions solve in workers

Maintain a solved-question queue.

If solver backlog occurs:
- temporarily draw from cached/analytic questions
- do not stall gameplay.

---

# 31. Architecture

Suggested high-level modules:

    /physics
        materials
        gaussian
        geometry
        rasterization
        mesh
        fde
        eigen
        normalization
        interpolation
        overlap
        validation

    /scenarios
        gaussian
        alignment
        silicon
        siliconNitride
        lithiumNiobate
        fiber
        heterogeneous
        multimode

    /game
        questionGenerator
        difficulty
        scoring
        session
        adaptiveModel
        explanations

    /workers
        solverWorker
        questionPoolWorker

    /storage
        solverCache
        playerStats
        challengeSeeds

    /visualization
        fieldData
        colormaps
        geometryData

Keep styling separate from physics/game state.

---

# 32. Data schema

A generated Question should resemble:

    {
      id,
      seed,
      category,
      difficulty,
      wavelength,
      interfaceType,

      structureA,
      structureB,

      modeA: {
        modeIndex,
        polarization,
        fieldRef
      },

      modeB: {
        modeIndex,
        polarization,
        fieldRef
      },

      transforms: {
        dx,
        dy,
        angularTilt,
        polarizationRotation
      },

      answer: {
        eta,
        lossDB,
        optionalModalDistribution
      },

      concepts: [
        "mode-size-mismatch",
        "vertical-offset"
      ],

      explanation,
      solverMetadata
    }

---

# 33. Physical presets versus randomized families

Do both.

## Presets

Recognizable reference scenarios:

    SMF-28-like
    HNA-like
    standard SOI strip
    thin SiN
    thick SiN
    TFLN ridge
    Si inverse taper

## Randomized

Perturb real ranges around presets.

The game must prevent memorizing:

    "SMF-28 to this picture = 0.82"

Geometry should vary continuously.

---

# 34. Educational progression

Suggested campaign:

Chapter 1:
    What overlap means

Chapter 2:
    Spot size

Chapter 3:
    Alignment

Chapter 4:
    Fiber ↔ chip

Chapter 5:
    High-index waveguides

Chapter 6:
    Heterogeneous integration

Chapter 7:
    Polarization and vector fields

Chapter 8:
    Phase and parity

Chapter 9:
    Multimode coupling

Chapter 10:
    Overlap is not the whole scattering problem

But the 20-question speedrun should be available immediately.

---

# 35. Interesting "boss questions"

Occasionally generate deliberately deceptive but physically meaningful cases.

Examples:

## Boss: identical intensity, zero overlap

Two modes look nearly identical under |E|² but have opposite field parity.

## Boss: smaller HNA fiber is worse

SSC mode is already larger than HNA mode.

## Boss: giant horizontal match, terrible vertical match

Thin/wide SiN mode aligns horizontally with fiber but remains vertically too confined.

## Boss: geometry centered, mode not centered

Asymmetric substrate or heterogeneous stack shifts optical centroid.

## Boss: 95% overlap is not 0.05 dB

Train intuitive conversion between η and loss.

## Boss: high TE0 overlap but abrupt interface still reflects

Ask:
    "Can η alone determine transmitted power here?"

Correct answer:
    no.

---

# 36. Research references / conceptual precedents

Use these as implementation/validation inspiration rather than copying code blindly:

- Femwell photonic eigenmode solver
- Femwell fiber-overlap example
- vector finite-difference waveguide mode solver literature
- Gaussian beam alignment / coupling literature
- silicon-photonics spot-size-converter literature
- heterogeneous SiN/LN taper literature
- standard fiber mode-field-diameter definitions

Verify licensing before reusing source code.

If an external project has an incompatible license, reproduce the numerical method from published formulations rather than copying implementation.

---

# 37. MVP prioritization

Build in this order.

## Milestone 1

Analytic Gaussian engine:
- size mismatch
- offset
- tilt
- polarization
- slider answers
- 20-question speedrun
- scoring

## Milestone 2

Browser FDE:
- isotropic rectangular materials
- Si / SiO2 / SiN
- TE-like fundamental modes
- mode viewer
- numerical overlap

## Milestone 3

Scenario generator:
- SOI width changes
- offsets
- fiber ↔ SSC
- thick/thin SiN
- balanced η distribution

## Milestone 4

Advanced vector concepts:
- TE/TM
- higher-order modes
- parity
- phase visualization
- multimode projections

## Milestone 5

Heterogeneous platforms:
- SiN/TFLN
- Si/TFLN
- anisotropic materials
- stacked modes

## Milestone 6

Adaptive learning + optimization gameplay.

Do not start with every material and arbitrary FEM polygon support.

The first genuinely fun version should already exist after Milestone 2–3.

---

# 38. Definition of success

The game succeeds if an experienced player gradually becomes able to look at two mode plots and say things like:

    "That's probably ~90%, not 99%."

    "The x match is fine; y mismatch is going to cost roughly a dB."

    "SMF-28 is way too large for that endpoint."

    "HNA should help, but that one is probably too small."

    "Those intensity plots fool you—the phase parity makes overlap nearly zero."

    "The local overlap is high, but I would not infer discontinuity transmission without considering reflection/radiation."

The product should reward physical intuition rather than memorization or numerical integration skill.