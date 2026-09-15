# Project: Fruit Fly Connectome Learns Coherent Beam Combining and Target Tracking

Build an entertaining, visually compelling simulation in which a fruit fly's published neural connectome is used as the controller for a coherent-beam-combining (CBC) laser array.

The project should progress through two stages:

1. The fly learns to phase/steer a simulated CBC array and maximize beam quality at a commanded target location.
2. A second animated fruit fly enters the scene and moves around. The learner fly must steer and continuously track it with the beam.

The final deliverable should include:
- an interactive visualization/environment,
- a learning/training implementation,
- plots and telemetry showing learning,
- a timelapse video of training,
- and a polished final video showing successful real-time target tracking.

This is a simulation/art project. Do not interface with or provide control code for a physical laser system. The "laser" and target fly are virtual.

---

# 1. Core idea

Treat the fruit-fly connectome as a large fixed recurrent dynamical system / reservoir:

    visual observation
          ↓
    sensory encoding
          ↓
    fly connectome dynamics
          ↓
    selected output / descending-neuron activity
          ↓
    trainable motor readout
          ↓
    CBC actuator commands
          ↓
    optical far-field
          ↓
    reward

The connectome should matter materially to the behavior.

Do NOT merely train an arbitrary MLP and render a fly next to it.

Prefer:

- fixed biological connectivity,
- biologically annotated neuron populations where possible,
- sensory drive injected into plausible sensory/visual populations,
- neural dynamics evolved over time,
- and a relatively small trainable interface/readout around the connectome.

Training all ~166k neurons' connectome weights end-to-end is NOT required and is probably undesirable.

A good first implementation is analogous to reservoir computing:

    x[t+1] = F_connectome(x[t], sensory[t])

    action[t] = policy(W_out · features(x[t]))

where:
- `F_connectome` is fixed or mostly fixed,
- `W_out` is learned,
- actions control CBC phases / steering,
- reward derives from optical performance.

Optionally add reward-modulated synaptic plasticity to a restricted set of readout connections.

---

# 2. Scientific source material

## 2.1 Male Drosophila CNS connectome

Primary contemporary source:

**Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system," Cell, 2026.**

The released male CNS contains:

- 166,691 neurons,
- brain + ventral nerve cord,
- annotated neuron types,
- sensory-to-motor connectivity at synaptic resolution.

Google Research announcement:
**"A connectomics milestone: Mapping the complete male fruit fly brain," September 3, 2026.**

Important framing from Google:

> "a complete map of the male fruit fly's brain and central nervous system"

Use the actual MaleCNS graph/dataset if reasonably accessible.

Do not describe it as a pretrained behavioral neural network. It is a wiring/connectivity dataset.

### Implementation implication

Represent the connectome approximately as a sparse directed graph:

    neuron j ──w_ji──> neuron i

with weights initially proportional to synapse count.

Where neurotransmitter / excitatory-inhibitory information exists, use it.

Normalize/scalewise tune the graph so that activity remains stable.

A simple leaky-integrate-and-fire or rate-neuron model is sufficient for V1.

Example:

    V_i[t+1] =
        alpha * V_i[t]
        + gain * Σ_j W_ji S_j[t]
        + I_i[t]
        + noise

    S_i[t] = H(V_i[t] - threshold_i)

then reset/decrease `V_i`.

For a rate model:

    x[t+1] =
        (1-alpha)x[t]
        + alpha*f(Wx[t] + I[t])

The architecture must make it easy to switch between LIF and continuous-rate dynamics.

Performance matters because this is a ~166k-node sparse graph.

Preferred backends, in descending order:

1. GPU sparse implementation / WebGPU if practical.
2. C++/Rust sparse engine.
3. Python/NumPy/SciPy/JAX prototype.
4. A biologically selected subgraph fallback if full-CNS realtime simulation proves too expensive.

Keep full-CNS support as the architectural goal.

---

## 2.2 Fly body / rendering reference

Reference:

**Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion," Nature 643, 1312–1320 (2025).**

Associated project:

**TuragaLab/flybody**

FlyBody provides:
- an anatomically detailed fruit-fly model,
- MuJoCo physics,
- dm_control environments,
- walking/flight tasks.

The published FlyBody locomotion system used an ANN controller trained by imitation learning + RL, not the anatomical connectome.

Relevant training detail from the paper:

> "We used deep RL to train our fruit fly model to generate realistic locomotor behaviours."

Their policy used:
- DMPO,
- Acme,
- replay,
- Ray distributed actors,
- MuJoCo.

Do NOT replicate their enormous locomotion-training task unless useful.

For our purpose, use the FlyBody mesh/model primarily as:
- the visual representation of the learner fly,
- the target fly,
- optional simple locomotion/flight animation.

The learner fly can remain stationary next to or attached to a CBC control console.

If importing FlyBody is cumbersome, convert/use an appropriate fly mesh and preserve the attribution/license requirements.

---

# 3. Visual concept

The environment should look like a small sci-fi optical laboratory / training arena.

Suggested composition:

                         moving target fly
                              🪰
                              ↑
                        bright beam spot
                              ↑
                         far-field plane
                              ↑
                    simulated CBC array
                       ● ● ● ●
                       ● ● ● ●
                              |
                    learner fruit fly
                            🪰
                    "neural controller"

Show the fly brain/connectome as a glowing 3D neural structure or schematic adjacent to the learner fly.

Activity should visibly pulse through the neural graph during control.

Important overlays:

- target position,
- actual beam centroid,
- target crosshair,
- beam intensity heatmap,
- current Strehl,
- PIB,
- pointing error,
- RMS phase error,
- reward,
- episode number,
- training progress.

Visualize the actuator phases around the CBC emitters with colored phase rings, e.g.:

    φ = 0        → one hue
    φ = π        → opposite hue
    φ = 2π       → wrap to original

Do not make the interface look like a generic ML dashboard. The primary visual should be the physical optical scene.

---

# 4. CBC optical model

Start with a monochromatic tiled coherent array.

V1 configuration:

    wavelength λ = 1550 nm
    N = 16 emitters

Use a 4×4 square array.

Later optionally scale to:
- 8×8 / 64 emitters.

Each emitter has:

    amplitude A_n
    phase φ_n
    position (x_n, y_n)

The far-field electric field should be:

    E(kx, ky)
      = Σ_n A_n
          exp[i(
              φ_n
              + kx*x_n
              + ky*y_n
          )]

Optionally multiply by a Gaussian single-emitter element factor.

Intensity:

    I(kx,ky) = |E(kx,ky)|²

Normalize appropriately.

Use an FFT or analytic array-factor evaluation.

The simulator should be fast enough for thousands/millions of control steps.

---

# 5. Hidden hardware errors

Do NOT let the learner control an ideal array indefinitely.

Each training episode should randomize some hidden CBC imperfections.

Possible hidden parameters:

    static phase offset:
        δφ_i

    gain imbalance:
        δA_i

    emitter placement error:
        δx_i, δy_i

    phase actuator nonlinearity

    actuator quantization

    phase noise / drift:
        φ_noise_i(t)

    thermal/electrical crosstalk:
        φ_i =
            command_i
            + Σ_j C_ij command_j

V1 can begin with:

    actual_phi_i =
        command_phi_i
        + static_offset_i
        + random_walk_i(t)

The neural controller should NOT receive the hidden phase offsets.

It should infer useful corrections indirectly through observations/reward.

---

# 6. CBC control knobs

Start with:

    action_i = Δφ_i

for each CBC emitter.

For N=16:

    action dimension = 16

Phase commands wrap modulo 2π.

Later add:

    ΔA_i

and possibly:
- global steering tip,
- global steering tilt,
- polarization,
- individual pointing errors.

But keep V1 manageable.

A useful decomposition is:

    φ_i =
        steering_phase_i(target)
        + correction_phase_i(connectome)

This allows the connectome initially to learn calibration / residual correction.

Harder mode:

    connectome directly controls all phases
    with no analytic steering solution.

Implement both.

---

# 7. Optical performance metrics

Calculate at every step:

## Beam centroid

    x_beam, y_beam

## Pointing error

    e_point =
        ||r_beam - r_target||

## Peak intensity

    I_peak

## Strehl ratio

    S =
        I_peak / I_peak_ideal

for the same array geometry and total optical power.

## Power in bucket (PIB)

    PIB =
        integral over target disk of I
        / total far-field power

PIB is one of the most meaningful primary objectives for this project.

## Sidelobe suppression

    SLSR =
        10 log10(
            I_main_peak /
            I_largest_sidelobe
        )

## RMS phase error

Only show this as simulator ground truth:

    σ_phi =
        RMS(wrapped(phi_actual - phi_ideal))

The learner should not directly observe this in the difficult mode.

---

# 8. CBC reference algorithms

Implement at least one non-neural baseline.

## SPGD

Coherent beam combining frequently uses stochastic parallel gradient descent.

Reference:

**"Bandwidth and stability of the stochastic parallel gradient descent algorithm for phase control in coherent beam combination," Applied Optics 60, 4366 (2021).**

The paper studies SPGD phase control experimentally and analytically in a tiled CBC setup.

Basic SPGD:

Generate simultaneous random phase perturbation:

    δφ_i ∈ {-δ, +δ}

Measure:

    J_plus  = J(phi + δphi)
    J_minus = J(phi - δphi)

Gradient-like estimate:

    g_i ∝
        (J_plus - J_minus) δφ_i

Update:

    phi_i <- phi_i + eta*g_i

Reference CBC result:

**"Coherent beam combination of ten fiber arrays via stochastic parallel gradient descent algorithm," JOT 82, 16 (2015).**

Reported approximately:
- 96.4% combining efficiency without added disturbance,
- 92.6% with added phase disturbance,
- RMS phase errors around λ/35 and λ/23 respectively.

Use SPGD as a baseline comparison against the connectome learner.

---

## LOCSET

Reference:

**"Scalable all-fiber coherent beam combination using digital control," Applied Optics 61, 4543–4548 (2022).**

The system used single-detector electronic-frequency tagging / LOCSET and achieved:

- >95% combining efficiency,
- ~20 W stabilized output,
- RMS phase stability around λ/493.

The important idea:

each channel receives a unique small phase dither frequency.

A single detector signal can then provide channel-specific phase-error information via lock-in/demodulation.

No need to implement LOCSET in V1, but mention it in documentation as an example of a highly engineered conventional solution.

---

# 9. OPA / beam-steering reference

Although this project is framed as CBC, steering uses the same phase-array physics as an OPA.

For emitter location r_n and target direction s:

    φ_n,ideal =
        -k r_n · s

The learner must eventually discover / reproduce behavior equivalent to phase gradients.

Useful OPA calibration reference:

**"Practical two-dimensional beam steering system using an integrated tunable laser and an optical phased array," Applied Optics 59, 9985 (2020).**

This demonstrated a 32-channel OPA and used particle-swarm optimization to correct array phase errors.

Reported:
- ~0.63° × 0.58° divergence,
- >10 dB sidelobe suppression,
- <10 µs response time,
- 18° × 7° alias-free sweep.

This supports the use of black-box optimization for phase-error calibration.

---

# 10. Connectome sensory interface

This is one of the most important parts of the project.

The connectome should receive observations representing the optical scene.

Do NOT feed the entire perfect simulator state directly into the output readout.

Create a plausible sensory bottleneck.

V1 observation channels:

    target_x
    target_y

    beam_x
    beam_y

    target_minus_beam_x
    target_minus_beam_y

    beam_peak

    PIB

Optionally:

    16×16 or 32×32 grayscale far-field image

For a more biological version:

convert the scene into a low-resolution compound-eye-like image.

For example:

    721 directional samples

or a coarser ~100-channel eye initially.

Feed this image into identified visual / sensory neurons where annotations make this practical.

If exact mapping from pixels to fly visual neurons is too difficult for V1:

1. identify several hundred/thousand visual/sensory neurons,
2. assign receptive-field channels deterministically,
3. document the approximation.

Do not randomly stimulate arbitrary neurons without documenting why.

---

# 11. Connectome output interface

Prefer biologically meaningful output neurons:

- descending neurons,
- premotor neurons,
- motor-related CNS populations.

Extract a feature vector:

    z_t =
        firing rates / activity of selected output neurons

over a short temporal window.

Example:

    z_t ∈ R^256

or:

    z_t ∈ R^1024

Then use a trainable readout:

    u_t =
        tanh(W_out z_t + b)

Map `u_t` to phase increments:

    Δφ_i =
        max_step * u_i

For N=16:

    W_out shape ≈ 16 × K

This keeps the biological connectome fixed while training only a relatively small output mapping.

---

# 12. How learning should work

Implement a reward-based learning method.

The connectome itself functions as the recurrent dynamical substrate.

The primary trainable parameters can initially be:

    W_out

Optionally also:
- sensory encoder gains,
- a small subset of plastic connectome/readout synapses.

Possible training implementations:

## Option A — PPO/SAC on readout

Observation/features:

    connectome state z_t

Action:

    Δφ

Reward:

    optical score

Train a small actor/readout.

This is easiest.

## Option B — evolution strategies / CMA-ES

Optimize `W_out`.

This is very easy to parallelize and makes a fun evolutionary visualization.

## Option C — reward-modulated eligibility traces

For extra biological flavor:

    eligibility_ij(t)
        = λ_e * eligibility_ij(t-1)
          + pre_i(t) * post_j(t)

and:

    ΔW_ij
        = η * reward_prediction_error
            * eligibility_ij

This would allow the story:

"The fly receives reward when the beam improves, strengthening neural-to-actuator associations."

Implement this if it is not too destabilizing.

---

# 13. Reward function

Stage 1 — beam formation:

Start extremely simple:

    reward =
        PIB

or:

    reward =
        I(target_x, target_y)
        / total_power

Then introduce penalties:

    reward =
          w1 * PIB
        + w2 * Strehl
        - w3 * pointing_error
        - w4 * phase_command_energy
        - w5 * sidelobe_power

Suggested normalized starting values:

    w1 = 1.0
    w2 = 0.25
    w3 = 0.25
    w4 = 0.002
    w5 = 0.1

But tune empirically.

Avoid making the reward too complicated at first.

---

# 14. Curriculum

Training should visibly progress through a curriculum.

## Level 0 — one-dimensional toy task

2 or 4 beams.

Only piston phases.

Fixed target at center.

Goal:

    maximize center intensity.

This verifies the learning loop.

---

## Level 1 — 4×4 CBC phase lock

16 emitters.

Random static phase offsets.

Target remains at center.

Goal:

    maximize PIB / Strehl.

Fly learns to phase-lock the array.

---

## Level 2 — random stationary target

Each episode generates a new target:

    target ∈ steering FOV

Goal:

    steer beam onto target.

Start with analytic steering phase + learned correction.

Then progressively remove analytic assistance.

---

## Level 3 — smoothly moving target

Target follows:

- sinusoidal path,
- circular path,
- Lissajous path.

Reward tracks continuously.

---

## Level 4 — second fruit fly

A second rendered fly enters the arena.

The target fly moves according to:
- smooth random flight,
- spline path,
- mild acceleration limits.

The learner receives visual observations only.

The CBC spot must track the target fly.

Do not model injury, burning, or real laser effects.

The beam may simply create a visible glowing target spot/ring on or near the target.

---

## Level 5 — disturbance rejection

While tracking:

- change random phase offsets,
- add drift,
- temporarily disable one emitter,
- change amplitude of one channel,
- perturb target direction.

Evaluate reacquisition.

---

# 15. Desired emergent behavior

The final tracking behavior should visibly show:

1. the target fly appears,
2. the learner's visual neural populations respond,
3. connectome activity propagates,
4. output neurons drive the CBC phases,
5. the far-field spot moves,
6. the spot catches the target,
7. the target changes direction,
8. the neural activity changes,
9. the CBC reacquires it.

This should happen continuously, not through scripted beam animation.

---

# 16. Baselines / scientific sanity checks

Compare at minimum:

### A. Random controller

Random phase updates.

Expected to fail.

### B. SPGD

Conventional CBC optimization.

Expected to converge quickly for stationary beam optimization.

### C. Connectome reservoir + learned readout

Main experiment.

### D. Random graph reservoir

Same:
- neuron count or reduced matched count,
- edge count,
- firing statistics,

but randomized connectivity.

This tests whether the actual connectome structure provides any measurable advantage.

Optional:

### E. plain MLP/LSTM

Match trainable parameter count to the connectome readout system.

Do not oversell results.

The project is an entertainment experiment first and a neuroscience claim only if evidence actually supports it.

---

# 17. Key experiment plots

Generate automatically:

## Learning curve

    episode
       vs
    mean reward

## Optical performance

    episode
       vs
    PIB

    episode
       vs
    Strehl

    episode
       vs
    pointing RMS

## Tracking plot

    target_x(t)
    beam_x(t)

and separately:

    target_y(t)
    beam_y(t)

## Phase view

Heatmap:

    emitter
       vs
    time

colored by phase.

## Neural activity

Raster or population heatmap:

    neuron group
       vs
    time

Highlight:
- sensory populations,
- central recurrent populations,
- descending/output populations.

---

# 18. Visual implementation

Preferred real-time presentation:

### Browser option

- TypeScript
- Three.js
- WebGL/WebGPU
- WebWorker(s)
- WASM connectome engine if necessary

This is ideal for sharing.

### Python/offline option

- Python
- NumPy/JAX/PyTorch
- moderngl / MuJoCo renderer / Blender integration

Either is acceptable.

Strong preference:

**separate simulation core from renderer.**

Architecture:

    /connectome
    /optics
    /environment
    /learning
    /renderer
    /video
    /analysis

The optics engine should be independently testable.

---

# 19. Far-field visualization

Render the CBC far field as an animated floating plane or screen.

Represent intensity with:
- perceptually useful nonlinear colormap,
- logarithmic option,
- contour rings.

Show:

    target crosshair
    beam centroid
    Airy/main-lobe circle
    PIB bucket

Also show a small conventional 2D scientific heatmap inset.

The audience should immediately understand:

"all these little emitters interfere, and the fly is trying to make one clean spot."

---

# 20. Neural visualization

Do not attempt to render all 166k neurons individually every frame if performance suffers.

Use two levels:

### High-detail paused/intro rendering
Render connectome geometry or representative neuron traces.

### Real-time rendering
Aggregate neurons into:
- anatomical regions,
- cell classes,
- or selected representative populations.

Visual encoding:

    brightness = firing rate
    pulse = spike burst
    color = neural population

Show sensory → central → output propagation.

---

# 21. Training timelapse video

Automatically save checkpoints, for example:

    episode 0
    episode 10
    episode 100
    episode 1,000
    episode 10,000
    final

At each checkpoint run the SAME evaluation episode.

Record a side-by-side sequence.

Desired visual narrative:

### Early

- broad noisy diffraction pattern,
- wandering beam,
- low PIB,
- target missed.

### Middle

- intermittent focusing,
- beam moves in approximately correct direction,
- corrections overshoot.

### Late

- high-quality central lobe,
- fast steering,
- stable target lock.

Overlay:

    Episode
    Reward
    PIB
    Strehl
    Tracking Error

Build a 20–60 second timelapse.

---

# 22. Final demo video

Produce a second polished video.

Suggested sequence:

### 0–5 s

Title:

    "Can a fruit fly brain control a coherent laser array?"

Show fly + neural graph + 4×4 CBC.

### 5–12 s

Show emitters initially out of phase.

Far field is messy.

Neural activity starts.

The beam sharpens.

### 12–20 s

Move commanded target around.

Beam follows.

### 20–25 s

Text:

    "Now give it something to chase."

Second fly enters.

### 25–45 s

Target fly moves dynamically.

Learner continuously tracks it.

Show:
- 3D scene,
- far-field inset,
- connectome activity,
- phase visualization,
- tracking error.

### Ending

Target suddenly changes direction.

Beam briefly loses it.

Learner reacquires.

End on:

    PIB
    Strehl
    RMS tracking error
    reacquisition time

---

# 23. Evaluation targets

Do not hard-code success, but reasonable goals are:

Static phase locking:

    Strehl > 0.8

    PIB > 80% of ideal-array PIB

Tracking:

    RMS pointing error
        < 0.25–0.5 beam FWHM

Disturbance:

    reacquire within
        < 1–2 seconds simulated time

The exact thresholds may need adjustment.

---

# 24. Interesting scientific question

The experiment should ultimately let us ask:

> Can the fixed dynamical structure of an actual biological connectome act as a useful recurrent substrate for a completely alien control task?

CBC is attractive because:

- the forward physics are clean,
- actions have obvious physical meaning,
- reward can be exactly measured,
- ground truth is available,
- dimensionality scales easily,
- conventional optimizers exist,
- disturbances can be controlled,
- and success is visually obvious.

This is NOT intended to argue that flies naturally implement optical phased-array control.

Instead:

**We are deliberately connecting a biological controller to an alien but mathematically clean environment and asking what it can learn through its interfaces.**

---

# 25. Important controls against "fake connectome AI"

We need to be able to defend the statement:

    "the fly connectome is actually doing something."

Therefore log and expose:

    sensory currents sent into connectome

    connectome internal state

    selected output-neuron state

    readout weights

    final actions

Support ablation:

    bypass connectome

and compare.

A good minimum experiment:

    real connectome reservoir
        vs
    edge-shuffled connectome
        vs
    random reservoir
        vs
    same readout with raw observations

Match trainable parameter counts when possible.

---

# 26. Development order

Implement incrementally.

## Milestone 1

Pure optics simulator.

Validate:

- all phases equal → diffraction-limited central spot,
- linear phase ramp → beam steering,
- random phases → speckled/multilobe pattern.

Unit tests required.

## Milestone 2

Implement SPGD.

Show it recovering from random phase offsets.

## Milestone 3

Implement simplified neural reservoir.

Verify neural → phase control training works.

## Milestone 4

Replace reservoir with MaleCNS connectivity.

## Milestone 5

Add target steering curriculum.

## Milestone 6

Add moving target.

## Milestone 7

Add rendered second fly.

## Milestone 8

Add neural + optics visualization.

## Milestone 9

Benchmark and ablations.

## Milestone 10

Generate training timelapse and final demo video.

Do NOT spend the first several days building elaborate 3D rendering before proving the control loop.

---

# 27. Engineering requirements

Keep random seeds reproducible.

Save:

    config
    code commit
    random seed
    connectome version
    reward parameters
    checkpoints

Use configuration files for:

    neuron model
    N emitters
    pitch
    wavelength
    noise
    target speed
    learning algorithm
    reward weights

Provide CLI commands roughly like:

    train
    evaluate
    render
    benchmark
    make-video

Example:

    python main.py train --task phase_lock

    python main.py train --task target_tracking

    python main.py evaluate --checkpoint ...

    python main.py make-video --run ...

Adapt syntax to the final stack.

---

# 28. Documentation / attribution

README should clearly distinguish:

1. Google's/Janelia's MaleCNS connectome.
2. FlyBody physical fly model.
3. Our neural dynamics approximation.
4. Our sensory encoding.
5. Our trainable readout.
6. Our CBC simulator.
7. Our reward-learning implementation.

Never imply Google released a fruit fly capable of performing arbitrary RL tasks.

The biological data provide connectivity, not a pretrained arbitrary-task policy.

---

# 29. References to include in README

### Connectome

Berg et al.
"Sexual dimorphism in the complete connectome of the Drosophila male central nervous system."
Cell, 2026.

Google Research.
"A connectomics milestone: Mapping the complete male fruit fly brain."
September 3, 2026.

Key factual number:

    166,691 neurons

---

### Embodied fly simulation

Vaxenburg et al.
"Whole-body physics simulation of fruit fly locomotion."
Nature 643, 1312–1320 (2025).
DOI: 10.1038/s41586-025-09029-4

FlyBody repository:
TuragaLab/flybody

Relevant technical points:

- MuJoCo physics
- dm_control
- 59-dimensional walking actuator space
- imitation learning
- DMPO / Acme
- distributed training with Ray

---

### CBC / phase optimization

"Bandwidth and stability of the stochastic parallel gradient descent algorithm for phase control in coherent beam combination."
Applied Optics 60, 4366 (2021).

"Coherent beam combination of ten fiber arrays via stochastic parallel gradient descent algorithm."
Journal of Optical Technology 82, 16 (2015).

"Scalable all-fiber coherent beam combination using digital control."
Applied Optics 61, 4543–4548 (2022).

"Real-time fully digital control scheme for pulse coherent beam combining."
Optics Letters 49, 6333–6336 (2024).

The last system combined:
- LOCSET for phase,
- SPGD for polarization,

and reported ~95.3% CBC efficiency.

---

### OPA calibration / beam steering

"Practical two-dimensional beam steering system using an integrated tunable laser and an optical phased array."
Applied Optics 59, 9985 (2020).

Useful conceptual result:

black-box optimization such as particle swarm optimization can calibrate per-channel OPA phase errors and recover good far-field beam quality.

---

# 30. Stretch goals

Only after the main experiment works.

### Dead emitter challenge

Randomly kill one CBC channel during tracking.

See whether the fly adapts.

### Drifting array

Increase phase drift continuously.

Find maximum disturbance bandwidth the controller can track.

### Scaling

Compare:

    4
    16
    64
    256

emitters.

### Competition

Put:

    Fly controller
    SPGD
    PPO
    CMA-ES

on identical randomized arrays.

Race them.

Visualize each far field live.

### Two targets

Require the fly to choose between moving targets according to reward.

### Beam-shape task

Instead of a point:

    ring
    two spots
    flat-top

### Evolution / lesion study

Ablate connectome regions and measure control degradation.

### Random-connectome tournament

Real MaleCNS topology versus many degree-matched randomized connectomes.

This could become scientifically interesting if differences are robust.

---

# 31. Final acceptance criteria

The project is complete when all of the following work:

- [ ] CBC far-field physics validated.
- [ ] Random phases produce visibly bad beam.
- [ ] Correct phases produce diffraction-limited beam.
- [ ] Steering phase gradient moves beam correctly.
- [ ] SPGD baseline works.
- [ ] Actual MaleCNS connectivity is loaded and simulated.
- [ ] Sensory information enters connectome.
- [ ] Connectome activity affects CBC control.
- [ ] Trainable readout learns static beam optimization.
- [ ] Same controller learns beam steering.
- [ ] Moving-target tracking works.
- [ ] Second rendered fruit fly can serve as target.
- [ ] Neural activity visualization works.
- [ ] CBC phase visualization works.
- [ ] Metrics are logged.
- [ ] Random-connectome/bypass ablation exists.
- [ ] Training timelapse is rendered automatically.
- [ ] Final target-tracking demo video is rendered automatically.
- [ ] README accurately describes what was biological vs engineered.

The final result should feel simultaneously like:

- a weird neuroscience experiment,
- a coherent-optics control demo,
- a reinforcement-learning visualization,
- and an absurd video of a fruit fly learning to operate a phased-array laser.