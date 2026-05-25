# DevLog-001: CDR / PLL Visualization Proposal

## Goal

Add a deep-dive visualization that explains how a deserializer recovers sampling timing from incoming serial data. The view should sit between pure block diagram and transistor-level design: it should show samplers, phase detector decisions, loop-filter charge/accumulation, phase generation, and resulting recovered sampling ticks, but it should not model device equations, noise spectra, or real analog loop stability.

Core idea:

```text
incoming serial waveform
  -> data / edge samplers
  -> early-late phase detector
  -> charge pump or digital accumulator
  -> loop filter
  -> VCO / phase interpolator
  -> recovered sampling clock
  -> deserializer sampler
```

## Educational Questions To Answer

- How does the receiver know where the middle of the UI is?
- What does "clock recovery" mean if the transmitted serial stream does not include a separate clock wire?
- How do data transitions create early/late information?
- How does a stream of noisy early/late decisions become a smooth recovered sampling phase?
- What is the difference between the bit-rate sampling cadence and the divided word clock?
- Why does the CDR need transitions, and what happens during long runs without transitions?

## Visual Model Level

Use a behavioral mixed-signal model:

- Incoming NRZ waveform is continuous-looking, with finite rise/fall time.
- Incoming edges include bounded transition jitter and the voltage trace includes small amplitude noise so the loop always has realistic residual correction work.
- Data and edge samplers are clocked comparators/slicers.
- Phase detector emits `UP`, `DOWN`, or `HOLD`.
- Charge pump / accumulator changes a visible control value.
- Loop filter smooths that control value.
- VCO / phase interpolator converts control value into sampling phase.
- Recovered sampling ticks move earlier/later until they settle near UI centers.
- After lock, the phase should not become perfectly static; detector uncertainty, VCO wander, and transition jitter should keep producing small UP/DOWN/HOLD activity.

Do not include:

- transistor-level CML latch internals
- actual loop bandwidth design
- jitter transfer math
- equalization, CTLE, DFE, DSP, FEC, PAM4, coding

## Main Panels

### 1. Incoming Data And Sampling Ticks

Show a rolling oscilloscope trace:

```text
incoming NRZ waveform
UI grid
recovered sample ticks
ideal UI centers, shown faintly for teaching
transition locations
```

Animation:

- Data rolls left, newest waveform enters at right.
- Recovered sample ticks drift slightly when the CDR is not locked.
- When locked, sample ticks settle near the center of each UI.
- Data decision markers appear at each sample tick.

Suggested labels:

```text
serial input
data sample
edge sample
UI center
transition edge
```

### 2. Bang-Bang Phase Detector

Show three conceptual samples around a transition:

```text
previous data sample | edge sample | next data sample
```

Use an early/late decision display:

```text
transition before expected edge -> DOWN / move phase earlier
transition after expected edge  -> UP / move phase later
no transition                  -> HOLD
```

The exact UP/DOWN sign convention should be labeled in the UI so the animation is self-consistent.

Visual activation:

- If a transition is detected, the edge-sampler path glows.
- `UP` pulses glow orange.
- `DOWN` pulses glow blue.
- `HOLD` is dim gray.

### 3. Charge Pump / Accumulator

Show a small charge bucket or digital accumulator:

```text
UP pulse   -> adds charge / increments accumulator
DOWN pulse -> removes charge / decrements accumulator
HOLD       -> no change
```

Animation:

- Each UP/DOWN decision launches a small packet into or out of the bucket.
- The bucket level changes in discrete steps.
- A smoothed loop-filter trace follows the bucket level with lag.

Waveforms:

```text
phase_error_decision: UP / DOWN / HOLD
raw_control
filtered_control
```

This is the best place to show "charge accumulation" without pretending to be a full transistor simulation.

### 4. Loop Filter

Show the loop filter as an RC-like smoothing block:

```text
raw UP/DOWN pulses -> smoothed control voltage / control word
```

Visual options:

- A capacitor icon fills and drains.
- A small low-pass trace shows filtered control.
- A label explains: "smooths noisy phase decisions so the recovered clock does not jump every UI."

### 5. Phase Generator

Support either PLL/VCO language or phase-interpolator language.

Recommended UI wording:

```text
Phase generator
PLL / VCO or phase interpolator
```

Behavioral model:

- Filtered control changes phase increment slightly.
- Recovered sample ticks shift left/right.
- A phase dial rotates or a tap selector moves across multi-phase clock taps.

Waveforms:

```text
reference / local multi-phase clock, conceptual
recovered sample phase
div-by-16 word clock
```

### 6. Lock Indicator

Show:

```text
phase error magnitude
locked / acquiring
```

Animation:

- Start with sample ticks too early or too late.
- Phase detector produces mostly one correction direction.
- Accumulator ramps.
- Sample ticks move toward UI centers.
- Lock indicator becomes active when recent corrections are balanced.

## Suggested Interaction Controls

- `Reset CDR`: start from an early or late phase.
- `Inject phase step`: shift incoming data phase suddenly.
- `Transition density`: choose random data, alternating data, or long-run data.
- `Noise / jitter`: optional slider for transition jitter, detector uncertainty, and input amplitude noise.
- `Show ideal centers`: toggle teaching overlay.
- `Show UP/DOWN pulses`: toggle detailed CDR internals.
- `Speed`: reuse existing animation speed control.

## Data Model Sketch

Use existing `UI` and `WORD` constants.

State variables:

```text
cdrPhasePs        // sampling phase offset within UI
phaseErrorPs      // conceptual error from edge samples
pdDecision        // -1, 0, +1
rawControl        // accumulated charge/control word
filteredControl   // smoothed control
lockedScore       // recent correction balance
```

Per animation step:

```text
1. Advance serial data.
2. Detect whether a transition occurred near the edge sampler.
3. Compare transition time to expected edge phase.
4. Emit UP, DOWN, or HOLD.
5. Update rawControl.
6. Low-pass rawControl into filteredControl.
7. Adjust cdrPhasePs from filteredControl.
8. Draw recovered sampling ticks using cdrPhasePs.
```

## Accuracy Notes

- A real receiver may use half-rate or quarter-rate clocks, interleaved samplers, phase interpolators, or VCO-based loops. The visualization should say "one sample per UI conceptually" rather than implying a large full-swing 112 GHz clock everywhere.
- This model explains timing recovery, not protocol framing.
- Long sequences with no transitions should show `HOLD` behavior or gradual reliance on the local oscillator.
- The CDR determines when to sample; the demux or shift register determines where each sampled bit is stored.

## Implementation Feedback

- Implemented in `serdes_cdr_shift_register_deep_dive.html` as CDR scope and loop-internals panels.
- Added bounded transition jitter, small amplitude noise, detector uncertainty, and oscillator wander so the CDR keeps making residual corrections after lock.
- Visualized UP/DOWN/HOLD, charge bucket, loop-filter capacitor, phase dial, recovered sample ticks, ideal centers, and div-by-16 word clock.
- Useful next improvement: expose noise/jitter amplitudes as UI sliders instead of fixed constants.
