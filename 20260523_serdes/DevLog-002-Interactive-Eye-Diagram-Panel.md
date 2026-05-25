# DevLog-002: Interactive Eye Diagram Panel

## Goal

Expand the optical / SerDes architecture page with a richer interactive eye-diagram experience. The panel should let a user see how channel properties close the eye, how transmitter and receiver compensation reopen it, and how timing recovery changes the effective sampling point.

The interaction should feel animated and causal:

```text
symbols
  -> Tx FIR / pre-emphasis
  -> bandwidth-limited driver
  -> channel loss / reflection / noise / jitter
  -> receiver front-end
  -> CDR sampling phase
  -> Rx FFE / DFE / threshold adaptation
  -> eye diagram and decision metrics
```

The eye should no longer be only a reconstructed visual from symbol samples. It should be generated from an oversampled waveform buffer, folded into a rolling eye heatmap, and updated continuously as the simulated stream moves.

## Why JLSD Is Useful Here

The JLSD repo already has a compact behavioral SerDes flow that is very close to what this page needs conceptually:

- PRBS / symbol generation.
- Tx FIR and finite-bandwidth driver impulse response.
- Deterministic and random TX jitter.
- Channel impulse response from either simple RC models or imported frequency response.
- AWGN.
- Phase-interpolator style sampling clock.
- Data slicers and error slicers.
- Baud-rate CDR with configurable loop gains.
- Eye heatmap generation by folding an oversampled receive waveform into UI windows.
- Interactive Makie widget with sliders for TX, channel, and RX settings.

The browser visualization should not try to run Julia directly in the first version. Instead, borrow the modeling structure and UI ideas:

```text
JLSD idea                     Browser panel adaptation
---------                     ------------------------
Eye buffer                    circular Float32 waveform buffer
heatmap_ob                    canvas heatmap / density texture
w_gen_eye_simple!             fold oversampled samples into eye bins each frame
shadow_weight                 persistence / decay control
TX FIR h1                     pre-emphasis / de-emphasis knob
channel impulse response      selectable behavioral channel models
clk_rj / skews                timing jitter and phase skew knobs
CDR Kp / Ki                   CDR loop gain controls
PI code                       recovered phase indicator
```

## Educational Questions To Answer

- Why does bandwidth loss close the eye horizontally and vertically?
- How do reflections create post-cursor and pre-cursor ISI?
- Why can Tx pre-emphasis make the transmitted waveform look worse but the received eye better?
- What is the difference between adding noise and adding jitter?
- How does receiver equalization change the eye before slicing?
- What does a CDR actually optimize: the waveform, the sampling instant, or both?
- Why can an eye look open in voltage but still have poor timing margin?
- How do NRZ and PAM4 eyes differ in threshold count, vertical margin, and sensitivity?

## Panel Layout

Use a focused eye lab view inside the optical architecture page, reachable from the `Rx DSP`, `CDR / Timing`, `Optical Channel`, and `after channel` / `equalized` signal cards.

Recommended layout:

```text
+--------------------------------------------------------------+
| Eye Diagram Lab                                               |
|                                                              |
|  [Before EQ eye]      [After EQ eye]      [Sample histogram] |
|                                                              |
|  waveform strip: Tx -> channel -> Rx samples -> equalized     |
|                                                              |
|  timing strip: ideal UI centers, CDR phase, jitter histogram  |
|                                                              |
|  controls: channel | Tx comp | Rx comp | CDR | display        |
+--------------------------------------------------------------+
```

Primary views:

- `Before EQ`: folded eye from the receive front-end waveform.
- `After EQ`: folded eye after Rx FFE / DFE or simple linear equalizer.
- `Difference / overlay`: optional mode that overlays before and after compensation.
- `PAM4 mode`: show three threshold lines and four density bands.
- `NRZ mode`: show one threshold line and two density bands.

Secondary views:

- Rolling oscilloscope trace for the current waveform.
- Sample-point markers from the CDR.
- Jitter histogram below the eye.
- Vertical sample histogram beside the eye.
- Tap bar charts for Tx FIR and Rx FFE / DFE.
- Metric tiles: eye height, eye width, estimated BER proxy, ISI energy, jitter RMS, CDR lock.

## Interaction Knobs

### Signal Mode

- `Modulation`: NRZ / PAM4.
- `Pattern`: PRBS, random, alternating, long run, stress pattern.
- `Data rate`: conceptual presets such as 25G, 56G, 112G.
- `Samples per UI`: display / simulation quality, e.g. 16, 32, 64.

### Channel

- `Channel preset`:
  - Ideal.
  - Mild low-pass.
  - Lossy backplane.
  - Reflective channel.
  - Optical direct-detect style bandwidth limit.
  - Custom simple FIR.
- `Bandwidth`: controls first-order or multi-pole low-pass corner.
- `Insertion loss at Nyquist`: dB-style educational knob.
- `Reflection amplitude`: adds delayed echo.
- `Reflection delay`: in UI.
- `Group-delay ripple`: optional later knob.
- `Noise`: RMS voltage or SNR.
- `Jitter`: random jitter and sinusoidal jitter.
- `Duty-cycle distortion`: separate rising/falling edge displacement.

### Tx Compensation

- `Tx FIR enable`.
- `Pre-cursor tap`.
- `Main tap`.
- `Post-cursor tap`.
- `Auto set Tx FIR`: choose taps from a simple channel inverse heuristic.
- `Driver bandwidth`: finite edge-speed control.
- `Swing`: output amplitude.
- `Show transmit penalty`: toggle showing how pre-emphasis increases overshoot / high-frequency energy.

### Rx Compensation

- `Rx FFE enable`.
- `FFE taps`: 3-tap first, expandable to 5 or 7 taps.
- `DFE enable`.
- `DFE tap 1`, `DFE tap 2`.
- `Auto adapt`: simple LMS-style tap update.
- `Freeze adaptation`.
- `Reset taps`.
- `Slicer threshold`: manual threshold for NRZ, three thresholds for PAM4.
- `Auto threshold`: align thresholds to sample clusters.

### CDR / Timing

- `CDR enable`.
- `Manual phase`: visible when CDR is disabled.
- `Loop gain`: concept mode uses one slider; DSP detail can expose `Kp` and `Ki`.
- `Inject phase step`.
- `Sampling jitter`.
- `Show sample phase`.
- `Show ideal UI centers`.
- `Transition density`: random / alternating / long-run stress.

### Eye Display

- `Before / after / overlay`.
- `Persistence`: equivalent to JLSD `shadow_weight`.
- `Heatmap brightness`.
- `Trace mode`: density heatmap / phosphor traces / both.
- `X span`: 1 UI / 2 UI / 4 UI.
- `Y scale`: auto / fixed.
- `Freeze eye`.
- `Clear accumulation`.
- `Show mask`: optional teaching mask for eye height and width.

## Behavioral Model

The first implementation should use a browser-native oversampled model.

State:

```text
uiPs
samplesPerUi
symbols[]
txSamples[]
channelSamples[]
rxSamples[]
equalizedSamples[]
sampledSymbols[]
eyeBeforeBins[xBins][yBins]
eyeAfterBins[xBins][yBins]
jitterBins[xBins]
sampleHistBins[yBins]
txFirTaps[]
rxFfeTaps[]
dfeTaps[]
cdrPhaseUi
cdrLockedScore
```

Per animation tick:

```text
1. Generate new symbols.
2. Upsample symbols into rectangular or pulse-shaped samples.
3. Apply Tx FIR / pre-emphasis.
4. Apply driver bandwidth impulse response.
5. Apply channel impulse response, reflection, noise, and jitter.
6. Update CDR phase from recent transitions or manual phase.
7. Sample the receive waveform at recovered phase.
8. Apply Rx FFE / DFE to sampled values.
9. Update slicer thresholds and decisions.
10. Fold before-EQ and after-EQ oversampled waveforms into eye heatmaps.
11. Update metrics and rolling strips.
```

The eye folding should follow the JLSD idea:

```text
for each oversampled waveform sample:
  x_bin = sample_phase_within_eye_window
  y_bin = quantized_voltage
  eye[x_bin][y_bin] = persistence * eye[x_bin][y_bin] + new_density
```

For performance, use fixed-size `Float32Array` buffers and update only the new samples each frame.

## Compensation Models

### Tx FIR

Start with a 3-tap FIR:

```text
y_tx[n] = c[-1] * x[n+1] + c[0] * x[n] + c[1] * x[n-1]
```

Visualize it as:

- Tap bars.
- Delayed symbol copies.
- Before / after Tx FIR waveform.
- After-channel eye with Tx FIR off versus on.

### Channel

Start with composable simple models:

```text
lowpass_ir(tau)
reflection_ir(delay_ui, amplitude)
noise_rms
timing_jitter_rms
dcd_ui
```

Then add optional imported-channel support later:

- Keep JLSD-style `frequency response -> impulse response` concept in mind.
- Browser import could accept a small JSON impulse response before trying Touchstone parsing.
- A later Python/Julia preprocessing script can convert `.s4p` / `.mat` into a compact JSON impulse response.

### Rx FFE

Start with 3 taps:

```text
y_ffe[n] = a[-1] * s[n+1] + a[0] * s[n] + a[1] * s[n-1]
```

Show both manual taps and an `Auto adapt` mode.

For teaching, the first auto mode can be a simple target-cursor heuristic:

- Estimate cursor samples from a known short training pattern or correlation.
- Set taps to reduce largest post-cursor ISI.
- Later replace with an LMS update.

### DFE

Start with one post-cursor decision feedback tap:

```text
y_dfe[n] = y_ffe[n] - b1 * decision[n-1]
```

Make the distinction visually clear:

- FFE filters the analog/sample stream and can boost noise.
- DFE subtracts estimated ISI from previous decisions and does not open the pre-slicer waveform in the same way.
- In DFE mode, show both `slicer input before DFE correction` and `decision variable after DFE correction`.

### CDR

Reuse the existing CDR model direction:

- Early / late / hold decisions from transitions.
- Accumulated phase control.
- Low-pass filtered phase.
- Sampling ticks over the rolling waveform.

The eye panel should show that CDR changes where the eye is sampled. It does not remove vertical noise or channel ISI by itself.

## Metrics

Display simple, explainable metrics rather than standards-accurate compliance numbers.

Recommended first metrics:

- `Eye height`: vertical opening at sampling phase.
- `Eye width`: horizontal region where opening stays above a threshold.
- `Sample mean levels`: NRZ two clusters or PAM4 four clusters.
- `Noise RMS`: cluster spread near sampling phase.
- `ISI estimate`: energy in neighboring cursor taps.
- `Jitter RMS`: transition-time spread.
- `Estimated BER proxy`: Q-style estimate for NRZ; nearest-threshold margin proxy for PAM4.
- `CDR lock`: recent UP/DOWN balance and phase stability.

Make the metrics explicitly educational:

```text
BER proxy is an intuition metric, not a compliance result.
```

## UI Depth Modes

The global technical-depth control from the optical architecture page can map to the eye lab:

### Concept

- Big before / after eyes.
- Few knobs: channel loss, noise, Tx precomp, Rx equalizer, CDR on/off.
- Short labels: "loss closes eye", "equalizer reopens vertical margin".

### Mechanism

- Show impulse response, taps, sample phase, jitter histogram, thresholds.
- Expose Tx FIR, Rx FFE, DFE, noise, reflection, and CDR gain controls.

### DSP Detail

- Show equations, tap adaptation state, cursor estimates, error counters, BER proxy, and slicer threshold adaptation.
- Allow custom FIR/channel tap editing.

## Suggested Presets

Presets are important because a blank set of sliders is too much work.

- `Clean Link`: ideal-ish channel, no compensation needed.
- `Bandwidth-Limited`: eye closes due to slow edges and ISI.
- `Tx Pre-Emphasis Helps`: channel loss plus Tx FIR preset.
- `Reflection Trouble`: echo creates post-cursor ISI.
- `Rx FFE Helps But Boosts Noise`: equalized eye opens but noise spread grows.
- `DFE Removes Post-Cursor`: decision eye improves without boosting high-frequency noise.
- `Jitter-Dominated`: vertical eye looks decent but timing margin collapses.
- `CDR Acquiring`: sample phase starts off-center and moves toward the eye opening.
- `PAM4 Margin Crunch`: four levels with smaller vertical margins.

## Implementation Plan

### Phase 1: Browser Eye Heatmap

- Add a full-size `Eye Diagram Lab` detail panel for the `Rx DSP` block and equalized signal card.
- Replace or supplement `drawEyeMini` with an eye heatmap generated from an oversampled waveform buffer.
- Add controls for channel loss, noise, Tx FIR on/off, Rx FFE on/off, and persistence.
- Show before-EQ and after-EQ eyes side by side.

### Phase 2: Channel And Compensation Knobs

- Add composable channel model: low-pass, reflection, noise, jitter, DCD.
- Add 3-tap Tx FIR and 3-tap Rx FFE with manual sliders.
- Add tap bar charts and impulse-response view.
- Add metric tiles for eye height, eye width, jitter RMS, and ISI estimate.

### Phase 3: CDR Integration

- Connect the existing CDR phase model to the eye sampler.
- Add manual phase versus CDR phase modes.
- Add phase-step injection.
- Show CDR sample markers and jitter histogram under the eye.

### Phase 4: Adaptation And DFE

- Add auto threshold adaptation.
- Add simple LMS / cursor-based Rx FFE adaptation.
- Add one-tap and two-tap DFE.
- Add freeze/reset adaptation controls.

### Phase 5: Imported Channel Data

- Add a small JSON impulse-response loader.
- Add a converter script later for Touchstone or MATLAB channel data.
- Include the JLSD `channel_4inch` style response as an optional demo preset if licensing and format are suitable.

## Accuracy Boundaries

- This is a behavioral teaching model, not a standards compliance simulator.
- The first eye metrics are educational estimates, not signoff measurements.
- Rx FFE / DFE adaptation can be simplified before becoming a true LMS loop.
- PAM4 support should show level compression and threshold sensitivity, but does not need full transmitter nonlinearity in the first version.
- Optical-specific impairments such as chromatic dispersion and coherent carrier recovery should remain optional future modules, separate from the first direct-detect / electrical-style eye lab.

## Open Clarification Questions

These do not block the first DevLog, but they matter before implementation:

- Should the first implementation target the existing self-contained HTML page only, or should we also create a Python/Julia helper for generating/importing channel impulse responses?
  - create a new subpage
- Should the initial eye lab focus on NRZ first, then PAM4, or keep both from day one?
  - both from day 1
- Do you want the controls to favor optical direct-detect language, electrical SerDes language, or a mode switch between the two?
  - show both
- Should imported channel data use JLSD's existing `.mat` / `.s4p` examples, or should we keep the first version entirely synthetic?
  - synthetic
- For compensation, should `DFE` be included in the first interactive release, or saved for the second pass after Tx FIR and Rx FFE are solid?
  - save for later

## Implementation Progress

### 2026-05-25: First Eye Lab Subpage

Implemented a first browser-native eye lab as `eye_diagram_lab.html`.

What landed:

- Created a dedicated dark-theme, sharp-corner `Eye Diagram Lab` subpage.
- Kept the first version self-contained in HTML/CSS/JS, with no Julia/Python runtime dependency.
- Added NRZ and PAM4 support from day one.
- Added a language selector for electrical SerDes wording, optical direct-detect wording, or both.
- Kept channel data synthetic for now.
- Deferred DFE, per clarification, while adding Tx FIR and Rx FFE first.
- Added live animation with pause, speed, new-data, clear-eye, and preset controls.
- Added presets:
  - `Clean Link`
  - `Bandwidth Limited`
  - `Tx Pre-Emphasis Helps`
  - `Reflection Trouble`
  - `Jitter Dominated`
  - `PAM4 Margin Crunch`
- Added synthetic channel controls:
  - channel loss
  - reflection amplitude
  - reflection delay in UI
  - noise
  - timing jitter
- Added Tx controls:
  - Tx FIR enable
  - pre-cursor tap
  - post-cursor tap
  - driver bandwidth
- Added Rx / timing controls:
  - Rx FFE enable
  - FFE pre tap
  - FFE post tap
  - CDR enable
  - manual phase
  - eye persistence
- Added an oversampled behavioral waveform model:
  - symbol generation
  - Tx FIR shaping
  - finite driver bandwidth
  - low-pass channel behavior
  - delayed reflection
  - AWGN-like noise
  - timing jitter
  - CDR/manual sample phase
  - 3-tap Rx FFE
- Added rolling before-EQ and after-EQ eye heatmaps using fixed `Float32Array` accumulators.
- Added sample and jitter histograms.
- Added waveform-chain strips for Tx FIR, channel output, and Rx FFE output.
- Added tap bar visualizations.
- Added educational metrics:
  - before eye height
  - after eye height
  - after eye width
  - Q proxy
  - ISI estimate
  - modulation mode

Architecture-page integration:

- Updated `optical_dsp_link_architecture.html` so these blocks now expose the lab:
  - `Tx DSP`
  - `Optical Channel`
  - `CDR / Timing`
  - `Rx DSP`
- Added an `Open Eye Diagram Lab` action to the relevant detail headers.
- Added a compact `Eye Diagram Lab` detail panel in the architecture page showing the expandable flow:

```text
Tx FIR -> Driver BW -> Channel -> CDR Phase -> Rx FFE
```

Verification:

- Ran `node --check` on the inline scripts for both `eye_diagram_lab.html` and `optical_dsp_link_architecture.html`.
- Both scripts passed syntax checks.

Git note:

- The parent repo ignores `*.html`, so `eye_diagram_lab.html` is currently ignored by default. Track it with `git add -f eye_diagram_lab.html` when committing.

## Updated Next Steps

- Visually inspect `eye_diagram_lab.html` in a browser and tune the default presets so each impairment / compensation story is obvious within a few seconds.
- Add a small phase-step button for the CDR section.
- Add an impulse-response mini-panel so channel loss and reflection have a direct time-domain visual.
- Add an `Auto Tx FIR` preset action and a simple `Auto Rx FFE` heuristic.
- Consider replacing the compact equalized eye card in `optical_dsp_link_architecture.html` with a thumbnail generated from the same style of oversampled heatmap model.
- Later: add DFE as a second-pass compensation mode after Tx FIR and Rx FFE behavior feels solid.

## Audit Notes From JLSD Cross-Check

### 2026-05-25: Calculation And Representation Audit

The first implementation is useful as a teaching sketch, but several pieces were more heuristic than the labels implied. These are the issues to fix before treating the eye lab as a faithful JLSD-inspired behavioral model:

- The CDR used the transmitted symbol stream to identify transitions. JLSD's CDR operates on data-slicer and error-slicer decisions, so the browser CDR should not peek at the ideal symbols.
- The after-EQ eye was reconstructed by interpolating symbol-rate FFE outputs. That can visually reopen an eye without representing an actual oversampled equalized waveform.
- Eye folding ignored recovered phase. The sample marker moved, but the density texture stayed locked to the original UI grid.
- Tx FIR taps were not normalized. Changing pre/post taps also changed the effective transmitter swing, which confounded pre-emphasis with amplitude gain.
- Eye width was derived directly from slider values rather than measured from the accumulated eye or sample clusters.
- The ISI estimate used adjacent-sample differences, so normal data transitions were counted as impairment.
- The simulation regenerated a full history window each frame instead of maintaining a true rolling circular waveform buffer. This is acceptable for the first browser pass, but the limitation should stay visible in the implementation plan.

### Immediate Fix Queue

- Normalize Tx FIR taps by absolute tap sum, similar to JLSD's `fir_norm`.
- Relabel CDR and metrics as educational estimates unless they are measured from simulated samples.
- Replace ideal-symbol CDR transition detection with decision-derived transition detection.
- Apply Rx FFE as a UI-spaced filter to the oversampled receive waveform for the displayed after-EQ eye.
- Fold eye heatmaps with a phase offset so CDR/manual phase changes the displayed timing reference.
- Compute eye width from vertical opening across phase bins instead of from impairment knobs.
- Replace the ISI estimate with a cursor/level-spread proxy around the sampling instant.

### 2026-05-25: Audit Fix Pass 1

Implemented the first set of corrections in `eye_diagram_lab.html`:

- Tx FIR taps are now normalized by absolute tap sum before waveform generation, so pre/post emphasis does not silently increase transmitter swing.
- The CDR no longer uses the transmitted symbol stream to find transitions. It now makes nearest-level decisions from sampled waveform values and updates phase from decision-derived boundary error.
- Before-EQ and after-EQ heatmaps are now folded with the active sampling phase offset, so manual phase and CDR phase change the eye timing reference instead of only moving the marker.
- The after-EQ waveform is now produced by applying a UI-spaced 3-tap FFE directly to the oversampled receive waveform, rather than interpolating between baud-rate equalizer outputs.
- Eye width is now estimated from accumulated heatmap density openings across phase bins, rather than from slider values.
- Q, height, and ISI proxies now use sample-cluster estimates instead of ideal transmitted symbols.
- UI labels now call out the CDR as a decision-derived estimate and the metrics as teaching-model measurements, not compliance results.

Validation:

- Re-ran `node --check` on the inline `eye_diagram_lab.html` script after the changes.

Remaining follow-up items:

- Visually tune CDR loop sign/gain and lock behavior in browser presets.
- Add explicit slicer threshold adaptation so PAM4 decisions do not rely on fixed nominal levels.
- Replace the regenerated-frame simulation with an actual rolling circular buffer.
- Add a phase-step button and CDR acquisition preset.
- Add auto Tx FIR / auto Rx FFE heuristics.

### 2026-05-25: Side-Panel Controls

Moved the eye lab controls from a second full-width section into a sticky right-side panel so the waveform/eye canvas and tuning knobs are visible together on desktop-sized screens.

Implementation notes:

- Added a two-column `.lab-layout` with the main canvas on the left and controls on the right.
- Made the controls panel sticky with its own vertical scroll area.
- Changed the controls grid to one column so the side panel stays readable.
- Let the canvas scale to the available column width instead of forcing a fixed 1360 px CSS width.
- Kept a responsive fallback that stacks controls below the canvas on narrower screens.

Validation:

- Re-ran `node --check` on the inline `eye_diagram_lab.html` script after the layout change.

### 2026-05-25: Higher Oversampling

Increased the browser simulation oversampling from 32 samples/UI to 128 samples/UI so folded eyes have more continuous horizontal density.

Implementation notes:

- Set `SPS = 128`.
- Changed eye folding to accumulate every oversampled point instead of every other point, so the added resolution contributes to the heatmap.

Validation:

- Re-ran `node --check` on the inline `eye_diagram_lab.html` script after the oversampling change.

### 2026-05-25: PAM4 Level And Reflection Audit

Double-checked the plotted waveforms and corrected a channel-model bug that made the reflection knob misleading.

Findings:

- The before-EQ and after-EQ eye panels intentionally plot continuous waveforms, not only slicer samples. In PAM4, the continuous eye can show more than four horizontal traces because transitions, bandwidth ISI, reflections, noise, and FFE combinations create trajectories between and around the four sampled levels.
- The sample histogram is the correct place to check whether the receiver is seeing four PAM4 sample clusters.
- The reflection model had a real bug: the delayed reflection and noise were included in the recursive low-pass state on the next sample. That made reflection behave like a feedback disturbance rather than a delayed echo path.
- Fixed the channel model by separating `channelMain` low-pass output from final `channel = channelMain + delayed echo + noise`.
- Changed the reflection echo to use a delayed copy of the low-pass channel path, which is closer to a simple reflected channel impulse response.
- Added measured sample-level guide lines to the eye plots and sample histogram instead of relying on fixed nominal PAM4 levels. This makes channel attenuation, reflection, and equalizer movement visible.
- Relabeled the waveform strip channel trace as `channel + echo` and added the current echo delay readout.

Validation:

- Re-ran `node --check` on the inline `eye_diagram_lab.html` script after the channel and visualization changes.
