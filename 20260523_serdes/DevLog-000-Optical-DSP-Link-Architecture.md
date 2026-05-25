# DevLog-000: Optical DSP Link Architecture

## Goal

Create a high-level architecture for an interactive, animated optical / photonic link visualization. The UI should feel like a Simulink-style block diagram: a full link is visible at the top level, signals animate through the chain, and each major block can be clicked or expanded into progressively deeper visual explanations.

This page becomes the project map for future implementation. Existing serializer, deserializer, CDR / PLL, and shift-register visuals should eventually become expandable subviews inside this architecture rather than separate one-off pages.

Core idea:

```text
payload bits
  -> framing / lane mapping
  -> FEC encode / interleave
  -> symbol mapper
  -> Tx DSP / precompensation
  -> DAC / driver
  -> optical modulator / photonic Tx
  -> fiber / optical channel
  -> photodetector / optical Rx front-end
  -> ADC / sampling
  -> Rx DSP / equalization / compensation
  -> soft demapper
  -> FEC decode / deinterleave
  -> recovered payload bits
```

The highest-level view should teach that an optical link is not only a serializer feeding a wire. It is a chain of representation changes:

```text
bits -> codewords -> symbols -> electrical waveform -> optical waveform
     -> noisy received samples -> equalized symbols -> soft bits -> corrected bits
```

## Educational Questions To Answer

- What are the major DSP blocks in a modern optical / photonic data link?
- Where do pure SerDes concepts fit inside a larger optical link?
- Why does the transmitter intentionally pre-distort or precompensate the signal?
- Why does the receiver need equalization after CDR / sampling?
- What does FEC correct, and why is pre-FEC BER allowed to be nonzero?
- What is the difference between hard decisions and soft decisions?
- How do impairments such as bandwidth loss, noise, ISI, dispersion, and nonlinearities show up visually?
- Which concepts belong to direct-detect IM-DD links, and which belong to coherent optical links?

## UI Direction

Use a top-level animated block diagram as the home screen.

Recommended interaction model:

- Every major block is clickable.
- Clicking a block expands an inline detail panel or opens a focused deep-dive view.
- The top-level signal token keeps moving even when details are collapsed.
- A global technical-depth control changes how much is shown:
  - `Concept`: simple block meaning and before/after visuals.
  - `Mechanism`: internal sub-blocks, traces, taps, thresholds, and state.
  - `DSP Detail`: equations, metrics, adaptation state, and error counters.
- Existing visualizations should be linked as deep dives:
  - Serializer mux tree.
  - Deserializer demux tree.
  - CDR / PLL timing recovery.
  - Shift-register word assembly.

Top-level layout sketch:

```text
+--------------+   +-------------+   +-----------+   +-------------------+
| Payload Bits |-->| FEC Encode  |-->| Mapper    |-->| Tx DSP / Precomp |
+--------------+   +-------------+   +-----------+   +-------------------+
                                                              |
                                                              v
+--------------+   +-------------+   +-----------+   +-------------------+
| FEC Decode   |<--| Demapper    |<--| Rx DSP    |<--| Optical Channel  |
+--------------+   +-------------+   +-----------+   +-------------------+
```

More complete version:

```text
Source
  -> Framer / Lane Mapper
  -> FEC Encoder
  -> Interleaver
  -> Scrambler
  -> Symbol Mapper
  -> Tx DSP: pulse shaping / FFE / precompensation
  -> Serializer / DAC / Driver
  -> Modulator / Photonic Tx
  -> Optical Channel
  -> Photodiode / TIA or Coherent Front-End
  -> ADC / Sampler / CDR
  -> Rx DSP: equalizer / compensation / timing / carrier recovery
  -> Soft Demapper
  -> Deinterleaver
  -> FEC Decoder
  -> Deframer / Recovered Payload
```

## Highest-Level Blocks

### 1. Payload Source / Traffic Pattern

Purpose:

Show the original information bits before coding and transmission.

Visuals:

- Bit stream ribbon.
- 16-lane or multi-lane word grid.
- PRBS / random / long-run pattern modes.
- Highlighted bit token that can be followed through later blocks.

Expandable detail:

- Existing serializer input-lane view.
- Lane ordering and word boundaries.
- Optional framing markers or alignment words.

### 2. Framer / Lane Mapper

Purpose:

Explain how payload bits are grouped into frames, lanes, or blocks before FEC and modulation.

Visuals:

- Payload bits packed into rows.
- Frame header / marker fields.
- Lane striping across parallel paths.

Accuracy note:

Keep protocol-specific details optional. The first version can use a generic frame and lane mapper rather than Ethernet-specific framing.

### 3. FEC Encoder

Purpose:

Show that extra parity bits are added before the channel so the receiver can correct errors later.

Visuals:

```text
payload bits -> codeword grid -> parity region appended
```

Interaction:

- Toggle parity visibility.
- Change FEC overhead.
- Show codeword length and payload/parity ratio.

Expandable detail levels:

- Concept: parity bits are added.
- Mechanism: codeword matrix, parity-check relationships, correction budget.
- DSP Detail: pre-FEC BER, post-FEC BER, coding gain, decoding threshold, latency.

### 4. Interleaver / Deinterleaver

Purpose:

Show how burst errors are spread across multiple FEC codewords so each decoder sees fewer errors.

Visuals:

- Rows of codewords.
- A burst error stripe before interleaving.
- Distributed red error marks after interleaving.

Interaction:

- `Interleaver on/off`.
- `Burst length`.
- `Interleaver depth`.

This block should visually connect to FEC success/failure.

### 5. Scrambler

Purpose:

Explain transition density and spectral whitening without turning into protocol coding.

Visuals:

- Before/after bit-run histogram.
- Long runs broken into more balanced random-looking data.
- Transition-density meter.

Connection to existing CDR:

Scrambler output should feed a CDR demo showing why transitions help timing recovery.

### 6. Symbol Mapper

Purpose:

Convert bits into transmitted symbols.

Visuals:

For NRZ:

```text
0 -> low
1 -> high
```

For PAM4:

```text
00 -> -3
01 -> -1
11 -> +1
10 -> +3
```

Interaction:

- `NRZ / PAM4` mode.
- Gray-code toggle for PAM4.
- Highlight current symbol and its source bits.

Expandable detail:

- PAM4 level spacing.
- Decision thresholds.
- How symbol errors can imply one or two bit errors depending on mapping.

### 7. Tx DSP / Precompensation

Purpose:

Show how the transmitter shapes or pre-distorts symbols so the received waveform is cleaner after channel impairments.

Sub-blocks:

```text
symbol stream
  -> pulse shaping
  -> feed-forward equalizer / FIR taps
  -> bandwidth / modulator precompensation
  -> optional nonlinear predistortion
```

Visuals:

- FIR tap bar chart.
- Weighted delayed symbol copies.
- Waveform before Tx DSP.
- Pre-emphasized waveform after Tx DSP.
- Eye after channel with precompensation on/off.

Interaction:

- `Channel loss`.
- `Pre-cursor tap`.
- `Post-cursor tap`.
- `Auto-tune taps`.
- `Compare uncompensated`.

Core teaching point:

Precompensation does not magically improve the transmitter waveform locally. It intentionally makes the launched waveform look distorted so the channel distortion partially cancels it.

### 8. Serializer / DAC / Driver

Purpose:

Bridge the existing pure SerDes visualizations into the optical DSP chain.

Visuals:

- Parallel digital symbols or samples become a high-rate stream.
- DAC stair-step or sample-and-hold waveform.
- Driver output amplitude.

Expandable detail:

- Existing serializer mux tree.
- Latch / mux / latch edge regeneration.
- DAC quantization and sample rate, when the DSP path uses DACs.

Accuracy note:

Some links are largely analog NRZ/PAM4 SerDes; others use DSP plus DACs. The UI should label the selected architecture.

### 9. Optical Modulator / Photonic Tx

Purpose:

Show conversion from electrical drive waveform to optical power or optical field.

Visuals:

Direct-detect IM-DD:

```text
electrical voltage -> modulator transfer curve -> optical power waveform
```

Coherent:

```text
I/Q electrical drive -> optical field constellation
```

Expandable detail:

- Mach-Zehnder modulator transfer curve.
- Bias point.
- Extinction ratio.
- Modulator bandwidth.
- Optional nonlinearity / clipping.

### 10. Optical Channel

Purpose:

Make channel impairments visible and controllable.

Direct-detect impairments:

- Optical/electrical bandwidth loss.
- ISI.
- Noise / OSNR.
- Extinction ratio penalty.
- Relative intensity noise.
- Photodiode/TIA noise.

Coherent impairments:

- Chromatic dispersion.
- Polarization mixing.
- Polarization mode dispersion.
- Laser frequency offset.
- Carrier phase noise.
- IQ imbalance.

Visuals:

- Impulse response.
- Frequency response.
- Eye closure.
- Constellation blur or rotation.
- Noise added as sample clouds.

Interaction:

- `Loss / bandwidth`.
- `Noise / OSNR`.
- `Dispersion`.
- `Nonlinearity`.
- `Polarization mixing`, for coherent mode.

### 11. Optical Rx Front-End

Purpose:

Show conversion from optical signal back to electrical samples.

Direct-detect:

```text
optical power -> photodiode current -> TIA voltage -> ADC samples
```

Coherent:

```text
optical field + local oscillator -> hybrid -> balanced photodiodes -> I/Q ADCs
```

Visuals:

- Photodiode current pulses.
- TIA bandwidth effect.
- ADC sample markers.
- Quantization levels.

Expandable detail:

- Noise sources.
- Saturation / clipping.
- ADC resolution.

### 12. CDR / Timing Recovery

Purpose:

Recover sampling phase and connect the existing CDR / PLL deep dive to the full link.

Visuals:

- Recovered sample ticks on received waveform.
- Timing error detector.
- Loop-filter control.
- Lock indicator.

Expandable detail:

- Existing CDR / PLL visualization.
- Transition-density dependency.
- Sampling phase step response.

### 13. Rx DSP / Equalization And Compensation

Purpose:

Show how receiver DSP removes channel memory and other impairments before decisions are made.

Direct-detect sub-blocks:

```text
ADC samples
  -> Rx FFE / CTLE-like equalization
  -> DFE
  -> slicer
```

Coherent sub-blocks:

```text
I/Q ADC samples
  -> chromatic dispersion compensation
  -> 2x2 MIMO polarization demux
  -> frequency recovery
  -> carrier phase recovery
  -> equalizer
```

Visuals:

- Closed eye before equalization.
- Open eye after equalization.
- ISI contribution stack: previous/current/next symbols.
- DFE feedback subtraction.
- Constellation before/after compensation.

Interaction:

- `FFE taps`.
- `DFE taps`.
- `Adaptation on/off`.
- `Show residual error`.
- `Show coherent DSP`, optional mode.

### 14. Slicer / Soft Demapper

Purpose:

Convert equalized samples or symbols into bit probabilities.

Visuals:

- NRZ threshold or PAM4 three-threshold slicer.
- Sample dots colored by confidence.
- Soft-decision bars / LLR bars.

Key teaching point:

The receiver often does not simply say "this bit is 0 or 1." It can pass confidence information to FEC, and soft information improves correction.

Interaction:

- `Hard decision`.
- `Soft decision`.
- `Noise level`.
- `Show LLR`.

### 15. FEC Decoder

Purpose:

Show the receiver correcting errors using the parity added by the transmitter.

Visuals:

- Received codeword grid with red error marks.
- Soft reliability shading.
- Syndrome / parity-check activity.
- Iterative correction animation.
- Corrected bits leaving the block.

Interaction:

- `Error rate`.
- `Burst errors`.
- `FEC strength`.
- `Hard vs soft decode`.

Metrics:

- Pre-FEC BER.
- Post-FEC BER.
- Corrected errors.
- Uncorrectable codewords.
- Decoder latency.

### 16. Recovered Payload / Link Metrics

Purpose:

Summarize whether the full link is working.

Visuals:

- Original bits and recovered bits aligned.
- Error marks before and after FEC.
- Link health dashboard.

Metrics:

- Eye height / eye width.
- EVM or symbol error rate.
- Pre-FEC BER.
- Post-FEC BER.
- FEC margin.
- Latency.
- Power or complexity, optional later.

## Recommended Page Structure

### Home View: Full Link Canvas

One full-width animated block diagram. Signal tokens move left to right through the Tx chain, through the optical channel, then right to left or continuing left to right through the Rx chain depending on layout.

Recommended first layout:

```text
Payload -> FEC -> Mapper -> Tx DSP -> SerDes/DAC -> Photonic Tx
                                                          |
                                                          v
Recovered <- FEC Decode <- Demapper <- Rx DSP <- ADC/Rx <- Optical Channel
```

Each block displays one compact live metric:

- FEC: `overhead`, `corrected errors`.
- Mapper: `NRZ` or `PAM4`.
- Tx DSP: dominant tap values.
- Channel: loss/noise/dispersion.
- Rx DSP: residual ISI or EVM.
- Demapper: confidence.
- FEC decode: pre/post-FEC BER.

### Expanded Detail View

When a block is expanded:

- Keep the block title and its input/output signal at top.
- Show 2-4 internal sub-blocks.
- Show one rolling trace, grid, eye, constellation, or codeword view.
- Include only controls relevant to the expanded block.

### Linked Deep-Dive Pages

Existing pages should be treated as block expansions:

- `Serializer / DAC / Driver` links to the serializer mux-tree visualization.
- `CDR / Timing Recovery` links to the CDR / PLL deep dive.
- `Deserializer / Word Assembly`, when shown, links to demux and shift-register visuals.

## Suggested Implementation Sequence

### Phase 1: Architecture Shell

- Create a new top-level HTML page with clickable blocks.
- Animate tokens through the link.
- Add block expansion state, but use placeholder mini-panels first.
- Add global controls:
  - `Architecture`: direct-detect / coherent placeholder.
  - `Modulation`: NRZ / PAM4.
  - `Technical depth`: concept / mechanism / DSP detail.
  - `Speed`.

### Phase 2: FEC Codeword Lab

- Implement FEC encode / channel-error / decode panels.
- Use a simplified behavioral code model first, not a production FEC implementation.
- Show codeword grid, error injection, correction budget, pre/post-FEC BER.

### Phase 3: Precompensation And Channel

- Implement Tx FIR taps, channel impulse response, waveform before/after channel, and eye comparison.
- Add auto-tune mode later.

### Phase 4: Rx Equalization / Soft Decisions

- Implement Rx FFE/DFE visual model.
- Show eye opening and sample confidence.
- Connect soft decisions into FEC decode.

### Phase 5: Integrate Existing SerDes Deep Dives

- Add clickable links or embedded panels for:
  - Serializer mux tree.
  - Deserializer demux tree.
  - CDR / PLL.
  - Shift register.

### Phase 6: Optional Coherent Mode

- Add coherent-specific blocks:
  - I/Q modulator.
  - Coherent receiver front-end.
  - Chromatic dispersion compensation.
  - 2x2 MIMO polarization demux.
  - Carrier frequency / phase recovery.
- Keep coherent mode separate enough that it does not confuse the direct-detect path.

## Modeling Boundaries

Use behavioral DSP models for teaching:

- FEC can be simplified to a correction-budget or parity-check-grid model.
- Tx precompensation can use small FIR filters and a simple channel impulse response.
- Equalization can use conceptual FFE/DFE taps and visible ISI cancellation.
- Optical channel can start with bandwidth loss, noise, and optional dispersion-like spreading.

Avoid in early versions:

- Production-grade FEC algorithms.
- Full electromagnetic / photonic device simulation.
- Vendor-specific Ethernet or OIF implementation details.
- Full coherent receiver math unless coherent mode is explicitly selected.

## Visual Design Notes

- Keep the top-level architecture dense and scannable.
- Use block colors consistently:
  - Digital / bits: blue.
  - FEC / coding: purple.
  - DSP filters: green.
  - Analog / photonic: orange.
  - Channel impairments: red.
- Keep cards flat and rectangular to match the current visualization style.
- Prefer rolling traces, codeword grids, eye diagrams, tap bars, and constellations over long explanatory text.
- Make signal representation changes explicit with labels:

```text
bits
codeword bits
symbols
samples
optical power / optical field
soft bits
corrected bits
```

## Open Decisions

- Should the first full-link implementation target NRZ, PAM4 direct-detect, or a mode switch?
- Should coherent optical DSP be a separate page or an optional advanced mode?
- Should FEC be shown as a simplified correction-budget model first, or should we implement a small real code such as Hamming/BCH-like behavior for educational correctness?
- Should existing SerDes visuals be embedded into the architecture page or opened as linked deep-dive pages?

## Initial Recommendation

Start with a direct-detect NRZ/PAM4 educational link:

```text
Payload
  -> FEC
  -> PAM4 Mapper
  -> Tx FFE / Precompensation
  -> Bandwidth-Limited Optical Channel + Noise
  -> ADC / CDR
  -> Rx Equalizer
  -> Soft Demapper
  -> FEC Decoder
  -> Recovered Payload
```

This path connects naturally to the existing SerDes, CDR, and shift-register work while introducing the most important DSP concepts: FEC, precompensation, equalization, soft decisions, and link metrics.

## Implementation Feedback

- Implemented first architecture shell in `optical_dsp_link_architecture.html`.
- Added a Simulink-style clickable full-link canvas with animated signal tokens.
- Added global controls for pause/resume, random data, technical depth, modulation mode, and speed.
- Added expandable compact details for implemented blocks:
  - `SerDes / DAC`: compact serializer mux-tree trace and final serialized output.
  - `CDR / Timing`: compact serial waveform, recovered sample ticks, word-valid clock, phase detector, accumulator, loop filter, and phase generator.
  - `Deserializer`: compact sampler, staged demux path, assembly register, and output latch view.
- Added links from expanded implemented blocks to the existing full deep-dive pages.
- Left planned DSP blocks clickable with placeholder expansion panels so future work can fill in FEC, precompensation, channel, equalization, demapper, and decoder panels one by one.
- Useful next improvement: split shared drawing/simulation primitives into a small common JS file once the architecture and existing pages need to share more than compact visual motifs.

### Revision Feedback

- Reworked the home canvas so the inter-block signals carry the main educational content instead of generic moving tokens.
- Added explicit signal representation tiles:
  - slow multi-lane payload bits,
  - FEC codeword grid with parity region,
  - NRZ/PAM4 symbol levels,
  - precompensated Tx waveform with FIR tap cue,
  - fast SerDes electrical lane bundle,
  - wavelength-colored optical launch for `4x200G CWDM4`,
  - impaired optical channel waveform,
  - noisy ADC samples,
  - CDR sample phase markers,
  - equalized eye,
  - soft-decision confidence bars,
  - decoded / recovered wide lanes.
- Added an `optics` selector with `1 optical lane` and `4x200G CWDM4` modes.
- Changed expanded block behavior so no detail panel is visible on the initial home page; details appear only after a block click.
- Rebuilt the compact serializer expansion so it no longer overflows the canvas and shows a representative mux-tree progression plus the final electrical lane bundle.
- Updated payload and recovered-payload tiles to show example realistic parallel lane counts:
  - `1 optical lane`: `16 x 12.5G = 200G`.
  - `4x200G CWDM4`: `64 = 4 x 16` core lanes, with `...` marking omitted middle lanes.
- Updated FEC/codeword tiles to display actual animated `0`/`1` bit values rather than only colored blocks.
- Kept SerDes electrical output lanes a single electrical color; wavelength coloring starts only at the optical launch / CWDM4 signal tile.
- Filled the optical launch and optical channel traces down to the local baseline with transparent wavelength-colored tone to make the waveform read as an optical intensity/envelope modulation.
- Changed payload, deserialized, and recovered payload lane tiles from word-jumping snapshots to smoothly scrolling slow-lane waveforms.
- Updated the Rx DSP tile to render an animated modulation-aware eye diagram; PAM4 mode now shows multiple level transitions and three eye openings with small jitter/shake.
- Updated the CDR/timing tile so sample phase markers scroll with the waveform and pulse/blink at the sample positions.

## Consistency Task: Shared Link Simulation

The architecture page should converge from visually related panels into mutually consistent panels. The same payload bits should feed FEC, the same FEC bits should feed the mapper, the same mapped symbols should feed Tx DSP / SerDes / optical channel, and the receiver panels should show noisy, sampled, equalized, demapped, and decoded versions of that same stream.

Target shared state:

```text
linkState = {
  payloadBits,
  fecBlocks,
  fecBits,
  mappedSymbols,
  txPrecompSamples,
  serdesElectricalLanes,
  opticalLanes,
  channelSamples,
  adcSamples,
  cdrSampledSymbols,
  rxEqualizedSymbols,
  softBits,
  decodedBits,
  recoveredPayloadBits,
  errorMask
}
```

The top-level page should become a set of views over this state:

- Payload tiles draw actual `payloadBits`.
- FEC tiles draw actual payload/parity `fecBits` and actual channel error marks.
- Mapper tiles draw symbols derived from the exact displayed FEC bits.
- Tx DSP / optical channel tiles draw waveform samples derived from the same symbols.
- Eye diagrams are generated by overlapping real waveform segments from `channelSamples` or `rxEqualizedSymbols`.
- Soft-decision tiles show actual confidence / LLR values from received sample distance to thresholds.
- Recovered payload tiles draw `recoveredPayloadBits`, so mismatches can be marked visibly when the decoder fails.

### Implementation TODOs

1. Add `linkState` generation and make it deterministic from the current random seed / payload stream.
2. Implement a simple educational FEC model first:
   - group payload bits into small blocks,
   - append actual parity bits,
   - inject channel errors after FEC,
   - correct limited errors behaviorally.
3. Replace illustrative payload/FEC/mapper/soft tiles with views over `linkState`.
4. Replace synthetic optical and Rx DSP traces with samples produced by a simple Tx FIR + channel FIR + noise model.
5. Generate actual eye diagrams by overlaying triggered waveform windows from the shared sample arrays.
6. Compute soft bits / LLRs from NRZ or PAM4 threshold distance.
7. Add explicit mismatch/error markers between original payload and recovered payload.
8. Once the shared model is stable, reuse it inside expanded block details so top-level and deep-dive views stay consistent.

Accuracy boundary:

The first shared model can be behavioral and educational. It should prioritize consistent signal lineage over production-grade FEC, exact optical device physics, or standards-accurate DSP implementation.

### First Consistency Implementation

- Added a first-pass shared `linkState` in `optical_dsp_link_architecture.html`.
- The shared state now derives:
  - `payloadBits` from the same generated payload stream,
  - simple educational FEC codewords with actual parity bits,
  - deterministic injected error masks,
  - behaviorally corrected decoded bits,
  - NRZ/PAM4 mapped symbols from the FEC bitstream,
  - noisy receive samples,
  - simple equalized symbol estimates,
  - soft bit confidence values,
  - recovered payload bits.
- Connected top-level tiles to this shared state:
  - payload and recovered lane tiles read from `payloadBits` / `recoveredPayloadBits`,
  - FEC encode/decode tiles show actual shared FEC bits,
  - mapper tile reads actual mapped symbols,
  - electrical/optical traces use mapped symbol levels,
  - ADC sample tile reads noisy receive samples,
  - Rx DSP eye tile overlays real equalized symbol windows,
  - soft demapper tile reads computed soft-bit confidence values.
- Remaining consistency work:
  - make Tx precomp use a real FIR sample array instead of a representative waveform,
  - make optical channel use stored channel sample arrays rather than direct drawing from mapped symbols,
  - feed CDR timing from the same ADC samples,
  - expose uncorrectable FEC blocks and payload mismatches visually.

### Interaction And Timing Refinements

- Added click-to-expand signal tiles on the top-level architecture page.
  - Clicking a waveform / grid / eye / soft-bit tile opens a modal with a larger rendering of the same signal view.
  - The modal continues to animate from the same shared `linkState`.
- Fixed the top-level CDR/timing tile so sample phase markers use the same UI-per-pixel scale as the waveform.
  - Before this fix, the waveform and markers used different implicit horizontal timebases.
  - Markers now scroll with the waveform and blink at recovered sample instants.
- Clarified the precompensation tile:
  - The tile currently shows one representative electrical lane, not all CWDM4 lanes.
  - The three small bars are FIR tap weights: precursor `-1`, main cursor `0`, and postcursor `+1`.
  - "FIR-shaped" means each output sample is a finite weighted sum of nearby input symbols, e.g. `y[n] = c[-1]x[n-1] + c[0]x[n] + c[+1]x[n+1]`, so the transmitter intentionally pre-distorts the waveform to counter expected channel memory.
- Fixed trace-stability issue in Tx precomp, optical launch/channel, and CDR timing mini-panels:
  - Noise/ripple terms are now functions of absolute signal/UI time instead of screen pixel plus animation time.
  - This makes the plotted signal behave like a fixed trace moving through a viewport instead of morphing while it scrolls.
- Fixed Rx DSP eye stability:
  - Removed continuous animation-time terms from the eye overlay jitter and opacity.
  - The eye now updates as a triggered snapshot when the UI trigger advances; traces do not drift continuously between trigger updates.
- Removed decorative Rx DSP eye traces:
  - The eye now uses one color only.
  - It draws only the actual equalized symbol samples from `linkState`, with straight line segments between adjacent sample points and dots at the sample locations.
  - Synthetic jitter and alternating trace colors were removed.
- Refined Rx DSP eye after review:
  - The eye still uses actual equalized sample levels from `linkState`.
  - Curved/noisy-looking transitions are allowed between real sample levels for visual readability, but the levels/endpoints remain data-derived.
- Fixed CWDM lane consistency:
  - `SerDes output`, `optical launch`, and `optical channel` now deinterleave the mapped symbol stream across optical/electrical lanes.
  - λ1..λ4 no longer show identical waveforms; each lane carries a different symbol subsequence from the shared data stream.
- Reduced default impairment severity:
  - Lowered deterministic channel bit-error injection probability in the educational FEC model.
  - Let the simplified FEC model correct up to two codeword bit flips for the displayed top-level view.
  - Reduced PAM4/NRZ receive sample noise so the default view usually shows FEC-cleaned recovered payload rather than frequent visible residual errors.
