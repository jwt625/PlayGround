read the following request and ask any clarification questions you have:
I want you to help me visualize a pure SerDes serializer example. Do not discuss encoding, PAM4, FEC, equalization, DSP, CTLE, DFE, etc. Focus only on the serializer mux, clocks, latches, retiming, and edge regeneration.

Context:

We discussed a concrete pure SerDes example:

- Start with 16 parallel binary bits presented by digital core logic.
- Serialize them into one high-speed serial NRZ bit stream.
- Example rate:
  - 16 bits in parallel at 7 GHz word rate
  - Output serial bit stream at 112 Gb/s
  - UI = 1 / 112 GHz = 8.93 ps
  - Word period = 16 UI = 143 ps

The serializer can be implemented as a tree of 2:1 mux stages:

16 bits @ 7 GHz
→ 8 streams @ 14 GHz
→ 4 streams @ 28 GHz
→ 2 streams @ 56 GHz
→ 1 stream @ 112 Gb/s

A 2:1 mux interleaves two streams:

A: a0 a1 a2 a3 ...
B: b0 b1 b2 b3 ...
output: a0 b0 a1 b1 a2 b2 ...

The actual high-speed serializer is not just passive muxing. It uses latch + mux + latch stages. Early stages may be CMOS logic. Fast stages may be CML / current-mode differential latch-muxes. Logically the mux is digital, but physically the final stages are mixed-signal custom high-speed logic.

Important concept to visualize:

A passive mux would usually degrade edges due to RC loading. But a real high-speed serializer stage has active latches / CML stages / predrivers. These stages can regenerate the data:

old degraded edge → sampled by latch → new edge launched on clean clock edge

So the serializer path can make transition edges sharper, not because muxing itself sharpens edges, but because each latch / active stage retimes and re-buffers the signal. The final edge speed is limited by the stage bandwidth, roughly set by node RC and device gm, not by the original slow core edge.

Request:

Please create a self-contained visualization, preferably in Python with matplotlib, that shows:

1. Parallel input bits:
   - 16 input bit lanes b0...b15.
   - Each input bit is valid over one 143 ps word period.
   - Show at least 2 or 3 word periods with arbitrary bit patterns.

2. Mux tree timing:
   - Show intermediate serialized stages:
     - 16 → 8 streams
     - 8 → 4 streams
     - 4 → 2 streams
     - 2 → 1 stream
   - It is OK to simplify and show only one representative stream per stage if the full 16-lane plot is too crowded, but please make the interleaving logic clear.

3. Clocks:
   - Show clocks or phase signals corresponding to 7 GHz, 14 GHz, 28 GHz, and 56 GHz/final half-rate timing.
   - Label UI = 8.93 ps and word period = 143 ps.
   - Show how the final output bit changes every UI.

4. Latch-based edge regeneration:
   - Show a degraded/slow input edge into a latch/mux stage.
   - Show that after sampling/relaunch, the output edge is sharper and aligned to a clean clock edge.
   - Model this phenomenologically, for example:
     - degraded input edge with rise time ~20 ps
     - regenerated output edge with rise time ~4 ps
   - Make clear that this is a simplified conceptual model, not transistor-level simulation.

5. Optional but helpful:
   - Include a block diagram of latch → 2:1 mux → latch repeated across stages.
   - Include a zoomed-in final-stage view showing even and odd streams interleaved into one 112 Gb/s output:
     even stream: e0 e1 e2 ...
     odd stream: o0 o1 o2 ...
     output: e0 o0 e1 o1 e2 o2 ...
   - Show the final mux controlled by a 56 GHz clock or equivalent two-phase timing.

Style requirements:

- Keep the visualization educational and uncluttered.
- Use clear labels on every trace.
- Use ps on the x-axis.
- Put the main outputs into a few figures:
  1. serializer mux tree / trace evolution
  2. final 2:1 interleaving zoom
  3. latch edge regeneration zoom
- Please write the code so I can run it directly.
- Do not include any PAM4, FEC, equalization, DSP, coding, or channel-loss discussion.

Follow-up requirements added after the initial request:

1. Make the visualization interactive / browser-based:
   - Prefer a self-contained HTML visualization, or a Python launcher that opens the HTML.
   - Show all mux-tree streams with actual random example data, not only representative streams.
   - Include a button to generate a new random bit-stream example.
   - Include a button or checkbox to toggle bit labels.
   - Keep the block diagram as a separate panel.

2. Use a dark, sharp-corner UI:
   - Use a dark theme for the full page, controls, panels, canvases, and plotted traces.
   - Use sharp corners everywhere: no rounded cards, buttons, canvases, or block-diagram boxes.

3. Add animation:
   - Create an animated version if useful.
   - Generate a continuous random stream of words/bits.
   - Let time play along the x-axis with rolling oscilloscope / strip-chart behavior:
     - newest samples enter from the right,
     - older waveform history moves left.
   - Interpolate transitions in time so traces move smoothly and are not jumpy.
   - Include play/pause and speed controls if practical.

4. Add an animated latch / mux path-detail panel:
   - Show one concrete first-stage example, such as generation of `S1.0`.
   - Use the actual mux-tree ordering used by the visualization. For example, if `S1.0` interleaves `b0` and `b8`, label that clearly.
   - Draw a small latch/mux/latch path, such as:
     - input lane `b0` waveform into latch A,
     - input lane `b8` waveform into latch B,
     - 2:1 mux,
     - output latch,
     - output waveform `S1.0`.
   - Plot the actual rolling input and output waveforms next to the corresponding wires.
   - Make wires or paths glow differently as the waveform activates them.
   - Emphasize the currently selected mux path.

5. Add clock and select traces to the latch / mux path-detail panel:
   - Show the relevant input latch clock trace, e.g. 7 GHz.
   - Show the relevant mux select / launch phase trace, e.g. 14 GHz for the `S1.0` first-stage example.
   - Draw the clock/select waveforms next to their corresponding control wires.
   - Make clock/select control paths glow according to their instantaneous level.
