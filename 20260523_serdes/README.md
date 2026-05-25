# SerDes Visualization Handoff

This folder contains browser-based educational visualizations for SerDes concepts and a newer top-level optical DSP link architecture page. The optical page is now the project map: it ties payload bits, FEC, PAM4/NRZ mapping, Tx precompensation, SerDes/optical lanes, CDR timing, Rx equalization, soft bits, and recovered payload into one animated view.

## Files

- `initial_prompt.md`: original requirements and follow-up requests.
- `optical_dsp_link_architecture.html`: top-level clickable optical DSP link architecture page.
  - Simulink-style full-link block diagram.
  - Expandable compact details for implemented SerDes / CDR / deserializer blocks.
  - Shared behavioral `linkState` drives payload, FEC, mapper, receive samples, equalized eye, soft bits, and recovered payload panels.
  - Includes modulation / optics / depth / speed / x-span controls.
  - Uses cached canvas logical sizes so browser zoom / DPR changes do not repeatedly inflate canvas backing stores.
- `serializer_visualization_animated.html`: main animated serializer/deserializer page.
  - Serializer tab: mux tree, final interleave, latch/mux path detail, edge regeneration.
  - Deserializer tab: demux tree, shift-register assembly, sampler/CDR selected-lane detail.
- `serdes_cdr_shift_register_deep_dive.html`: deeper CDR/PLL and shift-register page.
  - CDR scope and loop internals.
  - Basket register-bank selection circuits.
  - True shift chain.
  - Master/slave DFF snapshot timing.
- `serializer_visualization.py`: launcher for `serializer_visualization_animated.html`.
- `serializer_visualization.html`: earlier static/intermediate page.
- `DevLog-000-Optical-DSP-Link-Architecture.md`: full optical DSP architecture plan plus implementation feedback.
- `DevLog-001-CDR-PLL-Visualization.md`: CDR/PLL proposal plus implementation feedback.
- `DevLog-001-Shift-Register-Visualization.md`: shift-register proposal plus implementation feedback.

## Important Git Note

The parent repo has `.gitignore: *.html`, so new HTML pages are ignored by default. If adding another HTML file, use:

```sh
git add -f new_page.html
```

Tracked HTML files can still be modified normally once added.

## Verification

The pages are plain HTML/CSS/JS. For a syntax check, extract the inline script and run `node --check`:

```sh
python3 - <<'PY'
from pathlib import Path
import re, subprocess, tempfile, os
for name in ["optical_dsp_link_architecture.html", "serializer_visualization_animated.html", "serdes_cdr_shift_register_deep_dive.html"]:
    text = Path(name).read_text()
    script = re.search(r"<script>(.*)</script>", text, re.S).group(1)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script)
        p = f.name
    try:
        r = subprocess.run(["node", "--check", p], text=True, capture_output=True)
        print(name, r.returncode)
        if r.stderr:
            print(r.stderr)
    finally:
        os.unlink(p)
PY
```

Open locally in a browser:

```sh
open optical_dsp_link_architecture.html
open serializer_visualization_animated.html
open serdes_cdr_shift_register_deep_dive.html
```

## Modeling Boundaries

- The standalone SerDes/deep-dive pages keep the scope to serializer/deserializer muxing, latches, retiming, edge regeneration, CDR timing, and shift-register storage.
- The optical architecture page intentionally includes a behavioral optical-DSP chain: simplified FEC, NRZ/PAM4 mapping, Tx precomp, direct-detect optical lanes, channel impairment, ADC samples, CDR timing, Rx equalization, soft decisions, and recovered payload.
- The optical DSP model prioritizes consistent signal lineage and visual teaching over standards accuracy.
- CDR/PLL behavior is intentionally behavioral: transition jitter, detector noise, charge/control accumulation, loop filtering, and phase adjustment. It is not a transistor-level PLL simulator.
- The shift-register page intentionally shows both addressed basket-bank storage and cascaded DFF shifting. These are related teaching models, not necessarily the exact same high-speed implementation.
- The Rx DSP eye panel currently reconstructs a compact behavioral eye from equalized symbol samples. Transition variation is visual and must preserve sampled endpoint values; a future version should build the eye from a stored oversampled receive/equalized waveform.

## Recent Optical Page Progress

- Fixed a zoom crash caused by repeatedly scaling canvas backing dimensions from already-mutated canvas `width` attributes.
- Added an `x span` slider that changes the displayed horizontal range for time traces and discrete panels.
- Increased x resolution for the Tx precomp and timing-recovered mini traces.
- Corrected FEC grid parity coloring so it remains tied to actual codeword positions as x-span changes.
- Refined the Rx DSP eye: removed sample dots and vertical level noise, kept discrete recovered-UI updates, and constrained transition jitter to visual transition behavior rather than moving sample levels.

## Useful Next Improvements

- Generate a real oversampled Tx/channel/Rx/equalized waveform array and build the eye diagram from triggered waveform slices instead of reconstructing transitions in the draw function.
- Add sliders for CDR transition jitter, amplitude noise, detector uncertainty, and loop gain.
- Add a mode toggle connecting the same tracked bit across basket-bank, demux-tree, and shift-chain views.
- Add an optional word-boundary alignment/training-marker panel, while keeping it separate from coding/FEC topics.
- Add screenshot or Playwright visual checks if this becomes more than a local teaching artifact.
