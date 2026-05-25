# SerDes Visualization Handoff

This folder contains browser-based educational visualizations for a pure 16:1 NRZ SerDes example.

## Files

- `initial_prompt.md`: original requirements and follow-up requests.
- `optical_dsp_link_architecture.html`: top-level clickable optical DSP link architecture page.
  - Simulink-style full-link block diagram.
  - Expandable compact details for implemented SerDes / CDR / deserializer blocks.
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

- Keep the scope to serializer/deserializer muxing, latches, retiming, edge regeneration, CDR timing, and shift-register storage.
- Do not add PAM4, FEC, equalization, DSP, CTLE, DFE, coding, or channel-loss discussion unless the user explicitly changes scope.
- CDR/PLL behavior is intentionally behavioral: transition jitter, detector noise, charge/control accumulation, loop filtering, and phase adjustment. It is not a transistor-level PLL simulator.
- The shift-register page intentionally shows both addressed basket-bank storage and cascaded DFF shifting. These are related teaching models, not necessarily the exact same high-speed implementation.

## Useful Next Improvements

- Add sliders for CDR transition jitter, amplitude noise, detector uncertainty, and loop gain.
- Add a mode toggle connecting the same tracked bit across basket-bank, demux-tree, and shift-chain views.
- Add an optional word-boundary alignment/training-marker panel, while keeping it separate from coding/FEC topics.
- Add screenshot or Playwright visual checks if this becomes more than a local teaching artifact.
