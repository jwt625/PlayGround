# UI assets

Open [the UI preview](index.html) locally. Uses no dependencies, fonts, network requests, or game engine. All numbers are illustrative.

- 23 original SVG icons in icons/, drawn on a 24 × 24 grid with 1.8-unit strokes. These are code-native assets, not image-generation outputs.
- kit.css contains provisional colors and styles for counters, build buttons, inspector rows, warnings, sliders, progress indicators, and module ratings.
- index.html demonstrates build selection, relative-phase setting, an overlay toggle, and a cancellable commissioning animation. No simulation or gameplay state is connected.
- Rebuild the SVG set with `python3 assets/tools/build_ui.py` from the project directory.

Integration: retain accessible button labels, keyboard focus outlines, native range/progress controls, and state text. Color is supplementary. Runtime numbers and labels must stay editable text. The SVG stroke can be themed by embedding the SVG and setting stroke/currentColor, or by changing the asset style; img tags use the file's explicit light stroke.

Still needed for gameplay: selected-map ports, tooltips tied to real objects, causal path inspector, save/load feedback, confirmation of destructive demolition, research screen, and inventory item variants. Do not mistake the component preview for a game implementation.
