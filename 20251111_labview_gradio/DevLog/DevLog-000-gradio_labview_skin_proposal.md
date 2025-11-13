# Gradio LabVIEW Skin — Proposal (Moderately Hard Scope)

## 🎯 Objective
Create a **LabVIEW-style skin for Gradio** that provides an industrial, instrument-panel aesthetic while remaining lightweight and compatible with the Gradio theming and component systems.
This includes a theme, custom CSS, and a small suite of bespoke interactive components (Knob, LED, Gauge, SevenSegment) with a Plotly template for waveform-like displays.

**SCOPE CONSTRAINTS:**
- **Dark mode only** - focusing on modern industrial HMI aesthetic
- **Development/local use** - no PyPI publishing required for now
- **LabVIEW-specific design** - no generic npm packages, custom-built components

---

## 🧱 Scope Overview

**Deliverables:**
- A reusable Gradio theme (`LabVIEWTheme`) mimicking LabVIEW’s visual style.
- 4 custom components:
  1. **Knob** — rotary analog input.
  2. **LED** — on/off indicator light.
  3. **Gauge** — analog meter.
  4. **SevenSegment** — numeric display.
- Optional helper: `labview_plotly_template()` to restyle charts.
- All bundled in a local package: `gradio-labview-skin`.

**Tech stack:**
- Python 3.10+ (Gradio 4.x + Plotly)
- **Svelte** (Gradio's default frontend framework for custom components)
- CSS variables and SVG-based widgets
- Vite for bundling (included in Gradio CC workflow)

---

## 🧩 Components

### 1. Knob
**Purpose:** emulate LabVIEW rotary controls.

**Props:**
- `min`, `max`, `step`, `value`
- `detents` (bool)
- `tick_marks` (bool)
- `units` (str, optional)

**Behavior:**
- Mouse drag and wheel for value change
- Emits `input` (on drag) and `change` (on release)

**Render:** SVG arc with rotating indicator.

---

### 2. LED
**Purpose:** small indicator light.

**Props:**
- `value`: bool or enum (`"off" | "green" | "yellow" | "red"`)
- `blink`: bool
- `size`: `"sm" | "md" | "lg"`
- `label`: optional

**Render:** radial gradient with CSS glow; optional pulse animation when blinking.

---

### 3. Gauge
**Purpose:** analog circular meter.

**Props:**
- `min`, `max`, `zones` (e.g., `[(0, 60, "green"), (60, 80, "yellow"), (80, 100, "red")]`)
- `tick_interval`, `units`, `legend`

**Render:** SVG arcs with moving needle and colored bands.

---

### 4. SevenSegment
**Purpose:** digital numeric display.

**Props:**
- `digits`, `precision`, `unit`, `color`
- `align`: `"left" | "center" | "right"`

**Render:** CSS/SVG segments; dim inactive bars.

---

### 5. Plotly Template
**Purpose:** mimic LabVIEW waveform charts.

**Appearance:**
- Pale green LCD-style background (`#edf0e2`)
- Fine gridlines, dark ticks, thick axes
- Template applied via `fig.update_layout(template=labview_plotly_template())`

---

## 🎨 Visual System (Dark Mode Only)

**Palette** (ISA-101 HMI inspired)
| Element | Dark Mode |
|----------|------|
| Panel Background | `#2b2b2b` |
| Panel Border | `#1a1a1a` |
| Control Background | `#3a3a3a` |
| Accent (Green/OK) | `#6cf06f` |
| Warning (Yellow) | `#ffd84d` |
| Error (Red) | `#ff7676` |
| LCD Background | `#1e1e1e` |
| LCD Text | `#6cf06f` |
| Inactive/Dim | `#555555` |
| Text Primary | `#e0e0e0` |
| Text Secondary | `#a0a0a0` |

**Typography**
- UI: Arial / Helvetica / system-ui
- Readout/Numeric: IBM Plex Mono or Consolas (monospace)
- Font sizes: 12px (small), 14px (normal), 16px (large)

**Shape**
- Square corners (border-radius = 0 or 2px max)
- Subtle inset bevels using box-shadow
- 1-2px borders for definition

---

## ❓ Technical Q&A and Design Decisions

### 1. Gradio Custom Components Architecture

**Q: What is the current Gradio custom component API?**
- Gradio 4.x uses the `gradio cc` CLI for component development
- Components are built with **Svelte** (not React/TypeScript as originally proposed)
- Workflow: `create` → `dev` → `build` → `install`

**Q: How to scaffold components?**
```bash
gradio cc create ComponentName --template Slider
```
Available templates: `SimpleTextbox`, `Slider`, `Number`, `Image`, etc.

**Q: Build/bundling process?**
- Vite is automatically configured by Gradio CC
- Frontend (Svelte) and backend (Python) are separate
- Build with: `gradio cc build`
- Install locally: `pip install -e .`

### 2. Component Interactivity

**Knob Component:**
- **Primary interaction**: Click-and-drag (circular motion)
- **Secondary**: Mouse wheel for fine adjustment
- **Tertiary**: Double-click to type value directly (like LabVIEW)
- **Events**: `input` (during drag), `change` (on release)
- **Debouncing**: 16ms (60fps) during drag to avoid backend overload

**Gauge Component:**
- **Read-only** by default (indicator, not input)
- Optional: Click zones to set value (advanced feature, Phase 2)

**LED Component:**
- **Read-only** indicator
- Supports blink animation via CSS `@keyframes`

**SevenSegment Component:**
- **Read-only** display
- Updates via backend value changes

**Real-time updates:**
- All components support streaming via Gradio's `every=` parameter
- Use `gr.State` for high-frequency updates without UI lag

### 3. Theme System

**Q: Gradio version compatibility?**
- Target: **Gradio 4.x** (current stable)
- Theme API: `gr.themes.Base` subclass
- No need for 3.x backward compatibility

**Q: CSS injection method?**
```python
theme = LabVIEWTheme()
with gr.Blocks(theme=theme, css="path/to/labview.css") as demo:
    ...
```
- Theme object handles color tokens
- Separate CSS file for component-specific styles

**Q: Dark mode toggle?**
- **No toggle** - dark mode only (per requirements)
- Simplifies development and maintains consistency

### 4. Color Palette Refinement

**Rationale for color choices:**
- `#2b2b2b` panel: Common in industrial HMI (ISA-101 compliant)
- `#6cf06f` green: High contrast on dark, similar to LabVIEW indicators
- `#1e1e1e` LCD: Darker than original proposal for better dark mode integration
- Avoid pure black (#000) - causes eye strain in dark UIs

**LabVIEW reference:**
- Classic LabVIEW uses light gray (`#d7d7d7`)
- Modern LabVIEW NXG has dark theme option
- We're targeting the "Silver" control palette aesthetic in dark mode

### 5. Plotly Template

**Integration:**
```python
from gradio_labview import labview_plotly_template

fig = go.Figure(...)
fig.update_layout(template=labview_plotly_template())
```

**Features:**
- Dark background (`#2b2b2b`)
- Green trace color (`#6cf06f`)
- Fine gridlines (`#3a3a3a`)
- Thick axes, minimal chrome
- Supports both light and dark mode? **No** - dark only

### 6. Package Distribution

**Current scope:**
- **Local development only** - no PyPI publishing
- Install via: `pip install -e .` or `pip install .`
- Frontend assets bundled in Python wheel
- No CDN required - Gradio handles asset serving

---

## 🧰 Implementation Plan (Best Practices)

### 1. Theme and CSS

**Theme Definition** (`theme.py`):
```python
import gradio as gr

class LabVIEWTheme(gr.themes.Base):
    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        )
        # Dark mode only
        self.set(
            body_background_fill="#2b2b2b",
            body_background_fill_dark="#2b2b2b",
            panel_background_fill="#3a3a3a",
            panel_border_color="#1a1a1a",
            block_background_fill="#3a3a3a",
            block_border_width="1px",
            block_radius="0px",
            button_primary_background_fill="#6cf06f",
            button_primary_text_color="#1a1a1a",
            button_secondary_background_fill="#555555",
            input_background_fill="#1e1e1e",
            input_border_color="#555555",
        )
```

**Custom CSS** (`labview.css`):
```css
/* Dark mode industrial aesthetic */
:root {
  --lv-bevel-highlight: rgba(255, 255, 255, 0.1);
  --lv-bevel-shadow: rgba(0, 0, 0, 0.4);
  --lv-lcd-glow: #6cf06f;
}

/* Inset bevel effect for panels */
.gradio-container .block {
  box-shadow:
    inset 1px 1px 2px var(--lv-bevel-shadow),
    inset -1px -1px 1px var(--lv-bevel-highlight);
}

/* Square buttons with subtle 3D effect */
button {
  border-radius: 0;
  box-shadow:
    0 1px 0 var(--lv-bevel-highlight) inset,
    0 -1px 0 var(--lv-bevel-shadow) inset;
  transition: all 0.1s;
}

button:active {
  box-shadow:
    0 -1px 0 var(--lv-bevel-highlight) inset,
    0 1px 0 var(--lv-bevel-shadow) inset;
}

/* LCD-style displays */
.lv-lcd {
  background: #1e1e1e;
  border: 2px solid #3a3a3a;
  font-family: "IBM Plex Mono", monospace;
  color: var(--lv-lcd-glow);
  text-shadow: 0 0 4px var(--lv-lcd-glow);
  padding: 8px 12px;
  letter-spacing: 0.05em;
}
```

### 2. Component Structure (Svelte-based)

**Directory structure:**
```
gradio_labview/
  __init__.py
  theme.py
  components/
    knob/
      __init__.py              # Python component class
      knob.py                  # Backend logic
      frontend/
        Index.svelte           # Main Svelte component
        Example.svelte         # Demo/example
        style.css              # Component-specific styles
        package.json
        vite.config.js
    led/
      __init__.py
      led.py
      frontend/
        Index.svelte
        ...
```

**Python side** (`knob.py`):
```python
from gradio.components import FormComponent

class Knob(FormComponent):
    def __init__(
        self,
        value=0,
        minimum=0,
        maximum=100,
        step=1,
        label=None,
        **kwargs
    ):
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        super().__init__(value=value, label=label, **kwargs)

    def preprocess(self, x):
        return float(x)

    def postprocess(self, y):
        return float(y)
```

**Frontend side** (`Index.svelte`):
```svelte
<script lang="ts">
  export let value: number;
  export let minimum: number;
  export let maximum: number;
  export let step: number;

  let dragging = false;
  let startAngle = 0;

  function handleDrag(event: MouseEvent) {
    if (!dragging) return;
    // Calculate angle from center
    // Update value based on rotation
    // Dispatch change event
  }
</script>

<svg class="knob" on:mousedown={startDrag}>
  <!-- SVG knob rendering -->
</svg>

<style>
  .knob {
    cursor: pointer;
    user-select: none;
  }
</style>
```

**Best practices:**
- Use `FormComponent` for input components (Knob)
- Use `Component` for display-only (LED, Gauge, SevenSegment)
- Implement `preprocess()` and `postprocess()` for data conversion
- Use TypeScript in Svelte for type safety
- Debounce rapid updates with `requestAnimationFrame`

---

## 🧪 Example Usage

```python
import gradio as gr
from gradio_labview import LabVIEWTheme, Knob, LED, Gauge, SevenSegment

def compute(a, b):
    s = a + b
    status = s > 50
    return s, status

theme = LabVIEWTheme()

with gr.Blocks(theme=theme, css="labview.css") as demo:
    gr.Markdown("## LabVIEW Panel")

    with gr.Row():
        a = Knob(label="Input A", min=0, max=100)
        b = Knob(label="Input B", min=0, max=100)
        out = SevenSegment(label="Sum")
        led = LED(label="OK")

    calc = gr.Button("Compute")
    calc.click(compute, [a, b], [out, led])

demo.launch()
```

---

## 📦 Package Layout

```
gradio-labview-skin/
  pyproject.toml
  README.md
  gradio_labview/
    __init__.py
    theme.py
    assets/
      labview.css
      plotly_template.json
    components/
      knob/
      led/
      gauge/
      sevensegment/
```

---

## 🧠 Accessibility & UX
- Keyboard navigation: arrow keys and tab focus.
- Colorblind support: add icon or text fallback for LED indicators.
- Responsive SVGs for high-DPI displays.

---

## ⚙️ Testing & Validation
- Unit tests for Python side (Gradio value binding).
- Visual regression (Playwright screenshot tests).
- Interaction tests for Knob and Gauge (mouse + keyboard events).
- Theme snapshot validation in Storybook-style page.

---

## 🚀 Roadmap (Revised)

| Phase | Tasks | Duration | Priority |
|--------|--------|----------|----------|
| 0 | **Visual reference gathering** + color extraction | 0.5 day | CRITICAL |
| 1 | Base theme + CSS styling + proof-of-concept | 1 day | HIGH |
| 2 | LED component (simplest, validates workflow) | 0.5 day | HIGH |
| 3 | SevenSegment component | 0.5 day | HIGH |
| 4 | Gauge component (medium complexity) | 0.75 day | MEDIUM |
| 5 | Knob component (most complex) | 1 day | MEDIUM |
| 6 | Plotly template + integration | 0.5 day | LOW |
| 7 | Testing + refinement | 0.5 day | HIGH |

Total: **~5 days of focused development** (more realistic with learning curve)

**Rationale for order:**
1. **Visual references first** - ensures accuracy before coding
2. **LED first** - simplest component, validates Svelte workflow
3. **Knob last** - most complex interaction, build experience first

---

## 📸 Visual References Needed

To ensure LabVIEW-specific accuracy, gather these references:

### 1. LabVIEW Screenshots (Dark Theme Preferred)

**Control Elements:**
- [ ] Rotary knob (multiple angles if possible)
- [ ] Knob with tick marks and labels
- [ ] Knob in hover/active state
- [ ] Round LED (off, green, yellow, red states)
- [ ] Square LED variant
- [ ] Circular gauge with colored zones
- [ ] Gauge needle detail
- [ ] Seven-segment numeric display (3-5 digits)
- [ ] Boolean button (mechanical action style)

**UI Elements:**
- [ ] Front panel background (capture exact color)
- [ ] Panel borders and bevels
- [ ] Button states (normal, hover, pressed, disabled)
- [ ] Waveform chart with LCD background
- [ ] Control labels and typography

**Color Extraction:**
- [ ] Use color picker on LabVIEW to get exact hex codes
- [ ] Panel gray (classic: `#d7d7d7`, dark: TBD)
- [ ] Indicator green (exact shade)
- [ ] LCD background color
- [ ] Border colors

### 2. Alternative: LabVIEW Control Palette Export

If you have LabVIEW installed:
1. Open LabVIEW
2. Create a VI with all control types
3. Set to dark theme (if available)
4. Screenshot at 2x resolution for detail
5. Export individual controls as images if possible

### 3. Reference Sources

**Official NI Documentation:**
- LabVIEW UI design guidelines
- Control palette documentation
- Color scheme specifications

**Community Resources:**
- JKI Flat UI Controls (for modern LabVIEW aesthetic)
- DMC UI Styles Toolkit
- LabVIEW subreddit examples

### 4. SVG/Vector Assets (If Available)

**Ideal but not required:**
- Vector graphics of knob designs
- Gauge arc templates
- Seven-segment digit paths

**Fallback:**
- Hand-code SVG based on screenshots
- Use CSS for seven-segment (more performant)

---

## 🎨 Component-Specific Design Notes

### Knob Component
**SVG Structure:**
```svg
<svg viewBox="0 0 100 100">
  <!-- Outer ring with tick marks -->
  <circle cx="50" cy="50" r="45" fill="#3a3a3a" stroke="#1a1a1a"/>

  <!-- Tick marks (generated programmatically) -->
  <g class="ticks">
    <!-- 11 ticks for 0-100 range -->
  </g>

  <!-- Rotating knob body -->
  <g transform="rotate({angle} 50 50)">
    <circle cx="50" cy="50" r="35" fill="#555"/>
    <!-- Indicator line -->
    <line x1="50" y1="15" x2="50" y2="30" stroke="#6cf06f" stroke-width="3"/>
  </g>
</svg>
```

**Interaction:**
- Track mouse position relative to center
- Calculate angle: `Math.atan2(dy, dx)`
- Map angle to value range
- Clamp to min/max
- Emit events at 60fps max

### LED Component
**CSS-based (no SVG needed):**
```css
.led {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #6cf06f, #2a5a2a);
  box-shadow:
    inset 0 -2px 4px rgba(0,0,0,0.4),
    0 0 8px #6cf06f;
}

.led.off {
  background: #2a2a2a;
  box-shadow: inset 0 -2px 4px rgba(0,0,0,0.4);
}

.led.blink {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0.3; }
}
```

### Gauge Component
**SVG arc generation:**
- Use `<path>` with arc commands
- Color zones as separate arcs
- Needle as rotated `<line>` or `<polygon>`
- Labels positioned with trigonometry

### SevenSegment Component
**Pure CSS approach (most performant):**
```css
.segment {
  position: absolute;
  background: #6cf06f;
  box-shadow: 0 0 4px #6cf06f;
}

.segment.off {
  background: #2a2a2a;
  box-shadow: none;
}

/* Define 7 segment positions */
.seg-a { top: 0; left: 10%; width: 80%; height: 10%; }
.seg-b { top: 10%; right: 0; width: 10%; height: 40%; }
/* ... etc for segments c-g */
```

---

## 🧩 Future Extensions
- Toggle switches and rocker buttons
- Multi-LED bar graph
- “PanelFrame” component (etched grouping box)
- Dark theme auto-detection
- Wiring editor prototype using ReactFlow

---

## ✅ Summary
This proposal creates a mid-complexity **LabVIEW-style UI layer for Gradio** that:
- Keeps Gradio’s composability.
- Adds a distinctive, industrial control-panel aesthetic.
- Provides reusable instrument-style widgets.
- Stays fully Python-installable and theme-compatible.

It’s a sweet spot between “simple styling” and “full custom GUI framework,” ideal for demos, teaching, or engineering dashboards.

---

## 📚 References and Resources

**Gradio Documentation:**
- [Custom Components in 5 Minutes](https://www.gradio.app/guides/custom-components-in-five-minutes)
- [Theming Guide](https://www.gradio.app/guides/theming-guide)
- [Backend Component API](https://www.gradio.app/guides/backend)

**Design Standards:**
- ISA-101: Human-Machine Interfaces for Process Automation
- High-Performance HMI principles (minimal color, gray backgrounds)

**LabVIEW Resources:**
- NI LabVIEW UI Design Guidelines
- JKI Flat UI Controls 2.0
- DMC LabVIEW UI Styles Toolkit

**Technical References:**
- SVG Path Syntax (for gauges and arcs)
- Svelte Tutorial (for component development)
- CSS radial gradients (for LED effects)
- `requestAnimationFrame` (for smooth interactions)