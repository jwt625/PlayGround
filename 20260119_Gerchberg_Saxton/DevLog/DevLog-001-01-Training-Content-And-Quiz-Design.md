# DevLog-001-01: Training Content and Quiz Design

**Date:** 2026-01-20
**Author:** Wentao
**Status:** In Progress

---

## Progress Summary

**2026-01-20 (update 3)**

Frontend and generator improvements:
- Added modal popup for enlarged side-by-side GIF view in gallery
- Synchronized pause/play between phase and intensity GIFs in each card
- Added CLI options to generate_samples.py (--list, --samples for selective regeneration)
- Tuned sample parameters:
  - Coherent/Soft Aperture: start radius 2x smaller
  - Linear Ramp X/Y: 4x smaller end pitch, 2x longer (32 frames)
  - Linear Ramp Diagonal: 3x smaller pitch
  - Quadratic/Cubic Phase: 4x longer (64 frames)
- GIF pause-at-current-frame bug remains (deferred)

**2026-01-20 (update 2)**

L1 and L2 sample generation complete:
- Created slm_guessr Python package with pattern generators
- Implemented 16 L1 Foundations samples including aperture demos
- Implemented 8 L2 Periodic Structures samples
- Extended grating sweep range to 8px minimum period
- Added coherent aperture and soft aperture samples (spot size control via phase)
- Fixed GIF pause bug (continuous frame capture)
- Gallery loads and displays samples from manifest
- Total: 24 samples generated (48 GIFs)

**2026-01-20**

Frontend infrastructure complete:
- Initialized SvelteKit project with pnpm and TypeScript
- Configured static adapter for GitHub Pages deployment
- Implemented dark theme with sharp corners, monospace fonts
- Created Header, GifPlayer, SampleCard components
- Built Home, Gallery, and Quiz page structures
- Defined TypeScript types for samples and quiz state
- Build verified successful

---

## Design Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Content format | Pre-rendered GIF + PNG | Python generates, Svelte displays |
| GIF player | Play/pause toggle | Frame-by-frame scrubbing out of scope |
| Quiz directions | Both (phase->intensity, intensity->phase) | User can select mode |
| Wrong options | Same difficulty level | Near-misses for hard mode (future) |
| Quiz feedback | Show correct/incorrect + correct answer | No explanations in MVP |
| Difficulty selection | User picks upfront | Adaptive progression out of scope for MVP |
| Scoreboard | Local (per session) | Leaderboard requires backend (future) |
| Deployment | GitHub Pages | Static site |
| Future consideration | Browser/WebGPU FFT | Note for v2 |

---

## Training Sample Categories

### Level 1: Foundations (~20 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| Uniform phase | Static | Baseline reference |
| Single spot sweep X | GIF | Spot moves left to right, shows phase ramp tilt |
| Single spot sweep Y | GIF | Spot moves bottom to top |
| Single spot circular | GIF | Spot moves in circle, phase ramp rotates |
| Gaussian spot sweep | GIF | Soft spot moves, no sinc ringing |
| Rectangular slab sweep X | GIF | Rectangle moves horizontally, shows sinc envelope |
| Rectangular slab sweep Y | GIF | Rectangle moves vertically |
| Linear ramp X | GIF | kx: 0 to 2pi across grid |
| Linear ramp Y | GIF | ky: 0 to 2pi across grid |
| Linear ramp diagonal | GIF | Angle: 0 to 45 deg |
| Quadratic phase (positive) | GIF | Curvature: weak to strong |
| Quadratic phase (negative) | GIF | Curvature: weak to strong |
| Cubic phase X | GIF | Coefficient sweep |
| Cubic phase Y | GIF | Coefficient sweep |

### Level 2: Periodic Structures (~20 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| Binary grating vertical | GIF | Period: large to small |
| Binary grating horizontal | GIF | Period: large to small |
| Binary grating rotated | GIF | Angle: 0 to 90 deg |
| Sinusoidal grating | GIF | Period sweep |
| Blazed grating | GIF | Period sweep |
| Checkerboard | GIF | Period sweep |
| Crossed gratings | GIF | Relative angle |
| Multi-frequency grating | GIF | Frequency ratio |

### Level 3: Spot Arrays (~20 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| 2x2 uniform spots | GIF | Spot separation |
| 3x3 uniform spots | GIF | Spot separation |
| 4x4 uniform spots | GIF | Spot separation |
| Asymmetric spot array | Static | Fixed positions |
| Random spot positions | Static | Multiple random seeds |
| Weighted spots (varying brightness) | GIF | Weight distribution |
| Single off-center spot | GIF | Position sweep |

### Level 4: Special Beams (~20 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| Vortex l=1 | Static | Reference |
| Vortex l=1,2,3,4 | GIF | Charge increasing |
| Axicon | GIF | Slope sweep |
| Vortex + lens | GIF | Defocus sweep |
| Vortex + grating | GIF | Grating period |
| LG01, LG02, LG03 | GIF | Mode order |
| HG01, HG10, HG11 | Static | Mode comparison |
| Bessel beam | GIF | Ring radius |

### Level 5: Compound Patterns (~15 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| Grating + lens | GIF | Lens power |
| Dual gratings (orthogonal) | GIF | Relative strength |
| Vortex array (2x2) | Static | Fixed |
| Random phase (speckle) | GIF | Different seeds |
| Annular random phase | GIF | Annulus radius |

### Level 6: Practical Applications (~15 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| Tweezer rearrangement | GIF | Time evolution (reuse existing) |
| Defect-free array formation | GIF | Random to ordered |
| Gaussian to OAM conversion | GIF | Mode order |
| Beam steering | GIF | Angle sweep |
| Multi-plane focus | Static | Axial positions |

### Level 7: Shapes and Objects (~40 samples)

| Category | Examples | Count |
|----------|----------|-------|
| Letters | A-Z (subset: A,B,C,E,F,H,O,S,X) | 9 |
| Numbers | 0-9 | 10 |
| Animals | Cat, bird, fish, rabbit, dog | 5 |
| Objects | Cup, key, glasses, lightbulb, heart, star | 6 |
| Symbols | Arrow, music note, checkmark, cross | 4 |
| Geometric | Ring, triangle, hexagon, spiral | 4 |

All shapes rendered as binary silhouettes, GS-optimized phase masks.

---

## Total Sample Count

| Level | Count |
|-------|-------|
| L1 Foundations | 20 |
| L2 Periodic | 20 |
| L3 Spots | 20 |
| L4 Special Beams | 20 |
| L5 Compound | 15 |
| L6 Practical | 15 |
| L7 Shapes | 40 |
| **Total** | **150** |

---

## Quiz Mechanics

### Modes
- **Phase to Intensity**: Show phase GIF, pick intensity from 4 options
- **Intensity to Phase**: Show intensity GIF, pick phase from 4 options

### Difficulty Levels
- **Easy**: Levels 1-2 (foundations, periodic)
- **Medium**: Levels 3-5 (spots, beams, compound)
- **Hard**: Levels 6-7 (practical, shapes)

### Scoring
- Correct: +10 points
- Incorrect: +0 points
- Streak bonus: +5 per consecutive correct (cap at +25)
- Session high score stored in localStorage

### UI Flow
1. Select quiz mode (phase->intensity or intensity->phase)
2. Select difficulty (easy/medium/hard)
3. Present question with 4 options (GIF thumbnails)
4. User clicks answer
5. Show correct/incorrect, highlight correct option
6. Next question button
7. End: show final score, option to retry

---

## Data Format

### Manifest (samples.json)
```json
{
  "samples": [
    {
      "id": "linear_ramp_x",
      "level": 1,
      "category": "foundations",
      "name": "Linear Phase Ramp (X)",
      "description": "Phase gradient in X direction shifts beam",
      "phase_gif": "assets/L1/linear_ramp_x_phase.gif",
      "intensity_gif": "assets/L1/linear_ramp_x_intensity.gif",
      "parameters": {"kx_range": [0, "2pi"]}
    }
  ]
}
```

### Directory Structure
```
slm-guessr/
  public/
    assets/
      L1/, L2/, ... L7/    # GIFs organized by level
    samples.json           # Manifest
  src/
    lib/
      Gallery.svelte
      Quiz.svelte
      GifPlayer.svelte     # Play/pause component
      SampleCard.svelte
    routes/
      +page.svelte         # Landing
      gallery/+page.svelte
      quiz/+page.svelte
  static/
  package.json
```

---

## Implementation Steps

### Step 1: Python Content Generator
1. Create `slm_guessr/patterns.py` with all pattern generators
2. Create `slm_guessr/generator.py` to batch-generate all samples
3. Output: GIFs + `samples.json` manifest

### Step 2: Svelte Project Setup
1. Initialize SvelteKit project with pnpm
2. Configure for static adapter (GitHub Pages)
3. Create basic routing structure

### Step 3: Gallery Mode
1. Implement GifPlayer component with play/pause
2. Implement SampleCard component
3. Implement Gallery page with level/category filtering

### Step 4: Quiz Mode
1. Implement quiz state machine
2. Implement question display with 4 options
3. Implement scoring and feedback
4. Add localStorage for high scores

### Step 5: Polish and Deploy
1. Styling and responsiveness
2. GitHub Actions for deployment
3. Testing and bug fixes

---

## Style Guidelines

| Element | Specification |
|---------|---------------|
| Theme | Dark |
| Corners | Sharp (no border-radius) |
| Typography | Monospace for data, sans-serif for UI |
| Colors | Neutral grays, accent color for interactive elements |
| Icons | Clean minimal SVG only, no emoji anywhere |
| Layout | Minimal, technical, high information density |
| Spacing | Consistent, tight but readable |

### Color Palette
```
--bg-primary: #0a0a0a
--bg-secondary: #141414
--bg-tertiary: #1e1e1e
--text-primary: #e0e0e0
--text-secondary: #888888
--accent: #4a9eff
--accent-hover: #6bb3ff
--success: #4ade80
--error: #f87171
--border: #2a2a2a
```

### Component Style
- Buttons: solid background, no rounded corners, subtle hover state
- Cards: bordered, no shadow, sharp edges
- GIF player: minimal controls, progress bar underneath
- Quiz options: grid layout, clear hover/selected states

---

## Implementation Progress

| Task | Status |
|------|--------|
| Svelte project setup (pnpm, SvelteKit) | Complete |
| Static adapter configuration (GitHub Pages) | Complete |
| Dark theme CSS with style guidelines | Complete |
| Header component with navigation | Complete |
| Home page with mode selection | Complete |
| Gallery page structure (level sidebar) | Complete |
| Quiz page structure (mode/difficulty selection) | Complete |
| GifPlayer component (play/pause with frame capture) | Complete |
| SampleCard component | Complete |
| TypeScript types | Complete |
| Python pattern generators (L1, L2) | Complete |
| Sample generation pipeline | Complete |
| Gallery data integration | Complete |
| L3-L7 pattern generators | Not started |
| Quiz game logic | Not started |

---

## Next Steps

1. Implement L3 Spot Arrays generators
2. Implement L4 Special Beams generators
3. Implement L5-L7 generators
4. Implement Quiz game logic with scoring
5. Deploy to GitHub Pages

