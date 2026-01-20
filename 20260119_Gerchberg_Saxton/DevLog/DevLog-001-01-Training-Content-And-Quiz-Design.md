# DevLog-001-01: Training Content and Quiz Design

**Date:** 2026-01-20  
**Author:** Wentao  
**Status:** Planning

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

### Level 1: Foundations (~15 samples)

| Sample | Animation | Parameters Swept |
|--------|-----------|------------------|
| Uniform phase | Static | Baseline reference |
| Linear ramp X | GIF | kx: 0 to 2pi across grid |
| Linear ramp Y | GIF | ky: 0 to 2pi across grid |
| Linear ramp diagonal | GIF | Angle: 0 to 45 deg |
| Quadratic phase (positive) | GIF | Curvature: weak to strong |
| Quadratic phase (negative) | GIF | Curvature: weak to strong |
| Cubic phase | GIF | Coefficient sweep |

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
| L1 Foundations | 15 |
| L2 Periodic | 20 |
| L3 Spots | 20 |
| L4 Special Beams | 20 |
| L5 Compound | 15 |
| L6 Practical | 15 |
| L7 Shapes | 40 |
| **Total** | **145** |

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

## Next Steps

1. Create `slm_guessr/` Python package structure
2. Implement pattern generators (Level 1-2 first)
3. Generate sample batch for proof of concept
4. Initialize Svelte project

