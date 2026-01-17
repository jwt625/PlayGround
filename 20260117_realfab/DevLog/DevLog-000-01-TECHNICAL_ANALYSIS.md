# RealFab.org Technical Analysis

Based on analysis of realfood.gov (reference site)

## Framework & Stack

### Core Technologies
- **Framework**: Next.js 14+ (App Router with React Server Components)
- **UI Library**: React 18+
- **Animation**: Framer Motion (motion/react package)
- **Smooth Scroll**: Lenis (via LenisProvider)
- **Styling**: CSS Modules with CSS custom properties

### Key Animation Libraries Used
1. **Framer Motion** (`motion/react`)
   - `useScroll` - track scroll position
   - `useTransform` - map scroll to animation values
   - `useSpring` - physics-based spring animations
   - `motion.div` / `motion.span` - animated components
   - `AnimatePresence` - exit animations

2. **Lenis** - Smooth scroll library
   - Provides buttery-smooth scrolling
   - `scrollTo` method for programmatic scrolling

## Animation Patterns Identified

### 1. Scroll-Triggered Reveals
```javascript
const { scrollYProgress } = useScroll({
  target: sectionRef,
  offset: ["start end", "end start"]
});
```

### 2. Parallax Effects
- Elements move at different rates based on scroll
- Uses `useTransform` to map scroll progress to Y position

### 3. Number Counter Animations
- Statistics animate from 0 to target value
- Triggered when section enters viewport

### 4. Text Disintegration Effect
- Characters animate individually based on scroll
- Each char has staggered animation timing

### 5. Spring-Based UI Elements
- Navigation dots expand/collapse with spring physics
- Cards scale on hover with spring transition

### 6. Intro Animation Sequence
- Overlay with food items animating in
- Uses motion.div with initial/animate/exit states

## CSS Animation Techniques

### Keyframes Used
- `springUp` / `springDown` - Easter egg animations
- `spin` - Loading spinner
- `swipe-out-*` - Toast dismissal

### Transitions
- `cubic-bezier(0.34, 1.56, 0.64, 1)` - Overshoot bounce
- `cubic-bezier(0.23, 1, 0.32, 1)` - Smooth ease-out
- `cubic-bezier(0.165, 0.84, 0.44, 1)` - Ease-out-quart

### Performance Optimizations
- `will-change: transform, opacity` on animated elements
- `transform: translateZ(0)` for GPU acceleration
- `contain: layout style paint` for isolation

## Color Palette

```css
--off-white: #F3F0D6;     /* Cream background */
--off-black: #110000;     /* Near-black text */
--dark-green: #153f15;    /* Accent green */
--highlight: #f4ffae;     /* Highlight yellow-green */
--red4: var(--red4);      /* Statistics highlight */
--sand: (beige tone);     /* Section backgrounds */
```

## Typography

- **Primary Font**: Die Grotesk (or similar neo-grotesque)
- **Weights**: Bold (700), Medium (500), Regular (400)
- **Sizes**: Fluid typography using clamp()
- **Line Heights**: 100%, 120%, 140%

## Content Structure

1. **Hero** - Bold statement + video background
2. **Problem/Stats** - Scroll-triggered statistics
3. **Solution** - Text disintegration reveal
4. **Interactive Pyramid** - 3D food pyramid visualization
5. **Detail Cards** - Fan-out card animations
6. **FAQs** - Accordion with smooth expand
7. **Resources** - Stacked paper card effect
8. **Footer** - Minimal attribution

## Interactive Elements

1. **Sticky Navigation** with scroll-aware highlighting
2. **Minimap** pyramid navigation
3. **Lightbox** for images
4. **Video Player Modal** with blur background
5. **Accordion FAQs** with icon rotation
6. **Hover Card Effects** with spring physics

---

## Detailed Motion Specifications

### Spring Animation Configurations (Framer Motion)

| Component | Stiffness | Damping | Mass | Use Case |
|-----------|-----------|---------|------|----------|
| Navigation dots | 120 | 20 | - | Expand/collapse labels |
| Sticky header/content | 200 | 30 | - | Scroll-linked transforms |
| Hamburger menu | 300 | 30 | - | Menu icon morphing |
| Pyramid food items | 300 | 30 | 0.5 | Rearrange animations |
| Card hover | 400 | 25 | - | Scale on hover |
| Triangle intro | 400 | 40 | 1 | Scale reveal |
| Shimmer effect | 140 | 30 | 1 | Slow shimmer |
| Label animations | 400-500 | 32-40 | 1 | Text reveals |
| High-speed items | 800-1200 | 45-50 | 0.8-1 | Fast transitions |

### Duration-Based Animations

| Element | Duration | Easing | Delay |
|---------|----------|--------|-------|
| Hero content reveal | 0.8s | `[0.25, 1, 0.5, 1]` | 0-0.2s |
| Intro overlay | 0.8s | `[0.9, 0, 0, 0.9]` | - |
| Food items (mobile) | 0.5s | `[0.9, 0, 0, 0.9]` | staggered |
| Food items (desktop) | 0.7s | custom | staggered |
| Nav label fade | 0.3s | linear | 0.2s |
| Menu items stagger | 0.15s | linear | `0.03 * index` |
| Video modal | 0.25s | easeOut | 0.15s |
| Lenis scroll | 1.2s | - | - |

### CSS Cubic-Bezier Easing Curves

| Name | Value | Effect |
|------|-------|--------|
| Overshoot bounce | `(0.34, 1.56, 0.64, 1)` | Bouncy spring |
| Smooth ease-out | `(0.23, 1, 0.32, 1)` | Fast start, gentle end |
| Ease-out-quart | `(0.165, 0.84, 0.44, 1)` | Sharp deceleration |
| Snappy spring | `(0.28, 1.08, 0.4, 1)` | Slight overshoot |
| Soft settle | `(0.19, 1, 0.22, 1)` | Very soft landing |
| Bouncy icon | `(0.68, 0.05, 0.265, 1.55)` | Icon bounce |
| Exit curve | `(0.36, 0, 0.66, -0.56)` | Reverse spring |

### Scroll-Linked Transform Patterns

```javascript
// Navigation header parallax
inputRange: [0, 50, 150]
outputRange: [-100, -100, 0]  // Y position in pixels

// Text disintegration character animation
const charProgress = scrollProgress * (1 + charIndex / totalChars * 0.12)
// Maps to: scale, opacity, blur, y position per character

// Food pyramid rearrangement timing
inputRange: [te(row, 2.81), te(row, 3.04)]  // te = row-based timing function
outputRange: [0, 1]  // rearrange progress normalized

// Helper function for pyramid timing
const te = (row, offset) => (row - 1 + offset) / 12.4;
```

### Card/Carousel Motion Variants

```javascript
// Position variants for 3-card carousel
const positionVariants = {
  left:   { x: "-64%", scale: 0.67, backgroundColor: "rgb(245,245,245)" },
  center: { x: "0%",   scale: 1,    backgroundColor: "rgb(255,255,255)" },
  right:  { x: "64%",  scale: 0.67, backgroundColor: "rgb(245,245,245)" }
};

// Opacity variants
const opacityVariants = {
  left:   { opacity: 0.8 },
  center: { opacity: 1 },
  right:  { opacity: 0.8 }
};

// Card position transition
{ type: "spring", stiffness: 250, damping: 30 }

// Card text transition
{ type: "spring", stiffness: 150, damping: 16 }
```

### CSS Keyframe Animations

```css
/* Spring up animation (easter egg reveal) */
@keyframes springUp {
  0%  { transform: translateY(100%); opacity: 0; }
  50% { opacity: 1; }
  to  { transform: translateY(25%); opacity: 1; }
}
/* timing: 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) */

/* Spring down animation (exit) */
@keyframes springDown {
  0%  { transform: translateY(25%); opacity: 1; }
  to  { transform: translateY(100%); opacity: 0; }
}
/* timing: 0.3s cubic-bezier(0.36, 0, 0.66, -0.56) */

/* Loading spinner */
@keyframes spin {
  to { transform: rotate(1turn); }
}
/* timing: 0.8s linear infinite */
```

### GPU Acceleration Patterns

```css
/* Applied to all animated elements */
.animated-element {
  will-change: transform, opacity;
  transform: translateZ(0);
  -webkit-transform: translateZ(0);
}

/* For complex compositing */
.complex-animation {
  contain: layout style paint;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
}
```

### Intro Animation Sequence

```javascript
// Food items initial/animate states
const broccoli = {
  initial: { y: 600, opacity: 0.95 },
  animate: { y: 0, opacity: 1 },
  transition: { duration: 0.7, ease: [0.9, 0, 0, 0.9] }
};

const milk = {
  initial: { x: 400, y: 300, opacity: 0.95 },
  animate: { x: 0, y: 0, opacity: 1 },
  transition: {
    x: { duration: 0.7, delay: 0.05 },
    y: { duration: 0.7, delay: 0.1 },
    opacity: { duration: 0, delay: 0.25 }
  }
};

const steak = {
  initial: { x: -400, y: "90%", opacity: 0.95 },
  animate: { x: 0, y: 0, opacity: 1 },
  transition: {
    x: { duration: 0.7, delay: 0.1 },
    y: { duration: 0.7, delay: 0.15 },
    opacity: { duration: 0, delay: 0.25 }
  }
};

// Overlay fade
const overlay = {
  initial: { opacity: 1 },
  animate: { opacity: 0 },
  transition: { duration: 0.8 }
};
```

---

## Raw Animation Data from DevTools Console

Captured via `document.getAnimations()` while interacting with the site.

### Parsed Animation Summary

| # | Target Class | Duration | Delay | Easing | Property | From | To |
|---|--------------|----------|-------|--------|----------|------|-----|
| 0 | dga_h2_food | 700ms | 0 | spring | opacity | 1 | 0 |
| 1 | nav_label | 300ms | 0 | ease-out | opacity | 1 | 0 |
| 2 | nav_dot | 650ms | 0 | spring | opacity | 0 | 1 |
| 3 | nav_label | 300ms | 0 | ease-out | opacity | 0.65 | 0 |
| 4 | nav_dot | 650ms | 0 | spring | opacity | 0.076 | 1 |
| 5 | nav_label | 300ms | 200ms | ease-out | opacity | 0 | 1 |
| 6 | nav_dot | 650ms | 0 | spring | opacity | 1 | 0 |
| 7 | nav_label | 300ms | 0 | ease-out | opacity | 1 | 0 |
| 8 | nav_dot | 650ms | 0 | spring | opacity | 0 | 1 |
| 9 | (card) | 250ms | 0 | ease-out | opacity | 0.8 | 0 |
| 10 | dga_h2_food | 700ms | 0 | spring | opacity | 1 | 0 |
| 11 | (card) | 250ms | 0 | ease-out | filter | saturate(1.32) | saturate(1) |
| 12 | mobile-nav_label | 200ms | 0 | ease-out | opacity | 0 | 1 |

### Spring Easing Curve Analysis

The `linear()` function contains pre-calculated spring physics. Key characteristics:

```
Peak overshoot: 1.0664 at 47.8% of duration (6.6% overshoot)
Undershoot: 0.9959 at 92.7%
Final settle: 1.0 at 94.2%
Total keyframe steps: 69
```

This spring curve approximates:
```javascript
{ type: "spring", stiffness: 120, damping: 18, mass: 1 }
```

### Full Spring Easing (Copy-Paste Ready)

Two distinct spring curves were captured via `document.getAnimations()`:

**Spring A (700ms, with overshoot) - Food Title Transitions**
```css
linear(
  0 0%, 0.0073 1.45%, 0.0276 2.9%, 0.0588 4.35%, 0.0986 5.8%,
  0.1454 7.25%, 0.1972 8.7%, 0.2528 10.14%, 0.3106 11.59%,
  0.3696 13.04%, 0.4287 14.49%, 0.4871 15.94%, 0.5441 17.39%,
  0.599 18.84%, 0.6514 20.29%, 0.7009 21.74%, 0.7472 23.19%,
  0.7902 24.64%, 0.8297 26.09%, 0.8658 27.54%, 0.8983 28.99%,
  0.9274 30.43%, 0.9532 31.88%, 0.9758 33.33%, 0.9954 34.78%,
  1.012 36.23%, 1.0261 37.68%, 1.0376 39.13%, 1.0469 40.58%,
  1.0541 42.03%, 1.0595 43.48%, 1.0632 44.93%, 1.0655 46.38%,
  1.0664 47.83%,  /* PEAK OVERSHOOT: 6.64% */
  1.0663 49.28%, 1.0653 50.72%, 1.0635 52.17%, 1.0611 53.62%,
  1.0581 55.07%, 1.0548 56.52%, 1.0512 57.97%, 1.0474 59.42%,
  1.0435 60.87%, 1.0395 62.32%, 1.0356 63.77%, 1.0318 65.22%,
  1.0281 66.67%, 1.0245 68.12%, 1.0212 69.57%, 1.018 71.01%,
  1.0151 72.46%, 1.0123 73.91%, 1.0098 75.36%, 1.0076 76.81%,
  1.0056 78.26%, 1.0038 79.71%, 1.0022 81.16%, 1.0008 82.61%,
  0.9996 84.06%, 0.9986 85.51%, 0.9978 86.96%, 0.9971 88.41%,
  0.9966 89.86%, 0.9962 91.3%, 0.9959 92.75%,  /* SLIGHT UNDERSHOOT */
  1 94.2%, 1 95.65%, 1 97.1%, 1 98.55%, 1 100%
)
```
Framer Motion equivalent: `{ type: "spring", stiffness: 150, damping: 16 }`

**Spring B (650ms, no overshoot) - Navigation Dot**
```css
linear(
  0 0%, 0.0058 1.56%, 0.0216 3.13%, 0.0455 4.69%, 0.0757 6.25%,
  0.1107 7.81%, 0.1491 9.38%, 0.1901 10.94%, 0.2326 12.5%,
  0.2758 14.06%, 0.3192 15.63%, 0.3623 17.19%, 0.4045 18.75%,
  0.4457 20.31%, 0.4855 21.88%, 0.5237 23.44%, 0.5603 25%,
  0.595 26.56%, 0.6279 28.13%, 0.6589 29.69%, 0.688 31.25%,
  0.7153 32.81%, 0.7407 34.38%, 0.7644 35.94%, 0.7864 37.5%,
  0.8067 39.06%, 0.8254 40.63%, 0.8426 42.19%, 0.8585 43.75%,
  0.873 45.31%, 0.8862 46.88%, 0.8983 48.44%, 0.9094 50%,
  0.9194 51.56%, 0.9284 53.13%, 0.9366 54.69%, 0.944 56.25%,
  0.9507 57.81%, 0.9567 59.38%, 0.9621 60.94%, 0.9669 62.5%,
  0.9712 64.06%, 0.975 65.63%, 0.9784 67.19%, 0.9814 68.75%,
  0.9841 70.31%, 0.9865 71.88%, 0.9885 73.44%, 0.9904 75%,
  0.992 76.56%, 0.9934 78.13%, 0.9946 79.69%, 0.9956 81.25%,
  0.9965 82.81%, 0.9973 84.38%, 0.998 85.94%, 0.9985 87.5%,
  0.999 89.06%, 0.9994 90.63%, 0.9997 92.19%, 1 93.75%,
  1.0002 95.31%, 1 96.88%, 1 98.44%, 1 100%
)
```
Framer Motion equivalent: `{ type: "spring", stiffness: 120, damping: 20 }`

### Captured Animation Summary Table

| Target Class | Duration | Delay | Easing | Property | From | To |
|--------------|----------|-------|--------|----------|------|-----|
| dga_h2_food | 700ms | 0 | spring A | opacity | 1 | 0 |
| nav_label | 300ms | 0 | ease-out | opacity | 1 | 0 |
| nav_dot | 650ms | 0 | spring B | opacity | 0 | 1 |
| nav_label | 300ms | 200ms | ease-out | opacity | 0 | 1 |
| (card image) | 250ms | 0 | ease-out | opacity | 0.8 | 0 |
| (card image) | 250ms | 0 | ease-out | filter | saturate(1.32) | saturate(1) |
| mobile-nav_label | 200ms | 0 | ease-out | opacity | 0 | 1 |

### Limitations of Console Capture

The `document.getAnimations()` method only captures:
- Currently running Web Animations API animations
- Framer Motion spring animations (converted to linear() steps)

It does NOT capture:
- Scroll-linked transforms (updated per-frame via `useScroll`/`useTransform`)
- CSS transitions (only CSS keyframe animations)
- Completed animations
- Hover/focus state changes

To capture scroll-linked animations, use the Animation Panel in DevTools while scrolling,
or add logging to the `useScroll` callbacks in the source code.


## Video Analysis

Prompt:
```markdown
I'm reverse-engineering the scroll animations and motion design of this website (realfood.gov) to recreate similar effects. This video shows a complete scroll-through of the page.

Analyze the video frame-by-frame and describe ALL motion behaviors you observe. For each animated element or section, provide:

1. ELEMENT IDENTIFICATION
   - What type of element (text, image, navigation, background, etc.)
   - Approximate position on screen when animation triggers
   - Any identifying text or visual characteristics

2. TRIGGER CONDITIONS
   - When does the animation start? (on load, on scroll into view, on scroll past a threshold, on hover, etc.)
   - What scroll position or viewport percentage triggers it?
   - Does it animate once or continuously while scrolling?

3. MOTION PROPERTIES
   - Direction of movement (up, down, left, right, scale, rotate, fade)
   - Estimated distance traveled (in approximate pixels or % of viewport)
   - Duration (estimate in milliseconds or seconds)
   - Easing curve description (linear, ease-in, ease-out, spring/bounce, overshoot, etc.)
   - Any delays before animation starts

4. SCROLL-LINKED BEHAVIORS
   - Does the element move proportionally with scroll (parallax)?
   - What's the parallax ratio? (moves slower than scroll = 0.5x, faster = 1.5x, etc.)
   - Does it have sticky behavior? At what point does it stick/unstick?
   - Are there any "scrubbed" animations that progress with scroll position?

5. SPECIAL EFFECTS
   - Text disintegration/character-by-character effects
   - Blur, saturation, or filter changes
   - Staggered animations (multiple items animating in sequence)
   - Any morphing, masking, or clipping effects

6. SECTION-BY-SECTION BREAKDOWN
   For each major section of the page (Hero, Stats, Problem, Solution, Pyramid, FAQ, Footer), describe:
   - Entry animation when scrolling into view
   - Exit animation when scrolling out
   - Any internal scroll-linked behaviors while section is in view

7. NAVIGATION BEHAVIOR
   - How does the side navigation respond to scroll?
   - Dot/label transitions
   - Active state changes

Please be as specific and technical as possible. Use timestamps from the video when referencing specific moments. Format output as structured notes I can use as a reference spec for implementation.
```

## Reply from Gemini:

```markdown
# Motion Design Specification: RealFood.gov Reverse Engineering
**Target:** realfood.gov
**Analysis Source:** Screen Recording 2026-01-17 at 14.59.01.mov
**Tech Stack Inference:** WebGL or Canvas likely not required; achievable with GSAP (ScrollTrigger, Flip), Lenis (smooth scroll), and CSS Grid/Flexbox.

---

## 1. Global Motion Systems

### Smooth Scrolling
* **Behavior:** Inertial scrolling with heavy damping. Scrolling stops gradually after the user releases input.
* **Technical Spec:** Custom scroll container (likely `Lenis` or `Locomotive Scroll`).
* **Properties:** `lerp: 0.1` (approximate smoothing factor).

### Navigation Pill (Sticky Header)
* **Element:** Floating pill `div` at `top: 24px` centered horizontally.
* **Behavior:** Permanently fixed (Sticky).
* **Internal Animation:**
    * **Text:** Cross-fade or vertical slide (`translateY`) when entering new sections.
    * **Dots:** Active state indicator expands (`width: 6px` -> `24px`) with `ease-out` transition (~300ms).
    * **Trigger:** `IntersectionObserver` crossing section thresholds.

---

## 2. Section-by-Section Breakdown

### A. Hero Section (00:00 - 00:07)
**Concept:** Depth & Introduction
* **Floating Elements (Broccoli, Milk, Meat, etc.):**
    * **Trigger:** `window.onload` (continuous) + Scroll (parallax).
    * **Idle Animation:** `y: +/- 15px`, `rotation: +/- 5deg`, `duration: 3s`, `repeat: -1`, `yoyo: true`, `ease: sine.inOut`.
    * **Scroll Behavior (Parallax):**
        * Foreground items (Broccoli): `speed: 1.5` (Moves up faster than scroll).
        * Midground (Milk): `speed: 1.2`.
        * Background (Text "Real Food..."): `speed: 0.8` (Moves slower, creating depth).
* **Hero Video Container:**
    * **Initial State:** `scale: 0.9`, `border-radius: 40px`.
    * **Scroll Trigger:** Top of container hits 80% viewport height.
    * **Animation:** Scale to `1.0`, `border-radius` reduces to `20px` (or `0px` if full bleed).
    * **Easing:** `power2.out`.

### B. "America is Sick" Stats (00:07 - 00:25)
**Concept:** The Dark Mode Switch & Card Deck
* **Global Transition:**
    * **Background:** Interpolates from `#FDFBF7` (Cream) to `#1A0505` (Black/Dark Brown).
    * **Trigger:** Triggered when the "Stats" section hits 50% viewport.
* **Left Text Column ("America is sick...", "50% of Americans..."):**
    * **Behavior:** Sticky (Pinned).
    * **Motion:** Text swaps using `opacity` and `y-axis` slide.
        * Exit: `opacity: 0`, `y: -20px`.
        * Enter: `opacity: 1`, `y: 0`.
        * Sync: Ttied directly to the index of the visible card on the right.
* **Right Graphics Column (Red Cards):**
    * **Structure:** Stacked cards using `position: absolute`.
    * **Trigger:** Scroll Scrub (Scrubbing).
    * **Motion:** "Card Decking" effect.
        * Base Card: Static.
        * Overlay Cards (50%, 75%, 90%): Translate Y from `100vh` to `0` (stacking on top).
        * **Z-Index:** Managed incrementally (1, 2, 3, 4).
    * **Color Blocks:** Each card represents a percentage width or height, filling the screen progressively.

### C. The Problem (00:25 - 00:36)
**Concept:** Deconstruction & Clarity
* **1992 Pyramid Image:**
    * **Trigger:** Scroll Exit (Scrub).
    * **Motion:** `scale: 1` -> `0.8`, `opacity: 1` -> `0`.
* **"We can solve this crisis":**
    * **Trigger:** Center Viewport.
    * **Effect:** CSS Filter Blur Reveal.
        * Start: `filter: blur(20px)`, `opacity: 0`.
        * End: `filter: blur(0px)`, `opacity: 1`.
        * Duration: 1s.
* **"For the first time..." Paragraph:**
    * **Effect:** Scroll-scrubbed Text Highlighter.
    * **Implementation:** Split text into words/spans.
    * **Initial State:** `opacity: 0.2` (grayed out).
    * **Active State:** `opacity: 1.0` (white).
    * **Trigger:** As scroll position passes each line's Y-coordinate.

### D. The New Pyramid (00:36 - 01:00)
**Concept:** Re-assembly (The "Explosion")
* **Background:** Smooth transition back to `#FDFBF7` (Cream).
* **Title "The New Pyramid":**
    * **Start:** Center screen, Large (`5rem`).
    * **Scroll Action:** Shrinks (`scale: 0.6`) and moves to sticky header position (`top: 100px`).
* **Ingredients (The Core Effect):**
    * **Initial State:** Elements scattered radially outside the viewport (off-screen top, left, right, bottom).
    * **Trigger:** Pinned Section (Duration ~2000px scroll height).
    * **Motion (Scrubbed):**
        * Elements translate from `(x_random, y_random)` to `(0, 0)` (their final grid position in the pyramid).
        * **Stagger:** Groups arrive in sequence.
            * 0-30% Scroll: Protein (Meat/Fish) flies in from Left/Top.
            * 30-60% Scroll: Veggies fly in from Right.
            * 60-90% Scroll: Grains/Carbs fly in from Bottom.
    * **Sidebar Text:** Contextual text fades in/out on the left side corresponding to the active group (Protein -> Veggies -> Grains).

### E. Resources & Footer (01:10 - End)
**Concept:** Utility & Finale
* **Resources Cards:**
    * **Trigger:** Scroll Scrub.
    * **Effect:** Fan / Spread.
        * Origin: `transform-origin: bottom center`.
        * Left Card: `rotate: -10deg`, `x: -50px`.
        * Center Card: `y: -20px`.
        * Right Card: `rotate: 10deg`, `x: 50px`.
* **Footer Carousel (Slot Machine):**
    * **Element:** "Eat Real [Bread/Lettuce/Potatoes]".
    * **Trigger:** Autoplay (Loop) or Scroll Trigger.
    * **Motion:**
        * Text: Vertical slide (`y: -100%`). Masked overflow.
        * Image: Spring scale effect (`scale: 0.8` -> `1.0`) synced with text change.

---

## 3. Implementation Cheat Sheet

| Section | Key GSAP Method | CSS Property Focus |
| :--- | :--- | :--- |
| **Hero** | `ScrollTrigger` (scrub: true) | `transform: translate3d` |
| **Stats** | `ScrollTrigger` (pin: true) | `z-index`, `clip-path` (optional) |
| **Problem** | `SplitText` (or manual spans) | `filter: blur()`, `opacity` |
| **Pyramid** | `Timeline` synced to Scroll | `position: absolute`, `left/top` |
| **Cards** | `ScrollTrigger` (scrub: 1) | `transform: rotate()` |

```