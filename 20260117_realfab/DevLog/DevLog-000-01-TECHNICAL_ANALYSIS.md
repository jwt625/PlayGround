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
# Motion Design Specification: RealFood.gov (Revised)
**Target:** realfood.gov
**Analysis Source:** Screen Recording 2026-01-17 at 14.59.01.mov + Source Code Inspection
**Tech Stack:** Framer Motion (`motion/react`), Lenis (Smooth Scroll)

---

## 1. Global Motion Systems & Physics

### Smooth Scrolling (Lenis)
* **Behavior:** Inertial scrolling with heavy damping.
* **Implementation:** Lenis instance wrapping the main content.
* **Interactions:** `useScroll` hooks link animation progress to Lenis scroll position.

### Physics & Easing (Framer Motion)
Instead of standard CSS easings, the site relies heavily on spring physics for natural motion.
* **Spring A (Overshoot):** Used for food title transitions and entrance effects.
    * **Behavior:** Fast attack with distinct overshoot (peak 6.64%) and slight settling undershoot.
    * **Config:** `{ type: "spring", stiffness: 150, damping: 16 }`
    * **Duration:** ~700ms effective.
* **Spring B (Smooth Deceleration):** Used for navigation dots and UI elements.
    * **Behavior:** Smooth braking, no bounce.
    * **Config:** `{ type: "spring", stiffness: 120, damping: 20 }`
    * **Duration:** ~650ms effective.

### GPU Acceleration Patterns
Performance optimization is aggressive to handle heavy paint costs.
* **Properties:** `will-change: transform, opacity` applied to all animating elements.
* **Compositing:** `transform: translateZ(0)` forces layer promotion.
* **Containment:** `contain: layout style paint` used on static containers to prevent reflow propagation.
* **Visibility:** `backface-visibility: hidden` used on rotating cards.

---

## 2. Component Breakdown

### A. Navigation (Sticky Pill)
* **Container:** Fixed position `top: 24px`.
* **Scroll Offsets:**
    * Desktop: `600px` trigger points.
    * Mobile: `-100px` or `400px` (variable per section).
* **Dot Animation:**
    * **Motion:** `opacity` and `width` expansion.
    * **Timing:** Uses **Spring B** (650ms).
* **Label Animation:**
    * **Desktop:** 300ms `ease-out`, often with 200ms stagger delay.
    * **Mobile:** 200ms `ease-out` (snappier response).

### B. Hero Section (00:00 - 00:07)
* **Tech:** `useScroll({ offset: [...] })` mapped via `useTransform`.
* **Floating Elements (Parallax):**
    * **Implementation:** `motion.img` elements with `y` values mapped to `scrollYProgress`.
    * **Depth:** Foreground elements move faster than background (standard parallax).
* **Reveal:** Triggered at specific `scrollYProgress` thresholds.
* **Optimization:** `will-change: transform` active during scroll.

### C. "America is Sick" Stats (00:07 - 00:25)
* **Structure:** `position: sticky` container for the left text column.
* **Card Animation:**
    * **Transition:** `y` translation from `100vh` to `0`.
    * **Physics:** Linear mapping to scroll (scrubbing), but strictly clamped.
* **Card Hover State:**
    * **Scale:** `whileHover` uses Spring (`stiffness: 400, damping: 25`).
    * **Filter:** `filter: saturate(1)` → `saturate(1.32)` (250ms ease-out).
    * **Opacity:** `0.8` → `1` (250ms ease-out).

### D. The Problem (00:25 - 00:36)
* **Text Disintegration Effect:**
    * **Target:** "For the first time..." paragraph.
    * **Granularity:** **Character-level**. Each char is wrapped in a `motion.span` with class `disintegrating_char`.
    * **CSS:** `display: inline-block`, `transform-origin: center center`.
    * **Stagger Formula:**
        `charProgress = scrollProgress * (1 + charIndex / totalChars * 0.12)`
    * This formula ensures a wave-like propagation of opacity/blur across the text block rather than a flat linear fade.

### E. The New Pyramid (00:36 - 01:00)
* **Exit/Entry:** `AnimatePresence` manages the DOM mounting/unmounting of ingredients.
* **Motion Variants:**
    * **Hidden:** `{ opacity: 0, scale: 0.8, x: [off-screen-coordinate] }`
    * **Visible:** `{ opacity: 1, scale: 1, x: 0 }`
    * **Transition:** **Spring A** (Overshoot) for the "pop" effect as ingredients settle into the pyramid.
* **Labels:** Side labels sync with the ingredient animation completion using `onLayoutAnimationComplete` callbacks or synchronized delays.

---

## 3. Revised Implementation Cheat Sheet

| Section | Framer Motion Method | Critical CSS / Props |
| :--- | :--- | :--- |
| **Hero** | `useScroll({ offset: ["start start", "end start"] })` + `useTransform` | `will-change: transform`, `translateZ(0)` |
| **Stats** | `useScroll` (container ref), `motion.div` | `position: sticky`, `top: 20%` |
| **Problem** | `useScroll`, `useTransform` mapped per char index | `display: inline-block`, `transform-origin: center` |
| **Pyramid** | `AnimatePresence`, `motion.div` variants | `layout` prop (for smooth reflows), **Spring A** |
| **Cards** | `useTransform` (rotation/y), `whileHover` | `filter: saturate()`, `backface-visibility: hidden` |
| **Nav** | `useSpring` (linked to active index) | `opacity`, `width`, **Spring B** |
```

---

## 4. Implementation Reference

### Framer Motion Code Patterns

**Scroll-linked parallax:**
```javascript
const { scrollYProgress } = useScroll({ target: containerRef, offset: ["start end", "end start"] });
const y = useTransform(scrollYProgress, [0, 1], [100, -100]);
return <motion.div style={{ y }} />;
```

**Spring transition config:**
```javascript
const springA = { type: "spring", stiffness: 150, damping: 16 }; // overshoot
const springB = { type: "spring", stiffness: 120, damping: 20 }; // smooth
const springHover = { type: "spring", stiffness: 400, damping: 25 }; // snappy
```

**Character-level text animation:**
```javascript
const chars = text.split("");
return chars.map((char, i) => {
  const progress = useTransform(scrollYProgress, [0, 1], [0, 1 + (i / chars.length) * 0.12]);
  const opacity = useTransform(progress, [0, 0.5, 1], [0.2, 0.6, 1]);
  return <motion.span style={{ opacity, display: "inline-block" }}>{char}</motion.span>;
});
```

**Card hover state:**
```javascript
<motion.div
  whileHover={{ scale: 1.02, filter: "saturate(1.32)" }}
  transition={springHover}
/>
```

### Missing Section Details

**Blur Reveal (Problem Section):**
- Target: "We can solve this crisis" heading
- Start: `filter: blur(20px)`, `opacity: 0`
- End: `filter: blur(0px)`, `opacity: 1`
- Duration: ~1000ms, triggered at center viewport

**Footer Carousel (Slot Machine):**
- Text cycles: "Eat Real [Bread/Lettuce/Potatoes/...]"
- Vertical slide: `y: -100%` with masked overflow
- Image: Spring scale `0.8` to `1.0` synced with text change
- Autoplay interval: ~3000ms

**Resources Cards (Fan Effect):**
- Transform origin: `bottom center`
- Left card: `rotate: -10deg`, `x: -50px`
- Center card: `y: -20px`
- Right card: `rotate: 10deg`, `x: 50px`
- Trigger: Scroll scrub into view

### Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| Cream | `#FDFBF7` | Light mode background |
| Dark Brown | `#1A0505` | Stats section background |
| Red Accent | `#E53935` | Stats cards, highlights |
| White | `#FFFFFF` | Text on dark |
| Gray | `rgba(255,255,255,0.2)` | Inactive text |

### Typography

- Headings: Serif family (appears custom or licensed)
- Body: Sans-serif, ~18px base
- Hero title: ~5rem desktop, scales down mobile
- Stats numbers: Bold, large scale (~8rem)

### Breakpoints

| Name | Width | Notes |
|------|-------|-------|
| Mobile | < 768px | Single column, reduced parallax |
| Tablet | 768-1024px | Two column where applicable |
| Desktop | > 1024px | Full layout, all effects active |

Mobile adjustments:
- Parallax intensity reduced (~50%)
- Navigation label animations faster (200ms vs 300ms)
- Scroll offsets adjusted per section

### Z-Index Stack

| Layer | Z-Index | Elements |
|-------|---------|----------|
| Navigation | 100 | Sticky header pill |
| Modal/Overlay | 90 | Mobile menu |
| Cards (stacking) | 1-10 | Stats cards increment |
| Content | 1 | Default |
| Background | -1 | Parallax backgrounds |

### Performance Checklist

- [ ] Apply `will-change: transform, opacity` to animated elements
- [ ] Use `transform: translateZ(0)` for GPU compositing
- [ ] Add `contain: layout style paint` to static containers
- [ ] Ensure `backface-visibility: hidden` on 3D transforms
- [ ] Lazy load images below fold
- [ ] Debounce resize handlers
- [ ] Use passive scroll listeners via Lenis

### Dependencies

```json
{
  "framer-motion": "^11.x",
  "lenis": "^1.x",
  "next": "^14.x",
  "react": "^18.x"
}
```

### File Structure Suggestion

```
src/
  app/
    layout.tsx          # Lenis provider, global styles
    page.tsx            # Main page composition
  components/
    Navigation/         # Sticky pill nav
    Hero/               # Parallax hero
    Stats/              # Dark mode stats cards
    Problem/            # Text reveal, blur effects
    Pyramid/            # Ingredient assembly
    Resources/          # Fan cards
    Footer/             # Slot machine carousel
  hooks/
    useScrollProgress.ts
    useParallax.ts
  lib/
    springs.ts          # Spring configs
    animations.ts       # Shared variants
  styles/
    globals.css
    variables.css
```
