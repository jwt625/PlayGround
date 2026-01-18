# Design Specification: "State of Our Health" Scrolling Section

## 1. Global Visual Style & Palette

* **Background Color:** Deep, warm black (Hex approx: `#0A0505` or `#0F0A0A`). It is not pure `#000000`; it has a very subtle reddish/brown undertone.
* **Primary Accent Color (Red):** Vivid Vermilion/Orange-Red (Hex approx: `#D62718` or `#E33224`). Used for the geometric blocks and text highlighting.
* **Text Color (Primary):** White `#FFFFFF`.
* **Text Color (Secondary/Label):** Grey `#888888`.
* **Text Color (On Red Backgrounds):** Black `#000000` (for the large percentages).

## 2. Typography

* **Font Family:** A neo-grotesque sans-serif (e.g., *Helvetica Now*, *Inter*, *SF Pro Display*, or *Roboto*).
* **Headline Weight:** Heavy/Bold (700 or 800).
* **Body Weight:** Medium/Regular (400 or 500).
* **Tracking (Letter-spacing):** Tight tracking on large headlines (approx `-0.02em` to `-0.04em`).

## 3. Layout Structure (The "Stage")

The section functions as a sticky container where the visual elements evolve as the user scrolls.

* **Top Navigation Badge (Sticky):**
    * **Position:** Top Center (fixed or sticky).
    * **Shape:** Pill/Capsule shape (`border-radius: 50px`).
    * **Background:** Dark Grey/Brown (approx `#2A2020`).
    * **Content:**
        * A small bullet point (`•`).
        * Text: "State Of Our Health" (Font size: ~12-14px, White).
        * Pagination Dots: 4 dots to the right. The active dot is White; inactive dots are Grey.
* **Content Grid:**
    * The view is divided into a 2-column layout (conceptually).
    * **Left Column (Text):** aligned vertically center, with a left margin of approx 10-15% of the viewport width.
    * **Right Column (Graphics):** Anchored to the bottom-right corner of the viewport.

## 4. The Four States (Scroll Sequence)

### State 1: Introduction (The Hook)
* **Text Element (Left):**
    * **Label:** "The State of Our Health" (Small, Grey, Uppercase/Capitalized).
    * **Headline:** "America is sick.<br>The data is clear."
        * **Size:** Massive (approx `5rem` to `7rem` / `80px` to `112px`).
        * **Color:** White.
        * **Alignment:** Left-aligned.
* **Graphic Element (Right):** Empty/Invisible. The background is purely dark.

### State 2: 50% Statistic
* **Text Element (Left):**
    * **Label:** "The State of Our Health"
    * **Headline:** "50% of Americans have **prediabetes or diabetes**"
        * "50% of Americans have": White.
        * "prediabetes or diabetes": Highlighted in **Red Accent Color**.
        * **Size:** Slightly smaller than State 1 to accommodate longer text (approx `4rem` / `64px`).
* **Graphic Element (Right - Bottom):**
    * **Shape:** A square block anchored to the bottom-right corner.
    * **Color:** Red Accent.
    * **Dimensions:** Approx `400px` x `400px` (or 35% of viewport height).
    * **Content:** The number "50%" inside the block.
        * **Position:** Top-left of the red block.
        * **Font:** Black, Bold, Large (approx `120px`).

### State 3: 75% Statistic
* **Text Element (Left):**
    * **Headline:** "75% of adults report having at least one **chronic condition**"
        * "chronic condition": Highlighted in **Red Accent Color**.
* **Graphic Element (Right - Bottom):**
    * **Transition:** The red graphic expands upwards and leftwards. It looks like a new, larger square has grown behind the 50% square.
    * **Visual Stack:** The 50% block stays visible in the bottom right corner (opacity might lower slightly or stay solid). A new "L-shape" or larger square backing appears.
    * **Content:** The number "75%" appears in the newly revealed space (top-left of the new larger mass).
    * **Alignment:** The "75%" text is aligned significantly higher than the "50%" text.
    * **Color Match:** The 50% text in the corner remains visible.

### State 4: 90% Statistic
* **Text Element (Left):**
    * **Headline:** "90% of U.S. healthcare spending goes to treating **chronic disease**—much of which is linked to diet and lifestyle"
        * "chronic disease": Highlighted in **Red Accent Color**.
        * **Size:** Smaller font size (approx `3rem` / `48px`) to fit the paragraph.
* **Graphic Element (Right - Bottom):**
    * **Transition:** The red graphic expands again to its largest state.
    * **Visual Stack:**
        * Largest Block (Back layer): Contains "90%" text in the top-left area.
        * Middle Block (Middle layer): Contains "75%" text.
        * Smallest Block (Front layer/Bottom-Right): Contains "50%" text.
    * **Composition:** This forms a "step" chart visual, looking like nested squares radiating from the bottom right corner.
    * **Dimensions:** The red mass now takes up approx 60-70% of the screen width and 80% of height on the right side.

## 5. Animation & Implementation Notes for Coding Agent

* **CSS Grid/Flexbox:** Use a sticky container for the background and right-hand graphics. The left-hand text can scroll normally or cross-fade using opacity triggers.
* **Z-Index Layering:**
    * The "50%" block has the highest z-index (closest to viewer).
    * The "75%" block is behind it.
    * The "90%" block is at the back.
* **Motion:** As the user scrolls down:
    1.  Text elements on the left cross-fade (Opactiy 0 -> 1 -> 0) and translate slightly Y-axis.
    2.  The Red Blocks typically scale up from `scale(0)` or translate in from the corner.
    3.  Alternatively, the container size (`width`/`height`) of the red blocks animates from `0` to target size.

---

# Implementation Plan - RealFab Adaptation

## Section Overview
Adapt the "State of Our Health" reference design to create **"State of Our Fabs"** section for the RealFab website, showcasing semiconductor industry statistics with the same visual language and animation patterns.

## Persistent Section Title
- **Text:** "State of Our Fabs"
- **Position:** Top-left corner of the section, stays visible throughout all animations
- **Font size:** Match reference implementation (likely 1.5-2rem)
- **Font weight:** 600-700
- **Letter spacing:** -0.02em
- **Color:** White (on dark background)
- **Behavior:** Does not fade out during state transitions

## Content - 4 Statistics

### State 1: Introduction
- **Heading:** "Our industry is broken.<br>The data is clear."
- **Font size:** Large display heading (5-7rem range, match reference)
- **Color:** White
- **Alignment:** Left-aligned
- **Graphics:** None (dark background only)

### State 2: Monopoly Statistic
- **Text:** "**3** companies control 90%+ of advanced chip manufacturing"
- **Highlighted phrase:** "advanced chip manufacturing" (in red accent color)
- **Font size:** ~4rem (slightly smaller to accommodate text length)
- **Red Block:**
  - Shape: Square (~400px × 400px)
  - Position: Bottom-right corner
  - Color: Red accent (#D62718 or #E33224)
  - Number: "**3**" at top-left corner of block
  - Number font size: `clamp(44px, 8vw, 120px)`
  - Number color: Black (#000000)
  - Number animation: Rolling/counting from 0 to 3

### State 3: Complexity Statistic
- **Text:** "**500+** process steps in modern SOTA semiconductor fabrication"
- **Highlighted phrase:** "semiconductor fabrication" (in red accent color)
- **Font size:** ~4rem
- **Red Block:**
  - Shape: Larger rectangle/L-shape (expands from State 2)
  - Dimensions: ~400px × 600px (70% width of graphics container)
  - Position: Extends upward and leftward from State 2 block
  - Color: Slightly different red hue (variation of accent color)
  - Number: "**500+**" at top-left corner of new expanded area
  - Number animation: Rolling/counting from 0 to 500+
  - **Previous block remains visible:** "3" block stays in bottom-right

### State 4: Waste or Cost Statistic
**Option A (Material Waste):**
- **Text:** "**99%** of materials wasted in subtractive processes"
- **Number:** "99%"

**Option B (Cost):**
- **Text:** "**$20B+** cost to build a single leading-edge fab"
- **Number:** "$20B+"

- **Highlighted phrase:** "subtractive processes" or "leading-edge fab" (in red accent color)
- **Font size:** ~3rem (smaller to fit longer text)
- **Red Block:**
  - Shape: Largest stepped/layered graphic
  - Dimensions: ~500px × 700px (85% width of graphics container)
  - Position: Further expansion upward and leftward
  - Color: Third red hue variation
  - Number: "**99%**" or "**$20B+**" at top-left corner
  - Number animation: Rolling/counting
  - **All previous blocks remain visible:** Stepped chart effect with 3 layers

## Layout Structure

### Container Specifications
- **Sticky/Pinned Container:**
  - `position: sticky`
  - `top: 0`
  - `height: 100vh`
  - Main canvas does not scroll when section is entered
  - Scroll triggers animation state changes only

- **Scroll Container:**
  - `min-height: 400vh` (creates scroll distance for 4 states)
  - Each state occupies ~100vh of scroll distance

### 2-Column Grid Layout
- **Left Column (48% width):**
  - Section title "State of Our Fabs" (persistent, top-left)
  - State-specific text content
  - Sticky positioning at ~20% from top
  - Vertical center alignment for text content
  - Left margin: 10-15% of viewport width

- **Right Column (52% width):**
  - Red geometric blocks
  - Bottom-right anchored
  - Absolute positioning for layered blocks
  - Z-index stacking for depth

## Animation Specifications

### Scroll Progress Mapping
- **State 1:** scrollYProgress 0.0 - 0.25 (Introduction)
- **State 2:** scrollYProgress 0.25 - 0.50 (3 companies)
- **State 3:** scrollYProgress 0.50 - 0.75 (500+ steps)
- **State 4:** scrollYProgress 0.75 - 1.0 (99% waste or $20B+ cost)

### Text Transitions
- **Fade out:** opacity 1 → 0 (over ~0.1 scroll progress)
- **Fade in:** opacity 0 → 1 (over ~0.1 scroll progress)
- **Overlap:** Slight crossfade with 0.05 scroll progress overlap
- **Y-axis translation:** Optional slight upward movement on exit
- **Section title:** Never fades out, remains at 100% opacity

### Red Block Animations
- **Entry direction:** Slide from bottom-right corner
- **Movement:** Up and toward top-left
- **Transform:** `translateX(-Xpx) translateY(-Ypx)` or scale from 0
- **Timing:** Each block animates over ~0.15 scroll progress
- **Stacking:** Previous blocks remain visible with z-index layering
- **Z-index order:**
  - State 2 block (3): z-index 30 (front)
  - State 3 block (500+): z-index 20 (middle)
  - State 4 block (99%/$20B+): z-index 10 (back)

### Number Animations
- **Position:** Top-left corner of each red block (padding: ~20-30px)
- **Rolling effect:** Count from 0 to target value using `useSpring`
- **Spring config:** `{ stiffness: 50, damping: 30 }` for smooth counting
- **Font:** Tabular numbers (`font-variant-numeric: tabular-nums`)
- **Size:** `clamp(44px, 8vw, 120px)`
- **Weight:** 700-900 (bold/black)
- **Color:** Black (#000000) for contrast on red background

### Red Block Sizing (Progressive Growth)
- **State 2 (3 companies):**
  - Width: 400px (55% of graphics container)
  - Height: 400px
  - Shape: Square

- **State 3 (500+ steps):**
  - Width: 400px (70% of graphics container)
  - Height: 600px
  - Shape: L-shape or expanded rectangle

- **State 4 (99% waste or $20B+ cost):**
  - Width: 500px (85% of graphics container)
  - Height: 700px
  - Shape: Stepped chart (all 3 blocks visible)

### Red Color Palette Variations
- **Block 1 (State 2):** `#D62718` (base red)
- **Block 2 (State 3):** `#E33224` (slightly lighter/more orange)
- **Block 3 (State 4):** `#C41E1A` (slightly darker/deeper)
- All blocks remain visible in final state with distinct hues

## Typography Specifications

### Section Title ("State of Our Fabs")
- **Font size:** 1.5-2rem (match reference)
- **Font weight:** 600-700
- **Letter spacing:** -0.02em
- **Color:** White (#FFFFFF)
- **Position:** Top-left, persistent
- **Text transform:** None (title case)

### State Headings
- **Font size:**
  - State 1: 5-7rem (desktop), use `clamp(3rem, 8vw, 7rem)`
  - State 2-3: 4rem (desktop), use `clamp(2.5rem, 6vw, 4rem)`
  - State 4: 3rem (desktop), use `clamp(2rem, 5vw, 3rem)`
- **Font weight:** 700-800
- **Line height:** 1.0-1.1
- **Letter spacing:** -0.03em to -0.04em
- **Color:** White (#FFFFFF)
- **Highlighted phrases:** Red accent color (#D62718)

### Statistic Numbers (on red blocks)
- **Font size:** `clamp(44px, 8vw, 120px)`
- **Font weight:** 700-900
- **Font variant:** Tabular numbers
- **Color:** Black (#000000)
- **Position:** Top-left corner of block with padding

## Background & Colors

- **Background:** Deep warm black (`#0A0505` or `#0F0A0A`)
- **Text color (primary):** White (#FFFFFF)
- **Text color (highlighted):** Red accent (#D62718 or #E33224)
- **Number color:** Black (#000000) on red blocks
- **Red accent variations:** #D62718, #E33224, #C41E1A

## Technical Implementation Notes

### Framer Motion Setup
```typescript
const containerRef = useRef<HTMLDivElement>(null);
const { scrollYProgress } = useScroll({
  target: containerRef,
  offset: ["start start", "end end"]
});
```

### State Transitions
- Use `useTransform` to map scrollYProgress to opacity values
- Create separate motion values for each text block
- Example: `const state1Opacity = useTransform(scrollYProgress, [0, 0.2, 0.25], [1, 1, 0])`

### Block Positioning
- Use `useTransform` for translateX/translateY values
- Example: `const block1Y = useTransform(scrollYProgress, [0.25, 0.4], [100, 0])`

### Number Counting
- Use `useSpring` with `useTransform` for smooth counting
- Extract numeric value from string (handle "$", "+", "%" suffixes)
- Format output with proper separators and suffixes

### Container Structure
```tsx
<div style={{ minHeight: '400vh' }}>
  <div style={{ position: 'sticky', top: 0, height: '100vh' }}>
    {/* Section title (persistent) */}
    {/* Left column text (animated) */}
    {/* Right column graphics (animated) */}
  </div>
</div>
```

## Reference Alignment Checklist

- [x] Section title "The State of Our Fabs" persistent at top-left (grey #888888)
- [x] 5-state scroll sequence with sticky container
- [x] Main canvas does not scroll (sticky positioning)
- [x] Text fades out/in on state transitions (non-overlapping, with vertical movement)
- [x] Typography matches reference (font size, weight, spacing)
- [x] Dark background (#0A0505) with white text
- [x] Highlighted phrases in red accent color
- [x] Consistent font sizes across all text states
- [x] Text positioned ~15vh from top with proper spacing
- [x] Red blocks slide from bottom-right (diagonal entrance with x and y transforms)
- [x] Numbers positioned at top-left of blocks
- [x] Rolling number animations with spring physics (useSpring with useTransform)
- [x] Progressive block sizing (400x400, 500x570, 600x740, 700x910)
- [x] Layered z-index stacking for blocks (30, 20, 10, 5)
- [x] Red color palette with 4 variations (#D62718, #E33224, #C41E1A, #B01810)
- [x] Tooltip on "99%" with source note
- [x] Text content takes left 75% width
- [ ] Exit animation - NEEDS WORK
- [ ] Mobile responsive with adjusted font sizes

## Implementation Progress

### Completed
- Sticky scroll container (400vh) with 5 states
- Section title "The State of Our Fabs" in grey (#888888)
- 5 statistics (updated "500+" to "1,000+"):
  1. "Our industry is broken. The data is clear."
  2. "3 companies control 90%+ of advanced chip manufacturing"
  3. "1,000+ process steps in modern SOTA semiconductor fabrication"
  4. "99% of materials wasted in subtractive processes"
  5. "$20B+ cost to build a single leading-edge fab"
- Non-overlapping text transitions with vertical movement (fade in moves up, fade out moves down)
- Dark background (#0A0505) with white text and red highlights (#D62718)
- 2-column grid layout (48% text left, 52% graphics right)
- Text content width limited to 75%
- Red blocks with diagonal slide-in animation (x and y transforms from 100 to 0)
- Spring-based number counting animations
- Tooltip on "99%" claim: "Ratio of total chemical/gas input mass to final chip mass. Different processing steps can vary."
- Tooltip styling: dark background, drop shadow, border, proper hover area

### In Progress
- Exit animation sequence:
  - Buffer zone added (0.88-0.94) so viewer stays on state 5 longer
  - Text fades out first (0.94-0.95)
  - Numbers fade out (0.95-0.96)
  - Blocks should expand to fill canvas (0.96-1.0), back to front with stagger
  - Z-index elevation for blocks during exit (above text container)
  - Current issue: blocks not expanding correctly to cover full canvas; expansion logic needs rework (should move top-left corners toward canvas origin while growing width/height)