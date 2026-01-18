# Design Specification: "Real Food" Landing Page & Intro Animation

## 1. Overview
This is a two-stage landing page experience. It begins with a **fullscreen "Curtain" Overlay** acting as a pre-loader or intro, which then transitions via a vertical slide-up effect to reveal the **Main Hero Section**. The aesthetic is clean, bold, and government-institutional but modernized with "Gen Z" design trends (brutalism/neo-grotesque typography).

## 2. Color Palette
* **Intro Background (The Curtain):** Photolithography cleanroom yellow (`#ffcc00`)
* **Main Website Background:** Off-White (`#F3F0D6`)
* **Primary Text:** Off-Black (`#110000`)
* **Call-to-Action (CTA) Button:** Light yellow-green (`#f4ffae`)
* **Hero Description Text:** Brownish-gray (`#6b6159`)

## 3. Phase 1: Convergence Animation (00:00 - 01:02)
**State:** Fullscreen overlay (100vw, 100vh).
**Background:** Photolithography cleanroom yellow (`#ffcc00`).
**Elements:**
* **3D Sticker Assets:** Three pyramid fabrication assets (RealFab equivalent of food items):
    1.  **LPBF Metal Printer** (equivalent to Broccoli)
    2.  **Silicon Boule** (equivalent to Milk Carton)
    3.  **TPP System** (equivalent to Steak)

**Animation Sequence:**
* **Phase 1a: Convergence (0-1.2s)** - Items move toward center with smooth acceleration/deceleration
    * **Item 1 (LPBF Metal Printer):**
        * Starts: Just below top edge (5vh), already fully in view
        * Moves: DOWN toward center
        * Ends: Stops 20% earlier than center (-9vh)
        * Timing: Starts immediately at t=0, duration 600ms
    * **Item 2 (Silicon Boule):**
        * Starts: Bottom left (25% x position, 35vh - partially outside viewport)
        * Moves: UP-RIGHT toward center
        * Ends: Stops 20% earlier than center (-8vw, 9vh)
        * Timing: Starts at t=300ms (when Item 1 begins decelerating), duration 600ms
    * **Item 3 (TPP System):**
        * Starts: Bottom right (75% x position, 25vh)
        * Moves: LEFT toward center
        * Ends: Stops 20% earlier than center (8vw, 7vh)
        * Timing: Starts at t=600ms (when Item 2 begins decelerating), duration 600ms
    * **Overlap:** Items converge with slight (~10%) occlusion, not perfectly overlapping
    * **Easing:** Smooth acceleration/deceleration with flat middle segment - `cubic-bezier(0.45, 0, 0.55, 1)`
    * **No Rotation:** Items only translate, no rotation effects

* **Phase 1b: Separation & Exit (1.2s - 2.2s)** - Items move outward and upward simultaneously
    * All items move UP to -120vh (off screen)
    * Items separate outward to halfway back to original positions:
        * Item 1: Stays centered horizontally (0), moves up
        * Item 2: Moves LEFT/outward to -20vw while moving up
        * Item 3: Moves RIGHT/outward to 20vw while moving up
    * Duration: 1000ms
    * This happens simultaneously with curtain slide-up

## 4. Phase 2: Curtain Transition (01:02 - 02:02)
**Trigger:** After convergence animation completes (1.2s).
**Mechanism:** "Curtain Up" Reveal.
* The **Yellow Background Layer** (z-index: 100000) translates vertically upwards (`transform: translateY(-100%)`).
* **Easing:** Smooth ease-in-out (`cubic-bezier(0.65, 0, 0.35, 1)`).
* **Duration:** 1000ms
* **Simultaneous Action:** Items separate outward and move up during this phase (see Phase 1b above).

## 5. Phase 3: The Hero Section (00:01 - 00:02)
**State:** The static layer revealed underneath the yellow curtain.
**Background:** Off-White (`#F3F0D6`).
**Layout & UI Elements:**

### A. Global Header (Top Bar)
* **Position:** Top center.
* **Content:** TWM logo (1.5x size: 30px × 18px) followed by text: "AN OFFICIAL WEBSITE OF THE OUTSIDE FIVE SIGMA".
* **Style:** Uppercase, font-size 12px (10px mobile), monospace, letter-spacing 0.06em, padding 12px, border-bottom 0.5px.

### B. Hero Typography (Center)
* **Headline:** "Real Fab Starts Here"
    * **Font Size:** 170px desktop, 16.5vw mobile.
    * **Weight:** Bold (700).
    * **Line Height:** 0.84.
    * **Letter Spacing:** -0.02em.
    * **Alignment:** Centered, line break between "Fab" and "Starts Here" (forced no-wrap on second line).
* **Body Copy:**
    * **Content:** "Better technology begins with accessible fabrication—not gatekept mega-fabs. The new paradigm for semiconductor manufacturing defines real chips as locally-made, additive-manufactured, and democratically accessible, placing them back at the center of innovation."
    * **Style:** Font-weight 700 (bold), font-size 21px, line-height 1.3, color #6b6159, max-width 760px, text-wrap balance, center aligned.

### C. Call to Action (CTA)
* **Text:** "View the Guidelines"
    * **Button Shape:** Pill-shaped (border-radius 40px).
    * **Button Color:** Light yellow-green (`#f4ffae`).
    * **Text Color:** Off-black.
    * **Padding:** 18px 28px.
    * **Font:** 16px, line-height 1.5, font-weight 500.
    * **Hover:** Background changes to `#d4e767`.

## 6. Technical Implementation Prompts for Agent
* "Use a fixed position `div` for the Yellow Intro Layer with `z-index: 100000`."
* "Implement the convergence animation using Framer Motion with sequential delays and smooth easing."
* "The Yellow Layer should animate `transform: translateY(-100%)` to reveal the `<main>` content underneath."
* "Position animated items as independent fixed elements (z-index: 100001) above the curtain to prevent unmounting during exit."

## 7. Implementation Status

### Completed
* IntroCurtain component created with fullscreen photolithography yellow overlay (#ffcc00)
* Hero section updated with correct typography from reference
* Title: "Real Fab Starts Here" with line break, 170px desktop, 16.5vw mobile, line-height 0.84
* Description: Bold (700), 21px, line-height 1.3, color #6b6159, max-width 760px
* CTA button: Light yellow-green (#f4ffae), 18px 28px padding, 40px border-radius, 16px font
* Hero and banner background colors matched (#F3F0D6)
* Banner: TWM logo (30px × 18px), 12px font, 12px padding, 0.5px border
* Replaced placeholder elements with actual pyramid images (lpbf-metal-printer, silicon-boule, tpp-system)
* Animation timing configured to trigger after curtain reveal
* Sequential convergence animation implemented with proper delays
* Items stop 20% earlier during convergence phase
* Curtain slide-up animation implemented with correct easing
* Items positioned as fixed elements above curtain overlay
* Section component padding removed to eliminate gaps between sections
* Hero title second line forced to no-wrap ("Starts Here" stays together)

### Fixed
* **Exit animation now working correctly**
    * Root cause: Items were children of curtain overlay, got unmounted when curtain exited
    * Solution: Moved items outside curtain as independent fixed-position elements
    * Items now animate to exit positions (separate outward + move up) independently of curtain
    * Curtain slides up simultaneously without affecting item animations
* **Convergence order corrected**
    * Item 3 (bottom-right) now moves second with 0.3s delay
    * Item 2 (bottom-left) now moves third with 0.6s delay
* **Z-index layering fixed**
    * Later-moving items now appear on top of earlier-moving items
    * Item 1: z-index 1, Item 3: z-index 2, Item 2: z-index 3