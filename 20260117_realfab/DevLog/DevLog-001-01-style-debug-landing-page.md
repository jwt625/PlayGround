# Design Specification: "Real Food" Landing Page & Intro Animation

## 1. Overview
This is a two-stage landing page experience. It begins with a **fullscreen "Curtain" Overlay** acting as a pre-loader or intro, which then transitions via a vertical slide-up effect to reveal the **Main Hero Section**. The aesthetic is clean, bold, and government-institutional but modernized with "Gen Z" design trends (brutalism/neo-grotesque typography).

## 2. Color Palette
* **Intro Background (The Green Curtain):** Deep Forest Green (Hex approximation: `#104f35`)
* **Main Website Background:** Warm Cream / Off-White (Hex approximation: `#f7f6ef`)
* **Primary Text:** Jet Black (Hex approximation: `#0a0a0a`)
* **Call-to-Action (CTA) Button:** Acid Green / Lime (Hex approximation: `#dcfd56`)
* **Accent Colors (Illustrations):**
    * *Milk Carton:* Red (`#c92a2a`) and White.
    * *Broccoli:* Vibrant Green (`#2b8a3e`).
    * *Steak:* Dark Red/Pink (`#e03131`) with white marbling.

## 3. Phase 1: Convergence Animation (00:00 - 01:02)
**State:** Fullscreen overlay (100vw, 100vh).
**Background:** Solid Deep Forest Green.
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
* The **Green Background Layer** (z-index: 100000) translates vertically upwards (`transform: translateY(-100%)`).
* **Easing:** Smooth ease-in-out (`cubic-bezier(0.65, 0, 0.35, 1)`).
* **Duration:** 1000ms
* **Simultaneous Action:** Items separate outward and move up during this phase (see Phase 1b above).

## 5. Phase 3: The Hero Section (00:01 - 00:02)
**State:** The static layer revealed underneath the green curtain.
**Background:** Solid Warm Cream.
**Layout & UI Elements:**

### A. Global Header (Top Bar)
* **Position:** Top center.
* **Content:** A tiny US Flag icon followed by text: "AN OFFICIAL WEBSITE OF THE UNITED STATES GOVERNMENT".
* **Style:** Uppercase, very small font size (approx 10-11px), wide letter-spacing (tracking), sans-serif.

### B. Hero Typography (Center)
* **Headline:** "Real Food Starts Here"
    * **Font Style:** Massive Sans-Serif (resembling *Helvetica Now Display* or *Inter*).
    * **Weight:** Extra Bold / Black (800-900).
    * **Tracking:** Tight (negative letter-spacing, e.g., `-0.04em`).
    * **Alignment:** Centered, stacked on two lines if on mobile, likely 2-3 lines on desktop.
* **Body Copy:**
    * **Content:** "Better health begins on your plate—not in your medicine cabinet. The new Dietary Guidelines for Americans defines real food as whole, nutrient-dense, and naturally occurring, placing them back at the center of our diets."
    * **Style:** Clean Sans-Serif. Medium weight. Dark Grey. Max-width constraint (approx 60ch) for readability. Center aligned.

### C. Call to Action (CTA)
* **Text:** "View the Guidelines"
    * **Button Shape:** Pill-shaped / Stadium border radius (fully rounded ends).
    * **Button Color:** Acid Green (`#dcfd56`).
    * **Text Color:** Black.
    * **Typography:** Bold, small-medium size.

### D. Hero Image (Bottom)
* **Content:** A photograph of greenery/garden plants.
* **Shape:** It appears to be a card or section with rounded top-left and top-right corners, rising from the bottom of the viewport or sitting just below the fold.

## 6. Technical Implementation Prompts for Agent
* "Use a fixed position `div` for the Green Intro Layer with `z-index: 100000`."
* "Implement the convergence animation using Framer Motion with sequential delays and smooth easing."
* "The Green Layer should animate `transform: translateY(-100%)` to reveal the `<main>` content underneath."
* "Ensure the font rendering for the headline is set to `antialiased` and use a negative letter-spacing to match the brutalist aesthetic."

## 7. Implementation Status

### Completed
* IntroCurtain component created with fullscreen deep forest green overlay
* Hero section updated with correct typography (font-weight: 900, letter-spacing: -0.04em)
* Title changed to "Real Fab Starts Here"
* CTA button styled with acid green background and black text
* Hero background color set to exact spec (#f7f6ef)
* Replaced placeholder elements with actual pyramid images (lpbf-metal-printer, silicon-boule, tpp-system)
* Animation timing configured to trigger after curtain reveal
* Sequential convergence animation implemented with proper delays
* Items stop 20% earlier during convergence phase
* Curtain slide-up animation implemented with correct easing
* Items positioned as fixed elements above curtain overlay

### Issues / Not Working
* **CRITICAL: Items not animating during exit phase**
    * Items remain stuck at their converged positions
    * No outward separation movement visible
    * Items disappear when curtain finishes sliding up instead of animating out
    * Root cause: AnimatePresence exit animations not triggering properly for items
    * Items are using `animate={phase}` to switch between 'converge' and 'exit' states
    * When `isVisible` becomes false, entire component unmounts via AnimatePresence
    * Items need to complete their exit animation before component unmounts

### Next Steps
* Debug why items are not animating to their exit state
* Verify that phase state change from 'converge' to 'exit' is triggering item animations
* Ensure items complete their outward+upward movement before component unmounts
* Consider using exit animations via AnimatePresence instead of phase-based animate prop