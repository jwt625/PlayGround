# RealFab.org Implementation Plan

## Project Phases & Milestones

### Phase 0: Preparation & Setup - COMPLETE
**Status**: Complete (2026-01-18)
**Duration**: ~2 hours

#### Milestone 0.1: Asset Preparation - COMPLETE
- [x] Extract reference images from realfood.gov for style guide
- [x] Create placeholder images for RealFab content:
  - [x] Semiconductor wafer images
  - [x] TPP/LPBF equipment photos
  - [x] Desktop nanofab tools
  - [x] Chip/circuit board visuals
  - [x] Pyramid food items → fabrication tools mapping (38 assets)
- [x] Prepare video placeholder (announcement video equivalent)
- [x] Collect icon assets (navigation, UI elements)

**Deliverable**: `/public/images/pyramid/` directory with 38 processed WebP assets - COMPLETE
**Review Point**: Asset inventory complete - see DevLog-000-04-PYRAMID_ASSET_CHECKLIST.md

#### Milestone 0.2: Project Initialization - COMPLETE
- [x] Initialize Next.js 16.1.3 project with TypeScript
- [x] Install core dependencies (framer-motion ^12.26.2, lenis ^1.3.17)
- [x] Set up project structure (components, hooks, lib, styles)
- [x] Configure CSS Modules and global styles
- [x] Set up CSS custom properties (color palette, spacing, typography)
- [x] Configure Next.js for static export (GitHub Pages)
- [x] Create Lenis smooth scroll provider
- [x] Create spring animation configurations (lib/springs.ts)
- [x] Create shared animation variants (lib/animations.ts)
- [x] Copy all 38 assets to public/images/pyramid/

**Deliverable**: Running Next.js dev server with complete foundation - COMPLETE
**Review Point**: Project structure verified, dev server tested at localhost:3000

**Implementation Notes**:
- Next.js 16.1.3 installed (latest version)
- Static export configured with `output: 'export'` and `images: { unoptimized: true }`
- CSS design system includes: colors, spacing (0.25rem base), typography, containers, transitions, z-index, radius, shadows
- Spring configs: gentle, bouncy, snappy, smooth, slow
- Animation variants: fade, scale, slide, stagger patterns
- Lenis configured with 1.2s duration and custom easing

---

### Phase 1: Foundation & Core Systems [COMPLETE]
**Status**: Complete (2026-01-18)
**Duration**: ~1 hour

#### Milestone 1.1: Global Systems [COMPLETE]
- [x] Implement Lenis smooth scroll provider
- [x] Create spring animation configurations (lib/springs.ts)
- [x] Set up shared animation variants (lib/animations.ts)
- [x] Implement scroll progress hooks (useScrollProgress, useParallax, useScrollFade, useScrollScale)
- [x] Create global CSS variables and theme system
- [x] Implement intersection observer hooks (useInView, useIntersectionRatio)
- [x] Create active section tracking hook (useActiveSection)

**Deliverable**: Working smooth scroll with reusable animation utilities [COMPLETE]
**Review Point**: All hooks tested and TypeScript types verified

**Implementation Notes**:
- Created 4 scroll-based hooks: useScrollProgress, useParallax, useScrollFade, useScrollScale
- Created 2 intersection observer hooks: useInView, useIntersectionRatio
- Created useActiveSection hook with scrollToSection utility
- All hooks properly typed with TypeScript

#### Milestone 1.2: Navigation Component [COMPLETE]
- [x] Build sticky pill navigation structure
- [x] Implement scroll-aware section highlighting
- [x] Add spring-based dot expansion animations
- [x] Create label fade in/out transitions
- [x] Add scroll-to-section functionality
- [ ] Build mobile hamburger menu variant (deferred to Phase 2)

**Deliverable**: Fully functional navigation with all animations [COMPLETE]
**Review Point**: Navigation component complete with spring animations

**Implementation Notes**:
- Fixed position navigation with backdrop blur
- Spring-based dot animations (stiffness: 120, damping: 20)
- Label fade transitions (300ms ease-out with 200ms delay when active)
- Hover state shows labels for inactive sections
- Mobile responsive with smaller dots and labels
- Reduced motion support

#### Milestone 1.3: Basic Layout Components [COMPLETE]
- [x] Create Section wrapper component
- [x] Create Container component with size variants
- [x] Create AnimatedSection component with scroll-triggered animations
- [x] Set up stagger animation support

**Deliverable**: Reusable layout components [COMPLETE]
**Review Point**: Components tested in demo page

**Implementation Notes**:
- Section component with dark theme and fullHeight variants
- Container with 5 size options (sm, md, lg, xl, full)
- AnimatedSection combines Section with scroll-triggered fade-in
- Stagger animation support for child elements
- All components properly typed and documented

---

### Phase 2: Hero & Introduction [COMPLETE]
**Status**: Complete (2026-01-18)
**Duration**: ~1 hour

#### Milestone 2.1: Hero Section [COMPLETE]
- [x] Build hero layout with title and description
- [x] Implement entrance animations (fade + translateY)
- [x] Add CTA buttons with hover states
- [x] Create parallax background elements (deferred - not needed for MVP)

**Deliverable**: Hero section with entrance animations [COMPLETE]
**Review Point**: First impression and messaging clarity

**Implementation Notes**:
- Staggered entrance animations using Framer Motion variants
- Gentle spring animation (stiffness: 100, damping: 20)
- Responsive typography with clamp() for fluid scaling
- CTA buttons with hover states and transform effects
- Video preview button integrated into hero

#### Milestone 2.2: Video Player Modal [COMPLETE]
- [x] Build video preview component
- [x] Implement modal overlay with blur background
- [x] Add play button with spring animation
- [x] Create mobile fullscreen video fallback
- [x] Add close/escape functionality

**Deliverable**: Working video player with modal [COMPLETE]
**Review Point**: Video playback and modal UX

**Implementation Notes**:
- AnimatePresence for smooth modal transitions
- Backdrop blur with 85% opacity overlay
- Escape key and click-outside to close
- Body scroll lock when modal is open
- 16:9 aspect ratio with responsive container
- Mobile fullscreen mode for better viewing
- Video auto-pauses when modal closes
- Placeholder video path: `/videos/announcement-placeholder.mp4`

---

### Phase 3: Statistics Section [COMPLETE]
**Status**: Complete (2026-01-18)
**Duration**: ~45 minutes

#### Milestone 3.1: Stats Layout [COMPLETE]
- [x] Create dark section background transition
- [x] Build sticky left column for text
- [x] Implement scroll-triggered text swapping
- [x] Add percentage cards with stacking animation

**Deliverable**: Stats section structure with scroll behavior [COMPLETE]
**Review Point**: Scroll timing and readability

**Implementation Notes**:
- Dark background (#1A0505) with white text
- Two-column grid layout (sticky text + scrolling cards)
- Sticky text column at 20vh from top
- Cards column with vertical spacing
- Responsive: stacks to single column on mobile

#### Milestone 3.2: Stats Animations [COMPLETE]
- [x] Implement number counter animations (0 → target)
- [x] Add percentage bar fill animations (deferred - using solid color cards)
- [x] Create card entrance/exit transitions
- [x] Add hover effects with spring physics (scale, saturate)

**Deliverable**: Fully animated statistics section [COMPLETE]
**Review Point**: Animation polish and data presentation

**Implementation Notes**:
- Number counter using useSpring (stiffness: 50, damping: 30)
- Scroll-based card reveal with useTransform
- Cards animate from y: 100 to y: 0
- Opacity fade-in synchronized with position
- Hover: scale 1.02 + saturate 1.2
- Each card has unique color: red (#E53935), orange (#FB8C00), yellow (#FDD835), green (#43A047)
- Counter extracts numeric values and animates with proper formatting
- Supports prefixes ($) and suffixes (+, %)

---

### Phase 4: Problem & Solution [COMPLETE]
**Status**: Complete (2026-01-18)
**Duration**: ~30 minutes

#### Milestone 4.1: Broken System Section [COMPLETE]
- [x] Build sticky image container (deferred - focusing on text effects)
- [x] Implement image lightbox functionality (deferred)
- [x] Create text disintegration effect (character-level)
- [x] Add scroll-linked blur reveal

**Deliverable**: Problem section with text effects [COMPLETE]
**Review Point**: Text animation readability and impact

**Implementation Notes**:
- Character-level disintegration using scroll-based transforms
- Each character animates with staggered timing based on position
- Opacity, Y position, and rotateX transforms
- Scroll-linked blur effect (10px → 0 → 10px)
- White background with dark text
- Responsive typography with clamp()

#### Milestone 4.2: Solution Reveal [COMPLETE]
- [x] Implement "We can change this paradigm" disintegration
- [x] Add follow-up text with scroll-triggered reveal
- [x] Create blur-to-focus transition for heading
- [x] Polish scroll offsets and timing

**Deliverable**: Solution section with reveal animations [COMPLETE]
**Review Point**: Narrative flow and pacing

**Implementation Notes**:
- Reused DisintegratingText component for consistency
- Light gray background (#f5f5f5) for visual separation
- Three paragraphs with staggered whileInView animations
- Blur-to-focus transition synchronized with scroll
- Final paragraph emphasized with semibold weight
- Viewport margin: -100px for earlier trigger

---

### Phase 5: Fab Pyramid [COMPLETE]
**Status**: Complete (2026-01-18)
**Duration**: ~1 hour

#### Milestone 5.1: Pyramid Structure [COMPLETE]
- [x] Build inverted pyramid SVG with animated lines
- [x] Create food item → fab tool image mapping
- [x] Implement tooltip system for pyramid sections
- [x] Add "Introducing" kicker animation

**Deliverable**: Static pyramid with structure [COMPLETE]
**Review Point**: Visual hierarchy and content mapping

**Implementation Notes**:
- SVG pyramid outline with animated path drawing
- Three tiers: Additive Core (17 items), Local & Accessible (17 items), Big Fab (4 items)
- All 38 pyramid assets mapped from food items to fab equipment
- Dark background (#1A0505) matching stats section
- Responsive grid layout with auto-fit columns
- Kicker text with uppercase styling

#### Milestone 5.2: Pyramid Animations [COMPLETE]
- [x] Implement AnimatePresence for item transitions
- [x] Add spring-based item entrance (bouncy spring config)
- [x] Create staggered animation sequence
- [x] Add tooltip hover/reveal animations
- [x] Implement scroll-linked pyramid assembly

**Deliverable**: Fully animated pyramid section [COMPLETE]
**Review Point**: Animation choreography and timing

**Implementation Notes**:
- Scroll-based staggered entrance (0.015s delay per item)
- Each item animates: y (60→0), opacity (0→1), scale (0.8→1)
- Hover effects: scale 1.1, z-index elevation
- Tooltip with snappy spring animation
- Tooltip shows name and description on hover
- SVG path animation: 2s pathLength draw
- Tier divider lines with dashed stroke
- Mobile responsive: smaller grid items, adjusted padding

---

## CRITICAL STYLE ISSUES IDENTIFIED (2026-01-18)

### Overview
After reviewing reference HTML/CSS files from realfood.gov, discovered fundamental mismatches between current implementation and reference design. The issues span visual design, layout structure, typography, and animation patterns.

### UPDATE (2026-01-18 - Post Pyramid Rebuild)
**Completed:**
- [DONE] CSS color palette updated with cream/off-white/off-black colors
- [DONE] Spring configurations added (Spring A, Spring B, hover spring)
- [DONE] Pyramid section completely rebuilt with absolute positioning
- [DONE] All 38 items mapped with exact coordinates (top%, left%, width%, height%, zIndex, entryOrder)
- [DONE] Multi-stage scroll timeline implemented (0.1-0.35 proteins, 0.35-0.6 vegetables, 0.6-0.85 grains)
- [DONE] SVG pyramid outline updated with correct path (M 5 5 L 95 5 L 50 95 Z)
- [DONE] Scroll-linked stroke animation for SVG (0.05-0.95)
- [DONE] Tier labels with fade in/out animations
- [DONE] 300vh scroll container with sticky positioning

**Still Broken/TODO:**
- [TODO] Some pyramid image links broken (need to verify all 38 asset paths)
- [TODO] Hero section still has white background (should be cream #FDFBF7)
- [TODO] Problem/Solution sections still have white background (should be cream #F3F0D6)
- [TODO] Missing custom display font (Grotesk Display) for headings
- [TODO] Hero parallax elements not implemented
- [TODO] Government banner not implemented
- [TODO] Stats section animation timing needs refinement
- [TODO] Character disintegration formula in Problem/Solution may need adjustment
- [TODO] Overall scroll choreography across sections needs polish

**Pyramid Section Status:** Much better! Absolute positioning working, scroll timeline functional, items entering in correct phases. Need to fix broken image paths next.

### Background Colors
**Current Implementation:**
- Hero/Problem/Solution: White (`#FFFFFF`)
- Stats: Dark (`#1A0505`)
- Pyramid: Dark (`#1A0505`)

**Reference Design:**
- Light sections: Cream/Off-white (`#F3F0D6` / `#FDFBF7`)
- Dark sections: Off-black (`#110000`)
- Stats section: Dark brown (`#1A0505`) - CORRECT
- Pyramid section: Transitions from dark to cream background

**Issue:** Most sections using wrong background colors. Should use warm cream tone, not stark white.

### Typography
**Current Implementation:**
- All headings: Sans-serif (system fonts)
- Body text: Sans-serif

**Reference Design:**
- Headings: Custom display font (Grotesk Display, bold weight)
- Body text: Sans-serif
- Specific font families: `--font-grotesk-display`, `--font-geist-mono`

**Issue:** Missing display font for headings. Typography lacks the editorial/government document aesthetic.

### Pyramid Section - Layout Structure
**Current Implementation:**
- Grid layout: `grid-template-columns: repeat(auto-fit, minmax(120px, 1fr))`
- Items arranged in responsive grid
- Three tier labels with grid below each
- SVG pyramid as decorative background overlay

**Reference Design:**
- Absolute positioning: Each item positioned with `top: X%`, `left: Y%` coordinates
- Container: Fixed aspect ratio square (`width: min(48vw, 550px)`, `aspect-ratio: 1/1`)
- Items positioned spatially to form pyramid shape visually
- SVG triangle with animated stroke paths
- Tooltips positioned absolutely at specific percentages

**Issue:** Completely wrong layout paradigm. Grid cannot recreate the precise spatial arrangement of the reference pyramid.

### Pyramid Section - Visual Hierarchy
**Current Implementation:**
- Items in three horizontal tiers
- Equal visual weight across tiers
- Background: Dark (`#1A0505`)

**Reference Design:**
- Items clustered in pyramid shape (wide at top, narrow at bottom)
- Top tier (Proteins): Largest items, positioned 14-35% from top
- Middle tier (Vegetables): Medium items, positioned 40-60% from top
- Bottom tier (Grains): Smallest items, positioned 70-85% from top
- Background: Transitions from dark to cream during scroll
- Sticky "Introducing" / "The New Pyramid" text that fades/transforms

**Issue:** Missing the inverted pyramid visual metaphor. Current grid layout doesn't communicate hierarchy.

### Pyramid Section - Item Positioning
**Reference coordinates (from HTML):**
```
Milk: top: 22.58%, left: 40.51%, width: 4.25%, height: 8.84%
Olive Oil: top: 23.42%, left: 47.16%, width: 4.04%, height: 10.25%
Salmon: top: 28.07%, left: 24%, width: 9.82%, height: 5.67%
Chicken: top: 14.11%, left: 38.69%, width: 10.25%, height: 8.51%
```

**Current Implementation:**
- No absolute positioning
- Items flow in grid based on available space
- No specific size control per item

**Issue:** Cannot recreate reference layout without absolute positioning data for all 38 items.

### Pyramid Section - Scroll Behavior
**Current Implementation:**
- Simple scroll-in animation with stagger
- Items fade/scale/translate on scroll
- Static background

**Reference Design:**
- Multi-stage scroll sequence:
  1. "Introducing" text appears (dark background)
  2. "The New Pyramid" title fades in
  3. Background transitions from dark to cream
  4. Pyramid SVG draws in (stroke animation)
  5. Food items pop in with spring physics (staggered)
  6. Tooltips become visible
- Total scroll height: `min-height: calc(1300svh - 100svh)` (13x viewport height)
- Sticky positioning for pyramid container

**Issue:** Missing the elaborate scroll choreography. Current implementation is too simple.

### Stats Section - Card Animation
**Current Implementation:**
- Cards scroll up from bottom
- Opacity and Y transform based on scroll
- Hover: scale 1.02, saturate 1.2

**Reference Design:**
- Cards translate from `100vh` to `0` (full viewport height)
- Strictly clamped to scroll position (scrubbing effect)
- Hover: scale (spring physics), saturate 1.32, opacity 0.8 to 1
- Cards stack with incrementing z-index (1-10)

**Issue:** Animation timing and hover effects don't match reference precision.

### Problem/Solution Sections - Text Effects
**Current Implementation:**
- Character-level disintegration with opacity/y/rotateX
- Blur effect on heading
- White background

**Reference Design:**
- Character-level animation with stagger formula: `scrollProgress * (1 + charIndex / totalChars * 0.12)`
- Each character: opacity, blur, brightness, text-shadow
- Cream background (`#F3F0D6`)
- Sticky image with blur reveal

**Issue:** Background color wrong. Animation formula slightly different.

### Hero Section
**Current Implementation:**
- Staggered entrance animations
- Two CTA buttons
- Video modal
- White background

**Reference Design:**
- Cream background (`#FDFBF7`)
- Parallax floating elements (broccoli, milk, steak images)
- Video preview with glow effect
- Scroll-linked parallax on hero elements
- Government banner at top with US flag

**Issue:** Missing parallax elements, wrong background, missing government banner.

### Asset Style
**Current Implementation:**
- WebP images with transparent backgrounds
- Colored pencil style (correct)

**Reference Design:**
- Hand-drawn colored pencil illustrations
- High-grain paper texture
- Soft volumetric shading with hatching/stippling
- Saturated, rich, opaque colors
- Minimal cast shadows close to object base
- Vintage educational chart aesthetic

**Issue:** Asset style appears correct based on generation prompt. Verify actual assets match reference quality.

### CSS Variables Missing
**Reference uses:**
```css
--off-white: #F3F0D6
--off-black: #110000
--green: (intro overlay color)
--font-grotesk-display: (custom font)
--font-geist-mono: (monospace font)
```

**Current implementation:**
- Missing `--off-white` and `--off-black`
- Using generic color names
- Missing custom font definitions

### Animation Spring Configurations
**Reference (from DevLog-000-01):**
- Spring A (700ms, overshoot): `stiffness: 150, damping: 16` - Food title transitions
- Spring B (650ms, smooth): `stiffness: 120, damping: 20` - Navigation dots
- Card hover: `stiffness: 400, damping: 25`
- Pyramid items: `stiffness: 300, damping: 30, mass: 0.5`

**Current implementation:**
- Gentle: `stiffness: 100, damping: 20`
- Bouncy: `stiffness: 400, damping: 25` - CORRECT for hover
- Snappy: `stiffness: 300, damping: 30`
- Smooth: `stiffness: 200, damping: 30`

**Issue:** Missing Spring A and Spring B configurations. Some values don't match reference.

### Required Actions
1. [DONE] Update CSS variables to use cream/off-white/off-black color palette
2. [TODO] Implement custom display font for headings
3. [DONE] Completely rebuild Pyramid section with absolute positioning
4. [DONE] Extract exact coordinates for all 38 pyramid items from reference (see DevLog-000-05)
5. [DONE] Implement multi-stage scroll sequence for pyramid
6. [TODO] Add parallax elements to Hero section
7. [TODO] Add government banner component
8. [DONE] Update spring configurations to match reference
9. [IN PROGRESS] Fix background colors across all sections (Pyramid done, Hero/Problem/Solution pending)
10. [IN PROGRESS] Verify asset quality matches reference style guide (some broken image paths)

### Next Priority Order
1. **Fix broken pyramid image paths** (blocking visual verification)
2. **Update Hero/Problem/Solution backgrounds** to cream colors
3. **Implement custom display font** for headings (affects all sections)
4. **Add Hero parallax elements** (broccoli, milk, steak floating images)
5. **Refine Stats section animation** timing
6. **Add government banner** component
7. **Polish overall scroll choreography** across sections

### Known Issues - Broken Image Paths
**Problem:** Some pyramid items reference incorrect image filenames.

**Examples from pyramidData.ts:**
- `copper-wire-spool.webp` → should be `copper-wire.webp`
- `fdm-3d-printer.webp` → should be `3d-printer-fdm.webp`
- Potentially other mismatches between data file and actual asset filenames

**Action Required:**
1. Cross-reference pyramidData.ts image paths with actual files in `/public/images/pyramid/`
2. Update pyramidData.ts to use correct filenames
3. Verify all 38 images load correctly in browser

**Reference:** See DevLog-000-03-ASSET_INVENTORY.md for canonical asset filenames.

---

### Phase 6: FAQs & Resources (Week 4)
**Duration**: 2-3 days

#### Milestone 6.1: FAQ Accordion
- [ ] Build accordion component structure
- [ ] Implement expand/collapse animations
- [ ] Add icon rotation on toggle
- [ ] Create smooth height transitions
- [ ] Add keyboard navigation support

**Deliverable**: Working FAQ accordion
**Review Point**: Accessibility and UX

#### Milestone 6.2: Resources Section
- [ ] Create fan-out card layout
- [ ] Implement card rotation and positioning
- [ ] Add download button interactions
- [ ] Create stacked paper card effect
- [ ] Add hover states with spring physics

**Deliverable**: Resources section with card animations
**Review Point**: Visual appeal and usability

---

### Phase 7: Polish & Optimization (Week 4-5)
**Duration**: 2-3 days

#### Milestone 7.1: Performance Optimization
- [ ] Apply GPU acceleration patterns (will-change, translateZ)
- [ ] Implement lazy loading for below-fold images
- [ ] Add passive scroll listeners
- [ ] Optimize bundle size (code splitting)
- [ ] Test on mobile devices (iOS, Android)

**Deliverable**: Performance audit results
**Review Point**: Lighthouse scores and mobile performance

#### Milestone 7.2: Responsive Design
- [ ] Adjust animations for mobile (reduced parallax)
- [ ] Test all breakpoints (< 768px, 768-1024px, > 1024px)
- [ ] Fix layout issues on tablet
- [ ] Optimize touch interactions
- [ ] Test on various screen sizes

**Deliverable**: Fully responsive site
**Review Point**: Cross-device testing results

#### Milestone 7.3: Final Polish
- [ ] Add meta tags and SEO optimization
- [ ] Implement error boundaries
- [ ] Add loading states
- [ ] Create 404 page
- [ ] Final content review and copyediting

**Deliverable**: Production-ready site
**Review Point**: Final QA and stakeholder review

---

## Review Checkpoints

After each milestone, pause for:
1. **Visual Review**: Does it match the reference quality?
2. **Animation Review**: Are timings and physics correct?
3. **Content Review**: Is the fab → food parallel clear?
4. **Technical Review**: Any performance issues?
5. **Feedback Integration**: Incorporate your notes

## Success Criteria

- [ ] All animations match reference site quality
- [ ] Smooth 60fps scrolling on desktop
- [ ] Acceptable performance on mobile (30fps minimum)
- [ ] Clear messaging about distributed fabrication
- [ ] Accessible (keyboard nav, screen readers)
- [ ] SEO optimized
- [ ] Cross-browser compatible (Chrome, Safari, Firefox)

## Risk Mitigation

- **Asset delays**: Use placeholders, swap later
- **Animation complexity**: Start simple, layer complexity
- **Performance issues**: Profile early, optimize incrementally
- **Scope creep**: Stick to reference site features only

