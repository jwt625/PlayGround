# Pyramid Asset Generation Checklist

**Total Assets Needed**: 38 individual items
**Format**: WebP, transparent/white background, 2048px+ width
**Reference**: See DevLog-000-03-ASSET_INVENTORY.md for detailed mapping

---


## Style description

**Style Description & Prompt Guide:**

**Medium:** Layered colored pencil, soft wax pastel, or opaque gouache.
**Texture:** High-grain paper texture is dominant. Visible "tooth" of the paper, stippling noise, and dry-brush effects where the pigment breaks over the paper surface.
**Shading & Lighting:** Soft, volumetric shading using hatching and stippling. Highlights are matte and textured (white pigment), never glossy or sharp. IMPORTANT: Cast shadow should be small, subtle, and stay very close to the object base - minimal shadow footprint.
**Colors:** Saturated, rich, opaque colors. Natural but idealized tones (vintage educational chart aesthetic).
**Composition:** Single object isolated on a plain white background. 3/4 view or isometric perspective.
**Keywords for Gen AI:** "Hand-drawn colored pencil illustration," "textured paper grain," "stippled shading," "vintage cookbook style," "dry media," "matte finish," "isolated on white background," "minimal cast shadow," "wax pastel."


## Example prompt

```markdown
# Image Generation Prompt: Photoresist Bottle (milk.webp equivalent)

## Subject Description
A laboratory bottle of photoresist - an amber/brown glass bottle with a black screw cap, containing golden-yellow liquid photoresist. The bottle should be similar in size and proportion to a milk bottle (about 500ml-1L). Label visible on the bottle reading "SU-8" or "SPR-220" in clean technical typography. The liquid inside should be translucent and catch light beautifully, showing its golden honey-like color.

## Style Requirements
**Medium:** Hand-drawn colored pencil illustration with soft wax pastel highlights and opaque gouache for the liquid.

**Texture:** High-grain paper texture throughout. Visible "tooth" of the paper, stippling noise, and dry-brush effects where the pigment breaks over the paper surface. The glass should show texture, not photorealistic smoothness.

**Shading & Lighting:** Soft, volumetric shading using hatching and stippling on the bottle. The amber glass should have gentle gradients. Highlights on the glass and liquid are matte and textured (white pigment), never glossy or sharp. IMPORTANT: Cast shadow should be very small, subtle, and stay extremely close to the bottle base - minimal shadow footprint that won't interfere with background removal.

**Colors:** Rich amber/brown for the glass bottle, golden-yellow for the photoresist liquid inside, deep black for the cap. Natural but idealized tones with vintage educational chart aesthetic. The liquid should glow warmly like honey.

**Composition:** Single bottle isolated on SOLID BRIGHT GREEN BACKGROUND (#00FF00 chroma key green). 3/4 view showing the front label and one side. Slight isometric perspective. The bottle should be centered and take up similar space in frame as the reference milk bottle.

**Background:** CRITICAL - Solid, uniform bright green background (#00FF00 or similar chroma key green) for easy background removal in post-processing. No gradients, no texture, no shadows extending onto the background - just flat green. The cast shadow must be minimal and stay very close to the object.

**Keywords:** Hand-drawn colored pencil illustration, textured paper grain, stippled shading, vintage scientific illustration style, vintage educational chart, dry media, matte finish, green screen background, chroma key green, isolated object, minimal cast shadow, wax pastel, laboratory glassware, amber bottle, technical illustration.

## Technical Specs
- **Dimensions:** 2048px width minimum
- **Background:** SOLID BRIGHT GREEN (#00FF00) for chroma key removal
- **Format:** Generate as PNG, will be processed to remove green and convert to WebP with alpha
- **Aspect Ratio:** Portrait orientation, similar to milk bottle reference

## Post-Processing Plan
1. Generate with solid green background (#00FF00 chroma key green)
2. Crop to center 1/3 horizontally (removes excess green space on sides)
3. Remove green background using ImageMagick floodfill technique:
   ```bash
   magick input.png -gravity center -crop 33.33%x100%+0+0 +repage \
     -alpha set -channel RGBA -fuzz 35% -fill none \
     -floodfill +0+0 "srgba(25,235,52,1)" \
     -floodfill +[width-1]+0 "srgba(25,235,52,1)" \
     -floodfill +0+[height-1] "srgba(25,235,52,1)" \
     -floodfill +[width-1]+[height-1] "srgba(25,235,52,1)" \
     output.png
   ```
   Note: Floodfill from all 4 corners ensures complete green removal while preserving object pixels
4. Verify transparency and edge quality
5. Convert to WebP with alpha channel preserved:
   ```bash
   magick output.png -quality 90 output.webp
   ```

## Technical Notes
- **Fuzz value (35%)**: Handles green color variation from illustration texture
- **Floodfill approach**: Only removes green connected to edges, preserves dark pixels in object
- **Crop first**: Reduces processing time and removes unnecessary green space
- **WebP quality 90**: Good balance between file size and visual quality

## Reference Notes
The milk bottle in the reference is simple, clean, iconic - just a white bottle with minimal detail and minimal shadow. Our photoresist bottle should have the same visual simplicity and clarity, but with the amber glass and golden liquid making it distinctive and beautiful. Think vintage chemistry textbook illustration. Green background will be removed in post-processing to create transparent background for animation.
```


## Post processing

```bash
bash -c 'source .venv/bin/activate && python process_asset.py assets/raw/SolderPhaseSyringe.png assets/solder-paste.webp'
```

## HIGH Priority (21 items) - Generate First

### Top Tier - Additive Core (13 items)
- [x] 1. LPBF Metal Printer (steak) → `lpbf-metal-printer.webp` ✓
- [x] 3. TPP System (salmon) → `tpp-system.webp` ✓
- [x] 5. Direct-Write Tool (chicken) → `direct-write-tool.webp` ✓
- [x] 6. Metal Powder Container (eggs) → `metal-powder.webp` ✓
- [x] 7. Filament Spools (canned-tuna) → `filament-spools.webp` ✓
- [x] 8. Photoresist Bottle (milk) → `photoresist.webp` ✓
- [x] 10. Resin Vat (yogurt) → `resin-vat.webp` ✓
- [x] 12. DI Water Bottle (olive-oil) → `di-water.webp` ✓
- [x] 13. Silicon Boule (avocado) → `silicon-boule.webp` ✓

### Middle Tier - Local & Accessible (8 items)
- [x] 18. Desktop PCB Mill (broccoli) → `desktop-pcb-mill.webp` ✓
- [x] 19. Benchtop SEM (carrots) → `benchtop-sem.webp` ✓
- [x] 20. Reflow Oven (lettuce) → `reflow-oven.webp` ✓
- [x] 21. Pick-and-Place (tomatoes) → `pick-and-place.webp` ✓
- [x] 22. Oscilloscope (apples) → `oscilloscope.webp` ✓
- [x] 25. Multimeter (bananas) → `multimeter.webp` ✓
- [x] 27. Soldering Station (strawberry) → `soldering-station.webp` ✓
- [x] 14. Solder Paste Syringe (almond) → `solder-paste.webp` ✓

### Bottom Tier - Big Fab (3 items)
- [x] 35. ASML EUV Machine (bread) → `asml-euv.webp` ✓
- [x] 36. TSMC Fab Exterior (bowl-rice-beans) → `tsmc-fab-exterior.webp` ✓
- [x] 37. Cleanroom Worker (bowl-oats) → `cleanroom-worker.webp` ✓

---

## MEDIUM Priority (12 items) - Generate Second

### Top Tier (5 items)
- [x] 2. SLM Machine (ground-beef) → `slm-machine.webp` ✓
- [x] 4. Nanoscribe Tool (shrimp) → `nanoscribe-tool.webp` ✓
- [x] 9. Aerosol Jet Printer (cheese) → `aerosol-jet-printer.webp` ✓
- [x] 11. Conductive Ink (butter) → `conductive-ink.webp` ✓
- [x] 17. SMD Components (peanuts) → `smd-components.webp` ✓

### Middle Tier (6 items)
- [x] 23. Function Generator (cut-apple) → `function-generator.webp` ✓
- [x] 24. Power Supply (oranges) → `power-supply.webp` ✓
- [x] 26. Logic Analyzer (grapes) → `logic-analyzer.webp` ✓
- [x] 28. Hot Air Rework (strawberry-right) → `hot-air-rework.webp` ✓
- [x] 31. 3D Printer FDM (butternut) → `fdm-3d-printer.webp` ✓
- [x] 32. Laser Cutter (potato) → `laser-cutter.webp` ✓

### Bottom Tier (1 item)
- [x] 38. Wafer Fab Interior (oats) → `wafer-fab-interior.webp` ✓

---

## LOW Priority (5 items) - Generate Last or Simplify

### Top Tier (2 items)
- [x] 15. Copper Wire Spool (walnut-kernel) → `copper-wire-spool.webp` ✓
- [x] 16. Gold Wire Spool (walnut-shelled) → `gold-wire-spool.webp` ✓

### Middle Tier (4 items)
- [x] 29. Tweezers Set (blueberry) → `tweezers-set.webp` ✓
- [x] 30. Hand Tools Set (blueberries) → `hand-tools-set.webp` ✓
- [x] 33. Fume Extractor (green-beans) → `fume-extractor.webp` ✓
- [x] 34. Component Storage (frozen-peas) → `component-storage.webp` ✓

---

## Asset Generation Notes

### Recommended Sources by Category

**Equipment (Printers, Tools, Machines)**:
- Manufacturer websites (Formlabs, Nanoscribe, Bantam Tools, ASML, etc.)
- Look for product photos with clean backgrounds
- May need background removal

**Materials (Powders, Liquids, Spools)**:
- Stock photography (Unsplash, Pexels)
- Lab supply catalogs (Fisher Scientific, Sigma-Aldrich)
- AI generation for consistent style

**Facilities (Fabs, Cleanrooms)**:
- Stock photography
- Company press kits (TSMC, Intel, Samsung)
- Creative Commons from tech news sites

### Processing Pipeline
1. Source high-res image (3000px+ if possible)
2. Remove background (remove.bg or Photoshop)
3. Color correct for consistency
4. Resize to 2048px width
5. Export as WebP with quality=90
6. Generate responsive sizes (256px, 640px, 1080px, 2048px)

### Naming Convention
Use descriptive names for the RealFab items (NOT the food names):
- `lpbf-metal-printer.webp` (NOT steak.webp)
- `tpp-system.webp` (NOT salmon.webp)
- `metal-powder.webp` (NOT eggs.webp)
- `photoresist.webp` (NOT milk.webp)
- etc.

This makes the assets self-documenting and avoids confusion. The food item names in parentheses are only for reference mapping.

---

## Quick Stats
- **HIGH**: 21 assets (55%) - 21 completed (100%)
- **MEDIUM**: 12 assets (32%) - 12 completed (100%)
- **LOW**: 5 assets (13%) - 5 completed (100%)
- **TOTAL**: 38 assets - 38 completed (100%)

**Status**: All pyramid assets complete and processed.

**Processing Notes**:
- Assets with multiple objects (hand-tools-set, tweezers-set) processed with `--keep-multiple` flag
- Script modified to preserve objects overlapping with center 50% of image
- All assets converted to WebP with transparent backgrounds

