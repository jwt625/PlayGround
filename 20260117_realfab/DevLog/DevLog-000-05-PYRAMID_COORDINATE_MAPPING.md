# Pyramid Item Coordinate Mapping

**Purpose**: Maps Gemini's food item names to RealFab asset filenames with exact positioning coordinates.

**Source**: 
- Coordinates from DevLog-000-01-TECHNICAL_ANALYSIS.md (lines 961-1003)
- Asset mapping from DevLog-000-03-ASSET_INVENTORY.md (lines 133-184)

---

## Complete 38-Item Mapping

| Entry | Gemini Food Name | RealFab Asset | Filename | Top % | Left % | Width % | Height % | Z-Index |
|-------|------------------|---------------|----------|-------|--------|---------|----------|---------|
| **TIER 1: ADDITIVE CORE (Proteins/Dairy)** |
| 1 | Whole Milk | Photoresist Bottle | `photoresist.webp` | 10% | 35% | 16% | 16% | 20 |
| 2 | Roast Chicken | Direct-Write Tool | `direct-write-tool.webp` | 5% | 22% | 18% | 18% | 19 |
| 3 | Steak (Raw) | LPBF Metal Printer | `lpbf-metal-printer.webp` | 20% | 15% | 14% | 14% | 18 |
| 4 | Salmon Fillet | TPP System | `tpp-system.webp` | 28% | 25% | 15% | 15% | 17 |
| 5 | Eggs (Fried) | Metal Powder Container | `metal-powder.webp` | 25% | 5% | 12% | 12% | 16 |
| 6 | Yogurt Cup | Resin Vat | `resin-vat.webp` | 35% | 32% | 10% | 10% | 15 |
| 7 | Olive Oil | DI Water Bottle | `di-water.webp` | 15% | 45% | 8% | 8% | 14 |
| 8 | Cheese Block | Aerosol Jet Printer | `aerosol-jet-printer.webp` | 12% | 10% | 11% | 11% | 13 |
| 9 | Sardines Can | Filament Spools | `filament-spools.webp` | 38% | 12% | 10% | 10% | 12 |
| 10 | Tofu Block | Conductive Ink | `conductive-ink.webp` | 30% | 40% | 11% | 11% | 11 |
| 11 | Almonds | Solder Paste Syringe | `solder-paste.webp` | 42% | 20% | 7% | 7% | 10 |
| 12 | Walnuts | Silicon Boule | `silicon-boule.webp` | 5% | 38% | 6% | 6% | 9 |
| 13 | Black Beans | SMD Components | `smd-components.webp` | 18% | 2% | 9% | 9% | 8 |
| **TIER 2: LOCAL & ACCESSIBLE (Vegetables/Fruits)** |
| 14 | Broccoli | Desktop PCB Mill | `desktop-pcb-mill.webp` | 12% | 60% | 15% | 15% | 20 |
| 15 | Frozen Peas | Benchtop SEM | `benchtop-sem.webp` | 18% | 75% | 13% | 13% | 19 |
| 16 | Red Apple | Oscilloscope | `oscilloscope.webp` | 28% | 65% | 11% | 11% | 18 |
| 17 | Carrots | Reflow Oven | `reflow-oven.webp` | 20% | 85% | 12% | 12% | 17 |
| 18 | Banana Bunch | Multimeter | `multimeter.webp` | 35% | 55% | 14% | 14% | 16 |
| 19 | Lettuce Head | Pick-and-Place | `pick-and-place.webp` | 30% | 88% | 13% | 13% | 15 |
| 20 | Tomato | Function Generator | `function-generator.webp` | 15% | 55% | 9% | 9% | 14 |
| 21 | Orange | Power Supply | `power-supply.webp` | 10% | 70% | 10% | 10% | 13 |
| 22 | Grapes | Logic Analyzer | `logic-analyzer.webp` | 38% | 78% | 12% | 12% | 12 |
| 23 | Bell Pepper | Soldering Station | `soldering-station.webp` | 5% | 82% | 10% | 10% | 11 |
| 24 | Spinach Leaf | Hot Air Rework | `hot-air-rework.webp` | 42% | 62% | 8% | 8% | 10 |
| 25 | Blueberries | Tweezers Set | `tweezers-set.webp` | 8% | 52% | 6% | 6% | 9 |
| 26 | Cucumber | Hand Tools Set | `hand-tools-set.webp` | 40% | 90% | 8% | 8% | 8 |
| **TIER 3: BIG FAB (Grains - Minimize)** |
| 27 | Sourdough Loaf | ASML EUV Machine | `asml-euv.webp` | 55% | 42% | 16% | 16% | 10 |
| 28 | Oats Bowl | Cleanroom Worker | `cleanroom-worker.webp` | 65% | 32% | 14% | 14% | 9 |
| 29 | Brown Rice Bowl | TSMC Fab Exterior | `tsmc-fab-exterior.webp` | 65% | 54% | 14% | 14% | 9 |
| 30 | Whole Wheat Pasta | Wafer Fab Interior | `wafer-fab-interior.webp` | 75% | 45% | 12% | 12% | 8 |
| 31 | Barley Sack | SLM Machine | `slm-machine.webp` | 72% | 35% | 10% | 10% | 7 |
| 32 | Quinoa Bowl | Nanoscribe Tool | `nanoscribe-tool.webp` | 72% | 55% | 10% | 10% | 7 |
| 33 | Corn Cob | 3D Printer (FDM) | `3d-printer-fdm.webp` | 58% | 65% | 9% | 9% | 6 |
| 34 | Popcorn | Laser Cutter | `laser-cutter.webp` | 82% | 40% | 8% | 8% | 5 |
| 35 | Buckwheat | Fume Extractor | `fume-extractor.webp` | 82% | 50% | 8% | 8% | 5 |
| 36 | Millet | Component Storage | `component-storage.webp` | 88% | 45% | 7% | 7% | 4 |
| 37 | Rye Bread Slice | Copper Wire Spool | `copper-wire.webp` | 60% | 28% | 9% | 9% | 3 |
| 38 | Loose Wheat Stalks | Gold Wire Spool | `gold-wire.webp` | 85% | 30% | 12% | 12% | 2 |

---

## Notes

**Height Calculation**: Set equal to width% to maintain square aspect ratio for simplicity. Can be adjusted per item if needed.

**Tier Boundaries**:
- Tier 1 (Additive Core): Items 1-13, positioned in top-left wedge (0-50% width, 0-50% height)
- Tier 2 (Local & Accessible): Items 14-26, positioned in top-right wedge (50-100% width, 0-50% height)
- Tier 3 (Big Fab): Items 27-38, positioned in bottom-center wedge (25-75% width, 50-100% height)

**Scroll Animation Order**: Items enter in numerical order (1→38) with stagger delay calculated from scroll range.

