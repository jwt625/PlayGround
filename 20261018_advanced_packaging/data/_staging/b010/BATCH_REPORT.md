# b010 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b010` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51906.2022.00057 | distilled | 3 | 5 | 8 | 4 | 15 | 0 | 0 | 25 |
| ectc51906.2022.00179 | distilled | 5 | 14 | 28 | 15 | 59 | 0 | 0 | 83 |
| ectc51906.2022.00309 | distilled | 1 | 7 | 10 | 5 | 21 | 0 | 0 | 12 |
| total (3 papers rows) | | 9 | 26 | 46 | 24 | 95 | 0 | 0 | 120 |

No TEXT_LAYER_WARNING for any paper; the text layers were clean. Each primary_bucket from the batch file was kept.

## ectc51906.2022.00057 - Sony F2F/F2B ultra-fine pitch Cu-Cu hybrid bonding (ECTC 2022)

Status: distilled. primary_bucket hybrid_bonding.

TVs: (a) two-wafer F2F test module, 1 um pitch, 0.5 um pad, 0.5 um space, daisy chain and comb-comb TEG
(200K connections each); (b) three-wafer module with F2F + 1.4 um pitch F2B HB, a thinned middle wafer, TSVs and
200 mm wafers; (c) simulation model of Cu pad thermal expansion vs pad size (Fig. 4).

Figures opened: page_01..05.png. From source.pdf at 300 dpi: Fig. 5, 6, 7 (bar charts), Fig. 8, 9 (log CDF plots,
pixel-calibrated against the gridlines), Fig. 4, 11, 12, 13, 14 viewed on the page renders.

Judgment calls
- The CMP/wet/ECD protrusion splits are pad-process variants of TV a. They are entered in metrics.variant, not as new TVs.
- Contact resistance comes from Fig. 8 on a log axis: about 0.7 ohm/unit (advanced) vs about 8E7 ohm/unit median
  (conventional). The advanced value also goes into interconnects.resistance_mohm = 700 (approx). Leakage is about
  9E-13 A median.
- The F2B pitch is entered as 1.4 um (abstract, Fig. 14 STEM, conclusion). Sec. 3-1 says "1um pitch ... to
  face-to-back", which is logged as a conflict.
- The Fig. 4 x-axis is labelled "Cu-Pad size (nm)", but the text means um. um was used and the mismatch is logged as a conflict.
- TTV of about 1/3 and edge slope below 1/5 are entered as ratios (0.33 approx, 0.2 lt). Fig. 11 and 12 are in a.u., so no absolute values.
- n_dies on wafer-level TVs counts the stacked wafers (2 and 3).

Not reported: bonding/anneal conditions, alignment, wafer size of TV a, TSV dimensions, thinned Si thickness,
numeric yield, chain count, leakage bias, reliability.

## ectc51906.2022.00179 - imec low-T BS damascene on TBM carrier, 20/10/7/5 um microbumps (ECTC 2022)

Status: distilled. primary_bucket microbump_3d.

TVs: (a) F2F full-thickness D2W TCB reference; (b) F2B 50 um thin dies after BS processing on TBM; (c) 300 mm
B2F/F2B wafer with TSV-middle 5x50 um, M1B/VIA1B damascene and BS R-UBM/solder/spacer bumps; (d) B2F N>2 stack with
a different maskset (3x50 um TSV at 7 um pitch, chains at 40/20/15/10/7 um, N=5); (e) embedded bumps in a wafer-level
photo-patternable polymer.

Figures opened: page_01..06.png. From source.pdf at 400-600 dpi: Fig. 2, 3, 5, 6, 10, 11, 13, 14, 15. At 300 dpi:
Fig. 4, 7, 9, 16, 17 header.

Judgment calls
- Bump diameters (13/7/5/3 um at 20/10/7/5 um pitch) come from the Fig. 2/3 panel labels.
- TCB was done at 1 kgf (converted to 9.81 N) and a 250 C interface temperature on a Besi TC-advanced tool with flux dipping.
- Kelvin single-bump resistances (27/40/95/210 mohm) are log-axis reads of the Fig. 15 medians. They are entered in
  interconnects.resistance_mohm for TV a.
- The CDF plateaus in Fig. 5 and Fig. 6 are entered as "daisy chains in low-resistance population" (approx). Fig. 5
  (Sec. III lot) and Fig. 15 (reliability lot, 100/100/100% plus 5 um) disagree, so both are kept as separate variants.
- For 5 um yield, text says 50% and the Fig. 15 CDF reads about 46%. The plot value was entered (rule k) and the
  disagreement is logged as a conflict.
- The text and Fig. 16 disagree on test conditions: TCT 970 cycles (text, Fig. 17) vs 1000 cycles (Fig. 16 legend);
  reflow 3x 265 C/10 min (text) vs 250 C/5 min (Fig. 17 header). Both versions are kept in metrics.condition and
  logged as conflicts.
- Fig. 16 shows a 5% HTS drop at 10 um, but the text says the yield is unchanged. The figure value was entered.
- The Fig. 14 box-plot means (30 values: DC and Kelvin, 3 tiers, 5 pitches) were read to about +-3%.
- "5 by 50" / "3 by 50" TSV was taken to mean diameter x depth in um; TV c aspect ratio of 10 is derived.
- BS layer thicknesses (M1B 1060/1070 nm, VIA1B 520/530 nm) come from the Fig. 11 FIB labels.
- Wafer bow is about 292 um after oxide and about 149 um after CMP (Fig. 9 bars; text says about 300 and 100-150).
- n_dies = 5 for TV d comes from "N=5 die"; Fig. 18 labels a tier T34.

Not reported: die x/y, bump heights except TV b, BS bump diameters, TCB time/profile, CUF product/cure, alignment
value, TBM product and thermal-budget temperature, polymer product/thickness, embedded-bump yield, warpage sign.

## ectc51906.2022.00309 - ASE chip-last FOCoS for chiplets (ECTC 2022)

Status: distilled. primary_bucket organic_rdl_fanout. technology_name CL-FOCoS (the paper's own term).

TV: (a) 74x74 mm package, 34x42 mm fan-out interposer (3 RDLs, min 2/2 um, 30 um thick, 5 um photo-dielectric,
10 um via), main die 24x32 mm + 8 chiplets 16x5 mm, min die gap 70 um, hybrid microbumps 50/30 um (L-L 150, L-S 100,
S-S 45 um pitch), C4 Cu pillar/solder at 130 um pitch, E705G core 7-2-7 substrate, ring heat sink.

Figures opened: page_02..05.png. From source.pdf: Fig. 12 colorbars at 500 dpi, Fig. 9 and 11 at 300 dpi.

Judgment calls
- package_class organic_rdl_interposer (the paper says "fanout interposer"). carrier_format unspecified, because
  the wafer size is not given.
- Warpage: the text bound is < 250 um (RT to 260 C). The Fig. 12 colorbar ranges give 230.7 um at 25 C and 54.1 um
  at 260 C (derived max - min). The sign convention is not stated.
- "C4 bump diameter 70 x 90 um" (Table 1) is ambiguous, so it is noted but not entered.
- The RDL step order follows the text (SAP then photo-dielectric). Fig. 4 order differs, which is logged as a conflict.
- The L-S 100 um pitch is entered as its own interconnect row with no diameter.

Not reported: die/mold/package thickness, Cu RDL thickness, dielectric product, bump metallurgy and bonding
conditions, sample counts, RDL leakage/sheet resistance/via-chain values, electrical interface data.

## Schema gaps hit
- No min/max columns in metrics: the IMC 600-800 nm range and the CDF spreads went into value_text.
- No place for wafer-level "n_wafers": n_dies was reused for stacked wafers in W2W TVs.
- No dedicated field for test-structure yield definitions (CDF plateau vs open/short criterion).
- No column for bump height vs UBM split beyond height/cap_height; UBM thickness for remaining Ni was put in height_um.
- No warpage sign-convention enum; colorbar range on a 3D map has no standard parameter name.
