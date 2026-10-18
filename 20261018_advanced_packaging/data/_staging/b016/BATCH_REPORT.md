# b016 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b016` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| eptc67330.2025.11392108 | distilled | 2 | 7 | 6 | 6 | 20 | 0 | 0 | 21 |
| ectc51687.2025.00052 | distilled | 5 | 19 | 6 | 4 | 27 | 0 | 6 | 22 |
| ectc51909.2023.00174 | distilled | 5 | 13 | 17 | 6 | 21 | 2 | 0 | 12 |
| total (3 papers rows) | | 12 | 39 | 29 | 16 | 68 | 2 | 6 | 55 |

None of the sources had a TEXT_LAYER_WARNING, and none was truncated or the wrong paper.

## eptc67330.2025.11392108 - Unimicron AiP module (EPTC 2025)

Status: distilled. primary_bucket advanced_substrate (kept).

TVs: a = 2T3R 60 GHz radar AiP module (A antenna substrate / B coaxial-via interposer with cavity / C IC carrier, CSP RF IC
embedded in the B cavity); b = A/B substrate strip after panel-to-panel polymer hybrid bonding (sub_assembly). Only the 5x-reflow
delamination test and the void cross-section belong to b.

Figures opened: page_02.png, page_03.png, page_04.png, page_05.png. Fig. 6, 6(d), Fig. 8, Fig. 15, Fig. 16 and Table II were
re-rendered at 5-10x.

Judgment calls
- Package thickness: 1.78 mm is entered (text, Fig. 6(c) photo, Fig. 9 header). The abstract gives 1.6 mm and Fig. 10 gives 1.4 mm
  without BGA. Fig. 6(d) spans 1.25+0.12+0.23 = 1.60 mm without balls. The conflict is logged.
- Substrate thicknesses (A 0.45, B 0.8, C ~0.23 mm) come from the Fig. 6(d) schematic. The second "~0.23" is assigned to the BGA.
  BGA ball height 0.21 mm (Fig. 10) is entered; ~0.23 is logged.
- The Sn0.7Cu A/B-to-C joint: diameter 230 um and height 120 um are measured (Fig. 8(c) cross-section labels). The as-placed ball
  of 0.2 mm (Fig. 6(d)) is in the notes. The level is "other" (PoP-like substrate-on-substrate).
- The TLPS paste via in the bonding sheet: ic_type other, bonding_method sintering (Ormet TLPS). There are no dimensions.
  The coaxial via count of 6 comes from the Fig. 16 X-ray label.
- Fig. 4 says "UV on BS for via opening" and the text says laser drilling. The category is laser_drill and both are noted.
- Warpage: 5 temperature points, sign convention crying = negative (Fig. 9 key). The variation-vs-RT % values are in the notes.
- HTST: Fig. 16(c) says "1,000 Cycles" and the table says hours. The table was used.
- The Fig. 15 L= measurements are not entered because the labeled features are not identified.

Not reported: substrate layer counts, materials and line/space; panel size; BS product; TLPS via size and cure conditions; cavity
size; reflow profiles; underfill; BGA pitch; S-parameter data.

## ectc51687.2025.00052 - IBM CPO full module TVs with polymer waveguides (ECTC 2025)

Status: distilled. primary_bucket cpo (kept).

TVs: a = pooled carrier for results not attributed to a specific build (tv_kind other); b = IBM OTV1 single-PIC optics-first module;
c = IBM OTV1B single-PIC optics-last module (substrate cut-out); d = 3 PIC/module TV; e = reduced-pitch PWG prototype (test_coupon).

Figures opened: page_03.png to page_07.png. Fig. 8, Fig. 9, Fig. 10 and Fig. 11 were re-rendered at 5-7x.

Judgment calls
- Fig. 10 (DTC/HTS/T&H/LTS IL) and Fig. 9 data "came from single PIC / module test vehicles and three PIC / module test vehicles"
  (p.7). So they go once on pooled TV a and are not copied to b/c/d. The JEDEC stress set, the IL target, the final IL range
  (<1.5-3.0 dB) and the 12x bandwidth-density claim (author_estimate) also sit on a. Creating a pooled TV is a judgment call,
  because the paper has no single "general" build.
- Fig. 10: the DTC IL per facet at 1000 cycles is entered per channel pair (3 rows). For HTS/T&H/LTS one row each holds the max
  delta at the final readout, plus a range in value_text. Every per-channel bar read is in the evidence entries. Dashed lines at
  about +/-0.4 dB are unlabeled and not treated as a spec.
- Fig. 8 early-build values (8b 5.1, 8c 4.4 dB post reflow over 3.2 dB T0) and the enhanced 2.0 dB T0 values go on b and c as
  variants (early vs enhanced build), not new TVs (material/process changes, same structure). The post-reflow increment of the
  enhanced build is unlabeled and is not entered numerically.
- Optical interfaces: PIC-PWG adiabatic, 12 ch, 50 um pitch (in fiber_pitch_um, with a note), O-band; coupling_loss_db 2.0 = full
  PIC-to-SMF link at T0 (enhanced build). PWG-to-ferrule/SMF at 250 um, MT compatible. Alignment is unspecified (not stated).
- PIC flip-chip bumps at 150 um pitch are entered as c4 (lead-free solder bumps). Micro-BGA is at 500 um pitch.
- Source inconsistencies are logged: PWG pitch "50 mm" typo; power "200-500 mW" (p.3) vs "mA" (p.7-8); HTS 110 or 125 C. The
  cross talk "less than 30 dB" is entered as written (simulated).
- Table 2 flows go in as given (12 and 13 steps). "LD" in step 11 of the last flow is undefined in the source.

Not reported: PIC thickness/platform, substrate and lid materials, bump/ball sizes, adhesive products, reflow profiles, alignment
method, MFD, n_samples and pass criteria for the JEDEC tests, thermo-mechanical stress values. No electrical interface data.

## ectc51909.2023.00174 - TSMC CoWoS-L (ECTC 2023)

Status: distilled. **primary_bucket changed si_interposer -> si_bridge**: the full text describes LSI chiplets embedded in a
molding-compound reconstituted interposer with front/back RDL and TIV, replacing the Si interposer. buckets keeps all three.

TVs: a = CoWoS-L TV (3 SoC/chiplet + 8 HBM, >10 LSI, interposer 43x58 mm, ~2500 mm2, 3x reticle); b = ring-type package; c = lid-type
package with film TIM; d = LSI-1 die structures (dual damascene Cu/USG, 0.8/0.8 um, 2 um thick); e = LSI-2 die structures (SAP Cu
RDL/PI, 2/2 um, 2.3 um thick).

Figures opened: page_01.png to page_05.png. Fig. 8, Fig. 9, Fig. 11 and Fig. 12 were re-rendered at 6-7x.

Judgment calls
- Package size: text 70x76 mm vs Fig. 3 photo labels 70x78 mm (both panels). 78 is entered (photo > text) and the conflict is logged.
- Daisy-chain and reliability results are not attributed to ring or lid, so they sit on the general TV a. b and c carry only the
  ring/lid/TIM components and the attach step.
- Chain T0 resistances are cumulative-plot medians (approx). Post-stress (MSL4 precon, then TCG 1500 cycles, uHAST 264 h, HTS 1500 h)
  is "no significant change" and entered as pass, with the post-stress ranges in the evidence.
- TIV entered as ic_type tmv (through insulator via in molding compound), as named in notes.
- The LSI-1/LSI-2 GSG insertion loss at 50 GHz (~0.82 / ~1.2 dB, plot read) is entered as electrical_interfaces with protocol
  unspecified and channel si_bridge. The line length is not given. The text cites "Fig. 10" for IL, which is a typo for Fig. 9.
- eDTC: ~1100 nF/mm2 is measured (after voltage stress). The Fig. 6 max capacitance (600 uF at 3x, label) is author_estimate.
  The 1st-generation eDTC results are from ref [8] and are excluded.
- The "typical CoWoS" u-bump pitch of 30-60 um is a generic process description and is not attributed to this TV.
- The damascene steps are entered as electroplating (an inference, noted).

Not reported: pitch/size/metallurgy of u-bump, TSV, TIV and C4; RI RDL line/space and dielectric; RI/LSI thickness and LSI size;
substrate details; HBM generation; n_samples; warpage; thermal data.

## Schema gaps
- component_type has no RF IC / radar SoC (AiP CSP entered as other) and no polymer waveguide (PWG entered as other).
- optical_interfaces has no waveguide-pitch column distinct from fiber_pitch_um. The PIC-PWG 50 um waveguide pitch was put in
  fiber_pitch_um with a note.
- There is no interconnect_level for substrate-to-substrate / package-on-package stacking (A/B-to-C joints and the TLPS bonding-sheet
  vias are entered as other). There is no ic_type for a conductive-paste (TLPS) via or a coaxial via.
- interconnect_type has no TIV (through insulator via); tmv was used.
- There is no explicit way to attach results pooled across several builds. A pooled TV (tv_kind other) was used for IBM Fig. 9-10.
- metrics has no frequency column for RF metrics (impedance bandwidth went in value_text).
