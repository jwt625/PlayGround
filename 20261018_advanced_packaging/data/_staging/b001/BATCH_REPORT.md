# b001 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b001` -> `0 errors, 0 warnings (data/_staging/b001)`.

| table | ectc.2019.00-29 | ectc.2016.201 | ectc.2011.5898527 | total |
|---|---|---|---|---|
| papers | 1 | 1 | 1 | 3 |
| test_vehicles | 3 | 4 | 2 | 9 |
| components | 3 | 9 | 7 | 19 |
| stack_layers | 18 | 5 | 12 | 35 |
| interconnects | 3 | 5 | 4 | 12 |
| process_steps | 31 | 13 | 22 | 66 |
| electrical_interfaces | 0 | 6 | 0 | 6 |
| optical_interfaces | 0 | 0 | 0 | 0 |
| metrics | 35 | 6 | 33 | 74 |

## ectc.2019.00-29 - RDL-1st FOPLP for Heterogeneous and Economical Packaging (ECTC 2019, IME A*STAR)

Status: distilled. primary_bucket panel (kept); buckets add organic_rdl_fanout (RDL-first fan-out).

TVs: a RDL-First FOPLP (TV2) 15x15 mm package, 10x10 mm test chip, 2 Cu RDL / 3 photo dielectric, Gen 3 550x650 mm
glass carrier; b photo-dielectric via-opening short-loop coupon; c photoresist LW/LS short-loop coupon.

Figures opened: page_02 to page_08 (Table 1, Figs. 1-24); high-dpi re-renders of Fig. 12, Fig. 17, Fig. 22 from
source.pdf (scratch, not kept).

Judgment calls
- Layer thicknesses for TV a taken from the Fig. 22 SEM cross-section (D3 5.04, RDL2 7.97, D2 4.10, RDL1 5.98,
  D1 6.91 um); label-to-layer assignment follows the thin measurement line next to each label, checked against the
  image scale. Design targets (Cu 5 um p.2, ECP ~7 um p.6, dielectric ~10 um) logged as conflicts.
- Extra Fig. 22 labels (29.5, 26.6, 8.32, 15.0 um) put in metrics geometry with region notes.
- Short-loop DOE coupons (dielectric via, PR LW/LS) split into TVs b/c (different structures); release-layer, seed and
  ECP module metrics hung on TV a (they are the TV process modules; Fig. 19 also appears in the TV summary Fig. 20).
- Cu plating thickness variation: <10% (p.4) entered, <5% (p.8 summary) listed as conflict.
- Package UBM "50 um diameter" entered as pad_x/pad_y on the solder-ball interconnect; Fig. 19(b) 300/200 um arrows
  recorded as an unidentified-feature geometry metric.
- ~1000 I/Os kept in TV notes, not mapped to an interconnect count.
- Fig. 11 via CD at 160 mJ/cm2 = 10.0 um and Fig. 12/17 angles read from images; Fig. 12 CD labels illegible.

Not reported: electrical/optical interfaces, reliability, Cu pillar pitch/height/solder alloy, ball size/pitch, die,
mold and package thickness, material product names, cure conditions, meander parasitic results.

Source: clean text layer; caption numbering reused (two "Fig 3"); section title typos ("VECHILCE"). Not truncated.

## ectc.2016.201 - EMIB, A High Density, High Bandwidth Packaging Interconnect (ECTC 2016, Intel)

Status: distilled. primary_bucket si_bridge (kept).

TVs: a EMIB architecture envelope (Table II bridge attributes, process flow, bump pitches); b 5-die MCP TV with
4 bridges (void-free underfill); c signaling test coupon (bridge trace pair, 80 S-parameter samples);
d simulation_model of the I/O link (PDA eye-width sweep).

Figures opened: page_03 (Fig. 2), page_05 (Fig. 4, Table II), page_06 (Fig. 5), page_07 (Figs. 6-7), page_08
(Figs. 8-11), page_09 (Fig. 12); native Fig. 4(a) image (xref 85) cropped/upscaled for the SEM inset labels; Fig. 12
at 300 dpi.

Judgment calls
- 55 um (fine) and 130 um (coarse) bump pitch read from low-resolution labels in the Fig. 4(a) SEM inset; basis
  design_target, source figure_image. Coarse bump ic_type "other" (paper says only "coarse pitch bumps").
- Bridge x/y entered as lower bound of 2x2-8x8 mm range; thickness 75 um with qualifier lt (thinning "below 75 um").
- 5 mm reach: text gives 6-8 Gb/s "depending on termination"; 8 Gb/s assigned to terminated and 6 to unterminated
  from the Fig. 12 0.5 UI contours.
- 0.17 pJ/b entered as author_estimate (authors' V^2 scaling); the 0.7 pJ/b at 1.2 V it scales from is cited to [7]
  and excluded.
- 1 TB/s example (4096 x 2 Gb/s, 14 mm edge, 300 IO/mm) entered as author_estimate; 8 Tb/s and 0.571 Tb/s/mm logged
  as derived.
- Fig. 2(b) IO density points (250 at 2 um, 500 at 1 um half-line pitch) are a technology-class envelope incl.
  CoWoS/NTI/SLIM; entered on TV a with a note.
- IL ~3.5 dB at 10 GHz and NEXT ~-17 dB read from Fig. 10 (approx); channel length not stated.

Not reported: die/package/substrate dimensions, bridge metal thickness and dielectric, bump diameter/height/metallurgy,
TCB conditions, via dimensions, cavity depth, signaling-trace length and L/S, protocol, reliability/warpage.

Source: clean; equation symbols lost in the text layer (read from page render). Overview paper, no full stack data.

## ectc.2011.5898527 - Reliability of TSV Interposers and Interconnects for 28nm FPGA (ECTC 2011, Xilinx)

Status: distilled. primary_bucket si_interposer (kept). technology_name left empty (paper does not use CoWoS or SSI);
inference in TV notes.

TVs: a 28 nm 4-slice FPGA on 25x31 mm 100 um Cu TSV interposer, 42.5x42.5 mm substrate (phase 1/2 builds kept as
metric variants: UF, flux, bump height +20%, plasma clean); b ABAQUS quarter-package simulation model.

Figures opened: page_01 (Fig. 1a-b), page_02 (Fig. 1c, Fig. 2, Tables 1-2), page_03 (Table 3, Figs. 3-4),
page_04 (Figs. 5-7, Table 4), page_05 (Table 5, Figs. 8-9); Fig. 4(c) at 600 dpi (illegible).

Judgment calls
- Process flow taken from the Introduction description (via-first/via-middle) plus Conclusions, which confirm DRIE,
  thermal oxidation, barrier/seed, Cu fill, CMP for this work; Cu pillar FPGA bumping from the same paragraph.
- Table 2 RT material properties put in stack_layers.properties as model inputs.
- Lid thickness ~1 mm is a simulation optimum; component basis simulated.
- Table 3 warpage kept in mil as written (no conversion); spec <8 mil as design_target.
- HAST duration 264 h (text) vs 248 h (Fig. 8 caption) logged as conflict.
- Table 4 strain/energy have no units and no cycle count; entered unitless with note.

Not reported: electrical interfaces, ubump count/diameter/height, TSV depth/pitch, interposer RDL details, bonding
conditions, sample counts, EM results (abstract mentions EM testing, no data), Shadow Moire values (illegible).

Source: text-layer tables garbled (read from renders); otherwise complete, 6 pages.

## Schema gaps hit
- No generic "plating" or "bump formation" step_category; used deposition_other / other where the method is unstated.
- No "drilling (method unspecified)" category; laser_drill would over-claim, used other.
- Range-valued figure labels (bridge size 2x2-8x8 mm) fit only as a lower bound plus value_text; components lack
  x/y min/max.
- Interconnects lack a count qualifier for "thousands" / "up to 1000" (non-numeric counts).
- Round pads need pad_x/pad_y as a stand-in; no pad_diameter_um.
- metrics.value is single float; measured ranges (0.68-0.73 um) need two rows (min/max).
- No field for TC condition letter (written "TCB"); test_standard left empty, inference in notes.
