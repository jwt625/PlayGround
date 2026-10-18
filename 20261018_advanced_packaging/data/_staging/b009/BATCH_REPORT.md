# b009 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b009` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51906.2022.00176 | distilled | 4 | 2 | 8 | 2 | 11 | 0 | 0 | 34 |
| iwlpc52010.2020.9375855 | distilled | 2 | 8 | 14 | 8 | 5 | 2 | 0 | 20 |
| ectc32696.2021.00033 | distilled | 6 | 14 | 22 | 6 | 7 | 3 | 0 | 35 |
| total (3 papers rows) | | 12 | 24 | 44 | 16 | 23 | 5 | 0 | 89 |

## ectc51906.2022.00176 - TSMC 3 um pitch SoIC (ECTC 2022)

Status: distilled. primary_bucket hybrid_bonding (kept).

TVs: a = SoIC4 3 um pitch F2F CoW TV (6x6 mm top die on 16x16 mm bottom wafer die); b = SoIC1 9 um pitch reference
(Kelvin Rc, TCR only); c = FEM bond-pad compressive stress model (Table II, simulation_model); d = Table I scaling
roadmap SoIC1-SoIC5 (simulation_model, basis derived).

Figures opened: page_02.png, page_03.png, page_04.png. Fig. 3(b) and Fig. 5 were re-rendered at 5-6x in the scratchpad.

Judgment calls
- ic_type is cu_cu_hybrid, the same as the earlier SoIC row (ectc.2019.00095). The paper never names the dielectric or says
  "hybrid". Metallurgy Cu comes from the abstract ("Cu compressive stress") and the ECP step.
- Chain resistance: the text gives short chain ~165 kOhm and long chain ~12.5 ohm. The Fig. 3(b) long-chain slope reads
  about 165 ohm. The text values are entered and the conflict is logged (possibly a unit or label swap).
- Fig. 4 yield: n=70 dies x 24 chains is the overall count. The paper does not give the split per die-shift group.
- Fig. 5 normalized Rc: 0.67 at 25 C (stated) is entered. The -40 C, 125 C and SoIC1 100 C values are plot reads (approx).
  The other points are in the evidence notes.
- The Table II caption repeats the Table I caption (a source error), logged.
- uHAST/TCC: JEDEC is named without a standard number. 10/10 pass. Preconditioning is not stated.

Not reported: die thicknesses, wafer size, node, pad diameter, barrier/seed materials, bond height, dielectric, bonding
force/dwell/temperatures (T1/T2 unquantified), anneal, absolute Rc, resistance spec, EM/SM/VBD results.

## iwlpc52010.2020.9375855 - Samsung X-Cube (IWLPC 2020)

Status: distilled. primary_bucket microbump_3d (kept).

TVs: a = package TV (daisy chain, "multiple" stacked SRAM dies, count not given); b = SAINT-S functional TV (one 7nm
SRAM die on a 7nm logic die). Both are 9.5x9.5 mm logic and 9x9 mm SRAM, Cu pillar 125 um onto a 12x12 mm substrate,
X-Cube height 0.67 mm, total 1.1 mm, BGA 0.4 mm. The 7nm TSV (4 um diameter, 50 um pitch) is entered on both TVs.

Figures opened: page_03.png, page_04.png, page_05.png. Table 1 + Fig. 5 and the p.5 Fig. 7 region were re-rendered at 2.5-5x.

Judgment calls
- Table 1 qualification (pre-con MSL3 350/3 lots; TC(B) -55/125 C 300/700/1000 cycles 258/3 lots; uHAST 96/168 h 90/3 lots;
  HTS 150 C 500/1000 h 90/3 lots) is on TV a. TC and uHAST sit under the Pre-con row in the table tree. That is noted, not asserted.
- Electrical interface: custom DDR SRAM interface. lanes = 128 and lane rate = 1.52 Gb/s (2 x 760 Mb/s) are derived from
  "24.3 GB/s (=128 x 2 x 760Mbps)". Aggregate 0.1944 Tb/s is a unit conversion. Driver 0.07 pJ/bit and receiver
  0.03 pJ/bit are split into two rows. Their basis is unspecified (the paper does not say whether measured or simulated).
  Read latency 7.2 ns is in latency_ns; write latency 2.6 ns is a metric.
- Figs. 6/7 are credited to [10]/[11] (the authors' related talks). The numbers are stated in this paper's text for this TV, so
  they are kept.
- The micro-bump bonding method is entered as unspecified ("CoW solder bonding"). Pitch, size and alloy are not given.
- Measured functional results go on TV b: 600 MHz DDR, 740 MHz Fmax, 24 MHz monitor clock, SRAM connection and BIST pass.

Not reported: SRAM die count in TV a, die thicknesses, TSV depth, micro-bump dimensions and bond conditions, Cu pillar
size, substrate build, warpage data (named as a purpose only).

## ectc32696.2021.00033 - TSMC InFO_oS 2.5x reticle (ECTC 2021)

Status: distilled. primary_bucket organic_rdl_fanout (kept).

TVs: a = 2.5x mTV, 2 logic + 8 IO, ring heat spreader (the main build, all results); b = 1 logic + 8 IO floor plan (Fig. 4a,
no separate results); c = TV-1 (850 mm2, 3 RDL, 1x reticle); d = TV-2 (1250 mm2, 3 RDL); e = 2.5x lid-type HS 3 RDL model
case (Table VI); f = RDL via-structure SI/PI/strain models.

Figures opened: page_02.png, page_03.png, page_04.png, page_05.png. Table I, Table II and Fig. 8 were re-rendered at 4-5x.

Judgment calls
- InFO area: 2100 mm2 as printed is entered. 51x42 = 2142 (conflict logged).
- Results are assigned to the 2 logic + 8 IO build: the abstract describes it, and chains D-F run between two SOC dies.
- Table VI ring case (12x12 foot, 2.0 mm, substrate/core ~2.3/1.4 mm, lower CTE core) is attached to TV a, because Fig. 4(c)
  shows a ring HS. These values come from a modeling table.
- Table I warpage is normalized to TV-1 modeling: modeling rows are simulated, Exp. rows measured. The sign convention comes
  from the table icons ((-) at 250 C, (+) at 25 C, C4 side down).
- Fig. 8 S21 at 30 GHz is read off the plot (~0.09/0.09/0.07 dB), basis simulated. 32 Gbps is the "like PCIe" data rate, so
  protocol is unspecified.
- Text/table conflicts are logged: 4-tier stacking "45% higher strain" vs Table V (0.97 vs 0.54); the "28% decrease" equals
  1.40-1.12 absolute; 7 times (text) vs 7.3x (table); yield ">95%" (abstract) vs "95% in average" (text).
- The Fig. 9 caption duplicates the Fig. 8 caption (a source error).
- "PM" dielectric is entered as dielectric_polymer (inferred; noted).

Not reported: chiplet sizes/nodes, RDL thickness, via sizes, C4 size/alloy/count, FCB conditions, underfill, absolute
warpage/PDN values, substrate layer count, HS material, reliability sample sizes.

Source quality: all three are fine. No TEXT_LAYER_WARNING, not truncated, correct papers.
