# b014 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b014` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| iedm45625.2022.10019499 | distilled | 7 | 9 | 15 | 4 | 22 | 0 | 0 | 40 |
| ectc51906.2022.00061 | distilled | 1 | 1 | 11 | 1 | 12 | 0 | 0 | 8 |
| ectc51906.2022.00218 | distilled | 4 | 10 | 18 | 9 | 27 | 0 | 0 | 37 |
| total (3 papers rows) | | 12 | 20 | 44 | 14 | 61 | 0 | 0 | 85 |

There is no TEXT_LAYER_WARNING on any of the three papers. The text layers were clean and every paper was complete (4, 5 and 6 pages).

## iedm45625.2022.10019499 - Intel QMC (IEDM 2022)

Status: distilled. primary_bucket hybrid_bonding (kept).

TVs:
- a = 3 um pitch scaled HB layer build (Figs. 7, 9, 10)
- b = QMC two-chiplet-layer architecture and the generic Fig. 11 flow (tv_kind other, schematic)
- c = ultra-thick PECVD oxide fill / CMP module wafers (Figs. 12-17)
- d = HB pad contact-area model (Fig. 8)
- e = 2.5D D2D link power and driver-area analysis (Figs. 5, 6)
- f = oxide-stress FEM (Fig. 15(d))
- g = TSV vs TDV PI/SI assessment (Fig. 19)

d-g are simulation_model.

Figures opened: page_03.png, page_04.png. Figs. 5, 6, 7-9, 13, 15-16 and 19 were re-rendered at 3-9x in the scratchpad.

Judgment calls
- Almost all plots are A.U. (normalized). Bar values on readable normalized axes are entered with `value:approx` and unit A.U. Fig. 13 warpage has no axis values, so it is entered as qualitative value_text relative to the "max allowed" line.
- ic_type is cu_cu_hybrid, but the paper never names the pad metal or the dielectric (noted). bonding_method is hybrid_bond_d2w, taken from the Sec. III title "Scaled die to wafer hybrid bonding".
- "Below 3 um / >~100K connections/mm2" is entered on TV b as a capability claim (pitch_um:lt, density_per_mm2:gt).
- The "up to 10X" power reduction is from the text. The Fig. 6(a) bars give ~0.05-0.22 of solder, entered separately; the mismatch is logged as a conflict.
- Chiplets "20+ um" is entered as thickness_um 20 with qualifier gt.
- Electrical interface: none. The UCIe interface is only "being developed" and the link power is relative only.

Not reported: die sizes and thicknesses, wafer size, node, pad size and materials, bonding and anneal conditions, TDV dimensions, oxide thickness, absolute warpage, reliability, electrical yield.

## ectc51906.2022.00061 - ASE/NCKU Ga-assisted low-T bonding (ECTC 2022)

Status: distilled. primary_bucket microbump_3d (kept).

TV: a = blanket chip-scale coupon. It is two identical 10x10 mm Si dies (Ti/Cu seed, 7 um plated Cu, X-alloy, Ga), fixture-pressed at 1 MPa and heated in an oven at 150 C.

Figures opened: page_02.png, page_03.png (Fig. 3, Table I), page_04.png. No plot values were needed.

Judgment calls
- The stack layers record the measured post-bond structure: Cu ~6.5 / CuGa2 ~1.8 / X-alloy ~9 / CuGa2 / Cu um. The as-plated 7 um Cu and the Ga layer are in process_steps. The 7 vs 6.5 um difference is logged.
- No pitch was built. The abstract's "<20 um pitch" is the target application and is noted only, not entered in pitch_um.
- ic_type is other (Ga-assisted liquid/solid interdiffusion). bonding_method is tcb, as the paper calls it, although the bond was made in an oven with a fixture.
- Conflicts logged: abstract ">5 MPa" vs 5.5 MPa (entered), and abstract "~2 um" IMC vs 1.8 um (entered).
- The "1k/2k Ti/Cu" seed has no unit in the text. It was not converted (the thickness is left empty).
- "X-alloy" is undisclosed and recorded as written.

Not reported: X-alloy identity and thickness, Ga thickness, bonding time, n for the tensile test, reliability, electrical data.

## ectc51906.2022.00218 - Amkor S-SWIFT with embedded trace RDL (ECTC 2022)

Status: distilled. primary_bucket organic_rdl_fanout (kept). package_class is organic_rdl_interposer, because the paper calls it an HDFO interposer module.

TVs:
- a = S-SWIFT package (ASIC + HBMs, 4-layer ETR, Cu pillars, laminate substrate, lid); carries the Table II reliability results
- b = ETR patterning, stacking and unit-process wafers (Figs. 2-7)
- c / d = 2-layer ETR comb BHAST coupons, prototype vs modified dielectric (Table I). They are split because the dielectric material differs.

Figures opened: page_02.png to page_06.png. Fig. 2 and Fig. 7 were re-rendered at 4x.

Judgment calls
- 300 mm wafer is inferred from the Fig. 4 position axis (+/-150 mm). It is entered with override carrier_format:figure_image and noted.
- The 2/1 um L/S and the 2 um stacked via come from the TV b cross-sections. The package L/S is not stated, so TV a leaves it empty.
- The Fig. 2 via dimensions (top/bottom/depth) are entered as separate rdl_via rows. The bottom diameters are in notes.
- Fig. 7 dishing has 18 bar reads (approx; the error-bar statistic is not stated). The text says "not over 90 nm" while one error bar reaches ~96 nm; logged.
- MSL4 is listed as the "Precon." read point in Table II. The paper does not say whether it preceded TC/uHAST/HTS. JEDEC is cited without standard numbers.
- The step reductions (40% vs dual damascene, 33% vs SAP) are stated metrics.

Not reported: all package, die and substrate dimensions, HBM count, bump/pillar/BGA pitch and size, material trade names, cure and litho conditions, comb L/S, electrical interface data, warpage, yield values.

## Schema gaps
- There is no field for via bottom diameter, so tapered vias put it in notes (ETR vias, Fig. 2).
- Normalized (A.U.) plots have no "normalized to X" column. The reference goes in parameter or scope text.
- There is no enum for a blanket/unpatterned bonding coupon interconnect, and no bonding_method for oven-plus-fixture pressing (the Ga paper uses tcb + notes).
- The carrier wafer size is inferred from a plot axis, and the basis enum has no value for "inferred". The override uses source figure_image instead.
