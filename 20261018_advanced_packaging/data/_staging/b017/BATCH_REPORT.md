# b017 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b017` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51687.2025.00063 | distilled | 4 | 4 | 5 | 2 | 13 | 0 | 0 | 17 |
| ectc51909.2023.00307 | distilled | 2 | 3 | 8 | 6 | 27 | 0 | 0 | 7 |
| tcpmt.2023.3266666 | distilled | 2 | 5 | 14 | 1 | 9 | 0 | 0 | 97 |
| total (3 papers rows) | | 8 | 12 | 27 | 9 | 49 | 0 | 0 | 121 |

None of the papers has a TEXT_LAYER_WARNING. ectc51909.2023.00307 still has a garbled text layer on p.2, p.4 and p.5 (see below).
Each primary_bucket from the batch file was kept.

## ectc51687.2025.00063 - Corning glass cavity + TGV + RDL in cavity for CPO (ECTC 2025)

Status: distilled. primary_bucket glass. The CPO concept (Fig. 1) is cited to [1]-[3]. No CPO build or optical data is reported, so buckets stays `glass`.

TVs:
- (a) Blind-cavity 6-inch glass coupon. The Fig. 3 table gives three composition/process variants with side-wall length and cavity depth: A/P1 285/85, B/P1 130/75, B/P2 470/50 um. Taper ratio 0.1-0.6.
- (b) TGV inside a 75 um cavity, 47 um vias.
- (c) TGV inside an 85 um cavity, 65 um vias.
- (d) 1-layer Cu RDL inside a 75 um cavity on a 6-inch wafer: Ti 100-400 nm / Cu 150-1000 nm seed, spray-coated diluted WPR (JCR Micro), LDI, 3 um electrolytic Cu target.

Figures opened: page_02..04.png. From source.pdf: Fig. 2 at 600 dpi, Fig. 7 and Fig. 8 at 400 dpi.

Judgment calls
- Glass composition A/B and process 1/2 are unnamed splits. They go in metrics.variant, not new TVs.
- The two Fig. 4 coupons are separate TVs because via diameter and cavity depth both differ.
- Cu L/S: the abstract gives 10/10 um, while Sec. III and IV say "<~5 um". Both are entered as feature rows of the same seq. The ~2 um minimum is a resist (PR) pattern (Fig. 7) and is entered as a metric, not a Cu layer. This is logged as a conflict.
- Ti/Cu seed ranges are a general statement ("depending on cavity depth and TGV diameter"). They are entered once, on TV d.
- PR thickness: 1.4/2.0/4.3 um after 1/3/6 spray rounds (Fig. 6), and 3.9/5.0/4.3/4.1 um at wafer locations 1-4 (Fig. 8). The paper does not describe where locations 1-4 sit relative to the cavity.
- carrier_size_mm is "6-inch" as written (carrier_format other).

Not reported: glass thickness and composition, cavity x/y, TGV pitch, depth and metallization, HF/etch parameters, LDI
dose, plating chemistry, measured Cu L/S values (the abstract says they were measured but gives no numbers), whether TV d has TGVs, reliability.

## ectc51909.2023.00307 - AIST backside buried metal (BBM) PDN + via-last TSV, four-tier SLID stack (ECTC 2023)

Status: distilled. primary_bucket thermal_pdn.

Source problem: text.md is garbled by a symbol-font character shift in Sec. II on p.2 (node, bump sizes, trench depth, SiO2
thicknesses), on p.4 (warpage value) and on p.5 (BBM thickness, TSV size, chip gap). source.json says `garbled_pages: []` and
there is no TEXT_LAYER_WARNING. All of these passages were read from source.pdf re-rendered at 220-250 dpi. p.3 of text.md
also repeats the title/author block as stray text, which is harmless.

TVs:
- (a) Thin CMOS chip with BBM: 8-inch wafer, 0.13 um node, 8 x 12 x 0.045 mm chip. The BBM layer is 10.5 um thick Cu with a 15 um mesh. Signal TSV 10 um diameter, 39 um depth. SiO2 liner 0.9/0.6/0.4 um. BCB Cyclotene 4022-25 cured at 453 K for 1 h. The full 26-step flow (Fig. 3 a-q) includes tools and the Table I PPR plating recipe. Warpage about 77 um.
- (b) Four-tier stack: Cu-Ni-Au (30-35/15 um) to Cu-Ni-SnAg (35.5/26 um) SLID bonds, 573 K peak, 392 mN, 22 s load window, gap about 30 um.

Figures opened: page_02..05.png. Fig. 3, Fig. 4, Table I, Fig. 5 (250 dpi), Fig. 6, Fig. 7, Fig. 8, Fig. 9.

Judgment calls
- The single chip (TV a) and the stack (TV b) are separate builds. Process steps a-q go on TV a and step r on TV b.
- The frontside bump diameter "30-35 um" has no min/max column, so it is entered as two rows (30 and 35), per rule (d).
- The bonded-joint row carries the bonding conditions: bonding_method other (SLID), temperature converted from K, time 22 s = load window (8 s at peak), noted.
- Kelvin temperatures are converted to C (K - 273.15) and logged in the evidence derived list.
- Warpage sign convention and temperature are not stated. The Fig. 7 profile has its center higher than the edges.
- The chip's circuit function is not stated, so component_type is unspecified.

Not reported: bump pitch and count, BCB and Ti thickness, power-TSV dimensions, any electrical/PDN measurement (prior
work [8,9] only), reliability, yield.

## tcpmt.2023.3266666 - IBM 300-mm wafer-scale Si microchannel cooler (TCPMT 2023)

Status: distilled. primary_bucket thermal_pdn.

TVs:
- (a) Physical build. The Si microchannel wafer has staggered fins (75 um wide, 836 um long), 150 um channels, about 300 um deep, and an about 475 um base. It is frit-bonded to 5 CTE-matched borosilicate manifold layers (water-jet cut). Ti/Cu/Ni/Au both sides. About 225 um high-melting Pb-Sn attaches a 775 um 45 nm thermal test wafer (74 sites, 25.1 x 30.6 mm). Laminates use 150 um pitch SAC C4.
- (b) Simulation models: Fig. 4, 5, 6, 9(c), and the lower half of Fig. 10.

Figures opened: page_01..07.png. From source.pdf: Fig. 4(a) at 400 dpi (pixel-calibrated), Fig. 10 and Fig. 11 at 300 dpi.

Judgment calls
- The second cooler (Fig. 10), the "number of coolers" (Fig. 8) and the six coolers (Fig. 11) share one design. They are variants on TV a, not new TVs.
- The Fig. 10 labels carry two values per sensor. The upper/lower line is mapped to the 212 W/34 L/min and 234 W/38 L/min cases by power-density order (0.44 < 0.47 W/mm2). This mapping is an inference and is noted on each row. happ 98 000 / 104 000 follow the written order.
- The Fig. 9(b)/(c) and Fig. 10 sensor values are split by chip site: W/NW on the outlet manifold vs on the inlet manifold.
- The abstract's "about 14 kW" conflicts with 13.8 kW in the text. The text value is used and the conflict is logged.
- Fig. 8 and Fig. 11 points are entered as range/extreme reads (approx), not as per-point rows.
- Rsub 15.6, 6.2 C water rise, 12 um/g and 1.18 m/s are the paper's own calculations (basis derived). happ and the 76 kW projection are author_estimate.
- The "year or longer with no pressure-drop increase" claim is not tied to a specific unit. It is entered once on TV a with a note.

Not reported: etch chemistry/tool, glass layer thickness, frit product and firing, Pb-Sn composition and reflow
profile, metallization thicknesses, laminate details, C4 diameter/count, warpage, cycling reliability.

## Schema gaps
- There is no `*_min/*_max` for interconnect diameter or height. A written range ("30-35 um") forces duplicate rows.
- There is no field for coolant/fluidic properties (flow rate, pressure drop, fluid velocity, heat transfer coefficient). They go in metrics, with flow rate in condition text.
- Process steps have no force column (bonding load in mN), so the load went into conditions.
- carrier_format has no 150 mm / 6-inch wafer value; "other" plus "6-inch" was used.
- There is no structure enum for microchannel or manifold layers ("other" was used) and no component_type for a fluid manifold.
