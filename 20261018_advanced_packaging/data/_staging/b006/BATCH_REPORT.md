# b006 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b006` -> `0 errors, 0 warnings (data/_staging/b006)`.

| table | ectc51909.2023.00235 | ectc51909.2023.00108 | ectc51529.2024.00093 | total |
|---|---|---|---|---|
| papers | 1 | 1 | 1 | 3 |
| test_vehicles | 3 | 4 | 9 | 16 |
| components | 12 | 9 | 6 | 27 |
| stack_layers | 6 | 2 | 24 | 32 |
| interconnects | 14 | 3 | 14 | 31 |
| process_steps | 12 | 0 | 18 | 30 |
| electrical_interfaces | 0 | 3 | 3 | 6 |
| optical_interfaces | 0 | 2 | 0 | 2 |
| metrics | 34 | 12 | 14 | 60 |

## ectc51909.2023.00235 - AI hardware chiplet integration on high density organic substrates (ECTC 2023, IBM)

Status: distilled. primary_bucket advanced_substrate (kept).

TVs: a first-round 35x35 mm 4+4-2-5 HD organic substrate with Chip 1 logic (75/55 um mixed pitch), Chip 2 HBM2 test die
(55 um), Chip 3 accelerator (150 um); MR vs TCB and Cu pillar vs Cu pedestal as metric variants. b optimized module
(TCB, Cu pedestal, mixed UBM, substrate pad +>30%, BLM +50%). c ABAQUS global/submodel FEA.

Figures opened: page_02 to page_07 (Table I, Figs. 2-12, Tables II-V); Table I, II, III, Fig. 8, Fig. 10(b), Fig. 11
re-rendered at 250-600 dpi in scratch.

Judgment calls
- Layer count: Table I "4+4-2-5 (4 HD layers)" chosen over abstract "4+1 5-2-5" (conflict logged).
- Chip 1 C4 diameter cell "30 / 30 & 40 / 30" against pitch "75 / 55": 30 um entered for both pitches, 40 um alternative at 75 um in notes.
- Paper calls them "C4" in tables and "micro-bump" in text; ic_type microbump.
- Table II DTC result read from cell colors: first resistance shift at 250 (MR, both bumps), 500 (TCB pillar), 850 cycles (TCB pedestal).
- Fig. 10(b) SEM labels printed upside down; V1 20.64, V3 9.471, H2 5.467, H1 2.310 um entered as geometry metrics with my feature assignment; V2 illegible.
- Fig. 8 normalized strain values read by eye (approx). Gray solid curve unlabeled in caption, not entered.
- Table III shows smaller Al pad diameter lowers stress, text says larger pad; table entered, conflict logged.
- Underfill "14 ppm/C" attribution ambiguous; kept in process-step conditions only.
- TV b reuses TV a stack (stack_layers not duplicated); ~100 modules entered as n_samples 100 approx.

Not reported: bump heights and solder alloy of hardware, TCB conditions, underfill/lid/TIM materials, HD dielectric
material and thicknesses, design via diameters, core, package thickness, any electrical/protocol data (HBM2 PHY region
is a pad array only).

Source: clean text layer; Table II title duplicates Table I title ("CHIP AND SUBSTRATE CRITERIA"). Not truncated.

## ectc51909.2023.00108 - Integrated Optical Interconnect Systems (iOIS) (ECTC 2023, TSMC)

Status: distilled. primary_bucket cpo (kept); buckets add si_interposer, hybrid_bonding.

TVs: a iOIS platform (COUPE2.0 on CoWoS-based composite interposer, tv_kind other); b COUPE2.0 optical engine
(EIC/PIC SoIC bond + TSV, vertical broad-band coupler with embedded microlens, sub_assembly); c MCM and d InFO_oS
comparison cases from Table II (tv_kind other).

Figures opened: page_02 (Fig. 1, Table I, Fig. 2), page_04 (Figs. 6-8, Table II), page_05 (Table III, Fig. 9).

Judgment calls
- Concept/platform paper: no measurement method stated for any number; basis unspecified, except 44.9 Tbps/mm and the
  0.015 dB COUPE2.0-CI coupling (author_estimate; caption says "estimated").
- Table II relative values (routing density, BW density, energy vs MCM = 1.0x) entered as metrics with unit "x MCM" and
  reference_tv_id = MCM TV; link length and W/S in electrical_interfaces.
- BBC loss written "-0.3 dB" entered as 0.3 dB positive; spot size ~10 um entered as mfd_chip_um approx.
- SoIC bond entered as ic_type other (paper does not say hybrid bonding).
- Waveguide layer material class unspecified (only "dielectric waveguide").

Not reported: process steps, any dimensions, waveguide material/cross-section/layer count, wavelength, absolute
bandwidth or pJ/bit, bump pitches, reliability/thermal/warpage.

Source: clean; figures are block schematics without dimensions.

## ectc51529.2024.00093 - Glass core substrate stress, transmission, reliability (ECTC 2024, DNP)

Status: distilled. primary_bucket glass (kept); buckets add advanced_substrate.

TVs: a/b/c thermal-cycle coupons (fully filled 300 um glass 60/30 um vias; conformal 400 um 80/50 um; partially filled
400 um 40/60 um); d/e/f CPW S21 coupons (400 um glass, per via type); g FEA model; h high-AR TGV demos (AR 10 conformal,
AR 12 filled); i large-panel examples (550x650, 510x515, 300x400 mm).

Figures opened: page_02 (Figs. 2-4), page_03 (Figs. 5-7), page_04 (Figs. 8-11); Figs. 3, 4, 6, 8, 9, 10, 11 re-rendered at
500-1200 dpi in scratch.

Judgment calls
- Conformal S21 loss: 0.48 dB (text, conclusion) over 0.45 dB (abstract); Fig. 8 label illegible between 0.46/0.48.
- Partially filled TC upper temperature 120 C (text x2) over 122 C (Fig. 11 caption).
- Dk 5.8 / Df 0.006 assigned to the insulating layer per text, though figure boxes list them under "Glass property".
- TGV pitches 90/150/100 um from Fig. 4 SEM labels, attached to TC coupon TVs a/b/c with a note that Fig. 4 may be a
  different sample.
- S21 value scope = TGV-section difference (CPW+TGV minus CPW), per Fig. 8 arrows.
- Partially filled per-via resistance 34 mOhm read at sigma 0 of the Fig. 11 cumulative plot (approx), span 31-38.
- TC n_samples 10 counted from legend traces (Figs. 9, 10).

Not reported: via drilling/fill process conditions, wiring and insulating layer materials/thicknesses, CPW gap, S21 coupon
via diameters, chain via count, warpage, stress values for conformal/X-shape (Fig. 6 labels illegible), Fig. 3 GIP spec text.

Source: clean text layer but embedded figures are low resolution. Template note left in the author block; keyword line
garbled (font encoding). Not truncated.

## Schema gaps
- No column for relative/normalized interface figures (x vs a baseline): used metrics with unit "x MCM" + reference_tv_id.
- No alignment-tolerance column in optical_interfaces (BBC +/-10 um went to metrics).
- No beachfront/optical bandwidth density column in optical_interfaces (44.9 Tbps/mm went to metrics).
- No waveguide propagation/bend/transition loss columns (metrics).
- Interconnects lack a bottom-diameter column for tapered TGVs (top/bottom kept in notes).
- No column for a split bonding method (MR vs TCB) on one TV; recorded as metric variants and notes.
- Interconnect level/type have no entry for SoIC-type bonding without explicit hybrid-bond wording.
