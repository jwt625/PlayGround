# b008 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b008` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51909.2023.00333 | distilled | 2 | 5 | 8 | 0 | 12 | 0 | 0 | 11 |
| ectc51529.2024.00227 | distilled | 2 | 8 | 20 | 2 | 40 | 0 | 3 | 33 |
| ectc51529.2024.00092 | distilled | 4 | 3 | 3 | 4 | 13 | 0 | 0 | 41 |
| total (3 papers rows) | | 8 | 16 | 31 | 6 | 65 | 0 | 3 | 85 |

No TEXT_LAYER_WARNING on any paper. Text layers are clean, not truncated, and match the expected papers.

## ectc51909.2023.00333 - Georgia Tech / Corning die-embedded ARC (ECTC 2023)

Status: distilled. primary_bucket advanced_substrate (kept).

TVs: a = die-embedded bonded ARC (80 um ARC / 5 um GX-92P / 120 um ARC, fs-laser through cavity, TTC-1002 die,
30 um GL-102 ABF encapsulation, glass carrier with LTHC); b = ARC microstrip/CPW characterization (test_coupon; the values
appear only in the abstract).

Figures opened: page_02.png (Fig. 2, Fig. 3 flow), page_03.png (Fig. 4 SEM, Fig. 5), page_04.png (Fig. 6, 7, 8).
Fig. 4 inset and Fig. 5(a) were re-rendered at 6-8x in the scratchpad.

Judgment calls
- GX-92P thickness: the text says 5 um and the SEM shows 5.4772 um. The SEM value is entered (rule k) and the conflict logged.
- Cavity: the design is die plus 50 um per edge (gap_um 50). The measured bottom is 2.59 x 2.59 mm, which implies only
  25 um per edge, but the sidewall is tapered at 78.54 deg. The measured bottom and depth 207.57 um are entered with
  overrides. The Fig. 5(a) overlay labels are illegible, so these values come from the text.
- TTC-1002 thermal test chip: component_type other (no thermal test die enum). Die material is not stated (unspecified).
- The abstract RF numbers (Dk 10.12+/-0.061, tan d 0.000066-0.0013, microstrip 0.019-0.293 with no unit printed, CPW
  0.026-0.24 dB/mm, 3-170 GHz) are not supported anywhere in the body. They are entered on TV b as measured/text/p.1
  Abstract, with a note. The Intro values (er 10, tan d 0.0001, kth 36-38) are credited to ref [5] and excluded.
- The paper says "GX-92P" both as ABF and as a polyimide-based photo-imageable dry film. It is recorded as written.
- No interconnect rows: the die "connecting bumps" face up, and no dimensions or connection are given.

Not reported: substrate size, laser parameters, cure/lamination conditions, build-up wiring, warpage, reliability,
thermal, yield, RF structure details.

## ectc51529.2024.00227 - imec CoD2W for optically interconnected system-on-wafer (ECTC 2024)

Status: distilled. primary_bucket cpo (kept). buckets cpo;hybrid_bonding. The bond is collective D2W dielectric bonding,
not Cu hybrid.

TVs: a = first flow (release layer 200 nm on 30 um TBM / 700 um glass carrier, mechanical debond, 74% transfer);
b = optimized flow (release layer pre-coated on die backside, laser debond, 96% transfer, 2.4 um mean+3 sigma).
These are separate TVs because Fig. 7 shows two different temporary carrier stacks. The final bonded stack is the same.

Figures opened: page_01..page_05.png. Fig. 4, Fig. 5, Fig. 6(C), Fig. 7 and Fig. 8(C) were re-rendered at 5-7x to read
histogram percentages and stack labels.

Judgment calls
- package_class wafer_scale ("System-on-wafer"; 300 mm optical interconnect wafer). The target wafer is a
  component_type pic, as the paper calls it the "bottom PIC wafer".
- The die-to-wafer interface is entered as an interconnect with level optical / optical_coupling and bonding_method
  fusion (dielectric bond, N2 plasma, DI rinse), carrying transfer yield and alignment. EVC loss goes in
  optical_interfaces (evanescent, passive marker alignment, 1310 nm). There is one row per EVC length: 1.5/1/0.5 mm,
  with mean 0.36/0.37/0.32 dB and 3 sigma in notes.
- The EVC results are attached to TV b by inference: the Abstract and Conclusions link 96%, 2.4 um and <0.5 dB.
  The 76% of dies <0.5 dB (Abstract) and Sec. IV's "nearly all" are logged as a conflict.
- Fig. 5/6C/8C cumulative percentages (<1..<4 um) are entered as statistic percentile rows. Die-to-carrier
  mean+3 sigma X "around 6 um" is qualifier approx.
- TV b release-layer coat position in the flow is not stated precisely. It is placed before the tape transfer.
- Intro O-band <0.5 dB claim (ref [9]) excluded.

Not reported: die count, TBM/release/bonding dielectric materials, plasma and bond parameters, laser debond
parameters, MFD/polarization, bond strength, reliability, electrical I/O.

## ectc51529.2024.00092 - Absolics / YES HVM TGV wet etch (ECTC 2024)

Status: distilled. primary_bucket glass (kept).

TVs: a = lab coupon (borosilicate glass 200 um, 50 um TGV, AR 4, Fig. 4 49.49 um / 75 um pitch, Fig. 5 diameter vs
time 15-75 min); b = 510x515 mm blanket panel 400 um in the HVM tool (Fig. 6 flow, TTV 5 um, <2% within panel, ~3%
panel to panel, Fig. 8 sigma/avg 0.4%, +/-1 C bath, Fig. 10 dosing, 12 panels/batch, Ti carrier test);
c = glass-type survey (Table 1 types A-K in um/min, plus abstract 20 um/h at 110 C and 85 um/h at 130 C);
d = Fig. 3 aspect-ratio design matrix (3 via sizes x 4 thicknesses).

Figures opened: page_02..page_04.png. Fig. 3, 4, 6, 8, 9 and 10 were re-rendered at 6-8x. Fig. 9 axis labels are
illegible, so only the +/-1 C from the text is used.

Judgment calls
- Glass types A-K are kept as one TV with variants: blanket etch-rate coupons with no other stack attributes.
  The text says "up to 12 types" but the table lists 11 (logged).
- The abstract etch rates (20 um/h, 85 um/h) are on TV c in original units. The glass type is not stated.
- The Fig. 6 temperatures (50 C rinses, 130 C etch) are the generic tool flow. They are applied to the TV b steps with
  temp_c:schematic overrides. The blanket-panel etch temperature is not stated separately.
- Fig. 3 x-axis has no unit. Values of 0.2-0.7 are consistent with mm given AR = thickness/diameter (inferred, noted).
  The 75 um AR points (2.7/5.3/6.7/9.3) are qualifier approx. The text says AR 4 to 20 but the figure spans about
  2 to 14 (logged).
- Fig. 8 surface spans about 38-46 um etched against the text's "50 um etch" (logged). Fig. 10 minima are read off the
  plot (0.87 no dosing, 0.99 time-based) with qualifier approx. The +/-5% bath-health window is the pass criterion.
- Abstract 0.5 C equipment variation and measured +/-1 C are kept as separate rows (logged).

Not reported: chemistries, laser modification parameters, via shapes/sidewall angles for the measured vias, SEM
cross-section values, metallization, electrical data, reliability, ICP-MS numbers, equipment model names.

## Schema gaps
- No interconnect_level/bonding_method for die-to-wafer dielectric (oxide-oxide, no Cu) bonding. Used level optical +
  bonding_method fusion.
- No metric for cumulative alignment distributions (% of dies below threshold). Used statistic percentile with the
  threshold in parameter.
- No component_type for thermal test chip or temporary carrier (glass carrier). Used other.
- No step_category for laser modification (LMCT, as distinct from laser drilling). Used other.
- Material RF properties measured on unspecified structures (Dk, tan d, line loss in dB/mm) have no dedicated home.
  They went in metrics (electrical) on a test_coupon TV and in stack_layers.properties.
- Etch-rate and bath-control metrics (um/min, concentration fraction, bath temperature stability) fall under
  category other. A process/equipment-control category could help.
