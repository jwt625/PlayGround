# Schema / skill feedback from pilot_2 (2026-10-18)

Paper: ectc51906.2022.00072 (GF 90WG PIC, 12-Ch Vgroove, metamaterial SSC, IBM passive fiber attach, GR-468 stress).
An optical fiber-attach paper with no electrical I/O, no bumps, no RDL. Issues are ordered by impact.

## 1. No home for optical interfaces (biggest gap)
- `electrical_interfaces` is electrical-only (protocol enum, lane rate, pJ/bit). The paper's core result is an optical
  I/O: 12 ch, 250 um fiber pitch, O-band 1270-1330 nm, TE/TM, SMF MFD ~9 um to ~350 nm waveguide, coupler type
  (metamaterial SSC, edge, suspended), IL 1.5 dB TM / 1.7 dB TE, alignment tolerance 1.3 um (simulated).
- I squeezed these into `interconnects` (level optical, ic_type optical_coupling) + `metrics` + notes. Result:
  wavelength, polarization, coupler type and MFD are only in free text; not queryable.
- Proposal: an `optical_interfaces` table (or optical columns on interconnects): `coupler_type` enum (edge_ssc,
  grating, evanescent, adiabatic, lens, mirror, other), `fiber_type` (SMF, PMF, MCF), `channels`, `fiber_pitch_um`,
  `wavelength_min_nm`, `wavelength_max_nm`, `band` (O, C, ...), `polarization` (TE, TM, both), `mfd_fiber_um`,
  `mfd_chip_um`, `il_db`, `il_scope` (per facet | per loopback | per link), `alignment_method` (passive, active,
  self-aligned), `alignment_tolerance_um`, `optical_adhesive`.

## 2. Insertion-loss scope and sign convention
- Same paper uses three conventions: Fig. 3 plots IL negative (dB), Fig. 5 plots IL shift per facet (negative =
  worse), Fig. 6 plots IL per facet with sign reversed (positive = loss). Time-zero IL (1.5/1.7 dB) does not say per
  facet or per loopback (2 facets).
- Proposal: a rule "store optical loss as positive dB loss; record scope per_facet | per_loopback | per_link" and a
  `scope` column on metrics (also useful for pJ/bit scope and resistance per joint vs per chain).

## 3. package_class / component_type / structure enums miss photonic assembly items
- package_class: no value for a PIC fiber-attach sub-assembly (PIC + Si-submount + pigtail). Used `other`.
  `cpo_module` would be wrong (no ASIC/switch co-packaged). Suggest `pic_fiber_subassembly` / `optical_engine`.
- package_class: wafer-level mechanical coupons (DCB strip, BOR coupon) are characterized builds but not packages.
  Used `other`. Suggest `coupon` (and a `purpose` enum: product, test_vehicle, daisy_chain, coupon, demonstrator).
- component_type gaps hit here: `fiber_lid` (used other), `adhesive` (three distinct adhesives: optical
  index-matching, structural UV-tack, strain relief; used other), `submount` (Si-submount; used other),
  `connector` (MT/MPO-like ferrule visible in Fig. 4, unlabeled so not recorded). `lid` exists but means thermal lid.
- component columns: fiber needs `cladding_diameter_um`, `coating_diameter_um`, `mfd_um`; only notes available.
- structure: `photonic` exists but stack_layers role/material for a PIC BEOL with undisclosed materials has no fit;
  used layer_role other, material_class empty.

## 4. step_category gaps
- No `cure` (UV tack, batch oven full cure); used other. Very common in optical and adhesive assembly.
- No `dispense` / `adhesive_dispense`; used assembly.
- No `fiber_prep` (cleave, strip, ribbonize); used other.
- Anisotropic crystallographic Si etch (V-groove) has no clean category (etch_wet is an inference when the etchant is
  not named); used other. Suggest `etch_anisotropic` or allow `etch_wet` with "inferred" note convention.
- No `stress_test` category; GR-468 HTS/LTS/TC/DH went to metrics only (fine), but reflow-as-stress has no process home.

## 5. Per-cell basis is needed
- Rule (b) forces one basis per row. Interconnect a-i1 has pitch/count = design_target but alignment 1.3 um =
  simulated (Monte-Carlo). I moved alignment to a separate metrics row to avoid labeling simulated as design.
  The evidence `entries` list can carry per-value basis, but the CSV loses it.
- Proposal: allow `basis_overrides` list column ("alignment_um:simulated;count:design_target"), same syntax as
  qualifiers, validated against basis enum.

## 6. Reliability metrics need structure
- GR-468 results compress poorly: test name, standard (Telcordia GR-468), temperature(s), RH, duration h or cycles,
  sample size (45 / 36 modules), pass criterion (-0.71 dB per facet), result. Currently split across parameter,
  value (duration), unit, condition (free text), value_text ("Pass (45 modules)").
- Proposal: reliability columns or a sub-table: `standard`, `t_min_c`, `t_max_c`, `rh_pct`, `duration_h`, `cycles`,
  `n_samples`, `n_fail`, `criterion`, `result` (pass/fail).
- Distribution statistics (lognormal 1/50/99% lines in Fig. 6, box-plot medians in Fig. 2d) need a `statistic`
  column (mean, median, p1, p99, max, min) instead of encoding it in `parameter`.

## 7. TV granularity rule (f) overshoots for comparison coupons
- Strict reading gave 6 TVs: sub-assembly, Vgroove width split, DCB 90WG, DCB CMOS reference, BOR with Vgroove,
  BOR without Vgroove. The CMOS reference and no-Vgroove coupon are controls with almost no disclosed build data.
- Proposal: allow a `variant` / `split` column on metrics (e.g. "VGroove=Y|N", "tech=90WG|CMOS") for controls that
  differ by one factor, or state explicitly that controls get their own TV.

## 8. TV dimension columns are package-centric
- `package_x_mm/y_mm/thickness_mm` were reused for coupon size (2.43 x 70 mm strip, 25 x 25 mm x ~780 um coupon).
  PIC die size was not reported, so nothing for the actual TV. `carrier_format` is ambiguous between the die
  fab wafer (300 mm here) and the assembly format (unit-level fiber attach); I left it empty for TV a.
- Proposal: rename to `sample_x_mm` etc. or add `sample_kind`; clarify carrier_format = assembly carrier.

## 9. Locator conventions vs extracted text.md
- Fig. 1(a) label is at the end of text.md p.2 but the image renders on page_03.png; Fig. 3 renders on page_04.png
  while its caption text is on p.5. The skill should say: locator page = page where the figure image renders.
- Paper text cites "Fig. 2(c)" for results that the caption labels "(d)"; a rule "follow caption labels, note the
  mismatch" would help. Also a source-error convention (Fig. 5 panel says "LTS 500hrs", Table 1 says 72 h).
- No page_01/02 renders exist, so a figure on p.2 (if any) could not be opened; extraction only renders pages with
  image captions detected.

## 10. Minor
- `papers.technology_name`: rule (l) lists packaging brands; a foundry photonics platform name (CMS90WG / 90WG,
  45SPCLO) fits the letter but not the intent. Clarify whether process-platform names belong there.
- No place for qualitative findings with no number (crack path along <110>, "Vgroove did not induce crack onset"):
  used metrics value_text. Fine, but a `finding` category would separate them from numeric metrics.
- Cited-prior-work values (-20 to 150 C operation, reflow compatibility from ref [6]) need an explicit evidence key;
  I used `excluded_cited_values` in the YAML. Suggest standardizing that key.
- Normalized quantities (Norm_Gc, sigma_max,norm) have no unit; the normalization reference is not stated. A
  `normalized_to` field would help.
