# Schema / skill feedback from pilot_3 (ectc51906.2022.00180, 2026-10-18)

Paper: Georgia Tech Glass Panel Embedding (GPE), two TVs, emulator dies only, no electrical/reliability data.

## Missing columns

1. **Per-cell basis.** Rows routinely mix bases. Example: TV-1 HBM pillar (a-i4) has diameter and metallurgy from text
   (design_target) but the 55 um pitch only from Fig. 2/3(a) labels (extracted_from_figure). TV-2 SoC (b-c1) x/y from
   text, thickness from the Fig. 9 schematic. "Weakest basis" is undefined (is extracted_from_figure weaker than
   design_target? they are orthogonal: one is source, the other is design vs measured). Suggest splitting `basis` into
   `source` (text | table | figure) and `value_kind` (design | measured | simulated | derived | stated), or a
   `basis_overrides` list ("pitch_um:extracted_from_figure").
2. **Derived cells vs row basis.** Skill rule 12 says a derived cell's basis is `derived`, but basis is per row. TV-2
   interposer_area_mm2 = 50*50 sits in a design_target row. State explicitly that the evidence `derived` entry is
   sufficient and the row basis stays.
3. **pitch_max_um / multiple pitches.** TV-1 D2D pad arrays are "10, 20 and 30 um pitch"; TV-2 embedded die pads are
   50 um in x and 40 um in y. Only pitch_um + pitch_min_um exist. Add `pitch_max_um` (and maybe `pitch_x_um`/`pitch_y_um`),
   or a rule to emit one row per pitch variant.
4. **Die placement / cavity geometry.** Embedded-die packages need: cavity x/y/depth (TV-1 10.4x10.4 mm through-cavity;
   TV-2 7.5x7.5 mm, 110 um deep), die-to-die gap (100 um design, 92 um measured), cavity-to-die gap (150 um). No column
   fits; I used notes, a process-step conditions cell and two `metrics` rows with category `other`. Suggest
   `components.embedded` (bool), `components.cavity_x_mm/cavity_y_mm/cavity_depth_um`, and a `die_gap_um`.
5. **Emulator / dummy flag.** All dies are emulators (SoC, GPU, HBM, SRAM/router). component_type forces a choice between
   the emulated function (logic_soc, hbm) and `dummy_die`. Add `is_emulator` (bool) and keep component_type = emulated function.
6. **Build-up side.** stack_layers has no top/bottom side. TV-2 has build-up on both sides of a glass core (DL1/DL2 top,
   DL3/DL4 bottom). I continued interposer_rdl seq 1-7 above the core and 8-12 below it, with glass_core as a separate
   structure. Add `side` (front | back | core) or define seq as a single global order across structures.
7. **Lateral (same-level) layers.** In cavity embedding the die and the cavity glass occupy the same height. Rule (g)
   assumes a strictly vertical sequence. Clarify: use separate `structure`s (what I did: die vs glass_core) or a
   `parallel_to` column.
8. **Interconnect total length.** The paper's headline metric is vertical 3D interconnect length (<45 um, 44 um in the
   cross-section) spanning microbump + microvia. height_um on a single ic row does not capture a composite path. Add
   an `interconnect_length_um` metric convention or `path` rows.
9. **Pillar segment heights.** "20 um diameter pillar with 10 um Cu and 20 um SnAg": no column for Cu height vs solder cap
   height; total height not stated (summing would be my arithmetic). Add `pillar_height_um`, `solder_cap_um`.
10. **Conflicts between text and figures.** Several: package 15x15 mm (text, Fig. 3(b), 6(c), 7) vs 20 mm (Fig. 3(a));
    die 5x10 mm (text) vs 10.1x5 mm (Fig. 2); pad pitch 10/20/30 um (text) vs 20/40 um (Fig. 2); glass 200 um (text)
    vs 260 um (Fig. 9, 13); HBM 750 um (Fig. 3(a)) vs 775 um (Fig. 7). Add a `conflicts` list to the evidence YAML
    schema (field, values, locators, chosen) and a rule for which wins (suggest: measured cross-section > text > schematic).

## Enum gaps

- **step_category:** no generic `deposition` (paper says "depositing Ti/Cu" without PVD/evaporation) and no generic `etch`
  ("subtractively etching", "differentially etched" with no wet/dry stated). Had to use `other`. Also missing:
  `desmear`, `plasma_treatment` (surface_activation used), `laser_marking` (femtosecond fiducial marking), `surface_finish`
  (OSP), `cavity_formation`, `die_attach` / `pick_and_place` (used `assembly`), `ball_attach`.
- **step_category overlap:** `laser_drill` vs `tsv_tgv_formation` (laser-ablated TGVs); `lamination` covers DAF, ABF,
  solder mask and glass-to-glass bonding. Clarify precedence (method vs purpose).
- **structure:** no `cavity` / `embedded_die` and no `solder_mask`; TV-1 RDL on a glass cavity substrate is neither
  fan_out_rdl nor clearly substrate_buildup (used substrate_buildup); TV-2 used interposer_rdl. Same ABF/SAP process
  got two different structure values because the paper calls one a package and the other an interposer.
- **layer_role:** no `adhesion` for metal adhesion layers (Ti under Cu: used `barrier`); no `solder_mask` (used `passivation`);
  `adhesive` is ambiguous between a polymer bond layer (DAF, ABF bonding cavity glass) and a metal adhesion layer.
- **package_class:** no glass embedded-die / glass fan-out class. TV-1 (dies in glass cavity + RDL + BGA) mapped to
  `embedded_die_substrate`; consider `glass_embedded_die` or `fan_out_glass`.
- **component_type:** no `die_attach_film` (used `other`), no `solder_mask`, no `glass_core`/`glass_panel`.
- **interconnect_type:** no `via_in_via` (via drilled through a polymer-filled TGV); used `microvia`. Paper says
  "microbump" for 20 um Cu/SnAg pillars; `cu_pillar` vs `microbump` choice needs a rule (by metallurgy or by paper wording).
- **interconnect_level:** HBM emulator bonded onto RDL over an embedded die is both die-to-substrate and a 3D die-stack;
  used `die_to_substrate`. Consider `die_to_embedded_die`.
- **protocol:** labels like "HBM IO" / "D2D I/O" with no generation map to `unspecified`; fine, but say whether a
  physical-only pad array (no electrical data) should get an electrical_interfaces row at all. I emitted 4 rows
  carrying only bump_pitch + channel_type; they duplicate interconnect rows. Suggest a rule: only when the paper
  names a protocol or reports any electrical figure.
- **channel_type:** ABF RDL on a glass core: `glass` or `organic_rdl`? I used organic_rdl for TV-1 (RDL over embedded
  dies) and glass for TV-2 (glass interposer). Needs a rule.
- **metric_category:** geometric achievements (interconnect length, die gap) have no category; used `other`. Add `geometry`.

## Unclear rules

- **Values from a design schematic image** (Fig. 9: 260 um core, 650/800 um BGA, 775 um die) vs from a micrograph
  annotation (Fig. 7/13). Both are extracted_from_figure, but one is design and one is measured. Same issue as item 1.
- **Cited-prior-work values inside this paper's process description:** desmear roughness 300-500 nm is cited to [10];
  rule says do not attribute prior-work numbers to this TV, but the authors use it to justify their own process. I kept
  it in conditions with a note. Clarify.
- **Supplier attribution from acknowledgments:** AGC/Schott (glass), Ajinomoto (ABF), Hitachi Chemicals (dry film), Taiyo
  (solder resist). Text says PR "from TOK" while acknowledgment says Hitachi Chemicals. Rule needed: supplier only from
  the method text, acknowledgments to notes?
- **pages_read** when text.md has a references-only last page: I wrote 1-7.
- **Lists in a `float` cell** (e.g. pitch "10;20;30") are impossible; evidence `entries.value` then holds a string. Say
  that is acceptable.

## Remaining after v2 migration (2026-10-18)

- Duplicate homes: die gap and composite 3D length are both component/interconnect columns (gap_um, total_length_um) and
  geometry metrics (rule 9). Pick one canonical place or say both are expected.
- gap_um is one column but TV-1 has die-to-die (92 um) and cavity-to-die (150 um); I put them on different component
  rows. Suggest gap_die_um and gap_cavity_um.
- A cell differing in both basis and source needs two overrides for one field ("thickness_um:measured;thickness_um:figure_image").
  Works with the validator; say it is the intended form.
- Conflict precedence drops design intent from the tables (100 um design gap vs 92 um measured; 200 um specified glass
  vs 260 um cross-section). Consider *_design columns or allow a second row with a different basis.
- side for a one-sided build-up on a glass base (TV-1): front or single? Rule says single = no core; TV-1 has a glass
  base but no back build-up. Used front/core.
- structure `cavity` is unused here: unclear what layer it holds vs components.cavity_* (cavity fill dielectric?).
- pitch_um alongside pitch_min/max_um: meaning when both are given (TV-1: arrays 10/20/30 um, vias at 20 um) is undefined.
- Evidence has no section for values read but not entered (96 um span, 25 um joint width); I used `entries` with field notes.
- process_steps.target: may it point at a component (cavity etch -> embedded die comp_id)? Not stated.
