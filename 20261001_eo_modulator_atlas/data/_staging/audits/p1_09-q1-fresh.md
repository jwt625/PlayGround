# Audit Q1 (fresh context): batch p1_09 (li2026, li2026a, li2026ba, lin2025, niels2026)

Auditor: independent subagent, 2026-10-01. Read-only: no data or staging file edited; only this report written. No network. Sources used: references/<id>/text.md, crossref.json (li2026a, lin2025, niels2026), source.pdf (re-rendered with pymupdf: lin2025 PDF pp.23-25, niels2026 pp.4-5), figures/page_*.png (li2026 pp.3-4, li2026ba pp.2, 5, 6) and img_p02_4.png (li2026ba SEM), data/schema/devices.schema.yaml, SKILL.md, sims/SPEC.md (provenance classes only), both prior audits, data/organizations.csv, data/candidates.csv, staging batch CSV.
Mechanical check (script, uv + pyyaml): every non-empty evidence:true cell has an evidence entry with the same value; no evidence entry points at an empty cell or unknown column; no qualifier sits on an empty field; no absolute/home paths or non-ASCII in devices.csv, evidence/*.yaml or the two sim configs. All clean. Org names checked against data/organizations.csv and every other staged organizations.csv: none of EPFL / KIT / Ghent / imec / Luxtelligence exists elsewhere, so no spelling collision at audit time (p1_10/p1_11 not yet populated).
Severity: high = wrong/misleading in a way a plot or comparison would show, or provenance that presents unverified data as sourced; medium = convention/completeness defect that changes interpretation; low = cosmetic, locator, judgment call.

---------------------------------------------------------------------
## 0. Identity, version and duplication questions asked by the coordinator

| Question | Finding |
|---|---|
| li2026a vs li2026ba: same work? | Two distinct publications. li2026a = OFC 2026 paper W1A.5 "Lithium-Tantalate-on-Fused Silica Mach-Zehnder Modulators" (DOI 10.1364/ofc.2026.w1a.5, Crossref title/5 authors match). li2026ba = arXiv 2604.14836v1 (16 Apr 2026) "Low voltage and high-bandwidth thin-film lithium tantalate modulator on a silicon dioxide substrate", no DOI. Same group and same device family (18 mm LT-on-FS, T-segment CPW), but different numbers (BW 67 vs 64 GHz, Vpi 1.56/1.28 vs 1.53/1.21 V, etch/slab 400/200 vs 440/160 nm, net rate 437 vs 440.6 Gbit/s) and different wafer descriptions (BOX 4.7 um in OFC; 2 um in arXiv, both irrelevant after carrier removal). Keeping two papers and two row sets is correct. They are not linked in any structured field (notes only): a view that plots all rows will show the same device family twice. |
| Which version is each source? | li2026 = arXiv v1 (19 Jul 2026, stamp p.1); li2026ba = arXiv v1 (16 Apr 2026, stamp p.1); li2026a = OFC proceedings text (text-only); lin2025 = accepted unedited Nature Communications manuscript ("ARTICLE IN PRESS", Crossref published/issued 2026-02-26; numbers attributed correctly; arXiv 2505.04755 not consulted); niels2026 = arXiv v1 manuscript (PDF creator Firefox 136.0.1, creation date 2025-03-14, 9 pages incl. supplementary; Crossref journal = Nature Photonics 20(2) 225-231, first online 2026-01-13). Batch attribution is right in every case; the only structural weakness is niels2026 (see N3). |
| License / redistribution | li2026, li2026ba: no notice in the PDF, text.md header "CC-BY-4.0" is batch metadata only; empty license + unknown is correct per (k). li2026a: "(c) 2025 The Author(s)" only, Crossref has no license; empty + restricted_local_only OK. lin2025: Crossref tdm+vor CC-BY-NC-ND-4.0 and own notice p.1: correct, restricted_local_only OK. niels2026: Crossref gives only Springer Nature TDM terms; restricted_local_only OK (see N5 for the free-text license cell). |
| Authors | li2026, li2026ba, li2026a: 5 authors, order and spelling as in sources (Tobias J. vs Tobias Kippenberg preserved per source). lin2025: 11 authors match Crossref exactly. niels2026: 25 authors match Crossref one by one (script diff empty; "Cedric", "Gunther" diacritics preserved). |
| Organizations | See section 6. |

---------------------------------------------------------------------
## 1. li2026 (arXiv 2607.17436v1, suspended LTOI MZM)

Identity: title, authors, arXiv id, published_on 2026-07-19 (stamp), venue "arXiv", license empty / unknown, countries CH;DE, fab (CMi + IPHYS from Acknowledgements p.6), wafer NANOLN: all verified. repro_grade C plausible (no waveguide width, no ground width, no Wc).

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| L1 | medium | li2026-a bw_measured_to_ghz (empty), bw3db_ghz=110 | crossing claimed at 110, no measured-to value | p.3 text: "smoothed curve yields a 3 dB bandwidth of 110 GHz"; Fig. 2(f) (rendered p.4, panel e): raw gray data end at about 109-110 GHz, smoothed blue continues to about 114 GHz and reaches -3 dB at about 109-110 GHz, -3.7 dB at its end; probes are 110 GHz (p.2), VNA 125 GHz (p.2) | Convention (c): a claimed crossing at the instrument/data limit fills both fields. Add bw_measured_to_ghz=110, basis extracted_from_figure, locator "p.4, Fig. 2(f) as captioned"; keep the "smoothed fit, raw noisy" note. |
| L2 | low | extinction_ratio_db (empty) | "modulator IL and ER not reported" | Fig. 2(a) (p.4): normalized transmission on a dB axis dips to about -20 dB at 0 V; no text number | Either enter about 20 (approx, extracted_from_figure, static, lower-bound caveat from finite sampling) or add "ER visible only in Fig. 2(a), about 20 dB" to the notes; the BATCH_REPORT statement "ER not reported" is slightly wrong. |
| L3 | low | evidence basis `design_target` for drive=push_pull | design_target | p.2: "two 8 mm-long suspended electro-optic phase shifters ... to form a push-pull Mach-Zehnder modulator" - a statement about the fabricated device, not a target | No enum fits "stated by authors"; same choice is made in li2026a and lin2025. Acceptable, but document in the skill (see X6). |
| L4 | low | electrode_gap_um=6 (G) | G = gap between facing T-bars where the waveguide sits | Fig. 1 caption p.3: "T-shaped ground-electrode segments"; G is drawn beside the T-bar head, the opposing element is not identifiable from the schematic (text p.2 lists G=6 in the parameter tuple) | Evidence note already says "reading of the schematic"; also note that the caption describes T-segments on the ground side only, so the opposing element may be the signal electrode edge (still a metal-to-metal gap across the waveguide). |
| L5 | low | integration=monolithic vs li2026ba integration=other for the same LT platform family | inconsistent | Both use a commercial NANOLN LTOI (bonded film); li2026ba performs the LT-to-fused-silica bonding itself | Pick one rule for "film transferred to a different carrier by the authors" (see X7). |

### Checked and correct (sampled 14)
Vpi 5.1 V at 1550 nm, 100 Hz sawtooth 20 V pp (p.2; Fig. 2a arrow 0 to 5 V); VpiL 4 V cm (5.1 x 0.8 = 4.08, `derived` correct); 8 mm (p.2, Fig. 1a); 600 nm film, 4.7 um BOX, 525 um HR Si, 400 nm etch / 200 nm slab, 1.5 um ICP-CVD oxide, Ti 15 nm / Au 800 nm (p.2); (Wsig,P,G,LS,WS,LT,WT) = (70,35,6,20,1.5,33,1.5) um; bw3db 110 GHz normalized to 1 GHz (p.3, Fig. 2f caption); n_rf plateau 2.23-2.25 and dashed optical ng about 2.24 (Fig. 2(e) as captioned, right axis read: 2.235-2.24); microwave loss at 10.95 GHz^0.5 about 5.1 dB/cm (entered 5.0 approx; text S21 -4.3 dB at 120 GHz = 5.4 dB/cm, disclosed in the note); S11 < -20 dB, S21 -4.3 dB (p.3); 208 / 192 / 188 GBd (p.4), 460 Gbit/s at 180 GBd PAM8, AIR 485 (p.4); 564 = 188 x 3 in `derived`; laser 17.8 dBm and fiber-to-fiber coupling -5.2 dB correctly kept out of the power/IL columns; "LiNbO3 platform" typo (p.2) correctly noted; caption panel mismatch (a)-(f) vs rendered (a)-(e) confirmed. Locators page index: p.2 device design / VpiL text, p.3 bandwidth text, p.4 figures: all match the `<!-- page N -->` markers. vpi_convention mzm_push_pull marked `derived` with the "authors do not name the convention" note: correct per (f).

### Verdict: ACCEPT WITH CORRECTIONS (L1 should be fixed; no wrong value found).

---------------------------------------------------------------------
## 2. li2026a (OFC 2026 W1A.5, text-only)

Identity: title, 5 authors, DOI, page W1A.5, year all match Crossref; Crossref has only published-print year 2026 (created 2026-06-02 is a registration date), so empty published_on is right; license empty + restricted_local_only right. Affiliations 1-5 match text (EPFL IEM and IPhys; KIT IPQ and IMT; LUXTELLIGENCE SA, Saint-Sulpice), so Luxtelligence SA as company is correct here (contrast li2026/li2026ba/lin2025 where it appears only under competing interests: companies correctly empty there). No fab named in this paper: foundry_or_fab correctly empty. Nothing is claimed from figures: every value in rows and evidence traces to running text (Sec. 2-4, abstract); figure references appear only as pointers ("Fig. 2(d) not available").

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| A1 | low | etch_depth_nm=400, slab 200, basis `measured` | 400 / 200 | Sec. 2: "2 um wide, 400 nm high waveguide with a 200 nm slab" on a 600 nm film. "400 nm high" is ambiguous (rib height above slab vs total); 600 - 200 = 400 only makes the reading self-consistent (and matches li2026) | Keep the value; evidence note should say "interpreted as etch depth: 600 nm film minus 200 nm slab; text says waveguide height 400 nm". |
| A2 | low | li2026a-b qualifiers vpil_dc_vcm:approx | approx on 2.3 V cm | Sec. 2: "around 2.8 V cm for 1550 nm and 2.3 V cm for 1300 nm" - "around" attaches to the first number grammatically; 2.3 is stated plain (also 1.28 V and 1.56 V are plain "corresponds to") | Defensible either way; if kept, say so in the evidence note. |
| A3 | low | locators "Sec. 2", "Sec. 4", "Abstract" | section locators, no page index | No PDF and no page markers in text.md (text-only), so convention (i) cannot be applied | Document in the skill that text-only sources use section locators (X5). |
| A4 | low | tags `slow_wave` | present | li2026a text never says slow-wave (it says T-segment structure, microwave index matched to optical group index); "slow-wave" is the li2026ba wording | Drop the tag or mark as taken from li2026ba. |
| A5 | low | no structured link to li2026ba | notes only | Distinct papers on the same device family (section 0) | Add a tag (e.g. `ofc_version_of_li2026ba`) so views can de-duplicate or group. |

### Checked and correct (sampled 13)
VpiL "around 2.8 V cm" at 1550 and 2.3 V cm at 1300; 1.56 V / 1.28 V "corresponds to ... for an 18-mm long modulator" (2.8/1.8 = 1.556, 2.3/1.8 = 1.278, `derived` basis right since authors convert); 18 mm; 100 Hz; 600 nm film; 2 um wide waveguide; 1.5 um metal-to-waveguide spacing (electrode_gap 5 um = 2 x 1.5 + 2, in the `derived` list, not in `entries`: correct per (h)); fused silica 4-inch, direct bonding; 3 dB BW 67 GHz (Sec. 2; abstract; Summary) with reference `unspecified` (text gives no normalization); 208 / 192 / 184 GBd; line rate 552 stated in Summary ("single-carrier line rate of 552 Gbit/s (184 Gbd PAM8)"); net 437 at 180 GBd, AIR 466; laser 17.8 dBm, 3.7 dBm at the output fiber (notes only: correct); O-band row carries only Vpi/VpiL (no O-band bandwidth in text: correct). No numbers from Fig. 2(a)-(g) are used.

### Verdict: ACCEPT (A1-A5 cosmetic; I tried to break the VpiL/Vpi arithmetic, the rate arithmetic and the figure-free claim and found nothing wrong).

---------------------------------------------------------------------
## 3. li2026ba (arXiv 2604.14836v1, LT-on-fused-silica MZM)

Identity: title, 5 authors, arXiv id, published_on 2026-04-16 (stamp), license empty / unknown, repro_grade B, sim_config path all fine. paper_id suffix "ba" is a batch artifact (renamed from li2026b to avoid a seed collision; candidates.csv shows an unrelated li2026b, so the rename is necessary).

### Findings (rows and papers)

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| B1 | low | il_fiber_to_fiber_db (empty) | "fiber-to-fiber coupling loss ~12 dB (coupling only)" only in notes | p.3: facet "resulting in a fiber-to-fiber coupling loss of approximately 12 dB" | Judgment call; the sentence is the only optical-loss number in the paper and is named fiber-to-fiber. Either enter 12 with `il_fiber_to_fiber_db:approx` and a note "coupling-dominated, facets unpolished", or keep empty and state the reason. Same question for li2026 (-5.2 dB). |
| B2 | low | rib_width_nm=2000 approx, basis extracted_from_figure; note "SEM top ~1.8 um, base ~2.2 um" | 2.0 mid-height | Fig. 1(e) SEM (img_p02_4.png, 3 um bar): I measure top about 1.77 um and base about 2.0-2.1 um at the rib foot | Value 2.0 and approx qualifier are fine; correct the note to "base ~2.0-2.1 um". The config provenance carries the same 2.2 um base figure (it uses polygon +-1.1 at the base); impact small. |
| B3 | low | integration=other | other | LT film bonded to fused silica by the authors (p.3) | See L5 / X7. |
| B4 | low | li2026ba-b wavelength_nm=1300 | 1300 | p.4: Vpi measured at 1300 nm; EO response at 1310 nm (Fig. 3d); note already says so | OK; just confirming no row for the 1310 nm EO point (rule d would allow; no bandwidth is stated for O-band, so fine). |

### Checked and correct (sampled 18)
Vpi 1.53 V (C-band) / 1.21 V (O-band), device C1 F11 (Fig. 2a arrows 0 to 1.53 and 0 to 1.21 V); VpiL 2.76 / 2.19 from Fig. 2(c) cell F11 (read: 2.76 blue, 2.19 orange; 1.53 x 1.8 = 2.754, 1.21 x 1.8 = 2.18: consistent). Note: F11 is the minimum of the wafer map in both bands (C-band wafer values 2.76-3.20, mean of the 11 cells I computed = 2.86; O-band 2.19-3.65, mean 2.415), so the row carries the best field, which the notes disclose ("wafer average 2.86 / 2.42"); the paper's "(2.76, 3.20)" are min/max. bw3db 64 GHz and bw6db > 100 GHz (p.4 text; Fig. 3d blue crosses -3 dB at about 62-64 GHz, -6 dB at about 97-100 GHz, ends near 110 GHz at about -7 dB: eo_rolloff 7 approx at 110 correct, bw_measured_to 110 extracted correct); reference 1 GHz (Fig. 3 caption); ER about 12 / 10 dB (p.4, static, approx); S21 -8.2 dB at 120 GHz = 4.56 dB/cm (entered 4.6 as the authors' "approximately 4.6", `derived`); Z0 about 42 ohm (p.5); n_rf and ng 2.24 (Fig. 3c right axis read 2.238); 440 nm etch / 160 nm slab / 1.5 um HD-PECVD / 200 nm cap / Ti 15 nm + Au 800 nm / 500 um fused silica (p.3); (WT,LT,WS,LS,G,Wsig,P) = (1.5,32,2,5,5,100,35) um (p.4); electrode gap G 5 um vs SEM about 4.8-4.9 um (my read from the two inner Au edges: 4.8); 208 / 192 / 184 GBd, PAM4 KP4 to 176 GBd, 440.6 Gbit/s at 176 GBd with 18.7 % overhead, AIR 468.1 at 180 GBd; 552 = 184 x 3 in `derived`; laser 17.8 dBm, 3.7 dBm after the MZM, 7.7 dBm at the receiver (not on-chip: correct). drive=push_pull marked `derived` with "authors do not state push-pull in this version" (correct; the arXiv text only says "pair of modulation arms"; the OFC version states it). Locators match page markers (p.2 SEM, p.3 fabrication, p.4 EO modulation, p.5 Fig. 2 and Z0, p.6 Fig. 3 and IMDD, p.7 Fig. 4).

### Sim config sims/li2026ba/config.yaml

| # | Sev | Item | Issue | Suggested correction |
|---|-----|------|-------|----------------------|
| BC1 | high | materials.lithium_tantalate.eps_r (41/43) and r13 (8.4): provenance class standard_reference with citation "Weis and Gaylord, Appl. Phys. A 37, 191 (1985)" | The batch itself says the values are recalled and NOT verified; the attached citation is, to my knowledge, the LiNbO3 property summary ("Lithium niobate: summary of physical properties and crystal structure"), i.e. not a LiTaO3 source (cannot be checked offline; no copy in references/). A citation attached to an unverified recalled value reads as sourced. Neither value appears in any cached source (grep of all five texts: no LT permittivity, no r13). | Change class to a flagged state (e.g. `project_inference`, note "recalled, unverified; no cached source") and delete the citation until a LiTaO3 reference is actually opened; keep the limitation text. Same defect in lin2025 (LC1). |
| BC2 | medium | materials.gold.sigma_Sm 4.1e7, class standard_reference, citation "bulk gold 4.1e7 S/m" | No source. The value equals the example in sims/SPEC.md (line 43), not a measured/cited constant. A better-sourced number is in the cached lin2025 text (p.6-7, Fig. 2d): bulk Au 2.21 uOhm cm (4.5e7 S/m), thin-film evaporated Au 2.56 uOhm cm (3.9e7 S/m, 796 nm film, same group and process class). | Use `project_inference` with the SPEC-example origin stated, or switch to the lin2025 thin-film value with a locator (3.9e7), flagged as a cross-paper value. |
| BC3 | low | r33 = 30.5 pm/V | Verified: niels2026 p.9 "r33 = 30.5 pm/V is the EO coefficient of LiTaO3" (cached). The class is right, but it is a coefficient used in a different paper's model, not a measurement; the config says so. | OK. |
| BC4 | low | limitations/missing | 15 nm Ti adhesion layer and the 200 nm cap over the electrodes are not listed as omitted/assumed (lin2025 config lists its Ti barrier). Unit/geometry arithmetic otherwise consistent: signal x -109..-9, bars at +-(2.5..4), stems 5 um to bodies at 9 and -9, left arm at -118 with bars -115.5..-114 and -122..-120.5 (G = 5 um each), ground_l -227..-127; rib polygon height 0.44 um, slab 0.16 um. | Add the Ti layer to limitations. |
| BC5 | low | YAML | `sigma_Sm: 4.1e7` loads as a string with PyYAML (verified: type str); batch notes this (SPEC_PROPOSALS item 3) | Write 4.1e+7 so both loaders agree. |

Targets: vpi_l 2.76 (device F11; tol 15 %), n_rf 2.24, Z0 42, rf_loss 4.6 at 120 GHz, ng 2.24 all tied to evidence entries; the 64 GHz bandwidth is correctly not a target. Honest `missing` list (ground width, direction, RF eps, r22/r51, Au conductivity, stems, cladding profile).

### Verdict: ACCEPT WITH CORRECTIONS for rows (no wrong value; B1 judgment call); sim config needs BC1 (high) and BC2.

---------------------------------------------------------------------
## 4. lin2025 (Nature Communications 17(1) 3211, accepted manuscript)

Identity: title, 11 authors, journal, volume/issue/article number 3211, DOI, published_on 2026-02-26 (Crossref published/published-online/issued; also "First Online" assertion), CC-BY-NC-ND-4.0 (Crossref tdm 2026-02-26 and vor 2026-04-07; manuscript notice p.1), restricted_local_only, access open_access: all verified. Fab: "EPFL Center of MicroNano Technology (CMi) and the Institute of Physics (IPHYS) cleanroom" (Acknowledgments p.19): correct. Received 2025-08-26 / accepted 2026-01-29 (p.1) are consistent with Crossref.

### Findings (rows and papers)

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| N1 | medium | lin2025-a bw3db_ghz=40, bw_basis measured | 40 GHz (text value) | p.9 text: "3 dB EO bandwidth reaches 40 GHz for a 16 mm-long device"; Fig. 3(d) (PDF p.24, rendered 1.5x): the solid 16 mm trace starts at 0 dB at 1 GHz, is at about -2.3 dB at 40 GHz and reaches the -3 dB dashed line at about 50 GHz (about -4.5 dB at 70 GHz, -6.2 dB at 110). The 6 mm trace crosses -3 dB at about 98-100 GHz (text 100: consistent) | Keep the text value (do not replace by a figure read) but add to the evidence note: "Fig. 3(d) -3 dB crossing from 0 dB at 1 GHz reads about 50 GHz; text says 40 GHz; normalization not stated". Possibly the authors reference a different point; the discrepancy is worth carrying. Reference stays `unspecified`. |
| N2 | low | optical_power_handling_dbm (empty) | "Vpi measured up to 1.17 W on-chip power" only in lin2025-c notes | p.8 and abstract: stable Vpi "up to 1.17 W" on-chip, "watt-level on-chip optical power handling" as a headline claim; device not stated | 10 log10(1170 mW) = 30.7 dBm; enter 30.7 on lin2025-c (the unidentified-device row) with `optical_power_handling_dbm:gt`? It is a tested level, not a measured limit, so at least add the number to a column or state the decision. |
| N3 | low | lin2025-c scope | system-demo row with eo_material/platform/electrode_type but no geometry | Paper never identifies the IMDD device (p.9, p.13); geometry fields (film 600, etch 320, slab 280, gap 6, Cu 1.8, cladding) are identical for the 6 mm and 16 mm devices | Row design is defensible and honest (tag `system_demo_only`, notes). If kept, the shared geometry could be copied (same on both lengths); and the gap that no length/Vpi/bandwidth is attached must be handled by views (see X3). |
| N4 | low | ng_opt=2.22 basis author_estimate, qualifier ng_opt:approx | approx, author_estimate | p.11: "microwave group indices are all well matched to the optical group index of 2.22 within a tolerance of +-0.03": the tolerance belongs to the microwave indices; the method for 2.22 is not given (simulation vs measurement unknown) | `approx` is not supported by the wording; basis could stay author_estimate or become `unspecified`-like. Also the value is a wafer-level statement, not this device. |
| N5 | low | published_on 2026-02-26 | accepted-manuscript online date | Schema: "first public version"; the arXiv preprint (2025-05-07 per candidates.csv) predates it | Consistent with the numbers' version (the AM is exactly the 2026-02-26 object); note already explains. Fine. |
| N6 | low | drive evidence basis `design_target` | design_target | p.6: "EO phase shifters operating in a push-pull configuration"; Fig. 1 caption "a pair of push-pull phase shifters" | Same remark as L3. |

### Checked and correct (sampled 20)
Vpi 1.7 V (Fig. 3a arrow about -0.8 to +0.95 V; 100 Hz triangular; 16 mm, 4 um waveguide, 6 um gap in the panel and caption); VpiL 2.7 V cm (`derived`: 1.7 x 1.6 = 2.72, authors' stated; simulated 2.64); ER 20 dB (Fig. 3a inset dips to -20, text p.8); Vpi stable 1 Hz-1 MHz (Fig. 3b, 1.6-1.72 V); drift 0.4 dB over 15 h at 1.75 mW on a 6 mm device (Fig. 3c caption, p.21; correctly assigned to lin2025-b notes); EO response to 110 GHz; eo_rolloff 6.2 dB (16 mm) and 3.8 dB (6 mm) at 110 GHz (Fig. 3d reads: 6.2 and 3.8, matches); bw3db 6 mm 100 GHz; film 600 nm / etch 320 / slab 280 / BOX 4.7 / Si 525 um HR (p.5); 4 um waveguide, 6 um gap (p.7); Cu 1.8 um after CMP, Ti 10 nm barrier, 2.5 um PECVD oxide, 100 nm Si3N4 passivation (p.12-13); Cu resistivity 2.04 uOhm cm (p.7); test CPW 16 mm, 6 um gap, 27 um signal, loss about 10 % below Au (p.7, Fig. 2d) correctly NOT copied to the MZM rows; PAM4 208 GBd / 416 Gbit/s, PAM8 180 GBd / 540 Gbit/s (p.3, p.9, Fig. 4 labels), net 423 Gbit/s at 176 GBd PAM8, AIR 449 (p.10); laser 17.8 dBm at 1550 nm (p.13), output fiber 5.6 dBm (notes). Locators match page markers; Fig. 3 is on PDF p.24 as cited. Wavelength of the Vpi/EO measurement is indeed not stated in the main text (I searched all pages); wavelength left empty on a/b is right.

### Sim config sims/lin2025/config.yaml

| # | Sev | Item | Issue | Suggested correction |
|---|-----|------|-------|----------------------|
| LC1 | high | eps_r 41/43 and r13 8.4 with citation Weis and Gaylord 1985 | Same defect as BC1: recalled unverified values labelled standard_reference with a citation that (to my knowledge) concerns LiNbO3; no cached source for LT permittivity or r13 | As BC1. |
| LC2 | low | silicon_dioxide (n 1.444, eps 3.9) and silicon (n 3.476, eps 11.7) "standard handbook values" | No named source; eps_r(Si) = 11.7 is stated in li2026 p.1 (different paper) | Name a source or flag as unverified; same as pilot X9. |
| LC3 | low | targets rf_loss 5.3 dB/cm at 67 GHz (test CPW) | Verified: Fig. 2(d) axis runs to about 67 GHz (tick spacing 8.9 px/GHz, curve ends at 67); measured trace ends at about 5.3 dB/cm. Honest labelling (test structure, extracted_from_figure) | OK; the target is a different structure than the simulated cross-section (27 um signal is an inference), tolerance 20 % reasonable. |

Checked and correct: copper sigma 4.9e7 = 1/2.04e-8 (p.7); 16 mm; 50 ohm; substrate -529.7..-4.7 (525 um Si + 4.7 um BOX); slab 0.28; rib 4 um x 0.32 um (vertical walls disclosed); Cu 1.8 um from the slab top to 2.08 (flat top consistent with p.12-13); waveguides centred in the 6 um gaps (signal -30..-3 => gaps -36..-30 and -3..3, arms at -33 and 0: consistent); wavelength 1550 explicitly project_inference with the "not stated" caveat (honest). The `missing` list is complete and honest, including the unverified constants. Targets tied to evidence (2.7, 2.22) and the paper's COMSOL 2.64 labelled simulated. `sigma_Sm: 4.9e7` loads as a string with PyYAML (verified); write 4.9e+7.

### Verdict: ACCEPT WITH CORRECTIONS (N1 medium; LC1 high for the config only).

---------------------------------------------------------------------
## 5. niels2026 (Nature Photonics 20(2) 225-231; cached text = arXiv v1)

Identity: title, 25 authors, journal/volume/issue/pages, DOI, received 2025-02-13 / accepted 2025-12-06 / first online 2026-01-13 (Crossref); licensing as discussed. Cached source is the arXiv v1 manuscript (batch correct; the paper-level notes say so; PDF creation 2025-03-14, 9 pages with supplementary).

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| N1m | medium | drive=differential, evidence basis `derived` | "GSSG differential feed with oppositely poled arms; authors call it push-pull" | p.3: "the modulator operates in a push-pull configuration, with the printed LiTaO3 membranes oriented 180 degrees relative to one another ... allows ... integrated electronic drivers with a differential output"; p.4: "3.5 V in the push-pull amplitude modulation configuration (both arms of the MZM)"; data transmission p.4: "A differential signal ... applied using a GSSG probe" | Both terms are stated by the authors, so `derived` is wrong in the sense of (f) (inferred values are those the authors do not state). Under (f), "push_pull" = opposite fields in the arms is directly stated; `differential` = feed topology, stated for the data experiment. Choose one and record the other in a note; if differential is kept the evidence basis should be `measured`/stated, not derived. No evidence entry exists for vpi_convention=per_arm_phase_shifter either (allowed, but this is exactly the convention that needs a locator: p.4 "7.0 V for a single phase shifter (one arm)", Fig. 5b). |
| N2m | medium | vpil_dc_vcm (empty) while VpiL 2.3 V cm is the paper's stated figure | left empty "to avoid mixing conventions" | Abstract/p.2, p.4, Conclusions: VpiL = 2.3 V cm (3.5 V x 6.6 mm = 2.31, push-pull). Per-arm 7.0 V x 0.66 cm = 4.62 V cm | Correct decision under (a) (no conversions in the CSV), but the paper's headline figure of merit now exists only in notes, and the derived view will show 4.62 V cm (per arm) next to an author-stated 2.3 V cm. Add the proposed `vpil_mzm_pushpull_vcm` column (BATCH_REPORT gap) or enter 2.3 in a note-matched evidence entry. |
| N3 | medium | version attribution: published_on 2026-01-13, source_type journal, url DOI; all numbers from arXiv v1 | no structured marker of version | arXiv v1 PDF (2025-03); Crossref has no abstract to compare; journal numbers may differ (the batch admits this) | published_on describes the journal object while every value is from the 2025-03 manuscript. Either set published_on to the arXiv v1 date once verified (2025-03-13 per candidates.csv, unverifiable offline) or leave empty, and put "arXiv v1 2503.10557v1" in a structured place (arxiv_id with version, or a tag `numbers_from_arxiv_v1`). Pitfall explicitly listed in SKILL.md ("overlapping arXiv and journal versions"). |
| N4 | low | bw3db_ghz=70 gt, bw_measured_to 70 | per (c) | p.4: "3-dB cut-off frequency in excess of 70 GHz, limited by the experimental setup"; Fig. 4(d) (rendered 2.2x): HR trace starts at about +1 dB near 5 GHz, hovers -1 to -2 dB, and a noisy dip reaches about -3 dB near 65-66 GHz before recovering to -2 at 70 GHz; plot axis ends at 70; EOE normalization (0 dB at about 0 GHz) | Entry follows the text correctly. Add to the note that the trace touches -3 dB in the noise near 65 GHz and that the EOE (VNA S21, electrical-power) definition of "3 dB" is not stated. |
| N5 | low | papers.csv license cell | free text: "publisher terms (Springer Nature text-and-data-mining); arXiv v1 license not verifiable offline" | Crossref license = TDM terms page only | Cell mixes a claim and a caveat; use "publisher TDM terms only; no open license verified" or empty with the caveat in notes. redistribution restricted_local_only is right. |
| N6 | low | foundry_or_fab = imec only | imec | Acknowledgments p.5: imec 200 mm pilot line (Si/SiN wafers); author contributions p.5-6: M.N., T.Z., M.B. (Ghent University-imec) fabricated the hybrid LiTaO3/SiN devices and metallization; Ghent staff thanked for LiTaO3 device fabrication | The LiTaO3 membrane/printing/metal steps were done by the authors at Ghent/imec; no separate facility is named, so the entry is acceptable, but note that imec covers the PIC wafers only. |
| N7 | low | electrode_type=other | other | GSSG, no enum value; tw_gsg would be the nearest but GSSG is a two-signal line | OK; keep `other` and tag `gssg` (already). |
| N8 | low | extinction_ratio_db=28, er_type static | 28 | p.3, Fig. 5a: ER 28 dB at 1310 nm from the wavelength sweep of the passive unbalanced MZI (peak to dip, normalized for GC/routing); Supp.: ER > 25 dB for four devices | Value right; note that it is fringe contrast of the passive interferometer, not a voltage-swept modulation ER. |
| N9 | low | il_onchip_db=2.9 basis derived | sum 0.6 + 0.7 + 1.6 | Supp. Table II: C3 0.6 +-0.2, C4 0.7 +-0.3, C5 1.6 +-0.1 (C5 from COMSOL, p.8); abstract and p.3 state 2.9 dB for the hybrid phase modulators | Correct: derived by the authors, excludes C1 0.8 + C2 3.0 = 3.8 dB PDK and gratings; the "metal-misalignment loss" would not be present in a corrected process (already in includes). |

### Checked and correct (sampled 17)
Vpi 7.0 V single phase shifter / 3.5 V push-pull, 100 kHz triangular, quadrature at 1309.26 nm (p.3-4; Fig. 5b: transmission max near -3.5 V, min near +3.5 V, span 7 V); 2.3 V cm = 3.5 x 0.66; simulated 3.6 V (Eq. 1: r33 30.5, ne 2.1269, Gamma 0.454, neff 1.852, G 5.5 um, L 6.6 mm, p.9) kept out of the Vpi columns; 300 nm film, 30 um x 7 mm membrane, 50 nm BCB, 6.6 mm effective (200 um transitions, p.3); Ti 20 nm / Au 1 um (p.3); SiN 300 nm x 900 nm (p.3, p.8); propagation loss 1.0 +- 0.5 dB/cm and transition 0.3 dB (p.8; 0.7 dB over 6.6 mm = 1.06 dB/cm consistent); ER 28 dB at 1310, FSR 4 nm (p.3, Fig. 5a); bandwidth > 70 GHz limited by VNA and photodetector, estimated 90 GHz kept out (author estimate); NRZ 190 GBd (BER about 5e-3 below the 15 % FEC line at about 2e-2; 200 GBd about 6e-2 above), PAM4 160 GBd (about 1.6e-2, just below the line), 320 Gbit/s = 160 x 2 entered `gt` ("more than 320 Gbit/s", p.5): all read from Fig. 6(b); 256 GS/s AWG with differential output and no driver amplifier stated; 64-tap FFE; projected patterned-LT design (gap 3.4 um, 2.7 V, 1.8 V cm) correctly in notes only; student-driver and HR vs standard Si ambiguity for the Vpi/IL/ER device flagged. Locators match page markers (p.3 design/IL, p.4 Vpi/bandwidth/data, p.5 conclusions and Fig. 6, p.8-9 supplementary).

### Sim: none (grade C, electrode widths and oxide thicknesses not reported: agrees).

### Verdict: ACCEPT WITH CORRECTIONS (three medium items; no wrong numeric cell found).

---------------------------------------------------------------------
## 6. Organizations (convention e)

| Item | Finding |
|---|---|
| Swiss Federal Institute of Technology Lausanne | Paper spelling: "Swiss Federal Institute of Technology Lausanne (EPFL)" (lin2025, li2026a) and "Swiss Federal Institute of Technology, Lausanne (EPFL)" (li2026, li2026ba); the comma is dropped and the acronym removed: consistent with (e). No existing row to reuse. Risk: later batches may write "Ecole Polytechnique Federale de Lausanne"; integrator should unify. |
| Karlsruhe Institute of Technology | Spelled out; li2026a gives "Karlsruhe Inst. of Technol. (KIT)". OK. |
| EPFL Center of MicroNano Technology / EPFL Institute of Physics cleanroom | facility, parent set, country CH from host institution with note: correct per (e). The second name is an inference from "the Institute of Physics (IPHYS) cleanroom" (the paper does not write "EPFL" in front); the note should say so. Note the existing facility row "Binnig and Rohrer Nanotechnology Center" has an empty parent; the two styles differ. |
| Luxtelligence SA | Competing-interest spelling "Luxtelligence SA, St. Sulpice"; affiliation spelling LUXTELLIGENCE SA (li2026a, lin2025). Normal case chosen: fine. Listed in `companies` only for li2026a: correct. |
| Ghent University | Affiliation is "Ghent University-imec" (Department of Information Technology, INTEC). Stored as Ghent University plus imec: fine. |
| imec | Written "imec" (lowercase) in the paper; kept as the institution's own name, not expanded (not a pure acronym in current usage). org_type research_institute is used elsewhere in the DB ("Institute of Physics, Chinese Academy of Sciences") but not enumerated (prior audit X7). Placing a research_institute in papers.csv `companies` is consistent with the column description (non-university). |
| Countries | CH;DE for the Lausanne/Karlsruhe papers; BE for niels2026: correct. |

---------------------------------------------------------------------
## 7. Convention and skill ambiguities (for the coordinator)

| # | Issue | Evidence |
|---|-------|----------|
| X1 | The Pockels / RF-permittivity constants for LiTaO3 have no verified source in the repo; configs use `standard_reference` + a citation for recalled values. Need a provenance class for "recalled, unverified" (raised twice before as X9, S2, D4) and a rule that a citation may be given only if the source was opened. SPEC example values (sigma 4.1e7, LN constants) are being copied as `standard_reference`. | BC1, BC2, LC1 |
| X2 | Convention (a) has no home for a paper-reported VpiL that belongs to the MZM push-pull value while the headline Vpi is per arm: add `vpil_mzm_pushpull_vcm` or allow an evidence-only entry. | N2m |
| X3 | Unidentified device for system-demo results (lin2025-c): no rule on which fields may be filled, whether shared geometry is copied, and how views avoid orphan rows. | N3 |
| X4 | Convention (f) wording "inferred values ... basis derived" does not say what to do when the authors state both "push-pull" (arm fields) and "differential" (feed): which one is `drive`. | N1m |
| X5 | Text-only sources (li2026a): no page index exists; need an allowed locator form ("Sec. N") in (i). | A3 |
| X6 | Basis for "authors state the architecture" (drive, push-pull): the enum offers `design_target` or `measured`; both are used for the same kind of statement across batches. | L3, N6 |
| X7 | `integration` rule for a commercial thin-film wafer that the authors re-bond to another carrier (li2026ba) vs LTOI on Si (li2026): monolithic vs other vs bonded_heterogeneous. | L5, B3 |
| X8 | Convention (c) and a claimed crossing at the end of smoothed data (li2026): the smoothed fit may extend past the raw data; state that bw_measured_to_ghz is the end of raw data. | L1 |
| X9 | Version handling: published_on / source_type describe the journal while the cached numbers come from the arXiv manuscript (niels2026); need a structured `numbers_from` marker or a rule on published_on for this case. | N3 |
| X10 | Text-vs-figure disagreement for a stated bandwidth (lin2025-a 40 GHz text, about 50 GHz figure): rule that the text value is entered and the discrepancy goes into the evidence note. | N1 |
| X11 | Same-group sibling papers (li2026a / li2026ba / li2026 / lin2025) have no relation field; views double count. | A5 |

---------------------------------------------------------------------
## Counts

| Paper | high | medium | low | verdict |
|-------|------|--------|-----|---------|
| li2026 (rows, papers) | 0 | 1 | 4 | accept with corrections |
| li2026a (text-only) | 0 | 0 | 5 | accept |
| li2026ba (rows, papers) | 0 | 0 | 4 | accept with corrections |
| sims/li2026ba/config.yaml | 1 | 1 | 3 | accept with corrections |
| lin2025 (rows, papers) | 0 | 1 | 5 | accept with corrections |
| sims/lin2025/config.yaml | 1 | 0 | 2 | accept with corrections |
| niels2026 | 0 | 3 | 6 | accept with corrections |
| Total | 2 | 6 | 29 | none rejected |

Cross-paper convention/skill issues: 11 (X1-X11). Most important: BC1/LC1 (recalled LT constants carry a citation that is probably for LiNbO3 and are labelled standard_reference), N1m/N2m/N3 for niels2026 (drive basis, lost 2.3 V cm headline, version vs published_on), lin2025-a 40 GHz text vs about 50 GHz figure, li2026 missing bw_measured_to.
