# Advanced packaging literature triage (2022-2026)

Generated 2026-10-18 from pipeline state. Spec: `init-prompt.md`. Plan and log: `DevLog/DevLog-000-literature-triage-plan.md`. Deliverables: `outputs/candidates.csv`, `outputs/cache_list.csv`.

## Summary

- Records enumerated: 10342 unique DOIs. Excluding front matter (449) and duplicate index entries (308; the two sets overlap) leaves 9886 papers screened.
- Prefilter kept 4412 of the screened papers; 4413 screened papers carry an LLM score (a few scored papers were later dropped by the final prefilter and keep their score).
- LLM score >= 7 with a bucket: 428; of these 207 cached. The cap forced 257 drops.
- Cached: 300 (cap 300). Protected (spec-list technologies, overrides, anchored lineages): 82. Outside the screened set (pre-2022 or unscreened venue, spec-list only): 19. Cached with LLM score below 7: 74 (spec-list papers, overrides, lineage members and curator picks; see per-bucket table).
- Not cached despite LLM score >= 9: 49 (cap with per-bucket balancing; listed in candidates.csv with cache=N).
- Priority split: 1 = 171, 2 = 90, 3 = 39. Lineage ids written (complete series only): 19 of 71 series proposed (15 anchored by a spec-list paper and protected).
- Per-bucket floor (10 papers, 3 orgs, 3 years): met in all buckets; threshold lowering needed: none.

## Spec-list technologies (cache regardless of score)

| Technology | Cached papers selected for it (year, DOI; from selection_reason) | Gap / caveat |
|---|---|---|
| CoWoS-S (original CoWoS) | 2011 10.1109/ectc.2011.5898527; 2013 10.1109/ectc.2013.6575597; 2023 10.1109/ectc51909.2023.00174 |  |
| CoWoS-R | 2021 10.1109/ectc32696.2021.00125; 2022 10.1109/ectc51906.2022.00008; 2024 10.1109/ectc51529.2024.00049; 2025 10.1109/ectc51687.2025.00047 |  |
| CoWoS-L | 2023 10.1109/ectc51909.2023.00174; 2023 10.1109/ectc51909.2023.00091 |  |
| InFO-oS | 2020 10.1109/itherm45881.2020.9190542; 2021 10.1109/ectc32696.2021.00033; 2022 10.1109/ectc51906.2022.00232 | No TSMC-authored in-window InFO_oS paper in the screened set; MediaTek 2022 fan-out-on-substrate paper attributed by inference. |
| InFO-LSI | none | No peer-reviewed disclosure found; only TSMC Hot Chips 33 (2021) tutorial and press. |
| SoIC | 2019 10.1109/ectc.2019.00095; 2022 10.1109/ectc51906.2022.00176; 2022 10.1109/vlsitechnologyandcir46769.2022.9830182; 2025 10.1109/ectc51687.2025.00055 |  |
| EMIB | 2016 10.1109/ectc.2016.201; 2023 10.1109/tcpmt.2023.3312037 |  |
| EMIB-T | 2025 10.1109/ectc51687.2025.00051; 2026 10.1109/ectc51846.2026.00051 |  |
| Foveros | 2019 10.1109/iedm19573.2019.8993637; 2024 10.1109/vlsitechnologyandcir46783.2024.10631478 |  |
| Foveros Direct | 2021 10.1109/ectc32696.2021.00166; 2022 10.1109/iedm45625.2022.10019499; 2024 10.1109/ectc51529.2024.00017 | No peer-reviewed paper uses the brand; Intel hybrid-bonding papers mapped by inference. |
| Foveros Omni (ODI) | 2019 10.1109/iedm19573.2019.8993659 | Omni = ODI rebrand per trade press only. |
| Intel glass core | 2025 10.23919/icep-iaac64884.2025.11003038; 2026 10.1109/ectc51846.2026.00134 | Only two Intel glass-core papers found (ICEP-IAAC 2025 short paper, ECTC 2026). |
| Samsung I-Cube | 2021 10.1109/ectc32696.2021.00315; 2022 10.1109/ectc51906.2022.00108; 2025 10.1109/ectc51687.2025.00048 | I-Cube S attribution to the 2021 MIoS paper is low confidence (brand not named). |
| Samsung X-Cube | 2020 10.23919/iwlpc52010.2020.9375855; 2023 10.1109/ectc51909.2023.00178; 2024 10.1109/ectc51529.2024.00064 | In-window papers do not use the X-Cube name; attribution inferred. |
| Samsung H-Cube | none | No peer-reviewed disclosure found; 2021-11-11 press release only. |
| SK hynix HBM3E stack | 2023 10.1109/ectc51909.2023.00151 | No peer-reviewed 12-high HBM3E (Advanced MR-MUF) stack paper; earliest MR-MUF paper (ECTC 2023, 8-high) cached instead. |
| SK hynix HBM4 stack | 2026 10.1109/vlsitechnologyandcir65830.2026.11577576 | VLSI 2026 12-layer test vehicle; bonding method not stated in abstract. |
| Absolics/SKC glass substrate | 2024 10.1109/ectc51529.2024.00092 | Only one Absolics paper (ECTC 2024, TGV etch); no peer-reviewed stack-up or panel size. |
| FOPLP lines >= 500 x 500 mm | 2019 10.1109/ectc.2019.00-29; 2020 10.23919/iwlpc52010.2020.9375887; 2020 10.1109/ectc32862.2020.00185; 2020 10.4071/2380-4505-2020.1.000042; 2021 10.1109/eptc53413.2021.9663896; 2021 10.1109/ectc32696.2021.00174; +8 more | No peer-reviewed paper for Powertech, Innolux, JCET or any Samsung >= 500 mm line (press only). |

## Venue enumeration

| Venue | 2022 | 2023 | 2024 | 2025 | 2026 | Source | Notes |
|---|---|---|---|---|---|---|---|
| ECTC | 369 | 376 | 391 | 389 | 424 | IEEE Xplore | Sessions/committees from ectc.net final programs 2022-2026 |
| TCPMT | 296 | 305 | 339 | 352 | 402 | IEEE Xplore | Journal; year = Xplore publicationYear (early-access records included) |
| EPTC | 203 | 212 | 245 | 239 | 0 | IEEE Xplore | 2026 not yet posted on Xplore (2026-10-18) |
| 3DIC | 0 | 16 | 35 | 0 | 0 | IEEE Xplore | Only 2023 and 2024 found on Xplore; 2022/2025/2026 not found (not held or not yet indexed) |
| IEDM | 230 | 231 | 277 | 298 | - | IEEE Xplore | Filtered venue: full TOC fetched, packaging/3D topic filter (session data not in Xplore) |
| VLSI | 215 | 241 | 249 | 282 | 255 | IEEE Xplore | Filtered venue: Technology and Circuits combined proceedings; topic filter |
| EDAPS | 45 | 39 | 39 | 33 | - | IEEE Xplore | Filtered venue: cross-section disclosure required (scoring penalizes pure modeling) |
| EPEPS | 36 | 61 | 72 | 56 | - | IEEE Xplore | Filtered venue: as EDAPS |
| IRPS | 176 | 188 | 177 | 209 | 209 | IEEE Xplore | Filtered venue: test-vehicle stack required |
| ITherm | 195 | 160 | 244 | 202 | 0 | IEEE Xplore | Filtered venue; 2026 not yet on Xplore (2026-10-18) |
| ICEP | 129 | 139 | 174 | 181 | - | IEEE Xplore | 2025 = ICEP-IAAC |
| ESTC | 117 | 0 | 157 | - | - | IEEE Xplore | Biennial (even years); 2023 not held |
| EMPC | 0 | 84 | 0 | - | - | IEEE Xplore | Biennial (odd years); 2022/2024 not held |
| IWLPC | 18 | 21 | 18 | 13 | - | Crossref (SMTA prefix 10.37665, 'Wafer-Level Packaging Symposium' vols 19-22) | Not on Xplore; Crossref holds only part of each proceedings; edition-to-year mapping inferred (IWLPC 2017 = 14th, proceedings.com) |
| IMAPS | 84 | 54 | 76 | 65 | - | Crossref (prefix 10.4071, IMAPSource Proceedings, issue Symposium) | meridian.allenpress.com behind Cloudflare (403); Crossref Symposium records only (likely partial) |

Counts are unique records including front matter. '-' = year not in scope for that venue; '0' = not found in the source.

- IEEE Xplore: venue-year search by publication title, 100 rows/page, then completeness check against each proceedings/issue TOC totalRecords (all matched; +2 T-CPMT records backfilled). Records whose Xplore publicationYear falls outside a venue's scope were dropped (4 T-CPMT strays).
- Xplore abstracts in search/TOC responses are truncated (~400 chars); full abstracts from OpenAlex (by DOI), else Xplore /rest/document/<id>/abstract. 237 of 8613 OpenAlex abstracts compared (2.8%) were placeholders or swapped and were replaced.
- Fraunhofer/imec/ITRI technical reports: not enumerated as a separate source; their conference papers are covered through the venues above.
- Venues outside the screened list appear only as curator-verified primary disclosures (outside_screen = 'unscreened venue', e.g. ICEPT, ISSCC) or in the reference table.
- IWLPC and IMAPS records have no session data; access is member/subscription via publisher.
- ECTC session match: 1840 of 1949 records (94.4%) mapped to a program session (title prefix match against ectc.net final-program PDFs); unmatched rows have an empty session.

## Per-bucket summary

| Bucket | Screened (LLM-assigned) | LLM >= 7 | Cached | Cached < 7 | Cached orgs | Cached 2022/23/24/25/26 | Outside screen |
|---|---|---|---|---|---|---|---|
| 2.5D silicon interposer | 217 | 30 | 43 | 13 | 28 | 7/11/8/9/5 | 3 |
| Organic/RDL interposer and large-reticle fan-out | 509 | 122 | 43 | 5 | 18 | 7/9/6/14/4 | 3 |
| Local silicon bridge in organic substrate | 108 | 24 | 32 | 12 | 16 | 3/6/5/10/7 | 1 |
| Panel-level packaging | 124 | 45 | 43 | 5 | 23 | 7/8/7/9/6 | 7 |
| Hybrid bonding | 612 | 146 | 43 | 5 | 18 | 7/10/12/5/7 | 2 |
| Micro-bump / 3D stacking with solder | 272 | 66 | 43 | 2 | 23 | 8/9/9/5/9 | 3 |
| Glass core / glass substrate | 220 | 35 | 44 | 10 | 27 | 6/3/6/11/18 | 0 |
| Advanced substrates | 191 | 37 | 43 | 13 | 26 | 4/6/9/14/10 | 0 |
| Silicon photonics co-packaging | 45 | 16 | 26 | 14 | 19 | 1/5/6/9/5 | 0 |
| Thermal/power-delivery structures integral to the stack | 49 | 18 | 35 | 17 | 21 | 3/7/5/9/11 | 0 |

Buckets overlap (a paper can count in several). Primary bucket per paper is in cache_list.csv.

### 2.5D silicon interposer (`si_interposer`)

- Dominant organizations (cached): A*STAR IME (6), Resonac (4), Fraunhofer IZM (3), imec (3), Samsung (2), TSMC (2)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: Resonac-JOINT2-2.5D-size, Samsung-SiInterposer-size
- Gap: TSMC CoWoS-S interposer-size lineage (1x -> 1.5x -> 2x -> 3.3x reticle): only one in-window TSMC CoWoS-S paper in input (ECTC 2023, 2500 mm2 by four-mask stitching); no generational series formable. Other TSMC CoWoS papers in pool sit outside this input (e.g. 10.1109/irps61424.2026.11499180 [not cached], 10.1109/ectc51906.2022.00095 [not cached] per disclosures_tsmc.json).
- Gap: Samsung I-Cube S: no peer-reviewed paper found that names I-Cube S and discloses its stack; MIoS (ECTC 2021/2022) is the probable but unbranded disclosure (low confidence).
- Gap: TSMC SoIS: peer-reviewed disclosure exists (ECTC 2021, 10.1109/ectc32696.2021.00016 [not cached]) but it is a wafer-processed integrated substrate, not a Si interposer; no in-window SoIS paper in pool.
- Gap: AMD/Xilinx interposer: no multi-paper series in input; Xilinx 28 nm SSI (ECTC 2011) is out of window; MI300 primary disclosure is ISSCC 2024 (unscreened venue).
- Gap: Large-interposer reticle stitching: no dedicated process paper in input; stitching is only mentioned in TSMC ECTC 2023 and Samsung ECTC 2023 abstracts.
- Unverified: CoWoS-S second generation: DOI not checked (per disclosures_tsmc.json); not resolved here.
- Unverified: CoWoS-S (thin interposer variant): DOI verified in disclosures_tsmc.json, but abstract does not name CoWoS.
- Unverified: Samsung I-Cube S / I-Cube4 / I-Cube8 brand: Press only; I-Cube4 described as mold-free, so MIoS-to-I-Cube mapping unconfirmed. EPEPS 2024 (10.1109/epeps61853.2024.10753887) names I-CubeS but is SI-only and not in input.
- Unverified: Lightmatter Passage M-series: Crossref-listed talk record, not a peer-reviewed paper; content not checked.

### Organic/RDL interposer and large-reticle fan-out (`organic_rdl_fanout`)

- Dominant organizations (cached): TSMC (8), ASE/SPIL (6), Samsung (5), Amkor (4), Resonac (3), MediaTek Inc (2)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: ASE-FOPLP-panelsize, Samsung-PanelRDL-interposer_LS_size, TSMC-CoWoS-R-RDL_LS_and_size, TSMC-CoWoS-R-RDL_LS_and_size;TSMC-IMCSi-power, nepes-FOPLP-600mm
- Gap: InFO-oS: no in-window TSMC-authored InFO_oS paper in pool; in-window MediaTek candidates do not name InFO_oS. Primary disclosures are pre-window (ITherm 2020 MediaTek; ECTC 2021 TSMC).
- Gap: CoWoS-R primary disclosure (ECTC 2021) is pre-window; in-window coverage begins with CoWoS-R+ (2022) whose pool score (2) understates it.
- Gap: No in-window base InFO / InFO_PoP / InFO_LSI TSMC papers in pool; InFO_LSI has no peer-reviewed disclosure found (disclosures_tsmc.json).
- Gap: SPIL FO-MCM pre-window primary disclosure not confirmed (IMPACT 2020 candidate, affiliation address-only).
- Gap: Amkor SWIFT base generation and HDFO naming: only pre-window IMAPS DPC 2015 / IWLPC 2016 records; no in-window SWIFT (non-S) process paper in pool.
- Gap: JCET XDFOI only appears as CPI/warpage studies (10.4071/001c.74581 [not cached], 10.1109/ectc51687.2025.00249 [not cached]); no detailed process disclosure in pool.
- Gap: Thin year for vendor CoWoS-R process detail: 2023 (no TSMC CoWoS-R process paper in input).
- Unverified: SPIL FO-MCM primary disclosure: DOI verified via Crossref; OpenAlex gives only an address (No. 153, Sec. 3, Chung-Shan Rd., Tantzu, Taichung), SPIL attribution inferred; warpage study, not a platform description. EPTC 2021 10.1109/eptc53413.2021.9663928 (FOMCM X-ray) also exists, affiliation not checked.
- Unverified: ASE FOCoS (possible earlier in-window paper not in pool): Found in Crossref search only; not in pool; issued date may correspond to the 2021 symposium. Content not checked.
- Unverified: InFO-oS in-window attribution: Neither names InFO_oS in title/abstract; attribution inferred from structure; confirm in full text.
- Unverified: Amkor S-SWIFT pre-2022: Crossref bibliographic search returned no S-SWIFT-named paper before ECTC 2022; Amkor marketing material not checked.

### Local silicon bridge in organic substrate (`si_bridge`)

- Dominant organizations (cached): Intel (10), IBM (4), AOI Electronics (3), ASE/SPIL (2), A*STAR IME (2), TSMC (1)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: Intel-EMIB-integration, Intel-EMIB-integration;Intel-GlassCore-layercount, Intel-PanelHI-process, Samsung-PanelRDL-interposer_LS_size, nepes-FOPLP-600mm
- Gap: InFO-LSI (InFO_LI, Apple UltraFusion): no peer-reviewed disclosure found (disclosures_tsmc.json); only TSMC Hot Chips 33 (2021) tutorial and press.
- Gap: CoWoS-L LSI die itself (LSI BEOL, TSV in LSI, eDTC dimensions): no dedicated TSMC paper beyond ECTC 2023 overview; detail must come from full text.
- Gap: EMIB base technology: no in-window Intel paper in this input states bump pitch; in-window EMIB papers ectc51906.2022.00199 (score 3), iedm45625.2022.10019355 (1), ectc51529.2024.00050 (2), ectc51687.2025.00312 (3) are in pool but below threshold. EMIB-T Q&R irps61424.2026.11499343 (score 3) and EMIB-T HBM4E ectc51846.2026.00078 (0) also below threshold.
- Gap: Below-threshold bridge papers in pool that a selector may still want: ectc51906.2022.00337 SPIL FO-EB (2), ectc51909.2023.00047 SPIL FO-EB CPO (3), ectc51846.2026.00052 ASE FOCoS vs FOCoS-B warpage (0), ectc51846.2026.00101 Intel SRAM in FO embedded bridge (2), eptc67330.2025.11392386 Deca embedded bridge (3), ectc51687.2025.00138 PSB 5 um pitch vias (6, bucketed organic_rdl_fanout only).
- Gap: ASE FOCoS-B: pre-window origin named sFOCoS (ECTC 2021); FOCoS-B brand mapping from Chip Scale Review 2022, not from a peer-reviewed source.
- Gap: JCET XDFOI bridge variant: no first disclosure identified; earliest XDFOI papers found (IMAPS 2023 CPI) do not establish bridge content.
- Gap: No Samsung I-Cube E paper before ECTC 2025; earlier Samsung bridge activity only in press.
- Gap: Thin coverage: TSMC only 2023 and 2025 in input; 2022 dominated by IBM DBHi and SPIL FO-EB.
- Unverified: InFO-LSI (InFO_LI): From disclosures_tsmc.json: DOI verified but book chapter; InFO_LSI coverage not checked.
- Unverified: InFO-LSI (InFO_LI): Not peer-reviewed; no conference/journal paper found (disclosures_tsmc.json).
- Unverified: CoWoS-L: From disclosures_tsmc.json: secondary snippet claims CoWoS-L presented; abstract does not mention it; full text not checked.
- Unverified: Deca M-Series Embedded Bridge (MEB) name: LLM flag names MEB; whether this is the first use of the MEB brand was not verified.
- Unverified: EMIB pre-disclosure: From disclosures_intel.json: predates EMIB name; full text not read.

### Panel-level packaging (`panel`)

- Dominant organizations (cached): ASE/SPIL (6), Samsung (5), Nepes (4), Resonac (4), Intel (3), Institute of Microelectronics (2)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: ASE-FOPLP-panelsize, Intel-GlassCore-layercount, Intel-PanelHI-process, Samsung-PanelRDL-interposer_LS_size, nepes-FOPLP-600mm
- Gap: No in-window primary disclosure for any >= 500x500 FOPLP line from Samsung, PTI (Powertech), Innolux, or JCET; trade press only.
- Gap: No peer-reviewed source states a built ASE 700x700 mm panel (target only, ICEP 2025).
- Gap: ST 700x700 papers are FEA-centric; no ST stack-up or process flow paper found.
- Gap: Amkor has a single in-window panel paper (ECTC 2022); no generation follow-up found.
- Gap: Samsung panel size never stated in any in-window abstract (only 415x510 via trade press on ECTC 2018, below threshold).
- Gap: Deca 600 mm in-window papers not in pool: 10.4071/001c.116490 [not cached] (IMAPS 2024); Deca Gen 2 M-Series (10.1109/ectc51906.2022.00138 [not cached]) and MFIT (10.1109/ectc51687.2025.00091 [not cached]) are in organic_rdl_fanout input, not panel.
- Gap: Panel yield data (line-level yield vs panel size) is sparse; only layer yield (Samsung 2023) and QFN yield (nepes 2024) in abstracts.
- Unverified: FOPLP line: Amkor 650x650 mm: Crossref search found none with 650 mm panel; not exhaustively checked (carried from disclosures file).
- Unverified: FOPLP line: ASE 600x600 mm: DOI verified, not in pool; OpenAlex abstract identical to ICEP 2024 abstract, distinctness unconfirmed.
- Unverified: FOPLP line: ASE 700x700 mm: Only a future-target mention; no paper reports a built ASE 700x700 panel.
- Unverified: FOPLP line: nepes 600x600 mm: DOI verified via Crossref; not in pool; abstract not read.
- Unverified: Deca M-Series Direct (MDx): Abstract says MDx is new; one Crossref keyword search found no earlier MDx paper, but Deca IMAPS/IWLPC records not exhaustively checked.
- Unverified: Intel panel-level heterogeneous integration (PHI) panel size: Abstract gives area only (~2600 cm2, consistent with 510x515 mm) not edge lengths; not counted as a >= 500x500 line disclosure.
- Unverified: Samsung FOPLP >= 500x500 mm: Trade press only; no in-window Samsung abstract states panel size (carried from disclosures file).
- Unverified: Resonac consortium line (JOINT2): Input lineage field says JOINT2; abstracts do not name a consortium.

### Hybrid bonding (`hybrid_bonding`)

- Dominant organizations (cached): Intel (5), TSMC (5), Sony (5), Samsung (4), Applied Materials (4), CEA-Leti (4)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: Intel-HB-pitch, Samsung-SiInterposer-size, TSMC-SoIC-pitch
- Gap: Foveros Direct: no peer-reviewed paper names it; Intel HBI ECTC 2021 is the primary by inference, and all in-window Intel HB papers are mapped by inference (many are Components/Technology Research, not product flow).
- Gap: Samsung X-Cube bumpless: no branded paper; ECTC 2023 D2W HCB (10.1109/ectc51909.2023.00178 [cached]) is a medium-confidence mapping.
- Gap: SK hynix: no 12H/16H hybrid-bonded HBM stack paper in input; only 8Hi C2W (2023). HBM4 VLSI 2026 paper (in pool, other bucket) does not state bonding method in abstract.
- Gap: Micron: no HBM-specific hybrid-bond stack disclosure found; only generic C2W memory stacking (2023) and W2W carrier (2025).
- Gap: TSMC SoIC-X / SoIC-P naming appears in no verified paper; SoIC HBM 12/16-Hi (VLSI 2020) is out of window.
- Gap: Sony Cu-Cu hybrid bonding primary disclosure (pre-2022 CIS work) not researched here; not mandatory.
- Unverified: Intel Foveros Direct: No pool title/abstract uses the term; all Intel HB -> Foveros Direct mappings are inferences. Intel IEDM 2021 HB talk (press-reported) not located in Crossref.
- Unverified: Samsung X-Cube bumpless: OpenAlex title/abstract search for X-Cube (2020-2026) returns no packaging paper; only foundry-forum/press statements exist.
- Unverified: SK hynix HBM hybrid bonding (12H/16H, HBM4): Crossref-verified in disclosures_samsung_skhynix.json, in pool but not in this bucket's input; abstract does not state hybrid bonding vs MR-MUF.
- Unverified: Micron HBM hybrid bonding: None found via OpenAlex affiliation + 'hybrid bonding' search; Micron in-pool HB work is generic memory C2W (2023) and W2W carrier (2025). Micron DCB bond-energy paper 10.1109/ectc51529.2024.00062 is in pool but not in this input.

### Micro-bump / 3D stacking with solder (`microbump_3d`)

- Dominant organizations (cached): Intel (5), Samsung (4), imec (4), Resonac (3), ASE/SPIL (3), SK hynix (3)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: Intel-EMIB-integration, Samsung-3DIC-microbump-pitch, Samsung-SiInterposer-size, imec-CuSnMicrobump-pitch
- Gap: SK hynix HBM3E 12-high (Advanced MR-MUF): no peer-reviewed primary disclosure; press only. The earliest named MR-MUF paper is 8Hi (ECTC 2023 .00151).
- Gap: SK hynix HBM4: the only paper (VLSI 2026 .11577576) is in pool but not in this bucket's input. Bonding method (advanced MR-MUF vs hybrid) is not stated in the abstract. No 16-high HBM4 stack paper exists.
- Gap: Micron HBM3E TC-NCF stack: no peer-reviewed Micron HBM stack-process paper found; press only. The only Micron NCF process paper in input is EPTC 2022 slow-cure NCF (no HBM named).
- Gap: Foveros Omni (ODI): no in-window paper in this input. ODI-linked RDL-on-mold paper 10.1109/ectc51529.2024.00099 [not cached] sits in the organic_rdl_fanout input. Omni = ODI rests on trade press only.
- Gap: Foveros 25 um pitch generation: trade press only. In-window pitch data are 36 um only (VLSI 2024). The first-gen 50/55 um is only in IEDM 2019 (out_of_window).
- Gap: Samsung HBM TC-NCF: no pre-window primary identified. In-window process detail is limited to the ECTC 2022 NCF flow simulation (pool, score 0) and ECTC 2023 .00133 (thermal).
- Gap: Samsung X-Cube (micro-bump): no in-window paper uses the X-Cube name. The lineage rests on Samsung AVP micro-bump CoW papers.
- Unverified: SK hynix HBM3E 12-high stack: Crossref-verified circuit paper; it is 16-high, not 12-high, and the packaging process is not confirmed. Not in pool (disclosures file).
- Unverified: SK hynix MUF TSV stack (pre-window): Crossref-verified; MR-MUF/HBM not named in abstract; pre-window (disclosures file).
- Unverified: Samsung X-Cube detailed stack (pre-window): Crossref-verified; earliest detailed SRAM-on-logic stack but the abstract does not name X-Cube (disclosures file).
- Unverified: Micron HBM3E TC-NCF stack: Press only. Web and Crossref searches (2026-10-18) found no Micron peer-reviewed HBM3E stack paper. Micron VLSI 2025 'High Bandwidth Memory for AI' (10.23919/vlsitechnologyandcir65189.2025.11074880, in pool) is an architecture overview with no packaging detail in its abstract.
- Unverified: Foveros Omni 25 um bump pitch: Trade press only; no peer-reviewed paper reporting 25 um Foveros/Omni pitch found.
- Unverified: Amkor R-LAB priority: Crossref title found; content not checked; it may precede EPTC 2022-12 for the R-LAB concept.
- Unverified: Samsung HBM TC-NCF primary: Earliest in-window Samsung HBM TC-NCF process description found. It is simulation-centric, score 0, and not in input. No pre-window primary was identified.

### Glass core / glass substrate (`glass`)

- Dominant organizations (cached): Georgia Tech PRC (9), Corning (4), DNP (4), Intel (2), Inc. (2), Pennsylvania State University (2)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: GTPRC-GlassCoreEmbed-diethickness, GTPRC-GlassPackage-size, Intel-EMIB-integration;Intel-GlassCore-layercount, Intel-GlassCore-layercount
- Gap: Absolics/SKC: no peer-reviewed stack-up, panel size, core thickness, or build-up disclosure; only TGV etch (ECTC 2024).
- Gap: Intel glass core: only a 2-page ICEP 2025 warpage study plus one ECTC 2026 platform paper; press figures (510x515 mm panel, layer counts) not checked against full text.
- Gap: Samsung Electro-Mechanics, LG Innotek, Ibiden, TSMC: no glass-core-authored paper in input; SEMCO Crossref search found none.
- Gap: Schott and AGC appear only as glass suppliers/co-authors (GT 2022 AF32, UTokyo ENA1, Intel ICEP 2025); no first-party glass core platform paper.
- Gap: LPKF LIDE primary disclosure pre-window and not verified (ESTC 2014 candidate lacks affiliation and LIDE name).
- Gap: Corning: no single glass-core-for-HPC substrate disclosure; Corning output is CPO glass/IOX waveguides and die-embedded multilayer glass.
- Unverified: LPKF LIDE: Abstract describes laser-induced chemical etching TGV process but does not use the name LIDE; Crossref/OpenAlex list no affiliation, so LPKF attribution not verified.
- Unverified: Intel glass core substrate: Carried from disclosures_intel.json; not in pool or content unread; cannot rule out an earlier Intel non-first-author glass paper.
- Unverified: Absolics/SKC glass substrate: Carried from disclosures_glass_panel.json; none found, absence not proven for non-indexed venues.
- Unverified: Samsung Electro-Mechanics glass core substrate: Crossref bibliographic search (2026-10-18) returned no SEMCO-authored glass core paper; none in input.

### Advanced substrates (`advanced_substrate`)

- Dominant organizations (cached): Georgia Tech PRC (5), Resonac (3), Shinko (2), IBM (2), Luleå Technical University (2), TAIYO INK MFG.CO. (2)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: GTPRC-GlassCoreEmbed-diethickness, Intel-EMIB-integration;Intel-GlassCore-layercount, UCLA-HybridWiringLayer-pitch, USHIO-SAPLitho-pitch
- Gap: Samsung H-Cube: no peer-reviewed primary disclosure found (press release 2021-11-11 only); no H-Cube-named paper in pool or input.
- Gap: No first-party fine-line process paper in input from Ibiden, Kyocera, Samsung Electro-Mechanics, Nan Ya, Kinsus, or AT&S (AT&S only as Fraunhofer co-author); SEMCO's only pool paper (10.1109/eptc56328.2022.10013168 [not cached]) is inspection, score 1.
- Gap: No vendor paper disclosing Ajinomoto ABF generation roadmap (GX-T31/GL-102 etc.) as a process; ABF appears only as material in third-party papers.
- Gap: Applied Materials Si-core earliest paper is 10.1109/ectc51529.2024.00053 [not cached] (ECTC 2024, SI/PI modeling; pool score 0, not in input).
- Gap: Saras STile original 10.4071/001c.129008 [not cached] (IMAPS, Crossref-verified, published 2025-01-23) is absent from pool; input has only the later eVR STile paper (score 2, no dimensions).
- Gap: Shinko i-THOP 2.3D organic interposer in-window paper (10.23919/icep-iaac64884.2025.11002874 [not cached]) sits in organic_rdl_fanout input, not here.
- Unverified: Samsung H-Cube (hybrid substrate: fine-pitch ABF substrate + large HDI board): Press only; no peer-reviewed paper naming H-Cube found (OpenAlex search 2026-10-18 returns only reviews; prior group search same).
- Unverified: Samsung H-Cube: Crossref-verified by group search; same substrate-split idea but PLP RDL interposer, link to H-Cube unconfirmed; not in input.
- Unverified: Samsung H-Cube: Unbranded, SI-only; not in this bucket's input; no stack-up disclosure.

### Silicon photonics co-packaging (`cpo`)

- Dominant organizations (cached): A*STAR IME (3), Tohoku/AIST/Japan inst (2), TSMC (2), Ltd. (2), IBM (2), Intel (2)
- Thin years (<= 2 cached in-window): 2022
- Lineages cached as series: Intel-EMIB-integration
- Gap: In pool but excluded from input (stack_detailed=false or score<4); selector may override: 10.1109/ectc51909.2023.00108 [cached] TSMC iOIS (COUPE2.0 + CoWoS composite interposer, earliest in-window COUPE-family paper); 10.1109/ectc51529.2024.00144 [cached] TSMC COUPE with SoIC bond; 10.1109/iedm50854.2024.10873589 [not cached] TSMC EPIC-BOE; 10.1109/ectc51687.2025.00062 [cached] Celestial AI OMIB assembly process (score 6, process flow); 10.1109/ectc51846.2026.00311 [not cached] Marvell OMIB ATV (score 6); 10.1109/ectc51687.2025.00012 [cached] Intel EMIB fiber CPO; 10.1109/vlsitechnologyandcir46769.2022.9830221 [not cached] Intel FPGA + Ayar TeraPHY; 10.1109/vlsitechnologyandcir46769.2022.9830432 [not cached] POET Optical Interposer; 10.1109/ectc51906.2022.00072 [cached] IBM/GF self-aligned SMF attach; 10.1109/ectc51529.2024.00185 [not cached] IBM FAU HVM process; 10.1109/ectc51846.2026.00107 [not cached] GF/Corning GLASSBRIDGE; 10.1109/ectc51846.2026.00112 [not cached] Marvell PF chiplet FAU receptacle; 10.1109/ectc51687.2025.00063 [cached] Corning CPO glass substrate fab (glass bucket, score 7).
- Gap: Broadcom CPO: no peer-reviewed packaging stack disclosure found (51.2T paper has no stack detail).
- Gap: Intel OCI (optical compute interconnect) chiplet: no peer-reviewed packaging paper found in pool.
- Gap: NVIDIA Spectrum-X/Quantum-X Photonics CPO package: no stack-detailed peer-reviewed paper in input.
- Gap: Ayar Labs TeraPHY: no in-window stack-detailed paper in input; primary (OFC 2019) listed out_of_window.
- Unverified: Lightmatter Passage: Talk record only (abstract is one line); not a peer-reviewed paper; packaging stack detail unconfirmed.
- Unverified: CEA-Leti active photonic interposer (pre-2023): Earlier Leti photonic-interposer disclosure not located by Crossref search; ECTC 2023.00009 is earliest found.
- Unverified: Broadcom CPO (Bailly / 51.2T) packaging: Abstract gives no stack detail; no peer-reviewed Broadcom stack disclosure found.

### Thermal/power-delivery structures integral to the stack (`thermal_pdn`)

- Dominant organizations (cached): University of California (3), Intel (3), imec (3), A*STAR IME (3), Georgia Tech PRC (3), Center for Heterogeneous Integration and Performance Scaling (2)
- Thin years (<= 2 cached in-window): none
- Lineages cached as series: Intel-HB-pitch, TSMC-CoWoS-R-RDL_LS_and_size;TSMC-IMCSi-power, TSMC-IMCSi-power, imec-BSPDN-nTSV
- Gap: Microsoft direct-to-silicon microfluidic cooling: only peer-reviewed disclosure in pool is 10.1109/ectc51846.2026.00156 [cached] (Microsoft, ECTC 2026, GH200 backside-etched microchannels); scored 5 with buckets [] and stack_detailed=false, so absent from this input. Selector should consider adding it manually.
- Gap: TSMC SoIC-CS (Cool-Stacking, 10.1109/ectc51687.2025.00055 [cached]) is in the hybrid_bonding input, not here (stack_detailed=false).
- Gap: Saras STile: disclosures only at IMAPS (not screened); see out_of_window.
- Gap: Package-integrated IVR inductors from Intel: CoaxMIL 2.0 (10.1109/ectc51529.2024.00167 [not cached]) is stack_detailed=false, not in input; no Intel IVR stack paper in input.
- Gap: Murata base iPaS (10.1109/ectc51529.2024.00350 [not cached], 10.1109/ectc51529.2024.00170 [cached]) sit in advanced_substrate with stack_detailed=false.
- Gap: No detailed TSMC eDTC-in-interposer or Intel PowerVia + 3D stacking paper in input; Intel PowerVia is transistor-level and out of bucket scope.
- Gap: IME 10.1109/ectc51687.2025.00082 [not cached] (first demo of the same two-stacked-chip TV, same 157/187 W/cm2) is not in input; 2026.00096 is the process-detail version.
- Unverified: Intel PowerVia backside power delivery: Crossref keyword search did not return it; transistor-level BSPDN, out of bucket scope (no package/3D stack content) per bucket rule.
- Unverified: TSMC CoWoS-L / CoWoS-R embedded deep-trench capacitor (eDTC) or IPD in interposer: No paper in this input details eDTC/IPD stack; CoWoS-R+ IPD (10.1109/ectc51906.2022.00008) belongs to organic_rdl_fanout.

## Orientation reviews

1. `10.1109/tcpmt.2022.3144461` - Recent Advances and Trends in Advanced Packaging (TCPMT 2022). Broad technical survey of 2.5D/3D, fan-out, bridge and hybrid-bonding platforms (Unimicron, T-CPMT 2022); nominated by 2 curators.
2. `10.1109/iedm45625.2022.10019534` - Advanced Packaging Technology Platforms for Chiplets and Heterogeneous Integration (IEDM 2022). First-party OSAT platform overview (ASE, IEDM 2022) covering fan-out, bridge and 2.5D/3D stacks used across buckets.
3. `10.1109/iedm45741.2023.10413890` - Advanced Packaging Technologies in Memory Applications for Future Generative AI Era (IEDM 2023). First-party memory-maker overview of HBM stacking, MR-MUF and hybrid bonding (SK hynix, IEDM 2023); nominated by 2 curators.

Chosen by the orchestrator from 26 curator nominations: technical first-party or survey papers preferred over market overviews (e.g. the Yole ESTC 2024 fan-out overview, 2 nominations, was passed over).

## Out-of-window primary disclosures outside the spec list (reference only, not cached)

| Year | Venue | DOI | Technology | Bucket |
|---|---|---|---|---|
| 2016 | IMAPS International Symposium on Microelectronics 2016 | 10.4071/isom-2016-tp11 | Shinko i-THOP (context only; not a named mandatory item) | advanced_substrate |
| 2016 | IEEE JSTQE | 10.1109/jstqe.2016.2593637 | IBM self-aligned fiber-to-chip attach and hybrid integration on microelectronic packaging  | cpo |
| 2019 | OFC 2019 | 10.1364/ofc.2019.m4d.7 | Ayar Labs TeraPHY optical I/O chiplet co-packaged in MCM (assembly, thermals, fiber attach | cpo |
| 2019 | IEEE CPMT Symposium Japan (ICSJ) 2019 | 10.1109/icsj47124.2019.8998659 | Corning glass waveguide substrate for CPO (evanescent coupling to SiPh); detailed follow-u | cpo |
| 2021 | ECTC 2021 (71st), pp. 263-268 | 10.1109/ectc32696.2021.00052 | TSMC COUPE (first public peer-reviewed proposal of COUPE EIC-PIC stack) | cpo |
| 2010 | ECTC | 10.1109/ectc.2010.5490913 | Georgia Tech PRC glass interposer (origin context for Absolics lineage; Absolics link is t | glass |
| 2012 | IEEE TCPMT | 10.1109/tcpmt.2012.2204392 | Georgia Tech PRC thin glass interposer (panel-scalable) | glass |
| 2019 | ECTC | 10.1109/ectc.2019.00277 | Georgia Tech PRC Glass Panel Embedding (GPE); precursor of in-window 10.1109/ectc51906.202 | glass |
| 2020 | 2020 IEEE Symposium on VLSI Technology | 10.1109/vlsitechnology18217.2020.9265044 | HBM hybrid bonding (earliest found overall; foundry, not memory maker) | hybrid_bonding |
| 2021 | ECTC 2021 (71st), pp. 415-419 | 10.1109/ectc32696.2021.00076 | Samsung W2W Cu-Cu hybrid bonding multi-stack + TSV (pre-window context for Samsung HCB; no | hybrid_bonding |
| 2012 | IEDM 2012 | 10.1109/iedm.2012.6479039 | InFO original (context) | organic_rdl_fanout |
| 2015 | IMAPS Device Packaging Conference 2015 (Additional Conferences) | 10.4071/2015dpc-ta22 | Amkor SWIFT primary disclosure (abstract introduces SWIFT as new fan-out structure) | organic_rdl_fanout |
| 2016 | ECTC 2016 | 10.1109/ectc.2016.65 | InFO_PoP platform (context) | organic_rdl_fanout |
| 2016 | ECTC 2016 | 10.1109/ectc.2016.267 | ASE FOCoS: earliest paper naming Fan-Out Chip on Substrate found | organic_rdl_fanout |
| 2016 | IWLPC 2016 (Wafer-Level Packaging Symposium) | 10.37665/waedkkf53481 | Amkor SWIFT (first full symposium paper found) | organic_rdl_fanout |
| 2016 | IMAPS Device Packaging Conference 2016 (Additional Conferences) | 10.4071/2016dpc-tp22 | Deca M-Series (context; earliest M-Series record found, not exhaustively searched) | organic_rdl_fanout |
| 2020 | ECTC 2020 | 10.1109/ectc32862.2020.00055 | ASE FOCoS chip-last: first dedicated chip-last FOCoS disclosure | organic_rdl_fanout |
| 2020 | 2020 IEEE 70th ECTC, pp. 14-18 | 10.1109/ectc32862.2020.00015 | SPIL FO-EB | si_bridge |
| 2021 | 2021 IEEE 71st ECTC, pp. 8-13 | 10.1109/ectc32696.2021.00013 | ASE FOCoS-Bridge (introduced as stacked Si bridge FOCoS, sFOCoS; abstract: 'sFOCoS chip la | si_bridge |
| 2021 | 2021 IEEE 71st ECTC, pp. 96-100 | 10.1109/ectc32696.2021.00027 | Amkor S-Connect | si_bridge |
| 2021 | 2021 IEEE 71st ECTC, pp. 136-147 | 10.1109/ectc32696.2021.00034 | IBM DBHi | si_bridge |
| 2022 | 2022 6th IEEE EDTM, pp. 145-147 (venue not in screened set) | 10.1109/edtm53872.2022.9798272 | IBM s-DBHi (surface bridge, no substrate trench) | si_bridge |
| 2022 | IEEE/MTT-S IMS 2022, pp. 583-586 (venue not in screened set) | 10.1109/ims37962.2022.9865270 | Georgia Tech fused-silica stitch-chip (abstract: reported 'for the first time') | si_bridge |
| 2021 | ECTC 2021 (71st), pp. 28-33 | 10.1109/ectc32696.2021.00016 | TSMC SoIS (System on Integrated Substrate; wafer-processed integrated substrate, 91x91 mm  | si_interposer |
| 2024 | ISSCC 2024, pp. 490-492 (venue not in screened set) | 10.1109/isscc49657.2024.10454441 | AMD MI300 3.5D package (Si interposer + SoIC hybrid bonding) | si_interposer |
| 2020 | ECTC 2020 | 10.1109/ectc32862.2020.00020 | imec extreme wafer thinning on carrier + nano-TSV (basis of imec nTSV BSPDN) | thermal_pdn |
| 2020 | ECTC 2020 | 10.1109/ectc32862.2020.00071 | Intel package magnetic inductor arrays for FIVR (package-integrated IVR inductors) | thermal_pdn |
| 2021 | ECTC 2021 | 10.1109/ectc32696.2021.00029 | TSMC CoWoS liquid cooling (lidded and direct liquid cooling, TDP up to 2 kW); precursor to | thermal_pdn |
| 2025 | IMAPSource Proceedings (IMAPS; not a screened venue) | 10.4071/001c.129008 | Saras STile embedded IPD (capacitor) for PDN; eVR STile follow-up 10.4071/001c.147194 (202 | thermal_pdn |

## Method and overrides

- Prefilter: bucket regexes (config/buckets.yaml) plus a packaging-core keep rule for full venues and a packaging-context requirement for filtered venues; dropped records are listed in candidates.csv with score_method=rule, an empty score and their rule-based provisional score in rule_score.
- Scoring: each kept abstract scored 0-10 by an LLM subagent against the spec rubric (config/scoring_prompt.md), 100 per batch, with component points recorded; rescore batches replaced scores where the abstract changed. Same-abstract duplicates scored 7 and 9 in one case, so scores carry roughly +/-2 noise.
- Re-bucketing: 74 LLM >= 7 papers with no bucket were re-checked by a subagent (1 gained a bucket, 73 out of scope: process steps, sensors, power modules, device-level, board-level); 8 papers whose only bucket was removed by a curator reject were re-checked for a different bucket.
- Curation: 4 disclosure-research subagents (TSMC, Intel, Samsung+SK hynix, glass+FOPLP) verified primary-disclosure DOIs via Crossref; 10 per-bucket curator subagents proposed mandatory items, lineages, ranked picks, bucket-scoped rejects, gaps and reviews.
- Selection (scripts/select_cache.py): protected = spec-list technologies (curator entries matching the spec's named list) + orchestrator overrides + series anchored by a protected paper (completed). Candidates: curator picks with LLM score >= 5, LLM score >= 7, and near-complete series (>= half cached, <= 2 missing; members added but not protected). Per-bucket floor 10 papers / 3 orgs / 3 years with threshold lowering to 5. Cap 300: drop from the most over-represented bucket first (by cached count), lowest effective score = LLM score + 1 (curator pick) + 1 (curator priority 1), ties by raw score, then bucket-exclusive papers first, then curator rank; never breaking another bucket's floor.
- Curator 'mandatory' items not on the spec's named list (e.g. FOCoS, Deca M-Series, SPIL FO-EB, S-Connect, iPaS) are rank-0 priority-1 picks that compete under the cap; curator out-of-window items not on the spec list are in the reference table, not cached.
- Pre-2022 spec-list primary disclosures are cached with outside_screen = 'pre-2022' and count toward the cap (Wentao, 2026-10-18). Their score is empty (not LLM-scored).
- Where 'first disclosure' is ambiguous both candidates are cached: CoWoS (Xilinx ECTC 2011, name absent; TSMC ECTC 2013, first CoWoS-titled) and InFO-oS (MediaTek ITherm 2020, first naming; TSMC ECTC 2021, first process paper).
- Orchestrator overrides (config/selection.yaml, protected): five CPO papers excluded by the cpo input filter (stack_detailed=false) but flagged by the cpo curator as strong first-party disclosures (TSMC iOIS, TSMC COUPE+SoIC, Celestial AI OMIB, IBM/GF fiber attach, Intel EMIB fiber CPO), and Microsoft direct-to-silicon microfluidic cooling (thermal_pdn curator gap note).
- Duplicate Xplore index entries (same normalized title within venue-year) are collapsed to the canonical (shorter) DOI, which inherits the best score and buckets.
- lineage_id lists every series (';'-separated) of which the paper is a member and that is complete in cache_list.csv; identical or subset series proposed by different curators are collapsed, overlapping different series stay separate.
- Organization per paper (first_author_org) = first author's affiliation matched against config/orgs.yaml, else the first organization-like segment of the first author's affiliation; spelling variants (case, punctuation, corporate suffixes) merged. Co-author affiliations are not used.
- Scores are triage estimates from abstracts only; access status from Xplore accessType and OpenAlex OA/repository locations.
- Orchestrator overrides cached: 6 of 6.

