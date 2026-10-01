# Candidate seed list report

Generated 2026-10-01. Rows: 134 in `candidates_seed.csv` (unique paper_id, no duplicate DOI or arXiv id). Priority 1: 40, 2: 80, 3: 14. Priority and sim_candidate are best-effort from titles, abstracts and local summary tables; full texts were not read.

## Method and limits

- Crossref `/works/<doi>` for DOI leads and OFC 2026 papers (DOI pattern `10.1364/OFC.2026.<code>`, all resolved); Crossref `query.bibliographic` title searches (exact or near-exact title required) or journal+volume+issue+page matching for URL-only leads.
- arXiv metadata (title, authors, submitted date, rights, abstract) came from DataCite (`10.48550/arxiv.<id>`), not the arXiv API: the arXiv API returned HTTP 503 then 429 during the first attempt. No arXiv call was made afterwards. arXiv license values are the DataCite rights field.
- arXiv-version existence for journal-only rows was checked only by DataCite exact-title search on 37 DOI-resolved non-OFC leads and 15 search-resolved leads. A missing arXiv id therefore does not prove no preprint exists. OFC 2026 rows were not checked for preprints.
- OFC 2026 `published_on` is the presentation date from the local conference schedule file; authors and DOIs come from Crossref. `year` is the year of the version of record (Crossref) or the arXiv year for arXiv-only rows; `published_on` is the earliest of the arXiv submission date and the Crossref full-date.
- paper_id: the owner's lead names are kept where given (for example singer2025, lin2025, wu2025); other ids are family name + year, with a/b/c suffixes on collisions (reserved owner ids such as li2026 and sayem2026a/b force suffixes on the others, for example sayem2026c is arXiv 2604.09825).
- Local paths: paths into the private startup corpus are written as `@corpus_A/...` (traveling-wave MZM folder) and `@corpus_B/...` (BTO folder) in the CSV so the private repo name does not appear; absolute paths are in `candidates_seed_local_paths_private.csv` (not for the public repo). Corpus 5 (`agentic-multiphysics-harness`) and corpus 3 (`PlayGround/20260320_OFC`) paths are written in full; strip these too if either repo is private.

## Counts per platform_guess

| platform_guess | rows |
|---|---:|
| lithium_niobate | 35 |
| inp_mqw | 24 |
| silicon_plasma_dispersion | 21 |
| lithium_tantalate | 14 |
| barium_titanate | 13 |
| other | 10 |
| eo_polymer | 5 |
| ferroelectric_nematic_lc | 4 |
| germanium_silicon_eam | 2 |
| algaas_gaas | 2 |
| strontium_titanate | 2 |
| pzt | 1 |
| plzt | 1 |

## Counts per discovered_via tag (a row can carry several tags)

| tag | rows |
|---|---:|
| local_corpus | 102 |
| ofc2026 | 56 |
| drive_doc | 14 |
| blog:OFS-69 | 10 |
| tmp_eo_md | 9 |
| blog:OFS-118 | 4 |
| blog:OFS-109 | 3 |
| blog:OFS-37 | 2 |
| blog:OFS-102 | 1 |
| blog:OFS-104 | 1 |
| blog:OFS-107 | 1 |
| blog:OFS-35 | 1 |
| blog:OFS-47 | 1 |
| blog:OFS-51 | 1 |
| blog:OFS-61 | 1 |
| blog:OFS-77 | 1 |
| blog:OFS-79 | 1 |
| blog:OFS-83 | 1 |
| blog:OFS-84 | 1 |

## access_guess

| access_guess | rows |
|---|---:|
| unknown | 68 |
| arxiv | 28 |
| open_access | 22 |
| paywalled | 16 |

OFC 2026 proceedings rows carry `unknown` because Crossref has no license field for them.

## Unresolved identities

- `freitas2026` (OFS-84 lead, IEEE arnumber 11357954): in the CSV with `identity_unresolved`; title copied from the lead, no authors, DOI or date. Crossref title searches (two phrasings) returned no match.
- `soni2023` (VTEC InP MZM for 112 Gbps): in the CSV with `identity_unresolved`; title and authors read from the local PDF; no Crossref, arXiv or DataCite match; venue unknown.
- ECOC 2022 Mo4F.1 (NTT Class-80 coherent driver modulator, from the local InP notes): not in the CSV. The notes give no title; DOI guesses `10.1364/ECEOC.2022.Mo4F.1` and `10.1364/ECOC.2022.Mo4F.1` returned 404. Crossref shows a related 2022 PTL paper (doi 10.1109/lpt.2022.3212678, 'Coherent Driver Modulator With Flexible Printed Circuit RF Interface for 128-Gbaud Operations') but it was not matched to the lead.
- OECC/PSC 2025 WE3-4 (InP/Si p-i-n MZM by chip-on-wafer bonding, from the local InP notes): not in the CSV; the notes give a URL and metrics but no title; not searched.
- Meng2026 (Nature, doi 10.1038/s41586-026-11000-w): identity verified, but Crossref has no abstract, so whether it reports a modulator is unverified (kept as priority 3, platform other).
- Matched without a lead title (accepted on journal, volume, issue, page): yao2024, hiraki2021, tue2025. Matched on description only: ozaki2023 (IEEE arnumber in the notes not verified). weigel2018 accepted as near-exact (local arXiv-style title is a prefix of the published title, same authors).

## Discrepancies between leads and API / file truth (API value kept)

Lead titles that differ in substance from the API title for the cited DOI:
- girouard (doi 10.1109/jqe.2017.2718222): inventory title 'Integrated polarization diversity architecture for barium titanate-on-insulator electro-optic modulators'; API title '$\chi^{(2)}$ Modulator With 40-GHz Modulation Utilizing BaTiO3 Photonic Crystal Waveguides'.
- moor2021 (doi 10.1364/fio.2021.fth4d.2): inventory 'High-speed hybrid plasmonic BTO modulator'; API '>150 GHz Hybrid-Plasmonic BaTiO3-On-SOI Modulator for CMOS Foundry Integration'.
- kohli2023 (doi 10.1109/jlt.2023.3260064): inventory '200 Gbit/s Barium Titanate Modulator Using Weakly Guided Plasmonic Modes'; API 'Plasmonic Ferroelectric Modulator Monolithically Integrated on SiN for 216 GBd Data Transmission'.
- picavet (doi 10.1002/adfm.202403024): inventory 'Solution-Processed Textured Barium Titanate Thin Films for Integrated Electro-Optic Modulators'; API 'Integration Of Solution-Processed BaTiO3 Thin Films with High Pockels Coefficient on Photonic Platforms'.
- chen2022: lead calls it a T-rail electrode paper; API title is 'High performance thin-film lithium niobate modulator on a silicon substrate using periodic capacitively loaded traveling-wave electrode'.
- Onural/Chiang2025 lead: API first author is Li-Yuan Chiang, so the id is chiang2025.
- OFC titles: M2B.6 schedule title 'Technologies for 400G/Lane IM/DD Interconnects' vs Crossref 'Scaling IM/DD Interconnects to 400 Gb/s per Lane: Component and System-Level Tradeoffs'; Th1H.5 schedule title lacks the Crossref prefix 'Tutorial:'; W1A.4 Crossref title contains the typo 'Litihum'; M2A.3 and W1A.1 differ in minor wording. The local InP notes list M2A.6 under a different title (slow-light, 94.7 GHz); Crossref and the schedule abstract give 'A Compact Mach-Zehnder Modulator in 300 mm Silicon Photonic Platform towards 400Gbps/lane Transmission'.
Abbreviated or informal lead titles (API title kept; same paper): kakolaki2025, kakolaki2026, wu2025, rahman2025, liu2025a, ulrich, chromophore, onural, singer (mVpp), suceava, zhu2025a and declercq-style symbol differences.
Local corpus files that are not what the corpus README says:
- `2022__110ghz-110mw-hybrid-si-ln-mzm__nature-communications.pdf` is Mourgias-Alexandris et al., 'Noise-resilient and high-speed deep learning with coherent silicon photonics' (Nat Commun, doi 10.1038/s41467-022-33259-z); a text search finds no '110 GHz' or 'lithium'. The intended result is Valdez et al., '110 GHz, 110 mW hybrid silicon-lithium niobate Mach-Zehnder modulator' (Sci Rep 12, doi 10.1038/s41598-022-23403-6; arXiv 2210.14785), added as the `valdez2022` row with no local path (the Valdez arXiv 2211.05208 row is valdez2023).
- `2026__copper-damascene...nat-commun.pdf` is the accepted unedited Nat Commun version of Lin et al. (arXiv 2505.04755); merged with the Lin2025 lead (lin2025).
- `2026__high-bandwidth-traveling-wave-eom-1um-tflt__bell-labs-arxiv.pdf` is Al Sayem et al., arXiv 2604.09825; kept as its own row (sayem2026c), separate from Sayem2026a (arXiv 2604.27285).
- Local README says the hybrid Si-LN entries '>100 GHz' (2018) and 'O/C-band' (2022) are the Weigel et al. and Valdez et al. arXiv 2211.05208 manuscripts; both resolved (published: Optics Express 26, 23728 and doi 10.1364/oe.480519).
API year of record differs from the owner's id year (id kept): wu2025 (2026), lin2025 (2026), cai2025 (2026), kim2025 (2026).
Published versions found by exact-title Crossref match and merged with their arXiv rows: lee2026 (Nat Commun), cai2025 (Nat Commun), montifiore2026 (Optics Express), anderson2025 (Science), kim2025 (Optics Letters), suceava2025 (Adv Mater, via DataCite relation), taghavi2026 and geravand2025 (arXiv ids found by DataCite title search), kharel2021 (Optica 10.1364/optica.416155; Crossref also lists 10.1364/optica.440484 with the same title, not used), lt_si arXiv 2503.10557 (Nature Photonics), valdez 2211.05208 (Optics Express). Not merged because the title differs: rahman2025 (possible JLT 10.1109/jlt.2025.3646212), wakita2024 and ozaki2024 OFC papers (possible JLT 10.1109/jlt.2024.3472732 and 10.1109/jlt.2024.3453510).

## Excluded items and reasons

From the leads:
- ding2025 (arXiv 2507.14000, Photonic Fabric): system-level memory/switch architecture; abstract has no modulator metrics.
- valicourt (doi 10.1364/OE.555476, 1.6-Tbps linear-drive optical interface): packaged optical-engine system paper; abstract names no modulator type or metrics.
- zhou2025 (arXiv 2508.02444): microwave-to-optical AlN transducer link between dilution refrigerators; no modulator metrics (published as Nat Photon doi 10.1038/s41566-026-01866-7 with title '1-km photonic link').
- Mourgias-Alexandris Nat Commun 2022 PDF: wrong paper for its README entry (deep-learning system paper).
- Section A DSP/SerDes items (simplified coherent-lite transceivers, coherent-receiver digital filters, 224G SerDes): out of scope, not searched.
From the local corpora and OFC 2026 list (title and abstract scan of all 830 schedule entries; only modulator-device papers kept):
- M4D.5 (directly modulated membrane laser/PD link), W4E.5 (membrane surface-emitting laser), Th1G.4, W1B.4, W1B.5, W4E.3, Th4A.3, Th1A.3, Th1A.4 (VCSEL and DML sources): not EO modulators.
- M1B.2, M2A.4 (QD comb + microring transmitters), M4B.2 (3D-stacked DWDM I/O), M4B.3 (16-wavelength microring transceiver link), M4B.5 (16-channel SiPh optical engine), Th4A.7 (1.6T SiPh transceiver), W3E.4 (6.4 Tbps transmitter with inverse-designed multiplexer), Th2A.46 (comb IM/DD): system or comb transmission papers without modulator-device focus.
- Th3H.4 (photonic tensor core with membrane laser modulators), Th2A.6 (non-volatile charge-trap phase shifter), W2A.68 (radiation transients in ring modulators), M4D.3, Th2A.15, Th3F.1, Th3F.6, Th3F.7 (receivers and photodetectors), Tu3C.5 (piezo-tunable Ta2O5 MZI), M2E.5 (Kerr all-optical switch), Th1F.1 (supercontinuum), W4K.4 (withdrawn), W1D.2, M3F.*, W2A.x/Tu2x network, DSP, fiber and OCS items: not EO modulator devices or system-only.
- Non-peer-reviewed or non-paper sources in the InP/TW notes: NTT Technical Review articles (2005, 2006, 2018, 2019, 2022), Fraunhofer HHI group web page, NTT product pages, NeoPhotonics and Coherent press releases, Semantic Scholar record for the TFLT paper (same paper as wang2025).
- Corpus 4 (`jwt625.github.io/assets/doc/2025/`): only the Wang2025 TFLT PDF is a modulator paper; other PDFs (lasers, quantum, fibers, theses) are unrelated.
- Corpus 5 non-modulator references (Sellmeier and fused-silica index papers, Bao2026 TFLN phase-modulator multiphysics paper cached as HTML): material-property or off-scope entries; the Bao2026 multiphysics paper (PMCID PMC13302779, thin-film LN phase modulator for fiber-optic gyroscopes) was not looked up and could be added later.

## Kept but borderline

Material or review rows kept at priority 3: chelladurai2025, suceava2025, ulrich2025, anderson2025, chromophore paper (doi 10.1002/adma.202104174), smajic2024, meng2026, shin2011, M2B.6, M2A.7, Th1H.5, Tu3C.6, W1A.1, Tu3J.1. System-adjacent rows kept at priority 2 because the modulator is the featured component: wakita2024 (InP IQ modulator + driver front end), declercq2026, W4J.4, Th4A.6, Th4B.2, W1A.7, M1B.3, Th2A.12, W3F.4.
