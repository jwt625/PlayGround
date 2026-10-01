# candidates_landmark report

Generated 2026-10-01. Output: `data/_staging/candidates_landmark.csv` (76 rows, one per paper; arXiv + journal versions merged into one row).

## Counts

By `platform_guess`:

| platform_guess | rows |
|---|---|
| lithium_niobate (includes hybrid Si/SiN-LN rows) | 33 |
| eo_polymer (SOH + plasmonic-organic) | 10 |
| barium_titanate | 9 |
| lithium_tantalate | 7 |
| inp_mqw | 5 |
| silicon_plasma_dispersion | 5 |
| algaas_gaas | 3 |
| germanium_silicon_eam | 2 |
| pzt | 1 |
| plzt | 1 |

By priority: 1 = 33, 2 = 40, 3 = 3. By `sim_candidate`: yes 36, no 24, unknown 16.
By `source_type`: journal 68, arxiv_preprint 4, conference 3, review 1. `access_guess`: open_access 44, arxiv 12, paywalled 20.

Of the 33 lithium_niobate rows, 9 are hybrid Si/SiN-LN (Weigel 2018, He 2019, Valdez 2022/2023, Boynton 2020, Ahmed 2020, Churaev 2023, Mao 2022, Zhang 2021 SiN-loaded LNOI). One TFLT row is hybrid TFLT-Si (Niels 2026).
Years: 2016 to 2026 (outside the 2018-2026 window only: ogiso2016, ogiso2017, shin2005, shin2008, included per task brief for the NTT n-i-p-n lineage and Dagli GaAs T-rail).

## Method

1. Leads from about 20 WebSearch queries (per platform: TFLN records, TFLT, BTO, SOH/plasmonic, InP, GaAs, Si/GeSi, mid-IR, cryo, foundry, 2025-2026 records) plus the task brief. Search snippets and pages are leads only.
2. Identity: every row resolved on Crossref by `query.bibliographic` (title + surname) then re-fetched by DOI (`/works/{doi}`); title similarity and first author checked. arXiv ids (taken from web results) were verified on DataCite (`10.48550/arxiv.<id>`): title similarity, first author, v1 submission date. Weigel 2018 arXiv title differs from the journal title (similarity 0.48) but first author and topic match; flagged in notes.
3. Metadata fields (title, authors, venue, DOI, dates, license) come only from Crossref/DataCite responses. `published_on` = earliest of arXiv v1 date and Crossref published/online/issued date (partial dates kept as `YYYY` or `YYYY-MM`: yang2022, powell2024, eltes2023, haffner2018, ogiso2016, shin2005). `year` = Crossref issued year of the journal/proceedings record, which is the basis of `paper_id`.
4. Priority/notes from abstracts: Crossref abstract, else DataCite (arXiv) abstract, else OpenAlex abstract (by DOI). Priority 1 only where the retrieved abstract reports a measured Vpi (or VpiL) and bandwidth for a TWE MZM-type device. Notes carry only numbers present in those abstracts; anything seen only in a web snippet or not retrieved is listed after "Unverified:" in the row.
5. `license`: CC license from Crossref, else Optica OA / IEEE OAPA from Crossref, else the arXiv license from DataCite; blank when only a publisher TDM/copyright license was reported. `access_guess` is a guess from those licenses and arXiv presence.

## Corrections found during verification

- Alexander 2018 (Nat. Commun., "Nanophotonic Pockels modulators on a silicon nitride platform") is PZT on SiN, not BTO: platform_guess = pzt.
- Mao 2024 (Commun. Mater., "Ultra-fast perovskite electro-optic modulator ...") is lanthanum-modified PZT per its abstract: platform_guess = plzt.
- soh_fj (El Shamy 2021, J. Opt., 100 GHz fJ SOH modulator) is simulation-only; dropped from the list.
- DOI-level fixes: Ortmann 2019 and Riedhauser 2023 first matched Crossref supplementary-material DOIs (`.s001`); main article DOIs used. Churaev 2023 first matched an IET proceedings record; Nat. Commun. DOI used.

## identity_unresolved / not found (not in CSV)

- Hybrid amorphous-Si / LN modulator (Jian et al., Mookherjea group, ca. 2019): Crossref title search returned no match.
- Lumiphase/ETH BTO MZM with VpiL 4.8 V mm (web snippet): which paper this is was not established; Eltes 2019 (JLT) and the OFC 2023 Eltes paper are in the CSV but the number is not tied to either.
- Ge/GeSi EAM Srinivasan-type (imec, JLT 2016/2018 era): title query matched a different paper (score 0.62); not included.
- TU/e InP membrane / generic-platform modulators and Infinera InP MZM (Kato JLT 2013 resolved but is pre-2018): no in-window paper verified.
- Ghent / Georgia Tech / UW (Dalton-Jen) EO-polymer modulator papers 2018+: only Koos-group SOH papers verified; Ding 2010 (UW) is pre-window and not included.
- Northwestern (Girouard) / Delaware BTO modulators: only pre-2018 or thesis-level hits; not included.
- McGill BTO and TFLN optical DAC OFC 2026 papers: in seed list A, not repeated.
- Chen/Zhang AlGaAs item from the brief: not identified; the AlGaAs/GaAs rows are Shin 2005, Shin 2008 (Crossref year; task calls it 2010) and Li 2025 (suspended GaAs push-pull).
- Considered and dropped: Samani 2019 (PAM-4 architecture, no Vpi/BW in abstract), Sakib 2021 (Intel ring, CLEO, no abstract retrieved), Meighan 2021 (InP CPS MZM, 27.3 GHz), Ding 2010.

## Flags for the merge step

- paper_id collisions with seed A/B ids (different papers, same FirstAuthorYear): wang2018 (Nature CMOS-voltage paper vs seed Wang2018 Optics Express), li2026a/li2026b, sayem2026 (seed has Sayem2026a/b), lin2025, liu2025 (seed Liu2025a). Marked in `notes`.
- chen2022 is the same paper as seed B Chen2022 (DOI 10.1063/5.0077232), included because the task brief asks for it.
- Overlap with pilot work: `ogiso2016` here is the Electronics Letters paper (DOI 10.1049/el.2016.2987, first author Ogiso per Crossref) and `ogiso2017` the JLT paper; a `pilot_ogiso2016` folder already exists under `data/_staging`.
- Rows whose abstract was not retrievable (Springer Nature items: wang2018, youssefi2021, abel2018, eltes2020, haffner2018, koch2020; fragments only: mercante2018, wolf2018b): priority and platform for these rest on titles/knowledge and are flagged "Unverified" in notes. Platform for haffner2018, burla2019, koch2020 (eo_polymer) and youssefi2021 (lithium_niobate) is a guess not stated in a retrieved abstract.
- Values seen only in web snippets: wang2018 (210 Gbit/s, <0.5 dB), nirmir (VpiL list), tflt2407 (VpiL 2.8 V cm, PAM8 176 GBd), psiquantum2025 (0.33 dB V). Re-read from the primary paper at distillation.

## Rate limits / blocks

- No HTTP 429 or 403 from Crossref, DataCite or OpenAlex; all API calls spaced >= 2.1 s, one at a time, Crossref with the required User-Agent. About 280 API requests in total.
- Per coordinator instruction, no requests to export.arxiv.org or arxiv.org were made; arXiv metadata came from DataCite only. arXiv licenses come from DataCite rightsList; where absent, noted.
- WebFetch of nature.com article pages returned a 303 cookie-redirect to idp.nature.com (not followed); abstracts for those items were not retrieved that way.
- WebSearch was used for discovery only (about 20 queries); no anti-bot pages encountered.

Scripts and intermediate JSON (Crossref/DataCite/OpenAlex responses) are in the session scratchpad `lm/` folder, not in the project.
