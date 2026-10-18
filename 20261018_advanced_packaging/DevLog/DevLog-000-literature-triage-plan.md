# DevLog-000: Advanced packaging literature triage (2022-2026) - plan

Started 2026-10-18. Spec: `init-prompt.md` (verbatim, with clarification answers appended).

## Goal
Screen 2022-2026 packaging literature, score abstracts 0-10 for extractable stack/process detail, output
`outputs/candidates.csv`, `outputs/cache_list.csv` (~300), `outputs/triage_report.md`. No full-text download, no extraction.

## Decisions (2026-10-18, from Wentao)
- Cap N = 300.
- Scoring: hybrid. Regex/session prefilter assigns buckets and drops off-topic; LLM subagents score survivors against the rubric in batches (max 5 concurrent).
- Xplore: headless Playwright (cached Chromium 1234), same-origin REST calls, 3.5-5.5 s spacing, stop on 403/429/captcha/empty body.
- OpenAlex key requested and activated (stored in `.env`, git-ignored). Used for affiliations (raw strings), OA status, preprint URLs.

## Initial probe findings (2026-10-18)
- Plain curl to Xplore REST returns 200 with empty body (bot-blocked); browser-context fetch works.
- Xplore `/rest/search` records carry: abstract, authors, doi, articleNumber, publicationNumber, isNumber, accessType, publicationTitle, year. No affiliations, no session.
- OpenAlex 2022-2026 counts: ECTC 369/376/390/389/425; T-CPMT 265/319/379/367/319; EPTC 203/212/245/238/-; ICEP 129/139/174 + ICEP-IAAC 2025 181; ESTC 2022 117, 2024 157; EMPC 2023 84; 3DIC only 2024 (35); IWLPC, IMAPS Symposium: none for 2022-2026. OpenAlex is not a complete TOC source.
- Session titles: not in Xplore/OpenAlex. ECTC final programs 2022-2025 as PDFs on ectc.net/ectc-archive (72-75 ECTCFinal-Web.pdf); 2026 program page is JS-rendered (to check).

## Pipeline
1-2. `scripts/xplore_fetch.py`: enumerate every record per venue-year (search by publication title + year, 100 rows/page; discovery folded in), resumable, JSONL -> `cache/xplore/<venue>_<year>.jsonl`; `scripts/xplore_toc_check.py` completeness check vs TOC totals.
3. `scripts/openalex_enrich.py`: batch DOI lookup (50/request) for affiliations, OA status, best OA / preprint URL.
4. `scripts/ectc_sessions.py`: parse ECTC program PDFs -> title->session map, fuzzy-join.
5. Non-IEEE venues (IWLPC, IMAPS Symposium) and org reports: enumerate what is publicly indexed; record gaps.
6. `scripts/prefilter.py`: bucket regex (config/buckets.yaml), venue filter rules (filtered venues need packaging-topic match), off-topic drop, rule-based provisional score as a cross-check.
7. LLM scoring: batches of ~80 abstracts per subagent, JSON out (score components, buckets, rationale, is_review, first_disclosure flag, lineage hint). Max 5 concurrent.
8a. (2026-10-18, Wentao) Selection eval delegated to subagents: one curation subagent per bucket (10, max 5 concurrent)
    reviews that bucket's scored candidates, verifies first-disclosure flags and lineage series, proposes picks + priority.
8. `scripts/select.py`: threshold >=7, per-bucket floor (10 papers, >=3 orgs, >=3 years, down to 5), mandatory first-disclosure list, lineages, cap 300 with drop rule, priority 1/2/3.
9. Report + fresh-context audit subagent.

- 2026-10-18 (Wentao): mandatory primary disclosures that predate 2022 (CoWoS, InFO, EMIB, SoIC, ...) are added to cache_list.csv flagged out_of_window=Y and count toward the ~300 cap; the earliest detailed in-window paper per technology is cached too.

## Defaults assumed (not asked)
- CPO and thermal/PDN buckets: floor counts only papers passing the "packaging stack detailed" rule; shortfall reported as gap.
- IWLPC / IMAPS: public abstracts only; incomplete enumeration stated in report.
- Filtered venues (IEDM, VLSI, IRPS, ITherm, EDAPS, EPEPS): full TOC enumerated, then topic filter (session data not in Xplore); noted in report.

## TODO
- [x] Venue discovery
- [x] Full enumeration (Xplore)
- [x] OpenAlex enrichment
- [x] ECTC sessions
- [x] Non-IEEE venues (Crossref: IMAPS Symposium, IWLPC; partial)
- [x] Prefilter
- [x] LLM scoring (40 batches + rescore r000-r004; final rescore pending backfill)
- [x] Selection (interim; final rerun after dropped-row backfill)
- [x] Report (interim)
- [x] Audit 1 (fresh context) -> corrections below
- [x] Final rerun + audit 2 (2026-10-18 13:00)

## Progress log
- 2026-10-18 10:15 Probes done, scaffold created, OpenAlex key active.
- 2026-10-18 10:20 Xplore enumeration running (headless Chromium OK, no login needed). Fallback if blocked: Selenium Firefox with his logged-in profile (pattern from his blog repo's thread scraper; profile path from an untracked config.json), or ask him to log in.
- 2026-10-18 10:25 ECTC sessions mapped from ectc.net final programs (2022-2026): 1840/1949 records (94%).
- 2026-10-18 10:30 Xplore search/TOC abstracts truncated at ~400 chars -> full abstracts from OpenAlex (pilot 100/100).
- 2026-10-18 10:35 IMAPS: meridian.allenpress.com behind Cloudflare (403) -> Crossref prefix 10.4071 "IMAPSource Proceedings", issue "Symposium"/"IMAPS Symposium": 2022 84, 2023 54, 2024 76, 2025 65 (likely partial).
- 2026-10-18 10:35 IWLPC: SMTA Crossref prefix 10.37665 "Wafer-Level Packaging Symposium" vols 19-22 = IWLPC 2022-2025 (inferred from edition numbering: IWLPC 2017 = 14th, proceedings.com); landing page bot-walled; 13-23 records/yr (partial).
- 2026-10-18 10:40 Prefilter recall check on ECTC drops found misses (TSVs plural, generic interposer/RDL, SiCN bonding, glass stacking); regexes widened and packaging-core keep rule added for full venues.
- 2026-10-18 10:36 Xplore enumeration done (129 requests, no blocks): 10342 records in pool, 4388 kept by prefilter. Not in Xplore: ITherm 2026, ESTC 2023 (not held), EMPC 2022/2024 (biennial odd years), 3DIC 2022/2025/2026, EPTC 2026, IWLPC (all years; Crossref only).
- 2026-10-18 10:37 Xplore document pages sit behind an AWS WAF JS challenge; `/rest/document/<id>/abstract` works headless -> backfill script for 1149 truncated/no-abstract rows (background chain with TOC completeness check).
- 2026-10-18 10:37 LLM scoring started: 40 batches x 100 (rows with full abstracts), 2 batches per subagent, 5 concurrent. First 400 scored: 63 with score >= 7 (~16%).
- 2026-10-18 11:00 OpenAlex abstract QA: 237/8613 (2.8%) OpenAlex abstracts are placeholders ("International audience", page numbers, "Item does not contain fulltext") or swapped; prefilter now rejects OpenAlex abstracts <200 chars or not matching the Xplore snippet prefix -> Xplore REST backfill -> rescore batches (out_rNNN override out_NNN).
- 2026-10-18 11:20 All 40 primary batches scored (3919 papers). Rescore batches r000-r003 (427 papers) scored; r004 (70) running.
- 2026-10-18 11:25 Curation split (A) disclosure research per org group, independent of scores: disclosures_tsmc / intel / samsung_skhynix done, glass_panel running; (B) per-bucket curation started for si_bridge, advanced_substrate, organic_rdl_fanout.
- Disclosure findings needing report notes: InFO-LSI has no peer-reviewed paper (Hot Chips 2021 only); Samsung H-Cube and SK hynix HBM3E 12-high: press only; CoWoS original = Xilinx ECTC 2011 (name absent) vs TSMC ECTC 2013 (both cached); InFO-oS = MediaTek ITherm 2020 (first naming) vs TSMC ECTC 2021 (first process paper) (both cached); Foveros Direct/Omni brand mapping is trade-press inference.
- Duplicate index entries (same paper, two DOIs) seen in EPTC 2024 -> select_cache dedups by normalized title within venue-year.
- 2026-10-18 11:30-11:50 Curation (B) done for all 10 buckets (cache/curation/<bucket>.json). Re-score batches r004-r007 done.
- 2026-10-18 11:40 Orchestrator overrides (now in config/selection.yaml `orchestrator_overrides`, protected): 5 CPO papers flagged by the cpo curator but excluded by its stack_detailed input filter (TSMC iOIS, TSMC COUPE+SoIC, Celestial AI OMIB, IBM/GF fiber attach, Intel EMIB fiber CPO) + Microsoft direct-to-silicon cooling (thermal_pdn gap note).
- 2026-10-18 11:50 Interim merge: 257/300 slots protected because curators marked 129 'mandatory' -> protection restricted to the spec's named list (config/selection.yaml spec_mandatory_re); other curator 'mandatory' items -> rank-0 priority-1 picks; non-spec out-of-window items -> report reference table (not cached).
- 2026-10-18 11:55 Cap drop order: effective score = LLM score + pick bonus (+1) + priority-1 bonus (+1), ties by raw score (initial +2 bonus and 'picks always outrank' were too strong vs the spec's score rule).

## Audit 1 corrections (fresh-context subagent, 2026-10-18 12:00)
1. T-CPMT year mislabel: TOC backfill pulled the whole early-access issue into TCPMT_2025 (132 tcpmt.2026.* papers labelled 2025) -> prefilter now uses Xplore publicationYear and drops out-of-scope years (4 strays).
2. lineage_id written for incomplete series (always-true condition) -> written only if the series is complete in cache_list or anchored by a protected paper; same-series ids from different curators merged (union-find on shared members).
3. 'mandatory first disclosure' label over-applied -> reasons now 'spec-list primary disclosure', 'spec-list technology', 'curator first-party disclosure', 'curator pick', 'LLM score >= 7', 'lineage member', 'orchestrator override'; out-of-window rows have empty score and outside_screen = pre-2022 / unscreened venue; panel-size pattern added so Resonac 515x510 counts as a >= 500 mm line.
4. Report numbers hard-coded/inconsistent -> all counts computed from cache/selection_stats.json; screened excludes front matter and duplicates; below-threshold cached counts per bucket reported; reviews chosen explicitly (config/report_notes.yaml) with reasons; spec-list coverage table added; curator gap notes whose cited DOIs are now cached are marked resolved.
5. Duplicate merge shared a list object (a duplicate's bucket reject stripped the canonical record) -> list copy; rejects of duplicate entries ignored; picks/mandatory on duplicates mapped to the canonical uid.
6. Org attribution used the first org found in any affiliation -> first author's affiliation first; IME regex split A*STAR IME vs CAS IME; Micron regex excludes Unimicron/HANA Micron; fallback org = first organization-like affiliation segment.
7. HTML entities in titles/affiliations unescaped; garbled micro sign fixed.
8. candidates.csv: prefilter-dropped rows now have empty score, score_method=rule, rule score in a separate rule_score column.
9. 74 LLM >= 7 rows without a bucket re-checked by a subagent (cache/scoring/out_rebucket.jsonl): 1 re-bucketed, 73 out of scope.
10. Contact email moved to .env (CONTACT_EMAIL); absolute path removed from config/curation_prompt.md.
11. make_rescore compared raw vs unescaped abstracts (false 'changed') -> compares cleaned text.
- 2026-10-18 12:15 Dropped-row abstract backfill still running (~440 rows left, VLSI/T-CPMT); r008 (21 rescued/changed papers) scoring. Final: prefilter -> r009 -> select_cache -> report -> audit 2.

## Audit 2 corrections (fresh-context subagent, 2026-10-18 12:45; 9 of 11 audit-1 fixes confirmed, 2 partial)
1. Deliverables were git-ignored by PlayGround root patterns (*.csv, *cache*) -> project .gitignore re-includes outputs/*.csv and scripts/select_cache.py (verified with git check-ignore).
2. Low-scoring curator picks (score 4) survived while score-10 papers were dropped -> curator picks need LLM score >= 5 (pick_min_score); cap stays per-bucket (spec: most over-represented bucket first). Tested a global lowest-score drop: it unbalanced buckets (organic_rdl_fanout 94, cpo 15) and was reverted. Result: 8 cached papers below 5 (all spec-list or overrides); 49 papers with LLM >= 9 remain uncached because of the cap (reported).
3. Org attribution still fell back to co-author affiliations -> first_org from the first author's affiliation only; spelling variants merged; one shared field (cache_list first_author_org) used by selection and report.
4. Lineage union-find chained distinct series (CoWoS-R RDL + IMC-Si cooling) -> only identical/subset series collapse; multiple lineage ids per paper; only spec-anchored series are protected; near-complete series (>= half cached, <= 2 missing) get members added (unprotected); lineage_id written only for series complete in cache_list.
5. Spec-coverage table matched titles/rationales -> built from selection_reason only; AMD MI300 excluded from spec matching (reference table instead).
6. Gap notes: each cited DOI now tagged [cached]/[not cached].
7. Summary arithmetic (front matter and duplicates overlap) and the 'no retrievable abstract' claim corrected; front-matter regex widened (T-CPMT mastheads, TechRxiv/IEEE ads, cover/abstract pages, Index).
8. Field hygiene: LaTeX in Xplore titles converted (e.g. '2 x 4 um'); provenance notes stripped from curator affiliations; ECTC session strings cleaned of 'Time:' text.
9. 8 papers whose only bucket was removed by curator rejects re-checked (cache/scoring/out_rebucket2.jsonl): 2 re-bucketed (Intel TIM1 lidded 2.5D -> thermal_pdn; 80 um pitch Cu-Cu TCB -> microbump_3d).
10. Primary bucket = bucket of the curator that mandated the paper, else the one that ranked it best.
11. DevLog/README wording (personal repo name removed; candidates.csv scope described accurately).

## Final state (2026-10-18 13:00)
- Backfill complete (1394 Xplore abstract requests, no blocks; 244 records have no abstract on Xplore). All kept papers with an abstract are LLM-scored (49 primary batches + rescore r000-r010).
- outputs/cache_list.csv: 300 papers; per bucket 26-44 (floor met, no threshold lowering); 19 outside-screen spec-list primary disclosures; 19 complete lineage series; 6 orchestrator overrides.
- outputs/triage_report.md regenerated from selection_stats.json; outputs/candidates.csv 10342+19 rows.
- Not done / open: no full-text download (out of scope); IWLPC and IMAPS enumeration partial (Crossref only); 49 LLM >= 9 papers uncached by cap (candidates.csv cache=N) - raise cap or rebalance if the second stage has budget.
