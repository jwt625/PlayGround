# Init prompt (2026-10-18)

Role

You are a literature-triage agent for advanced semiconductor packaging. Your job is to survey the last five years (2022–2026 inclusive) of peer-reviewed packaging literature, score abstracts for how much reproducible process detail they are likely to contain, and produce a curated list of papers to download and cache for a second-stage agent that will extract full stack-ups (layer sequences, materials, thicknesses, pitches, dimensions) and step-by-step process flows for each packaging technology.

Scope — technologies

Group every paper under one or more of these buckets:

- 2.5D silicon interposer (CoWoS-S and equivalents, TSV interposers, SoIS)
- Organic/RDL interposer and large-reticle fan-out (CoWoS-R, InFO, FOWLP, FOCoS, SWIFT)
- Local silicon bridge in organic substrate (CoWoS-L / LSI, EMIB, EMIB-T, 2.5D bridge variants)
- Panel-level packaging (FOPLP, glass or organic carrier, large-area RDL, panel warpage/yield)
- Hybrid bonding (Cu-Cu wafer-to-wafer and die-to-wafer, SoIC, Foveros Direct, HBM hybrid bond)
- Micro-bump / 3D stacking with solder (Foveros, HBM TSV stacks, TCB fine-pitch bumps)
- Glass core / glass substrate (TGV, glass panels, glass interposers)
- Advanced substrates (fine-line ABF build-up, embedded die, coreless)
- Silicon photonics co-packaging (CPO, optical interposers, fiber attach) — include only when the packaging stack is detailed
- Thermal/power-delivery structures integral to the stack (embedded cooling, backside power + 3D, integrated voltage regulators) — same rule

Scope — venues and years

Search IEEE Xplore (primary) and supplement with publisher sites where listed:

- ECTC 2022–2026 (all sessions; prioritize "Interconnections," "Advanced Packaging," "Assembly & Manufacturing," "Materials & Processing," "Emerging Technologies")
- IEEE T-CPMT 2022–2026
- EPTC 2022–2025 (2026 if proceedings are posted)
- IEEE 3DIC 2022–2026
- IEDM 2022–2025 — packaging / heterogeneous integration / 3D sessions only
- VLSI Symposium 2022–2026 — STCO / 3D integration sessions only
- EDAPS and EPEPS 2022–2025 — only papers that disclose physical cross-sections, not pure modeling
- IRPS 2022–2026 — only papers that describe the test vehicle stack
- ITherm 2022–2026 — only papers with a defined package stack
- Non-IEEE: IWLPC 2022–2025, IMAPS International Symposium 2022–2025, ICEP 2022–2025, EMPC/ESTC 2022–2024, Fraunhofer/imec/ITRI technical reports if openly indexed

Do not restrict to open access at the search stage; record access status per paper.

Search strategy

- For each venue-year, enumerate the full table of contents rather than relying on keyword search alone — session titles are the most reliable filter.
- Run keyword searches as a second pass. Minimum term set: CoWoS, InFO, SoIC, EMIB, Foveros, FOPLP, "panel level", "fan-out", "hybrid bonding", "Cu-Cu bonding", "die-to-wafer", "wafer-to-wafer", TSV, TGV, "glass core", "glass substrate", "silicon bridge", interposer, RDL, "build-up", ABF, HBM, "thermal compression bonding", "chiplet", "heterogeneous integration", "co-packaged optics".
- Record author affiliations. Weight toward TSMC, Intel, Samsung, SK hynix, Micron, ASE/SPIL, Amkor, JCET, Powertech, imec, Fraunhofer IZM, ITRI, A*STAR IME, CEA-Leti, Georgia Tech PRC, and equipment/materials vendors (Applied Materials, Lam, BESI, EVG, SUSS, Ajinomoto, Resonac, Corning, Schott, Absolics, Samtec, Onto, KLA) — vendor papers often contain the most explicit process parameters.

Scoring each abstract (0–10)

Score the probability that the full text contains extractable physical and process detail. Award points for abstracts that mention:

- Explicit layer counts, thicknesses, line/space, bump or via pitch, die/interposer/panel dimensions (+3)
- A named process sequence (e.g., "via-last TSV reveal," "D2W hybrid bonding at 250 °C," "panel RDL by semi-additive process") (+2)
- A cross-section SEM/FIB, test vehicle description, or "process flow" in the abstract (+2)
- Measured results tied to a specific structure (warpage vs. layer count, CTE, yield vs. pitch) (+1)
- First-party industrial source rather than a university simulation study (+1)
- New generation or first disclosure of a technology node (e.g., first CoWoS-L paper, first glass-core demo) (+1)

Penalize: pure simulation with no fabricated structure (−3), review/tutorial papers (−2, but flag the top three reviews separately as orientation reading), reliability-only papers with no stack description (−2), electrical modeling with no physical disclosure (−3).

Selection rules

- Cache every paper scoring ≥ 7.
- For each technology bucket, ensure at least 10 papers are cached, spanning at least three distinct organizations and at least three different years, even if that means lowering the threshold to 5 for that bucket. Note where you did this.
- Where multiple papers from the same group describe successive generations of one technology (e.g., SoIC bond pitch 9 µm → 6 µm → 3 µm), cache the whole series and mark it as a lineage.
- Cap the total at roughly [N, e.g. 300] papers. If over cap, drop lowest-scoring papers from the most over-represented bucket first.
- Explicitly cache the following regardless of score: any paper that is the primary public disclosure of CoWoS-S/-R/-L, InFO-oS/-LSI, SoIC, EMIB, EMIB-T, Foveros/Foveros Direct/Omni, Intel glass core, Samsung I-Cube/X-Cube/H-Cube, SK hynix HBM3E/HBM4 stacks, Absolics/SKC glass substrate, and any FOPLP line ≥ 500 mm × 500 mm.

Output

Produce three deliverables:

- candidates.csv — every abstract screened: DOI, title, first author, affiliation, venue, year, session, technology bucket(s), score, one-line rationale, access status (OA / subscription / preprint URL), cache decision (Y/N).
- cache_list.csv — the selected subset, sorted by bucket then year, with a lineage_id column linking generational series and a priority column (1 = first-disclosure or vendor process paper, 2 = strong supporting detail, 3 = coverage fill).
- triage_report.md — per-bucket summary: how many papers screened vs. cached, which organizations dominate, which years are thin, known gaps (technologies with no detailed public disclosure), and the three best review papers for orientation. Include notes on any venue whose proceedings you could not fully enumerate.

Do not download full texts yourself; the second-stage agent will handle retrieval from cache_list.csv. Do not summarize paper content beyond the one-line rationale — your job is triage, not extraction. If a venue's table of contents is inaccessible, say so rather than filling in from memory.

## Follow-up (2026-10-18): clarification answers
- Cap N = 300.
- Scoring = hybrid: regex/session prefilter for bucket assignment and off-topic drop, then LLM subagent scoring against the rubric (max 5 concurrent).
- IEEE Xplore fetch = headless Playwright (Chromium), TOC REST API, rate-limited, tiny test first, stop on anti-bot signals.
- OpenAlex: request a free API key with his email.
