# Per-bucket curation instructions (selection subagents)

Project: this project directory (paths below are relative to it)
Goal: help pick ~300 papers (all buckets combined) whose full texts a second-stage agent will mine for
stack-ups and process flows. You curate ONE bucket. Triage only: do not summarize paper content.

## Inputs
- `cache/curation/input_<bucket>.jsonl`: scored candidates in your bucket (score >= 4, or flagged), sorted by score.
  Fields: uid, doi, title, year, venue, session, first_author, first_affiliation, orgs, score, rationale,
  buckets, first_disclosure, lineage, is_review, access.
- `cache/curation/first_disclosure_flags.jsonl`: every LLM first-disclosure flag (any bucket; unverified).
- `cache/curation/reviews.jsonl`: papers flagged as reviews/tutorials.
- `config/scoring_prompt.md`: the scoring rubric (for context).
- `init-prompt.md`: the full original spec.

## Tasks
1. Mandatory first disclosures for your bucket (list given in your task message). For each technology,
   determine its primary public disclosure paper (the first peer-reviewed paper that publicly describes it).
   Use web search and Crossref/OpenAlex to identify and verify (verify the DOI resolves and title matches:
   `curl -s "https://api.crossref.org/works/<doi>?mailto=$CONTACT_EMAIL"`, max 1 request/s).
   - If the paper is in your input (uid present) -> add to `mandatory`.
   - If it is pre-2022 or from a venue not in the screened set -> add to `out_of_window` with verified DOI.
   - Also mark the earliest detailed in-window (2022-2026) paper on that technology as mandatory if present.
   - If no detailed public disclosure exists, say so in `gaps`. Never invent a DOI; if you cannot verify, put
     the candidate in `unverified` instead.
2. Verify the LLM first_disclosure flags for papers in your input; keep true ones in `mandatory` only if they
   are genuinely the primary disclosure of a named technology/generation, otherwise ignore.
3. Lineages: group generational series from the same organization (e.g. SoIC bond pitch 9 -> 6 -> 3 um,
   EMIB -> EMIB-T, CoWoS-R interposer size growth, HBM 8H -> 12H -> 16H). Only uids from your input.
   lineage_id format: `<Org>-<Tech>-<axis>` e.g. `TSMC-SoIC-pitch`.
   Lineage members are protected from the cap, so be strict: only a true generational series of ONE technology
   from ONE organization (>= 2 papers, explicit generation step such as pitch, size, layer count, stack height).
   At most 6 lineages per bucket; same-org topical clusters without a generation step are NOT lineages (mention
   them in notes instead).
4. picks: ranked list (rank 1 = most valuable) of up to 45 uids from your input to cache. Favor: explicit
   dimensions + process flow + cross-sections, first-party industrial/vendor sources, new generations, and
   coverage across organizations and years. Must contain >= 10 papers spanning >= 3 organizations and
   >= 3 years if the input allows (score >= 5). Priority: 1 = first-disclosure or vendor/foundry/OSAT process
   paper, 2 = strong supporting detail, 3 = coverage fill.
5. reject: uids in your input with score >= 7 that do not belong in your bucket (mis-bucketed, off-topic,
   duplicate index entry, abstract mismatch, pure simulation that slipped through).
6. Notes: dominant organizations, thin years, known gaps (technologies with no detailed public disclosure),
   and up to 3 best review/tutorial papers for orientation in your bucket (from reviews.jsonl or input; give uid
   or verified DOI).

## Output: write `cache/curation/<bucket>.json` (Write tool), valid JSON:
{
  "bucket": "<bucket>",
  "mandatory": [{"uid": "...", "tech": "...", "basis": "why this is the primary/earliest in-window disclosure"}],
  "out_of_window": [{"doi": "...", "title": "...", "year": 2019, "venue": "...", "first_author": "...",
                     "affiliation": "...", "tech": "...", "verified_via": "crossref|openalex|publisher page URL",
                     "access": "OA|subscription|N/A", "url": "..."}],
  "unverified": [{"tech": "...", "candidate": "...", "reason": "..."}],
  "lineages": [{"lineage_id": "...", "uids": ["..."], "note": "generation axis, e.g. 9->6->3 um pitch"}],
  "picks": [{"uid": "...", "rank": 1, "priority": 1, "reason": "<= 15 words"}],
  "reject": ["uid", "..."],
  "gaps": ["..."],
  "reviews": [{"uid_or_doi": "...", "title": "...", "why": "<= 15 words"}],
  "notes": "dominant orgs; thin years; anything the selector should know (<= 120 words)"
}
Validate with `python3 -I -c "import json; json.load(open('cache/curation/<bucket>.json')); print('ok')"`.

Rules: no emoji; never fabricate titles, DOIs, numbers, or affiliations; label anything unverified;
do not edit any other project file; rate-limit web/API requests.
