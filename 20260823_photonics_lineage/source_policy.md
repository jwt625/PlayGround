# Source and Evidence Policy

Version: 0.1.0  
Applies from: Phase 0

## Core rule

The graph is claim-based. A biography is not accepted wholesale: each employment, role, date, training relationship, acquisition, license, or team transfer is a separate claim with its own source references and grade. Seed-list presence means “research this,” not “this is true.”

## Evidence grades

| Grade | Meaning | Typical sources | Default graph treatment |
|---|---|---|---|
| A | Primary evidence | Official company/university bio, CV, regulatory or acquisition filing, official press release, conference bio, paper affiliation, thesis, patent, archived official page | Render |
| B | Strong secondary evidence | IEEE, Optica, reputable technical publication, authoritative interview, respected conference organizer | Render |
| C | Weak but useful evidence | LinkedIn, Crunchbase, The Org, company directory, maintained startup database | Optional layer; corroborate where possible |
| D | Unverified media or promotional claim | Startup PR, a single general-news story, promotional profile, reposted biography | Do not render by default |
| X | Inference | Timeline inference, coauthorship-only inference, “team from X” without named people | Never include in validated graph |

Grade describes the source class, not certainty. A primary source can be incomplete or self-serving. Conflicting A-grade claims must remain in `conflicts` until resolved.

## Source capture requirements

Create one stable `src_…` record per distinct artifact. Capture:

- canonical URL and archive URL when available;
- exact title and publisher/issuing organization;
- publication date at its actual precision;
- access date;
- language;
- source type and evidence grade;
- a page, paragraph, timestamp, patent number, or other locator for every supported claim;
- notes about ambiguity, page changes, machine translation, or inaccessible content.

For mutable pages—company leadership pages, biographies, LinkedIn, and databases—save an archive URL or content hash when legally and technically practical. Do not treat a search-result snippet as a source.

## Claim grading procedure

1. Split the statement into atomic claims: organization, title/group, relationship type, and dates.
2. Record every source that directly supports each claim. A source cited elsewhere in the same biography does not automatically support it.
3. Assign the best directly supporting grade, then note corroboration count. Do not average grades.
4. Use the least precise date supported by evidence. Never infer a start or end month from neighboring roles.
5. Set status to `validated` only for A/B evidence that directly supports the relationship and has no unresolved material conflict.
6. Keep C/D claims as `asserted`; keep reasoning-only claims as `inferred` with grade X.
7. Put contradictory dates, titles, identities, or acquisition values into `conflicts` rather than silently choosing one.

## Special cases

### Employment and technical relevance

Corporate employment alone does not prove membership in a photonics group. Record both the employer and `photonics_relevance`. Use `direct` only when role, group, publication, patent, or technical biography supports it. If a publication proves affiliation on one date but not employment duration, use `publication_affiliation`, not an employment interval.

### Education and lab lineage

A university degree does not prove membership in a particular PI’s lab. Use `research_training` to the university when only enrollment is known; target the lab only when a thesis, CV, lab roster, publication affiliation, or equivalent evidence supports lab membership. Advisors belong in edge attributes as person IDs.

### Patents and papers

A paper affiliation establishes affiliation at publication time, not a full employment history. A patent establishes inventor/assignee linkage but may not establish ordinary employment, technical-group membership, or current role. Preserve these as their explicit edge types unless another source supports employment.

### Acquisitions

Separate announcement date from effective/close date. Store headline, upfront, earnout, and stock components independently; never equate enterprise value, purchase price, or maximum consideration. Claims that a team was retained require named people or an explicit team-retention source.

### Team and institutional lineage

Generic phrases such as “the core team came from Intel, Huawei, or Broadcom” are discovery leads only. A `team_transfer` requires named people and evidence they moved as a meaningful group. A `spinout` requires explicit institutional, corporate, or technology-transfer evidence; alumni founding a company does not by itself make it a formal spinout.

### Chinese-language sources and names

Preserve original Chinese names and company names as aliases with `language: zh`. Do not merge people solely from matching romanization. Validate identity using at least one additional discriminator such as employer sequence, degree, title, patent authorship, location, or photograph. Record whether an English name is self-used, official, or a researcher-supplied transliteration.

### LinkedIn and databases

LinkedIn and commercial databases are C-grade even when first-person maintained, unless the same fact is independently supported by A/B evidence. They are useful for discovery and date-range hypotheses. Record access dates because profiles change.

## Deduplication and identity

- IDs are immutable after canonical merge.
- Keep legal names, former names, native-script names, brands, and abbreviations as typed aliases.
- Model labs and business units separately from parents when they are meaningful lineage nodes.
- Use predecessor/successor links for rebrands and restructurings; use acquisition events for ownership changes.
- Do not collapse Bell Labs, Lucent, Alcatel-Lucent Bell Labs, and Nokia Bell Labs without time-aware evidence.
- Do not collapse Finisar, II-VI, and Coherent or Acacia and Cisco into one historical node.

## Publication and visualization gates

The default Sankey includes only validated A/B atomic edges. Grade C may appear in a visually distinct optional layer. D and X are excluded. Every aggregate `talent_flow` edge must retain its contributing person IDs and atomic edge IDs so the rendered width can be audited and regenerated.

Before release, run checks for orphan IDs, invalid edge endpoints, missing required attributes, duplicate canonical names, impossible date ordering, unsupported A-D claims, unresolved high-impact conflicts, and aggregate weights that cannot be reproduced.
