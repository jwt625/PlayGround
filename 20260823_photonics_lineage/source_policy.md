# Source and Evidence Policy

Version: 0.2.0
Applies from: timeline-first revision

## Core rule

The graph is claim-based. A biography is not accepted wholesale: each employment, role, date, training relationship, acquisition, license, or team transfer is a separate claim with its own source references and grade. Seed-list presence means “research this,” not “this is true.”

For every person admitted to the dataset, collect as complete a publicly documented professional timeline as practical. Do not stop after finding the one affiliation that motivated inclusion. Search upstream and downstream roles, education and research training, concurrent appointments, business units, job titles, locations, start/end dates, founder events, and current status. Timeline completeness and evidence strength are separate dimensions: a useful C-grade LinkedIn interval should be captured as asserted evidence rather than omitted, while the default validated graph remains A/B-only.

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

## Timeline-first collection procedure

For each included person:

1. Establish identity using name aliases plus at least one discriminator such as employer sequence, education, title, location, publication, patent, or photograph.
2. Search the person's official biography, university page or CV, conference biographies, papers, patents, archived employer pages, LinkedIn, and relevant professional databases.
3. Capture every publicly documented professional and research interval that helps reconstruct sequence, including roles outside photonics. Mark each edge's `photonics_relevance` rather than deleting adjacent history.
4. Record the exact employer label and business unit stated by the source. Link parent and unit entities explicitly; do not silently normalize a specific group into the parent corporation.
5. Preserve month-level LinkedIn dates when shown. Preserve “present” as `timeline_status: ongoing` plus an `active_on` observation dated to profile access; do not invent a future end date.
6. Keep concurrent roles concurrent. Do not force non-overlapping intervals merely to produce a clean career path.
7. If a profile supplies ordered roles but no dates, preserve `profile_order` and `sequence_hint` as low-confidence metadata; never convert page order alone into a confirmed transition.
8. Create asserted grade-C edges immediately for useful LinkedIn-only claims. Add A/B corroboration later and promote the claim only when the stronger source directly supports it.
9. Record what was checked and what remains missing in `person.timeline_research` and/or a `coverage_record`. “Complete” means complete to the public sources checked, not objectively complete.
10. Search both directions around major hubs. For Intel, for example, research where each person came from as well as where they went; a diaspora-only pass is not coverage of Intel's talent exchange.

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

LinkedIn and commercial databases are C-grade even when first-person maintained, unless the same fact is independently supported by A/B evidence. They are a required timeline-discovery source when a public or legitimately accessible profile can be resolved, not merely a last-resort source.

For LinkedIn, capture at minimum:

- canonical profile URL and access date;
- the exact displayed employer, business unit, title, location, and start/end month or year;
- whether the role is displayed as current;
- the profile's explicit role order when dates are absent;
- a locator identifying the relevant Experience or Education entry;
- an archive URL or content hash when lawful and technically practical;
- ambiguity caused by merged employer pages, abbreviated dates, overlapping roles, or profile changes.

Do not copy personal contact information, home addresses, family information, or other non-professional sensitive data. Do not bypass access controls or misrepresent inaccessible content as reviewed. Store only the professional facts needed for lineage research and comply with applicable platform terms and access restrictions.

## Deduplication and identity

- IDs are immutable after canonical merge.
- Keep legal names, former names, native-script names, brands, and abbreviations as typed aliases.
- Model labs and business units separately from parents when they are meaningful lineage nodes.
- Use predecessor/successor links for rebrands and restructurings; use acquisition events for ownership changes.
- Do not collapse Bell Labs, Lucent, Alcatel-Lucent Bell Labs, and Nokia Bell Labs without time-aware evidence.
- Do not collapse Finisar, II-VI, and Coherent or Acacia and Cisco into one historical node.

## Publication and visualization gates

The default confirmed layer includes only validated A/B atomic edges. Grade C appears only in a visibly distinct optional research layer, with dashed or muted styling and direct source disclosure. D and X are excluded from factual flow views. Every aggregate relationship must retain its contributing person IDs and atomic edge IDs so it can be audited and regenerated.

The visualization must not suppress a real shared-person relationship solely because chronology is missing. Such a relationship is rendered as direction unknown. Confirmed transitions, semantic founder ancestry, and undirected co-affiliation are separate relationship classes and must never share the same arrow or legend entry.

Before release, run checks for orphan IDs, invalid edge endpoints, missing required attributes, duplicate canonical names, impossible date ordering, unsupported A-D claims, unresolved high-impact conflicts, and aggregate weights that cannot be reproduced.
