# Photonics Lineage Visualization Design

Status: target design for schema v0.2.0  
Primary question: how did people, teams, and technical lineages circulate among photonics institutions without overstating direction or completeness?

## Product decision

Replace the global Sankey as the primary interface with an interactive organization exchange network plus a person-level timeline inspector. Retain a narrowly defined founder-ancestry Sankey as a secondary story view.

The visualization must distinguish what is known from what is merely connected:

| Relationship | Meaning | Rendering |
|---|---|---|
| Confirmed transition | Dates or an explicit move statement establish A before B | Solid arrow A → B |
| Semantic direction | Founder/biography semantics suggest A before B but chronology is incomplete | Dashed arrow A ⇢ B |
| Direction unknown | At least one person has supported affiliations with A and B, but order is unresolved | Neutral line A — B |
| Bidirectional exchange | Different people have confirmed moves in both directions | Paired arrows A ⇄ B |
| Conflicted chronology | Sources support incompatible order | Warning-colored line with conflict marker |
| Acquisition or spinout | Organization event, not automatically a person move | Separate event overlay |

Unknown direction is first-class data. It must not be dropped merely because it cannot enter a directed acyclic Sankey.

## Default organization network

- Start with a searchable one-hop ego network; allow expansion to two hops.
- Size nodes by unique represented people, with an optional normalization for research coverage.
- Size links by unique contributing people. Role-weighted influence is an optional, explicitly labeled index.
- Color nodes by institution type by default; allow technology, geography, or status coloring.
- Show separate node metrics for confirmed inbound, confirmed outbound, bidirectional, direction unknown, founders produced, and founders recruited.
- Do not position nodes on a fake left-to-right chronology. Use either a stable force layout or explicit time slices when a time filter is active.
- Make display clustering reversible. Parent/business-unit and university/lab aggregation must never rewrite canonical evidence.

### Intel acceptance test

Selecting Intel must show all supported connections involving Intel Corporation and the Intel Silicon Photonics group. Luxtera, UCSB/Bowers, Lucent Bell Labs, Broadcom, Rockley, Aurrion, Juniper, Ayar, Lumentum, Meta, Finisar/JDSU/Oclaro/NeoPhotonics, Xscape, ANELLO, and other represented neighbors must not disappear solely because dates are missing.

The Intel panel must partition neighbors into confirmed inbound, confirmed outbound, bidirectional, semantic ancestry, and direction unknown. It must also disclose how many Intel-associated people have no represented non-Intel affiliation and the upstream/downstream research coverage of the cohort.

## Person timeline inspector

Selecting a person or an organization-to-organization relationship opens a timeline with one lane per person and one segment per atomic affiliation edge.

- Exact intervals use solid segments.
- Year/month precision uses bounded segments with visible uncertainty.
- Ongoing roles use an open right endpoint and an `active_on` observation.
- Unknown starts or ends use open/faded tails rather than disappearing.
- Concurrent roles remain vertically aligned and are never forced into sequence.
- LinkedIn-only grade-C claims are visually distinct and disabled in the confirmed-only layer.
- Papers and patents appear as point observations unless separate evidence supports employment.
- Acquisitions, spinouts, and explicit team transfers appear as event markers.

Every segment links to the underlying edge, source, locator, access date, evidence grade, and any conflict.

## Founder ancestry view

The existing Sankey is retained only after these changes:

- Rename it from talent flow to founder ancestry.
- Use unique founders as the default width.
- Label links as direct predecessor, nonconsecutive ancestry, or chronology unknown.
- Do not connect every historical affiliation to every founder destination as if each were a direct move.
- Do not mix acquisition edges with person movement.
- State the represented-person denominator and coverage limitations in the chart header.

## Data required by the renderer

The renderer consumes canonical person–organization edges plus derived transitions. Each transition retains its person, an unordered organization pair, optional from/to endpoints when direction is supported, direction status, directness, basis, and contributing edge IDs. It also consumes coverage records so missing inbound or outbound research is visible rather than interpreted as zero flow.

At minimum, every organization detail panel reports:

- represented people;
- people with one versus multiple known institutions;
- confirmed inbound/outbound unique people;
- direction-unknown shared people;
- usable start/end date coverage;
- A/B/C evidence composition;
- upstream/downstream and inbound/outbound research coverage;
- last research date.

## Implementation sequence

1. Build a v0.2 transition derivation that emits confirmed, semantic, unknown, and conflicted relationships without suppressing undated pairs.
2. Add deterministic coverage metrics and the Intel acceptance-test fixture.
3. Build the organization ego-network interaction and evidence drawer.
4. Add person timeline lanes and uncertainty rendering.
5. Recast the current Sankey as the secondary founder-ancestry view.
6. Add technology, geography, time, evidence, and role filters after metadata coverage is measured.

## Claims the interface must not make

- Link absence means no movement.
- A thicker node produced or absorbed more total industry talent.
- Founder-weighted units are headcount.
- A/B evidence implies a complete career history.
- Acquisition implies employee retention or movement.
- A shared affiliation establishes direction.
- Layout position establishes chronology.
