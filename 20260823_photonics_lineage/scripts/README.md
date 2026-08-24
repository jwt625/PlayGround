# Data integrity tooling

Run the Phase 0 integrity checks from the repository root:

```sh
ruby scripts/validate_data.rb
```

The validator uses only the Ruby standard library. Its seed checks cover YAML parsing, schema-version consistency, required fields, unique and correctly prefixed person/organization IDs, technology taxonomy membership, organization parent/predecessor/successor references, current-organization references, and unique edge-type IDs. It is read-only and exits nonzero on failure.

Validate the complete canonical snapshot with one option:

```sh
ruby scripts/validate_data.rb --canonical canonical
```

Canonical validation additionally checks:

- unique, correctly prefixed source, event, and edge IDs;
- schema-required source, event, and edge fields;
- entity and evidence source references;
- event participants and edge endpoints;
- edge-type endpoint rules and required attributes;
- referenced event IDs;
- evidence grades versus edge status, including the requirement that A-D evidence cite at least one source;
- ISO date values and date ordering when the available precision makes ordering conclusive.

An explicitly documented event publication proxy after closing is reported as a warning, not an error. This preserves the exceptional InfiniLink record while keeping the chronology visible.

Collections can also be selected individually. Repeat flags to validate multiple disjoint files as a single ID/reference namespace:

```sh
ruby scripts/validate_data.rb \
  --organization-file organizations_seed.yaml \
  --organization-file canonical_organizations.yaml \
  --people-file people_seed.yaml \
  --people-file canonical_people.yaml \
  --source-file canonical_sources.yaml \
  --event-file canonical_events.yaml \
  --edge-file canonical_edges.yaml
```

When canonical files replace rather than partition seed records, pass only the canonical organization/person files to avoid legitimate duplicate-ID failures. Use `--schema` and `--edge-types` to test a future contract version. Run `ruby scripts/validate_data.rb --help` for all options.

## A/B graph prototype

Build the reproducible validated-evidence graph from the canonical snapshot:

```sh
ruby scripts/validate_data.rb --canonical canonical
ruby scripts/build_graph.rb
```

The builder writes:

- `generated/graph_ab.json` — filtered person→organization evidence, conservative derived institution flows, exclusions, and input hashes;
- `generated/photonics_lineage_sankey.html` — self-contained institution-flow visualization and searchable evidence browser;
- `generated/graph_build_report.md` — counts, ordering policy, flows, and exclusions.

Talent flow is derived only when an earlier edge's latest possible `end_date` is before a later edge's earliest possible `start_date`. It does not use biography list order or undated roles. Override paths with `--canonical DIR` and `--output DIR`.
