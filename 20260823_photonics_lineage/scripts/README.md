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
- schema-v0.2 source access metadata, per-person timeline-research metadata, edge timeline status, and source-backed timeline observations when present.

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

## A/B graph data

The current builder retains the conservative v0.1 projection semantics. Under schema v0.2 it remains useful for regression comparison, but it is not the target ecosystem visualization: it drops direction-unknown relationships and may represent nonconsecutive founder ancestry as institution flow. See `VISUALIZATION_DESIGN.md` for the replacement transition model and interface.

Build the reproducible validated-evidence graph from the canonical snapshot:

```sh
ruby scripts/validate_data.rb --canonical canonical
ruby scripts/build_graph.rb
```

The Ruby builder writes:

- `generated/graph_ab.json` — filtered person→organization evidence, conservative derived institution flows, exclusions, and input hashes;
- `generated/graph_build_report.md` — counts, ordering policy, flows, and exclusions.

Talent flow is derived only when an earlier edge's latest possible `end_date` is before a later edge's earliest possible `start_date`. It does not use biography list order or undated roles. Override paths with `--canonical DIR` and `--output DIR`.

## Python/Plotly Sankey

Build the actual interactive Sankey with pinned, isolated dependencies:

```sh
uv run \
  --with plotly==6.3.0 \
  --with pyyaml==6.0.2 \
  --with kaleido==1.1.0 \
  scripts/build_sankey.py
```

Alternatively, install `requirements-viz.txt` in a virtual environment and run
`python scripts/build_sankey.py`. The builder reads
`config/institution_display_map.yaml` and writes:

- `generated/photonics_lineage_sankey.html` — self-contained Plotly HTML with founder-lineage and strict-chronology Sankey modes, searchable person-level provenance, and an acquisition table;
- `generated/photonics_lineage_sankey.png` — static founder-lineage export when Kaleido has a usable Chrome installation;
- `generated/archive/photonics_lineage_sankey.pre_plotly.html` — the archived pre-Plotly visualization.

The display map collapses university/internal-lab clusters in both talent-flow
modes. Corporate families and industrial-lab histories collapse only in founder
mode when explicitly configured. Canonical organization IDs, time-aware Bell Labs
eras, acquisition counterparties, edge IDs, and source URLs remain available in
hover and evidence details. Non-collapsible audit groups remain separate. Use
`--no-png` when only the HTML is needed and `--no-archive` during deterministic
reruns.

## Release reports

After canonical validation and the graph build, generate deterministic release metrics:

```sh
ruby scripts/validate_data.rb --canonical canonical
ruby scripts/build_graph.rb --canonical canonical
ruby scripts/build_reports.rb
```

The report builder uses only the Ruby standard library and writes:

- `generated/validation_report.md` — release-facing collection, evidence, edge-type, graph-coverage, acquisition-completeness, disclosure, and definition-of-done tables;
- `generated/research_briefing_metrics.yaml` — the same core metrics in machine-readable form, including canonical input hashes and graph-freshness results.

The builder does not replace `validate_data.rb`. It checks that all canonical collections parse with one schema version, that the graph's recorded SHA-256 inputs match the current canonical files, that the default graph filter remains validated A/B-only, and that acquisition event/edge links and graph exclusions are disclosed. Explicit conflict IDs are extracted from `canonical/MERGE_NOTES.md`; other negative claims remain a documented limitation because there is not yet a canonical structured conflicts/non-edges collection.

Reports contain no generation timestamp and sort IDs and count keys, so identical inputs produce byte-identical outputs. Override inputs with `--canonical DIR`, `--graph FILE`, `--merge-notes FILE`, and `--output DIR`.
