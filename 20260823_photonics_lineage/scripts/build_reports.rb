#!/usr/bin/env ruby
# frozen_string_literal: true

require "date"
require "digest"
require "fileutils"
require "json"
require "optparse"
require "yaml"

ROOT = File.expand_path("..", __dir__)
COLLECTIONS = %w[organizations people sources events edges].freeze

options = {
  canonical: File.join(ROOT, "canonical"),
  graph: File.join(ROOT, "generated", "graph_ab.json"),
  merge_notes: File.join(ROOT, "canonical", "MERGE_NOTES.md"),
  output: File.join(ROOT, "generated")
}

OptionParser.new do |parser|
  parser.banner = "Usage: ruby scripts/build_reports.rb [options]"
  parser.on("--canonical DIR", "Canonical YAML directory") { |value| options[:canonical] = File.expand_path(value) }
  parser.on("--graph FILE", "A/B graph JSON path") { |value| options[:graph] = File.expand_path(value) }
  parser.on("--merge-notes FILE", "Canonical merge notes path") { |value| options[:merge_notes] = File.expand_path(value) }
  parser.on("--output DIR", "Generated output directory") { |value| options[:output] = File.expand_path(value) }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit 0
  end
end.parse!

def load_yaml(path)
  YAML.safe_load(File.read(path), permitted_classes: [Date], aliases: true)
rescue Errno::ENOENT, Psych::SyntaxError => e
  abort "Unable to load #{path}: #{e.message}"
end

def load_json(path)
  JSON.parse(File.read(path))
rescue Errno::ENOENT, JSON::ParserError => e
  abort "Unable to load #{path}: #{e.message}"
end

def sorted_counts(items)
  items.compact.map(&:to_s).tally.sort.to_h
end

def percent(numerator, denominator)
  return 0.0 if denominator.zero?

  ((numerator.to_f / denominator) * 100).round(1)
end

def date_bounds(date_value)
  return nil unless date_value.is_a?(Hash) && date_value["value"]

  value = date_value["value"].to_s
  case date_value["precision"]
  when "day"
    day = Date.iso8601(value)
    [day, day]
  when "month"
    year, month = value.split("-").map(&:to_i)
    [Date.new(year, month, 1), Date.new(year, month, -1)]
  when "year"
    year = value.to_i
    [Date.new(year, 1, 1), Date.new(year, 12, 31)]
  end
rescue Date::Error, ArgumentError
  nil
end

def nonempty_value?(value)
  case value
  when Hash then value.values.any? { |item| nonempty_value?(item) }
  when Array then value.any? { |item| nonempty_value?(item) }
  else !value.nil? && value != ""
  end
end

def markdown_table(headers, rows)
  output = []
  output << "| #{headers.join(' | ')} |"
  output << "|#{headers.map { '---' }.join('|')}|"
  rows.each { |row| output << "| #{row.join(' | ')} |" }
  output.join("\n")
end

paths = COLLECTIONS.to_h { |name| [name, File.join(options[:canonical], "#{name}.yaml")] }
documents = paths.to_h { |name, path| [name, load_yaml(path)] }
collections = COLLECTIONS.to_h { |name| [name, documents.fetch(name).fetch(name)] }
schema_versions = documents.values.map { |document| document["schema_version"] }.uniq.sort
abort "Canonical schema versions disagree: #{schema_versions.join(', ')}" unless schema_versions.size == 1

graph = load_json(options[:graph])
merge_notes = File.exist?(options[:merge_notes]) ? File.read(options[:merge_notes]) : ""

sources = collections.fetch("sources")
organizations = collections.fetch("organizations")
people = collections.fetch("people")
events = collections.fetch("events")
edges = collections.fetch("edges")

canonical_hashes = paths.to_h { |name, path| [name, Digest::SHA256.file(path).hexdigest] }
graph_hashes = graph.dig("metadata", "source_sha256") || {}
graph_hash_match = COLLECTIONS.to_h { |name| [name, canonical_hashes[name] == graph_hashes[name]] }

edge_grades = sorted_counts(edges.map { |edge| edge.dig("evidence", "grade") })
edge_statuses = sorted_counts(edges.map { |edge| edge["status"] })
edge_types = sorted_counts(edges.map { |edge| edge["edge_type"] })
source_grades = sorted_counts(sources.map { |source| source["evidence_grade"] })
source_types = sorted_counts(sources.map { |source| source["source_type"] })
organization_types = sorted_counts(organizations.map { |organization| organization["organization_type"] })
organization_statuses = sorted_counts(organizations.map { |organization| organization["status"] })
research_statuses = {
  "organizations" => sorted_counts(organizations.map { |organization| organization["research_status"] }),
  "people" => sorted_counts(people.map { |person| person["research_status"] })
}
duplicate_ids = COLLECTIONS.to_h do |name|
  duplicates = collections.fetch(name).map { |item| item["id"] }.tally.select { |_id, count| count > 1 }.keys.sort
  [name, duplicates]
end
duplicate_source_urls = sources.group_by { |source| source["canonical_url"] || source["url"] }
  .select { |url, grouped| url && grouped.size > 1 }
  .transform_values { |grouped| grouped.map { |source| source["id"] }.sort }
  .sort.to_h

person_ids = people.map { |person| person["id"] }.to_h { |id| [id, true] }
organization_ids = organizations.map { |organization| organization["id"] }.to_h { |id| [id, true] }
person_organization_edges = edges.select do |edge|
  person_ids[edge["source_id"]] && organization_ids[edge["target_id"]]
end

graph_person_edges = graph.fetch("person_organization_edges")
graph_people = graph.dig("nodes", "people") || []
graph_organizations = graph.dig("nodes", "organizations") || []
flows = graph.fetch("talent_flows")
sequence_exclusions = graph.fetch("sequence_exclusions")
ordered_contributions = flows.sum { |flow| (flow["people"] || []).size }
exclusion_reasons = sorted_counts(sequence_exclusions.map { |item| item["reason"] })

acquisition_events = events.select { |event| event["event_type"] == "acquisition" }.sort_by { |event| event["id"] }
acquisition_edges = edges.select { |edge| edge["edge_type"] == "acquisition" }.sort_by { |edge| edge["id"] }
acquisition_event_ids = acquisition_events.map { |event| event["id"] }.to_h { |id| [id, true] }
linked_event_ids = acquisition_edges.filter_map { |edge| edge.dig("attributes", "event_id") }

acquisition_metrics = {
  "events" => acquisition_events.size,
  "edges" => acquisition_edges.size,
  "events_with_announced_date" => acquisition_events.count { |event| nonempty_value?(event["announced_date"]) },
  "events_with_effective_date" => acquisition_events.count { |event| nonempty_value?(event["effective_date"]) },
  "events_with_disclosed_value" => acquisition_events.count { |event| nonempty_value?(event["value"]) },
  "events_with_two_or_more_participants" => acquisition_events.count { |event| (event["participants"] || []).size >= 2 },
  "events_with_ab_evidence" => acquisition_events.count { |event| %w[A B].include?(event.dig("evidence", "grade")) },
  "edges_linked_to_event" => acquisition_edges.count { |edge| acquisition_event_ids[edge.dig("attributes", "event_id")] },
  "event_ids_without_acquisition_edge" => acquisition_events.map { |event| event["id"] }.reject { |id| linked_event_ids.include?(id) }.sort,
  "edge_ids_without_valid_event" => acquisition_edges.reject { |edge| acquisition_event_ids[edge.dig("attributes", "event_id")] }.map { |edge| edge["id"] }.sort,
  "event_ids_missing_announced_date" => acquisition_events.reject { |event| nonempty_value?(event["announced_date"]) }.map { |event| event["id"] },
  "event_ids_missing_effective_date" => acquisition_events.reject { |event| nonempty_value?(event["effective_date"]) }.map { |event| event["id"] },
  "event_ids_missing_disclosed_value" => acquisition_events.reject { |event| nonempty_value?(event["value"]) }.map { |event| event["id"] }
}

chronology_warnings = acquisition_events.filter_map do |event|
  announced = date_bounds(event["announced_date"])
  effective = date_bounds(event["effective_date"])
  next unless announced && effective && announced.first > effective.last

  {
    "id" => "warning_#{event['id']}_announced_after_effective",
    "event_id" => event["id"],
    "message" => "announced_date is after effective_date; inspect event notes for a publication-proxy explanation"
  }
end

conflict_ids = merge_notes.scan(/\bconflict_[a-z0-9_]+\b/).uniq.sort
conflicts = conflict_ids.map do |id|
  {
    "id" => id,
    "status" => "open_or_preserved",
    "source" => "canonical/MERGE_NOTES.md"
  }
end
explicit_non_edge_sections = merge_notes.lines.grep(/^### .*?(?:non-edges|excluded claims)/i).map(&:strip).uniq

limitations = [
  "The default graph includes only validated grade-A/B person-to-organization edges; lower-grade and asserted claims remain canonical but are excluded.",
  "Institution flows require strict date ordering. Missing or overlapping date precision produces explicit sequence exclusions rather than inferred order.",
  "Canonical conflicts and negative claims are currently narrative in MERGE_NOTES.md; only explicitly named conflict IDs and section counts are machine-surfaced.",
  "Acquisition completeness reports disclosure, not an assumption that unknown effective dates or values are zero."
]

default_filter = graph.dig("metadata", "evidence_filter") || {}
filter_is_ab = default_filter["statuses"] == ["validated"] && default_filter["grades"] == %w[A B]
all_hashes_match = graph_hash_match.values.all?
acquisitions_linked = acquisition_metrics["event_ids_without_acquisition_edge"].empty? && acquisition_metrics["edge_ids_without_valid_event"].empty?
ids_unique = duplicate_ids.values.all?(&:empty?)
source_urls_unique = duplicate_source_urls.empty?

definition_of_done = [
  { "id" => "canonical_collections_loaded", "status" => "pass", "detail" => "Five canonical YAML collections parsed with schema #{schema_versions.first}." },
  { "id" => "canonical_ids_unique", "status" => ids_unique ? "pass" : "fail", "detail" => ids_unique ? "No duplicate canonical IDs found." : "Duplicate IDs found; inspect research_briefing_metrics.yaml." },
  { "id" => "canonical_source_urls_unique", "status" => source_urls_unique ? "pass" : "warn", "detail" => source_urls_unique ? "No duplicate canonical source URLs found." : "Duplicate source URLs found; inspect research_briefing_metrics.yaml." },
  { "id" => "graph_matches_canonical_inputs", "status" => all_hashes_match ? "pass" : "fail", "detail" => all_hashes_match ? "All graph input SHA-256 hashes match." : "Rebuild generated/graph_ab.json before release." },
  { "id" => "default_graph_is_validated_ab", "status" => filter_is_ab ? "pass" : "fail", "detail" => "Graph evidence filter is #{default_filter.inspect}." },
  { "id" => "acquisition_events_and_edges_linked", "status" => acquisitions_linked ? "pass" : "warn", "detail" => "#{acquisition_metrics['edges_linked_to_event']} of #{acquisition_edges.size} acquisition edges link to canonical acquisition events." },
  { "id" => "open_conflicts_disclosed", "status" => conflicts.empty? ? "pass" : "warn", "detail" => conflicts.empty? ? "No explicitly named open conflicts found." : "#{conflicts.size} explicitly named conflicts remain open or preserved." },
  { "id" => "graph_exclusions_disclosed", "status" => "pass", "detail" => "#{sequence_exclusions.size} sequence exclusions are reported by reason." },
  { "id" => "deterministic_generation", "status" => "pass", "detail" => "Reports contain no runtime timestamp and use sorted IDs/count keys plus canonical input hashes." }
]

metrics = {
  "schema_version" => schema_versions.first,
  "report_version" => "0.1.0",
  "inputs" => {
    "canonical_sha256" => canonical_hashes,
    "graph_source_sha256_match" => graph_hash_match,
    "graph_reproducible" => graph.dig("metadata", "reproducible") == true
  },
  "collections" => COLLECTIONS.to_h { |name| [name, collections.fetch(name).size] },
  "integrity" => {
    "duplicate_ids" => duplicate_ids,
    "duplicate_source_urls" => duplicate_source_urls
  },
  "evidence" => {
    "source_grades" => source_grades,
    "source_types" => source_types,
    "edge_grades" => edge_grades,
    "edge_statuses" => edge_statuses
  },
  "taxonomy" => {
    "edge_types" => edge_types,
    "organization_types" => organization_types,
    "organization_statuses" => organization_statuses,
    "research_statuses" => research_statuses
  },
  "default_ab_graph" => {
    "evidence_filter" => default_filter,
    "canonical_person_organization_edges" => person_organization_edges.size,
    "included_person_organization_edges" => graph_person_edges.size,
    "excluded_by_grade_or_status" => person_organization_edges.size - graph_person_edges.size,
    "people" => graph_people.size,
    "organizations" => graph_organizations.size,
    "canonical_people_covered_percent" => percent(graph_people.size, people.size),
    "canonical_organizations_covered_percent" => percent(graph_organizations.size, organizations.size),
    "ordered_person_contributions" => ordered_contributions,
    "institution_flows" => flows.size,
    "sequence_exclusions" => sequence_exclusions.size,
    "sequence_exclusions_by_reason" => exclusion_reasons
  },
  "acquisitions" => acquisition_metrics,
  "release_disclosures" => {
    "named_conflicts" => conflicts,
    "chronology_warnings" => chronology_warnings,
    "explicit_non_edge_section_count" => explicit_non_edge_sections.size,
    "explicit_non_edge_sections" => explicit_non_edge_sections,
    "limitations" => limitations
  },
  "definition_of_done" => definition_of_done
}

collection_rows = COLLECTIONS.map { |name| [name.capitalize, collections.fetch(name).size] }
grade_rows = (source_grades.keys | edge_grades.keys).sort.map do |grade|
  [grade, source_grades.fetch(grade, 0), edge_grades.fetch(grade, 0)]
end
edge_type_rows = edge_types.map { |type, count| [type, count] }
acquisition_rows = [
  ["Canonical acquisition events", acquisition_metrics["events"]],
  ["Canonical acquisition edges", acquisition_metrics["edges"]],
  ["Edges linked to canonical events", acquisition_metrics["edges_linked_to_event"]],
  ["Events with announced date", acquisition_metrics["events_with_announced_date"]],
  ["Events with effective date", acquisition_metrics["events_with_effective_date"]],
  ["Events with disclosed value", acquisition_metrics["events_with_disclosed_value"]],
  ["Events with ≥2 participants", acquisition_metrics["events_with_two_or_more_participants"]],
  ["Events with A/B evidence", acquisition_metrics["events_with_ab_evidence"]]
]
graph_rows = [
  ["Canonical person→organization edges", person_organization_edges.size],
  ["Included validated A/B edges", graph_person_edges.size],
  ["Excluded by grade/status", person_organization_edges.size - graph_person_edges.size],
  ["People in graph", graph_people.size],
  ["Organizations in graph", graph_organizations.size],
  ["Ordered person contributions", ordered_contributions],
  ["Institution flows", flows.size],
  ["Sequence exclusions", sequence_exclusions.size]
]

markdown = []
markdown << "# Canonical Validation and Release Report"
markdown << ""
markdown << "Generated deterministically from canonical YAML and `generated/graph_ab.json`. This report does not run the standalone validator; it performs release-oriented consistency and freshness checks and should be paired with `ruby scripts/validate_data.rb --canonical canonical`."
markdown << ""
markdown << "## Snapshot"
markdown << ""
markdown << markdown_table(["Collection", "Count"], collection_rows)
markdown << ""
markdown << "Schema version: `#{schema_versions.first}`. Graph input hashes current: **#{all_hashes_match ? 'yes' : 'no'}**."
markdown << "Canonical duplicate IDs: **#{duplicate_ids.values.sum(&:size)}**. Duplicate source URLs: **#{duplicate_source_urls.size}**."
markdown << ""
markdown << "## Evidence"
markdown << ""
markdown << markdown_table(["Grade", "Sources", "Edges"], grade_rows)
markdown << ""
markdown << "Edge statuses: #{edge_statuses.map { |status, count| "`#{status}` #{count}" }.join(', ')}."
markdown << ""
markdown << "## Edge types"
markdown << ""
markdown << markdown_table(["Edge type", "Count"], edge_type_rows)
markdown << ""
markdown << "## Default validated A/B graph"
markdown << ""
markdown << markdown_table(["Metric", "Count"], graph_rows)
markdown << ""
markdown << "Coverage: #{percent(graph_people.size, people.size)}% of canonical people and #{percent(graph_organizations.size, organizations.size)}% of canonical organizations appear in the default graph."
markdown << ""
markdown << "Sequence exclusions: #{exclusion_reasons.map { |reason, count| "`#{reason}` #{count}" }.join(', ')}."
markdown << ""
markdown << "## Acquisition completeness"
markdown << ""
markdown << markdown_table(["Metric", "Count"], acquisition_rows)
markdown << ""
markdown << "Missing announced date: #{acquisition_metrics['event_ids_missing_announced_date'].empty? ? 'none' : acquisition_metrics['event_ids_missing_announced_date'].map { |id| "`#{id}`" }.join(', ')}."
markdown << ""
markdown << "Missing effective date: #{acquisition_metrics['event_ids_missing_effective_date'].empty? ? 'none' : acquisition_metrics['event_ids_missing_effective_date'].map { |id| "`#{id}`" }.join(', ')}."
markdown << ""
markdown << "Missing disclosed value: #{acquisition_metrics['event_ids_missing_disclosed_value'].empty? ? 'none' : acquisition_metrics['event_ids_missing_disclosed_value'].map { |id| "`#{id}`" }.join(', ')}."
markdown << ""
markdown << "## Conflicts, warnings, and limitations"
markdown << ""
if conflicts.empty?
  markdown << "No explicitly named conflict IDs were found in `canonical/MERGE_NOTES.md`."
else
  markdown << "Named open or preserved conflicts: #{conflicts.map { |conflict| "`#{conflict['id']}`" }.join(', ')}."
end
markdown << ""
if chronology_warnings.empty?
  markdown << "No acquisition chronology warnings were derived."
else
  chronology_warnings.each { |warning| markdown << "- `#{warning['event_id']}`: #{warning['message']}." }
end
markdown << ""
limitations.each { |limitation| markdown << "- #{limitation}" }
markdown << ""
markdown << "MERGE_NOTES contains #{explicit_non_edge_sections.size} explicitly labeled non-edge/excluded-claim sections. Negative claims remain narrative until a canonical structured conflicts/non-edges collection is introduced."
markdown << ""
markdown << "## Definition of done"
markdown << ""
definition_of_done.each do |item|
  marker = item["status"] == "pass" ? "x" : " "
  markdown << "- [#{marker}] **#{item['id']}** (`#{item['status']}`): #{item['detail']}"
end
markdown << ""
markdown << "## Input fingerprints"
markdown << ""
markdown << markdown_table(["Input", "SHA-256", "Matches graph"], COLLECTIONS.map { |name| [name, "`#{canonical_hashes[name]}`", graph_hash_match[name] ? "yes" : "no"] })
markdown << ""

FileUtils.mkdir_p(options[:output])
markdown_path = File.join(options[:output], "validation_report.md")
metrics_path = File.join(options[:output], "research_briefing_metrics.yaml")
File.write(markdown_path, markdown.join("\n"))
File.write(metrics_path, YAML.dump(metrics))

puts "Release reports built"
puts "  canonical collections: #{COLLECTIONS.map { |name| "#{name}=#{collections.fetch(name).size}" }.join(', ')}"
puts "  graph: edges=#{graph_person_edges.size}, flows=#{flows.size}, exclusions=#{sequence_exclusions.size}"
puts "  acquisitions: events=#{acquisition_events.size}, linked_edges=#{acquisition_metrics['edges_linked_to_event']}"
puts "  disclosures: conflicts=#{conflicts.size}, warnings=#{chronology_warnings.size}"
puts "  wrote: #{markdown_path}, #{metrics_path}"
exit(all_hashes_match && filter_is_ab ? 0 : 1)
