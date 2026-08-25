#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require "yaml"
require "date"

class DataValidator
  attr_reader :errors, :warnings, :counts

  def initialize(schema_path:, edge_types_path:, organization_paths:, people_paths:, source_paths:, event_paths:, edge_paths:)
    @schema_path = schema_path
    @edge_types_path = edge_types_path
    @organization_paths = organization_paths
    @people_paths = people_paths
    @source_paths = source_paths
    @event_paths = event_paths
    @edge_paths = edge_paths
    @errors = []
    @warnings = []
    @counts = Hash.new(0)
  end

  def run
    schema = load_yaml(@schema_path)
    edge_types = load_yaml(@edge_types_path)
    organization_documents = @organization_paths.filter_map { |path| load_collection(path, "organizations") }
    people_documents = @people_paths.filter_map { |path| load_collection(path, "people") }
    source_documents = @source_paths.filter_map { |path| load_collection(path, "sources") }
    event_documents = @event_paths.filter_map { |path| load_collection(path, "events") }
    edge_documents = @edge_paths.filter_map { |path| load_collection(path, "edges") }

    return false unless schema.is_a?(Hash)

    validate_schema_contract(schema)
    validate_schema_versions(
      schema,
      edge_types,
      organization_documents + people_documents + source_documents + event_documents + edge_documents
    )

    organizations = organization_documents.flat_map { |document| document.fetch("organizations") }
    people = people_documents.flat_map { |document| document.fetch("people") }
    sources = source_documents.flat_map { |document| document.fetch("sources") }
    events = event_documents.flat_map { |document| document.fetch("events") }
    edges = edge_documents.flat_map { |document| document.fetch("edges") }

    validate_entities(
      organizations,
      kind: "organization",
      prefix: schema.dig("conventions", "ids", "prefixes", "organization"),
      required: schema.dig("entities", "organization", "required"),
      id_pattern: schema.dig("conventions", "ids", "pattern")
    )
    validate_entities(
      people,
      kind: "person",
      prefix: schema.dig("conventions", "ids", "prefixes", "person"),
      required: schema.dig("entities", "person", "required"),
      id_pattern: schema.dig("conventions", "ids", "pattern")
    )
    validate_person_timeline_research(people, schema)
    validate_technology_membership(organizations + people, schema)
    validate_organization_references(organizations)
    validate_organization_dates(organizations)
    validate_edge_types(edge_types, schema)
    validate_sources(sources, schema) unless @source_paths.empty?
    validate_events(events, organizations, people, sources, schema) unless @event_paths.empty?
    validate_edges(edges, organizations, people, events, sources, edge_types, schema) unless @edge_paths.empty?
    validate_entity_source_references(organizations + people, sources) unless @source_paths.empty?
    validate_current_organization_references(people, organizations)

    @counts["organization_files"] = organization_documents.length
    @counts["people_files"] = people_documents.length
    @counts["organizations"] = organizations.length
    @counts["people"] = people.length
    @counts["source_files"] = source_documents.length
    @counts["sources"] = sources.length
    @counts["event_files"] = event_documents.length
    @counts["events"] = events.length
    @counts["edge_files"] = edge_documents.length
    @counts["edges"] = edges.length
    errors.empty?
  end

  private

  def load_yaml(path)
    # Canonical files use YAML anchors to deduplicate repeated evidence blocks.
    # Inputs are repository-owned data, so aliases are intentionally enabled.
    document = YAML.safe_load_file(path, aliases: true)
    unless document.is_a?(Hash)
      error(path, "top level must be a mapping")
      return nil
    end
    document
  rescue Errno::ENOENT
    error(path, "file not found")
    nil
  rescue Psych::Exception => e
    error(path, "invalid YAML: #{e.message.lines.first.strip}")
    nil
  end

  def load_collection(path, key)
    document = load_yaml(path)
    return nil unless document

    collection = document[key]
    unless collection.is_a?(Array)
      error(path, "#{key} must be an array")
      return nil
    end
    { "path" => path, "schema_version" => document["schema_version"], key => collection }
  end

  def validate_schema_contract(schema)
    required_paths = [
      %w[schema_version],
      %w[conventions ids pattern],
      %w[conventions ids prefixes organization],
      %w[conventions ids prefixes person],
      %w[entities organization required],
      %w[entities person required],
      %w[taxonomies technology]
    ]
    required_paths.each do |path|
      value = path.reduce(schema) { |memo, key| memo.is_a?(Hash) ? memo[key] : nil }
      error(@schema_path, "missing schema contract field #{path.join('.')}") if value.nil?
    end
  end

  def validate_schema_versions(schema, edge_types, collection_documents)
    expected = schema["schema_version"]
    return if expected.nil?

    documents = []
    documents << [@edge_types_path, edge_types && edge_types["schema_version"]]
    collection_documents.each { |doc| documents << [doc["path"], doc["schema_version"]] }
    documents.each do |path, version|
      error(path, "schema_version must equal #{expected.inspect}; got #{version.inspect}") unless version == expected
    end
  end

  def validate_entities(rows, kind:, prefix:, required:, id_pattern:)
    unless prefix.is_a?(String) && required.is_a?(Array) && id_pattern.is_a?(String)
      error(@schema_path, "cannot validate #{kind} records because its schema contract is incomplete")
      return
    end

    regexp = Regexp.new("\\A(?:#{id_pattern.sub(/^\^/, '').sub(/\$$/, '')})\\z")
    ids = Hash.new { |hash, key| hash[key] = [] }
    rows.each_with_index do |row, index|
      label = "#{kind}[#{index}]"
      unless row.is_a?(Hash)
        error(label, "record must be a mapping")
        next
      end

      missing = required.reject { |field| row.key?(field) }
      error(label, "missing required fields: #{missing.join(', ')}") unless missing.empty?

      id = row["id"]
      if !id.is_a?(String)
        error(label, "id must be a string")
        next
      end
      error(id, "must start with #{prefix.inspect}") unless id.start_with?(prefix)
      error(id, "does not match schema ID pattern #{id_pattern.inspect}") unless regexp.match?(id)
      ids[id] << index
    end

    ids.each do |id, indexes|
      error(id, "duplicate #{kind} ID at indexes #{indexes.join(', ')}") if indexes.length > 1
    end
  rescue RegexpError => e
    error(@schema_path, "invalid conventions.ids.pattern: #{e.message}")
  end

  def validate_technology_membership(rows, schema)
    taxonomy = schema.dig("taxonomies", "technology")
    unless taxonomy.is_a?(Array)
      error(@schema_path, "taxonomies.technology must be an array")
      return
    end

    allowed = taxonomy.to_h { |technology| [technology, true] }
    rows.each_with_index do |row, index|
      next unless row.is_a?(Hash)

      technologies = row["technologies"]
      unless technologies.is_a?(Array)
        error(row["id"] || "record[#{index}]", "technologies must be an array")
        next
      end
      technologies.each do |technology|
        error(row["id"] || "record[#{index}]", "unknown technology #{technology.inspect}") unless allowed[technology]
      end
    end
  end

  def validate_organization_references(organizations)
    ids = organizations.filter_map { |row| row.is_a?(Hash) ? row["id"] : nil }.to_h { |id| [id, true] }
    organizations.each_with_index do |row, index|
      next unless row.is_a?(Hash)

      label = row["id"] || "organization[#{index}]"
      parent = row["parent_organization_id"]
      validate_org_reference(label, "parent_organization_id", parent, ids) unless parent.nil?

      %w[predecessor_ids successor_ids].each do |field|
        references = row[field]
        next if references.nil?
        unless references.is_a?(Array)
          error(label, "#{field} must be an array")
          next
        end
        references.each { |reference| validate_org_reference(label, field, reference, ids) }
      end
    end
  end

  def validate_org_reference(label, field, reference, ids)
    unless reference.is_a?(String)
      error(label, "#{field} reference must be a string; got #{reference.inspect}")
      return
    end
    error(label, "#{field} references missing organization #{reference.inspect}") unless ids[reference]
    error(label, "#{field} cannot reference itself") if label == reference
  end

  def validate_edge_types(document, schema)
    unless document.is_a?(Hash)
      error(@edge_types_path, "cannot validate edge types")
      return
    end
    rows = document["edge_types"]
    unless rows.is_a?(Array)
      error(@edge_types_path, "edge_types must be an array")
      return
    end

    pattern_text = schema.dig("conventions", "ids", "pattern")
    pattern = pattern_text.is_a?(String) ? Regexp.new("\\A(?:#{pattern_text.sub(/^\^/, '').sub(/\$$/, '')})\\z") : nil
    ids = Hash.new { |hash, key| hash[key] = [] }
    rows.each_with_index do |row, index|
      unless row.is_a?(Hash)
        error("edge_type[#{index}]", "record must be a mapping")
        next
      end
      id = row["id"]
      if !id.is_a?(String)
        error("edge_type[#{index}]", "id must be a string")
        next
      end
      error(id, "does not match schema ID pattern #{pattern_text.inspect}") if pattern && !pattern.match?(id)
      ids[id] << index
    end
    ids.each do |id, indexes|
      error(id, "duplicate edge-type ID at indexes #{indexes.join(', ')}") if indexes.length > 1
    end
    @counts["edge_types"] = rows.length
  rescue RegexpError => e
    error(@schema_path, "invalid conventions.ids.pattern: #{e.message}")
  end

  def validate_sources(sources, schema)
    validate_prefixed_collection(
      sources,
      kind: "source",
      prefix: schema.dig("conventions", "ids", "prefixes", "source"),
      required: schema.dig("entities", "source", "required"),
      id_pattern: schema.dig("conventions", "ids", "pattern")
    )
    allowed_grades = schema.dig("entities", "source", "fields", "evidence_grade", "enum") || []
    allowed_types = schema.dig("entities", "source", "fields", "source_type", "enum") || []
    allowed_access_scopes = schema.dig("entities", "source", "fields", "access_scope", "enum") || []
    allowed_retrieval_methods = schema.dig("entities", "source", "fields", "retrieval_method", "enum") || []
    sources.each_with_index do |source, index|
      next unless source.is_a?(Hash)

      label = source["id"] || "source[#{index}]"
      error(label, "evidence_grade #{source['evidence_grade'].inspect} is not defined") unless allowed_grades.include?(source["evidence_grade"])
      error(label, "source_type #{source['source_type'].inspect} is not defined") unless allowed_types.include?(source["source_type"])
      if source["access_scope"] && !allowed_access_scopes.include?(source["access_scope"])
        error(label, "access_scope #{source['access_scope'].inspect} is not defined")
      end
      if source["retrieval_method"] && !allowed_retrieval_methods.include?(source["retrieval_method"])
        error(label, "retrieval_method #{source['retrieval_method'].inspect} is not defined")
      end
      publication = date_bounds(source["publication_date"], "#{label}.publication_date")
      validate_iso_day(source["accessed_date"], "#{label}.accessed_date") unless source["accessed_date"].nil?
      begin
        accessed = Date.iso8601(source["accessed_date"]) if source["accessed_date"].is_a?(String)
        error(label, "publication_date is definitively after accessed_date") if publication && accessed && publication.first > accessed
      rescue Date::Error
        # The format-specific error is already emitted by validate_iso_day.
      end
    end
  end

  def validate_events(events, organizations, people, sources, schema)
    validate_prefixed_collection(
      events,
      kind: "event",
      prefix: schema.dig("conventions", "ids", "prefixes", "event"),
      required: schema.dig("entities", "event", "required"),
      id_pattern: schema.dig("conventions", "ids", "pattern")
    )
    entity_ids = (organizations + people).filter_map { |row| row.is_a?(Hash) ? row["id"] : nil }.to_h { |id| [id, true] }
    source_ids = sources.filter_map { |row| row.is_a?(Hash) && row["id"] ? [row["id"], row] : nil }.to_h
    allowed_event_types = schema.dig("entities", "event", "fields", "event_type", "enum") || []

    events.each_with_index do |event, index|
      next unless event.is_a?(Hash)

      label = event["id"] || "event[#{index}]"
      error(label, "unknown event_type #{event['event_type'].inspect}") unless allowed_event_types.include?(event["event_type"])
      participants = event["participants"]
      if participants.is_a?(Array)
        error(label, "participants must not be empty") if participants.empty?
        participants.each_with_index do |participant, participant_index|
          participant_label = "#{label}.participants[#{participant_index}]"
          unless participant.is_a?(Hash)
            error(participant_label, "must be a mapping")
            next
          end
          %w[entity_id role].each { |field| error(participant_label, "missing #{field}") unless participant.key?(field) }
          entity_id = participant["entity_id"]
          error(participant_label, "references missing entity #{entity_id.inspect}") unless entity_ids[entity_id]
        end
      else
        error(label, "participants must be an array")
      end

      validate_evidence(event["evidence"], label, source_ids, schema)
      announced = date_bounds(event["announced_date"], "#{label}.announced_date")
      effective = date_bounds(event["effective_date"], "#{label}.effective_date")
      if announced && effective && effective.last < announced.first
        if event["notes"].to_s.match?(/post-dates close|publication proxy/i)
          warning(label, "announced_date is after effective_date but notes document a publication proxy")
        else
          error(label, "effective_date is definitively earlier than announced_date")
        end
      end
    end
  end

  def validate_edges(edges, organizations, people, events, sources, edge_types_document, schema)
    validate_prefixed_collection(
      edges,
      kind: "edge",
      prefix: schema.dig("conventions", "ids", "prefixes", "edge"),
      required: schema.dig("entities", "edge", "required"),
      id_pattern: schema.dig("conventions", "ids", "pattern")
    )
    org_ids = organizations.filter_map { |row| row.is_a?(Hash) ? row["id"] : nil }.to_h { |id| [id, true] }
    person_ids = people.filter_map { |row| row.is_a?(Hash) ? row["id"] : nil }.to_h { |id| [id, true] }
    event_ids = events.filter_map { |row| row.is_a?(Hash) ? row["id"] : nil }.to_h { |id| [id, true] }
    source_ids = sources.filter_map { |row| row.is_a?(Hash) && row["id"] ? [row["id"], row] : nil }.to_h
    edge_types = (edge_types_document.is_a?(Hash) ? edge_types_document["edge_types"] : []).to_a.to_h { |row| [row["id"], row] }

    edges.each_with_index do |edge, index|
      next unless edge.is_a?(Hash)

      label = edge["id"] || "edge[#{index}]"
      source_type = endpoint_type(edge["source_id"], org_ids, person_ids)
      target_type = endpoint_type(edge["target_id"], org_ids, person_ids)
      error(label, "source_id references missing entity #{edge['source_id'].inspect}") unless source_type
      error(label, "target_id references missing entity #{edge['target_id'].inspect}") unless target_type

      definition = edge_types[edge["edge_type"]]
      if definition.nil?
        error(label, "unknown edge_type #{edge['edge_type'].inspect}")
      else
        allowed_sources = definition["source_types"] || []
        allowed_targets = definition["target_types"] || []
        error(label, "#{edge['edge_type']} does not allow source endpoint type #{source_type.inspect}") if source_type && !allowed_sources.include?(source_type)
        error(label, "#{edge['edge_type']} does not allow target endpoint type #{target_type.inspect}") if target_type && !allowed_targets.include?(target_type)
        attributes = edge["attributes"]
        unless attributes.is_a?(Hash)
          error(label, "attributes must be a mapping")
          attributes = {}
        end
        (definition["required_attributes"] || []).each do |attribute|
          error(label, "missing required edge attribute #{attribute}") unless attributes.key?(attribute)
        end
        event_id = attributes["event_id"]
        error(label, "event_id references missing event #{event_id.inspect}") if event_id && !event_ids[event_id]
      end

      validate_evidence(edge["evidence"], label, source_ids, schema, status: edge["status"])
      start_bounds = date_bounds(edge["start_date"], "#{label}.start_date")
      end_bounds = date_bounds(edge["end_date"], "#{label}.end_date")
      error(label, "end_date is definitively earlier than start_date") if start_bounds && end_bounds && end_bounds.last < start_bounds.first
      validate_timeline_fields(edge, label, source_ids, schema)
    end
  end

  def validate_person_timeline_research(people, schema)
    allowed_statuses = schema.dig("entities", "person", "fields", "timeline_research", "fields", "status", "enum") || []
    allowed_source_types = schema.dig("entities", "source", "fields", "source_type", "enum") || []
    people.each_with_index do |person, index|
      next unless person.is_a?(Hash) && person.key?("timeline_research")

      label = person["id"] || "person[#{index}]"
      timeline = person["timeline_research"]
      next if timeline.nil?
      unless timeline.is_a?(Hash)
        error(label, "timeline_research must be a mapping or null")
        next
      end
      status = timeline["status"]
      error(label, "timeline_research.status #{status.inspect} is not defined") unless allowed_statuses.include?(status)
      %w[earliest_known_date latest_known_date].each do |field|
        date_bounds(timeline[field], "#{label}.timeline_research.#{field}") if timeline[field]
      end
      source_types = timeline["source_types_checked"]
      unless source_types.is_a?(Array)
        error(label, "timeline_research.source_types_checked must be an array")
        next
      end
      source_types.each do |source_type|
        error(label, "timeline_research contains unknown source type #{source_type.inspect}") unless allowed_source_types.include?(source_type)
      end
    end
  end

  def validate_timeline_fields(edge, label, source_ids, schema)
    allowed_statuses = schema.dig("entities", "edge", "fields", "timeline_status", "enum") || []
    status = edge["timeline_status"]
    error(label, "timeline_status #{status.inspect} is not defined") if status && !allowed_statuses.include?(status)
    warning(label, "timeline_status is ongoing but end_date is populated") if status == "ongoing" && edge["end_date"]

    return unless edge.key?("timeline_observations")
    observations = edge["timeline_observations"]
    unless observations.is_a?(Array)
      error(label, "timeline_observations must be an array")
      return
    end
    allowed_types = schema.dig("common_types", "timeline_observation", "fields", "observation_type", "enum") || []
    observations.each_with_index do |observation, index|
      observation_label = "#{label}.timeline_observations[#{index}]"
      unless observation.is_a?(Hash)
        error(observation_label, "must be a mapping")
        next
      end
      %w[observation_type date source_reference].each do |field|
        error(observation_label, "missing #{field}") unless observation.key?(field)
      end
      unless allowed_types.include?(observation["observation_type"])
        error(observation_label, "observation_type #{observation['observation_type'].inspect} is not defined")
      end
      date_bounds(observation["date"], "#{observation_label}.date") if observation["date"]
      validate_source_references([observation["source_reference"]], observation_label, source_ids) if observation["source_reference"]
    end
  end

  def validate_entity_source_references(entities, sources)
    source_ids = sources.filter_map { |row| row.is_a?(Hash) && row["id"] ? [row["id"], row] : nil }.to_h
    entities.each_with_index do |entity, index|
      next unless entity.is_a?(Hash)

      label = entity["id"] || "entity[#{index}]"
      refs = entity["sources"]
      unless refs.is_a?(Array)
        error(label, "sources must be an array")
        next
      end
      validate_source_references(refs, label, source_ids)
    end
  end

  def validate_current_organization_references(people, organizations)
    org_ids = organizations.filter_map { |row| row.is_a?(Hash) ? row["id"] : nil }.to_h { |id| [id, true] }
    people.each_with_index do |person, index|
      next unless person.is_a?(Hash)

      organization_id = person["current_organization_id"]
      next if organization_id.nil?
      error(person["id"] || "person[#{index}]", "current_organization_id references missing organization #{organization_id.inspect}") unless org_ids[organization_id]
    end
  end

  def validate_organization_dates(organizations)
    organizations.each_with_index do |organization, index|
      next unless organization.is_a?(Hash)

      label = organization["id"] || "organization[#{index}]"
      founded = date_bounds(organization["founded_date"], "#{label}.founded_date")
      dissolved = date_bounds(organization["dissolved_date"], "#{label}.dissolved_date")
      error(label, "dissolved_date is definitively earlier than founded_date") if founded && dissolved && dissolved.last < founded.first
    end
  end

  def validate_prefixed_collection(rows, kind:, prefix:, required:, id_pattern:)
    unless prefix.is_a?(String) && required.is_a?(Array) && id_pattern.is_a?(String)
      error(@schema_path, "cannot validate #{kind} records because its schema contract is incomplete")
      return
    end
    regexp = Regexp.new("\\A(?:#{id_pattern.sub(/^\^/, '').sub(/\$$/, '')})\\z")
    ids = Hash.new { |hash, key| hash[key] = [] }
    rows.each_with_index do |row, index|
      label = "#{kind}[#{index}]"
      unless row.is_a?(Hash)
        error(label, "record must be a mapping")
        next
      end
      missing = required.reject { |field| row.key?(field) }
      error(label, "missing required fields: #{missing.join(', ')}") unless missing.empty?
      id = row["id"]
      unless id.is_a?(String)
        error(label, "id must be a string")
        next
      end
      error(id, "must start with #{prefix.inspect}") unless id.start_with?(prefix)
      error(id, "does not match schema ID pattern #{id_pattern.inspect}") unless regexp.match?(id)
      ids[id] << index
    end
    ids.each { |id, indexes| error(id, "duplicate #{kind} ID at indexes #{indexes.join(', ')}") if indexes.length > 1 }
  rescue RegexpError => e
    error(@schema_path, "invalid conventions.ids.pattern: #{e.message}")
  end

  def validate_evidence(evidence, label, source_ids, schema, status: nil)
    unless evidence.is_a?(Hash)
      error(label, "evidence must be a mapping")
      return
    end
    grade = evidence["grade"]
    allowed_grades = schema.fetch("evidence_grades", {}).keys
    error(label, "unknown evidence grade #{grade.inspect}") unless allowed_grades.include?(grade)
    refs = evidence["sources"]
    unless refs.is_a?(Array)
      error(label, "evidence.sources must be an array")
      refs = []
    end
    error(label, "grade #{grade} evidence requires at least one source") if %w[A B C D].include?(grade) && refs.empty?
    validate_source_references(refs, label, source_ids)
    if %w[A B C D].include?(grade)
      grade_rank = {"A" => 0, "B" => 1, "C" => 2, "D" => 3}
      cited_grades = refs.filter_map { |reference| source_ids.dig(reference["source_id"], "evidence_grade") if reference.is_a?(Hash) }
      best_cited_rank = cited_grades.filter_map { |source_grade| grade_rank[source_grade] }.min
      if best_cited_rank && grade_rank[grade] < best_cited_rank
        error(label, "claim grade #{grade} is stronger than all cited source grades #{cited_grades.uniq.sort.inspect}")
      end
    end

    return if status.nil?

    allowed_statuses = schema.dig("entities", "edge", "fields", "status", "enum") || []
    error(label, "unknown edge status #{status.inspect}") unless allowed_statuses.include?(status)
    error(label, "validated status requires A/B evidence, got #{grade.inspect}") if status == "validated" && !%w[A B].include?(grade)
    error(label, "grade X evidence requires inferred status") if grade == "X" && status != "inferred"
    error(label, "inferred status requires grade X evidence") if status == "inferred" && grade != "X"
  end

  def validate_source_references(refs, label, source_ids)
    refs.each_with_index do |reference, index|
      reference_label = "#{label}.sources[#{index}]"
      unless reference.is_a?(Hash)
        error(reference_label, "must be a mapping")
        next
      end
      %w[source_id supports].each { |field| error(reference_label, "missing #{field}") unless reference.key?(field) }
      source_id = reference["source_id"]
      error(reference_label, "references missing source #{source_id.inspect}") unless source_ids[source_id]
      validate_iso_day(reference["accessed_date"], "#{reference_label}.accessed_date") unless reference["accessed_date"].nil?
    end
  end

  def endpoint_type(id, org_ids, person_ids)
    return "organization" if org_ids[id]
    return "person" if person_ids[id]

    nil
  end

  def date_bounds(value, label)
    return nil if value.nil?
    return nil unless validate_date_value(value, label)

    text = value["value"]
    case value["precision"]
    when "day"
      day = Date.iso8601(text)
      [day, day]
    when "month"
      year, month = text.split("-").map(&:to_i)
      first = Date.new(year, month, 1)
      [first, Date.new(year, month, -1)]
    when "year"
      year = Integer(text, 10)
      [Date.new(year, 1, 1), Date.new(year, 12, 31)]
    when "unknown"
      nil
    end
  rescue ArgumentError, Date::Error
    nil
  end

  def validate_date_value(value, label)
    unless value.is_a?(Hash)
      error(label, "must be a {value, precision} mapping")
      return false
    end
    text = value["value"]
    precision = value["precision"]
    patterns = {"day" => /\A\d{4}-\d{2}-\d{2}\z/, "month" => /\A\d{4}-\d{2}\z/, "year" => /\A\d{4}\z/}
    unless precision == "unknown" || patterns.key?(precision)
      error(label, "invalid precision #{precision.inspect}")
      return false
    end
    return true if precision == "unknown" && text.nil?
    unless text.is_a?(String) && patterns[precision]&.match?(text)
      error(label, "value #{text.inspect} does not match precision #{precision.inspect}")
      return false
    end
    date_bounds_without_validation(text, precision)
    true
  rescue ArgumentError, Date::Error
    error(label, "contains an invalid calendar date #{text.inspect}")
    false
  end

  def date_bounds_without_validation(text, precision)
    case precision
    when "day" then Date.iso8601(text)
    when "month"
      year, month = text.split("-").map(&:to_i)
      Date.new(year, month, 1)
    when "year" then Date.new(Integer(text, 10), 1, 1)
    end
  end

  def validate_iso_day(value, label)
    unless value.is_a?(String) && /\A\d{4}-\d{2}-\d{2}\z/.match?(value)
      error(label, "must be an ISO-8601 day string")
      return
    end
    Date.iso8601(value)
  rescue Date::Error
    error(label, "contains an invalid calendar date #{value.inspect}")
  end

  def warning(location, message)
    warnings << "#{location}: #{message}"
  end

  def error(location, message)
    errors << "#{location}: #{message}"
  end
end

options = {
  schema_path: "schema.yaml",
  edge_types_path: "edge_types.yaml",
  organization_paths: [],
  people_paths: [],
  source_paths: [],
  event_paths: [],
  edge_paths: []
}

parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby scripts/validate_data.rb [options]"
  opts.on("--schema PATH", "Schema contract (default: schema.yaml)") { |path| options[:schema_path] = path }
  opts.on("--edge-types PATH", "Edge-type catalog (default: edge_types.yaml)") { |path| options[:edge_types_path] = path }
  opts.on("--organization-file PATH", "Organization collection; repeatable") { |path| options[:organization_paths] << path }
  opts.on("--people-file PATH", "People collection; repeatable") { |path| options[:people_paths] << path }
  opts.on("--source-file PATH", "Source collection; repeatable") { |path| options[:source_paths] << path }
  opts.on("--event-file PATH", "Event collection; repeatable") { |path| options[:event_paths] << path }
  opts.on("--edge-file PATH", "Edge collection; repeatable") { |path| options[:edge_paths] << path }
  opts.on("--canonical DIR", "Validate DIR/{organizations,people,sources,events,edges}.yaml") do |dir|
    options[:organization_paths] << File.join(dir, "organizations.yaml")
    options[:people_paths] << File.join(dir, "people.yaml")
    options[:source_paths] << File.join(dir, "sources.yaml")
    options[:event_paths] << File.join(dir, "events.yaml")
    options[:edge_paths] << File.join(dir, "edges.yaml")
  end
end

begin
  parser.parse!
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 2
end

unless ARGV.empty?
  warn "Unexpected positional arguments: #{ARGV.join(' ')}"
  warn parser
  exit 2
end

options[:organization_paths] = ["organizations_seed.yaml"] if options[:organization_paths].empty?
options[:people_paths] = ["people_seed.yaml"] if options[:people_paths].empty?

validator = DataValidator.new(**options)
if validator.run
  puts "Validation passed"
  puts "  organization files: #{validator.counts['organization_files']}"
  puts "  organizations: #{validator.counts['organizations']}"
  puts "  people files: #{validator.counts['people_files']}"
  puts "  people: #{validator.counts['people']}"
  puts "  source files: #{validator.counts['source_files']}"
  puts "  sources: #{validator.counts['sources']}"
  puts "  event files: #{validator.counts['event_files']}"
  puts "  events: #{validator.counts['events']}"
  puts "  edge files: #{validator.counts['edge_files']}"
  puts "  edges: #{validator.counts['edges']}"
  puts "  edge types: #{validator.counts['edge_types']}"
  unless validator.warnings.empty?
    puts "Warnings (#{validator.warnings.length}):"
    validator.warnings.each { |message| puts "  - #{message}" }
  end
  exit 0
end

warn "Validation failed with #{validator.errors.length} error(s):"
validator.errors.each { |message| warn "  - #{message}" }
exit 1
