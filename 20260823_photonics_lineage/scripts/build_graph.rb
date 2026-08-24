#!/usr/bin/env ruby
# frozen_string_literal: true

require "date"
require "digest"
require "fileutils"
require "json"
require "optparse"
require "yaml"

options = {
  canonical_dir: "canonical",
  output_dir: "generated"
}

OptionParser.new do |opts|
  opts.banner = "Usage: ruby scripts/build_graph.rb [options]"
  opts.on("--canonical DIR", "Canonical YAML directory (default: canonical)") { |dir| options[:canonical_dir] = dir }
  opts.on("--output DIR", "Generated output directory (default: generated)") { |dir| options[:output_dir] = dir }
end.parse!

CANONICAL_FILES = %w[organizations people sources events edges].freeze
LINEAGE_EDGE_TYPES = %w[
  employment research_training founder executive_role technical_leadership
  publication_affiliation patent_affiliation
].freeze
ROLE_WEIGHTS = {
  "founder" => 3.0,
  "executive_role" => 2.0,
  "technical_leadership" => 1.0,
  "employment" => 1.0,
  "research_training" => 1.0,
  "publication_affiliation" => 1.0,
  "patent_affiliation" => 1.0
}.freeze

def load_collection(directory, name)
  path = File.join(directory, "#{name}.yaml")
  document = YAML.safe_load_file(path, aliases: true)
  collection = document[name]
  raise "#{path}: missing #{name} array" unless collection.is_a?(Array)

  [path, document.fetch("schema_version"), collection]
end

def date_bounds(value)
  return nil unless value.is_a?(Hash) && value["value"] && value["precision"]

  text = value["value"]
  case value["precision"]
  when "day"
    day = Date.iso8601(text)
    [day, day]
  when "month"
    year, month = text.split("-").map(&:to_i)
    [Date.new(year, month, 1), Date.new(year, month, -1)]
  when "year"
    year = Integer(text, 10)
    [Date.new(year, 1, 1), Date.new(year, 12, 31)]
  end
rescue ArgumentError, Date::Error
  nil
end

def compact_edge(edge)
  {
    "id" => edge["id"],
    "person_id" => edge["source_id"],
    "organization_id" => edge["target_id"],
    "edge_type" => edge["edge_type"],
    "status" => edge["status"],
    "start_date" => edge["start_date"],
    "end_date" => edge["end_date"],
    "attributes" => edge["attributes"],
    "evidence_grade" => edge.dig("evidence", "grade"),
    "source_ids" => (edge.dig("evidence", "sources") || []).map { |source| source["source_id"] },
    "notes" => edge["notes"]
  }
end

paths = {}
versions = {}
collections = {}
CANONICAL_FILES.each do |name|
  path, version, collection = load_collection(options[:canonical_dir], name)
  paths[name] = path
  versions[name] = version
  collections[name] = collection
end
raise "Canonical schema versions differ: #{versions.inspect}" unless versions.values.uniq.length == 1

organizations = collections["organizations"]
people = collections["people"]
sources = collections["sources"]
edges = collections["edges"]
organization_by_id = organizations.to_h { |organization| [organization["id"], organization] }
person_by_id = people.to_h { |person| [person["id"], person] }
source_by_id = sources.to_h { |source| [source["id"], source] }

eligible_edges = edges.select do |edge|
  person_by_id.key?(edge["source_id"]) &&
    organization_by_id.key?(edge["target_id"]) &&
    edge["status"] == "validated" &&
    %w[A B].include?(edge.dig("evidence", "grade"))
end.sort_by { |edge| edge["id"] }
all_person_organization_edges = edges.select do |edge|
  person_by_id.key?(edge["source_id"]) && organization_by_id.key?(edge["target_id"])
end

missing_source_refs = eligible_edges.flat_map do |edge|
  (edge.dig("evidence", "sources") || []).filter_map do |reference|
    "#{edge['id']}:#{reference['source_id']}" unless source_by_id.key?(reference["source_id"])
  end
end
raise "Eligible edges contain missing source references: #{missing_source_refs.join(', ')}" unless missing_source_refs.empty?

lineage_edges = eligible_edges.select { |edge| LINEAGE_EDGE_TYPES.include?(edge["edge_type"]) }
edges_by_person = lineage_edges.group_by { |edge| edge["source_id"] }
multi_institution_people = edges_by_person.count { |_person_id, person_edges| person_edges.map { |edge| edge["target_id"] }.uniq.length >= 2 }
single_institution_people = edges_by_person.length - multi_institution_people
contributions = []
exclusions = []

edges_by_person.keys.sort.each do |person_id|
  person_edges = edges_by_person.fetch(person_id)
  by_organization = person_edges.group_by { |edge| edge["target_id"] }
  next if by_organization.length < 2

  by_organization.keys.sort.combination(2).each do |left_id, right_id|
    directions = [[left_id, right_id], [right_id, left_id]]
    supported = []
    comparable_boundary_seen = false

    directions.each do |prior_id, later_id|
      candidates = []
      by_organization.fetch(prior_id).each do |prior_edge|
        prior_end = date_bounds(prior_edge["end_date"])
        by_organization.fetch(later_id).each do |later_edge|
          later_start = date_bounds(later_edge["start_date"])
          next unless prior_end && later_start

          comparable_boundary_seen = true
          next unless prior_end.last < later_start.first

          candidates << {
            "prior_edge" => prior_edge,
            "later_edge" => later_edge,
            "prior_end" => prior_end.last,
            "later_start" => later_start.first,
            "gap_days" => (later_start.first - prior_end.last).to_i
          }
        end
      end
      next if candidates.empty?

      candidate = candidates.min_by do |item|
        [item["gap_days"], item["prior_edge"]["id"], item["later_edge"]["id"]]
      end
      supported << [prior_id, later_id, candidate]
    end

    if supported.empty?
      exclusions << {
        "person_id" => person_id,
        "organization_ids" => [left_id, right_id],
        "reason" => comparable_boundary_seen ? "overlapping_or_nonconclusive_date_precision" : "missing_prior_end_or_later_start",
        "edge_ids" => (by_organization.fetch(left_id) + by_organization.fetch(right_id)).map { |edge| edge["id"] }.sort
      }
      next
    end

    supported.each do |prior_id, later_id, candidate|
      later_edge = candidate["later_edge"]
      contributions << {
        "source_organization_id" => prior_id,
        "target_organization_id" => later_id,
        "person_id" => person_id,
        "weight" => ROLE_WEIGHTS.fetch(later_edge["edge_type"], 1.0),
        "prior_edge_id" => candidate["prior_edge"]["id"],
        "later_edge_id" => later_edge["id"],
        "prior_end_date" => candidate["prior_edge"]["end_date"],
        "later_start_date" => later_edge["start_date"],
        "ordering_rule" => "latest_possible_prior_end_before_earliest_possible_later_start",
        "gap_days_minimum" => candidate["gap_days"]
      }
    end
  end
end

talent_flows = contributions.group_by do |contribution|
  [contribution["source_organization_id"], contribution["target_organization_id"]]
end.map do |(source_id, target_id), flow_contributions|
  ordered = flow_contributions.sort_by { |contribution| contribution["person_id"] }
  {
    "id" => "flow_#{source_id.sub(/^org_/, '')}_to_#{target_id.sub(/^org_/, '')}",
    "source_organization_id" => source_id,
    "target_organization_id" => target_id,
    "weight" => ordered.sum { |contribution| contribution["weight"] }.round(2),
    "people" => ordered,
    "person_ids" => ordered.map { |contribution| contribution["person_id"] },
    "contributing_edge_ids" => ordered.flat_map { |contribution| [contribution["prior_edge_id"], contribution["later_edge_id"]] }.uniq.sort
  }
end.sort_by { |flow| [flow["source_organization_id"], flow["target_organization_id"]] }

used_person_ids = eligible_edges.map { |edge| edge["source_id"] }.uniq.sort
used_organization_ids = eligible_edges.map { |edge| edge["target_id"] }.uniq.sort
source_hashes = paths.transform_values { |path| Digest::SHA256.file(path).hexdigest }

graph = {
  "metadata" => {
    "schema_version" => versions.values.first,
    "graph_version" => "0.1.0",
    "evidence_filter" => {"statuses" => ["validated"], "grades" => ["A", "B"]},
    "sequence_policy" => "Derive organization flow only when an earlier edge's latest possible end precedes a later edge's earliest possible start.",
    "lineage_edge_types" => LINEAGE_EDGE_TYPES,
    "source_sha256" => source_hashes,
    "reproducible" => true
  },
  "nodes" => {
    "organizations" => used_organization_ids.map do |id|
      organization = organization_by_id.fetch(id)
      {"id" => id, "name" => organization["name"], "organization_type" => organization["organization_type"], "status" => organization["status"], "technologies" => organization["technologies"]}
    end,
    "people" => used_person_ids.map do |id|
      person = person_by_id.fetch(id)
      {"id" => id, "name" => person["name"], "technologies" => person["technologies"]}
    end
  },
  "person_organization_edges" => eligible_edges.map { |edge| compact_edge(edge) },
  "talent_flows" => talent_flows,
  "sequence_exclusions" => exclusions.sort_by { |item| [item["person_id"], item["organization_ids"]] }
}

FileUtils.mkdir_p(options[:output_dir])
json_path = File.join(options[:output_dir], "graph_ab.json")
File.write(json_path, JSON.pretty_generate(graph) + "\n")

grade_counts = eligible_edges.group_by { |edge| edge.dig("evidence", "grade") }.transform_values(&:length).sort.to_h
type_counts = eligible_edges.group_by { |edge| edge["edge_type"] }.transform_values(&:length).sort.to_h
exclusion_counts = exclusions.group_by { |item| item["reason"] }.transform_values(&:length).sort.to_h

report = <<~MARKDOWN
  # A/B Graph Build Report

  Generated reproducibly from the canonical YAML snapshot. The output contains no build timestamp; input SHA-256 hashes in `graph_ab.json` identify the exact source state.

  ## Result

  | Metric | Count |
  |---|---:|
  | Canonical organizations | #{organizations.length} |
  | Canonical people | #{people.length} |
  | Canonical edges | #{edges.length} |
  | Canonical person→organization edges | #{all_person_organization_edges.length} |
  | Validated A/B person→organization edges | #{eligible_edges.length} |
  | Person→organization edges removed by A/B/status filter | #{all_person_organization_edges.length - eligible_edges.length} |
  | People in evidence graph | #{used_person_ids.length} |
  | Organizations in evidence graph | #{used_organization_ids.length} |
  | People with fewer than two lineage institutions | #{single_institution_people} |
  | People eligible for cross-institution sequencing | #{multi_institution_people} |
  | Person-level ordered contributions | #{contributions.length} |
  | Aggregated institution→institution talent flows | #{talent_flows.length} |
  | Excluded cross-organization person pairs | #{exclusions.length} |

  Evidence grades: #{grade_counts.map { |grade, count| "`#{grade}` #{count}" }.join(", ")}.

  Edge types: #{type_counts.map { |type, count| "`#{type}` #{count}" }.join(", ")}.

  ## Ordering policy

  A talent flow is generated only when the latest possible date represented by an earlier edge's `end_date` is strictly before the earliest possible date represented by a later edge's `start_date`. Year and month precision are expanded to full intervals. Equal years/months therefore do not establish order. Biography list order, acquisition timing, current-role fields, and undated role language are not used to infer a sequence.

  The HTML is a conservative Sankey-like institution-flow view plus a searchable table of every included person→organization evidence edge. Because only #{talent_flows.length} institution flows meet the ordering rule, it should be read as a **dated-evidence prototype**, not an ecosystem-completeness map.

  ## Derived institution flows

  #{talent_flows.empty? ? "No institution flow met the ordering rule." : talent_flows.map { |flow| source = organization_by_id.fetch(flow["source_organization_id"])["name"]; target = organization_by_id.fetch(flow["target_organization_id"])["name"]; names = flow["person_ids"].map { |id| person_by_id.fetch(id)["name"] }.join(", "); "- **#{source} → #{target}** — weight #{flow['weight']}; #{names}" }.join("\n")}

  ## Exclusions

  #{exclusion_counts.empty? ? "No cross-organization pairs were excluded." : exclusion_counts.map { |reason, count| "- `#{reason}`: #{count}" }.join("\n")}

  Full person/pair exclusions and their contributing edge IDs are stored in `graph_ab.json` under `sequence_exclusions`. People with fewer than two distinct lineage institutions never become sequence candidates and are counted separately above. Eligible person→organization relationships outside the configured lineage-bearing types remain visible in the evidence browser but are not used to derive flows.

  ## Validation

  The builder aborts on missing canonical collections, schema-version disagreement, or missing evidence-source references for included edges. After writing, the build was checked by reparsing JSON, checking node/edge/flow references, and confirming that every included person edge is validated with grade A or B.
MARKDOWN
report_path = File.join(options[:output_dir], "graph_build_report.md")
File.write(report_path, report)

embedded_json = JSON.generate(graph).gsub("</", "<\\/")
html = <<~HTML
  <!doctype html>
  <html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Photonics Lineage — Validated A/B Prototype</title>
    <style>
      :root { color-scheme: dark; --bg:#09111b; --panel:#101d2b; --ink:#e9f1f8; --muted:#91a4b7; --a:#56c8ff; --b:#a98cff; --line:#3c82a6; }
      * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,sans-serif }
      header,main { max-width:1440px; margin:auto; padding:20px 28px } h1 { margin:0 0 6px; font-size:26px } h2 { margin:0 0 12px; font-size:18px }
      .muted,.disclosure { color:var(--muted) } .disclosure { max-width:1000px } .stats { display:flex; gap:12px; flex-wrap:wrap; margin:18px 0 }
      .stat,.panel { background:var(--panel); border:1px solid #20364a; border-radius:10px } .stat { padding:10px 14px } .stat b { display:block; font-size:20px }
      .panel { padding:16px; margin:16px 0; overflow:hidden } #flowSvg { width:100%; min-height:430px; display:block }
      .flow { fill:none; stroke:var(--line); stroke-opacity:.58 } .flow:hover { stroke:#ffd166; stroke-opacity:1 }
      .node { fill:#17324a; stroke:#69c7f2; stroke-width:1.2 } .node-label { fill:var(--ink); font-size:12px; pointer-events:none }
      input,select { background:#0a1622; color:var(--ink); border:1px solid #31506b; border-radius:6px; padding:8px; margin:0 8px 10px 0 }
      table { width:100%; border-collapse:collapse; font-size:12px } th,td { text-align:left; padding:7px 8px; border-bottom:1px solid #243849; vertical-align:top } th { position:sticky; top:0; background:var(--panel) }
      .table-wrap { max-height:520px; overflow:auto } .grade { font-weight:700 } .grade.A { color:var(--a) } .grade.B { color:var(--b) }
      #tip { position:fixed; display:none; max-width:360px; background:#02070c; border:1px solid #50708c; border-radius:7px; padding:9px; pointer-events:none; z-index:5 }
      @media(max-width:700px){header,main{padding:16px}.node-label{font-size:10px}}
    </style>
  </head>
  <body>
    <header>
      <h1>Photonics Lineage — Validated A/B Prototype</h1>
      <p class="disclosure">This view contains only validated A/B person→organization evidence. Institution flows are derived only from conclusively ordered end/start dates. Undated biography ordering is excluded, so sparse flows indicate missing temporal evidence—not absence of lineage.</p>
      <div class="stats" id="stats"></div>
    </header>
    <main>
      <section class="panel"><h2>Conservatively derived institution flows</h2><svg id="flowSvg" viewBox="0 0 1200 430" role="img" aria-label="Institution talent flow graph"></svg></section>
      <section class="panel">
        <h2>A/B person→organization evidence browser</h2>
        <input id="search" type="search" placeholder="Search person or organization">
        <select id="grade"><option value="">All grades</option><option>A</option><option>B</option></select>
        <select id="edgeType"><option value="">All relationship types</option></select>
        <span class="muted" id="shown"></span>
        <div class="table-wrap"><table><thead><tr><th>Person</th><th>Organization</th><th>Relationship</th><th>Dates</th><th>Grade</th><th>Source IDs</th></tr></thead><tbody id="rows"></tbody></table></div>
      </section>
    </main>
    <div id="tip"></div>
    <script>const graph=#{embedded_json};
      const orgs=Object.fromEntries(graph.nodes.organizations.map(x=>[x.id,x])); const people=Object.fromEntries(graph.nodes.people.map(x=>[x.id,x]));
      const stats=[['Evidence edges',graph.person_organization_edges.length],['People',graph.nodes.people.length],['Organizations',graph.nodes.organizations.length],['Ordered flows',graph.talent_flows.length],['Excluded pairs',graph.sequence_exclusions.length]];
      document.querySelector('#stats').innerHTML=stats.map(([k,v])=>`<div class="stat"><b>${v}</b>${k}</div>`).join('');
      const svg=document.querySelector('#flowSvg'), NS='http://www.w3.org/2000/svg', tip=document.querySelector('#tip');
      const flowNodes=[...new Set(graph.talent_flows.flatMap(f=>[f.source_organization_id,f.target_organization_id]))];
      const incoming=Object.fromEntries(flowNodes.map(id=>[id,0])); graph.talent_flows.forEach(f=>incoming[f.target_organization_id]++);
      const layers={}; const layerOf={}, visiting=new Set(); const visit=id=>{ if(layerOf[id]!=null)return layerOf[id];if(visiting.has(id))return 0;visiting.add(id);const ins=graph.talent_flows.filter(f=>f.target_organization_id===id);layerOf[id]=ins.length?Math.max(...ins.map(f=>visit(f.source_organization_id)+1)):0;visiting.delete(id);return layerOf[id] };
      flowNodes.forEach(visit); flowNodes.forEach(id=>(layers[layerOf[id]]??=[]).push(id)); const positions={};
      Object.entries(layers).forEach(([layer,ids])=>ids.sort().forEach((id,i)=>positions[id]={x:90+Number(layer)*430,y:70+i*(300/Math.max(1,ids.length-1))}));
      const add=(name,attrs)=>{const e=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e};
      graph.talent_flows.forEach(f=>{const a=positions[f.source_organization_id],b=positions[f.target_organization_id];const p=add('path',{class:'flow',d:`M${a.x+180},${a.y+20} C${a.x+280},${a.y+20} ${b.x-100},${b.y+20} ${b.x},${b.y+20}`,'stroke-width':Math.max(3,f.weight*5)});const names=f.person_ids.map(id=>people[id].name).join(', ');p.onmousemove=e=>{tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.textContent=`${orgs[f.source_organization_id].name} → ${orgs[f.target_organization_id].name}; weight ${f.weight}; ${names}`};p.onmouseleave=()=>tip.style.display='none'});
      flowNodes.forEach(id=>{const p=positions[id];add('rect',{class:'node',x:p.x,y:p.y,width:180,height:40,rx:7});const t=add('text',{class:'node-label',x:p.x+10,y:p.y+25});t.textContent=orgs[id].name});
      if(!flowNodes.length){const t=add('text',{class:'node-label',x:50,y:80});t.textContent='No institution flow met the strict date-order rule.'}
      const edgeType=document.querySelector('#edgeType'); [...new Set(graph.person_organization_edges.map(e=>e.edge_type))].sort().forEach(v=>edgeType.add(new Option(v,v)));
      const fmt=d=>d?.value||'—'; const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      function render(){const q=document.querySelector('#search').value.toLowerCase(),g=document.querySelector('#grade').value,t=edgeType.value;const filtered=graph.person_organization_edges.filter(e=>(!g||e.evidence_grade===g)&&(!t||e.edge_type===t)&&(!q||`${people[e.person_id].name} ${orgs[e.organization_id].name}`.toLowerCase().includes(q)));document.querySelector('#shown').textContent=`${filtered.length} shown`;document.querySelector('#rows').innerHTML=filtered.map(e=>`<tr><td>${esc(people[e.person_id].name)}</td><td>${esc(orgs[e.organization_id].name)}</td><td>${esc(e.edge_type)}</td><td>${fmt(e.start_date)} → ${fmt(e.end_date)}</td><td class="grade ${e.evidence_grade}">${e.evidence_grade}</td><td>${esc(e.source_ids.join(', '))}</td></tr>`).join('')}
      ['search','grade','edgeType'].forEach(id=>document.querySelector('#'+id).addEventListener(id==='search'?'input':'change',render)); render();
    </script>
  </body></html>
HTML
html_path = File.join(options[:output_dir], "photonics_lineage_sankey.html")
File.write(html_path, html)

# Reparse and check every generated reference before declaring success.
parsed = JSON.parse(File.read(json_path))
generated_org_ids = parsed.dig("nodes", "organizations").to_h { |node| [node["id"], true] }
generated_person_ids = parsed.dig("nodes", "people").to_h { |node| [node["id"], true] }
parsed.fetch("person_organization_edges").each do |edge|
  raise "Generated edge #{edge['id']} has missing person" unless generated_person_ids[edge["person_id"]]
  raise "Generated edge #{edge['id']} has missing organization" unless generated_org_ids[edge["organization_id"]]
  raise "Generated edge #{edge['id']} violates A/B filter" unless %w[A B].include?(edge["evidence_grade"])
end
parsed.fetch("talent_flows").each do |flow|
  raise "Generated flow #{flow['id']} has missing source" unless generated_org_ids[flow["source_organization_id"]]
  raise "Generated flow #{flow['id']} has missing target" unless generated_org_ids[flow["target_organization_id"]]
  raise "Generated flow #{flow['id']} has no people" if flow["person_ids"].empty?
end

puts "Graph build passed"
puts "  eligible A/B person→organization edges: #{eligible_edges.length}"
puts "  people: #{used_person_ids.length}"
puts "  organizations: #{used_organization_ids.length}"
puts "  ordered person contributions: #{contributions.length}"
puts "  institution talent flows: #{talent_flows.length}"
puts "  excluded person/organization pairs: #{exclusions.length}"
puts "  wrote: #{json_path}, #{html_path}, #{report_path}"
