#!/usr/bin/env python3
"""Build an auditable Plotly Sankey from canonical lineage data.

Two views are emitted in one self-contained HTML file:

* founder-lineage: individually sourced employment, research-training, or technical-
  leadership institutions feeding a later founder destination. Founder semantics provide
  direction when dates are absent; known contradictory chronology is excluded.
* strict chronology: any supported person path whose earlier edge has a latest possible
  end before the later edge's earliest possible start.

The institution display map changes presentation only. Hover and evidence rows retain
canonical organization IDs, atomic edge IDs, people, grades, and sources.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go
import yaml


LINEAGE_TYPES = {
    "employment",
    "research_training",
    "founder",
    "executive_role",
    "technical_leadership",
    "publication_affiliation",
    "patent_affiliation",
}
FOUNDER_SOURCE_TYPES = {"employment", "research_training", "technical_leadership"}
DEFAULT_WEIGHTS = {
    "founder": 3.0,
    "executive_role": 2.0,
    "technical_leadership": 1.0,
    "employment": 1.0,
    "research_training": 1.0,
    "publication_affiliation": 1.0,
    "patent_affiliation": 1.0,
}
NODE_PALETTE = {
    "university": "#4C78A8",
    "academic_lab": "#6B8FB3",
    "research_institute": "#72B7B2",
    "industrial_lab": "#54A24B",
    "government_lab": "#59A14F",
    "startup": "#F28E2B",
    "company": "#E15759",
    "business_unit": "#B279A2",
    "acquired_business_unit": "#9D755D",
    "venture_fund": "#BAB0AC",
    "other": "#79706E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default="canonical", help="Canonical YAML directory")
    parser.add_argument(
        "--display-map",
        default="config/institution_display_map.yaml",
        help="Institution display/dedup YAML",
    )
    parser.add_argument("--output-dir", default="generated", help="Generated artifact directory")
    parser.add_argument("--no-archive", action="store_true", help="Do not preserve an existing HTML artifact")
    parser.add_argument("--no-png", action="store_true", help="Skip optional Kaleido PNG export")
    return parser.parse_args()


def load_collection(directory: Path, name: str) -> tuple[str, list[dict[str, Any]], Path]:
    path = directory / f"{name}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get(name), list):
        raise ValueError(f"{path}: missing {name} array")
    return str(document.get("schema_version")), document[name], path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def date_bounds(value: Any) -> tuple[date, date] | None:
    if not isinstance(value, dict) or not value.get("value") or not value.get("precision"):
        return None
    text = str(value["value"])
    precision = value["precision"]
    try:
        if precision == "day":
            parsed = date.fromisoformat(text)
            return parsed, parsed
        if precision == "month":
            year, month = map(int, text.split("-"))
            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)
            return date(year, month, 1), date.fromordinal(next_month.toordinal() - 1)
        if precision == "year":
            year = int(text)
            return date(year, 1, 1), date(year, 12, 31)
    except (TypeError, ValueError):
        return None
    return None


def date_label(value: Any) -> str:
    return str(value.get("value")) if isinstance(value, dict) and value.get("value") else "unknown"


def normalize_display_map(document: Any) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Accept the frozen map plus simple map/list variants for forward compatibility."""
    if not isinstance(document, dict):
        return {}, DEFAULT_WEIGHTS.copy(), {}

    weights = DEFAULT_WEIGHTS.copy()
    configured_weights = document.get("role_weights") or document.get("weights") or document.get("talent_weights")
    if isinstance(configured_weights, dict):
        for key, value in configured_weights.items():
            if key in weights and isinstance(value, (int, float)):
                weights[key] = float(value)

    raw = (
        document.get("institution_display_map")
        or document.get("display_map")
        or document.get("mappings")
        or document.get("institutions")
        or document.get("clusters")
        or document.get("display_clusters")
        or {}
    )
    result: dict[str, dict[str, Any]] = {}

    if isinstance(raw, dict):
        for organization_id, value in raw.items():
            if not str(organization_id).startswith("org_"):
                continue
            if isinstance(value, str):
                result[organization_id] = {"display_id": value, "display_name": value}
            elif isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("display_id", entry.get("cluster_id") or entry.get("id") or organization_id)
                entry.setdefault("display_name", entry.get("name") or entry["display_id"])
                result[organization_id] = entry
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            members = entry.get("member_org_ids") or entry.get("organization_ids") or entry.get("members") or entry.get("canonical_ids") or []
            if isinstance(members, str):
                members = [members]
            display_id = entry.get("display_id") or entry.get("cluster_id") or entry.get("id")
            display_name = entry.get("display_name") or entry.get("label") or entry.get("name") or display_id
            for organization_id in members:
                result[str(organization_id)] = {
                    **entry,
                    "display_id": display_id or str(organization_id),
                    "display_name": display_name or str(organization_id),
                }

    settings = document.get("settings") if isinstance(document.get("settings"), dict) else {}
    return result, weights, settings


@dataclass(frozen=True)
class DisplayNode:
    id: str
    name: str
    member_ids: tuple[str, ...]
    organization_type: str


class DisplayResolver:
    def __init__(
        self,
        organizations: dict[str, dict[str, Any]],
        mapping: dict[str, dict[str, Any]],
        *,
        allow_person_affiliation_collapse: bool,
    ):
        self.organizations = organizations
        self.mapping = mapping
        self.allow_person_affiliation_collapse = allow_person_affiliation_collapse

    def id_for(self, organization_id: str) -> str:
        entry = self.mapping.get(organization_id, {})
        if not self.allow_person_affiliation_collapse or not entry.get("aggregation_allowed_for_sankey_only", True):
            return organization_id
        return str(entry.get("display_id") or organization_id)

    def nodes_for(self, organization_ids: Iterable[str]) -> dict[str, DisplayNode]:
        members: dict[str, list[str]] = defaultdict(list)
        for organization_id in sorted(set(organization_ids)):
            members[self.id_for(organization_id)].append(organization_id)
        nodes: dict[str, DisplayNode] = {}
        for display_id, member_ids in sorted(members.items()):
            entries = [self.mapping.get(item, {}) for item in member_ids if self.id_for(item) != item]
            names = [entry.get("display_name") for entry in entries if entry.get("display_name")]
            canonical = [self.organizations[item] for item in member_ids]
            name = str(names[0]) if names else canonical[0]["name"]
            types = [item.get("organization_type", "other") for item in canonical]
            organization_type = max(sorted(set(types)), key=types.count)
            nodes[display_id] = DisplayNode(display_id, name, tuple(member_ids), organization_type)
        return nodes


def source_refs(edge: dict[str, Any]) -> list[str]:
    return sorted({item["source_id"] for item in edge.get("evidence", {}).get("sources", []) if item.get("source_id")})


def compact_contribution(
    person_id: str,
    source_edge: dict[str, Any],
    target_edge: dict[str, Any],
    source_display_id: str,
    target_display_id: str,
    weight: float,
    ordering_basis: str,
) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "source_display_id": source_display_id,
        "target_display_id": target_display_id,
        "source_organization_id": source_edge["target_id"],
        "target_organization_id": target_edge["target_id"],
        "source_edge_id": source_edge["id"],
        "target_edge_id": target_edge["id"],
        "source_edge_type": source_edge["edge_type"],
        "target_edge_type": target_edge["edge_type"],
        "source_end": date_label(source_edge.get("end_date")),
        "target_start": date_label(target_edge.get("start_date")),
        "grade": "B" if "B" in {source_edge["evidence"]["grade"], target_edge["evidence"]["grade"]} else "A",
        "source_ids": sorted(set(source_refs(source_edge) + source_refs(target_edge))),
        "weight": weight,
        "ordering_basis": ordering_basis,
    }


def founder_contributions(
    edges_by_person: dict[str, list[dict[str, Any]]],
    resolver: DisplayResolver,
    weights: dict[str, float],
    acquisition_pairs: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contributions: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for person_id in sorted(edges_by_person):
        person_edges = edges_by_person[person_id]
        founder_edges = [edge for edge in person_edges if edge["edge_type"] == "founder"]
        source_edges = [edge for edge in person_edges if edge["edge_type"] in FOUNDER_SOURCE_TYPES]
        for founder_edge in sorted(founder_edges, key=lambda item: item["id"]):
            target_display = resolver.id_for(founder_edge["target_id"])
            for source_edge in sorted(source_edges, key=lambda item: item["id"]):
                source_display = resolver.id_for(source_edge["target_id"])
                if source_display == target_display:
                    continue
                if (founder_edge["target_id"], source_edge["target_id"]) in acquisition_pairs:
                    exclusions.append({
                        "person_id": person_id,
                        "source_edge_id": source_edge["id"],
                        "founder_edge_id": founder_edge["id"],
                        "reason": "source_is_acquirer_of_founder_destination",
                    })
                    continue
                source_start = date_bounds(source_edge.get("start_date"))
                founder_end = date_bounds(founder_edge.get("end_date"))
                if source_start and founder_end and source_start[0] > founder_end[0]:
                    exclusions.append({
                        "person_id": person_id,
                        "source_edge_id": source_edge["id"],
                        "founder_edge_id": founder_edge["id"],
                        "reason": "source_role_conclusively_after_founder_role",
                    })
                    continue
                source_end = date_bounds(source_edge.get("end_date"))
                founder_start = date_bounds(founder_edge.get("start_date"))
                if source_end and founder_start and source_end[1] < founder_start[0]:
                    basis = "strict_date_order_plus_founder_semantics"
                else:
                    basis = "founder_semantics_dates_incomplete"
                contributions.append(compact_contribution(
                    person_id,
                    source_edge,
                    founder_edge,
                    source_display,
                    target_display,
                    weights["founder"],
                    basis,
                ))
    return deduplicate_contributions(contributions), exclusions


def strict_contributions(
    edges_by_person: dict[str, list[dict[str, Any]]],
    resolver: DisplayResolver,
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contributions: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for person_id in sorted(edges_by_person):
        by_display: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges_by_person[person_id]:
            by_display[resolver.id_for(edge["target_id"])].append(edge)
        display_ids = sorted(by_display)
        for left_index, left_id in enumerate(display_ids):
            for right_id in display_ids[left_index + 1 :]:
                found = False
                for source_id, target_id in ((left_id, right_id), (right_id, left_id)):
                    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
                    for source_edge in by_display[source_id]:
                        source_end = date_bounds(source_edge.get("end_date"))
                        if not source_end:
                            continue
                        for target_edge in by_display[target_id]:
                            target_start = date_bounds(target_edge.get("start_date"))
                            if not target_start or source_end[1] >= target_start[0]:
                                continue
                            candidates.append(((target_start[0] - source_end[1]).days, source_edge, target_edge))
                    if not candidates:
                        continue
                    found = True
                    _, source_edge, target_edge = min(candidates, key=lambda item: (item[0], item[1]["id"], item[2]["id"]))
                    contributions.append(compact_contribution(
                        person_id,
                        source_edge,
                        target_edge,
                        source_id,
                        target_id,
                        weights.get(target_edge["edge_type"], 1.0),
                        "latest_possible_source_end_before_earliest_possible_target_start",
                    ))
                if not found:
                    exclusions.append({"person_id": person_id, "display_ids": [left_id, right_id], "reason": "no_conclusive_date_order"})
    return deduplicate_contributions(contributions), exclusions


def deduplicate_contributions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One person's weight counts once per displayed source→target pair."""
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in sorted(items, key=lambda row: (row["person_id"], row["source_display_id"], row["target_display_id"], row["source_edge_id"], row["target_edge_id"])):
        key = (item["person_id"], item["source_display_id"], item["target_display_id"])
        chosen.setdefault(key, item)
    return list(chosen.values())


def aggregate_flows(contributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in contributions:
        grouped[(item["source_display_id"], item["target_display_id"])].append(item)
    flows = []
    for (source_id, target_id), people in sorted(grouped.items()):
        people = sorted(people, key=lambda item: item["person_id"])
        flows.append({
            "source_display_id": source_id,
            "target_display_id": target_id,
            "weight": round(sum(item["weight"] for item in people), 2),
            "contributions": people,
        })
    return flows


def rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def build_figure(
    title: str,
    flows: list[dict[str, Any]],
    nodes: dict[str, DisplayNode],
    people: dict[str, dict[str, Any]],
    organizations: dict[str, dict[str, Any]],
    mode_note: str,
) -> go.Figure:
    used_ids = sorted({flow["source_display_id"] for flow in flows} | {flow["target_display_id"] for flow in flows})
    index = {node_id: position for position, node_id in enumerate(used_ids)}
    labels = [nodes[node_id].name for node_id in used_ids]
    colors = [NODE_PALETTE.get(nodes[node_id].organization_type, NODE_PALETTE["other"]) for node_id in used_ids]
    node_custom = ["<br>".join(f"{organizations[item]['name']} ({item})" for item in nodes[node_id].member_ids) for node_id in used_ids]
    link_custom = []
    for flow in flows:
        rows = []
        for contribution in flow["contributions"]:
            rows.append(
                f"{people[contribution['person_id']]['name']} [{contribution['grade']}]"
                f"<br>{contribution['source_edge_id']} → {contribution['target_edge_id']}"
                f"<br>{contribution['source_organization_id']} → {contribution['target_organization_id']}"
            )
        link_custom.append("<br><br>".join(rows))
    figure = go.Figure(go.Sankey(
        arrangement="snap",
        valueformat=".1f",
        node={
            "pad": 18,
            "thickness": 18,
            "line": {"color": "rgba(30,41,59,.45)", "width": 0.7},
            "label": labels,
            "color": colors,
            "customdata": node_custom,
            "hovertemplate": "%{label}<br>Canonical members:<br>%{customdata}<extra></extra>",
        },
        link={
            "source": [index[flow["source_display_id"]] for flow in flows],
            "target": [index[flow["target_display_id"]] for flow in flows],
            "value": [flow["weight"] for flow in flows],
            "color": [rgba(colors[index[flow["source_display_id"]]], 0.3) for flow in flows],
            "customdata": link_custom,
            "hovertemplate": "%{source.label} → %{target.label}<br>Talent weight: %{value}<br><br>%{customdata}<extra></extra>",
        },
    ))
    figure.update_layout(
        title={"text": f"{title}<br><sup>{mode_note}</sup>", "x": 0.01, "xanchor": "left"},
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "size": 12, "color": "#172033"},
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#F8FAFC",
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
        height=max(650, min(1200, 420 + len(used_ids) * 12)),
    )
    return figure


def evidence_rows(
    mode: str,
    flows: list[dict[str, Any]],
    nodes: dict[str, DisplayNode],
    people: dict[str, dict[str, Any]],
    organizations: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> str:
    rows = []
    for flow in flows:
        for item in flow["contributions"]:
            links = " ".join(
                f'<a href="{html.escape(sources[source_id]["url"])}" target="_blank" rel="noreferrer">{html.escape(source_id)}</a>'
                for source_id in item["source_ids"] if source_id in sources
            )
            search = " ".join([
                mode,
                people[item["person_id"]]["name"],
                nodes[item["source_display_id"]].name,
                nodes[item["target_display_id"]].name,
                organizations[item["source_organization_id"]]["name"],
                organizations[item["target_organization_id"]]["name"],
                item["source_edge_id"], item["target_edge_id"], *item["source_ids"],
            ]).lower()
            rows.append(
                f'<tr data-mode="{mode}" data-search="{html.escape(search)}">'
                f'<td>{html.escape(people[item["person_id"]]["name"])}</td>'
                f'<td>{html.escape(nodes[item["source_display_id"]].name)}<small>{html.escape(organizations[item["source_organization_id"]]["name"])} · {item["source_edge_type"]}</small></td>'
                f'<td>{html.escape(nodes[item["target_display_id"]].name)}<small>{html.escape(organizations[item["target_organization_id"]]["name"])} · {item["target_edge_type"]}</small></td>'
                f'<td>{item["weight"]:g}</td><td>{html.escape(item["grade"])}</td>'
                f'<td><code>{html.escape(item["source_edge_id"])}</code><br><code>{html.escape(item["target_edge_id"])}</code></td>'
                f'<td>{links}<small>{html.escape(item["ordering_basis"])}</small></td></tr>'
            )
    return "\n".join(rows)


def acquisition_rows(
    edges: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    organizations: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> str:
    rows = []
    for edge in sorted(edges, key=lambda item: item["id"]):
        if edge.get("edge_type") != "acquisition" or edge.get("status") != "validated" or edge.get("evidence", {}).get("grade") not in {"A", "B"}:
            continue
        if edge.get("source_id") not in organizations or edge.get("target_id") not in organizations:
            continue
        event_id = edge.get("attributes", {}).get("event_id")
        event = events.get(event_id, {})
        amount = event.get("value", {}).get("headline_amount") if isinstance(event.get("value"), dict) else None
        currency = event.get("value", {}).get("currency") if isinstance(event.get("value"), dict) else None
        value = f"{currency} {amount:,.0f}" if amount is not None and currency else "not disclosed"
        source_links = " ".join(
            f'<a href="{html.escape(sources[ref["source_id"]]["url"])}" target="_blank" rel="noreferrer">{html.escape(ref["source_id"])}</a>'
            for ref in edge.get("evidence", {}).get("sources", []) if ref.get("source_id") in sources
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(organizations[edge['source_id']]['name'])}</td>"
            f"<td>{html.escape(organizations[edge['target_id']]['name'])}</td>"
            f"<td>{html.escape(date_label(edge.get('start_date')))}</td>"
            f"<td>{html.escape(value)}</td><td>{html.escape(edge['evidence']['grade'])}</td>"
            f"<td><code>{html.escape(edge['id'])}</code><br>{source_links}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="6">No validated A/B acquisition edges.</td></tr>'


def render_html(
    founder_figure: go.Figure,
    strict_figure: go.Figure,
    evidence_html: str,
    acquisition_html: str,
    metadata: dict[str, Any],
) -> str:
    founder_div = founder_figure.to_html(full_html=False, include_plotlyjs=True, div_id="founder-chart", config={"responsive": True, "displaylogo": False})
    strict_div = strict_figure.to_html(full_html=False, include_plotlyjs=False, div_id="strict-chart", config={"responsive": True, "displaylogo": False})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photonics lineage Sankey</title>
<style>
:root{{--ink:#172033;--muted:#5d687b;--line:#d8dee9;--paper:#f8fafc;--panel:#fff;--accent:#2855a6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:22px}} h1{{margin:0 0 4px;font-size:25px}} h2{{margin-top:30px;font-size:19px}} p{{color:var(--muted);line-height:1.5}}
.toolbar{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}} button,input{{font:inherit;border:1px solid var(--line);border-radius:8px;background:white;padding:9px 12px}} button.active{{background:var(--accent);color:white;border-color:var(--accent)}}
.chart,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}} #strict-wrap{{display:none}}
.method{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}} .method div{{background:white;border:1px solid var(--line);border-radius:10px;padding:13px}}
.table-wrap{{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:10px;background:white}} table{{border-collapse:collapse;width:100%;font-size:12px}} th{{position:sticky;top:0;background:#edf2f7;z-index:1;text-align:left}} th,td{{padding:9px;border-bottom:1px solid #e8edf3;vertical-align:top}} small{{display:block;color:var(--muted);margin-top:3px}} code{{font-size:10px}} a{{color:var(--accent)}} .meta{{font:11px ui-monospace,SFMono-Regular,monospace;color:var(--muted);word-break:break-all}}
</style></head><body><main>
<h1>Photonics institution lineage</h1>
<p>Auditable person-level talent flows from canonical A/B validated claims. Display clusters deduplicate institution labels without erasing canonical evidence.</p>
<div class="toolbar"><button id="founder-btn" class="active" onclick="setMode('founder')">Founder lineage</button><button id="strict-btn" onclick="setMode('strict')">Strict chronology</button></div>
<section id="founder-wrap" class="chart">{founder_div}</section><section id="strict-wrap" class="chart">{strict_div}</section>
<h2>Method disclosure</h2><section class="method">
<div><strong>Founder-lineage mode</strong><p>Employment, research-training, and technical-leadership institutions feed an individually evidenced founder destination. Founder semantics justify direction where dates are incomplete; conclusively post-founder source roles are excluded.</p></div>
<div><strong>Strict chronology mode</strong><p>A link exists only when the latest possible end of the source edge precedes the earliest possible start of the target edge. Two generic undated employments are never connected.</p></div>
<div><strong>Weighting and display</strong><p>Each person counts once per displayed source→target pair. Destination weights are founder 3, executive 2, technical/employment/training/paper/patent 1 unless overridden by the frozen display-map configuration.</p></div>
<div><strong>Evidence gate</strong><p>Only validated grade A/B atomic edges enter either Sankey. Acquisition events are shown separately and never imply employee retention or team transfer.</p></div>
</section>
<h2>Searchable contributing evidence</h2><div class="toolbar"><input id="search" type="search" placeholder="Search person, institution, edge or source ID" size="48" oninput="filterRows()"></div>
<div class="table-wrap"><table><thead><tr><th>Person</th><th>Source</th><th>Destination</th><th>Weight</th><th>Grade</th><th>Atomic edges</th><th>Sources / basis</th></tr></thead><tbody id="evidence-body">{evidence_html}</tbody></table></div>
<h2>Validated acquisition overlay</h2><p>Corporate events are context, not talent-flow links.</p>
<div class="table-wrap"><table><thead><tr><th>Acquired organization</th><th>Acquirer</th><th>Effective/start date</th><th>Headline value</th><th>Grade</th><th>Evidence</th></tr></thead><tbody>{acquisition_html}</tbody></table></div>
<h2>Deterministic inputs</h2><pre class="meta">{html.escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre>
</main><script>
let activeMode='founder';
function setMode(mode){{activeMode=mode;document.getElementById('founder-wrap').style.display=mode==='founder'?'block':'none';document.getElementById('strict-wrap').style.display=mode==='strict'?'block':'none';document.getElementById('founder-btn').classList.toggle('active',mode==='founder');document.getElementById('strict-btn').classList.toggle('active',mode==='strict');filterRows();setTimeout(()=>window.dispatchEvent(new Event('resize')),20)}}
function filterRows(){{const q=document.getElementById('search').value.toLowerCase().trim();document.querySelectorAll('#evidence-body tr').forEach(row=>{{row.style.display=(row.dataset.mode===activeMode&&(!q||row.dataset.search.includes(q)))?'':'none'}})}}
filterRows();
</script></body></html>"""


def main() -> int:
    args = parse_args()
    canonical_dir = Path(args.canonical)
    display_path = Path(args.display_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collections: dict[str, list[dict[str, Any]]] = {}
    versions: set[str] = set()
    paths: dict[str, Path] = {}
    for name in ("organizations", "people", "sources", "events", "edges"):
        version, collection, path = load_collection(canonical_dir, name)
        versions.add(version)
        collections[name] = collection
        paths[name] = path
    if len(versions) != 1:
        raise ValueError(f"Canonical schema versions differ: {sorted(versions)}")

    organizations = {item["id"]: item for item in collections["organizations"]}
    people = {item["id"]: item for item in collections["people"]}
    sources = {item["id"]: item for item in collections["sources"]}
    events = {item["id"]: item for item in collections["events"]}

    display_document = yaml.safe_load(display_path.read_text(encoding="utf-8")) if display_path.exists() else {}
    display_map, weights, settings = normalize_display_map(display_document)
    # The frozen map permits selected collapse only for person-affiliation/talent
    # views. Strict chronology and transaction tables retain canonical endpoints.
    founder_resolver = DisplayResolver(
        organizations,
        display_map,
        allow_person_affiliation_collapse=True,
    )
    strict_resolver = DisplayResolver(
        organizations,
        display_map,
        allow_person_affiliation_collapse=False,
    )

    eligible = sorted([
        edge for edge in collections["edges"]
        if edge.get("source_id") in people
        and edge.get("target_id") in organizations
        and edge.get("status") == "validated"
        and edge.get("evidence", {}).get("grade") in {"A", "B"}
        and edge.get("edge_type") in LINEAGE_TYPES
    ], key=lambda item: item["id"])
    missing_sources = sorted({source_id for edge in eligible for source_id in source_refs(edge) if source_id not in sources})
    if missing_sources:
        raise ValueError(f"Eligible edges reference missing sources: {missing_sources}")

    edges_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in eligible:
        edges_by_person[edge["source_id"]].append(edge)

    acquisition_pairs = {
        (edge["source_id"], edge["target_id"])
        for edge in collections["edges"]
        if edge.get("edge_type") == "acquisition"
        and edge.get("status") == "validated"
        and edge.get("evidence", {}).get("grade") in {"A", "B"}
        and edge.get("source_id") in organizations
        and edge.get("target_id") in organizations
    }
    founder_items, founder_exclusions = founder_contributions(edges_by_person, founder_resolver, weights, acquisition_pairs)
    strict_items, strict_exclusions = strict_contributions(edges_by_person, strict_resolver, weights)
    founder_flows = aggregate_flows(founder_items)
    strict_flows = aggregate_flows(strict_items)
    founder_org_ids = {item["source_organization_id"] for item in founder_items} | {item["target_organization_id"] for item in founder_items}
    strict_org_ids = {item["source_organization_id"] for item in strict_items} | {item["target_organization_id"] for item in strict_items}
    founder_nodes = founder_resolver.nodes_for(founder_org_ids)
    strict_nodes = strict_resolver.nodes_for(strict_org_ids)

    founder_figure = build_figure(
        "Founder-lineage talent flows",
        founder_flows,
        founder_nodes,
        people,
        organizations,
        f"{len(founder_flows)} displayed flows · {len(founder_items)} named contributors · founder semantics may supply direction",
    )
    strict_figure = build_figure(
        "Strict chronological talent flows",
        strict_flows,
        strict_nodes,
        people,
        organizations,
        f"{len(strict_flows)} displayed flows · {len(strict_items)} named contributors · conclusive date ordering only",
    )

    evidence_html = evidence_rows("founder", founder_flows, founder_nodes, people, organizations, sources) + evidence_rows("strict", strict_flows, strict_nodes, people, organizations, sources)
    acquisition_html = acquisition_rows(collections["edges"], events, organizations, sources)
    metadata = {
        "schema_version": next(iter(versions)),
        "canonical_sha256": {name: sha256(path) for name, path in sorted(paths.items())},
        "display_map_path": str(display_path),
        "display_map_sha256": sha256(display_path) if display_path.exists() else None,
        "display_map_missing": not display_path.exists(),
        "display_policy": "Allowed clusters collapse only in founder/person-affiliation mode; strict chronology and acquisitions keep canonical endpoints.",
        "evidence_filter": {"status": "validated", "grades": ["A", "B"]},
        "role_weights": weights,
        "eligible_atomic_edges": len(eligible),
        "founder_mode": {"flows": len(founder_flows), "contributors": len(founder_items), "exclusions": len(founder_exclusions)},
        "strict_mode": {"flows": len(strict_flows), "contributors": len(strict_items), "exclusions": len(strict_exclusions)},
        "display_settings": settings,
    }

    html_path = output_dir / "photonics_lineage_sankey.html"
    if html_path.exists() and not args.no_archive:
        archive = output_dir / "archive" / "photonics_lineage_sankey.pre_plotly.html"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(html_path, archive)
    html_path.write_text(render_html(founder_figure, strict_figure, evidence_html, acquisition_html, metadata), encoding="utf-8")

    png_path = output_dir / "photonics_lineage_sankey.png"
    png_status = "skipped"
    if not args.no_png:
        try:
            founder_figure.write_image(png_path, width=1800, height=max(900, founder_figure.layout.height or 900), scale=1.25)
            png_status = str(png_path)
        except Exception as error:  # Kaleido/Chrome availability is environment-specific.
            png_status = f"unavailable: {type(error).__name__}: {error}"

    print(f"Wrote {html_path}")
    print(f"Founder mode: {len(founder_flows)} flows / {len(founder_items)} contributors")
    print(f"Strict mode: {len(strict_flows)} flows / {len(strict_items)} contributors")
    print(f"PNG: {png_status}")
    if not display_path.exists():
        print(f"WARNING: {display_path} not found; canonical organizations were rendered without display deduplication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
