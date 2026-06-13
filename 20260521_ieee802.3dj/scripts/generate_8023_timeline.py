#!/usr/bin/env python3
"""Generate the IEEE 802.3 AI map timeline from cached meeting indexes."""

from __future__ import annotations

import html.parser
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "ieee802_3_ai_map" / "ieee8023-ai-relationships.svg"
OUT_JSON = ROOT / "ieee802_3_ai_map" / "timeline_data.json"

START_MARKER = "  <!-- TIMELINE:START -->"
END_MARKER = "  <!-- TIMELINE:END -->"

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


class TextParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "li", "p", "tr"}:
            self.parts.append("\n")


@dataclass(frozen=True)
class TimelineItem:
    key: str
    label: str
    lane: str
    start: str
    end: str
    status: str
    color: str
    source: str
    note: str = ""
    precision: str = "day"
    open_ended: bool = False


def html_text(path: Path) -> str:
    parser = TextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.parts))


def parse_date_phrase(text: str) -> list[date]:
    dates: list[date] = []

    # 11-12, 2025 after a month token is handled as day 11.
    patterns = [
        r"\b(\d{1,2})(?:\s*[-–]\s*\d{1,2})?\s+([A-Za-z]+)\s+(\d{4})\b",
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:\s*[-–]\s*\d{1,2})?,?\s+(\d{4})\b",
        r"\b([A-Za-z]+)\s+(\d{4})\b",
        r"\b(\d{2})_(\d{2})(\d{2})?\b",
    ]

    for day, month_name, year in re.findall(patterns[0], text):
        month = MONTHS.get(month_name.lower())
        if month:
            dates.append(date(int(year), month, int(day)))

    for month_name, day, year in re.findall(patterns[1], text):
        month = MONTHS.get(month_name.lower())
        if month:
            dates.append(date(int(year), month, int(day)))

    for month_name, year in re.findall(patterns[2], text):
        month = MONTHS.get(month_name.lower())
        if month:
            dates.append(date(int(year), month, 1))

    for year, month, day in re.findall(patterns[3], text):
        full_year = 2000 + int(year)
        if 2020 <= full_year <= 2026:
            dates.append(date(full_year, int(month), int(day or "1")))

    return [d for d in dates if date(2020, 1, 1) <= d <= date(2026, 12, 31)]


def range_from_dates(dates: list[date]) -> tuple[date, date]:
    if not dates:
        raise ValueError("no dates")
    return min(dates), max(dates)


def dj_range() -> tuple[date, date]:
    slugs = [p.name for p in (ROOT / "ieee802_3dj_cache").iterdir() if p.is_dir()]
    dates = []
    for slug in slugs:
        match = re.fullmatch(r"(\d{2})_(\d{2})(\d{2})?", slug)
        if match:
            dates.append(date(2000 + int(match.group(1)), int(match.group(2)), int(match.group(3) or "1")))
    return range_from_dates(dates)


def e4ai_range() -> tuple[date, date]:
    pages = json.loads((ROOT / "ieee802_e4ai_metadata/pages.json").read_text(encoding="utf-8"))["pages"]
    dates: list[date] = []
    for page in pages:
        if page.get("kind") not in {"meeting", "workshop", "external_workshop"}:
            continue
        dates.extend(parse_date_phrase(f"{page.get('slug', '')} {page.get('title', '')} {page.get('url', '')}"))
    high_level = ROOT / "ieee802_3_ai_map" / "high_level_pages" / "e4ai.html"
    if high_level.exists():
        text = html_text(high_level)
        meeting_materials = text.split("Meeting materials", 1)[-1].split("Footer", 1)[0]
        dates.extend(parse_date_phrase(meeting_materials))
    return range_from_dates(dates)


def ai_related_range(project_slug: str) -> tuple[date, date]:
    pages = json.loads((ROOT / "ieee802_3_ai_related_metadata/projects.json").read_text(encoding="utf-8"))["pages"]
    dates: list[date] = []
    for page in pages:
        if page.get("project_slug") != project_slug or page.get("kind") != "meeting":
            continue
        if page.get("slug") == "presentproc.html":
            continue
        dates.extend(parse_date_phrase(f"{page.get('slug', '')} {page.get('title', '')}"))
    return range_from_dates(dates)


def high_level_public_range(slug: str) -> tuple[date, date]:
    path = ROOT / "ieee802_3_ai_map" / "high_level_pages" / f"{slug}.html"
    return range_from_dates(parse_date_phrase(html_text(path)))


def build_items() -> list[TimelineItem]:
    items: list[TimelineItem] = []

    def add(
        key: str,
        label: str,
        lane: str,
        rng: tuple[date, date],
        status: str,
        color: str,
        source: str,
        note: str = "",
        precision: str = "day",
        open_ended: bool = False,
    ) -> None:
        items.append(
            TimelineItem(
                key=key,
                label=label,
                lane=lane,
                start=rng[0].isoformat(),
                end=rng[1].isoformat(),
                status=status,
                color=color,
                source=source,
                note=note,
                precision=precision,
                open_ended=open_ended,
            )
        )

    add(
        "B400G",
        "B400G SG",
        "PHY speed lineage",
        high_level_public_range("b400g-public"),
        "completed SG",
        "orange",
        "b400g-public.html",
        "start month-level from public index",
        "month",
    )
    add("802.3df", "P802.3df TF", "PHY speed lineage", high_level_public_range("df-public"), "completed TF", "blue", "df-public.html", "range from public index", "month")
    add("802.3dj", "P802.3dj TF", "PHY speed lineage", dj_range(), "cached active/late-stage TF", "blue", "ieee802_3dj_cache/*", "range from cached meeting-folder slugs", "month", True)
    add("E4AI", "E4AI / NEA", "AI requirements", e4ai_range(), "cached assessment", "green", "ieee802_e4ai_metadata/pages.json + high_level_pages/e4ai.html", "includes E4AI workshop pages and high-level meeting-materials listing", "day", True)
    add("200GMMF", "200GMMF SG", "MMF short reach", ai_related_range("200GMMF"), "completed SG", "green", "ieee802_3_ai_related_metadata/projects.json")
    add("802.3ds", "P802.3ds TF", "MMF short reach", ai_related_range("ds"), "cached active TF", "green", "ieee802_3_ai_related_metadata/projects.json", open_ended=True)
    add("400GPL", "400GPL SG", "400G/lane", ai_related_range("400GPL"), "cached active SG", "blue", "ieee802_3_ai_related_metadata/projects.json", "right edge includes scheduled/future cached meetings", open_ended=True)
    add(
        "COM",
        "COM ad hoc",
        "Support / tools",
        high_level_public_range("com-public"),
        "support ad hoc",
        "orange",
        "com-public.html",
        "public index plus next-meeting date",
    )
    # High-level only; included as milestones, not meeting-range bars.
    add("802.3dq", "P802.3dq", "Support / adjacent", (date(2026, 1, 1), date(2026, 6, 13)), "high-level cached", "orange", "dq.html", "meeting cache pending", open_ended=True)
    add("802.3dt", "P802.3dt", "Support / adjacent", (date(2026, 1, 1), date(2026, 6, 13)), "high-level cached", "muted", "dt.html", "meeting cache pending", open_ended=True)
    return items


def color_classes(color: str) -> tuple[str, str]:
    return {
        "green": ("node", "edge"),
        "blue": ("node2", "edgeBlue"),
        "orange": ("node3", "edgeOrange"),
        "muted": ("mutedNode", "edgeMuted"),
    }[color]


def render_timeline(items: list[TimelineItem]) -> str:
    min_date = date(2021, 1, 1)
    max_date = date(2026, 12, 31)
    x0, x1 = 130, 1660
    y_top, y_bottom = 920, 1290
    lane_y = {
        "MMF short reach": 930,
        "PHY speed lineage": 1010,
        "400G/lane": 1076,
        "AI requirements": 1142,
        "Support / tools": 1208,
        "Support / adjacent": 1208,
    }
    lane_labels = {
        "MMF short reach": 953,
        "PHY speed lineage": 1033,
        "400G/lane": 1099,
        "AI requirements": 1165,
        "Support / tools": 1231,
        "Support / adjacent": 1231,
    }

    total_days = (max_date - min_date).days

    def x_for(value: date) -> float:
        return x0 + ((value - min_date).days / total_days) * (x1 - x0)

    placements: list[tuple[TimelineItem, float, float, int]] = []
    for lane in lane_y:
        lane_items = sorted([item for item in items if item.lane == lane], key=lambda item: (item.start, item.end))
        sublane_ends: list[float] = []
        for item in lane_items:
            start = datetime.fromisoformat(item.start).date()
            end = datetime.fromisoformat(item.end).date()
            x = x_for(start)
            w = max(112, x_for(end) - x)
            if x + w > x1:
                w = x1 - x
            sublane = 0
            while sublane < len(sublane_ends) and x < sublane_ends[sublane] + 14:
                sublane += 1
            if sublane == len(sublane_ends):
                sublane_ends.append(x + w)
            else:
                sublane_ends[sublane] = x + w
            placements.append((item, x, w, sublane))

    lines: list[str] = [
        START_MARKER,
        '  <rect class="panel" x="50" y="850" width="1700" height="500"/>',
        '  <text class="section" x="76" y="887">Timeline View</text>',
        '  <text class="tiny" x="76" y="910">Workstream chronology and lineage</text>',
    ]

    for year in range(2021, 2027):
        x = x_for(date(year, 1, 1))
        lines.append(f'  <line class="tick" x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bottom}"/>')
        lines.append(f'  <text class="tiny" x="{x - 16:.1f}" y="{y_bottom + 25}">{year}</text>')

    labels_by_y: dict[int, list[str]] = {}
    for lane, y in lane_labels.items():
        labels_by_y.setdefault(y, []).append(lane)
    for y, labels in labels_by_y.items():
        label = " + ".join(label.replace("Support / ", "") for label in labels)
        if len(labels) > 1 and all(item.startswith("Support / ") for item in labels):
            label = f"Support / {label}"
        lines.append(f'  <text class="tiny" x="76" y="{y}">{escape_xml(label)}</text>')

    for item, x, w, sublane in sorted(placements, key=lambda value: (lane_y[value[0].lane], value[3], value[0].start)):
        start = datetime.fromisoformat(item.start).date()
        end = datetime.fromisoformat(item.end).date()
        y = lane_y[item.lane] + sublane * 50
        node_class, _ = color_classes(item.color)
        lines.append(f'  <rect class="{node_class}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="46"/>')
        if item.open_ended:
            right_x = x + w
            if right_x < x1:
                mid_y = y + 23
                lines.append(f'  <line class="tick" x1="{right_x:.1f}" y1="{mid_y}" x2="{x1:.1f}" y2="{mid_y}" stroke-dasharray="5 5"/>')
                right_x = x1
            lines.append(f'  <line class="tick" x1="{right_x:.1f}" y1="{y}" x2="{right_x:.1f}" y2="{y + 46}" stroke-dasharray="5 5"/>')
        lines.append(f'  <text class="small" x="{x + 16:.1f}" y="{y + 20}">{escape_xml(item.label)}</text>')
        if item.precision == "month":
            label = f"{start.strftime('%Y-%m')} - {end.strftime('%Y-%m')}"
        else:
            label = f"{start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}"
        suffix = " -> active" if item.open_ended else ""
        lines.append(f'  <text class="tiny" x="{x + 16:.1f}" y="{y + 38}">{escape_xml(label + suffix)}</text>')

    # Lineage / transition arrows between lanes.
    arrows = [
        ("B400G", "802.3df", "edgeBlue"),
        ("802.3df", "802.3dj", "edgeBlue"),
        ("802.3dj", "400GPL", "edgeBlue"),
        ("E4AI", "400GPL", "edge"),
        ("200GMMF", "802.3ds", "edge"),
        ("COM", "400GPL", "edgeOrange"),
    ]
    item_by_key = {item.key: item for item in items}
    placement_by_key = {item.key: (x, w, sublane) for item, x, w, sublane in placements}
    for src, dst, edge_class in arrows:
        s = item_by_key[src]
        d = item_by_key[dst]
        sx, sw, ss = placement_by_key[src]
        dx, _dw, ds = placement_by_key[dst]
        end_x = dx
        end_y = lane_y[d.lane] + ds * 50 + 23

        source_end_x = sx + sw
        source_center_y = lane_y[s.lane] + ss * 50 + 23
        if end_x <= source_end_x:
            # Overlapping efforts should point to the target start, not loop back
            # from the source end. Anchor on the source box edge near the
            # target's start date so the arrow still reads left-to-right.
            source_y = lane_y[s.lane] + ss * 50
            start_x = max(sx + 24, min(end_x - 90, sx + sw - 30))
            if end_y >= source_center_y:
                start_y = source_y + 46
            else:
                start_y = source_y
            gap = max(24, end_x - start_x)
            c1 = start_x + gap * 0.55
            c2 = end_x - gap * 0.35
            d_attr = f"M{start_x:.1f} {start_y:.1f} C{c1:.1f} {start_y:.1f} {c2:.1f} {end_y:.1f} {end_x:.1f} {end_y:.1f}"
        else:
            start_x = source_end_x
            start_y = source_center_y
            gap = end_x - start_x
            if gap < 96:
                c1 = start_x + gap * 0.5
                c2 = end_x - gap * 0.5
            else:
                c1 = start_x + max(42, gap * 0.45)
                c2 = end_x - max(42, gap * 0.45)
            d_attr = f"M{start_x:.1f} {start_y:.1f} C{c1:.1f} {start_y:.1f} {c2:.1f} {end_y:.1f} {end_x:.1f} {end_y:.1f}"
        lines.append(f'  <path class="{edge_class}" d="{d_attr}"/>')

    lines.append('  <text class="tiny" x="76" y="1330">Footnote: dates from cached public IEEE 802.3 meeting indexes; sources/ranges in ieee802_3_ai_map/timeline_data.json.</text>')
    lines.append(END_MARKER)
    return "\n".join(lines)


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def update_svg(fragment: str) -> None:
    text = SVG_PATH.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        old_start = text.index('  <!-- Bottom timeline panel -->')
        old_end = text.index('  <text class="tiny" x="76" y="1145">', old_start)
        old_end = text.index("\n</svg>", old_end)
        text = text[:old_start] + fragment + text[old_end:]
    else:
        start = text.index(START_MARKER)
        end = text.index(END_MARKER, start) + len(END_MARKER)
        text = text[:start] + fragment + text[end:]
    SVG_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    items = build_items()
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "note": "Ranges are derived from cached public meeting indexes / local cache manifests. High-level-only adjacent items are marked in notes.",
                "items": [asdict(item) for item in items],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    update_svg(render_timeline(items))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"updated {SVG_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
