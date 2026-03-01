#!/usr/bin/env python3
"""Repair top-100 external reference metadata using DOI/arXiv authoritative metadata.

Updates:
- tests/inference_test/output/top_100_external_refs_R2.md
- tests/inference_test/output/dedup_stats.json (top_100_external only)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / "tests/inference_test/output"
TOP100_MD = OUTPUT_DIR / "top_100_external_refs_R2.md"
DEDUP_STATS = OUTPUT_DIR / "dedup_stats.json"

DOI_ACCEPT = "application/vnd.citationstyles.csl+json"
UA = "LangExtractTop100Fixer/1.0"


@dataclass
class Entry:
    idx: int
    citations: int
    title: str
    link: str
    year: Optional[int]
    authors: list[str]
    raw_head: str


@dataclass
class Meta:
    title: Optional[str]
    year: Optional[int]
    authors: list[str]


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def split_authors(s: str) -> list[str]:
    if not s:
        return []
    return [a.strip() for a in s.split(",") if a.strip()]


def parse_md(path: Path) -> tuple[list[str], list[Entry]]:
    lines = path.read_text().splitlines()
    entries: list[Entry] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\d+)\. \*\*\[(\d+) citations\]\*\* \[(.*)\]\((https?://[^)]+)\) \((\d{4})\)\s*$", line)
        if not m:
            i += 1
            continue

        idx = int(m.group(1))
        citations = int(m.group(2))
        title = m.group(3).strip()
        link = m.group(4).strip()
        year = int(m.group(5))
        raw_head = line

        authors: list[str] = []
        if i + 1 < len(lines):
            m2 = re.match(r"^\s*-\s*(.*)$", lines[i + 1])
            if m2:
                authors = split_authors(m2.group(1).strip())
                i += 1

        entries.append(Entry(idx, citations, title, link, year, authors, raw_head))
        i += 1

    return lines, entries


def extract_doi(link: str) -> Optional[str]:
    m = re.search(r"doi\.org/(.+)$", link, flags=re.I)
    if not m:
        return None
    doi = urllib.parse.unquote(m.group(1)).strip()
    doi = doi.rstrip(".")
    return doi.lower() if doi else None


def extract_arxiv(link: str) -> Optional[str]:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", link, flags=re.I)
    if not m:
        return None
    aid = m.group(1)
    aid = aid.removesuffix(".pdf")
    aid = re.sub(r"v\d+$", "", aid, flags=re.I)
    return aid.lower()


def fetch_doi_meta(doi: str) -> Optional[Meta]:
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/:;()-._")
    req = urllib.request.Request(url, headers={"Accept": DOI_ACCEPT, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None

    title = norm_space(data.get("title") or "") or None

    year = None
    for k in ("issued", "published-print", "published-online", "created"):
        d = data.get(k, {})
        if isinstance(d, dict):
            parts = d.get("date-parts")
            if parts and parts[0] and parts[0][0]:
                year = int(parts[0][0])
                break

    authors = []
    for a in data.get("author", []):
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        full = norm_space(f"{given} {family}")
        if full:
            authors.append(full)

    return Meta(title=title, year=year, authors=authors)


def fetch_arxiv_meta(arxiv_id: str) -> Optional[Meta]:
    query_url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    req = urllib.request.Request(query_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            xml_text = r.read().decode("utf-8", "replace")
    except Exception:
        return None

    try:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title = norm_space(entry.findtext("atom:title", default="", namespaces=ns)) or None
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            norm_space(a.findtext("atom:name", default="", namespaces=ns))
            for a in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]
        return Meta(title=title, year=year, authors=authors)
    except Exception:
        return None


def format_authors(authors: list[str], max_names: int = 8) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= max_names:
        return ", ".join(authors)
    return ", ".join(authors[:max_names]) + ", et al."


def write_corrected_md(lines: list[str], entries: list[Entry], updates: dict[int, Meta]) -> None:
    new_lines: list[str] = []
    i = 0
    entry_map = {e.idx: e for e in entries}

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\d+)\. \*\*\[(\d+) citations\]\*\* \[(.*)\]\((https?://[^)]+)\) \((\d{4})\)\s*$", line)
        if not m:
            new_lines.append(line)
            i += 1
            continue

        idx = int(m.group(1))
        if idx in updates:
            meta = updates[idx]
            e = entry_map[idx]
            title = meta.title or e.title
            year = meta.year or e.year or 0
            head = f"{idx}. **[{e.citations} citations]** [{title}]({e.link}) ({year})"
            new_lines.append(head)

            if i + 1 < len(lines) and re.match(r"^\s*-\s*", lines[i + 1]):
                i += 1
            authors = meta.authors if meta.authors else e.authors
            new_lines.append(f"   - {format_authors(authors)}")
        else:
            new_lines.append(line)

        i += 1

    backup = TOP100_MD.with_suffix(".md.bak")
    if not backup.exists():
        backup.write_text("\n".join(lines) + "\n")
    TOP100_MD.write_text("\n".join(new_lines) + "\n")


def patch_dedup_stats(updates_by_id: dict[tuple[Optional[str], Optional[str]], Meta]) -> int:
    data = json.loads(DEDUP_STATS.read_text())
    top = data.get("top_100_external", [])
    changed = 0
    for r in top:
        doi = (r.get("doi") or "").strip().lower() or None
        arxiv = (r.get("arxiv_id") or "").strip().lower() or None
        key = (doi, arxiv)
        meta = updates_by_id.get(key)
        if not meta:
            continue
        if meta.title:
            r["title"] = meta.title
        if meta.year:
            r["year"] = meta.year
        if meta.authors:
            r["authors"] = meta.authors
        changed += 1

    DEDUP_STATS.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def main() -> int:
    if not TOP100_MD.exists():
        print(f"Missing: {TOP100_MD}", file=sys.stderr)
        return 1
    if not DEDUP_STATS.exists():
        print(f"Missing: {DEDUP_STATS}", file=sys.stderr)
        return 1

    lines, entries = parse_md(TOP100_MD)
    doi_cache: dict[str, Optional[Meta]] = {}
    arxiv_cache: dict[str, Optional[Meta]] = {}

    updates: dict[int, Meta] = {}
    updates_by_id: dict[tuple[Optional[str], Optional[str]], Meta] = {}

    for e in entries:
        doi = extract_doi(e.link)
        arxiv = extract_arxiv(e.link)

        meta: Optional[Meta] = None
        if doi:
            if doi not in doi_cache:
                doi_cache[doi] = fetch_doi_meta(doi)
            meta = doi_cache[doi]
        elif arxiv:
            if arxiv not in arxiv_cache:
                arxiv_cache[arxiv] = fetch_arxiv_meta(arxiv)
            meta = arxiv_cache[arxiv]

        if meta and (meta.title or meta.year or meta.authors):
            updates[e.idx] = meta
            updates_by_id[(doi, arxiv)] = meta

    write_corrected_md(lines, entries, updates)
    changed_stats = patch_dedup_stats(updates_by_id)

    print(f"Parsed entries: {len(entries)}")
    print(f"Metadata updates applied to markdown: {len(updates)}")
    print(f"Metadata updates applied to dedup_stats top_100_external: {changed_stats}")
    print(f"Wrote: {TOP100_MD}")
    print(f"Wrote: {DEDUP_STATS}")
    print(f"Backup: {TOP100_MD.with_suffix('.md.bak')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
