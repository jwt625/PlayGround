#!/usr/bin/env python3
"""Summarize visible participant affiliations from cached IEEE 802.3 materials."""

from __future__ import annotations

import collections
import html.parser
import json
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ieee802_3_ai_map" / "participant_summary.json"


class RowParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(clean_cell("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def clean_cell(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


ORG_NORMALIZATION = {
    "Futurewei, US Subsidiary of Huawei": "Huawei / Futurewei",
    "Futurewei, U.S. Subsidiary of Huawei": "Huawei / Futurewei",
    "Futurewei, US Affiliate of Huawei": "Huawei / Futurewei",
    "Futurewei, U.S. Affiliate of Huawei": "Huawei / Futurewei",
    "Huawei Technologies": "Huawei",
    "Huawei Technologies Co., Ltd.": "Huawei",
    "Broadcom, Inc.": "Broadcom",
    "Broadcom, Inc": "Broadcom",
    "Broadcom Inc.": "Broadcom",
    "Nvidia": "NVIDIA",
    "Ghiasi Quantum/Marvell": "Marvell / Ghiasi Quantum",
    "Ghiasi Quantum / Marvell": "Marvell / Ghiasi Quantum",
    "Ghiasi Quantum": "Marvell / Ghiasi Quantum",
    "Keysight Technologies": "Keysight",
    "Genuine Optics": "Genuine Optics",
    "TE Connectivity": "TE Connectivity",
    "CommScope": "CommScope",
    "TIA TR-42.11": "TIA TR-42.11",
    "IEC TC86": "IEC TC86",
}

SKIP_ORGS = {
    "",
    "Study Group",
    "Task Force",
    "IEEE 802.3",
}


def normalize_org(value: str) -> str:
    value = clean_cell(value)
    value = value.strip(" ,.;")
    return ORG_NORMALIZATION.get(value, value)


def split_affiliations(value: str) -> list[str]:
    parts: list[str] = []
    for line in value.split("\n"):
        line = clean_cell(line)
        if not line:
            continue
        # Keep "Huawei / Futurewei" and "Marvell / Ghiasi Quantum" intact.
        for chunk in re.split(r"\s{2,}|;\s*", line):
            org = normalize_org(chunk)
            if org and org not in SKIP_ORGS:
                parts.append(org)
    return parts


def summarize_metadata(path: Path, key: str) -> collections.Counter[str]:
    data = json.loads(path.read_text(encoding="utf-8"))[key]
    counter: collections.Counter[str] = collections.Counter()
    for doc in data:
        for org in doc.get("affiliations", []) or []:
            org = normalize_org(org)
            if org not in SKIP_ORGS:
                counter[org] += 1
    return counter


def summarize_cached_html(project_slug: str) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for path in sorted((ROOT / "ieee802_3_ai_related_cache" / project_slug).glob("*/index.html")):
        if path.parent.name in {"_home", "public_index", "presentproc.html"}:
            continue
        parser = RowParser()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        for row in parser.rows:
            if len(row) < 4:
                continue
            headerish = " ".join(row).lower()
            if "affiliation" in headerish or "contributions" in headerish:
                continue
            for org in split_affiliations(row[3]):
                counter[org] += 1
    return counter


def top(counter: collections.Counter[str], n: int = 8) -> list[dict[str, int | str]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(n)]


def main() -> int:
    summaries = {
        "802.3dj": {
            "source": "ieee802_3dj_browser/metadata/talks.json",
            "method": "presentation metadata affiliation counts",
            "top_affiliations": top(
                summarize_metadata(ROOT / "ieee802_3dj_browser/metadata/talks.json", "talks")
            ),
        },
        "E4AI": {
            "source": "ieee802_e4ai_metadata/documents.json",
            "method": "presentation metadata affiliation counts",
            "top_affiliations": top(
                summarize_metadata(ROOT / "ieee802_e4ai_metadata/documents.json", "documents")
            ),
        },
        "200GMMF": {
            "source": "ieee802_3_ai_related_cache/200GMMF/*/index.html",
            "method": "meeting table affiliation-column counts",
            "top_affiliations": top(summarize_cached_html("200GMMF")),
        },
        "802.3ds": {
            "source": "ieee802_3_ai_related_cache/ds/*/index.html",
            "method": "meeting table affiliation-column counts",
            "top_affiliations": top(summarize_cached_html("ds")),
        },
        "400GPL": {
            "source": "ieee802_3_ai_related_cache/400GPL/*/index.html",
            "method": "meeting table affiliation-column counts",
            "top_affiliations": top(summarize_cached_html("400GPL")),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "note": "Counts are visible public presentation/meeting-page affiliation signals, not formal sponsor or membership counts.",
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
