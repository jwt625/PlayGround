#!/usr/bin/env python3
"""
Light-touch probe for OFC schedule paper links.

This script:
1. Fetches the schedule API once.
2. Summarizes presentations for a selected day.
3. Finds paper URLs from both the structured field and embedded description HTML.
4. Optionally tests a small sample of paper URLs without cookies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import unescape
from http.cookiejar import CookieJar
from typing import Any


SCHEDULE_URL = "https://www.ofcconference.org/api/schedule/"
UA = "Mozilla/5.0 (compatible; ofc-schedule-probe/1.0)"
PDF_TEXT_RE = re.compile(
    r"Access the Technical Paper.*?<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>\s*Download PDF\s*</a>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class PresentationRecord:
    day: str
    session_id: int
    session_title: str
    presentation_id: int
    code: str
    title: str
    paper_is_ready: bool
    paper_download_link: str
    embedded_pdf_link: str


def build_opener() -> urllib.request.OpenerDirector:
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", UA)]
    return opener


def fetch_json(opener: urllib.request.OpenerDirector, url: str) -> dict[str, Any]:
    with opener.open(url, timeout=30) as response:
        return json.load(response)


def find_embedded_pdf_link(description: str | None) -> str:
    if not description:
        return ""
    match = PDF_TEXT_RE.search(description)
    if not match:
        return ""
    return unescape(match.group(1))


def collect_presentations(schedule: dict[str, Any]) -> list[PresentationRecord]:
    rows: list[PresentationRecord] = []
    for day in schedule.get("selectedDays", []):
        day_name = day.get("dayOfWeek", "")
        for time_block in day.get("timeBlocks", []):
            for session in time_block.get("sessions", []):
                for presentation in session.get("presentations", []) or []:
                    rows.append(
                        PresentationRecord(
                            day=day_name,
                            session_id=session.get("id"),
                            session_title=session.get("title", ""),
                            presentation_id=presentation.get("id"),
                            code=presentation.get("code", ""),
                            title=presentation.get("title", ""),
                            paper_is_ready=bool(presentation.get("paperIsReady")),
                            paper_download_link=presentation.get("paperDownloadLink", "") or "",
                            embedded_pdf_link=find_embedded_pdf_link(presentation.get("description")),
                        )
                    )
    return rows


def probe_url(opener: urllib.request.OpenerDirector, url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with opener.open(request, timeout=30) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            return {
                "ok": True,
                "status": getattr(response, "status", 200),
                "final_url": final_url,
                "content_type": content_type,
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "final_url": exc.geturl(),
            "content_type": exc.headers.get("Content-Type", ""),
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "status": None,
            "final_url": url,
            "content_type": "",
            "error": repr(exc),
        }


def summarize(day: str, rows: list[PresentationRecord]) -> None:
    day_rows = [row for row in rows if row.day.lower() == day.lower()]
    field_links = sum(1 for row in day_rows if row.paper_download_link)
    embedded_links = sum(1 for row in day_rows if row.embedded_pdf_link)
    ready_with_missing_field = sum(
        1 for row in day_rows if row.paper_is_ready and not row.paper_download_link and row.embedded_pdf_link
    )

    print(f"Day: {day}")
    print(f"Presentations: {len(day_rows)}")
    print(f"Structured paperDownloadLink present: {field_links}")
    print(f"Embedded Download PDF link in description: {embedded_links}")
    print(f"paperIsReady=true but paperDownloadLink missing and embedded link present: {ready_with_missing_field}")


def print_samples(day: str, rows: list[PresentationRecord], limit: int) -> list[PresentationRecord]:
    samples = [row for row in rows if row.day.lower() == day.lower() and row.embedded_pdf_link][:limit]
    for idx, row in enumerate(samples, start=1):
        print("")
        print(f"[sample {idx}] {row.code} {row.title}")
        print(f"session: {row.session_id} {row.session_title}")
        print(f"paperIsReady: {row.paper_is_ready}")
        print(f"paperDownloadLink: {row.paper_download_link or '<empty>'}")
        print(f"embeddedPdfLink: {row.embedded_pdf_link}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default="Monday")
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--probe-links", action="store_true")
    args = parser.parse_args()

    opener = build_opener()
    schedule = fetch_json(opener, SCHEDULE_URL)
    rows = collect_presentations(schedule)

    summarize(args.day, rows)
    samples = print_samples(args.day, rows, args.sample_limit)

    if args.probe_links:
        for idx, row in enumerate(samples, start=1):
            result = probe_url(opener, row.embedded_pdf_link)
            print("")
            print(f"[probe {idx}] {row.code}")
            print(f"status: {result.get('status')}")
            print(f"final_url: {result.get('final_url')}")
            print(f"content_type: {result.get('content_type')}")
            if result.get("error"):
                print(f"error: {result['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
