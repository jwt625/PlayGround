#!/usr/bin/env python3
"""Cache IEEE P802.3dj public meeting materials and write a checklist."""

from __future__ import annotations

import hashlib
import html.parser
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOME_URL = "https://www.ieee802.org/3/dj/"
BASE_URL = "https://www.ieee802.org/3/dj/public/index.html"
CACHE_DIR = ROOT / "ieee802_3dj_cache"
CHECKLIST = ROOT / "IEEE-802p3dj-cache-checklist.md"
KNOWN_REPLACEMENTS = {
    "https://www.ieee802.org/3/dj/public/24_09/lusted_3dj_04a_2409.pdf": (
        "https://www.ieee802.org/3/dj/public/24_09/lusted_3dj_04_2409.pdf",
        "Source page link returns 404; corrected by checking adjacent filename.",
    ),
}
KNOWN_MISSING = {
    "https://www.ieee802.org/3/dj/public/24_09/nicholl_3dj_01b_2409.pdf": (
        "Source page link returns 404. Adjacent revisions 01 and 01a exist; no 01b/01c file found."
    ),
}

DOWNLOAD_EXTENSIONS = {
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".doc",
    ".docx",
    ".zip",
}
REQUEST_DELAY_SECONDS = 0.75


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        self._href = attrs_dict.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = " ".join("".join(self._text).split())
        self.links.append((self._href, text))
        self._href = None
        self._text = []


@dataclass(frozen=True)
class Meeting:
    slug: str
    title: str
    url: str


@dataclass(frozen=True)
class Download:
    meeting: Meeting
    title: str
    url: str
    path: Path
    status: str
    size: int
    sha256: str
    note: str = ""


def fetch(url: str, *, attempts: int = 3) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; local IEEE 802.3dj cache)"}
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt != attempts:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def polite_pause() -> None:
    time.sleep(REQUEST_DELAY_SECONDS)


def parse_links(html: bytes) -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    return parser.links


def meeting_sort_key(meeting: Meeting) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d{2})_(\d{2})(\d{2})?", meeting.slug)
    if not match:
        return (0, 0, 0)
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or "0")
    return (year, month, day)


def discover_meetings() -> list[Meeting]:
    html = fetch(BASE_URL)
    polite_pause()
    (CACHE_DIR / "_parent").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "_parent" / "index.html").write_bytes(html)
    meetings: list[Meeting] = []
    for href, title in parse_links(html):
        url = urllib.parse.urljoin(BASE_URL, href)
        parsed = urllib.parse.urlparse(url)
        match = re.search(r"/3/dj/public/(\d{2}_\d{2}(?:\d{2})?)/index\.html$", parsed.path)
        if not match:
            continue
        slug = match.group(1)
        meetings.append(Meeting(slug=slug, title=title or slug, url=url))
    return sorted(meetings, key=meeting_sort_key)


def file_name_for(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        name = hashlib.sha256(url.encode()).hexdigest()[:16]
    return name


def cache_file(url: str, path: Path) -> tuple[str, int, str]:
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
        return ("cached", len(data), hashlib.sha256(data).hexdigest())
    data = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return ("downloaded", len(data), hashlib.sha256(data).hexdigest())


def discover_and_cache(meetings: list[Meeting]) -> list[Download]:
    downloads: list[Download] = []
    for meeting in meetings:
        meeting_dir = CACHE_DIR / meeting.slug
        meeting_dir.mkdir(parents=True, exist_ok=True)
        index_html = fetch(meeting.url)
        polite_pause()
        (meeting_dir / "index.html").write_bytes(index_html)
        seen_urls: set[str] = set()
        for href, title in parse_links(index_html):
            url = urllib.parse.urljoin(meeting.url, href)
            parsed = urllib.parse.urlparse(url)
            ext = Path(parsed.path).suffix.lower()
            if ext not in DOWNLOAD_EXTENSIONS:
                continue
            if "/3/dj/public/" not in parsed.path and "ieee802.org" in parsed.netloc:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            cache_url, note = KNOWN_REPLACEMENTS.get(url, (url, ""))
            local_path = meeting_dir / file_name_for(cache_url)
            try:
                status, size, sha256 = cache_file(cache_url, local_path)
                if status == "downloaded":
                    polite_pause()
            except Exception as exc:  # Keep the manifest useful if one file fails.
                status, size, sha256 = (f"error: {exc}", 0, "")
                note = KNOWN_MISSING.get(url, note)
            downloads.append(
                Download(
                    meeting=meeting,
                    title=title or file_name_for(url),
                    url=cache_url,
                    path=local_path.relative_to(ROOT),
                    status=status,
                    size=size,
                    sha256=sha256,
                    note=note,
                )
            )
            print(f"{meeting.slug}: {status} {local_path.name}", flush=True)
    return downloads


def write_checklist(meetings: list[Meeting], downloads: list[Download]) -> None:
    by_meeting: dict[str, list[Download]] = {meeting.slug: [] for meeting in meetings}
    for download in downloads:
        by_meeting[download.meeting.slug].append(download)

    total_size = sum(download.size for download in downloads)
    cached_count = sum(1 for download in downloads if download.size)
    lines = [
        "# IEEE P802.3dj Presentation Cache Checklist",
        "",
        f"Source parent page: {BASE_URL}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Meetings checked: {len(meetings)}",
        f"Downloadable links discovered: {len(downloads)}",
        f"Files cached: {cached_count}",
        f"Total cached bytes: {total_size}",
        "",
        "## Parent/Homepage Links Checked",
        "",
        f"- [x] IEEE P802.3dj task force home page - {HOME_URL}",
        f"- [x] IEEE P802.3dj public index - {BASE_URL}",
        "",
        "## Month-by-Month Checklist",
        "",
    ]
    for meeting in meetings:
        meeting_downloads = by_meeting[meeting.slug]
        lines.append(f"### {meeting.slug} - {meeting.title}")
        lines.append("")
        lines.append(f"- [x] Parent page: {meeting.url}")
        lines.append(f"- [x] Cached parent HTML: `{CACHE_DIR.relative_to(ROOT)}/{meeting.slug}/index.html`")
        lines.append(f"- [x] Cached downloadable materials: {len(meeting_downloads)} files")
        lines.append("")
        for download in sorted(meeting_downloads, key=lambda item: str(item.path).lower()):
            checkbox = "x" if download.size else " "
            size = f"{download.size:,} bytes" if download.size else "not cached"
            note = f" Note: {download.note}" if download.note else ""
            lines.append(
                f"- [{checkbox}] `{download.path}` ({size}) - [{download.title}]({download.url}){note}"
            )
        lines.append("")

    CHECKLIST.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meetings = discover_meetings()
    if not meetings:
        print("No meeting pages discovered.", file=sys.stderr)
        return 1
    downloads = discover_and_cache(meetings)
    write_checklist(meetings, downloads)
    print(f"Wrote {CHECKLIST}")
    print(f"Cached {len(downloads)} files across {len(meetings)} meetings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
