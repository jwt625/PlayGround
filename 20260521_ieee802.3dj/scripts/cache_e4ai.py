#!/usr/bin/env python3
"""Cache IEEE 802.3 E4AI public materials and write metadata manifests."""

from __future__ import annotations

import hashlib
import html.parser
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HOME_URL = "https://ieee802.org/3/ad_hoc/E4AI/public/index.html"
PUBLIC_ROOT_URL = "https://ieee802.org/3/ad_hoc/E4AI/public/"
CACHE_DIR = ROOT / "ieee802_e4ai_cache"
METADATA_DIR = ROOT / "ieee802_e4ai_metadata"
CHECKLIST = ROOT / "IEEE-802p3-e4ai-cache-checklist.md"
DOCUMENTS_JSON = METADATA_DIR / "documents.json"
PAGES_JSON = METADATA_DIR / "pages.json"
REQUEST_DELAY_SECONDS = 0.75

DOWNLOAD_EXTENSIONS = {
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".doc",
    ".docx",
    ".zip",
    ".s4p",
    ".s8p",
    ".s12p",
    ".s16p",
    ".s32p",
    ".mat",
    ".csv",
    ".txt",
}


class LinkParser(html.parser.HTMLParser):
    def __init__(self, parent_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.parent_url = parent_url
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        self._href = urllib.parse.urljoin(self.parent_url, href)
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        self.links.append({"href": self._href, "text": clean_inline("".join(self._text))})
        self._href = None
        self._text = []


BR_MARK = "\u241e"


class TableMetadataParser(html.parser.HTMLParser):
    def __init__(self, parent_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.parent_url = parent_url
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._link: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = {"chunks": [], "links": []}
        elif tag == "br" and self._cell is not None:
            self._cell["chunks"].append(BR_MARK)
        elif tag == "a" and self._cell is not None:
            href = dict(attrs).get("href")
            if href:
                self._link = {"href": urllib.parse.urljoin(self.parent_url, href), "text": ""}

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["chunks"].append(data)
        if self._link is not None:
            self._link["text"] += data

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._cell is not None and self._link is not None:
            self._link["text"] = clean_inline(self._link["text"])
            self._cell["links"].append(self._link)
            self._link = None
        elif tag == "td" and self._row is not None and self._cell is not None:
            self._row.append(
                {
                    "text": clean_multiline("".join(self._cell["chunks"])),
                    "links": self._cell["links"],
                }
            )
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell["text"] or cell["links"] for cell in self._row):
                self.rows.append(self._row)
            self._row = None


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    url: str
    kind: str


@dataclass(frozen=True)
class Download:
    doc_id: str
    page_slug: str
    page_title: str
    page_kind: str
    title: str
    presentation_date: str
    presentation_date_iso: str
    presenters: list[str]
    affiliations: list[str]
    presenter_affiliations: list[dict[str, str]]
    url: str
    source_parent_url: str
    source_parent_path: str
    path: str
    status: str
    size: int
    sha256: str
    note: str = ""


def fetch(url: str, *, attempts: int = 3) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; local IEEE 802.3 E4AI cache)"}
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt != attempts:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def polite_pause() -> None:
    time.sleep(REQUEST_DELAY_SECONDS)


def decode_html(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    sample = data[:2000]
    if sample.count(b"\x00") > len(sample) // 8:
        return data.decode("utf-16", errors="replace")
    for encoding in ("utf-8", "windows-1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_inline(value: str) -> str:
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    value = value.replace("\xa0", " ")
    value = value.replace("\u2019", "'")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_multiline(value: str) -> str:
    value = value.replace("\xa0", " ")
    lines = [clean_inline(line) for line in value.split(BR_MARK)]
    return "\n".join(line for line in lines if line)


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


def parse_date_iso(value: str) -> str:
    value = clean_inline(value)
    match = re.search(r"(\d{1,2}|xx)[-\s]+([A-Za-z]+)[-\s]+(\d{2,4})", value)
    if not match or match.group(1).lower() == "xx":
        return ""
    month = MONTHS.get(match.group(2).lower())
    if month is None:
        return ""
    year = int(match.group(3))
    if year < 100:
        year += 2000
    try:
        return datetime(year, month, int(match.group(1))).date().isoformat()
    except ValueError:
        return ""


def split_people(value: str) -> list[str]:
    return [clean_inline(line) for line in value.split("\n") if clean_inline(line)]


def pair_presenters_affiliations(
    presenters: list[str], affiliations: list[str]
) -> list[dict[str, str]]:
    if len(affiliations) == 1 and len(presenters) > 1:
        affiliations = affiliations * len(presenters)
    paired = []
    for index, presenter in enumerate(presenters):
        affiliation = affiliations[index] if index < len(affiliations) else ""
        paired.append({"presenter": presenter, "affiliation": affiliation})
    return paired


def parse_links(html_text: str, parent_url: str) -> list[dict[str, str]]:
    parser = LinkParser(parent_url)
    parser.feed(html_text)
    return parser.links


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def is_download_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return Path(parsed.path).suffix.lower() in DOWNLOAD_EXTENSIONS


def is_cacheable_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "ieee802.org" not in host and host != "mentor.ieee.org":
        return False
    if "/private/" in path:
        return False
    return True


def file_name_for(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        name = hashlib.sha256(url.encode()).hexdigest()[:16]
    return name


def slug_for_url(url: str, fallback: str = "page") -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return fallback
    parts = [part for part in path.split("/") if part and part != "index.html"]
    slug = parts[-1] if parts else fallback
    if slug == "public" and len(parts) > 1:
        slug = parts[-2]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", slug).strip("._") or fallback


def doc_id_for(page_slug: str, url: str) -> str:
    stem = Path(file_name_for(url)).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{page_slug}__{safe}"


def page_sort_key(page: Page) -> tuple[int, int, int, str]:
    match = re.search(r"(\d{2})_(\d{2})(\d{2})?", page.slug)
    if not match:
        return (9999, 99, 99, page.slug)
    return (2000 + int(match.group(1)), int(match.group(2)), int(match.group(3) or "0"), page.slug)


def discover_pages() -> list[Page]:
    data = fetch(HOME_URL)
    polite_pause()
    parent_dir = CACHE_DIR / "_parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "index.html").write_bytes(data)
    html_text = decode_html(data)
    pages: dict[str, Page] = {
        "public_index": Page(
            slug="public_index",
            title="E4AI public index",
            url=HOME_URL,
            kind="index",
        )
    }
    for link in parse_links(html_text, HOME_URL):
        url = link["href"]
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc == "1.ieee802.org" and not is_download_url(url):
            slug = slug_for_url(url, "workshop")
            pages[slug] = Page(slug=slug, title=link["text"] or slug, url=url, kind="workshop")
        elif parsed.path.endswith("/index.html") and (
            "/3/ad_hoc/E4AI/public/" in parsed.path
            or "/3/ad_hoc/ngrates/public/25_01/" in parsed.path
        ):
            slug = slug_for_url(url)
            kind = "workshop" if parsed.netloc == "1.ieee802.org" else "meeting"
            pages[slug] = Page(slug=slug, title=link["text"] or slug, url=url, kind=kind)
        elif parsed.path.endswith("/channel/index.html"):
            pages["channel"] = Page(
                slug="channel",
                title=link["text"] or "E4AI Channel Data",
                url=url,
                kind="channel",
            )
    return sorted(pages.values(), key=page_sort_key)


def cache_file(url: str, path: Path) -> tuple[str, int, str]:
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
        return ("cached", len(data), hashlib.sha256(data).hexdigest())
    data = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return ("downloaded", len(data), hashlib.sha256(data).hexdigest())


def metadata_from_row(
    page: Page,
    local_page_path: Path,
    row: list[dict[str, Any]],
    link_index: int,
    link: dict[str, str],
) -> dict[str, Any]:
    title = clean_inline(link["text"]) or clean_inline(row[link_index]["text"]) or file_name_for(link["href"])
    cells_after = row[link_index + 1 :]
    date_text = ""
    date_iso = ""
    date_index = None
    for index, cell in enumerate(cells_after):
        parsed_date = parse_date_iso(cell["text"])
        if parsed_date or re.search(r"\bxx[-\s]+[A-Za-z]+[-\s]+\d{2,4}\b", cell["text"]):
            date_text = clean_inline(cell["text"])
            date_iso = parsed_date
            date_index = index
            break
    presenters: list[str] = []
    affiliations: list[str] = []
    if date_index is not None:
        tail = [cell["text"] for cell in cells_after[date_index + 1 :] if cell["text"]]
        presenters = split_people(tail[0]) if tail else []
        affiliations = split_people(tail[1]) if len(tail) > 1 else []
    return {
        "title": title,
        "presentation_date": date_text,
        "presentation_date_iso": date_iso,
        "presenters": presenters,
        "affiliations": affiliations,
        "presenter_affiliations": pair_presenters_affiliations(presenters, affiliations),
        "source_parent_url": page.url,
        "source_parent_path": rel(local_page_path),
    }


def discover_downloads_from_page(page: Page, local_page_path: Path, html_text: str) -> list[Download]:
    parser = TableMetadataParser(page.url)
    parser.feed(html_text)
    downloads: list[Download] = []
    seen: set[str] = set()

    for row in parser.rows:
        for cell_index, cell in enumerate(row):
            for link in cell["links"]:
                url = link["href"]
                if not is_download_url(url) or not is_cacheable_url(url) or url in seen:
                    continue
                seen.add(url)
                metadata = metadata_from_row(page, local_page_path, row, cell_index, link)
                downloads.append(
                    Download(
                        doc_id=doc_id_for(page.slug, url),
                        page_slug=page.slug,
                        page_title=page.title,
                        page_kind=page.kind,
                        title=metadata["title"],
                        presentation_date=metadata["presentation_date"],
                        presentation_date_iso=metadata["presentation_date_iso"],
                        presenters=metadata["presenters"],
                        affiliations=metadata["affiliations"],
                        presenter_affiliations=metadata["presenter_affiliations"],
                        url=url,
                        source_parent_url=metadata["source_parent_url"],
                        source_parent_path=metadata["source_parent_path"],
                        path="",
                        status="pending",
                        size=0,
                        sha256="",
                    )
                )

    for link in parse_links(html_text, page.url):
        url = link["href"]
        if not is_download_url(url) or not is_cacheable_url(url) or url in seen:
            continue
        seen.add(url)
        downloads.append(
            Download(
                doc_id=doc_id_for(page.slug, url),
                page_slug=page.slug,
                page_title=page.title,
                page_kind=page.kind,
                title=link["text"] or file_name_for(url),
                presentation_date="",
                presentation_date_iso="",
                presenters=[],
                affiliations=[],
                presenter_affiliations=[],
                url=url,
                source_parent_url=page.url,
                source_parent_path=rel(local_page_path),
                path="",
                status="pending",
                size=0,
                sha256="",
            )
        )
    return downloads


def page_local_path(page: Page) -> Path:
    return CACHE_DIR / page.slug / "index.html"


def discover_and_cache(pages: list[Page]) -> list[Download]:
    downloads: list[Download] = []
    for page in pages:
        page_dir = CACHE_DIR / page.slug
        page_dir.mkdir(parents=True, exist_ok=True)
        local_page_path = page_local_path(page)
        page_data = fetch(page.url)
        polite_pause()
        local_page_path.write_bytes(page_data)
        html_text = decode_html(page_data)
        page_downloads = discover_downloads_from_page(page, local_page_path, html_text)
        for download in page_downloads:
            local_path = page_dir / file_name_for(download.url)
            note = ""
            try:
                status, size, sha256 = cache_file(download.url, local_path)
                if status == "downloaded":
                    polite_pause()
            except Exception as exc:
                status, size, sha256 = (f"error: {exc}", 0, "")
                note = "Download failed; source link preserved in metadata."
            downloads.append(
                Download(
                    **(
                        asdict(download)
                        | {
                            "path": rel(local_path),
                            "status": status,
                            "size": size,
                            "sha256": sha256,
                            "note": note,
                        }
                    )
                )
            )
            print(f"{page.slug}: {status} {local_path.name}", flush=True)
    return downloads


def write_metadata(pages: list[Page], downloads: list[Download]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    page_rows = [
        asdict(page)
        | {
            "source_parent_path": rel(page_local_path(page)) if page_local_path(page).exists() else "",
            "download_count": sum(1 for download in downloads if download.page_slug == page.slug),
        }
        for page in pages
    ]
    document_rows = [asdict(download) for download in downloads]
    PAGES_JSON.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "source_home_url": HOME_URL,
                "cache_root": rel(CACHE_DIR),
                "page_count": len(page_rows),
                "pages": page_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    DOCUMENTS_JSON.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "source_home_url": HOME_URL,
                "cache_root": rel(CACHE_DIR),
                "document_count": len(document_rows),
                "documents": document_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_checklist(pages: list[Page], downloads: list[Download]) -> None:
    by_page: dict[str, list[Download]] = {page.slug: [] for page in pages}
    for download in downloads:
        by_page.setdefault(download.page_slug, []).append(download)

    total_size = sum(download.size for download in downloads)
    cached_count = sum(1 for download in downloads if download.size)
    lines = [
        "# IEEE 802.3 E4AI Presentation Cache Checklist",
        "",
        f"Source parent page: {HOME_URL}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Pages checked: {len(pages)}",
        f"Downloadable links discovered: {len(downloads)}",
        f"Files cached: {cached_count}",
        f"Total cached bytes: {total_size}",
        "",
        "## Parent/Homepage Links Checked",
        "",
        f"- [x] E4AI public index - {HOME_URL}",
        "",
        "## Page-by-Page Checklist",
        "",
    ]
    for page in pages:
        page_downloads = by_page.get(page.slug, [])
        lines.append(f"### {page.slug} - {page.title}")
        lines.append("")
        lines.append(f"- [x] Parent page: {page.url}")
        lines.append(f"- [x] Cached parent HTML: `{rel(page_local_path(page))}`")
        lines.append(f"- [x] Cached downloadable materials: {len(page_downloads)} files")
        lines.append("")
        for download in sorted(page_downloads, key=lambda item: item.path.lower()):
            checkbox = "x" if download.size else " "
            size = f"{download.size:,} bytes" if download.size else "not cached"
            note = f" Note: {download.note}" if download.note else ""
            lines.append(
                f"- [{checkbox}] `{download.path}` ({size}) - "
                f"[{download.title}]({download.url}){note}"
            )
        lines.append("")
    CHECKLIST.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pages = discover_pages()
    if not pages:
        print("No E4AI pages discovered.", file=sys.stderr)
        return 1
    downloads = discover_and_cache(pages)
    write_metadata(pages, downloads)
    write_checklist(pages, downloads)
    print(f"Wrote {CHECKLIST}")
    print(f"Wrote {DOCUMENTS_JSON}")
    print(f"Cached {len(downloads)} files across {len(pages)} pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
