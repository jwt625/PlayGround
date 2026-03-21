#!/usr/bin/env python3
"""
Export OFC schedule metadata into JSON and CSV.

Modes:
- metadata: API-only export with no browser usage
- paper-status: protected paper-link resolution without downloading PDF bytes
- paper-cache: protected paper-link resolution plus PDF caching
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


SCHEDULE_URL = "https://www.ofcconference.org/api/schedule/"
UA = "Mozilla/5.0 (compatible; ofc-metadata-export/2.0)"
DEFAULT_OUTDIR = Path("output")
DEFAULT_CACHE_DIR = Path(".cache/ofc")
DEFAULT_FIREFOX_EXECUTABLE = Path("/Applications/Firefox.app/Contents/MacOS/firefox")
DEFAULT_CHROME_EXECUTABLE = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

PDF_LINK_RE = re.compile(
    r"Access the Technical Paper.*?<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>\s*Download PDF\s*</a>",
    re.IGNORECASE | re.DOTALL,
)
AUTHORS_RE = re.compile(
    r"<strong>\s*Authors\s*</strong>\s*:\s*(.*?)(?:<br\s*/?>\s*<br\s*/?>|<strong>\s*Access the Technical Paper\s*</strong>|$)",
    re.IGNORECASE | re.DOTALL,
)
ACCESS_SECTION_RE = re.compile(
    r"<br\s*/?>\s*<br\s*/?>\s*<strong>\s*Access the Technical Paper\s*</strong>.*$",
    re.IGNORECASE | re.DOTALL,
)
AUTHORS_SECTION_RE = re.compile(
    r"<br\s*/?>\s*<br\s*/?>\s*<strong>\s*Authors\s*</strong>\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)


class HtmlToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = unescape(text)
        text = re.sub(r"\r", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()


def html_to_text(value: str | None) -> str:
    if not value:
        return ""
    parser = HtmlToTextParser()
    parser.feed(value)
    return parser.get_text()


def strip_tags(value: str | None) -> str:
    return html_to_text(value)


def fetch_schedule() -> dict[str, Any]:
    request = urllib.request.Request(SCHEDULE_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_embedded_pdf_link(description_html: str | None) -> str:
    if not description_html:
        return ""
    match = PDF_LINK_RE.search(description_html)
    return unescape(match.group(1)) if match else ""


def parse_authors_text(description_html: str | None) -> str:
    if not description_html:
        return ""
    match = AUTHORS_RE.search(description_html)
    if not match:
        return ""
    return strip_tags(match.group(1))


def parse_abstract_text(description_html: str | None) -> str:
    if not description_html:
        return ""
    value = ACCESS_SECTION_RE.sub("", description_html)
    value = AUTHORS_SECTION_RE.sub("", value)
    return strip_tags(value)


def parse_author_entries(authors_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if not authors_text:
        return entries

    for part in [item.strip() for item in authors_text.split(" / ") if item.strip()]:
        pieces = [item.strip() for item in part.split(",")]
        if not pieces:
            continue
        entries.append(
            {
                "name": pieces[0],
                "affiliation": ", ".join(pieces[1:]).strip(),
                "raw": part,
            }
        )
    return entries


def normalize_tags(tags: list[str] | None) -> str:
    return " | ".join(tags or [])


def classify_record(session: dict[str, Any], presentation: dict[str, Any] | None) -> str:
    if presentation is not None:
        return "presentation"
    lowered = {tag.lower() for tag in session.get("tags", [])}
    if "short course" in lowered:
        return "short-course"
    if "special event" in lowered:
        return "special-session"
    return "session"


def derive_link_source(paper_download_link: str, embedded_pdf_link: str) -> str:
    if paper_download_link:
        return "structured"
    if embedded_pdf_link:
        return "description"
    return "none"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "item"


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def parse_cookie_export(path: Path) -> list[dict[str, Any]]:
    cookies: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name = parts[0].strip()
        value = parts[1].strip() if len(parts) > 1 else ""
        domain = parts[2].strip() if len(parts) > 2 else ""
        path_value = parts[3].strip() if len(parts) > 3 else "/"
        same_site = parts[8].strip() if len(parts) > 8 else ""
        secure = any(part.strip() == "✓" for part in parts[5:7])
        if not name or not domain:
            continue
        cookie: dict[str, Any] = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": path_value or "/",
            "secure": secure,
        }
        cookies.append(cookie)
    return cookies


def save_pdf_with_cookies(url: str, cookies: list[dict[str, Any]], destination: Path) -> tuple[bool, str, str]:
    import http.cookiejar

    jar = http.cookiejar.CookieJar()
    for item in cookies:
        cookie = http.cookiejar.Cookie(
            version=0,
            name=item["name"],
            value=item["value"],
            port=None,
            port_specified=False,
            domain=item["domain"],
            domain_specified=True,
            domain_initial_dot=item["domain"].startswith("."),
            path=item.get("path", "/"),
            path_specified=True,
            secure=bool(item.get("secure", False)),
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )
        jar.set_cookie(cookie)

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", UA)]
    with opener.open(urllib.request.Request(url), timeout=30) as response:
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    ensure_parent(destination)
    destination.write_bytes(data)
    is_pdf = destination.suffix.lower() == ".pdf" or content_type.lower().startswith("application/pdf")
    return is_pdf, final_url, content_type


@dataclass
class ExportRecord:
    record_kind: str
    day: str
    day_iso: str
    timeblock_id: str
    session_id: int
    session_code: str
    session_title: str
    session_location: str
    session_track: str
    session_tags: str
    session_start: str
    session_end: str
    presentation_id: int | None
    presentation_code: str
    presentation_title: str
    presenter_name: str
    presenter_affiliation: str
    presenter_country: str
    authors_text: str
    authors_json: str
    abstract_text: str
    description_text: str
    description_html: str
    extra_people_json: str
    paper_is_ready: bool
    paper_link_source: str
    paper_download_link: str
    embedded_pdf_link: str
    best_pdf_link: str
    paper_link_tested: bool
    paper_link_status: str
    paper_link_http_status: int | None
    paper_link_final_url: str
    paper_link_content_type: str
    paper_link_error: str
    paper_cached: bool
    paper_cache_path: str
    source: str


def build_record(
    *,
    day: dict[str, Any],
    timeblock: dict[str, Any],
    session: dict[str, Any],
    presentation: dict[str, Any] | None,
) -> ExportRecord:
    extra_people_json = "[]"
    if presentation is None:
        special = session.get("specialEventDetails") or {}
        description_html = special.get("description", "") or ""
        authors_text = ""
        authors_json = "[]"
        abstract_text = strip_tags(description_html)
        description_text = abstract_text
        presenter_name = ""
        presenter_affiliation = ""
        presenter_country = ""
        presentation_id = None
        presentation_code = ""
        presentation_title = ""
        paper_is_ready = False
        paper_download_link = ""
        embedded_pdf_link = ""
        person_groups = special.get("personGroups") or []
        flattened_people: list[dict[str, str]] = []
        for group in person_groups:
            group_title = group.get("title", "") or ""
            for person in group.get("people", []) or []:
                flattened_people.append(
                    {
                        "group_title": group_title,
                        "name": person.get("name", "") or "",
                        "affiliation": person.get("affiliation", "") or "",
                        "country": person.get("country", "") or "",
                        "presentation_title": person.get("presentationTitle", "") or "",
                    }
                )
        extra_people_json = json.dumps(flattened_people, ensure_ascii=False)
    else:
        description_html = presentation.get("description", "") or ""
        authors_text = parse_authors_text(description_html)
        authors_json = json.dumps(parse_author_entries(authors_text), ensure_ascii=False)
        abstract_text = parse_abstract_text(description_html)
        description_text = strip_tags(description_html)
        presenter = presentation.get("presenter") or {}
        presenter_name = presenter.get("name", "") or ""
        presenter_affiliation = presenter.get("affiliation", "") or ""
        presenter_country = presenter.get("country", "") or ""
        presentation_id = presentation.get("id")
        presentation_code = presentation.get("code", "") or ""
        presentation_title = presentation.get("title", "") or ""
        paper_is_ready = bool(presentation.get("paperIsReady"))
        paper_download_link = presentation.get("paperDownloadLink", "") or ""
        embedded_pdf_link = parse_embedded_pdf_link(description_html)

    best_pdf_link = paper_download_link or embedded_pdf_link
    paper_link_source = derive_link_source(paper_download_link, embedded_pdf_link)

    return ExportRecord(
        record_kind=classify_record(session, presentation),
        day=day.get("dayOfWeek", "") or "",
        day_iso=day.get("selectedDay", "") or "",
        timeblock_id=timeblock.get("id", "") or "",
        session_id=session.get("id"),
        session_code=session.get("code", "") or "",
        session_title=session.get("title", "") or "",
        session_location=session.get("location", "") or "",
        session_track=session.get("track", "") or "",
        session_tags=normalize_tags(session.get("tags")),
        session_start=session.get("startDateTimeOffset", "") or "",
        session_end=session.get("endDateTimeOffset", "") or "",
        presentation_id=presentation_id,
        presentation_code=presentation_code,
        presentation_title=presentation_title,
        presenter_name=presenter_name,
        presenter_affiliation=presenter_affiliation,
        presenter_country=presenter_country,
        authors_text=authors_text,
        authors_json=authors_json,
        abstract_text=abstract_text,
        description_text=description_text,
        description_html=description_html,
        extra_people_json=extra_people_json,
        paper_is_ready=paper_is_ready,
        paper_link_source=paper_link_source,
        paper_download_link=paper_download_link,
        embedded_pdf_link=embedded_pdf_link,
        best_pdf_link=best_pdf_link,
        paper_link_tested=False,
        paper_link_status="not_tested" if best_pdf_link else "not_available",
        paper_link_http_status=None,
        paper_link_final_url="",
        paper_link_content_type="",
        paper_link_error="",
        paper_cached=False,
        paper_cache_path="",
        source="https://www.ofcconference.org/schedule/",
    )


def collect_records(schedule: dict[str, Any]) -> list[ExportRecord]:
    records: list[ExportRecord] = []
    for day in schedule.get("selectedDays", []):
        for timeblock in day.get("timeBlocks", []):
            for session in timeblock.get("sessions", []):
                presentations = session.get("presentations") or []
                if presentations:
                    for presentation in presentations:
                        records.append(
                            build_record(
                                day=day,
                                timeblock=timeblock,
                                session=session,
                                presentation=presentation,
                            )
                        )
                else:
                    records.append(
                        build_record(
                            day=day,
                            timeblock=timeblock,
                            session=session,
                            presentation=None,
                        )
                    )
    return records


def load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(cache_path)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key_for_record(record: ExportRecord) -> str:
    identity = f"{record.session_id}:{record.presentation_id}:{record.best_pdf_link}"
    return sha1_text(identity)


def apply_cached_status(records: list[ExportRecord], cache: dict[str, Any]) -> None:
    for record in records:
        if not record.best_pdf_link:
            continue
        cached = cache.get(cache_key_for_record(record))
        if not cached:
            continue
        record.paper_link_tested = bool(cached.get("paper_link_tested", False))
        record.paper_link_status = cached.get("paper_link_status", record.paper_link_status)
        record.paper_link_http_status = cached.get("paper_link_http_status")
        record.paper_link_final_url = cached.get("paper_link_final_url", "")
        record.paper_link_content_type = cached.get("paper_link_content_type", "")
        record.paper_link_error = cached.get("paper_link_error", "")
        record.paper_cached = bool(cached.get("paper_cached", False))
        record.paper_cache_path = cached.get("paper_cache_path", "")


def selected_records(records: list[ExportRecord], *, batch_limit: int, retry_tested: bool) -> list[ExportRecord]:
    candidates = [
        record
        for record in records
        if record.best_pdf_link and (retry_tested or not record.paper_link_tested)
    ]
    candidates.sort(
        key=lambda record: (
            record.day,
            record.session_start,
            record.presentation_code or "",
            record.presentation_id or 0,
        )
    )
    return candidates[:batch_limit]


@contextmanager
def file_lock(lock_path: Path):
    ensure_parent(lock_path)
    if lock_path.exists():
        raise RuntimeError(f"Lock already exists: {lock_path}")
    lock_path.write_text(str(time.time()), encoding="utf-8")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def status_from_probe_result(final_url: str, content_type: str, page_text: str, file_path: str) -> str:
    lowered_url = final_url.lower()
    lowered_text = page_text.lower()
    lowered_content = content_type.lower()
    if file_path:
        return "cached_pdf"
    if "application/pdf" in lowered_content:
        return "pdf_response"
    if "captcha" in lowered_text or "perfdrive" in lowered_url or "validate.perfdrive.com" in lowered_url:
        return "blocked_by_bot"
    if "accountlogin" in lowered_url or "login" in lowered_text or "sign in" in lowered_text:
        return "login_required"
    if "upcoming_conference_pdf" in lowered_url:
        return "pdf_landing"
    return "html_response"


def clone_profile_if_requested(profile_template: Path | None, run_dir: Path) -> Path:
    if profile_template is None:
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(profile_template, run_dir)
    return run_dir


def validate_papers_with_firefox(
    records: list[ExportRecord],
    *,
    sample_limit: int,
    retry_tested: bool,
    firefox_executable: Path,
    cache_dir: Path,
    cache_path: Path,
    firefox_profile_template: Path | None,
    headless: bool,
    wait_ms: int,
) -> None:
    targets = selected_records(records, batch_limit=sample_limit, retry_tested=retry_tested)
    if not targets:
        return

    cache = load_cache(cache_path)
    browser_profile_dir = cache_dir / "firefox-profile-runtime"
    download_dir = cache_dir / "pdfs"
    download_dir.mkdir(parents=True, exist_ok=True)

    clone_profile_if_requested(firefox_profile_template, browser_profile_dir)
    if firefox_profile_template is None:
        browser_profile_dir.mkdir(parents=True, exist_ok=True)

    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    opts = Options()
    opts.binary_location = str(firefox_executable)
    if headless:
        opts.add_argument("-headless")
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", str(download_dir))
    opts.set_preference("browser.download.useDownloadDir", True)
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf,application/octet-stream")
    opts.set_preference("pdfjs.disabled", True)
    opts.set_preference("browser.download.manager.showWhenStarting", False)
    opts.set_preference("browser.download.always_ask_before_handling_new_types", False)
    opts.add_argument("-profile")
    opts.add_argument(str(browser_profile_dir))

    driver = webdriver.Firefox(options=opts)
    driver.set_page_load_timeout(max(5, wait_ms // 1000))

    try:
        for record in targets:
            final_url = ""
            content_type = ""
            page_text = ""
            file_path = ""
            http_status: int | None = None
            error = ""
            status = "not_tested"
            before_files = {path.name for path in download_dir.glob("*")}

            try:
                driver.get(record.best_pdf_link)
                time.sleep(min(4.0, wait_ms / 1000))
                final_url = driver.current_url
                page_text = (driver.page_source or "")[:50000]
                after_files = [path for path in download_dir.glob("*") if path.name not in before_files]
                pdf_candidates = [path for path in after_files if path.is_file() and path.suffix.lower() == ".pdf"]
                if pdf_candidates:
                    pdf_candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
                    file_path = str(pdf_candidates[0])
                    content_type = "application/pdf"
                status = status_from_probe_result(final_url, content_type, page_text, file_path)
            except Exception as exc:  # pragma: no cover
                error = repr(exc)
                status = "browser_error"

            record.paper_link_tested = True
            record.paper_link_status = status
            record.paper_link_http_status = http_status
            record.paper_link_final_url = final_url
            record.paper_link_content_type = content_type
            record.paper_link_error = error
            record.paper_cached = bool(file_path)
            record.paper_cache_path = file_path

            cache[cache_key_for_record(record)] = {
                "paper_link_tested": record.paper_link_tested,
                "paper_link_status": record.paper_link_status,
                "paper_link_http_status": record.paper_link_http_status,
                "paper_link_final_url": record.paper_link_final_url,
                "paper_link_content_type": record.paper_link_content_type,
                "paper_link_error": record.paper_link_error,
                "paper_cached": record.paper_cached,
                "paper_cache_path": record.paper_cache_path,
                "updated_at": int(time.time()),
            }
            save_cache(cache_path, cache)
    finally:
        driver.quit()


def validate_papers_with_chrome_cookies(
    records: list[ExportRecord],
    *,
    sample_limit: int,
    retry_tested: bool,
    download_pdfs: bool,
    chrome_executable: Path,
    chrome_profile_template: Path | None,
    cookie_export_path: Path,
    cache_dir: Path,
    cache_path: Path,
    headless: bool,
    wait_ms: int,
) -> None:
    targets = selected_records(records, batch_limit=sample_limit, retry_tested=retry_tested)
    if not targets:
        return

    cookies = parse_cookie_export(cookie_export_path)
    cache = load_cache(cache_path)
    user_data_root = cache_dir / "chrome-user-data"
    profile_dir = user_data_root / "Default"
    download_dir = cache_dir / "pdfs"
    download_dir.mkdir(parents=True, exist_ok=True)

    clone_profile_if_requested(chrome_profile_template, profile_dir)
    if chrome_profile_template is None:
        profile_dir.mkdir(parents=True, exist_ok=True)

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.binary_location = str(chrome_executable)
    opts.add_argument(f"--user-data-dir={user_data_root}")
    opts.add_argument("--profile-directory=Default")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if headless:
        opts.add_argument("--headless=new")
    opts.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
        },
    )

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(max(5, wait_ms // 1000))

    try:
        driver.execute_cdp_cmd("Network.enable", {})
        for cookie in cookies:
            payload = {
                "name": cookie["name"],
                "value": cookie["value"],
                "domain": cookie["domain"],
                "path": cookie.get("path", "/"),
                "secure": bool(cookie.get("secure", False)),
            }
            if payload["domain"].startswith("."):
                payload["domain"] = payload["domain"][1:]
            try:
                driver.execute_cdp_cmd("Network.setCookie", payload)
            except Exception:
                pass

        for record in targets:
            final_url = ""
            content_type = ""
            page_text = ""
            file_path = ""
            http_status: int | None = None
            error = ""
            status = "not_tested"

            try:
                driver.get(record.best_pdf_link)
                time.sleep(min(5.0, wait_ms / 1000))
                final_url = driver.current_url
                page_text = (driver.page_source or "")[:50000]
                status = status_from_probe_result(final_url, content_type, page_text, file_path)

                if "directpdfaccess/" in final_url:
                    status = "authenticated_pdf_url"
                if download_pdfs and "directpdfaccess/" in final_url:
                    filename = Path(urllib.parse.urlparse(final_url).path).name or f"{cache_key_for_record(record)}.pdf"
                    destination = download_dir / safe_slug(f"{record.presentation_code or record.session_id}-{filename}")
                    is_pdf, fetched_final_url, fetched_content_type = save_pdf_with_cookies(final_url, cookies, destination)
                    final_url = fetched_final_url
                    content_type = fetched_content_type
                    if is_pdf and destination.exists():
                        file_path = str(destination)
                        status = "cached_pdf"
            except Exception as exc:  # pragma: no cover
                error = repr(exc)
                status = "browser_error"

            record.paper_link_tested = True
            record.paper_link_status = status
            record.paper_link_http_status = http_status
            record.paper_link_final_url = final_url
            record.paper_link_content_type = content_type
            record.paper_link_error = error
            record.paper_cached = bool(file_path)
            record.paper_cache_path = file_path

            cache[cache_key_for_record(record)] = {
                "paper_link_tested": record.paper_link_tested,
                "paper_link_status": record.paper_link_status,
                "paper_link_http_status": record.paper_link_http_status,
                "paper_link_final_url": record.paper_link_final_url,
                "paper_link_content_type": record.paper_link_content_type,
                "paper_link_error": record.paper_link_error,
                "paper_cached": record.paper_cached,
                "paper_cache_path": record.paper_cache_path,
                "updated_at": int(time.time()),
            }
            save_cache(cache_path, cache)
    finally:
        driver.quit()


def write_json(records: list[ExportRecord], path: Path) -> None:
    payload = [asdict(record) for record in records]
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(records: list[ExportRecord], path: Path) -> None:
    rows = [asdict(record) for record in records]
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_batch_log(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(records: list[ExportRecord]) -> dict[str, int]:
    summary: dict[str, int] = {
        "records": len(records),
        "presentations": 0,
        "sessions_without_presentations": 0,
        "paper_ready": 0,
        "paper_links_structured": 0,
        "paper_links_description": 0,
        "paper_links_available": 0,
        "paper_links_tested": 0,
        "paper_cached": 0,
    }
    for record in records:
        if record.record_kind == "presentation":
            summary["presentations"] += 1
        else:
            summary["sessions_without_presentations"] += 1
        if record.paper_is_ready:
            summary["paper_ready"] += 1
        if record.paper_link_source == "structured":
            summary["paper_links_structured"] += 1
        if record.paper_link_source == "description":
            summary["paper_links_description"] += 1
        if record.best_pdf_link:
            summary["paper_links_available"] += 1
        if record.paper_link_tested:
            summary["paper_links_tested"] += 1
        if record.paper_cached:
            summary["paper_cached"] += 1
        key = f"status_{record.paper_link_status}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["metadata", "paper-status", "paper-cache"],
        default="metadata",
        help="metadata is API-only; paper-status and paper-cache touch protected Optica links.",
    )
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--basename", default="ofc_schedule_metadata")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--firefox-executable", default=str(DEFAULT_FIREFOX_EXECUTABLE))
    parser.add_argument("--chrome-executable", default=str(DEFAULT_CHROME_EXECUTABLE))
    parser.add_argument("--chrome-profile-template", default="")
    parser.add_argument("--cookie-export-path", default="")
    parser.add_argument("--firefox-profile-template", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-pdf-download", action="store_true")
    parser.add_argument("--wait-ms", type=int, default=15000)
    parser.add_argument("--retry-tested", action="store_true")
    parser.add_argument("--lock-path", default="")
    parser.add_argument("--batch-log-path", default="")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cache_dir = Path(args.cache_dir)
    cache_path = cache_dir / "paper_cache.json"
    firefox_executable = Path(args.firefox_executable)
    chrome_executable = Path(args.chrome_executable)
    chrome_profile_template = Path(args.chrome_profile_template) if args.chrome_profile_template else None
    cookie_export_path = Path(args.cookie_export_path) if args.cookie_export_path else None
    firefox_profile_template = Path(args.firefox_profile_template) if args.firefox_profile_template else None
    lock_path = Path(args.lock_path) if args.lock_path else None
    batch_log_path = Path(args.batch_log_path) if args.batch_log_path else None

    @contextmanager
    def maybe_lock():
        if lock_path is None:
            yield
        else:
            with file_lock(lock_path):
                yield

    with maybe_lock():
        schedule = fetch_schedule()
        records = collect_records(schedule)
        apply_cached_status(records, load_cache(cache_path))

        preselected = selected_records(records, batch_limit=args.sample_limit, retry_tested=args.retry_tested)

        if args.mode != "metadata":
            if cookie_export_path is not None:
                validate_papers_with_chrome_cookies(
                    records,
                    sample_limit=args.sample_limit,
                    retry_tested=args.retry_tested,
                    download_pdfs=args.mode == "paper-cache" and not args.no_pdf_download,
                    chrome_executable=chrome_executable,
                    chrome_profile_template=chrome_profile_template,
                    cookie_export_path=cookie_export_path,
                    cache_dir=cache_dir,
                    cache_path=cache_path,
                    headless=args.headless,
                    wait_ms=args.wait_ms,
                )
            else:
                validate_papers_with_firefox(
                    records,
                    sample_limit=args.sample_limit,
                    retry_tested=args.retry_tested,
                    firefox_executable=firefox_executable,
                    cache_dir=cache_dir,
                    cache_path=cache_path,
                    firefox_profile_template=firefox_profile_template,
                    headless=args.headless,
                    wait_ms=args.wait_ms,
                )

        json_path = outdir / f"{args.basename}.json"
        csv_path = outdir / f"{args.basename}.csv"
        write_json(records, json_path)
        write_csv(records, csv_path)

        summary = summarize(records)
        if batch_log_path is not None:
            write_batch_log(
                batch_log_path,
                {
                    "mode": args.mode,
                    "started_candidates": [asdict(record) for record in preselected],
                    "summary": summary,
                    "json_path": str(json_path),
                    "csv_path": str(csv_path),
                },
            )

        print(json.dumps(summary, indent=2))
        print(json_path)
        print(csv_path)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
