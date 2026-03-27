#!/usr/bin/env python3
"""Cache COMSOL release pages and extract embedded Wistia media."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


BASE_URL = "https://www.comsol.com"
RELEASE_HISTORY_URL = f"{BASE_URL}/release-history"
WISTIA_MEDIA_JSON = "https://fast.wistia.com/embed/medias/{media_id}.json"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)
DEFAULT_RELEASES = ["5.4", "5.5", "5.6", "6.0", "6.1", "6.2", "6.3", "6.4"]


@dataclass(frozen=True)
class WistiaAsset:
    media_id: str
    name: str
    url: str
    width: int
    height: int
    ext: str
    bitrate: int
    asset_type: str

    @property
    def score(self) -> tuple[int, int, int]:
        return (self.width * self.height, self.bitrate, self.width)


class Scraper:
    def __init__(self, root: Path, force: bool = False, sleep_seconds: float = 0.15):
        self.root = root
        self.force = force
        self.sleep_seconds = sleep_seconds
        self.cache_dir = root / "cache"
        self.metadata_dir = root / "metadata"
        self.media_dir = root / "media"
        self.index_dir = root / "indexes"
        for directory in (self.cache_dir, self.metadata_dir, self.media_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def fetch_text(self, url: str, cache_path: Path) -> str:
        if cache_path.exists() and not self.force:
            return cache_path.read_text(encoding="utf-8")
        text = self._download_text(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text

    def fetch_json(self, url: str, cache_path: Path) -> dict:
        if cache_path.exists() and not self.force:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        payload = self._download_text(url)
        data = json.loads(payload)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data

    def download_binary(self, url: str, output_path: Path) -> None:
        if output_path.exists() and not self.force:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request) as response, output_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        time.sleep(self.sleep_seconds)

    def _download_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request) as response:
                payload = response.read()
        except HTTPError as exc:
            raise SystemExit(f"HTTP error for {url}: {exc.code}") from exc
        except URLError as exc:
            raise SystemExit(f"Network error for {url}: {exc.reason}") from exc
        time.sleep(self.sleep_seconds)
        return payload.decode("utf-8", "ignore")


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", value.lower()).strip("_")


def release_history_cache_path(scraper: Scraper) -> Path:
    return scraper.cache_dir / "release-history.html"


def release_page_cache_path(scraper: Scraper, release: str, page_slug: str) -> Path:
    return scraper.cache_dir / release / f"{page_slug}.html"


def release_metadata_path(scraper: Scraper, release: str) -> Path:
    return scraper.index_dir / f"release_{release}.json"


def releases_index_path(scraper: Scraper) -> Path:
    return scraper.index_dir / "releases.json"


def page_media_path(scraper: Scraper, release: str, page_slug: str) -> Path:
    return scraper.metadata_dir / release / f"{page_slug}.json"


def media_json_cache_path(scraper: Scraper, media_id: str) -> Path:
    return scraper.metadata_dir / "wistia" / f"{media_id}.json"


def extract_release_roots(html: str) -> list[dict]:
    pattern = re.compile(
        r'<a\s+href="(?P<href>/release/(?P<version>\d+\.\d+))"[^>]*>\s*(?P<label>\d+\.\d+\s+Release Highlights)\s*</a>',
        re.IGNORECASE,
    )
    seen = set()
    releases = []
    for match in pattern.finditer(html):
        version = match.group("version")
        if version in seen:
            continue
        seen.add(version)
        releases.append(
            {
                "version": version,
                "url": urljoin(BASE_URL, match.group("href")),
                "label": " ".join(unescape(match.group("label")).split()),
            }
        )
    return sorted(releases, key=lambda item: [int(part) for part in item["version"].split(".")], reverse=True)


def extract_release_subpages(html: str, release: str) -> list[dict]:
    pattern = re.compile(rf'href="(/release/{re.escape(release)}/[^"#?]+)"', re.IGNORECASE)
    seen = set()
    pages = []
    for href in pattern.findall(html):
        href = unescape(href)
        if href in seen:
            continue
        seen.add(href)
        slug = href.rstrip("/").split("/")[-1]
        pages.append(
            {
                "slug": slug,
                "path": href,
                "url": urljoin(BASE_URL, href),
            }
        )
    return sorted(pages, key=lambda item: item["slug"])


def extract_wistia_ids(html: str) -> list[str]:
    ids = set(re.findall(r"wistia_async_([a-z0-9]{10})", html, re.IGNORECASE))
    ids.update(re.findall(r"fast\.wistia\.com/embed/medias/([a-z0-9]{10})", html, re.IGNORECASE))
    return sorted(ids)


def pick_best_asset(media_json: dict) -> WistiaAsset | None:
    media = media_json.get("media") or {}
    media_id = str(media.get("hashedId") or "")
    name = str(media.get("name") or media_id)
    assets = media.get("assets") or []
    candidates = []
    for asset in assets:
        if not asset.get("public"):
            continue
        ext = asset.get("ext")
        container = asset.get("container")
        url = asset.get("url")
        width = int(asset.get("width") or 0)
        height = int(asset.get("height") or 0)
        bitrate = int(asset.get("bitrate") or 0)
        if not url:
            continue
        if ext == "mp4" or container == "mp4":
            candidates.append(
                WistiaAsset(
                    media_id=media_id,
                    name=name,
                    url=url,
                    width=width,
                    height=height,
                    ext="mp4",
                    bitrate=bitrate,
                    asset_type=str(asset.get("type") or ""),
                )
            )
    if candidates:
        return max(candidates, key=lambda item: item.score)
    for asset in assets:
        if asset.get("public") and asset.get("type") == "original" and asset.get("url"):
            parsed = urlparse(asset["url"])
            suffix = Path(parsed.path).suffix.lstrip(".") or "bin"
            return WistiaAsset(
                media_id=media_id,
                name=name,
                url=asset["url"],
                width=int(asset.get("width") or 0),
                height=int(asset.get("height") or 0),
                ext=suffix,
                bitrate=int(asset.get("bitrate") or 0),
                asset_type="original",
            )
    return None


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def discover_releases(scraper: Scraper) -> list[dict]:
    html = scraper.fetch_text(RELEASE_HISTORY_URL, release_history_cache_path(scraper))
    releases = extract_release_roots(html)
    save_json(releases_index_path(scraper), releases)
    return releases


def scrape_release(scraper: Scraper, release: str, download_media: bool = True) -> dict:
    release_url = f"{BASE_URL}/release/{release}"
    release_html = scraper.fetch_text(
        release_url,
        release_page_cache_path(scraper, release, "index"),
    )
    pages = extract_release_subpages(release_html, release)

    page_summaries = []
    for page in pages:
        html = scraper.fetch_text(
            page["url"],
            release_page_cache_path(scraper, release, page["slug"]),
        )
        wistia_ids = extract_wistia_ids(html)
        media_entries = []
        for media_id in wistia_ids:
            media_json = scraper.fetch_json(
                WISTIA_MEDIA_JSON.format(media_id=media_id),
                media_json_cache_path(scraper, media_id),
            )
            best_asset = pick_best_asset(media_json)
            saved_path = None
            if best_asset and download_media:
                output_name = (
                    f"{release}_{page['slug']}_{sanitize_slug(best_asset.name)}_{best_asset.media_id}."
                    f"{best_asset.ext}"
                )
                output_path = scraper.media_dir / release / output_name
                scraper.download_binary(best_asset.url, output_path)
                saved_path = str(output_path)
            media_entries.append(
                {
                    "media_id": media_id,
                    "name": (media_json.get("media") or {}).get("name"),
                    "duration": (media_json.get("media") or {}).get("duration"),
                    "best_asset": None
                    if not best_asset
                    else {
                        "url": best_asset.url,
                        "ext": best_asset.ext,
                        "width": best_asset.width,
                        "height": best_asset.height,
                        "bitrate": best_asset.bitrate,
                        "asset_type": best_asset.asset_type,
                        "saved_path": saved_path,
                    },
                }
            )
        page_summary = {
            "release": release,
            "page_slug": page["slug"],
            "page_url": page["url"],
            "wistia_ids": wistia_ids,
            "media": media_entries,
        }
        save_json(page_media_path(scraper, release, page["slug"]), page_summary)
        page_summaries.append(page_summary)

    summary = {
        "release": release,
        "release_url": release_url,
        "page_count": len(page_summaries),
        "pages": page_summaries,
    }
    save_json(release_metadata_path(scraper, release), summary)
    return summary


def iter_requested_releases(scraper: Scraper, requested: Iterable[str] | None) -> list[str]:
    if requested:
        return list(requested)
    return list(DEFAULT_RELEASES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="build/comsol_release_scrape",
        help="Output root for cache, metadata, and downloaded media",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch files even if a cached copy already exists",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("discover", help="Cache release-history and write release roots")

    release_parser = subparsers.add_parser("scrape-release", help="Scrape one release root and its subpages")
    release_parser.add_argument("release", help="Release version like 5.4")
    release_parser.add_argument(
        "--no-download-media",
        action="store_true",
        help="Only cache pages and media JSON, without downloading the video files",
    )

    all_parser = subparsers.add_parser("scrape-all", help="Scrape one or more releases")
    all_parser.add_argument(
        "--release",
        dest="releases",
        action="append",
        help="Limit scraping to one or more specific releases; otherwise use the built-in release list",
    )
    all_parser.add_argument(
        "--no-download-media",
        action="store_true",
        help="Only cache pages and media JSON, without downloading the video files",
    )
    return parser


def print_release_summary(releases: list[dict]) -> None:
    print(f"Discovered {len(releases)} release highlight pages")
    for item in releases:
        print(f"{item['version']}\t{item['url']}")


def print_scrape_summary(summary: dict) -> None:
    media_count = sum(len(page["media"]) for page in summary["pages"])
    pages_with_media = sum(1 for page in summary["pages"] if page["media"])
    print(
        f"Release {summary['release']}: {summary['page_count']} pages, "
        f"{pages_with_media} pages with media, {media_count} Wistia embeds"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    scraper = Scraper(Path(args.root), force=args.force)

    if args.command == "discover":
        releases = discover_releases(scraper)
        print_release_summary(releases)
        return 0

    if args.command == "scrape-release":
        summary = scrape_release(
            scraper,
            args.release,
            download_media=not args.no_download_media,
        )
        print_scrape_summary(summary)
        return 0

    if args.command == "scrape-all":
        releases = iter_requested_releases(scraper, args.releases)
        print(f"Scraping {len(releases)} release(s)")
        for release in releases:
            summary = scrape_release(
                scraper,
                release,
                download_media=not args.no_download_media,
            )
            print_scrape_summary(summary)
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
