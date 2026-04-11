#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait


DEFAULT_ACCOUNT = "jwt0625"
LIST_PATHS = {
    "verified_followers": "/verified_followers",
    "followers": "/followers",
    "following": "/following",
}
DEFAULT_TARGET_KIND = "verified_followers"
DEFAULT_FIREFOX_BINARY = "/Applications/Firefox.app/Contents/MacOS/firefox"
DEFAULT_FIREFOX_PROFILE = Path.home() / "Library/Application Support/Firefox/Profiles/9ons1v9u.default-release"
OUTPUT_DIR = Path("output")
NOISE_LINES = {
    "Follows you",
    "Follow",
    "Follow back",
    "Following",
    "Unfollow",
    "Subscribe",
    "Blocked",
    "Muted",
}


@dataclass
class FollowerRecord:
    name: str
    handle: str
    description: str
    profile_url: str
    profile_image_url: str
    downloaded_image_path: str = ""


@dataclass
class ScrollAudit:
    scroll_number: int
    visible_handles: list[str]
    new_handles: int
    overlap_with_previous: int
    overlap_ratio: float
    scroll_y: float
    resumed_from_checkpoint: bool = False


@dataclass
class CheckpointState:
    target_kind: str
    target_url: str
    output_prefix: str
    records: list[dict]
    audit_entries: list[dict]
    scroll_number: int
    stagnant_rounds: int
    last_count: int
    previous_visible: list[str]
    previous_scroll_y: float
    saved_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape X follower/following lists using an existing Firefox profile.")
    parser.add_argument("--mode", choices=["smoke", "inspect", "scrape"], default="smoke")
    parser.add_argument("--target-kind", choices=sorted(LIST_PATHS.keys()), default=DEFAULT_TARGET_KIND)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--url", default=None, help="Optional explicit list URL. Overrides --account and --target-kind.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_FIREFOX_PROFILE)
    parser.add_argument("--binary", default=DEFAULT_FIREFOX_BINARY)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds to wait between scrolls.")
    parser.add_argument("--max-scrolls", type=int, default=250)
    parser.add_argument("--stagnant-limit", type=int, default=8, help="Stop after this many scrolls without new accounts.")
    parser.add_argument("--limit", type=int, default=0, help="Optional hard cap on collected accounts.")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--image-dir", type=Path, default=OUTPUT_DIR / "profile_images")
    parser.add_argument(
        "--scroll-fraction",
        type=float,
        default=0.8,
        help="Scroll by this fraction of the viewport height each step to preserve overlap.",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.15,
        help="Warn if adjacent visible windows overlap by less than this fraction.",
    )
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write checkpoint every N scrolls.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint if present.")
    return parser.parse_args()


def build_target_url(account: str, target_kind: str) -> str:
    return f"https://x.com/{account}{LIST_PATHS[target_kind]}"


def derive_output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    return f"{args.account}_{args.target_kind}"


def derive_paths(args: argparse.Namespace) -> tuple[str, str, Path, Path]:
    target_url = args.url or build_target_url(args.account, args.target_kind)
    output_prefix = derive_output_prefix(args)
    audit_path = args.audit_path or (OUTPUT_DIR / f"{output_prefix}_audit.json")
    checkpoint_path = args.checkpoint_path or (OUTPUT_DIR / f"{output_prefix}_checkpoint.json")
    return target_url, output_prefix, audit_path, checkpoint_path


def clone_firefox_profile(profile_path: Path) -> Path:
    if not profile_path.exists():
        raise FileNotFoundError(f"Firefox profile not found: {profile_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="x-firefox-profile-"))
    destination = temp_dir / "profile"

    def ignore_files(_src: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {".parentlock", "parent.lock", "lock", "lockwise.sqlite-wal", "cookies.sqlite-wal"}
        }

    shutil.copytree(profile_path, destination, ignore=ignore_files, dirs_exist_ok=True)
    return destination


def build_driver(profile_copy: Path, firefox_binary: str, headless: bool) -> webdriver.Firefox:
    options = Options()
    options.binary_location = firefox_binary
    options.add_argument("-profile")
    options.add_argument(str(profile_copy))
    options.set_preference("remote.active-protocols", 1)
    options.set_preference("media.volume_scale", "0.0")
    options.set_preference("dom.webnotifications.enabled", False)
    options.set_preference("toolkit.telemetry.reportingpolicy.firstRun", False)
    if headless:
        options.add_argument("-headless")

    service = Service()
    driver = webdriver.Firefox(options=options, service=service)
    driver.set_window_size(1440, 1400)
    return driver


def wait_for_any(driver: webdriver.Firefox, timeout: int, selectors: Iterable[tuple[str, str]]) -> tuple[str, str] | None:
    end_time = time.time() + timeout
    while time.time() < end_time:
        for by, selector in selectors:
            elements = driver.find_elements(by, selector)
            visible = [element for element in elements if element.is_displayed()]
            if visible:
                return by, selector
        time.sleep(0.5)
    return None


def wait_for_x_shell(driver: webdriver.Firefox, timeout: int) -> None:
    selectors = [
        (By.CSS_SELECTOR, '[data-testid="primaryColumn"]'),
        (By.CSS_SELECTOR, '[data-testid="AppTabBar_Home_Link"]'),
        (By.CSS_SELECTOR, 'a[href="/home"]'),
    ]
    match = wait_for_any(driver, timeout, selectors)
    if not match:
        raise TimeoutException("Timed out waiting for X app shell.")


def detect_login_state(driver: webdriver.Firefox) -> str:
    if "login" in driver.current_url or "i/flow" in driver.current_url:
        return "login_required"
    login_like = driver.find_elements(By.CSS_SELECTOR, 'input[autocomplete="username"], input[name="text"]')
    if login_like:
        return "login_required"
    app_shell = driver.find_elements(By.CSS_SELECTOR, '[data-testid="AppTabBar_Home_Link"], a[href="/home"]')
    if app_shell:
        return "logged_in"
    return "unknown"


def normalize_profile_image(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"_(normal|mini|bigger)(?=\.)", "", url)


def clean_text_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in NOISE_LINES:
            continue
        cleaned.append(stripped)
    return cleaned


def maybe_find_user_cells(driver: webdriver.Firefox) -> list:
    selectors = [
        'div[data-testid="UserCell"]',
        'button[data-testid="UserCell"]',
        '[data-testid="cellInnerDiv"] div[data-testid="UserCell"]',
    ]
    for selector in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        visible = [element for element in elements if element.is_displayed()]
        if visible:
            return visible
    return []


def extract_from_user_cell(cell) -> FollowerRecord | None:
    try:
        links = cell.find_elements(By.CSS_SELECTOR, 'a[href^="/"]')
        profile_link = ""
        handle = ""
        for link in links:
            href = link.get_attribute("href") or ""
            text = (link.text or "").strip()
            if re.fullmatch(r"https://x\.com/[^/]+", href):
                profile_link = href
                if text.startswith("@"):
                    handle = text
                elif not handle:
                    handle = f"@{href.rstrip('/').split('/')[-1]}"
                break

        if not profile_link:
            for link in links:
                href = link.get_attribute("href") or ""
                if href.startswith("https://x.com/") and "/status/" not in href and "/photo/" not in href:
                    username = href.rstrip("/").split("/")[-1]
                    if username and username not in {"i", "home", "explore", "messages"}:
                        profile_link = href
                        handle = f"@{username}"
                        break

        text_lines = clean_text_lines((cell.text or "").splitlines())
        if not text_lines:
            return None

        name = ""
        description = ""
        if handle and handle in text_lines:
            handle_index = text_lines.index(handle)
            if handle_index > 0:
                name = text_lines[handle_index - 1]
            description = "\n".join(text_lines[handle_index + 1 :]).strip()
        else:
            for line in text_lines:
                if line.startswith("@"):
                    handle = line
                    break
            if handle:
                handle_index = text_lines.index(handle)
                if handle_index > 0:
                    name = text_lines[handle_index - 1]
                description = "\n".join(text_lines[handle_index + 1 :]).strip()
            elif len(text_lines) >= 2:
                name = text_lines[0]
                handle = text_lines[1] if text_lines[1].startswith("@") else ""
                description = "\n".join(text_lines[2:]).strip()

        if not name:
            name = text_lines[0]

        profile_image_url = ""
        try:
            img = cell.find_element(By.CSS_SELECTOR, "img")
            profile_image_url = normalize_profile_image(img.get_attribute("src") or "")
        except NoSuchElementException:
            pass

        if not handle and profile_link:
            handle = f"@{profile_link.rstrip('/').split('/')[-1]}"
        if not profile_link and handle:
            profile_link = f"https://x.com/{handle.lstrip('@')}"
        if not handle:
            return None

        return FollowerRecord(
            name=name,
            handle=handle,
            description=description,
            profile_url=profile_link,
            profile_image_url=profile_image_url,
        )
    except WebDriverException:
        return None


def inspect_page(driver: webdriver.Firefox, timeout: int, output_prefix: str) -> None:
    wait_for_x_shell(driver, timeout)
    time.sleep(5)
    cells = maybe_find_user_cells(driver)
    print(f"URL: {driver.current_url}", flush=True)
    print(f"TITLE: {driver.title}", flush=True)
    print(f"VISIBLE_USER_CELLS: {len(cells)}", flush=True)
    for index, cell in enumerate(cells[:5], start=1):
        print(f"\n--- CELL {index} ---", flush=True)
        print(cell.text, flush=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    screenshot_path = OUTPUT_DIR / f"{output_prefix}_inspect.png"
    driver.save_screenshot(str(screenshot_path))
    print(f"\nSaved screenshot to: {screenshot_path.resolve()}", flush=True)


def collect_visible_records(cells: list) -> list[FollowerRecord]:
    visible_records: list[FollowerRecord] = []
    for cell in cells:
        record = extract_from_user_cell(cell)
        if record:
            visible_records.append(record)
    return visible_records


def scroll_by_viewport_fraction(driver: webdriver.Firefox, fraction: float) -> float:
    viewport_height = driver.execute_script("return window.innerHeight;")
    current_y = driver.execute_script("return window.pageYOffset;")
    target_y = current_y + (viewport_height * fraction)
    max_y = driver.execute_script(
        "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight) - window.innerHeight;"
    )
    bounded_target = min(target_y, max_y)
    driver.execute_script("window.scrollTo(0, arguments[0]);", bounded_target)
    return float(bounded_target)


def write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def write_audit(audit_entries: list[ScrollAudit], audit_path: Path) -> None:
    write_json_atomic([asdict(entry) for entry in audit_entries], audit_path)


def write_output(records: list[FollowerRecord], prefix: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / f"{prefix}.json"
    csv_path = OUTPUT_DIR / f"{prefix}.csv"

    write_json_atomic([asdict(record) for record in records], json_path)

    temp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "handle",
                "description",
                "profile_url",
                "profile_image_url",
                "downloaded_image_path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    temp_csv.replace(csv_path)
    return json_path, csv_path


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def download_profile_images(records: list[FollowerRecord], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        if not record.profile_image_url:
            continue

        parsed = urllib.parse.urlparse(record.profile_image_url)
        extension = Path(parsed.path).suffix or ".jpg"
        filename = f"{index:05d}_{sanitize_filename(record.handle.lstrip('@'))}{extension}"
        destination = image_dir / filename

        try:
            request = urllib.request.Request(
                record.profile_image_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) "
                        "Gecko/20100101 Firefox/138.0"
                    ),
                    "Referer": "https://x.com/",
                },
            )
            with urllib.request.urlopen(request) as response, destination.open("wb") as fh:
                shutil.copyfileobj(response, fh)
            record.downloaded_image_path = str(destination.resolve())
            print(f"downloaded_image={index}/{len(records)} handle={record.handle}", flush=True)
            time.sleep(0.15)
        except Exception as exc:
            print(f"image_download_failed handle={record.handle} error={exc}", flush=True)


def write_checkpoint(
    checkpoint_path: Path,
    target_kind: str,
    target_url: str,
    output_prefix: str,
    collected: dict[str, FollowerRecord],
    audit_entries: list[ScrollAudit],
    scroll_number: int,
    stagnant_rounds: int,
    last_count: int,
    previous_visible: set[str],
    previous_scroll_y: float,
) -> None:
    payload = CheckpointState(
        target_kind=target_kind,
        target_url=target_url,
        output_prefix=output_prefix,
        records=[asdict(record) for record in collected.values()],
        audit_entries=[asdict(entry) for entry in audit_entries],
        scroll_number=scroll_number,
        stagnant_rounds=stagnant_rounds,
        last_count=last_count,
        previous_visible=sorted(previous_visible),
        previous_scroll_y=previous_scroll_y,
        saved_at=time.time(),
    )
    write_json_atomic(asdict(payload), checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> CheckpointState:
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return CheckpointState(**payload)


def hydrate_records(serialized: list[dict]) -> dict[str, FollowerRecord]:
    records: dict[str, FollowerRecord] = {}
    for item in serialized:
        record = FollowerRecord(**item)
        records[record.handle] = record
    return records


def hydrate_audit_entries(serialized: list[dict]) -> list[ScrollAudit]:
    return [ScrollAudit(**item) for item in serialized]


def maybe_load_resume_state(
    args: argparse.Namespace,
    checkpoint_path: Path,
    target_kind: str,
    target_url: str,
    output_prefix: str,
) -> tuple[dict[str, FollowerRecord], list[ScrollAudit], int, int, int, set[str], float]:
    if not args.resume or not checkpoint_path.exists():
        return {}, [], 0, 0, 0, set(), -1.0

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.target_url != target_url or checkpoint.output_prefix != output_prefix or checkpoint.target_kind != target_kind:
        raise ValueError("Checkpoint does not match the requested scrape target or output prefix.")

    collected = hydrate_records(checkpoint.records)
    audit_entries = hydrate_audit_entries(checkpoint.audit_entries)
    previous_visible = set(checkpoint.previous_visible)
    print(
        f"Resuming from checkpoint: scroll={checkpoint.scroll_number} collected={len(collected)}",
        flush=True,
    )
    return (
        collected,
        audit_entries,
        checkpoint.scroll_number,
        checkpoint.stagnant_rounds,
        checkpoint.last_count,
        previous_visible,
        checkpoint.previous_scroll_y,
    )


def scrape_followers(
    driver: webdriver.Firefox,
    args: argparse.Namespace,
    target_kind: str,
    target_url: str,
    output_prefix: str,
    checkpoint_path: Path,
) -> tuple[list[FollowerRecord], list[ScrollAudit]]:
    wait_for_x_shell(driver, args.timeout)

    wait = WebDriverWait(driver, args.timeout)
    wait.until(lambda d: len(maybe_find_user_cells(d)) > 0)
    time.sleep(2)

    (
        collected,
        audit_entries,
        completed_scrolls,
        stagnant_rounds,
        last_count,
        previous_visible,
        previous_scroll_y,
    ) = maybe_load_resume_state(args, checkpoint_path, target_kind, target_url, output_prefix)

    if completed_scrolls and previous_scroll_y >= 0:
        driver.execute_script("window.scrollTo(0, arguments[0]);", previous_scroll_y)
        time.sleep(args.pause)

    for scroll_number in range(completed_scrolls + 1, args.max_scrolls + 1):
        cells = maybe_find_user_cells(driver)
        visible_records = collect_visible_records(cells)
        visible_handles = [record.handle for record in visible_records]

        new_handles = 0
        for record in visible_records:
            if record.handle not in collected:
                collected[record.handle] = record
                new_handles += 1
            elif not collected[record.handle].description and record.description:
                collected[record.handle] = record

        current_visible = set(visible_handles)
        overlap_count = len(previous_visible & current_visible) if previous_visible else 0
        overlap_denominator = max(1, min(len(previous_visible), len(current_visible))) if previous_visible else 1
        overlap_ratio = overlap_count / overlap_denominator if previous_visible else 1.0
        scroll_y = float(driver.execute_script("return window.pageYOffset;"))

        audit_entry = ScrollAudit(
            scroll_number=scroll_number,
            visible_handles=visible_handles,
            new_handles=new_handles,
            overlap_with_previous=overlap_count,
            overlap_ratio=overlap_ratio,
            scroll_y=scroll_y,
            resumed_from_checkpoint=bool(completed_scrolls and scroll_number == completed_scrolls + 1),
        )
        audit_entries.append(audit_entry)

        current_count = len(collected)
        print(
            "scroll="
            f"{scroll_number} collected={current_count} visible_cells={len(cells)} "
            f"new_handles={new_handles} overlap={overlap_count} overlap_ratio={overlap_ratio:.3f}",
            flush=True,
        )

        if previous_visible and overlap_ratio < args.min_overlap_ratio:
            print(
                f"overlap_warning scroll={scroll_number} overlap_ratio={overlap_ratio:.3f} "
                f"min_expected={args.min_overlap_ratio:.3f}",
                flush=True,
            )

        if scroll_number % max(1, args.checkpoint_every) == 0:
            write_checkpoint(
                checkpoint_path=checkpoint_path,
                target_kind=target_kind,
                target_url=target_url,
                output_prefix=output_prefix,
                collected=collected,
                audit_entries=audit_entries,
                scroll_number=scroll_number,
                stagnant_rounds=stagnant_rounds,
                last_count=last_count,
                previous_visible=current_visible,
                previous_scroll_y=scroll_y,
            )

        if args.limit and current_count >= args.limit:
            break

        if current_count == last_count:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
            last_count = current_count

        if stagnant_rounds >= args.stagnant_limit:
            print(f"Stopping after {stagnant_rounds} stagnant scrolls.", flush=True)
            break

        target_scroll_y = scroll_by_viewport_fraction(driver, args.scroll_fraction)
        time.sleep(args.pause)

        if math.isclose(target_scroll_y, previous_scroll_y, abs_tol=1.0):
            print("Reached the current bottom of the page; waiting for more content or stop condition.", flush=True)

        previous_visible = current_visible
        previous_scroll_y = target_scroll_y

    write_checkpoint(
        checkpoint_path=checkpoint_path,
        target_kind=target_kind,
        target_url=target_url,
        output_prefix=output_prefix,
        collected=collected,
        audit_entries=audit_entries,
        scroll_number=min(args.max_scrolls, len(audit_entries)),
        stagnant_rounds=stagnant_rounds,
        last_count=last_count,
        previous_visible=previous_visible,
        previous_scroll_y=previous_scroll_y,
    )
    return list(collected.values()), audit_entries


def maybe_remove_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()


def save_emergency_checkpoint(
    checkpoint_path: Path,
    target_kind: str,
    target_url: str,
    output_prefix: str,
    records: list[FollowerRecord],
    audit_entries: list[ScrollAudit],
) -> None:
    write_checkpoint(
        checkpoint_path=checkpoint_path,
        target_kind=target_kind,
        target_url=target_url,
        output_prefix=output_prefix,
        collected={record.handle: record for record in records},
        audit_entries=audit_entries,
        scroll_number=len(audit_entries),
        stagnant_rounds=0,
        last_count=len(records),
        previous_visible=set(audit_entries[-1].visible_handles) if audit_entries else set(),
        previous_scroll_y=audit_entries[-1].scroll_y if audit_entries else -1.0,
    )


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    target_url, output_prefix, audit_path, checkpoint_path = derive_paths(args)
    profile_copy = clone_firefox_profile(args.profile)
    print(f"Using temporary profile copy: {profile_copy}", flush=True)
    print(f"Target URL: {target_url}", flush=True)

    driver = None
    try:
        driver = build_driver(profile_copy, args.binary, args.headless)
        driver.get(target_url if args.mode != "smoke" else "https://x.com/home")

        if args.mode == "smoke":
            wait_for_x_shell(driver, args.timeout)
            login_state = detect_login_state(driver)
            print(f"LOGIN_STATE: {login_state}", flush=True)
            print(f"CURRENT_URL: {driver.current_url}", flush=True)
            print(f"TITLE: {driver.title}", flush=True)
            return 0 if login_state == "logged_in" else 2

        if detect_login_state(driver) == "login_required":
            print("Firefox profile did not arrive logged in to X.", flush=True)
            return 2

        if args.mode == "inspect":
            inspect_page(driver, args.timeout, output_prefix)
            return 0

        records, audit_entries = scrape_followers(
            driver=driver,
            args=args,
            target_kind=args.target_kind,
            target_url=target_url,
            output_prefix=output_prefix,
            checkpoint_path=checkpoint_path,
        )
        if args.limit:
            records = records[: args.limit]
        if args.download_images:
            download_profile_images(records, args.image_dir)
        json_path, csv_path = write_output(records, output_prefix)
        write_audit(audit_entries, audit_path)
        maybe_remove_checkpoint(checkpoint_path)
        print(f"Saved {len(records)} records to:", flush=True)
        print(json_path.resolve(), flush=True)
        print(csv_path.resolve(), flush=True)
        print(audit_path.resolve(), flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user. Preserving checkpoint for resume.", flush=True)
        if args.mode == "scrape":
            try:
                if checkpoint_path.exists():
                    print(checkpoint_path.resolve(), flush=True)
                else:
                    save_emergency_checkpoint(
                        checkpoint_path=checkpoint_path,
                        target_kind=args.target_kind,
                        target_url=target_url,
                        output_prefix=output_prefix,
                        records=[],
                        audit_entries=[],
                    )
                    print(checkpoint_path.resolve(), flush=True)
            except Exception as exc:
                print(f"Checkpoint preservation failed: {exc}", flush=True)
        return 130
    finally:
        if driver is not None:
            driver.quit()
        shutil.rmtree(profile_copy.parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
