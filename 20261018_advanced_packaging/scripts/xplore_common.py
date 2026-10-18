"""Shared IEEE Xplore helpers: headless Chromium session + rate-limited same-origin REST calls."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CHROMIUM = (
    Path.home()
    / "Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/"
    "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
)
BASE = "https://ieeexplore.ieee.org"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


class AntiBotError(RuntimeError):
    pass


def log_path(name: str) -> Path:
    d = ROOT / "logs"
    d.mkdir(exist_ok=True)
    return d / f"{name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"


class Xplore:
    """Rate-limited wrapper. Every request waits base_wait + jitter seconds first."""

    def __init__(self, page: Page, log: Path, base_wait: float = 3.5, jitter: float = 2.0):
        self.page = page
        self.log = log
        self.base_wait = base_wait
        self.jitter = jitter
        self.n_requests = 0

    def _write(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with self.log.open("a") as f:
            f.write(line + "\n")

    def _wait(self) -> None:
        time.sleep(self.base_wait + random.uniform(0, self.jitter))

    def call(self, method: str, url: str, body: dict | None = None) -> Any:
        self._wait()
        self.n_requests += 1
        res = self.page.evaluate(
            """async ({method, url, body}) => {
                const r = await fetch(url, {
                    method,
                    headers: {"content-type": "application/json",
                              accept: "application/json, text/plain, */*"},
                    body: body ? JSON.stringify(body) : undefined,
                    credentials: "same-origin",
                });
                return {status: r.status, text: await r.text()};
            }""",
            {"method": method, "url": url, "body": body},
        )
        status, text = res["status"], res["text"]
        self._write(f"{method} {url} body={json.dumps(body) if body else ''} -> {status} {len(text)}B")
        if status in (403, 429) or "captcha" in text[:2000].lower():
            raise AntiBotError(f"anti-bot signal: {status} on {url}")
        if status != 200:
            raise RuntimeError(f"{status} on {url}: {text[:300]}")
        if not text.strip():
            raise AntiBotError(f"empty body on {url}")
        return json.loads(text)

    def get(self, path: str) -> Any:
        return self.call("GET", BASE + path)

    def post(self, path: str, body: dict) -> Any:
        return self.call("POST", BASE + path, body)

    def page_html(self, path: str) -> str:
        self._wait()
        self.n_requests += 1
        self.page.goto(BASE + path, wait_until="domcontentloaded", timeout=60000)
        html = self.page.content()
        self._write(f"GOTO {path} -> {len(html)}B")
        return html


@contextmanager
def xplore_session(log_name: str, **kw: Any) -> Iterator[Xplore]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=str(CHROMIUM))
        ctx = browser.new_context(user_agent=UA, locale="en-US")
        page = ctx.new_page()
        page.goto(BASE + "/Xplore/home.jsp", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        try:
            yield Xplore(page, log_path(log_name), **kw)
        finally:
            browser.close()
