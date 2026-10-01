"""Tests for scripts/fetch_source.py against a local HTTP server: serialization, breaker, atomic cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "fetch_source.py"
PDF = b"%PDF-1.4\n" + b"x" * 4000 + b"\n%%EOF\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/ok"):
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(PDF)))
            self.end_headers()
            self.wfile.write(PDF)
        elif self.path.startswith("/html"):
            body = b"<html>challenge</html>" + b" " * 5000
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/429"):
            self.send_response(429)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path.startswith("/short"):
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(PDF) + 5000))
            self.end_headers()
            self.wfile.write(PDF)
            self.wfile.flush()
            self.close_connection = True
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture()
def server() -> str:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def env_for(tmp: Path, delay: str = "1.0") -> dict[str, str]:
    return {
        **os.environ,
        "EO_ATLAS_ROOT": str(tmp),
        "EO_ATLAS_MIN_DELAY": delay,
        "EO_ATLAS_JITTER": "0",
        "EO_ATLAS_MIN_PDF_BYTES": "100",
        "EO_ATLAS_COOLDOWN_S": "60",
    }


def run(tmp: Path, pid: str, url: str, delay: str = "1.0") -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), "--paper-id", pid, "--url", url],
        env=env_for(tmp, delay),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def fetch_times(tmp: Path) -> list[float]:
    ts: list[float] = []
    for f in (tmp / "logs").glob("fetch_*.jsonl"):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "fetch":
                ts.append(e["ts"])
    return sorted(ts)


def test_ok_download_is_atomic_and_cached(tmp_path: Path, server: str) -> None:
    p = run(tmp_path, "a2025", f"{server}/ok.pdf")
    out, err = p.communicate(timeout=30)
    assert p.returncode == 0, err
    dest = tmp_path / "references/a2025/source.pdf"
    assert dest.read_bytes() == PDF
    assert not (dest.parent / "source.pdf.part").exists()
    p2 = run(tmp_path, "a2025", f"{server}/ok.pdf")
    out2, _ = p2.communicate(timeout=30)
    assert "skip-existing" in out2


def test_simultaneous_fetches_are_serialized(tmp_path: Path, server: str) -> None:
    import datetime as dt

    procs = [run(tmp_path, f"p{i}", f"{server}/ok{i}.pdf", delay="1.5") for i in range(4)]
    for p in procs:
        p.communicate(timeout=60)
        assert p.returncode == 0
    stamps = [dt.datetime.fromisoformat(t).timestamp() for t in fetch_times(tmp_path)]
    assert len(stamps) == 4
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert min(gaps) >= 1.0, gaps  # >= MIN_DELAY minus second-level timestamp rounding


def test_429_arms_breaker_and_blocks_others_without_network(tmp_path: Path, server: str) -> None:
    p = run(tmp_path, "b2025", f"{server}/429")
    _, err = p.communicate(timeout=30)
    assert p.returncode == 3
    assert (tmp_path / "logs/.fetch_blocked_127.0.0.1.json").exists()
    before = len(fetch_times(tmp_path))
    p2 = run(tmp_path, "c2025", f"{server}/ok.pdf")
    _, err2 = p2.communicate(timeout=30)
    assert p2.returncode == 4, err2
    assert len(fetch_times(tmp_path)) == before  # no second request was made
    assert not (tmp_path / "references/c2025/source.pdf").exists()


def test_breaker_is_per_host(tmp_path: Path, server: str) -> None:
    p = run(tmp_path, "f2025", f"{server}/429")
    p.communicate(timeout=30)
    assert p.returncode == 3
    other = server.replace("127.0.0.1", "localhost")
    p2 = run(tmp_path, "g2025", f"{other}/ok.pdf")
    _, err2 = p2.communicate(timeout=30)
    assert p2.returncode == 0, err2  # different host: not blocked
    assert (tmp_path / "references/g2025/source.pdf").exists()


def test_html_challenge_is_not_cached(tmp_path: Path, server: str) -> None:
    p = run(tmp_path, "d2025", f"{server}/html")
    p.communicate(timeout=30)
    assert p.returncode == 3
    assert not (tmp_path / "references/d2025/source.pdf").exists()
    assert (tmp_path / "logs/.fetch_blocked_127.0.0.1.json").exists()


def test_truncated_download_never_becomes_a_cache_hit(tmp_path: Path, server: str) -> None:
    p = run(tmp_path, "e2025", f"{server}/short")
    p.communicate(timeout=30)
    assert p.returncode == 3
    d = tmp_path / "references/e2025"
    assert not (d / "source.pdf").exists()
    assert not (d / "source.pdf.part").exists()
