#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VIEWER_DIR = ROOT / "viewer"
OUTPUT_DIR = ROOT / "output"


def list_datasets() -> list[dict[str, object]]:
    datasets: list[dict[str, object]] = []
    if not OUTPUT_DIR.exists():
        return datasets

    for path in sorted(OUTPUT_DIR.glob("*.json")):
        if path.name.endswith("_audit.json") or path.name.endswith("_checkpoint.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            datasets.append(
                {
                    "name": path.stem,
                    "file": path.name,
                    "count": len(data),
                }
            )
        except Exception:
            continue
    return datasets


class ViewerHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed_path = urllib.parse.urlparse(path).path
        relative = posixpath.normpath(urllib.parse.unquote(parsed_path)).lstrip("/")
        if not relative or relative == ".":
            relative = "index.html"
        return str(VIEWER_DIR / relative)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/datasets":
            self.send_json(list_datasets())
            return

        if parsed.path == "/api/data":
            params = urllib.parse.parse_qs(parsed.query)
            requested = params.get("name", [""])[0]
            dataset_path = (OUTPUT_DIR / requested).resolve()
            if not requested.endswith(".json") or dataset_path.parent != OUTPUT_DIR.resolve() or not dataset_path.exists():
                self.send_error(404, "Dataset not found")
                return
            self.send_json(json.loads(dataset_path.read_text(encoding="utf-8")))
            return

        super().do_GET()

    def send_json(self, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:
        guessed, _ = mimetypes.guess_type(path)
        return guessed or "application/octet-stream"


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler)
    print(f"Viewer running at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
