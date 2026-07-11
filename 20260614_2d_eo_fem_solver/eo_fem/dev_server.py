from __future__ import annotations

import argparse
import errno
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Static handler that always revalidates development assets."""

    def __init__(self, *args: object, directory: str | None = None, **kwargs: object) -> None:
        self.workspace_root = Path(directory or ".").resolve()
        super().__init__(*args, directory=str(self.workspace_root), **kwargs)  # type: ignore[arg-type]

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json({"status": "ok", "service": "eo-fem-dev", "api_version": "v1"})
            return
        if path == "/api/examples":
            self._send_json({"examples": _discover_examples(self.workspace_root)})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path != "/api/mesh/generate":
            self.send_error(404, "unknown API endpoint")
            return
        try:
            payload = self._read_json()
            config_path = _safe_workspace_path(self.workspace_root, str(payload["config_path"]), "examples")
            output_name = _safe_name(str(payload.get("name", config_path.stem)))
            output_dir = self.workspace_root / "artifacts" / "api" / output_name
            job = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eo_fem.backends.mesh_job",
                    "--config",
                    str(config_path),
                    "--out",
                    str(output_dir),
                    "--name",
                    output_name,
                ],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if job.returncode != 0:
                detail = (job.stderr or job.stdout).strip().splitlines()[-1]
                raise RuntimeError(f"Gmsh job exited {job.returncode}: {detail}")
            result = json.loads(job.stdout.strip().splitlines()[-1])
            mesh_path = Path(result["mesh_path"])
            manifest_path = Path(result["manifest_path"])
            self._send_json(
                {
                    "status": "generated",
                    "mesh_url": _workspace_url(self.workspace_root, mesh_path),
                    "manifest_url": _workspace_url(self.workspace_root, manifest_path),
                    "node_count": result["node_count"],
                    "element_count": result["element_count"],
                },
                status=201,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:  # pragma: no cover - protects the long-running dev server
            self._send_json({"error": f"mesh generation failed: {error}"}, status=500)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("request body must contain at most 1 MB of JSON")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def create_server(
    host: str,
    preferred_port: int,
    directory: str | Path,
    *,
    max_attempts: int = 100,
) -> tuple[ThreadingHTTPServer, int]:
    """Bind a static-file server, scanning upward if a port is occupied."""

    handler = partial(NoCacheHTTPRequestHandler, directory=str(directory))
    for port in range(preferred_port, preferred_port + max_attempts):
        try:
            server = ThreadingHTTPServer((host, port), handler)
            return server, int(server.server_address[1])
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    end_port = preferred_port + max_attempts - 1
    raise OSError(f"no available port in range {preferred_port}-{end_port}")


def _discover_examples(root: Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    examples_root = root / "examples"
    if examples_root.is_dir():
        for path in sorted(examples_root.rglob("*.yaml")):
            relative = path.relative_to(root).as_posix()
            examples.append(
                {
                    "id": f"config:{relative}",
                    "label": path.stem.replace("_", " ").title(),
                    "kind": "config",
                    "dimension": 2,
                    "url": f"/{relative}",
                    "config_path": relative,
                }
            )
    artifacts_root = root / "artifacts"
    if artifacts_root.is_dir():
        for path in sorted(artifacts_root.rglob("*.msh")):
            relative = path.relative_to(root).as_posix()
            dimension = 3 if "mesh_controls_3d" in path.parts else 2
            manifest = path.with_suffix(".tags.json")
            entry: dict[str, object] = {
                "id": f"mesh:{relative}",
                "label": path.stem.replace("_", " ").title(),
                "kind": "mesh",
                "dimension": dimension,
                "url": f"/{relative}",
                "bytes": path.stat().st_size,
            }
            if manifest.exists():
                entry["manifest_url"] = _workspace_url(root, manifest)
            examples.append(entry)
    return examples


def _safe_workspace_path(root: Path, value: str, required_parent: str) -> Path:
    candidate = (root / value).resolve()
    allowed = (root / required_parent).resolve()
    if not candidate.is_relative_to(allowed) or not candidate.is_file():
        raise ValueError(f"path must reference an existing file under {required_parent}/")
    return candidate


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not name:
        raise ValueError("name must contain a letter or number")
    return name


def _workspace_url(root: Path, path: Path) -> str:
    return f"/{path.resolve().relative_to(root).as_posix()}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the EO FEM browser frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173, help="Preferred port; scans upward when occupied.")
    parser.add_argument("--directory", default=".")
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args(argv)

    server, selected_port = create_server(
        args.host,
        args.port,
        args.directory,
        max_attempts=args.max_attempts,
    )
    display_host = "localhost" if args.host in {"0.0.0.0", "127.0.0.1", "::"} else args.host
    if selected_port != args.port:
        print(f"Port {args.port} is occupied; selected {selected_port}.", flush=True)
    print(f"Frontend: http://{display_host}:{selected_port}/web/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
