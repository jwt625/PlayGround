from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from eo_fem.dev_server import create_server


def test_create_server_scans_past_occupied_port(tmp_path) -> None:
    occupied = ThreadingHTTPServer(("127.0.0.1", 0), None)
    occupied_port = int(occupied.server_address[1])
    server = None
    try:
        server, selected_port = create_server("127.0.0.1", occupied_port, tmp_path, max_attempts=100)
        assert selected_port > occupied_port
    finally:
        occupied.server_close()
        if server is not None:
            server.server_close()


def test_development_server_disables_browser_caching(tmp_path) -> None:
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    server, port = create_server("127.0.0.1", 0, tmp_path)
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:  # noqa: S310 - loopback test server
            assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
            assert response.read() == b"ok"
    finally:
        server.server_close()
        thread.join(timeout=2)


def test_api_lists_configs_and_mesh_artifacts(tmp_path) -> None:
    config = tmp_path / "examples" / "mesh_controls" / "demo.yaml"
    mesh = tmp_path / "artifacts" / "mesh_controls" / "demo" / "demo.msh"
    config.parent.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    config.write_text("Domain: {}", encoding="utf-8")
    mesh.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    server, port = create_server("127.0.0.1", 0, tmp_path)
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/examples", timeout=2) as response:  # noqa: S310
            payload = json.loads(response.read())
        assert [entry["kind"] for entry in payload["examples"]] == ["config", "mesh"]
        assert payload["examples"][1]["url"] == "/artifacts/mesh_controls/demo/demo.msh"
    finally:
        server.server_close()
        thread.join(timeout=2)


def test_api_generates_mesh_from_example_config(tmp_path) -> None:
    pytest.importorskip("gmsh")
    config = tmp_path / "examples" / "demo.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """Simulation:
  mesh_nx: 9
  mesh_ny: 7
Domain:
  x_min: -2
  x_max: 2
  y_min: -2
  y_max: 2
Materials:
  background:
    eps_r: 1
Electrodes:
  signal:
    shape: circle
    potential: 1
    x: -0.7
    y: 0
    radius: 0.2
  ground:
    shape: circle
    potential: 0
    x: 0.7
    y: 0
    radius: 0.2
""",
        encoding="utf-8",
    )
    server, port = create_server("127.0.0.1", 0, tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{port}/api/mesh/generate",
        data=json.dumps({"config_path": "examples/demo.yaml", "name": "api_demo"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read())
        assert response.status == 201
        assert payload["node_count"] > 0
        assert (tmp_path / payload["mesh_url"].lstrip("/")).is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
