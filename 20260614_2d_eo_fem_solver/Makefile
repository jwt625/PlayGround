PYTHON ?= .venv/bin/python
UV ?= uv
FE_HOST ?= 127.0.0.1
FE_PORT ?= 5173
CONFIG ?= examples/parallel_plate.yaml
GMSH_OUT ?= artifacts/gmsh_preview
GMSH_NAME ?= mesh_preview

.PHONY: help setup frontend solve backend-check gmsh-mesh gmsh-examples gmsh-examples-3d gmsh-test test check

help:
	@echo "EO FEM development targets"
	@echo "  make setup          Install project, backend, and dev dependencies"
	@echo "  make frontend       Serve the browser app at http://$(FE_HOST):$(FE_PORT)/web/"
	@echo "  make solve          Run the Python reference solver (CONFIG=$(CONFIG))"
	@echo "  make backend-check  Verify that the Gmsh and PyMFEM modules import"
	@echo "  make gmsh-mesh      Write a persistent Gmsh mesh for FE loading"
	@echo "  make gmsh-examples  Build advanced 2D mesh-control validation artifacts"
	@echo "  make gmsh-examples-3d Build initial 3D validation artifacts"
	@echo "  make gmsh-test      Generate and validate the tagged Gmsh test mesh"
	@echo "  make test           Run Python and browser test suites"
	@echo "  make check          Run tests, Ruff, and mypy"
	@echo ""
	@echo "No backend API server exists yet; see DevLog 004 milestones 3-5."

setup:
	$(UV) sync --extra dev --extra backend

frontend:
	$(PYTHON) -m eo_fem.dev_server --host $(FE_HOST) --port $(FE_PORT) --directory .

solve:
	$(PYTHON) -m eo_fem $(CONFIG)

backend-check:
	$(PYTHON) -c 'import gmsh, mfem.ser as mfem; print("Gmsh:", gmsh.__version__); print("PyMFEM: import OK"); print("MFEM H1:", hasattr(mfem, "H1_FECollection")); print("MFEM H(curl):", hasattr(mfem, "ND_FECollection"))'

gmsh-mesh:
	$(PYTHON) -c 'from eo_fem.backends.gmsh_mesh import generate_legacy_gmsh_mesh; from eo_fem.config import load_config; artifact = generate_legacy_gmsh_mesh(load_config("$(CONFIG)"), "$(GMSH_OUT)", model_name="$(GMSH_NAME)"); print(artifact.mesh_path); print(artifact.tag_map_path)'

gmsh-examples:
	$(MAKE) gmsh-mesh CONFIG=examples/mesh_controls/parallel_plate_proximity.yaml GMSH_OUT=artifacts/mesh_controls/parallel_plate_proximity GMSH_NAME=parallel_plate_proximity
	$(MAKE) gmsh-mesh CONFIG=examples/mesh_controls/two_cylinders_curvature.yaml GMSH_OUT=artifacts/mesh_controls/two_cylinders_curvature GMSH_NAME=two_cylinders_curvature
	$(MAKE) gmsh-mesh CONFIG=examples/mesh_controls/sharp_corner_box.yaml GMSH_OUT=artifacts/mesh_controls/sharp_corner_box GMSH_NAME=sharp_corner_box
	$(MAKE) gmsh-mesh CONFIG=examples/mesh_controls/multi_electrode_hotspots.yaml GMSH_OUT=artifacts/mesh_controls/multi_electrode_hotspots GMSH_NAME=multi_electrode_hotspots

gmsh-examples-3d:
	$(PYTHON) -m eo_fem.backends.gmsh_examples_3d --out artifacts/mesh_controls_3d

gmsh-test:
	$(PYTHON) -m pytest -q tests/test_backend_imports.py tests/test_gmsh_mesh.py

test:
	$(PYTHON) -m pytest -q
	node --test tests-js/*.test.js

check: test
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy
