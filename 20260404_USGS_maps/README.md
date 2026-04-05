# USGS China Minerals And Infrastructure Atlas

Minimal, utilitarian frontend for exploring the USGS geospatial data release for the mineral industries and related infrastructure of the People's Republic of China.

## What Is Included

- `viewer/`: static frontend and pre-exported GeoJSON layers used by the map
- `scripts/export_layers.py`: rebuilds the browser-ready layer bundle from the extracted USGS file geodatabase
- `pyproject.toml`: Python dependencies for the export pipeline

The checked-in `viewer/data/` files are enough to render the visualization locally without re-running the export.

## Features

- Full-screen interactive map
- Layer groups and visibility toggles
- Per-layer opacity controls
- Fit-to-layer actions
- Monochrome and satellite basemap modes
- Hover tooltips and click details
- Search across loaded visible layers
- Point clustering for denser datasets

## Run The Viewer

From this directory:

```bash
python -m http.server 8123 --directory viewer
```

Open `http://127.0.0.1:8123/`.

## Rebuild The Layer Bundle

The export script expects the extracted USGS geodatabase and metadata to exist under:

- `extracted/64348094d34ee8d4add91365/CHN_GIS_gdb/CHN_GIS.gdb`
- `extracted/64348094d34ee8d4add91365/data_level_metadata/SI_CHN_GIS_Data-Level_Metadata/`

Set up the environment with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python scripts/export_layers.py
```

If you already have the project virtualenv, only the last command is needed to regenerate `viewer/data/`.

## Notes

- The viewer is intentionally static: no backend is required.
- Raw `downloads/`, extracted source data, and `.venv/` are local working artifacts and are not required to render the checked-in visualization.
- The browser may briefly cache an older `manifest.json` after changes; a hard refresh fixes that.
