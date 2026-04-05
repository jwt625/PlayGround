# USGS NMIC Global Minerals And Infrastructure Atlas

Minimal, utilitarian frontend for exploring multiple USGS National Minerals Information Center regional GIS releases in one interactive atlas.

Currently bundled regions:

- Africa
- China
- Indo-Pacific
- Latin America & Caribbean
- Southwest Asia

## Related NMIC Map Items

The China release sits inside the USGS National Minerals Information Center collection:

- Collection: [National Minerals Information Center](https://www.sciencebase.gov/catalog/item/5c8c03e4e4b0938824529f7d)
- Collection JSON: `https://www.sciencebase.gov/catalog/items?parentId=5c8c03e4e4b0938824529f7d&max=200&format=json`

Other relevant GIS / map-oriented ScienceBase items in that collection:

- Africa: [catalog item](https://www.sciencebase.gov/catalog/item/607611a9d34e018b3201cbbf)
- Africa JSON: `https://www.sciencebase.gov/catalog/item/607611a9d34e018b3201cbbf?format=json`
- Latin America and the Caribbean: [catalog item](https://www.sciencebase.gov/catalog/item/5804d720e4b0824b2d1c19c6)
- Latin America and the Caribbean JSON: `https://www.sciencebase.gov/catalog/item/5804d720e4b0824b2d1c19c6?format=json`
- Southwest Asia: [catalog item](https://www.sciencebase.gov/catalog/item/63891269d34ed907bf78e9cc)
- Southwest Asia JSON: `https://www.sciencebase.gov/catalog/item/63891269d34ed907bf78e9cc?format=json`
- China: [catalog item](https://www.sciencebase.gov/catalog/item/64348094d34ee8d4add91365)
- China JSON: `https://www.sciencebase.gov/catalog/item/64348094d34ee8d4add91365?format=json`
- Indo-Pacific: [catalog item](https://www.sciencebase.gov/catalog/item/65caa1aed34ef4b119cb3427)
- Indo-Pacific JSON: `https://www.sciencebase.gov/catalog/item/65caa1aed34ef4b119cb3427?format=json`

The Latin America and the Caribbean item is also a child collection with additional map resources:

- Child collection JSON: `https://www.sciencebase.gov/catalog/items?parentId=5804d720e4b0824b2d1c19c6&max=200&format=json`
- Mineral commodity exporting ports of Latin America and the Caribbean: [catalog item](https://www.sciencebase.gov/catalog/item/58093603e4b0f497e78f3f31)
- Mineral facilities of Latin America and the Caribbean: [catalog item](https://www.sciencebase.gov/catalog/item/5809354ee4b0f497e78f3f02)
- Mineral exploration sites of Latin America and the Caribbean: [catalog item](https://www.sciencebase.gov/catalog/item/58093596e4b0f497e78f3f0e)

## What Is Included

- `viewer/`: static frontend and pre-exported GeoJSON layers used by the map
- `scripts/export_layers.py`: rebuilds the browser-ready multi-region layer bundle from the extracted USGS source datasets
- `pyproject.toml`: Python dependencies for the export pipeline

The checked-in `viewer/data/` files are enough to render the visualization locally without re-running the export.

Current exported bundle:

- 5 regions
- 60 layers
- about 98 MB of static GeoJSON

## Features

- Full-screen interactive map
- Layer groups and visibility toggles
- Per-layer opacity controls
- Fit-to-layer actions
- Monochrome and satellite basemap modes
- Hover tooltips and click details
- Search across loaded visible layers
- Point clustering for denser datasets
- Foldable regional layer groups
- Resizable left control panel
- Draggable, foldable legend
- Light and dark mode toggle

## Run The Viewer

From this directory:

```bash
python -m http.server 8123 --directory viewer
```

Open `http://127.0.0.1:8123/`.

## Rebuild The Layer Bundle

The export script currently expects the extracted USGS source data to exist under:

- `extracted/607611a9d34e018b3201cbbf/gdb/Africa_GIS.gdb`
- `extracted/64348094d34ee8d4add91365/CHN_GIS_gdb/CHN_GIS.gdb`
- `extracted/64348094d34ee8d4add91365/data_level_metadata/SI_CHN_GIS_Data-Level_Metadata/`
- `extracted/65caa1aed34ef4b119cb3427/gdb/INDOPAC_GIS.gdb`
- `extracted/65caa1aed34ef4b119cb3427/metadata/`
- `extracted/63891269d34ed907bf78e9cc/gdb/SWAsia_GIS.gdb`
- `extracted/63891269d34ed907bf78e9cc/metadata/`
- `extracted/5804d720e4b0824b2d1c19c6/MINFAC_LAC.csv`
- `extracted/5804d720e4b0824b2d1c19c6/EXPLORE_LAC.csv`
- `extracted/5804d720e4b0824b2d1c19c6/PORTS_LAC.csv`

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
- The checked-in atlas includes the mineral and related infrastructure layers for the five regions above.
- Some generic Africa context layers that duplicate the basemap experience were intentionally omitted from the exported viewer bundle to keep the static app practical in size.
- Raw `downloads/`, extracted source data, and `.venv/` are local working artifacts and are not required to render the checked-in visualization.
- The browser may briefly cache an older `manifest.json` after changes; a hard refresh fixes that.
