from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
from pyogrio import list_layers, read_dataframe, read_info


ROOT = Path(__file__).resolve().parents[1]
GDB_PATH = ROOT / "extracted/64348094d34ee8d4add91365/CHN_GIS_gdb/CHN_GIS.gdb"
METADATA_DIR = ROOT / "extracted/64348094d34ee8d4add91365/data_level_metadata/SI_CHN_GIS_Data-Level_Metadata"
OUTPUT_DIR = ROOT / "viewer/data"
LAYERS_DIR = OUTPUT_DIR / "layers"

GROUPS = {
    "CHN_Mineral_Facilities": "Minerals",
    "CHN_Mineral_Exploration": "Minerals",
    "CHN_Mineral_Deposits": "Minerals",
    "CHN_Mineral_Resources_Antimony": "Resources",
    "CHN_Mineral_Resources_Coal": "Resources",
    "CHN_Mineral_Resources_Copper": "Resources",
    "CHN_Mineral_Resources_Phosphate": "Resources",
    "CHN_Mineral_Resources_Potash": "Resources",
    "CHN_Infra_Dams": "Infrastructure",
    "CHN_Infra_Power_Stations": "Infrastructure",
    "CHN_Infra_Power_Transmission": "Infrastructure",
    "CHN_Infra_OG_LNG_Terminals": "Infrastructure",
    "CHN_Infra_Transport_Ports": "Infrastructure",
    "CHN_OG_Provinces_Continuous": "Oil & Gas",
    "CHN_OG_Province_Groups_Conventional": "Oil & Gas",
    "CHN_OG_Provinces_Conventional": "Oil & Gas",
    "CHN_OG_Resources_Recoverable": "Oil & Gas",
}

COLORS = {
    "CHN_Mineral_Facilities": "#0f766e",
    "CHN_Mineral_Exploration": "#0f766e",
    "CHN_Mineral_Deposits": "#115e59",
    "CHN_Mineral_Resources_Antimony": "#92400e",
    "CHN_Mineral_Resources_Coal": "#4b5563",
    "CHN_Mineral_Resources_Copper": "#b45309",
    "CHN_Mineral_Resources_Phosphate": "#6b8e23",
    "CHN_Mineral_Resources_Potash": "#9a3412",
    "CHN_Infra_Dams": "#1d4ed8",
    "CHN_Infra_Power_Stations": "#f59e0b",
    "CHN_Infra_Power_Transmission": "#7c3aed",
    "CHN_Infra_OG_LNG_Terminals": "#0ea5e9",
    "CHN_Infra_Transport_Ports": "#2563eb",
    "CHN_OG_Provinces_Continuous": "#be123c",
    "CHN_OG_Province_Groups_Conventional": "#e11d48",
    "CHN_OG_Provinces_Conventional": "#fb7185",
    "CHN_OG_Resources_Recoverable": "#7f1d1d",
}

SIMPLIFY_TOLERANCE = {
    "CHN_Mineral_Resources_Coal": 0.02,
    "CHN_Mineral_Resources_Phosphate": 0.012,
    "CHN_Mineral_Resources_Copper": 0.01,
    "CHN_Mineral_Resources_Antimony": 0.01,
    "CHN_Mineral_Resources_Potash": 0.01,
    "CHN_OG_Provinces_Continuous": 0.015,
    "CHN_OG_Province_Groups_Conventional": 0.015,
    "CHN_OG_Provinces_Conventional": 0.015,
    "CHN_OG_Resources_Recoverable": 0.015,
    "CHN_Infra_Power_Transmission": 0.0025,
}

DEFAULT_VISIBLE = {
    "CHN_Mineral_Facilities",
    "CHN_Mineral_Deposits",
    "CHN_Infra_Power_Stations",
    "CHN_Infra_Transport_Ports",
    "CHN_Mineral_Resources_Coal",
    "CHN_OG_Resources_Recoverable",
}

PREFERRED_FIELDS = [
    "FeatureUID",
    "FeatureNam",
    "FeatureTyp",
    "Country",
    "ADM1",
    "MemoLoc",
    "LocOpStat",
    "OperateNam",
    "OwnerName1",
    "DsgAttr01",
    "DsgAttr02",
    "DsgAttr03",
    "DsgAttr04",
    "Latitude",
    "Longitude",
    "LocConfid",
    "InfSource1",
    "LocSource1",
    "Shape_Length",
    "Shape_Area",
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    if hasattr(value, "item"):
        return clean_value(value.item())
    text = str(value).strip()
    return text or None


def pick_fields(frame: gpd.GeoDataFrame) -> list[str]:
    fields = [column for column in frame.columns if column != "geometry"]
    selected = [field for field in PREFERRED_FIELDS if field in fields]

    if not selected:
        selected = fields[:12]

    if len(selected) < 8:
        extras = [field for field in fields if field not in selected]
        selected.extend(extras[: 8 - len(selected)])

    return selected[:12]


def parse_metadata(layer_name: str) -> tuple[str, str]:
    xml_path = METADATA_DIR / f"{layer_name}.xml"
    if not xml_path.exists():
        return layer_name, ""

    tree = ET.parse(xml_path)
    title = tree.findtext(".//title") or layer_name
    purpose = tree.findtext(".//purpose") or ""
    return " ".join(title.split()), " ".join(purpose.split())


def feature_title(properties: dict[str, object], layer_name: str) -> str:
    for key in ("FeatureNam", "Label", "FeatureUID", "ADM1"):
        value = clean_value(properties.get(key))
        if value:
            return str(value)
    return layer_name.replace("_", " ")


def feature_subtitle(properties: dict[str, object]) -> str:
    bits = []
    for key in ("FeatureTyp", "DsgAttr01", "DsgAttr02", "ADM1"):
        value = clean_value(properties.get(key))
        if value and value not in bits:
            bits.append(str(value))
    return " · ".join(bits[:3])


def prepare_features(frame: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    fields = pick_fields(frame)
    prepared = frame[fields + ["geometry"]].copy()

    if prepared.crs and str(prepared.crs) != "EPSG:4326":
        prepared = prepared.to_crs(4326)

    tolerance = SIMPLIFY_TOLERANCE.get(layer_name)
    if tolerance:
        prepared["geometry"] = prepared.geometry.simplify(tolerance, preserve_topology=True)

    prepared = prepared[prepared.geometry.notnull()].copy()
    prepared["geometry"] = prepared.geometry.make_valid()

    titles = []
    subtitles = []
    search_terms = []

    for _, row in prepared.iterrows():
        props = {field: clean_value(row[field]) for field in fields}
        title = feature_title(props, layer_name)
        subtitle = feature_subtitle(props)
        titles.append(title)
        subtitles.append(subtitle)

        terms = [title, subtitle]
        terms.extend(str(value) for value in props.values() if value)
        search_terms.append(" ".join(dict.fromkeys(term for term in terms if term)))

    prepared["title"] = titles
    prepared["subtitle"] = subtitles
    prepared["searchText"] = search_terms

    for field in fields:
        prepared[field] = prepared[field].map(clean_value)

    return prepared


def export_layer(layer_name: str, geometry_type: str) -> dict[str, object]:
    title, purpose = parse_metadata(layer_name)
    info = read_info(GDB_PATH, layer=layer_name)
    frame = read_dataframe(GDB_PATH, layer=layer_name)
    prepared = prepare_features(frame, layer_name)

    output_path = LAYERS_DIR / f"{slugify(layer_name)}.geojson"
    output_path.write_text(prepared.to_json(drop_id=True), encoding="utf-8")

    return {
        "id": layer_name,
        "slug": slugify(layer_name),
        "title": title,
        "summary": purpose,
        "group": GROUPS.get(layer_name, "Other"),
        "geometryType": geometry_type,
        "featureCount": int(info["features"]),
        "crs": str(info["crs"]),
        "color": COLORS.get(layer_name, "#111827"),
        "visibleByDefault": layer_name in DEFAULT_VISIBLE,
        "source": f"./data/layers/{slugify(layer_name)}.geojson",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)

    layers = []
    for layer_name, geometry_type in list_layers(GDB_PATH):
        layers.append(export_layer(str(layer_name), str(geometry_type)))

    manifest = {
        "title": "USGS China Minerals and Infrastructure Atlas",
        "defaultCenter": [104.2, 35.8],
        "defaultZoom": 3.4,
        "layers": layers,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
