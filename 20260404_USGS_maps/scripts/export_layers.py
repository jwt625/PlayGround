from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyogrio import list_layers, read_dataframe, read_info


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "viewer/data"
LAYERS_DIR = OUTPUT_DIR / "layers"

REGIONS = [
    {
        "slug": "africa",
        "name": "Africa",
        "type": "gdb",
        "gdb_path": ROOT / "extracted/607611a9d34e018b3201cbbf/gdb/Africa_GIS.gdb",
        "metadata_dir": None,
        "prefix": "AFR_",
    },
    {
        "slug": "china",
        "name": "China",
        "type": "gdb",
        "gdb_path": ROOT / "extracted/64348094d34ee8d4add91365/CHN_GIS_gdb/CHN_GIS.gdb",
        "metadata_dir": ROOT / "extracted/64348094d34ee8d4add91365/data_level_metadata/SI_CHN_GIS_Data-Level_Metadata",
        "prefix": "CHN_",
    },
    {
        "slug": "indo-pacific",
        "name": "Indo-Pacific",
        "type": "gdb",
        "gdb_path": ROOT / "extracted/65caa1aed34ef4b119cb3427/gdb/INDOPAC_GIS.gdb",
        "metadata_dir": ROOT / "extracted/65caa1aed34ef4b119cb3427/metadata",
        "prefix": "INDOPAC_",
    },
    {
        "slug": "latin-america-caribbean",
        "name": "Latin America & Caribbean",
        "type": "csv",
        "layers": [
            {
                "id": "LAC_Mineral_Exploration",
                "title": "Mineral Exploration",
                "summary": "Mineral exploration sites of Latin America and the Caribbean.",
                "path": ROOT / "extracted/5804d720e4b0824b2d1c19c6/EXPLORE_LAC.csv",
                "encoding": "latin-1",
                "lat_field": "DDLAT",
                "lon_field": "DDLONG",
            },
            {
                "id": "LAC_Mineral_Facilities",
                "title": "Mineral Facilities",
                "summary": "Mineral facilities of Latin America and the Caribbean.",
                "path": ROOT / "extracted/5804d720e4b0824b2d1c19c6/MINFAC_LAC.csv",
                "encoding": "latin-1",
                "lat_field": "DDLAT",
                "lon_field": "DDLONG",
            },
            {
                "id": "LAC_Infra_Transport_Ports",
                "title": "Transport Ports",
                "summary": "Mineral commodity exporting ports of Latin America and the Caribbean.",
                "path": ROOT / "extracted/5804d720e4b0824b2d1c19c6/PORTS_LAC.csv",
                "encoding": "latin-1",
                "lat_field": "DDLAT",
                "lon_field": "DDLONG",
            },
        ],
    },
    {
        "slug": "southwest-asia",
        "name": "Southwest Asia",
        "type": "gdb",
        "gdb_path": ROOT / "extracted/63891269d34ed907bf78e9cc/gdb/SWAsia_GIS.gdb",
        "metadata_dir": ROOT / "extracted/63891269d34ed907bf78e9cc/metadata",
        "prefix": "SWA_",
    },
]

DEFAULT_VISIBLE = {
    "AFR_Mineral_Facilities",
    "CHN_Mineral_Facilities",
    "INDOPAC_Mineral_Facilities",
    "LAC_Mineral_Facilities",
    "SWA_Mineral_Facilties",
}

SKIP_LAYERS = {
    "AFR_Infra_Transport_Road",
    "AFR_Political_Cities",
    "AFR_NaturalFeatures_Rivers",
    "AFR_NaturalFeatures_Lakes",
    "AFR_Political_ADM0_Boundaries",
    "AFR_Political_ADM1_Boundaries",
}

PREFERRED_FIELDS = [
    "FeatureUID",
    "Label",
    "Label1",
    "FeatureNam",
    "PROJNAME",
    "LOCNAME",
    "PORTNAME",
    "NameVar",
    "FeatureTyp",
    "FACTYPE",
    "PROJTYPE",
    "Country",
    "COUNTRY",
    "ADM1",
    "MemoLoc",
    "LOCDESC",
    "LocOpStat",
    "STATUS",
    "OperateNam",
    "OPERATOR",
    "OwnerName1",
    "OWNER",
    "DsgAttr01",
    "DsgAttr02",
    "DsgAttr03",
    "DsgAttr04",
    "COMMODITY",
    "COMM",
    "COMM_EXPD",
    "FORM_COMM",
    "Latitude",
    "Longitude",
    "DDLAT",
    "DDLONG",
    "LocConfid",
    "LocConf",
    "LOCACC",
    "InfSource1",
    "SOURCEID",
    "Shape_Length",
    "Shape_Area",
]

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
    "SWA_Mineral_Resources_Coal": 0.02,
    "SWA_Mineral_Resources_Phosphate": 0.012,
    "SWA_Mineral_Resources_Copper": 0.01,
    "SWA_Mineral_Resources_Potash": 0.01,
    "SWA_OG_Provinces_Continuous": 0.015,
    "SWA_OG_Provinces_Conventional": 0.015,
    "SWA_OG_Resources_Recoverable": 0.015,
    "SWA_Infra_Power_Transmission": 0.0025,
    "INDOPAC_Mineral_Resources_Coal": 0.02,
    "INDOPAC_Mineral_Resources_Copper": 0.01,
    "INDOPAC_OG_Provinces_Continuous": 0.015,
    "INDOPAC_OG_Provinces_Conventional": 0.015,
    "INDOPAC_OG_Resources_Recoverable": 0.015,
    "AFR_Mineral_Resources_Coal": 0.02,
    "AFR_Mineral_Resources_Copper": 0.01,
    "AFR_Mineral_Resources_Gabon": 0.01,
    "AFR_Mineral_Resources_Mauritania": 0.01,
    "AFR_Mineral_Resources_PGE": 0.01,
    "AFR_Mineral_Resources_Potash": 0.01,
    "AFR_OG_Provinces_Continuous": 0.015,
    "AFR_OG_Provinces_Conventional": 0.015,
    "AFR_OG_Resources_Recoverable": 0.015,
    "AFR_Infra_Power_Transmission": 0.0025,
    "AFR_Infra_OG_Pipelines": 0.004,
    "AFR_Infra_Transport_Rail": 0.004,
    "AFR_Infra_Transport_Road": 0.004,
    "AFR_NaturalFeatures_Rivers": 0.004,
    "AFR_Political_ADM0_Boundaries": 0.01,
    "AFR_Political_ADM1_Boundaries": 0.01,
}

TOKEN_REPLACEMENTS = {
    "ADM0": "Admin 0",
    "ADM1": "Admin 1",
    "AFR": "Africa",
    "CHN": "China",
    "SWA": "Southwest Asia",
    "INDOPAC": "Indo-Pacific",
    "OG": "Oil & Gas",
    "PGE": "PGE",
    "LNG": "LNG",
}


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
    if not text or text in {"<null>", "nan", "None"}:
        return None
    return text


def pick_fields(frame: gpd.GeoDataFrame) -> list[str]:
    fields = [column for column in frame.columns if column != "geometry"]
    selected = [field for field in PREFERRED_FIELDS if field in fields]

    if len(selected) < 10:
        extras = [field for field in fields if field not in selected]
        selected.extend(extras[: 10 - len(selected)])

    return selected[:14]


def find_metadata_file(metadata_dir: Path | None, layer_name: str) -> Path | None:
    if not metadata_dir or not metadata_dir.exists():
        return None

    candidates = [f"{layer_name}.xml", f"{layer_name}_metadata.xml"]
    for candidate in candidates:
        matches = list(metadata_dir.rglob(candidate))
        if matches:
            return matches[0]
    return None


def simplify_title(text: str, region_name: str) -> str:
    cleaned = re.sub(r"\((GIS Feature Class|GIS Layer)\)", "", text, flags=re.I)
    cleaned = cleaned.replace(region_name, "").replace("  ", " ").strip(" -")
    return " ".join(cleaned.split())


def humanize_layer_name(layer_name: str, prefix: str = "") -> str:
    core = layer_name[len(prefix) :] if prefix and layer_name.startswith(prefix) else layer_name
    tokens = [TOKEN_REPLACEMENTS.get(token, token) for token in core.split("_")]
    return " ".join(tokens)


def parse_metadata(layer_name: str, region_name: str, metadata_dir: Path | None, prefix: str) -> tuple[str, str]:
    xml_path = find_metadata_file(metadata_dir, layer_name)
    if not xml_path:
        return humanize_layer_name(layer_name, prefix), ""

    tree = ET.parse(xml_path)
    title = tree.findtext(".//title") or humanize_layer_name(layer_name, prefix)
    purpose = tree.findtext(".//purpose") or tree.findtext(".//abstract") or ""
    return simplify_title(" ".join(title.split()), region_name), " ".join(purpose.split())


def normalize_geometry_type(value: str) -> str:
    lowered = value.lower()
    if "point" in lowered:
        return "Point"
    if "line" in lowered:
        return "MultiLineString"
    if "polygon" in lowered:
        return "MultiPolygon"
    return value


def infer_color(layer_id: str) -> str:
    lowered = layer_id.lower()
    if "facilit" in lowered:
        return "#0f766e"
    if "development" in lowered:
        return "#047857"
    if "exploration" in lowered:
        return "#059669"
    if "deposit" in lowered:
        return "#115e59"
    if "resources" in lowered:
        return "#8b5e34"
    if "power_stations" in lowered:
        return "#d97706"
    if "power_transmission" in lowered:
        return "#7c3aed"
    if "transport_ports" in lowered:
        return "#2563eb"
    if "transport_rail" in lowered:
        return "#4f46e5"
    if "transport_road" in lowered:
        return "#ea580c"
    if "lakes" in lowered:
        return "#0ea5e9"
    if "rivers" in lowered:
        return "#0284c7"
    if "cities" in lowered:
        return "#64748b"
    if "political" in lowered:
        return "#6b7280"
    if "lng" in lowered or "pipelines" in lowered:
        return "#0891b2"
    if "recoverable" in lowered or "provinces" in lowered:
        return "#be123c"
    return "#334155"


def feature_title(properties: dict[str, object], layer_name: str) -> str:
    for key in ("FeatureNam", "PROJNAME", "LOCNAME", "PORTNAME", "Label", "Label1", "FeatureUID", "ADM1", "Country", "COUNTRY"):
        value = clean_value(properties.get(key))
        if value:
            return str(value)
    return layer_name.replace("_", " ")


def feature_subtitle(properties: dict[str, object]) -> str:
    bits = []
    for key in (
        "FeatureTyp",
        "FACTYPE",
        "PROJTYPE",
        "DsgAttr01",
        "DsgAttr02",
        "COMMODITY",
        "COMM",
        "COMM_EXPD",
        "Country",
        "COUNTRY",
    ):
        value = clean_value(properties.get(key))
        if value and value not in bits:
            bits.append(str(value))
    return " · ".join(bits[:3])


def prepare_features(frame: gpd.GeoDataFrame, layer_name: str, region_name: str) -> gpd.GeoDataFrame:
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

        terms = [region_name, title, subtitle]
        terms.extend(str(value) for value in props.values() if value)
        search_terms.append(" ".join(dict.fromkeys(term for term in terms if term)))

    prepared["title"] = titles
    prepared["subtitle"] = subtitles
    prepared["searchText"] = search_terms

    for field in fields:
        prepared[field] = prepared[field].map(clean_value)

    return prepared


def export_gdb_region(region: dict[str, object]) -> list[dict[str, object]]:
    layers = []
    gdb_path = region["gdb_path"]
    metadata_dir = region.get("metadata_dir")
    prefix = region.get("prefix", "")

    for layer_name, geometry_type in list_layers(gdb_path):
        layer_name = str(layer_name)
        if layer_name in SKIP_LAYERS:
            continue
        geometry_type = normalize_geometry_type(str(geometry_type))
        title, purpose = parse_metadata(layer_name, region["name"], metadata_dir, prefix)
        info = read_info(gdb_path, layer=layer_name)
        frame = read_dataframe(gdb_path, layer=layer_name)
        prepared = prepare_features(frame, layer_name, region["name"])

        output_path = LAYERS_DIR / f"{slugify(layer_name)}.geojson"
        output_path.write_text(prepared.to_json(drop_id=True), encoding="utf-8")

        layers.append(
            {
                "id": layer_name,
                "slug": slugify(layer_name),
                "title": title,
                "summary": purpose or f"{region['name']} · {title}",
                "group": region["name"],
                "region": region["name"],
                "geometryType": geometry_type,
                "featureCount": int(info["features"]),
                "crs": str(info["crs"]),
                "color": infer_color(layer_name),
                "visibleByDefault": layer_name in DEFAULT_VISIBLE,
                "source": f"./data/layers/{slugify(layer_name)}.geojson",
            }
        )

    return layers


def export_csv_region(region: dict[str, object]) -> list[dict[str, object]]:
    layers = []
    for layer in region["layers"]:
        frame = pd.read_csv(layer["path"], encoding=layer.get("encoding", "utf-8"))
        frame[layer["lat_field"]] = pd.to_numeric(frame[layer["lat_field"]], errors="coerce")
        frame[layer["lon_field"]] = pd.to_numeric(frame[layer["lon_field"]], errors="coerce")
        frame = frame[frame[layer["lat_field"]].notna() & frame[layer["lon_field"]].notna()].copy()

        geo = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame[layer["lon_field"]], frame[layer["lat_field"]]),
            crs="EPSG:4326",
        )
        geo["Latitude"] = frame[layer["lat_field"]]
        geo["Longitude"] = frame[layer["lon_field"]]

        prepared = prepare_features(geo, layer["id"], region["name"])
        output_path = LAYERS_DIR / f"{slugify(layer['id'])}.geojson"
        output_path.write_text(prepared.to_json(drop_id=True), encoding="utf-8")

        layers.append(
            {
                "id": layer["id"],
                "slug": slugify(layer["id"]),
                "title": layer["title"],
                "summary": layer["summary"],
                "group": region["name"],
                "region": region["name"],
                "geometryType": "Point",
                "featureCount": int(len(prepared)),
                "crs": "EPSG:4326",
                "color": infer_color(layer["id"]),
                "visibleByDefault": layer["id"] in DEFAULT_VISIBLE,
                "source": f"./data/layers/{slugify(layer['id'])}.geojson",
            }
        )

    return layers


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)

    for path in LAYERS_DIR.glob("*.geojson"):
        path.unlink()

    layers: list[dict[str, object]] = []
    for region in REGIONS:
        if region["type"] == "gdb":
            layers.extend(export_gdb_region(region))
        else:
            layers.extend(export_csv_region(region))

    manifest = {
        "title": "USGS NMIC Global Minerals and Infrastructure Atlas",
        "defaultCenter": [35.0, 18.0],
        "defaultZoom": 1.65,
        "layers": sorted(layers, key=lambda item: (item["group"], item["title"])),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
