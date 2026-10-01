"""Build the app data view (app/static/data/atlas.json) and an optional flat CSV from the database.

Reads <data>/papers.csv, devices.csv, organizations.csv, evidence/*.yaml and <data>/schema/devices.schema.yaml.
`--with-staging DIR ...` merges staged fragments (same layout as data/) into a temp copy via merge_staging.merge;
the data directory itself is never modified.

Derived per-device quantities (all flagged as derived in the output, never written back to the CSV):
  vpil_dc_vcm_derived  = vpi_dc_v * length_mm / 10                       [V*cm]   only when vpil_dc_vcm is empty
  il_rf_total_db       = rf_loss_db_per_cm * length_mm / 10              [dB]
  vpi_il_vdb           = Vpi_ref * il_onchip_db                          [V*dB]   Vpi_ref = vpi_dc_v, else vpi_rf_v
  completeness         = reported core fields / 7   (vpi any, bw3db, il_onchip, rf_loss, z0, n_rf, ng)
  fom                  = f3dB[GHz] / (Vpi_eff[V] * 10^(IL_onchip[dB]/10))
                         Vpi_eff = Vpi_ref * 10^(il_rf_total_db/20) when il_rf_total_db is known, else Vpi_ref

Usage:
  uv run python scripts/build_views.py [--data data] [--with-staging DIR ...] [--out app/static/data/atlas.json]
                                       [--flat-csv PATH] [--sims sims]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_staging import merge  # noqa: E402
from validate_db import read_csv, split_list  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FOM_CLASSES = {"mzm", "phase_shifter", "iq_mzm", "plasmonic_mzm"}
VPI_FIELDS = ["vpi_dc_v", "vpi_rf_v", "vpil_dc_vcm", "vpil_rf_vcm", "vpi_mzm_pushpull_dc_v"]
CORE_FIELDS: list[tuple[str, list[str]]] = [
    ("vpi", VPI_FIELDS),
    ("bw3db", ["bw3db_ghz"]),
    ("il_onchip", ["il_onchip_db"]),
    ("rf_loss", ["rf_loss_db_per_cm"]),
    ("z0", ["z0_ohm"]),
    ("n_rf", ["n_rf"]),
    ("ng", ["ng_opt"]),
]
SIM_BASES = {"simulated", "predicted", "design_target"}

LABEL_OVERRIDES = {
    "mzm": "MZM",
    "iq_mzm": "IQ MZM",
    "eam": "EAM",
    "plasmonic_mzm": "Plasmonic MZM",
    "inp_mqw": "InP MQW",
    "algaas_gaas": "AlGaAs / GaAs",
    "ktn": "KTN",
    "pzt": "PZT",
    "plzt": "PLZT",
    "inp": "InP",
    "algaas": "AlGaAs",
    "sin": "SiN",
    "soi_slot": "SOI slot",
    "soi_rib": "SOI rib",
    "lnoi_rib": "LNOI rib",
    "lnoi_loaded_sin": "LNOI loaded SiN",
    "lnoi_loaded_si": "LNOI loaded Si",
    "ltoi_rib": "LTOI rib",
    "ltoi_loaded_sin": "LTOI loaded SiN",
    "bto_on_si": "BTO on Si",
    "bto_on_sin": "BTO on SiN",
    "bto_on_oxide_substrate": "BTO on oxide substrate",
    "plasmonic_mim": "Plasmonic MIM",
    "suspended_lnoi": "Suspended LNOI",
    "cl_twe": "Capacitively loaded TWE",
    "tw_cpw": "Traveling-wave CPW",
    "tw_cps": "Traveling-wave CPS",
    "tw_gsg": "Traveling-wave GSG",
    "o_band": "O-band",
    "c_band": "C-band",
    "l_band": "L-band",
    "cl_band": "C+L band",
    "one_um": "1 um band",
    "visible_nir": "Visible / NIR",
    "mid_ir": "Mid-IR",
    "bw_reference": "BW reference",
    "dc": "DC",
    "1ghz": "1 GHz",
    "10ghz": "10 GHz",
    "arxiv_preprint": "arXiv preprint",
    "arxiv": "arXiv",
    "open_license_ok": "Open license, redistribution OK",
    "restricted_local_only": "Restricted, local only",
    "mzm_push_pull": "MZM push-pull",
    "mzm_series_push_pull": "MZM series push-pull",
    "mzm_differential": "MZM differential",
    "mzm_single_arm": "MZM single arm",
    "per_arm_phase_shifter": "Per-arm phase shifter",
    "resonance_tuning_derived": "Resonance-tuning derived",
    "extracted_from_figure": "Extracted from figure",
    "design_target": "Design target",
    "author_estimate": "Author estimate",
    "silicon_plasma_dispersion": "Silicon plasma dispersion",
    "germanium_silicon_eam": "Germanium-silicon EAM",
    "graphene_2d": "Graphene / 2D",
    "ferroelectric_nematic_lc": "Ferroelectric nematic LC",
    "strontium_titanate": "Strontium titanate",
    "eo_polymer": "EO polymer",
    "tfln": "TFLN",
    "single_ended": "Single-ended",
    "push_pull": "Push-pull",
    "series_push_pull": "Series push-pull",
    "dual_drive": "Dual drive",
    "north_america": "North America",
    "east_asia": "East Asia",
    "south_asia": "South Asia",
    "southeast_asia": "Southeast Asia",
    "middle_east": "Middle East",
    "national_lab": "National laboratory",
    "research_institute": "Research institute",
    "eo_material": "EO material",
}

REGIONS = ["north_america", "europe", "east_asia", "south_asia", "southeast_asia", "oceania", "middle_east", "other"]
ORG_TYPES = [
    "university",
    "company",
    "national_lab",
    "research_institute",
    "foundry",
    "facility",
    "consortium",
    "other",
]

COUNTRY_NAMES = {
    "US": "United States",
    "CA": "Canada",
    "MX": "Mexico",
    "BR": "Brazil",
    "CH": "Switzerland",
    "DE": "Germany",
    "FR": "France",
    "GB": "United Kingdom",
    "NL": "Netherlands",
    "BE": "Belgium",
    "IT": "Italy",
    "ES": "Spain",
    "PT": "Portugal",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "IE": "Ireland",
    "AT": "Austria",
    "PL": "Poland",
    "CZ": "Czechia",
    "GR": "Greece",
    "RU": "Russia",
    "IL": "Israel",
    "TR": "Turkey",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
    "JP": "Japan",
    "CN": "China",
    "HK": "Hong Kong",
    "TW": "Taiwan",
    "KR": "South Korea",
    "IN": "India",
    "SG": "Singapore",
    "MY": "Malaysia",
    "TH": "Thailand",
    "VN": "Vietnam",
    "AU": "Australia",
    "NZ": "New Zealand",
    "ZA": "South Africa",
    "LU": "Luxembourg",
    "HU": "Hungary",
    "RO": "Romania",
    "UA": "Ukraine",
    "LT": "Lithuania",
    "EE": "Estonia",
}

COLUMN_GROUPS = {
    "identity": [
        "device_id",
        "paper_id",
        "device_label",
        "device_class",
        "tags",
        "eo_material",
        "waveguide_platform",
        "integration",
        "electrode_type",
        "drive",
        "vpi_convention",
        "temperature_class",
        "band",
    ],
    "operating point": ["wavelength_nm", "length_mm"],
    "drive efficiency": [
        "vpi_dc_v",
        "vpi_rf_v",
        "vpi_rf_freq_ghz",
        "vpil_dc_vcm",
        "vpil_rf_vcm",
        "vpi_mzm_pushpull_dc_v",
        "bias_for_vpi_v",
        "drive_vpp_v",
        "vpi_basis",
    ],
    "bandwidth": [
        "bw3db_ghz",
        "bw3db_reference",
        "bw3db_reference_freq_ghz",
        "bw6db_ghz",
        "bw_measured_to_ghz",
        "bw_basis",
    ],
    "optical loss": [
        "il_onchip_db",
        "il_onchip_includes",
        "il_onchip_excludes",
        "il_fiber_to_fiber_db",
        "prop_loss_db_per_cm",
        "extinction_ratio_db",
        "er_type",
        "optical_input_power_dbm",
        "il_basis",
    ],
    "RF": ["rf_loss_db_per_cm", "rf_loss_freq_ghz", "z0_ohm", "n_rf", "ng_opt", "optical_power_handling_dbm"],
    "geometry": [
        "eo_film_thickness_nm",
        "etch_depth_nm",
        "rib_width_nm",
        "slab_thickness_nm",
        "sidewall_angle_deg",
        "electrode_gap_um",
        "signal_width_um",
        "electrode_thickness_um",
        "electrode_metal",
        "buffer_oxide_um",
        "substrate",
        "cladding",
        "crystal_cut",
        "waveguide_orientation",
        "epitaxy_or_stack",
    ],
    "system demonstration": [
        "max_baud_gbd",
        "modulation_format",
        "max_line_rate_gbps",
        "max_net_rate_gbps",
        "energy_per_bit_fj",
        "capacitance_ff",
        "q_loaded",
        "fsr_nm",
        "tuning_nm_per_v",
        "driver",
    ],
    "bookkeeping": ["qualifiers", "evidence_ref", "notes"],
}


COLUMN_LABELS = {
    "device_id": "Device id",
    "paper_id": "Paper id",
    "device_label": "Device label",
    "device_class": "Device class",
    "eo_material": "EO material",
    "waveguide_platform": "Waveguide platform",
    "vpi_convention": "Vpi convention",
    "temperature_class": "Temperature class",
    "wavelength_nm": "Wavelength",
    "length_mm": "Length",
    "vpi_dc_v": "Vpi (DC)",
    "vpi_rf_v": "Vpi (RF)",
    "vpi_rf_freq_ghz": "Vpi (RF) frequency",
    "vpil_dc_vcm": "Vpi*L (DC)",
    "vpil_rf_vcm": "Vpi*L (RF)",
    "vpi_mzm_pushpull_dc_v": "Vpi (MZM push-pull, DC)",
    "bias_for_vpi_v": "Bias for Vpi",
    "drive_vpp_v": "Drive amplitude",
    "vpi_basis": "Vpi basis",
    "bw3db_ghz": "3 dB bandwidth",
    "bw3db_reference": "3 dB BW reference",
    "bw3db_reference_freq_ghz": "3 dB BW reference frequency",
    "bw6db_ghz": "6 dB bandwidth",
    "bw_measured_to_ghz": "Bandwidth measured to",
    "bw_basis": "Bandwidth basis",
    "il_onchip_db": "On-chip insertion loss",
    "il_onchip_includes": "On-chip IL includes",
    "il_onchip_excludes": "On-chip IL excludes",
    "il_fiber_to_fiber_db": "Fiber-to-fiber insertion loss",
    "prop_loss_db_per_cm": "Propagation loss",
    "extinction_ratio_db": "Extinction ratio",
    "er_type": "Extinction ratio type",
    "optical_input_power_dbm": "Optical input power",
    "il_basis": "Loss basis",
    "rf_loss_db_per_cm": "RF loss",
    "rf_loss_freq_ghz": "RF loss frequency",
    "z0_ohm": "Z0",
    "n_rf": "RF index n_RF",
    "ng_opt": "Optical group index n_g",
    "optical_power_handling_dbm": "Optical power handling",
    "eo_film_thickness_nm": "EO film thickness",
    "etch_depth_nm": "Etch depth",
    "rib_width_nm": "Rib width",
    "slab_thickness_nm": "Slab thickness",
    "sidewall_angle_deg": "Sidewall angle",
    "electrode_gap_um": "Electrode gap",
    "signal_width_um": "Signal width",
    "electrode_thickness_um": "Electrode thickness",
    "electrode_metal": "Electrode metal",
    "buffer_oxide_um": "Buffer oxide",
    "crystal_cut": "Crystal cut",
    "waveguide_orientation": "Waveguide orientation",
    "epitaxy_or_stack": "Epitaxy / stack",
    "max_baud_gbd": "Max baud rate",
    "modulation_format": "Modulation format",
    "max_line_rate_gbps": "Max line rate",
    "max_net_rate_gbps": "Max net rate",
    "energy_per_bit_fj": "Energy per bit",
    "capacitance_ff": "Capacitance",
    "q_loaded": "Loaded Q",
    "fsr_nm": "Free spectral range",
    "tuning_nm_per_v": "Tuning",
    "evidence_ref": "Evidence file",
}


def label_of(value: str) -> str:
    if value in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[value]
    s = value.replace("_", " ")
    return s[:1].upper() + s[1:]


def to_float(cell: str | None) -> float | None:
    if cell is None or cell.strip() == "":
        return None
    return float(cell)


def typed_row(columns: list[dict[str, Any]], row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        cell = (row.get(col["name"]) or "").strip()
        t = col["type"]
        if cell == "":
            out[col["name"]] = [] if t == "list" else None
        elif t == "float":
            out[col["name"]] = float(cell)
        elif t == "int":
            out[col["name"]] = int(cell)
        elif t == "list":
            out[col["name"]] = split_list(cell)
        else:
            out[col["name"]] = cell
    return out


def load_evidence(data: Path, paper_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """device_id -> field -> {locator, basis, note, unit, derived, formula, inputs}"""
    path = data / "evidence" / f"{paper_id}.yaml"
    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        ev = yaml.safe_load(f) or {}
    for e in ev.get("entries") or []:
        out.setdefault(str(e.get("device_id")), {})[str(e.get("field"))] = {
            "locator": str(e.get("locator", "")),
            "basis": e.get("basis"),
            "note": str(e.get("note", "") or ""),
            "unit": str(e.get("unit", "") or ""),
            "derived": False,
        }
    for e in ev.get("derived") or []:
        out.setdefault(str(e.get("device_id")), {})[str(e.get("field"))] = {
            "locator": "derived",
            "basis": "derived",
            "note": "",
            "unit": "",
            "derived": True,
            "formula": str(e.get("formula", "")),
            "inputs": list(e.get("inputs") or []),
        }
    return out


def compute_derived(d: dict[str, Any]) -> dict[str, Any]:
    """Return the derived block for one typed device record."""
    quals: dict[str, str] = d["qualifiers_map"]
    conv = d.get("vpi_convention") or "unspecified"
    L = d.get("length_mm")
    out: dict[str, Any] = {}

    if d.get("vpil_dc_vcm") is None and d.get("vpi_dc_v") is not None and L is not None:
        out["vpil_dc_vcm_derived"] = {
            "value": d["vpi_dc_v"] * L / 10,
            "unit": "V*cm",
            "formula": "vpi_dc_v * length_mm / 10",
            "inputs": ["vpi_dc_v", "length_mm"],
            "qualifier": quals.get("vpi_dc_v") or quals.get("length_mm"),
            "warning": f"inherits Vpi convention {conv}; not converted between conventions",
        }
    if d.get("rf_loss_db_per_cm") is not None and L is not None:
        f = d.get("rf_loss_freq_ghz")
        out["il_rf_total_db"] = {
            "value": d["rf_loss_db_per_cm"] * L / 10,
            "unit": "dB",
            "formula": "rf_loss_db_per_cm * length_mm / 10",
            "inputs": ["rf_loss_db_per_cm", "length_mm"],
            "qualifier": quals.get("rf_loss_db_per_cm") or quals.get("length_mm"),
            "warning": (f"loss taken at {f:g} GHz" if f is not None else "frequency of the RF loss figure not stated"),
        }
    vpi_ref_field = (
        "vpi_dc_v" if d.get("vpi_dc_v") is not None else ("vpi_rf_v" if d.get("vpi_rf_v") is not None else None)
    )
    vpi_ref = d[vpi_ref_field] if vpi_ref_field else None
    if vpi_ref_field and d.get("il_onchip_db") is not None:
        out["vpi_il_vdb"] = {
            "value": vpi_ref * d["il_onchip_db"],
            "unit": "V*dB",
            "formula": f"{vpi_ref_field} * il_onchip_db",
            "inputs": [vpi_ref_field, "il_onchip_db"],
            "qualifier": quals.get(vpi_ref_field) or quals.get("il_onchip_db"),
            "warning": f"inherits Vpi convention {conv}",
        }
    # completeness
    reported = [name for name, fields in CORE_FIELDS if any(d.get(f) is not None for f in fields)]
    out["completeness"] = {
        "value": len(reported) / len(CORE_FIELDS),
        "reported": reported,
        "missing": [name for name, _ in CORE_FIELDS if name not in reported],
        "unit": "fraction",
    }
    # FOM
    bw = d.get("bw3db_ghz")
    if (
        d.get("device_class") in FOM_CLASSES
        and bw is not None
        and bw > 0
        and vpi_ref_field
        and vpi_ref is not None
        and vpi_ref > 0
        and d.get("il_onchip_db") is not None
    ):
        il = d["il_onchip_db"]
        rf_total = out.get("il_rf_total_db", {}).get("value")
        assert vpi_ref is not None
        vpi_eff = vpi_ref * (10 ** (rf_total / 20) if rf_total is not None else 1.0)
        fom = bw / (vpi_eff * 10 ** (il / 10))
        used = ["bw3db_ghz", vpi_ref_field, "il_onchip_db"] + (
            ["rf_loss_db_per_cm", "length_mm"] if rf_total is not None else []
        )
        out["fom"] = {
            "value": fom,
            "unit": "GHz/V",
            "formula": (
                "bw3db_ghz / (Vpi_eff * 10^(il_onchip_db/10)); "
                "Vpi_eff = Vpi * 10^(il_rf_total_db/20) if known else Vpi"
            ),
            "inputs": used,
            "rf_corrected": rf_total is not None,
            "qualifiers": [f"{f}:{quals[f]}" for f in used if f in quals],
            "warning": f"Vpi convention {conv}",
        }
    return out


def vpil_best(d: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any] | None:
    if d.get("vpil_dc_vcm") is not None:
        return {"value": d["vpil_dc_vcm"], "source": "reported_dc", "qualifier": d["qualifiers_map"].get("vpil_dc_vcm")}
    if "vpil_dc_vcm_derived" in derived:
        x = derived["vpil_dc_vcm_derived"]
        return {"value": x["value"], "source": "derived_dc", "qualifier": x.get("qualifier")}
    if d.get("vpil_rf_vcm") is not None:
        return {"value": d["vpil_rf_vcm"], "source": "reported_rf", "qualifier": d["qualifiers_map"].get("vpil_rf_vcm")}
    return None


def headline_basis(d: dict[str, Any], ev: dict[str, dict[str, Any]], derived: dict[str, Any]) -> dict[str, str | None]:
    def basis(field: str, col_basis: str | None) -> str | None:
        e = ev.get(field)
        if e and e.get("basis"):
            return str(e["basis"])
        return col_basis if d.get(field) is not None else None

    vpi_field = "vpi_dc_v" if d.get("vpi_dc_v") is not None else "vpi_rf_v"
    vb = basis(vpi_field, d.get("vpi_basis"))
    out: dict[str, str | None] = {
        "vpi": vb,
        "vpil": (
            basis("vpil_dc_vcm", d.get("vpi_basis"))
            if d.get("vpil_dc_vcm") is not None
            else ("derived" if "vpil_dc_vcm_derived" in derived else basis("vpil_rf_vcm", d.get("vpi_basis")))
        ),
        "bw3db": basis("bw3db_ghz", d.get("bw_basis")),
        "il_onchip": basis("il_onchip_db", d.get("il_basis")),
        "il_f2f": basis("il_fiber_to_fiber_db", d.get("il_basis")),
        "rf_loss": basis("rf_loss_db_per_cm", None),
        "rate": basis("max_line_rate_gbps", None) or basis("max_baud_gbd", None),
    }
    return out


def author_label(authors: list[str], year: int | None) -> str:
    if not authors:
        return str(year or "")
    first = authors[0].strip().split()
    last = first[-1] if first else authors[0]
    suffix = " et al." if len(authors) > 2 else (" and " + authors[1].strip().split()[-1] if len(authors) == 2 else "")
    return f"{last}{suffix} {year}" if year else f"{last}{suffix}"


def rep_order_key(dev: dict[str, Any]) -> tuple[Any, ...]:
    fom = dev["derived"].get("fom", {}).get("value")
    vp = dev["vpil_best"]["value"] if dev["vpil_best"] else None
    return (
        -dev["derived"]["completeness"]["value"],
        -(fom if fom is not None else float("-inf")),
        vp if vp is not None else float("inf"),
        dev["device_id"],
    )


def pick_reps(devs: list[dict[str, Any]]) -> dict[str, str]:
    default: str = min(devs, key=rep_order_key)["device_id"]

    def best(metric: Any, reverse: bool) -> str:
        cand = [(metric(x), x) for x in devs if metric(x) is not None]
        if not cand:
            return default
        cand.sort(key=lambda t: ((-t[0] if reverse else t[0]), rep_order_key(t[1])))
        return str(cand[0][1]["device_id"])  # noqa

    return {
        "default": default,
        "lowest_vpil": best(lambda x: x["vpil_best"]["value"] if x["vpil_best"] else None, False),
        "highest_bw": best(lambda x: x.get("bw3db_ghz"), True),
        "highest_fom": best(lambda x: x["derived"].get("fom", {}).get("value"), True),
        "lowest_vpi_il": best(lambda x: x["derived"].get("vpi_il_vdb", {}).get("value"), False),
    }


def load_sims(sims_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not sims_dir.is_dir():
        return out
    for cfg in sorted(sims_dir.glob("*/config.yaml")):
        try:
            with cfg.open() as f:
                c = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            print(f"WARNING sim config {cfg} unreadable: {exc}", file=sys.stderr)
            continue
        out.append(
            {
                "paper_id": str(c.get("paper_id") or cfg.parent.name),
                "id": str(c.get("id") or cfg.parent.name),
                "device_id": c.get("device_id"),
                "title": c.get("title"),
                "repro_grade": c.get("repro_grade"),
                "validation_status": c.get("validation_status"),
                "chain": c.get("chain") or [],
                "n_targets": len(c.get("targets") or []),
                "path": f"sims/{cfg.parent.name}/config.yaml",
            }
        )
    return out


def build(data: Path, sims_dir: Path | None = None) -> dict[str, Any]:
    schema = yaml.safe_load((data / "schema" / "devices.schema.yaml").read_text())
    enums: dict[str, list[str]] = schema["enums"]
    p_cols, d_cols = schema["papers_columns"], schema["devices_columns"]
    _, p_rows = read_csv(data / "papers.csv")
    _, d_rows = read_csv(data / "devices.csv")
    _, o_rows = read_csv(data / "organizations.csv")
    sims = load_sims(sims_dir if sims_dir is not None else ROOT / "sims")
    sims_by_paper: dict[str, list[str]] = {}
    for s in sims:
        sims_by_paper.setdefault(s["paper_id"], []).append(s["id"])

    sim_device_ids = {str(s["device_id"]) for s in sims if s["device_id"]}
    warnings: list[str] = []
    orgs = {
        r["org_name"]: {k: r[k] for k in ("org_name", "org_type", "country", "region", "parent_org", "notes")}
        for r in o_rows
    }

    papers: dict[str, dict[str, Any]] = {}
    for r in p_rows:
        p = typed_row(p_cols, r)
        p["label"] = author_label(p["authors"], p["year"])
        papers[p["paper_id"]] = p

    devices: list[dict[str, Any]] = []
    ev_missing = 0
    ev_fields = {c["name"] for c in d_cols if c.get("evidence")}
    nonempty_ev_fields = 0
    for pid in sorted(papers):
        ev_all = load_evidence(data, pid)
        for r in [x for x in d_rows if x["paper_id"] == pid]:
            d = typed_row(d_cols, r)
            qmap: dict[str, str] = {}
            for q in d["qualifiers"]:
                fld, _, op = q.partition(":")
                qmap[fld] = op
            d["qualifiers_map"] = qmap
            ev = ev_all.get(d["device_id"], {})
            for fld in ev_fields:
                if d.get(fld) is not None:
                    nonempty_ev_fields += 1
                    if fld not in ev:
                        ev_missing += 1
            derived = compute_derived(d)
            d["derived"] = derived
            d["vpil_best"] = vpil_best(d, derived)
            hb = headline_basis(d, ev, derived)
            d["headline_basis"] = hb
            d["is_sim"] = any(hb[k] in SIM_BASES for k in ("vpi", "vpil", "bw3db", "il_onchip", "il_f2f"))
            d["measured_only"] = not d["is_sim"]
            d["evidence"] = ev
            d["sim_available"] = d["device_id"] in sim_device_ids or d["device_id"] in sims_by_paper.get(pid, [])
            d.pop("qualifiers", None)
            d["qualifiers"] = qmap
            d.pop("qualifiers_map")
            devices.append(d)

    by_paper: dict[str, list[dict[str, Any]]] = {}
    for d in devices:
        by_paper.setdefault(d["paper_id"], []).append(d)

    for pid, p in papers.items():
        devs = by_paper.get(pid, [])
        p["device_ids"] = [d["device_id"] for d in devs]
        p["rep"] = pick_reps(devs) if devs else {}
        affil_names = list(dict.fromkeys(p["universities"] + p["companies"]))
        fab_names = list(p["foundry_or_fab"])
        for n in affil_names + fab_names:
            if n not in orgs:
                warnings.append(f"{pid}: organization {n!r} not in organizations.csv")
        p["orgs_affil"] = [orgs[n] for n in affil_names if n in orgs]
        p["orgs_fab"] = [orgs[n] for n in fab_names if n in orgs]
        derived_countries = sorted({o["country"] for o in p["orgs_affil"]})
        if p["countries"] and derived_countries and set(p["countries"]) != set(derived_countries):
            warnings.append(
                f"{pid}: papers.countries {p['countries']} differs from organizations-derived {derived_countries}"
            )
        p["countries_derived"] = derived_countries or sorted(set(p["countries"]))
        p["regions_derived"] = sorted({o["region"] for o in p["orgs_affil"]} or set())
        if not p["regions_derived"]:
            fallback = {orgs[n]["region"] for n in fab_names if n in orgs}
            p["regions_derived"] = sorted(fallback)
        p["sim_ids"] = sims_by_paper.get(pid, [])
        p["has_sim"] = bool(p["sim_ids"]) or bool(p["sim_config"])
        p["n_devices"] = len(devs)

    paper_list = [papers[k] for k in sorted(papers)]
    countries_all = sorted({c for p in paper_list for c in p["countries_derived"]})

    # integrity summary
    basis_counts: Counter[str] = Counter()
    for d in devices:
        for e in d["evidence"].values():
            basis_counts[str(e.get("basis"))] += 1
    headline_counts: dict[str, Counter[str]] = {}
    for d in devices:
        for k, v in d["headline_basis"].items():
            if v:
                headline_counts.setdefault(k, Counter())[v] += 1

    def cnt(items: list[Any]) -> dict[str, int]:
        return dict(
            sorted(Counter(str(x) if x else "unspecified" for x in items).items(), key=lambda kv: (-kv[1], kv[0]))
        )

    comp = [d["derived"]["completeness"]["value"] for d in devices]
    integrity = {
        "counts": {
            "papers": len(paper_list),
            "devices": len(devices),
            "organizations": len({o["org_name"] for p in paper_list for o in p["orgs_affil"] + p["orgs_fab"]}),
            "organizations_listed": len(orgs),
            "countries": len(countries_all),
            "sim_configs": len(sims),
            "devices_with_fom": sum(1 for d in devices if "fom" in d["derived"]),
            "devices_with_derived_vpil": sum(1 for d in devices if "vpil_dc_vcm_derived" in d["derived"]),
        },
        "per_platform": cnt([d["waveguide_platform"] for d in devices]),
        "per_material": cnt([d["eo_material"] for d in devices]),
        "per_device_class": cnt([d["device_class"] for d in devices]),
        "per_basis": dict(sorted(basis_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_headline_basis": {k: dict(v) for k, v in sorted(headline_counts.items())},
        "per_source_type": cnt([p["source_type"] for p in paper_list]),
        "per_access": cnt([p["access"] for p in paper_list]),
        "per_repro_grade": cnt([p["repro_grade"] for p in paper_list]),
        "per_redistribution": cnt([p["redistribution"] for p in paper_list]),
        "evidence": {"nonempty_evidence_fields": nonempty_ev_fields, "without_evidence_entry": ev_missing},
        "completeness_mean": (sum(comp) / len(comp)) if comp else None,
    }

    enum_labels = {name: [{"value": v, "label": label_of(v)} for v in values] for name, values in enums.items()}
    enum_labels["region"] = [{"value": v, "label": label_of(v)} for v in REGIONS]
    enum_labels["org_type"] = [{"value": v, "label": label_of(v)} for v in ORG_TYPES]
    enum_labels["country"] = [{"value": c, "label": COUNTRY_NAMES.get(c, c)} for c in countries_all]

    columns = [
        {
            "name": c["name"],
            "type": c["type"],
            "unit": c.get("unit", ""),
            "desc": c.get("desc", ""),
            "enum": c.get("enum", ""),
            "evidence": bool(c.get("evidence")),
            "label": COLUMN_LABELS.get(c["name"], label_of(c["name"])),
        }
        for c in d_cols
    ]
    paper_columns = [
        {"name": c["name"], "type": c["type"], "desc": c.get("desc", ""), "label": label_of(c["name"])} for c in p_cols
    ]

    return {
        "meta": {
            "generated_on": dt.date.today().isoformat(),
            "schema_version": schema.get("version"),
            "core_fields": [{"name": n, "fields": f} for n, f in CORE_FIELDS],
            "fom_classes": sorted(FOM_CLASSES),
            "fom_definition": (
                "f3dB[GHz] / (Vpi_eff[V] * 10^(IL_onchip[dB]/10)); "
                "Vpi_eff = Vpi * 10^(RF_loss_total[dB]/20) when RF loss total is known, else Vpi"
            ),
            "vpi_il_definition": "Vpi[V] * IL_onchip[dB] (V*dB); Vpi = vpi_dc_v, else vpi_rf_v",
            "completeness_definition": (
                "reported core fields / 7: Vpi or Vpi*L (any convention), 3 dB BW, on-chip IL, RF loss, Z0, n_RF, ng"
            ),
            "column_groups": [{"name": k, "fields": v} for k, v in COLUMN_GROUPS.items()],
        },
        "integrity": integrity,
        "enums": enum_labels,
        "columns": columns,
        "paper_columns": paper_columns,
        "papers": paper_list,
        "devices": devices,
        "organizations": [orgs[k] for k in sorted(orgs)],
        "sims": sims,
        "warnings": sorted(set(warnings)),
    }


FLAT_PAPER_COLS = [
    "label",
    "title",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
    "source_type",
    "access",
    "license",
    "redistribution",
    "repro_grade",
]
FLAT_DERIVED = ["vpil_dc_vcm_derived", "il_rf_total_db", "vpi_il_vdb", "completeness", "fom"]


def write_flat_csv(atlas: dict[str, Any], path: Path) -> None:
    papers = {p["paper_id"]: p for p in atlas["papers"]}
    dev_cols = [c["name"] for c in atlas["columns"] if c["name"] not in ("qualifiers",)]
    header = (
        ["is_representative"]
        + FLAT_PAPER_COLS
        + ["organizations", "foundry_or_fab", "countries", "regions", "has_sim"]
        + dev_cols
        + ["qualifiers"]
        + FLAT_DERIVED
        + ["is_sim"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in atlas["devices"]:
            p = papers[d["paper_id"]]
            row: list[Any] = [int(p["rep"].get("default") == d["device_id"])]
            row += [p[c] if p[c] is not None else "" for c in FLAT_PAPER_COLS]
            row += [
                ";".join(o["org_name"] for o in p["orgs_affil"]),
                ";".join(o["org_name"] for o in p["orgs_fab"]),
                ";".join(p["countries_derived"]),
                ";".join(p["regions_derived"]),
                int(p["has_sim"]),
            ]
            for c in dev_cols:
                v = d[c]
                row.append(";".join(v) if isinstance(v, list) else ("" if v is None else v))
            row.append(";".join(f"{k}:{v}" for k, v in d["qualifiers"].items()))
            for k in FLAT_DERIVED:
                x = d["derived"].get(k)
                row.append("" if x is None else x["value"])
            row.append(int(d["is_sim"]))
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--with-staging", nargs="+", default=[])
    ap.add_argument("--out", default=str(ROOT / "app" / "static" / "data" / "atlas.json"))
    ap.add_argument("--flat-csv", default="")
    ap.add_argument("--sims", default=str(ROOT / "sims"))
    a = ap.parse_args()
    data = Path(a.data)
    with tempfile.TemporaryDirectory() as tmp:
        src = data
        if a.with_staging:
            src = Path(tmp) / "data"
            shutil.copytree(data, src, ignore=shutil.ignore_patterns("_staging"))
            msgs, counts = merge([Path(s) for s in a.with_staging], src, set())
            for m in msgs:
                print("MERGE", m, file=sys.stderr)
            print(f"merged staging in memory: {counts}", file=sys.stderr)
        atlas = build(src, Path(a.sims))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    if a.flat_csv:
        write_flat_csv(atlas, Path(a.flat_csv))
    c = atlas["integrity"]["counts"]
    print(
        f"wrote {out}: {c['papers']} papers, {c['devices']} devices, {c['organizations']} orgs, "
        f"{len(atlas['warnings'])} warnings"
    )
    for w in atlas["warnings"]:
        print("WARNING", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
