"""Validate data/papers.csv, data/devices.csv, data/organizations.csv and data/evidence/*.yaml against the schema.

Integrity rule: every non-empty evidence-required value in devices.csv must have a matching evidence entry
(device_id, field, same value, locator, basis) or a `derived` entry with a formula. Empty cell means not reported.

Usage: uv run python scripts/validate_db.py [--data data] [--staging DIR]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
ORG_COLUMNS = ["org_name", "org_type", "country", "region", "parent_org", "notes"]
ORG_TYPES = {"university", "company", "national_lab", "research_institute", "foundry", "facility", "consortium", "other"}
REGIONS = {"north_america", "europe", "east_asia", "south_asia", "southeast_asia", "oceania", "middle_east", "other"}


def load_schema(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


def split_list(cell: str) -> list[str]:
    return [p.strip() for p in cell.split(";") if p.strip()]


def check_cell(col: dict[str, Any], cell: str, enums: dict[str, list[str]]) -> str | None:
    """Return an error string for a malformed non-empty cell, else None."""
    if cell == "":
        return None
    t = col["type"]
    if t == "float":
        try:
            v = float(cell)
        except ValueError:
            return f"not a float: {cell!r}"
        if not math.isfinite(v):
            return f"non-finite: {cell!r}"
    elif t == "int":
        try:
            int(cell)
        except ValueError:
            return f"not an int: {cell!r}"
    elif t == "enum":
        if cell not in enums[col["enum"]]:
            return f"{cell!r} not in enum {col['enum']}"
    elif t == "date":
        try:
            dt.date.fromisoformat(cell)
        except ValueError:
            return f"not an ISO date: {cell!r}"
    return None


def values_match(csv_value: str, ev_value: Any) -> bool:
    try:
        a = float(csv_value)
        b = float(ev_value)
    except (TypeError, ValueError):
        return str(ev_value).strip() == csv_value.strip()
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)


def validate_table(
    name: str, columns: list[dict[str, Any]], header: list[str], rows: list[dict[str, str]], enums: dict[str, list[str]]
) -> list[str]:
    errs: list[str] = []
    expected = [c["name"] for c in columns]
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        errs.append(f"{name}: header mismatch; missing={missing} extra={extra}; order must follow the schema")
        return errs
    for i, row in enumerate(rows, start=2):
        for col in columns:
            cell = row[col["name"]].strip()
            if col.get("required") and cell == "":
                errs.append(f"{name}:{i} {col['name']} is required")
            msg = check_cell(col, cell, enums)
            if msg:
                errs.append(f"{name}:{i} {col['name']}: {msg}")
    return errs


def validate_orgs(path: Path) -> tuple[set[str], list[str]]:
    errs: list[str] = []
    names: set[str] = set()
    if not path.exists():
        return names, [f"{path.name}: missing"]
    header, rows = read_csv(path)
    if header != ORG_COLUMNS:
        return names, [f"{path.name}: header must be {ORG_COLUMNS}, got {header}"]
    for i, r in enumerate(rows, start=2):
        n = r["org_name"].strip()
        if not n:
            errs.append(f"organizations.csv:{i} org_name empty")
        if n in names:
            errs.append(f"organizations.csv:{i} duplicate org_name {n!r}")
        names.add(n)
        if r["org_type"] not in ORG_TYPES:
            errs.append(f"organizations.csv:{i} org_type {r['org_type']!r} not in {sorted(ORG_TYPES)}")
        if len(r["country"]) != 2 or not r["country"].isupper():
            errs.append(f"organizations.csv:{i} country must be ISO alpha-2 uppercase, got {r['country']!r}")
        if r["region"] not in REGIONS:
            errs.append(f"organizations.csv:{i} region {r['region']!r} not in {sorted(REGIONS)}")
    return names, errs


def validate_evidence(
    data: Path, devices: list[dict[str, str]], dev_cols: list[dict[str, Any]], enums: dict[str, list[str]]
) -> list[str]:
    errs: list[str] = []
    ev_cols = {c["name"] for c in dev_cols if c.get("evidence")}
    all_cols = {c["name"] for c in dev_cols}
    by_paper: dict[str, list[dict[str, str]]] = {}
    for d in devices:
        by_paper.setdefault(d["paper_id"], []).append(d)
    for paper_id, devs in by_paper.items():
        ev_path = data / "evidence" / f"{paper_id}.yaml"
        if not ev_path.exists():
            errs.append(f"evidence: {ev_path.relative_to(data.parent)} missing for paper {paper_id}")
            continue
        with ev_path.open() as f:
            ev = yaml.safe_load(f) or {}
        if ev.get("paper_id") != paper_id:
            errs.append(f"{ev_path.name}: paper_id field must be {paper_id!r}")
        entries = {(e.get("device_id"), e.get("field")): e for e in ev.get("entries") or []}
        derived = {(e.get("device_id"), e.get("field")): e for e in ev.get("derived") or []}
        valid_devices = {d["device_id"] for d in devs}
        for key, e in entries.items():
            if key[0] not in valid_devices:
                errs.append(f"{ev_path.name}: entry for unknown device_id {key[0]!r}")
            if key[1] not in all_cols:
                errs.append(f"{ev_path.name}: entry for unknown field {key[1]!r}")
            if not str(e.get("locator", "")).strip():
                errs.append(f"{ev_path.name}: {key} missing locator")
            if e.get("basis") not in enums["basis"]:
                errs.append(f"{ev_path.name}: {key} basis {e.get('basis')!r} invalid")
        for key, e in derived.items():
            if not str(e.get("formula", "")).strip():
                errs.append(f"{ev_path.name}: derived {key} missing formula")
        for d in devs:
            for field in ev_cols:
                cell = d[field].strip()
                if cell == "":
                    continue
                key = (d["device_id"], field)
                if key in entries:
                    if not values_match(cell, entries[key].get("value")):
                        errs.append(f"{ev_path.name}: {key} value {entries[key].get('value')!r} != CSV {cell!r}")
                elif key in derived:
                    if not values_match(cell, derived[key].get("value")):
                        errs.append(f"{ev_path.name}: derived {key} value != CSV {cell!r}")
                else:
                    errs.append(f"{d['device_id']}.{field}={cell!r} has no evidence entry")
    return errs


def validate(data: Path) -> list[str]:
    schema = load_schema(data / "schema" / "devices.schema.yaml")
    enums: dict[str, list[str]] = schema["enums"]
    errs: list[str] = []
    org_names, org_errs = validate_orgs(data / "organizations.csv")
    errs += org_errs

    p_path, d_path = data / "papers.csv", data / "devices.csv"
    if not p_path.exists() or not d_path.exists():
        return errs + ["papers.csv and devices.csv must exist"]
    p_header, papers = read_csv(p_path)
    d_header, devices = read_csv(d_path)
    errs += validate_table("papers.csv", schema["papers_columns"], p_header, papers, enums)
    errs += validate_table("devices.csv", schema["devices_columns"], d_header, devices, enums)
    if errs and any("header mismatch" in e for e in errs):
        return errs

    paper_ids: set[str] = set()
    for i, p in enumerate(papers, start=2):
        pid = p["paper_id"]
        if pid in paper_ids:
            errs.append(f"papers.csv:{i} duplicate paper_id {pid}")
        paper_ids.add(pid)
        if not (p["doi"] or p["arxiv_id"] or p["url"]):
            errs.append(f"papers.csv:{i} {pid} needs doi, arxiv_id or url")
        for col in ("universities", "companies", "foundry_or_fab"):
            for org in split_list(p[col]):
                if org not in org_names:
                    errs.append(f"papers.csv:{i} {pid} {col}: {org!r} not in organizations.csv")
        for c in split_list(p["countries"]):
            if len(c) != 2 or not c.isupper():
                errs.append(f"papers.csv:{i} {pid} countries: {c!r} must be ISO alpha-2 uppercase")
        if p["sim_config"] and not (ROOT / p["sim_config"]).exists():
            errs.append(f"papers.csv:{i} {pid} sim_config {p['sim_config']} does not exist")
        if "tweet" in p["url"] or "x.com" in p["url"] or "twitter.com" in p["url"]:
            errs.append(f"papers.csv:{i} {pid} url must be a primary source, not a tweet")

    dev_ids: set[str] = set()
    for i, d in enumerate(devices, start=2):
        did = d["device_id"]
        if did in dev_ids:
            errs.append(f"devices.csv:{i} duplicate device_id {did}")
        dev_ids.add(did)
        if d["paper_id"] not in paper_ids:
            errs.append(f"devices.csv:{i} {did} unknown paper_id {d['paper_id']}")
        if not did.startswith(d["paper_id"] + "-"):
            errs.append(f"devices.csv:{i} device_id must start with '<paper_id>-'")
        for q in split_list(d["qualifiers"]):
            fld, _, op = q.partition(":")
            if fld not in d or op not in enums["qualifier_ops"]:
                errs.append(f"devices.csv:{i} {did} bad qualifier {q!r} (field:op with op in lt|gt|approx)")
            elif d[fld].strip() == "":
                errs.append(f"devices.csv:{i} {did} qualifier on empty field {fld!r}")
        if d["evidence_ref"] != f"data/evidence/{d['paper_id']}.yaml":
            errs.append(f"devices.csv:{i} {did} evidence_ref must be data/evidence/{d['paper_id']}.yaml")
    errs += validate_evidence(data, devices, schema["devices_columns"], enums)
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    errs = validate(Path(args.data))
    for e in errs:
        print("ERROR", e)
    print(f"{len(errs)} error(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
