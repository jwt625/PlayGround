"""Validate packaging extraction tables against data/schema/packaging.schema.yaml.

Checks: headers, required columns, types (int/float/date/enum/list), enum membership, unique keys, parent references,
id prefixes (child ids start with their parent id), locator format ("p.N" with N <= pages in text.md), qualifier
syntax, evidence YAML (paper_id, derived entries reference existing rows), and that paper references exist.

Usage: uv run python scripts/validate_db.py                 # data/
       uv run python scripts/validate_db.py --staging data/_staging/<batch>
       uv run python scripts/validate_db.py --write-templates   # data/_staging/TEMPLATE/*.csv headers
Exit 0 = no errors (warnings allowed), 1 = errors.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = yaml.safe_load((ROOT / "data" / "schema" / "packaging.schema.yaml").read_text())
ENUMS = SCHEMA["enums"]
TABLES = SCHEMA["tables"]
COMMON = list(SCHEMA.get("common", {}))
for _t in TABLES.values():  # YAML merge keys land first; put the shared basis/source/... columns last
    cols = _t["columns"]
    _t["columns"] = {**{k: v for k, v in cols.items() if k not in COMMON},
                     **{k: cols[k] for k in COMMON if k in cols}}
LOC_RE = re.compile(r"\bp\.(\d+)")
QUAL_RE = re.compile(r"^[a-z0-9_]+:(gt|lt|ge|le|approx)$")


def pages_of(paper_id: str) -> int | None:
    t = ROOT / "references" / paper_id / "text.md"
    if not t.exists():
        return None
    m = re.search(r"^pages: (\d+)", t.read_text()[:2000], re.M)
    return int(m.group(1)) if m else None


def read(d: Path, table: str) -> tuple[list[str], list[dict]]:
    p = d / f"{table}.csv"
    if not p.exists():
        return [], []
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def check_cell(col: str, spec: dict, v: str) -> str | None:
    if v == "":
        return "required" if spec.get("required") else None
    t = spec["type"]
    try:
        if t == "int":
            int(v)
        elif t == "float":
            float(v)
        elif t == "date":
            dt.date.fromisoformat(v)
        elif t == "enum":
            if v not in ENUMS[spec["enum"]]:
                return f"'{v}' not in enum {spec['enum']}"
        elif t == "list" and spec.get("enum"):
            bad = [x for x in v.split(";") if x.strip() and x.strip() not in ENUMS[spec["enum"]]]
            if bad:
                return f"{bad} not in enum {spec['enum']}"
    except ValueError:
        return f"'{v}' is not {t}"
    return None


def validate(d: Path, base: Path | None = None) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    data: dict[str, list[dict]] = {}
    for table, ts in TABLES.items():
        hdr, rows = read(d, table)
        if not hdr:
            if table == "papers":
                errs.append(f"{table}.csv missing")
            continue
        expected = list(ts["columns"])
        if hdr != expected:
            missing = [c for c in expected if c not in hdr]
            extra = [c for c in hdr if c not in expected]
            errs.append(f"{table}.csv header mismatch (missing {missing}, extra {extra})")
        data[table] = rows
        key = ts["key"]
        seen = set()
        for i, r in enumerate(rows, start=2):
            rid = r.get(key, "")
            where = f"{table}.csv:{i} ({rid})"
            if rid in seen:
                errs.append(f"{where}: duplicate {key}")
            seen.add(rid)
            for col, spec in ts["columns"].items():
                e = check_cell(col, spec, (r.get(col) or "").strip())
                if e:
                    errs.append(f"{where}: {col} {e}")
            for o in filter(None, (r.get("overrides") or "").split(";")):
                f_, _, v_ = o.strip().partition(":")
                if f_ not in ts["columns"] or v_ not in ENUMS["basis"] + ENUMS["source"]:
                    errs.append(f"{where}: bad override '{o}' (field:basis|source)")
            for q in filter(None, (r.get("qualifiers") or "").split(";")):
                if not QUAL_RE.match(q.strip()):
                    errs.append(f"{where}: bad qualifier '{q}'")
                elif q.split(":")[0] not in ts["columns"]:
                    errs.append(f"{where}: qualifier field '{q.split(':')[0]}' not a column")
            if "locator" in ts["columns"] and r.get("locator") and not LOC_RE.findall(r["locator"]):
                errs.append(f"{where}: locator '{r['locator']}' has no p.N")
            notes = r.get("notes") or ""
            if len(notes.split()) > 40:
                warns.append(f"{where}: notes > 40 words")
    # seq uniqueness (warning): stack_layers per (tv, structure, side, feature); process_steps per (tv, flow)
    for table, keyf in (("stack_layers", ("tv_id", "structure", "side", "feature", "seq")),
                        ("process_steps", ("tv_id", "flow_name", "seq"))):
        seen_k: set = set()
        for r in data.get(table, []):
            k = tuple(r.get(x, "") for x in keyf)
            if k in seen_k:
                warns.append(f"{table}: duplicate {keyf} = {k}")
            seen_k.add(k)

    # locator page range (paper via test_vehicles)
    tv2p = {r["tv_id"]: r["paper_id"] for r in data.get("test_vehicles", [])}
    if base is not None:
        tv2p.update({r["tv_id"]: r["paper_id"] for r in read(base, "test_vehicles")[1]})
    for table, ts in TABLES.items():
        if "locator" not in ts["columns"]:
            continue
        for r in data.get(table, []):
            pid = r.get("paper_id") or tv2p.get(r.get("tv_id", ""))
            np = pages_of(pid) if pid else None
            pages = [int(x) for x in LOC_RE.findall(r.get("locator") or "")]
            if np and any(pg < 1 or pg > np for pg in pages):
                errs.append(f"{table}: {r[ts['key']]} locator page out of range 1..{np}")

    # parent references and id prefixes
    def keys(table: str) -> set[str]:
        own = {r[TABLES[table]["key"]] for r in data.get(table, [])}
        if base is not None:
            _, rows = read(base, table)
            own |= {r[TABLES[table]["key"]] for r in rows}
        return own
    for table, ts in TABLES.items():
        par = ts.get("parent")
        if not par or table not in data:
            continue
        pk = keys(par["table"])
        for r in data[table]:
            v = r.get(par["column"], "")
            if v and v not in pk:
                errs.append(f"{table}: {ts['key']}={r[ts['key']]} references missing {par['table']}.{par['column']}={v}")
            if v and not r[ts["key"]].startswith(v + "-"):
                errs.append(f"{table}: id {r[ts['key']]} does not start with parent id {v}-")
    # papers: one evidence file each, source cached
    for r in data.get("papers", []):
        pid = r["paper_id"]
        ev = d / "evidence" / f"{pid}.yaml"
        if not ev.exists():
            errs.append(f"papers: {pid} has no evidence/{pid}.yaml")
        else:
            try:
                y = yaml.safe_load(ev.read_text()) or {}
                if y.get("paper_id") != pid:
                    errs.append(f"evidence/{pid}.yaml: paper_id mismatch")
                all_ids = {rr[TABLES[t]["key"]] for t in data for rr in data[t]}
                for sect in ("derived", "entries", "conflicts"):
                    for e in (y.get(sect) or []):
                        if sect == "derived" and not e.get("formula"):
                            errs.append(f"evidence/{pid}.yaml: derived entry without formula ({e.get('field')})")
                        rid = e.get("row_id")
                        if rid and rid not in all_ids:
                            errs.append(f"evidence/{pid}.yaml: {sect} row_id {rid} not found in staged tables")
            except yaml.YAMLError as exc:
                errs.append(f"evidence/{pid}.yaml: YAML error {str(exc)[:80]}")
        if r.get("source_status") == "full_text" and not (ROOT / "references" / pid / "text.md").exists():
            errs.append(f"papers: {pid} source_status full_text but no references/{pid}/text.md")
        n_tv = sum(1 for t in data.get("test_vehicles", []) if t["paper_id"] == pid)
        if r.get("n_test_vehicles") and int(r["n_test_vehicles"]) != n_tv:
            errs.append(f"papers: {pid} n_test_vehicles={r['n_test_vehicles']} but {n_tv} test_vehicles rows")
    # privacy
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in (".csv", ".yaml", ".md"):
            s = p.read_text(errors="ignore")
            if "/Users/" in s or "\\Users\\" in s:
                errs.append(f"{p.name}: absolute local path")
    return errs, warns


def write_templates() -> None:
    t = ROOT / "data" / "_staging" / "TEMPLATE"
    t.mkdir(parents=True, exist_ok=True)
    for table, ts in TABLES.items():
        (t / f"{table}.csv").write_text(",".join(ts["columns"]) + "\n")
    print(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging")
    ap.add_argument("--write-templates", action="store_true")
    a = ap.parse_args()
    if a.write_templates:
        write_templates()
        return 0
    d = ROOT / a.staging if a.staging else ROOT / "data"
    errs, warns = validate(d, base=ROOT / "data" if a.staging else None)
    for w in warns:
        print("WARN", w)
    for e in errs:
        print("ERROR", e)
    print(f"{len(errs)} errors, {len(warns)} warnings ({d.relative_to(ROOT)})")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
