"""Merge staged fragments into data/ safely.

A staging dir mirrors the data layout: <dir>/papers.csv, <dir>/devices.csv, <dir>/organizations.csv, <dir>/evidence/*.yaml.
Rules: a paper_id already present in data/papers.csv is a conflict unless --replace-paper-ids lists it; organizations that
already exist must be identical (else conflict); evidence files never overwrite. The merged result is validated in a temp
copy first; data/ is only modified when validation passes (or --force).

Usage: uv run python scripts/merge_staging.py DIR [DIR ...] [--apply] [--replace-paper-ids id1,id2]
Default is a dry run that prints what would change.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_db import ORG_COLUMNS, read_csv, validate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def write_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def merge(dirs: list[Path], base: Path, replace: set[str]) -> tuple[list[str], dict[str, int]]:
    msgs: list[str] = []
    counts = {"papers": 0, "devices": 0, "orgs": 0, "evidence": 0}
    p_h, papers = read_csv(base / "papers.csv")
    d_h, devices = read_csv(base / "devices.csv")
    o_h, orgs = read_csv(base / "organizations.csv")
    org_by_name = {o["org_name"]: o for o in orgs}
    paper_ids = {p["paper_id"] for p in papers}
    for d in dirs:
        if (d / "papers.csv").exists():
            _, new_p = read_csv(d / "papers.csv")
            for p in new_p:
                pid = p["paper_id"]
                if pid in paper_ids:
                    if pid in replace:
                        papers = [x for x in papers if x["paper_id"] != pid]
                        devices = [x for x in devices if x["paper_id"] != pid]
                    else:
                        msgs.append(f"CONFLICT paper_id {pid} already in data/ (from {d.name})")
                        continue
                papers.append(p)
                paper_ids.add(pid)
                counts["papers"] += 1
        if (d / "devices.csv").exists():
            _, new_d = read_csv(d / "devices.csv")
            have = {x["device_id"] for x in devices}
            for x in new_d:
                if x["device_id"] in have:
                    msgs.append(f"CONFLICT device_id {x['device_id']} (from {d.name})")
                    continue
                devices.append(x)
                counts["devices"] += 1
        if (d / "organizations.csv").exists():
            _, new_o = read_csv(d / "organizations.csv")
            for o in new_o:
                cur = org_by_name.get(o["org_name"])
                if cur is None:
                    orgs.append(o)
                    org_by_name[o["org_name"]] = o
                    counts["orgs"] += 1
                elif {k: cur[k] for k in ("org_type", "country", "region")} != {
                    k: o[k] for k in ("org_type", "country", "region")
                }:
                    msgs.append(f"CONFLICT org {o['org_name']!r} differs (from {d.name}): {cur} vs {o}")
        if (d / "evidence").is_dir():
            (base / "evidence").mkdir(exist_ok=True)
            for ev in sorted((d / "evidence").glob("*.yaml")):
                dest = base / "evidence" / ev.name
                if dest.exists() and ev.stem not in replace:
                    msgs.append(f"CONFLICT evidence {ev.name} exists (from {d.name})")
                    continue
                shutil.copyfile(ev, dest)
                counts["evidence"] += 1
    papers.sort(key=lambda r: r["paper_id"])
    devices.sort(key=lambda r: r["device_id"])
    orgs.sort(key=lambda r: r["org_name"])
    write_rows(base / "papers.csv", p_h, papers)
    write_rows(base / "devices.csv", d_h, devices)
    write_rows(base / "organizations.csv", o_h, orgs)
    return msgs, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--replace-paper-ids", default="")
    a = ap.parse_args()
    replace = {s for s in a.replace_paper_ids.split(",") if s}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "data"
        shutil.copytree(DATA, work)
        msgs, counts = merge([Path(d) for d in a.dirs], work, replace)
        errs = validate(work)
        for m in msgs:
            print(m)
        for e in errs[:80]:
            print("VALIDATION", e)
        print(f"merge counts: {counts}; conflicts: {len(msgs)}; validation errors: {len(errs)}")
        if a.apply and (not msgs and not errs or a.force):
            for name in ("papers.csv", "devices.csv", "organizations.csv"):
                shutil.copyfile(work / name, DATA / name)
            (DATA / "evidence").mkdir(exist_ok=True)
            for ev in (work / "evidence").glob("*.yaml"):
                shutil.copyfile(ev, DATA / "evidence" / ev.name)
            print("applied")
            return 0
        print("dry run (nothing written)" if not a.apply else "NOT applied (conflicts or errors)")
        return 0 if not msgs and not errs else 1


if __name__ == "__main__":
    sys.exit(main())
