"""Tests for scripts/merge_staging.py merge semantics (conflicts, replacement, org comparison)."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from merge_staging import merge  # noqa: E402
from test_validate_db import make_db  # noqa: E402
from validate_db import ORG_COLUMNS, read_csv, validate  # noqa: E402


def stage(tmp: Path, name: str, base: Path, paper_id: str, org_notes: str = "", org_type: str = "university") -> Path:
    d = tmp / name
    (d / "evidence").mkdir(parents=True)
    p_h, p_rows = read_csv(base / "papers.csv")
    d_h, d_rows = read_csv(base / "devices.csv")
    paper = dict(p_rows[0], paper_id=paper_id, title="Other", universities="Staged University")
    dev = dict(d_rows[0], device_id=f"{paper_id}-a", paper_id=paper_id, evidence_ref=f"data/evidence/{paper_id}.yaml")
    for fn, h, rows in (("papers.csv", p_h, [paper]), ("devices.csv", d_h, [dev])):
        with (d / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=h)
            w.writeheader()
            w.writerows(rows)
    with (d / "organizations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ORG_COLUMNS)
        w.writeheader()
        w.writerow(
            dict(
                org_name="Staged University",
                org_type=org_type,
                country="DE",
                region="europe",
                parent_org="",
                notes=org_notes,
            )
        )
    ev = {
        "paper_id": paper_id,
        "entries": [
            {"device_id": f"{paper_id}-a", "field": "vpi_dc_v", "value": 2.5, "basis": "measured", "locator": "p.1"}
        ],
    }
    (d / "evidence" / f"{paper_id}.yaml").write_text(yaml.safe_dump(ev))
    return d


def test_new_paper_merges_and_validates(tmp_path: Path) -> None:
    base = make_db(tmp_path)
    d = stage(tmp_path, "s1", base, "new2026")
    msgs, counts = merge([d], base, set())
    assert msgs == [] and counts["papers"] == 1 and counts["devices"] == 1 and counts["orgs"] == 1
    assert validate(base) == []


def test_duplicate_paper_is_conflict_unless_replaced(tmp_path: Path) -> None:
    base = make_db(tmp_path)
    d = stage(tmp_path, "s1", base, "test2026")
    msgs, _ = merge([d], base, set())
    assert any("CONFLICT paper_id test2026" in m for m in msgs)
    base2 = make_db(tmp_path / "again")
    d2 = stage(tmp_path / "again", "s1", base2, "test2026")
    msgs2, counts2 = merge([d2], base2, {"test2026"})
    assert not any("paper_id" in m for m in msgs2)
    assert counts2["papers"] == 1
    devices = read_csv(base2 / "devices.csv")[1]
    assert [r["device_id"] for r in devices] == ["test2026-a"]  # old rows replaced, not duplicated


def test_org_conflict_only_on_type_country_region_not_notes(tmp_path: Path) -> None:
    base = make_db(tmp_path)
    d1 = stage(tmp_path, "s1", base, "a2026", org_notes="first")
    d2 = stage(tmp_path, "s2", base, "b2026", org_notes="second notes differ")
    msgs, _ = merge([d1, d2], base, set())
    assert msgs == []
    d3 = stage(tmp_path, "s3", base, "c2026", org_type="company")
    msgs3, _ = merge([d3], base, set())
    assert any("CONFLICT org" in m for m in msgs3)


def test_existing_evidence_file_is_never_overwritten(tmp_path: Path) -> None:
    base = make_db(tmp_path)
    shutil.copyfile(base / "evidence/test2026.yaml", base / "evidence/dup2026.yaml")
    d = stage(tmp_path, "s1", base, "dup2026")
    msgs, _ = merge([d], base, set())
    assert any("CONFLICT evidence dup2026.yaml" in m for m in msgs)
