"""Tests for scripts/validate_db.py using a tiny synthetic database written to a tmp dir."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_db import ORG_COLUMNS, validate  # noqa: E402


def make_db(tmp: Path) -> Path:
    data = tmp / "data"
    (data / "schema").mkdir(parents=True)
    (data / "evidence").mkdir()
    shutil.copyfile(ROOT / "data/schema/devices.schema.yaml", data / "schema/devices.schema.yaml")
    schema = yaml.safe_load((data / "schema/devices.schema.yaml").read_text())
    p_cols = [c["name"] for c in schema["papers_columns"]]
    d_cols = [c["name"] for c in schema["devices_columns"]]
    paper = dict.fromkeys(p_cols, "")
    paper.update(
        paper_id="test2026",
        title="T",
        authors="A;B",
        year="2026",
        url="https://doi.org/10.0/x",
        source_type="journal",
        access="open_access",
        redistribution="restricted_local_only",
        discovered_via="web",
        cache_status="full_extract",
        verified_on="2026-10-01",
        universities="Test University",
        countries="US",
    )
    dev = dict.fromkeys(d_cols, "")
    dev.update(
        device_id="test2026-a",
        paper_id="test2026",
        device_label="a",
        device_class="mzm",
        eo_material="lithium_niobate",
        vpi_dc_v="2.5",
        evidence_ref="data/evidence/test2026.yaml",
    )
    for name, cols, rows in (("papers.csv", p_cols, [paper]), ("devices.csv", d_cols, [dev])):
        with (data / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    with (data / "organizations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ORG_COLUMNS)
        w.writeheader()
        w.writerow(
            dict(
                org_name="Test University",
                org_type="university",
                country="US",
                region="north_america",
                parent_org="",
                notes="",
            )
        )
    ev = {
        "paper_id": "test2026",
        "entries": [
            {"device_id": "test2026-a", "field": "vpi_dc_v", "value": 2.5, "basis": "measured", "locator": "p.1"}
        ],
    }
    (data / "evidence/test2026.yaml").write_text(yaml.safe_dump(ev))
    return data


def test_valid_db_has_no_errors(tmp_path: Path) -> None:
    assert validate(make_db(tmp_path)) == []


def test_value_without_evidence_is_rejected(tmp_path: Path) -> None:
    data = make_db(tmp_path)
    (data / "evidence/test2026.yaml").write_text(yaml.safe_dump({"paper_id": "test2026", "entries": []}))
    errs = validate(data)
    assert any("no evidence entry" in e for e in errs)


def test_evidence_value_mismatch_is_rejected(tmp_path: Path) -> None:
    data = make_db(tmp_path)
    ev = yaml.safe_load((data / "evidence/test2026.yaml").read_text())
    ev["entries"][0]["value"] = 3.0
    (data / "evidence/test2026.yaml").write_text(yaml.safe_dump(ev))
    assert any("!= CSV" in e for e in validate(data))


def test_unknown_org_and_tweet_url_rejected(tmp_path: Path) -> None:
    data = make_db(tmp_path)
    rows = list(csv.DictReader((data / "papers.csv").open()))
    rows[0]["universities"] = "Nowhere University"
    rows[0]["url"] = "https://x.com/someone/status/1"
    with (data / "papers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    errs = validate(data)
    assert any("not in organizations.csv" in e for e in errs)
    assert any("not a tweet" in e for e in errs)
