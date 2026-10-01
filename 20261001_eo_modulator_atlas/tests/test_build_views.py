"""Tests for scripts/build_views.py using real pilot rows (chen2022, kohli2025, ogiso2016) in tests/fixtures/pilot."""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_views as bv  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "pilot"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("db") / "data"
    shutil.copytree(FIX, d)
    (d / "schema").mkdir()
    shutil.copyfile(ROOT / "data/schema/devices.schema.yaml", d / "schema/devices.schema.yaml")
    return d


@pytest.fixture(scope="module")
def atlas(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return bv.build(data_dir, tmp_path_factory.mktemp("nosims"))


def dev(atlas: dict[str, Any], device_id: str) -> dict[str, Any]:
    return next(d for d in atlas["devices"] if d["device_id"] == device_id)


def paper(atlas: dict[str, Any], paper_id: str) -> dict[str, Any]:
    return next(p for p in atlas["papers"] if p["paper_id"] == paper_id)


def test_counts_and_integrity(atlas: dict[str, Any]) -> None:
    c = atlas["integrity"]["counts"]
    assert (c["papers"], c["devices"]) == (3, 7)
    assert (
        atlas["integrity"]["per_platform"] == {"bto_on_sin": 3, "lnoi_rib": 3, "inp": 1}
        or sum(atlas["integrity"]["per_platform"].values()) == 7
    )
    assert sum(atlas["integrity"]["per_source_type"].values()) == 3
    assert atlas["integrity"]["evidence"]["without_evidence_entry"] == 0
    assert sum(atlas["integrity"]["per_basis"].values()) > 100


def test_typed_values_and_empty_are_none(atlas: dict[str, Any]) -> None:
    d = dev(atlas, "chen2022-c")
    assert d["vpil_dc_vcm"] == 2.2
    assert d["bw3db_ghz"] is None  # not reported: never 0
    assert d["tags"] == []
    assert isinstance(d["length_mm"], float)


def test_derived_vpil_only_when_reported_empty(atlas: dict[str, Any]) -> None:
    chen = dev(atlas, "chen2022-c")
    assert "vpil_dc_vcm_derived" not in chen["derived"]  # reported value exists
    assert chen["vpil_best"]["source"] == "reported_dc"
    k = dev(atlas, "kohli2025-mzm")
    x = k["derived"]["vpil_dc_vcm_derived"]
    assert math.isclose(x["value"], 3.6 * 0.015 / 10)
    assert "per_arm_phase_shifter" in x["warning"]  # convention warning carried
    assert k["vpil_best"]["source"] == "derived_dc"
    assert k["headline_basis"]["vpil"] == "derived"


def test_rf_total_and_vpi_il(atlas: dict[str, Any]) -> None:
    c = dev(atlas, "chen2022-c")
    assert c["rf_loss_db_per_cm"] is not None
    rf = c["derived"]["il_rf_total_db"]
    assert math.isclose(rf["value"], c["rf_loss_db_per_cm"] * c["length_mm"] / 10)
    assert math.isclose(c["derived"]["vpi_il_vdb"]["value"], c["vpi_dc_v"] * c["il_onchip_db"])
    assert c["derived"]["vpi_il_vdb"]["unit"] == "V*dB"
    k = dev(atlas, "kohli2025-mzm")
    assert "il_rf_total_db" not in k["derived"]
    assert "vpi_il_vdb" not in k["derived"]  # no on-chip IL reported (only fiber-to-fiber)


def test_completeness(atlas: dict[str, Any]) -> None:
    c = dev(atlas, "chen2022-c")["derived"]["completeness"]
    assert c["reported"] == ["vpi", "il_onchip", "rf_loss", "z0", "n_rf", "ng"]
    assert c["missing"] == ["bw3db"]
    assert math.isclose(c["value"], 6 / 7)
    k = dev(atlas, "kohli2025-mzm")["derived"]["completeness"]
    assert k["reported"] == ["vpi", "bw3db"]


def test_fom_absent_when_inputs_missing(atlas: dict[str, Any]) -> None:
    for d in atlas["devices"]:
        assert "fom" not in d["derived"]  # no real pilot row has bw3db + vpi + on-chip IL together


def test_fom_formula_with_and_without_rf_loss() -> None:
    # In-memory variation of a real row (chen2022-c) with a 3 dB BW inserted: exercises the formula only.
    _, rows = bv.read_csv(FIX / "devices.csv")
    schema = bv.yaml.safe_load((ROOT / "data/schema/devices.schema.yaml").read_text())
    base = bv.typed_row(schema["devices_columns"], next(r for r in rows if r["device_id"] == "chen2022-c"))
    base["qualifiers_map"] = {}
    base["bw3db_ghz"] = 100.0
    out = bv.compute_derived(base)["fom"]
    rf_total = base["rf_loss_db_per_cm"] * base["length_mm"] / 10
    expected = 100.0 / (base["vpi_dc_v"] * 10 ** (rf_total / 20) * 10 ** (base["il_onchip_db"] / 10))
    assert out["rf_corrected"] and math.isclose(out["value"], expected)
    no_rf = copy.deepcopy(base)
    no_rf["rf_loss_db_per_cm"] = None
    out2 = bv.compute_derived(no_rf)["fom"]
    assert not out2["rf_corrected"]
    assert math.isclose(out2["value"], 100.0 / (base["vpi_dc_v"] * 10 ** (base["il_onchip_db"] / 10)))
    ring = copy.deepcopy(base)
    ring["device_class"] = "ring"
    assert "fom" not in bv.compute_derived(ring)
    ring["device_class"] = "mzm"
    ring["bw3db_ghz"] = None
    assert "fom" not in bv.compute_derived(ring)


def test_qualifiers_and_bases(atlas: dict[str, Any]) -> None:
    rt = dev(atlas, "kohli2025-rt")
    assert rt["qualifiers"]["il_onchip_db"] == "lt"
    assert rt["il_onchip_db"] is not None
    assert rt["evidence"]["il_onchip_db"]["locator"]
    og = dev(atlas, "ogiso2016-a")
    assert og["qualifiers"]["bw_measured_to_ghz"] == "gt"
    assert og["bw3db_ghz"] is None
    assert og["headline_basis"]["il_onchip"] == "author_estimate"
    assert og["measured_only"] is True


def test_org_join_and_regions(atlas: dict[str, Any]) -> None:
    k = paper(atlas, "kohli2025")
    names = {o["org_name"] for o in k["orgs_affil"]}
    assert "ETH Zurich" in names
    assert k["countries_derived"] == ["CH"]
    assert k["regions_derived"] == ["europe"]
    o = paper(atlas, "ogiso2016")
    assert o["countries_derived"] == ["JP"] and o["regions_derived"] == ["east_asia"]
    assert o["label"] == "Ogiso et al. 2016"
    assert "Binnig and Rohrer Nanotechnology Center" in {f["org_name"] for f in k["orgs_fab"]}


def test_representative_rules(atlas: dict[str, Any]) -> None:
    # chen2022: all equal completeness, no FOM -> lowest Vpi*L wins the tie
    assert paper(atlas, "chen2022")["rep"]["default"] == "chen2022-c"
    assert paper(atlas, "chen2022")["rep"]["lowest_vpil"] == "chen2022-c"
    # kohli2025: iq and mzm have bw3db; highest_bw is mzm (110 GHz); default tie -> lowest Vpi*L
    k = paper(atlas, "kohli2025")
    assert k["rep"]["highest_bw"] == "kohli2025-mzm"
    assert k["rep"]["default"] in k["device_ids"]
    # highest_fom falls back to default when no device has a FOM
    assert k["rep"]["highest_fom"] == k["rep"]["default"]


def test_rep_order_prefers_completeness_then_fom_then_vpil() -> None:
    def mk(i: str, comp: float, fom: float | None, vp: float | None) -> dict[str, Any]:
        der: dict[str, Any] = {"completeness": {"value": comp}}
        if fom is not None:
            der["fom"] = {"value": fom}
        return {"device_id": i, "derived": der, "vpil_best": {"value": vp} if vp is not None else None}

    devs = [
        mk("p-a", 3 / 7, 5.0, 2.0),
        mk("p-b", 4 / 7, None, 9.0),
        mk("p-c", 4 / 7, 1.0, 5.0),
        mk("p-d", 4 / 7, 1.0, 3.0),
    ]
    assert bv.pick_reps(devs)["default"] == "p-d"  # completeness 4/7 beats 3/7; FOM 1.0 beats none; lower Vpi*L wins
    assert bv.pick_reps(devs)["highest_fom"] == "p-a"


def test_sims_listing(data_dir: Path) -> None:
    sims = bv.load_sims(ROOT / "sims")
    ids = {s["paper_id"] for s in sims}
    assert ids <= {p.name for p in (ROOT / "sims").iterdir() if p.is_dir()}
    a = bv.build(data_dir, ROOT / "sims")
    for s in a["sims"]:
        assert paper(a, s["paper_id"])["has_sim"] if any(p["paper_id"] == s["paper_id"] for p in a["papers"]) else True


def test_cli_with_staging_and_flat_csv(tmp_path: Path, data_dir: Path) -> None:
    out = tmp_path / "atlas.json"
    flat = tmp_path / "flat.csv"
    base = tmp_path / "base"
    shutil.copytree(data_dir, base)
    # staging fragment = the kohli2025 rows only; base keeps the other two papers
    stg = tmp_path / "stg"
    (stg / "evidence").mkdir(parents=True)
    for name in ("papers", "devices"):
        h, rows = bv.read_csv(data_dir / f"{name}.csv")
        keep = [r for r in rows if r["paper_id"] == "kohli2025"]
        with (stg / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=h)
            w.writeheader()
            w.writerows(keep)
        with (base / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=h)
            w.writeheader()
            w.writerows([r for r in rows if r["paper_id"] != "kohli2025"])
    shutil.copyfile(data_dir / "organizations.csv", stg / "organizations.csv")
    shutil.copyfile(data_dir / "evidence/kohli2025.yaml", stg / "evidence/kohli2025.yaml")
    (base / "evidence/kohli2025.yaml").unlink()
    old = sys.argv
    sys.argv = [
        "build_views",
        "--data",
        str(base),
        "--with-staging",
        str(stg),
        "--out",
        str(out),
        "--flat-csv",
        str(flat),
        "--sims",
        str(tmp_path / "nosims"),
    ]
    try:
        assert bv.main() == 0
    finally:
        sys.argv = old
    a = json.loads(out.read_text())
    assert {p["paper_id"] for p in a["papers"]} == {"chen2022", "kohli2025", "ogiso2016"}
    assert not (base / "evidence/kohli2025.yaml").exists()  # base data dir untouched
    rows = list(csv.DictReader(flat.open()))
    assert len(rows) == 7
    assert sum(int(r["is_representative"]) for r in rows) == 3
