"""Create empty data/papers.csv, data/devices.csv, data/organizations.csv with schema headers (never overwrites)."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ORG_COLUMNS = ["org_name", "org_type", "country", "region", "parent_org", "notes"]


def main() -> None:
    schema = yaml.safe_load((ROOT / "data/schema/devices.schema.yaml").read_text())
    targets = {
        "papers.csv": [c["name"] for c in schema["papers_columns"]],
        "devices.csv": [c["name"] for c in schema["devices_columns"]],
        "organizations.csv": ORG_COLUMNS,
    }
    for name, cols in targets.items():
        path = ROOT / "data" / name
        if path.exists():
            print(f"exists, skipped: {path}")
            continue
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(cols)
        print(f"created {path}")


if __name__ == "__main__":
    main()
