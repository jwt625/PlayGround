"""Re-extract every cached PDF with the current extractor and list papers whose garbled flag changed.
Writes logs/recheck_text_layers_<ts>.jsonl. Usage: uv run python scripts/recheck_text_layers.py"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_source import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
log = ROOT / "logs" / f"recheck_text_layers_{dt.datetime.now():%Y%m%dT%H%M%S}.jsonl"
for d in sorted((ROOT / "references").iterdir()):
    if d.name == "_inbox" or not (d / "source.pdf").exists():
        continue
    before = json.loads((d / "source.json").read_text()).get("garbled_text_layer") if (d / "source.json").exists() else None
    info = extract(d.name)
    row = {"paper_id": d.name, "before": before, "after": info["garbled_text_layer"],
           "pages": info.get("garbled_pages"), "number_missing_pages": info.get("number_missing_pages")}
    with log.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
