#!/usr/bin/env python3
"""
Batch planning and scheduling helpers for OFC PDF caching.

This script does not fetch PDFs itself. It prepares reproducible plans and
inspects candidate inventory so a separate worker can be called later.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path(".cache/ofc")
DEFAULT_PLAN_DIR = Path("output/pdf_batch_plans")


@dataclass
class BatchPlanItem:
    run_at_local: str
    batch_size: int


@dataclass
class BatchCandidate:
    session_id: int
    presentation_id: int | None
    presentation_code: str
    presentation_title: str
    best_pdf_link: str
    paper_link_status: str
    paper_cached: bool


def load_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def eligible_candidates(records: list[dict[str, Any]]) -> list[BatchCandidate]:
    candidates: list[BatchCandidate] = []
    for row in records:
        if not row.get("best_pdf_link"):
            continue
        if row.get("paper_cached"):
            continue
        status = row.get("paper_link_status", "not_tested")
        if status not in {"not_tested", "authenticated_pdf_url", "html_response"}:
            continue
        candidates.append(
            BatchCandidate(
                session_id=row["session_id"],
                presentation_id=row.get("presentation_id"),
                presentation_code=row.get("presentation_code", ""),
                presentation_title=row.get("presentation_title", ""),
                best_pdf_link=row["best_pdf_link"],
                paper_link_status=status,
                paper_cached=bool(row.get("paper_cached", False)),
            )
        )
    return candidates


def make_daily_plan(
    *,
    seed: int,
    start: datetime,
    active_hours_min: int,
    active_hours_max: int,
    active_minutes_min: int,
    active_minutes_max: int,
    batch_size_min: int,
    batch_size_max: int,
) -> list[BatchPlanItem]:
    rng = random.Random(seed)
    hour_count = rng.randint(active_hours_min, active_hours_max)
    selected_hours = sorted(rng.sample(range(24), k=hour_count))

    items: list[BatchPlanItem] = []
    for hour in selected_hours:
        minute_count = rng.randint(active_minutes_min, active_minutes_max)
        chosen_minutes = sorted(rng.sample(range(60), k=minute_count))
        for minute in chosen_minutes:
            run_at = start.replace(hour=hour, minute=minute, second=0, microsecond=0)
            items.append(
                BatchPlanItem(
                    run_at_local=run_at.isoformat(),
                    batch_size=rng.randint(batch_size_min, batch_size_max),
                )
            )
    items.sort(key=lambda item: item.run_at_local)
    return items


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--records-json", required=True)

    plan_parser = subparsers.add_parser("plan-day")
    plan_parser.add_argument("--out", default=str(DEFAULT_PLAN_DIR / "daily-plan.json"))
    plan_parser.add_argument("--seed", type=int, default=1)
    plan_parser.add_argument("--date", default=datetime.now().date().isoformat())
    plan_parser.add_argument("--active-hours-min", type=int, default=2)
    plan_parser.add_argument("--active-hours-max", type=int, default=3)
    plan_parser.add_argument("--active-minutes-min", type=int, default=5)
    plan_parser.add_argument("--active-minutes-max", type=int, default=10)
    plan_parser.add_argument("--batch-size-min", type=int, default=3)
    plan_parser.add_argument("--batch-size-max", type=int, default=5)

    args = parser.parse_args()

    if args.command == "inspect":
        records = load_records(Path(args.records_json))
        candidates = eligible_candidates(records)
        summary = {
            "eligible_candidates": len(candidates),
            "sample": [asdict(item) for item in candidates[:10]],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "plan-day":
        start = datetime.fromisoformat(f"{args.date}T00:00:00")
        items = make_daily_plan(
            seed=args.seed,
            start=start,
            active_hours_min=args.active_hours_min,
            active_hours_max=args.active_hours_max,
            active_minutes_min=args.active_minutes_min,
            active_minutes_max=args.active_minutes_max,
            batch_size_min=args.batch_size_min,
            batch_size_max=args.batch_size_max,
        )
        payload = {
            "date": args.date,
            "seed": args.seed,
            "items": [asdict(item) for item in items],
            "planned_total_papers": sum(item.batch_size for item in items),
        }
        write_json(Path(args.out), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
