from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import InteractionEvent, Person, event_people

TYPE_WEIGHTS = {
    "One-on-one": 1.6,
    "Meeting": 1.2,
    "Meal": 1.5,
    "Call": 1.0,
    "Message": 0.45,
    "Email": 0.35,
    "Intro": 0.9,
    "Work session": 1.4,
    "Event attendance": 0.7,
    "Personal milestone": 1.3,
    "Note": 0.25,
    "Other": 0.8,
}


def relationship_category_for_score(score: int) -> str:
    if score >= 85:
        return "Close"
    if score >= 55:
        return "Strong"
    if score >= 32:
        return "Warm"
    if score >= 16:
        return "Light"
    if score >= 6:
        return "Dormant"
    return "New"


def relationship_reason_for_person(person: Person, event_count: int) -> str:
    if not person.last_interaction_date or event_count == 0:
        return "No interaction history yet."

    last_date = person.last_interaction_date.date().isoformat()
    return f"{event_count} events logged; last interaction on {last_date}; score favors recent higher-signal activity."


def _recency_factor(started_at: datetime, now: datetime) -> float:
    age_days = max((now - started_at).days, 0)
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.72
    if age_days <= 180:
        return 0.48
    return 0.28


def recompute_person_relationship(db: Session, person_id: UUID) -> None:
    person = db.get(Person, person_id)
    if not person:
        return

    events = list(
        db.scalars(
            select(InteractionEvent)
            .join(event_people, event_people.c.event_id == InteractionEvent.id)
            .where(
                event_people.c.person_id == person_id,
                InteractionEvent.deleted_at.is_(None),
            )
            .order_by(InteractionEvent.started_at.desc())
        )
    )

    if not events:
        person.last_interaction_date = None
        person.relationship_score = 0
        person.relationship_category = "New"
        return

    now = datetime.now(timezone.utc)
    total = 0.0

    for event in events:
        participant_count = db.scalar(
            select(func.count())
            .select_from(event_people)
            .where(event_people.c.event_id == event.id)
        ) or 1
        type_weight = TYPE_WEIGHTS.get(event.type, TYPE_WEIGHTS["Other"])
        duration_factor = 1 + min((event.duration_minutes or 0) / 60, 2) * 0.4
        group_factor = 1.0 if participant_count <= 2 else 0.7
        total += 10 * type_weight * duration_factor * group_factor * _recency_factor(event.started_at, now)

    person.last_interaction_date = events[0].started_at
    person.relationship_score = int(round(total))
    person.relationship_category = relationship_category_for_score(person.relationship_score)


def recompute_people_relationships(db: Session, person_ids: set[UUID]) -> None:
    for person_id in person_ids:
        recompute_person_relationship(db, person_id)
