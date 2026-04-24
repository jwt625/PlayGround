from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ContactMethod, InteractionEvent, Organization, Person, Reminder


def seed_demo_data(db: Session) -> None:
    if db.scalar(select(Person.id).limit(1)):
        return

    organization = Organization(
        name="Kizuna Labs",
        type="Company",
        industry="Relationship software",
        location="San Francisco, USA",
        website="https://example.com",
        description="Demo organization for local testing.",
    )
    person = Person(
        display_name="Ada Lovelace",
        given_name="Ada",
        family_name="Lovelace",
        primary_location="London, UK",
        relationship_summary="Computing pioneer and frequent thought partner.",
        how_we_met="Introduced through a systems history reading group.",
        notes="Strong local demo contact.",
    )
    person.contact_methods = [ContactMethod(type="Email", value="ada@example.com", label="Imported", is_primary=True)]

    event = InteractionEvent(
        title="Coffee with Ada",
        type="One-on-one",
        started_at=datetime.now(timezone.utc) - timedelta(days=2),
        duration_minutes=75,
        summary="Discussed local-first tooling and knowledge systems.",
    )
    event.people = [person]
    event.organizations = [organization]

    reminder = Reminder(
        title="Send Ada the latest prototype notes",
        due_at=datetime.now(timezone.utc) + timedelta(days=2),
        status="Open",
        priority="High",
        entity_type="Person",
    )

    db.add_all([organization, person, event])
    db.flush()
    reminder.entity_id = person.id
    db.add(reminder)
    db.commit()
