from csv import DictWriter
from io import StringIO

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import DbSession
from app.models import EntityTag, InteractionEvent, Organization, Person, PersonOrganization, Reminder, Tag

router = APIRouter(prefix="/exports", tags=["exports"])


def csv_response(filename: str, rows: list[dict[str, object]], fieldnames: list[str]) -> StreamingResponse:
    buffer = StringIO()
    writer = DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/people-csv")
def export_people_csv(db: DbSession) -> StreamingResponse:
    people = list(db.scalars(select(Person).where(Person.deleted_at.is_(None)).order_by(Person.display_name)))
    rows: list[dict[str, object]] = [
        {
            "id": person.id,
            "display_name": person.display_name,
            "primary_location": person.primary_location or "",
            "relationship_category": person.relationship_category,
            "relationship_score": person.relationship_score,
            "last_interaction_date": person.last_interaction_date.isoformat() if person.last_interaction_date else "",
            "next_reminder_date": person.next_reminder_date.isoformat() if person.next_reminder_date else "",
            "organizations": ", ".join(
                db.scalars(
                    select(Organization.name)
                    .join(PersonOrganization, PersonOrganization.organization_id == Organization.id)
                    .where(PersonOrganization.person_id == person.id)
                )
            ),
            "tags": ", ".join(
                db.scalars(
                    select(Tag.name)
                    .join(EntityTag, EntityTag.tag_id == Tag.id)
                    .where(EntityTag.entity_type == "Person", EntityTag.entity_id == person.id)
                )
            ),
            "notes": person.notes or "",
        }
        for person in people
    ]
    return csv_response(
        "people.csv",
        rows,
        [
            "id",
            "display_name",
            "primary_location",
            "relationship_category",
            "relationship_score",
            "last_interaction_date",
            "next_reminder_date",
            "organizations",
            "tags",
            "notes",
        ],
    )


@router.get("/organizations-csv")
def export_organizations_csv(db: DbSession) -> StreamingResponse:
    organizations = list(
        db.scalars(select(Organization).where(Organization.deleted_at.is_(None)).order_by(Organization.name))
    )
    rows: list[dict[str, object]] = [
        {
            "id": organization.id,
            "name": organization.name,
            "type": organization.type,
            "industry": organization.industry or "",
            "location": organization.location or "",
            "website": organization.website or "",
            "tags": ", ".join(
                db.scalars(
                    select(Tag.name)
                    .join(EntityTag, EntityTag.tag_id == Tag.id)
                    .where(EntityTag.entity_type == "Organization", EntityTag.entity_id == organization.id)
                )
            ),
            "notes": organization.notes or "",
        }
        for organization in organizations
    ]
    return csv_response(
        "organizations.csv",
        rows,
        ["id", "name", "type", "industry", "location", "website", "tags", "notes"],
    )


@router.get("/events-csv")
def export_events_csv(db: DbSession) -> StreamingResponse:
    events = list(
        db.scalars(select(InteractionEvent).where(InteractionEvent.deleted_at.is_(None)).order_by(InteractionEvent.started_at.desc()))
    )
    rows: list[dict[str, object]] = [
        {
            "id": event.id,
            "title": event.title,
            "type": event.type,
            "started_at": event.started_at.isoformat(),
            "duration_minutes": event.duration_minutes or "",
            "context": event.context or "",
            "summary": event.summary or "",
            "notes": event.notes or "",
        }
        for event in events
    ]
    return csv_response(
        "events.csv",
        rows,
        ["id", "title", "type", "started_at", "duration_minutes", "context", "summary", "notes"],
    )


@router.get("/reminders-csv")
def export_reminders_csv(db: DbSession) -> StreamingResponse:
    reminders = list(db.scalars(select(Reminder).where(Reminder.deleted_at.is_(None)).order_by(Reminder.due_at)))
    rows: list[dict[str, object]] = [
        {
            "id": reminder.id,
            "title": reminder.title,
            "due_at": reminder.due_at.isoformat(),
            "status": reminder.status,
            "priority": reminder.priority,
            "entity_type": reminder.entity_type or "",
            "entity_id": reminder.entity_id or "",
            "notes": reminder.notes or "",
        }
        for reminder in reminders
    ]
    return csv_response(
        "reminders.csv",
        rows,
        ["id", "title", "due_at", "status", "priority", "entity_type", "entity_id", "notes"],
    )
