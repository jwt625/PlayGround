from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession
from app.models import InteractionEvent, Location, Organization, Person
from app.schemas.event import EventCreate, EventRead, EventUpdate
from app.services.relationship_scoring import recompute_people_relationships

router = APIRouter(prefix="/events", tags=["events"])


def event_statement() -> Select[tuple[InteractionEvent]]:
    return (
        select(InteractionEvent)
        .options(
            selectinload(InteractionEvent.people),
            selectinload(InteractionEvent.organizations),
            selectinload(InteractionEvent.locations),
        )
        .where(InteractionEvent.deleted_at.is_(None))
    )


def serialize_event(event: InteractionEvent) -> EventRead:
    return EventRead.model_validate(
        {
            **event.__dict__,
            "person_ids": [person.id for person in event.people],
            "organization_ids": [organization.id for organization in event.organizations],
            "location_ids": [location.id for location in event.locations],
        }
    )


def assign_event_relations(db: DbSession, event: InteractionEvent, payload: EventCreate | EventUpdate) -> None:
    if payload.person_ids is not None:
        event.people = list(db.scalars(select(Person).where(Person.id.in_(payload.person_ids)))) if payload.person_ids else []
    if payload.organization_ids is not None:
        event.organizations = (
            list(db.scalars(select(Organization).where(Organization.id.in_(payload.organization_ids))))
            if payload.organization_ids
            else []
        )
    if payload.location_ids is not None:
        event.locations = list(db.scalars(select(Location).where(Location.id.in_(payload.location_ids)))) if payload.location_ids else []


@router.get("", response_model=list[EventRead])
def list_events(
    db: DbSession,
    response: Response,
    q: str | None = Query(default=None),
    person_id: UUID | None = None,
    organization_id: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[EventRead]:
    statement = event_statement()
    count_statement = select(func.count()).select_from(InteractionEvent).where(InteractionEvent.deleted_at.is_(None))
    if q:
        pattern = f"%{q}%"
        search_filter = or_(
            InteractionEvent.title.ilike(pattern),
            InteractionEvent.context.ilike(pattern),
            InteractionEvent.summary.ilike(pattern),
            InteractionEvent.notes.ilike(pattern),
        )
        statement = statement.where(search_filter)
        count_statement = count_statement.where(search_filter)
    if person_id:
        person_filter = InteractionEvent.people.any(Person.id == person_id)
        statement = statement.where(person_filter)
        count_statement = count_statement.where(person_filter)
    if organization_id:
        organization_filter = InteractionEvent.organizations.any(Organization.id == organization_id)
        statement = statement.where(organization_filter)
        count_statement = count_statement.where(organization_filter)
    response.headers["X-Total-Count"] = str(db.scalar(count_statement) or 0)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Offset"] = str(offset)
    statement = statement.order_by(InteractionEvent.started_at.desc()).limit(limit).offset(offset)
    return [serialize_event(event) for event in db.scalars(statement)]


@router.post("", response_model=EventRead, status_code=status.HTTP_201_CREATED)
def create_event(payload: EventCreate, db: DbSession) -> EventRead:
    event = InteractionEvent(**payload.model_dump(exclude={"person_ids", "organization_ids", "location_ids"}))
    assign_event_relations(db, event, payload)
    touched_people = {person.id for person in event.people}
    db.add(event)
    db.flush()
    recompute_people_relationships(db, touched_people)
    db.commit()
    db.refresh(event)
    return get_event(event.id, db)


@router.get("/{event_id}", response_model=EventRead)
def get_event(event_id: UUID, db: DbSession) -> EventRead:
    event = db.scalar(event_statement().where(InteractionEvent.id == event_id))
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return serialize_event(event)


@router.patch("/{event_id}", response_model=EventRead)
def update_event(event_id: UUID, payload: EventUpdate, db: DbSession) -> EventRead:
    event = db.scalar(event_statement().where(InteractionEvent.id == event_id))
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")

    touched_people = {person.id for person in event.people}
    for field, value in payload.model_dump(exclude_unset=True, exclude={"person_ids", "organization_ids", "location_ids"}).items():
        setattr(event, field, value)
    assign_event_relations(db, event, payload)
    touched_people.update(person.id for person in event.people)
    recompute_people_relationships(db, touched_people)
    db.commit()
    db.refresh(event)
    return get_event(event.id, db)


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_event(event_id: UUID, db: DbSession) -> None:
    event = db.scalar(event_statement().where(InteractionEvent.id == event_id))
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    touched_people = {person.id for person in event.people}
    from datetime import datetime, timezone

    event.deleted_at = datetime.now(timezone.utc)
    recompute_people_relationships(db, touched_people)
    db.commit()
