from datetime import datetime, timezone
from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession
from app.models import (
    ContactMethod,
    EntityLocation,
    EntityTag,
    ExternalProfile,
    InteractionEvent,
    Location,
    Organization,
    Person,
    PersonOrganization,
    PipelineItem,
    Reminder,
    Tag,
)
from app.schemas.event import EventRead
from app.schemas.metadata import EntityLocationCreate, EntityLocationRead, EntityTagRead, PersonOrganizationCreate, TagCreate
from app.schemas.person import (
    ContactMethodCreate,
    ExternalProfileCreate,
    PersonCreate,
    PersonDetailRead,
    PersonRead,
    PersonUpdate,
)
from app.schemas.pipeline import PipelineItemRead
from app.schemas.reminder import ReminderRead
from app.services.relationship_scoring import relationship_reason_for_person

router = APIRouter(prefix="/people", tags=["people"])


def person_detail_statement() -> Select[tuple[Person]]:
    return (
        select(Person)
        .options(
            selectinload(Person.contact_methods),
            selectinload(Person.external_profiles),
            selectinload(Person.organization_roles),
        )
        .where(Person.deleted_at.is_(None))
    )


@router.get("", response_model=list[PersonRead])
def list_people(
    db: DbSession,
    response: Response,
    q: str | None = Query(default=None),
    relationship_category: str | None = None,
    city: str | None = None,
    organization_id: UUID | None = None,
    tag: str | None = None,
    reminder_status: str | None = None,
    pipeline_id: UUID | None = None,
    pipeline_stage_id: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Person]:
    statement = person_detail_statement()
    count_statement = select(func.count()).select_from(Person).where(Person.deleted_at.is_(None))
    if q:
        pattern = f"%{q}%"
        search_filter = or_(
            Person.display_name.ilike(pattern),
            Person.given_name.ilike(pattern),
            Person.family_name.ilike(pattern),
            Person.nickname.ilike(pattern),
            Person.primary_location.ilike(pattern),
            Person.notes.ilike(pattern),
            Person.relationship_summary.ilike(pattern),
            Person.how_we_met.ilike(pattern),
        )
        statement = statement.where(search_filter)
        count_statement = count_statement.where(search_filter)
    if relationship_category:
        category_filter = Person.relationship_category == relationship_category
        statement = statement.where(category_filter)
        count_statement = count_statement.where(category_filter)
    if city:
        city_filter = Person.primary_location.ilike(f"%{city}%")
        statement = statement.where(city_filter)
        count_statement = count_statement.where(city_filter)
    if organization_id:
        organization_filter = Person.organization_roles.any(PersonOrganization.organization_id == organization_id)
        statement = statement.where(organization_filter)
        count_statement = count_statement.where(organization_filter)
    if tag:
        tag_filter = Person.id.in_(
            select(EntityTag.entity_id)
            .join(Tag, Tag.id == EntityTag.tag_id)
            .where(EntityTag.entity_type == "Person", Tag.name == tag)
        )
        statement = statement.where(tag_filter)
        count_statement = count_statement.where(tag_filter)
    if reminder_status:
        reminder_filter = Person.id.in_(
            select(Reminder.entity_id).where(
                Reminder.deleted_at.is_(None),
                Reminder.entity_type == "Person",
                Reminder.status == reminder_status,
            )
        )
        statement = statement.where(reminder_filter)
        count_statement = count_statement.where(reminder_filter)
    if pipeline_id:
        pipeline_filter = Person.id.in_(select(PipelineItem.primary_person_id).where(PipelineItem.pipeline_id == pipeline_id))
        statement = statement.where(pipeline_filter)
        count_statement = count_statement.where(pipeline_filter)
    if pipeline_stage_id:
        stage_filter = Person.id.in_(select(PipelineItem.primary_person_id).where(PipelineItem.stage_id == pipeline_stage_id))
        statement = statement.where(stage_filter)
        count_statement = count_statement.where(stage_filter)
    response.headers["X-Total-Count"] = str(db.scalar(count_statement) or 0)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Offset"] = str(offset)
    statement = statement.order_by(Person.display_name).limit(limit).offset(offset)
    return list(db.scalars(statement))


@router.post("", response_model=PersonDetailRead, status_code=status.HTTP_201_CREATED)
def create_person(payload: PersonCreate, db: DbSession) -> PersonDetailRead:
    existing = db.scalar(select(Person).where(Person.deleted_at.is_(None), Person.display_name == payload.display_name))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Person with this display name already exists")
    data = payload.model_dump(exclude={"contact_methods", "external_profiles"})
    person = Person(**data)
    person.contact_methods = [ContactMethod(**item.model_dump()) for item in payload.contact_methods]
    person.external_profiles = [ExternalProfile(**item.model_dump()) for item in payload.external_profiles]
    db.add(person)
    db.commit()
    db.refresh(person)
    return get_person(person.id, db)


def build_person_detail(person: Person, db: DbSession) -> PersonDetailRead:
    entity_locations = list(
        db.execute(
            select(EntityLocation, Location)
            .join(Location, Location.id == EntityLocation.location_id)
            .where(EntityLocation.entity_type == "Person", EntityLocation.entity_id == person.id)
            .order_by(EntityLocation.created_at.desc())
        )
    )
    entity_tags = list(
        db.execute(
            select(EntityTag, Tag)
            .join(Tag, Tag.id == EntityTag.tag_id)
            .where(EntityTag.entity_type == "Person", EntityTag.entity_id == person.id)
            .order_by(Tag.name)
        )
    )
    recent_events = list(
        db.scalars(
            select(InteractionEvent)
            .where(InteractionEvent.deleted_at.is_(None), InteractionEvent.people.any(Person.id == person.id))
            .options(
                selectinload(InteractionEvent.people),
                selectinload(InteractionEvent.organizations),
                selectinload(InteractionEvent.locations),
            )
            .order_by(InteractionEvent.started_at.desc())
            .limit(25)
        )
    )
    active_reminders = list(
        db.scalars(
            select(Reminder)
            .where(
                Reminder.deleted_at.is_(None),
                Reminder.entity_type == "Person",
                Reminder.entity_id == person.id,
                Reminder.status != "Done",
            )
            .order_by(Reminder.due_at)
            .limit(25)
        )
    )
    pipeline_items = list(
        db.scalars(
            select(PipelineItem)
            .where(PipelineItem.primary_person_id == person.id)
            .order_by(PipelineItem.created_at.desc())
            .limit(25)
        )
    )
    return PersonDetailRead.model_validate(
        {
            **PersonRead.model_validate(person).model_dump(),
            "organization_roles": [
                {
                    "id": role.id,
                    "organization_id": role.organization_id,
                    "organization_name": role.organization.name if role.organization else None,
                    "title": role.title,
                    "role_type": role.role_type,
                    "start_date": role.start_date,
                    "end_date": role.end_date,
                    "is_current": role.is_current,
                    "notes": role.notes,
                }
                for role in person.organization_roles
            ],
            "locations": [
                EntityLocationRead.model_validate(
                    {
                        "id": entity_location.id,
                        "location_id": location.id,
                        "is_primary": entity_location.is_primary,
                        "notes": entity_location.notes,
                        "location": location,
                    }
                )
                for entity_location, location in entity_locations
            ],
            "tags": [
                EntityTagRead.model_validate(
                    {
                        "id": entity_tag.id,
                        "tag_id": tag.id,
                        "tag": tag,
                    }
                )
                for entity_tag, tag in entity_tags
            ],
            "recent_events": [
                EventRead.model_validate(
                    {
                        **event.__dict__,
                        "person_ids": [participant.id for participant in event.people],
                        "organization_ids": [organization.id for organization in event.organizations],
                        "location_ids": [location.id for location in event.locations],
                    }
                )
                for event in recent_events
            ],
            "active_reminders": [ReminderRead.model_validate(reminder) for reminder in active_reminders],
            "pipeline_items": [PipelineItemRead.model_validate(item) for item in pipeline_items],
            "relationship_score_reason": relationship_reason_for_person(person, len(recent_events)),
        }
    )


@router.get("/{person_id}", response_model=PersonDetailRead)
def get_person(person_id: UUID, db: DbSession) -> PersonDetailRead:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    return build_person_detail(person, db)


@router.patch("/{person_id}", response_model=PersonDetailRead)
def update_person(person_id: UUID, payload: PersonUpdate, db: DbSession) -> PersonDetailRead:
    person = get_person(person_id, db)
    db_person = cast(Person, db.get(Person, person.id))
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(db_person, field, value)
    db.commit()
    db.refresh(db_person)
    return get_person(db_person.id, db)


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(person_id: UUID, db: DbSession) -> None:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    person.deleted_at = datetime.now(timezone.utc)
    db.commit()


@router.post("/{person_id}/organization-roles", response_model=PersonDetailRead, status_code=status.HTTP_201_CREATED)
def add_person_organization_role(person_id: UUID, payload: PersonOrganizationCreate, db: DbSession) -> PersonDetailRead:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    organization = db.get(Organization, payload.organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    db.add(PersonOrganization(person_id=person_id, **payload.model_dump()))
    db.commit()
    db.expire_all()
    return get_person(person_id, db)


@router.post("/{person_id}/tags", response_model=PersonDetailRead, status_code=status.HTTP_201_CREATED)
def add_person_tag(person_id: UUID, payload: TagCreate, db: DbSession) -> PersonDetailRead:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    tag = db.scalar(select(Tag).where(Tag.name == payload.name))
    if not tag:
        tag = Tag(**payload.model_dump())
        db.add(tag)
        db.flush()
    existing = db.scalar(
        select(EntityTag).where(EntityTag.entity_type == "Person", EntityTag.entity_id == person_id, EntityTag.tag_id == tag.id)
    )
    if not existing:
        db.add(EntityTag(entity_type="Person", entity_id=person_id, tag_id=tag.id))
    db.commit()
    db.expire_all()
    return get_person(person_id, db)


@router.post("/{person_id}/locations", response_model=PersonDetailRead, status_code=status.HTTP_201_CREATED)
def add_person_location(person_id: UUID, payload: EntityLocationCreate, db: DbSession) -> PersonDetailRead:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    location = Location(**payload.location.model_dump())
    db.add(location)
    db.flush()
    db.add(
        EntityLocation(
            entity_type="Person",
            entity_id=person_id,
            location_id=location.id,
            is_primary=payload.is_primary,
            notes=payload.notes,
        )
    )
    if payload.is_primary:
        parts = [location.city, location.region, location.country]
        person.primary_location = ", ".join(part for part in parts if part) or location.label
    db.commit()
    db.expire_all()
    return get_person(person_id, db)


@router.post("/{person_id}/contact-methods", response_model=PersonDetailRead, status_code=status.HTTP_201_CREATED)
def add_person_contact_method(person_id: UUID, payload: ContactMethodCreate, db: DbSession) -> PersonDetailRead:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    db.add(ContactMethod(person_id=person_id, **payload.model_dump()))
    db.commit()
    db.expire_all()
    return get_person(person_id, db)


@router.post("/{person_id}/external-profiles", response_model=PersonDetailRead, status_code=status.HTTP_201_CREATED)
def add_person_external_profile(person_id: UUID, payload: ExternalProfileCreate, db: DbSession) -> PersonDetailRead:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    db.add(ExternalProfile(person_id=person_id, **payload.model_dump()))
    db.commit()
    db.expire_all()
    return get_person(person_id, db)
