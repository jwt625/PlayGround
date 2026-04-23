from datetime import datetime, timezone
from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import Select, or_, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession
from app.models import ContactMethod, ExternalProfile, Person
from app.schemas.person import PersonCreate, PersonRead, PersonUpdate

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
    q: str | None = Query(default=None),
    relationship_category: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Person]:
    statement = person_detail_statement()
    if q:
        pattern = f"%{q}%"
        statement = statement.where(
            or_(
                Person.display_name.ilike(pattern),
                Person.given_name.ilike(pattern),
                Person.family_name.ilike(pattern),
                Person.nickname.ilike(pattern),
                Person.primary_location.ilike(pattern),
                Person.notes.ilike(pattern),
                Person.relationship_summary.ilike(pattern),
                Person.how_we_met.ilike(pattern),
            )
        )
    if relationship_category:
        statement = statement.where(Person.relationship_category == relationship_category)
    statement = statement.order_by(Person.display_name).limit(limit).offset(offset)
    return list(db.scalars(statement))


@router.post("", response_model=PersonRead, status_code=status.HTTP_201_CREATED)
def create_person(payload: PersonCreate, db: DbSession) -> Person:
    data = payload.model_dump(exclude={"contact_methods", "external_profiles"})
    person = Person(**data)
    person.contact_methods = [ContactMethod(**item.model_dump()) for item in payload.contact_methods]
    person.external_profiles = [ExternalProfile(**item.model_dump()) for item in payload.external_profiles]
    db.add(person)
    db.commit()
    db.refresh(person)
    return get_person(person.id, db)


@router.get("/{person_id}", response_model=PersonRead)
def get_person(person_id: UUID, db: DbSession) -> Person:
    person = cast(Person | None, db.scalar(person_detail_statement().where(Person.id == person_id)))
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    return person


@router.patch("/{person_id}", response_model=PersonRead)
def update_person(person_id: UUID, payload: PersonUpdate, db: DbSession) -> Person:
    person = get_person(person_id, db)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(person, field, value)
    db.commit()
    db.refresh(person)
    return get_person(person.id, db)


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(person_id: UUID, db: DbSession) -> None:
    person = get_person(person_id, db)
    person.deleted_at = datetime.now(timezone.utc)
    db.commit()
