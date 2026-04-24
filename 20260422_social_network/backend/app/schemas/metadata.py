from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel, TimestampedSchema


class TagBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    color: str | None = None


class TagCreate(TagBase):
    pass


class TagRead(TagBase, TimestampedSchema):
    id: UUID


class EntityTagRead(ORMModel):
    id: UUID
    tag_id: UUID
    tag: TagRead


class LocationBase(BaseModel):
    label: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    address_line: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    location_type: str = "Other"
    notes: str | None = None


class LocationCreate(LocationBase):
    pass


class LocationRead(LocationBase, TimestampedSchema):
    id: UUID


class EntityLocationCreate(BaseModel):
    location: LocationCreate
    is_primary: bool = False
    notes: str | None = None


class EntityLocationRead(ORMModel):
    id: UUID
    location_id: UUID
    is_primary: bool
    notes: str | None = None
    location: LocationRead


class PersonOrganizationBase(BaseModel):
    organization_id: UUID
    title: str | None = None
    role_type: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool = True
    notes: str | None = None


class PersonOrganizationCreate(PersonOrganizationBase):
    pass


class PersonOrganizationRead(ORMModel):
    id: UUID
    organization_id: UUID
    organization_name: str | None = None
    title: str | None = None
    role_type: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool
    notes: str | None = None
