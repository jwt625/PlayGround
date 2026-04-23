from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import TimestampedSchema


class OrganizationBase(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    type: str = "Other"
    website: str | None = None
    description: str | None = None
    industry: str | None = None
    location: str | None = None
    notes: str | None = None


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)
    type: str | None = None
    website: str | None = None
    description: str | None = None
    industry: str | None = None
    location: str | None = None
    notes: str | None = None


class OrganizationRead(OrganizationBase, TimestampedSchema):
    id: UUID
    deleted_at: datetime | None = None
