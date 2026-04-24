from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.metadata import EntityLocationRead, EntityTagRead, PersonOrganizationRead
from app.schemas.pipeline import PipelineItemRead
from app.schemas.common import TimestampedSchema


OrganizationType = Literal["Company", "Fund", "Supplier", "School", "Community", "Partner", "Nonprofit", "Government", "Other"]


class OrganizationBase(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    type: OrganizationType = "Other"
    website: str | None = None
    description: str | None = None
    industry: str | None = None
    location: str | None = None
    notes: str | None = None


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)
    type: OrganizationType | None = None
    website: str | None = None
    description: str | None = None
    industry: str | None = None
    location: str | None = None
    notes: str | None = None


class OrganizationRead(OrganizationBase, TimestampedSchema):
    id: UUID
    deleted_at: datetime | None = None


class OrganizationDetailRead(OrganizationRead):
    locations: list[EntityLocationRead] = []
    tags: list[EntityTagRead] = []
    people: list[PersonOrganizationRead] = []
    pipeline_items: list[PipelineItemRead] = []
