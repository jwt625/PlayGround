from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import TimestampedSchema


EventType = Literal[
    "Meeting",
    "One-on-one",
    "Group meeting",
    "Call",
    "Email",
    "Message",
    "Intro",
    "Meal",
    "Event attendance",
    "Work session",
    "Supplier discussion",
    "Personal milestone",
    "Note",
    "Other",
]


class EventBase(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    type: EventType
    started_at: datetime
    ended_at: datetime | None = None
    duration_minutes: int | None = None
    context: str | None = None
    summary: str | None = None
    notes: str | None = None
    sentiment: str | None = None


class EventCreate(EventBase):
    person_ids: list[UUID] = []
    organization_ids: list[UUID] = []
    location_ids: list[UUID] = []


class EventUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    type: EventType | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_minutes: int | None = None
    context: str | None = None
    summary: str | None = None
    notes: str | None = None
    sentiment: str | None = None
    person_ids: list[UUID] | None = None
    organization_ids: list[UUID] | None = None
    location_ids: list[UUID] | None = None


class EventRead(EventBase, TimestampedSchema):
    person_ids: list[UUID]
    organization_ids: list[UUID]
    location_ids: list[UUID]
    deleted_at: datetime | None = None
