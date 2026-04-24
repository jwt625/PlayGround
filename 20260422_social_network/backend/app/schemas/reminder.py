from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import TimestampedSchema

ReminderStatus = Literal["Open", "Snoozed", "Done", "Canceled"]
ReminderPriority = Literal["Low", "Normal", "High"]
ReminderEntityType = Literal["Person", "Organization", "Event", "PipelineItem", "General"]


class ReminderBase(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    notes: str | None = None
    due_at: datetime
    status: ReminderStatus = "Open"
    priority: ReminderPriority = "Normal"
    snoozed_until: datetime | None = None
    completed_at: datetime | None = None
    entity_type: ReminderEntityType | None = None
    entity_id: UUID | None = None


class ReminderCreate(ReminderBase):
    pass


class ReminderUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    notes: str | None = None
    due_at: datetime | None = None
    status: ReminderStatus | None = None
    priority: ReminderPriority | None = None
    snoozed_until: datetime | None = None
    completed_at: datetime | None = None
    entity_type: ReminderEntityType | None = None
    entity_id: UUID | None = None


class ReminderRead(ReminderBase, TimestampedSchema):
    id: UUID
    deleted_at: datetime | None = None
