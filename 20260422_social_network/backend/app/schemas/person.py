from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import TimestampedSchema
from app.schemas.event import EventRead
from app.schemas.metadata import EntityLocationRead, EntityTagRead, PersonOrganizationRead
from app.schemas.pipeline import PipelineItemRead
from app.schemas.reminder import ReminderRead


class ContactMethodBase(BaseModel):
    type: str
    value: str
    label: str | None = None
    is_primary: bool = False
    notes: str | None = None


class ContactMethodCreate(ContactMethodBase):
    pass


class ContactMethodRead(ContactMethodBase, TimestampedSchema):
    id: UUID


class ExternalProfileBase(BaseModel):
    platform: str
    url_or_handle: str
    label: str | None = None
    notes: str | None = None
    last_checked_at: datetime | None = None


class ExternalProfileCreate(ExternalProfileBase):
    pass


class ExternalProfileRead(ExternalProfileBase, TimestampedSchema):
    id: UUID


class PersonBase(BaseModel):
    display_name: str = Field(min_length=1, max_length=240)
    given_name: str | None = None
    family_name: str | None = None
    nickname: str | None = None
    pronouns: str | None = None
    short_bio: str | None = None
    relationship_summary: str | None = None
    how_we_met: str | None = None
    first_met_date: date | None = None
    primary_location: str | None = None
    notes: str | None = None


class PersonCreate(PersonBase):
    contact_methods: list[ContactMethodCreate] = []
    external_profiles: list[ExternalProfileCreate] = []


class PersonUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=240)
    given_name: str | None = None
    family_name: str | None = None
    nickname: str | None = None
    pronouns: str | None = None
    short_bio: str | None = None
    relationship_summary: str | None = None
    how_we_met: str | None = None
    first_met_date: date | None = None
    primary_location: str | None = None
    notes: str | None = None


class PersonRead(PersonBase, TimestampedSchema):
    id: UUID
    relationship_score: int
    relationship_category: str
    last_interaction_date: datetime | None = None
    next_reminder_date: datetime | None = None
    deleted_at: datetime | None = None
    contact_methods: list[ContactMethodRead] = []
    external_profiles: list[ExternalProfileRead] = []
    organization_roles: list[PersonOrganizationRead] = []


class PersonDetailRead(PersonRead):
    locations: list[EntityLocationRead] = []
    tags: list[EntityTagRead] = []
    recent_events: list[EventRead] = []
    active_reminders: list[ReminderRead] = []
    pipeline_items: list[PipelineItemRead] = []
    relationship_score_reason: str | None = None
