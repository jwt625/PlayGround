from datetime import date, datetime
from uuid import UUID

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Table, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


event_people = Table(
    "event_people",
    Base.metadata,
    Column("event_id", ForeignKey("interaction_events.id", ondelete="CASCADE"), primary_key=True),
    Column("person_id", ForeignKey("people.id", ondelete="CASCADE"), primary_key=True),
)

event_organizations = Table(
    "event_organizations",
    Base.metadata,
    Column("event_id", ForeignKey("interaction_events.id", ondelete="CASCADE"), primary_key=True),
    Column("organization_id", ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True),
)

event_locations = Table(
    "event_locations",
    Base.metadata,
    Column("event_id", ForeignKey("interaction_events.id", ondelete="CASCADE"), primary_key=True),
    Column("location_id", ForeignKey("locations.id", ondelete="CASCADE"), primary_key=True),
)


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    display_name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class Person(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "people"

    display_name: Mapped[str] = mapped_column(String(240), index=True)
    given_name: Mapped[str | None] = mapped_column(String(120))
    family_name: Mapped[str | None] = mapped_column(String(120))
    nickname: Mapped[str | None] = mapped_column(String(120))
    pronouns: Mapped[str | None] = mapped_column(String(80))
    short_bio: Mapped[str | None] = mapped_column(Text)
    relationship_summary: Mapped[str | None] = mapped_column(Text)
    how_we_met: Mapped[str | None] = mapped_column(Text)
    first_met_date: Mapped[date | None] = mapped_column(Date)
    primary_location: Mapped[str | None] = mapped_column(String(240), index=True)
    relationship_score: Mapped[int] = mapped_column(Integer, default=0)
    relationship_category: Mapped[str] = mapped_column(String(40), default="New", index=True)
    last_interaction_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_reminder_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    contact_methods: Mapped[list["ContactMethod"]] = relationship(back_populates="person", cascade="all, delete-orphan")
    external_profiles: Mapped[list["ExternalProfile"]] = relationship(back_populates="person", cascade="all, delete-orphan")
    organization_roles: Mapped[list["PersonOrganization"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(240), index=True)
    type: Mapped[str] = mapped_column(String(80), default="Other", index=True)
    website: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    industry: Mapped[str | None] = mapped_column(String(160), index=True)
    location: Mapped[str | None] = mapped_column(String(240), index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    created_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    people: Mapped[list["PersonOrganization"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    contact_methods: Mapped[list["ContactMethod"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    external_profiles: Mapped[list["ExternalProfile"]] = relationship(back_populates="organization", cascade="all, delete-orphan")


class PersonOrganization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "person_organizations"

    person_id: Mapped[UUID] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(String(200))
    role_type: Mapped[str | None] = mapped_column(String(120), index=True)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text)

    person: Mapped[Person] = relationship(back_populates="organization_roles")
    organization: Mapped[Organization] = relationship(back_populates="people")


class ContactMethod(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "contact_methods"

    person_id: Mapped[UUID | None] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(80), index=True)
    value: Mapped[str] = mapped_column(String(500), index=True)
    label: Mapped[str | None] = mapped_column(String(120))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    person: Mapped[Person | None] = relationship(back_populates="contact_methods")
    organization: Mapped[Organization | None] = relationship(back_populates="contact_methods")


class ExternalProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "external_profiles"

    person_id: Mapped[UUID | None] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    platform: Mapped[str] = mapped_column(String(120), index=True)
    url_or_handle: Mapped[str] = mapped_column(String(500))
    label: Mapped[str | None] = mapped_column(String(120))
    notes: Mapped[str | None] = mapped_column(Text)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    person: Mapped[Person | None] = relationship(back_populates="external_profiles")
    organization: Mapped[Organization | None] = relationship(back_populates="external_profiles")


class Location(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "locations"

    label: Mapped[str | None] = mapped_column(String(160))
    city: Mapped[str | None] = mapped_column(String(120), index=True)
    region: Mapped[str | None] = mapped_column(String(120), index=True)
    country: Mapped[str | None] = mapped_column(String(120), index=True)
    address_line: Mapped[str | None] = mapped_column(String(500))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    location_type: Mapped[str] = mapped_column(String(80), default="Other", index=True)
    notes: Mapped[str | None] = mapped_column(Text)


class EntityLocation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "entity_locations"

    entity_type: Mapped[str] = mapped_column(String(80), index=True)
    entity_id: Mapped[UUID] = mapped_column(index=True)
    location_id: Mapped[UUID] = mapped_column(ForeignKey("locations.id", ondelete="CASCADE"), index=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)


class Tag(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    color: Mapped[str | None] = mapped_column(String(40))


class EntityTag(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "entity_tags"

    entity_type: Mapped[str] = mapped_column(String(80), index=True)
    entity_id: Mapped[UUID] = mapped_column(index=True)
    tag_id: Mapped[UUID] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), index=True)


class InteractionEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "interaction_events"

    title: Mapped[str] = mapped_column(String(240), index=True)
    type: Mapped[str] = mapped_column(String(80), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_minutes: Mapped[int | None] = mapped_column(Integer)
    context: Mapped[str | None] = mapped_column(String(240), index=True)
    summary: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(String(80))
    created_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    people: Mapped[list[Person]] = relationship(secondary=event_people)
    organizations: Mapped[list[Organization]] = relationship(secondary=event_organizations)
    locations: Mapped[list[Location]] = relationship(secondary=event_locations)


class SourceLink(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "source_links"

    entity_type: Mapped[str] = mapped_column(String(80), index=True)
    entity_id: Mapped[UUID] = mapped_column(index=True)
    source_type: Mapped[str] = mapped_column(String(120), index=True)
    url_or_reference: Mapped[str] = mapped_column(String(800))
    label: Mapped[str | None] = mapped_column(String(160))
    notes: Mapped[str | None] = mapped_column(Text)


class Reminder(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "reminders"

    title: Mapped[str] = mapped_column(String(240), index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(40), default="Open", index=True)
    priority: Mapped[str] = mapped_column(String(40), default="Normal", index=True)
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    entity_type: Mapped[str | None] = mapped_column(String(80), index=True)
    entity_id: Mapped[UUID | None] = mapped_column(index=True)
    created_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class Pipeline(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "pipelines"

    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    template_type: Mapped[str] = mapped_column(String(120), default="Relationship nurture", index=True)
    created_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))


class PipelineStage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "pipeline_stages"

    pipeline_id: Mapped[UUID] = mapped_column(ForeignKey("pipelines.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    color: Mapped[str | None] = mapped_column(String(40))
    is_terminal: Mapped[bool] = mapped_column(Boolean, default=False)


class PipelineItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "pipeline_items"

    pipeline_id: Mapped[UUID] = mapped_column(ForeignKey("pipelines.id", ondelete="CASCADE"), index=True)
    stage_id: Mapped[UUID] = mapped_column(ForeignKey("pipeline_stages.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(240), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    primary_person_id: Mapped[UUID | None] = mapped_column(ForeignKey("people.id"), index=True)
    primary_organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("organizations.id"), index=True)
    status: Mapped[str] = mapped_column(String(40), default="Open", index=True)
    priority: Mapped[str] = mapped_column(String(40), default="Normal", index=True)
    expected_date: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)
