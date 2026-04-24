from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import TimestampedSchema


class PipelineStageBase(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    sort_order: int = 0
    color: str | None = None
    is_terminal: bool = False


class PipelineStageCreate(PipelineStageBase):
    pass


class PipelineStageRead(PipelineStageBase, TimestampedSchema):
    id: UUID
    pipeline_id: UUID


class PipelineBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    template_type: str = "Relationship nurture"


class PipelineCreate(PipelineBase):
    stages: list[PipelineStageCreate] = []


class PipelineRead(PipelineBase, TimestampedSchema):
    id: UUID
    stages: list[PipelineStageRead] = []


class PipelineItemBase(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    description: str | None = None
    primary_person_id: UUID | None = None
    primary_organization_id: UUID | None = None
    status: str = "Open"
    priority: str = "Normal"
    expected_date: date | None = None
    notes: str | None = None


class PipelineItemCreate(PipelineItemBase):
    stage_id: UUID


class PipelineItemUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = None
    stage_id: UUID | None = None
    primary_person_id: UUID | None = None
    primary_organization_id: UUID | None = None
    status: str | None = None
    priority: str | None = None
    expected_date: date | None = None
    notes: str | None = None


class PipelineItemMove(BaseModel):
    stage_id: UUID


class PipelineItemRead(PipelineItemBase, TimestampedSchema):
    id: UUID
    pipeline_id: UUID
    stage_id: UUID


class PipelineDetailRead(PipelineRead):
    items: list[PipelineItemRead] = []
