from uuid import UUID

from pydantic import BaseModel


class SearchResult(BaseModel):
    entity_type: str
    id: UUID
    title: str
    subtitle: str | None = None


class SearchResponse(BaseModel):
    people: list[SearchResult]
    organizations: list[SearchResult]
    events: list[SearchResult]
    reminders: list[SearchResult]
