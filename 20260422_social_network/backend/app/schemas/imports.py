from pydantic import BaseModel


class PeopleImportResult(BaseModel):
    created: int
    skipped: int
    errors: list[str]
