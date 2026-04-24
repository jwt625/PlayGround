from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str | list[object]
