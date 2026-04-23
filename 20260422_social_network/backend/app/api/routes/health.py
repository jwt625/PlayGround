from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbSession

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: DbSession) -> dict[str, str]:
    db.execute(text("select 1"))
    return {"status": "ok", "database": "ok"}
