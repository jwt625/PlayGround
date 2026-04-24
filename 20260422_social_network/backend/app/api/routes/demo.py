from fastapi import APIRouter, status

from app.api.deps import DbSession
from app.services.demo_seed import seed_demo_data

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post("/seed", status_code=status.HTTP_204_NO_CONTENT)
def seed_demo(db: DbSession) -> None:
    seed_demo_data(db)
