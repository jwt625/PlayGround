from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import or_, select

from app.api.deps import DbSession
from app.models import Organization
from app.schemas.organization import OrganizationCreate, OrganizationRead, OrganizationUpdate

router = APIRouter(prefix="/organizations", tags=["organizations"])


@router.get("", response_model=list[OrganizationRead])
def list_organizations(
    db: DbSession,
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Organization]:
    statement = select(Organization).where(Organization.deleted_at.is_(None))
    if q:
        pattern = f"%{q}%"
        statement = statement.where(
            or_(
                Organization.name.ilike(pattern),
                Organization.industry.ilike(pattern),
                Organization.location.ilike(pattern),
                Organization.notes.ilike(pattern),
            )
        )
    statement = statement.order_by(Organization.name).limit(limit).offset(offset)
    return list(db.scalars(statement))


@router.post("", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED)
def create_organization(payload: OrganizationCreate, db: DbSession) -> Organization:
    organization = Organization(**payload.model_dump())
    db.add(organization)
    db.commit()
    db.refresh(organization)
    return organization


@router.get("/{organization_id}", response_model=OrganizationRead)
def get_organization(organization_id: UUID, db: DbSession) -> Organization:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return organization


@router.patch("/{organization_id}", response_model=OrganizationRead)
def update_organization(organization_id: UUID, payload: OrganizationUpdate, db: DbSession) -> Organization:
    organization = get_organization(organization_id, db)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(organization, field, value)
    db.commit()
    db.refresh(organization)
    return organization


@router.delete("/{organization_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_organization(organization_id: UUID, db: DbSession) -> None:
    organization = get_organization(organization_id, db)
    organization.deleted_at = datetime.now(timezone.utc)
    db.commit()
