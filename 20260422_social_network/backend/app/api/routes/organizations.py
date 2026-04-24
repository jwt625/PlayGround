from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select

from app.api.deps import DbSession
from app.models import ContactMethod, EntityLocation, EntityTag, ExternalProfile, Location, Organization, PersonOrganization, PipelineItem, Tag
from app.schemas.metadata import EntityLocationCreate, EntityLocationRead, EntityTagRead, TagCreate
from app.schemas.organization import OrganizationCreate, OrganizationDetailRead, OrganizationRead, OrganizationUpdate
from app.schemas.person import ContactMethodCreate, ExternalProfileCreate
from app.schemas.pipeline import PipelineItemRead

router = APIRouter(prefix="/organizations", tags=["organizations"])


@router.get("", response_model=list[OrganizationRead])
def list_organizations(
    db: DbSession,
    response: Response,
    q: str | None = Query(default=None),
    industry: str | None = None,
    tag: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Organization]:
    statement = select(Organization).where(Organization.deleted_at.is_(None))
    count_statement = select(func.count()).select_from(Organization).where(Organization.deleted_at.is_(None))
    if q:
        pattern = f"%{q}%"
        search_filter = or_(
            Organization.name.ilike(pattern),
            Organization.industry.ilike(pattern),
            Organization.location.ilike(pattern),
            Organization.notes.ilike(pattern),
        )
        statement = statement.where(search_filter)
        count_statement = count_statement.where(search_filter)
    if industry:
        industry_filter = Organization.industry.ilike(f"%{industry}%")
        statement = statement.where(industry_filter)
        count_statement = count_statement.where(industry_filter)
    if tag:
        tag_filter = Organization.id.in_(
            select(EntityTag.entity_id)
            .join(Tag, Tag.id == EntityTag.tag_id)
            .where(EntityTag.entity_type == "Organization", Tag.name == tag)
        )
        statement = statement.where(tag_filter)
        count_statement = count_statement.where(tag_filter)
    response.headers["X-Total-Count"] = str(db.scalar(count_statement) or 0)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Offset"] = str(offset)
    statement = statement.order_by(Organization.name).limit(limit).offset(offset)
    return list(db.scalars(statement))


@router.post("", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED)
def create_organization(payload: OrganizationCreate, db: DbSession) -> Organization:
    existing = db.scalar(select(Organization).where(Organization.deleted_at.is_(None), Organization.name == payload.name))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Organization with this name already exists")
    organization = Organization(**payload.model_dump())
    db.add(organization)
    db.commit()
    db.refresh(organization)
    return organization


def build_organization_detail(organization: Organization, db: DbSession) -> OrganizationDetailRead:
    entity_locations = list(
        db.execute(
            select(EntityLocation, Location)
            .join(Location, Location.id == EntityLocation.location_id)
            .where(EntityLocation.entity_type == "Organization", EntityLocation.entity_id == organization.id)
        )
    )
    entity_tags = list(
        db.execute(
            select(EntityTag, Tag)
            .join(Tag, Tag.id == EntityTag.tag_id)
            .where(EntityTag.entity_type == "Organization", EntityTag.entity_id == organization.id)
        )
    )
    people = list(
        db.scalars(
            select(PersonOrganization)
            .where(PersonOrganization.organization_id == organization.id)
            .order_by(PersonOrganization.created_at.desc())
        )
    )
    pipeline_items = list(
        db.scalars(select(PipelineItem).where(PipelineItem.primary_organization_id == organization.id).order_by(PipelineItem.created_at.desc()))
    )
    return OrganizationDetailRead.model_validate(
        {
            **OrganizationRead.model_validate(organization).model_dump(),
            "locations": [
                EntityLocationRead.model_validate(
                    {
                        "id": entity_location.id,
                        "location_id": location.id,
                        "is_primary": entity_location.is_primary,
                        "notes": entity_location.notes,
                        "location": location,
                    }
                )
                for entity_location, location in entity_locations
            ],
            "tags": [
                EntityTagRead.model_validate(
                    {
                        "id": entity_tag.id,
                        "tag_id": tag.id,
                        "tag": tag,
                    }
                )
                for entity_tag, tag in entity_tags
            ],
            "people": [
                {
                    "id": role.id,
                    "organization_id": role.organization_id,
                    "organization_name": organization.name,
                    "title": role.title,
                    "role_type": role.role_type,
                    "start_date": role.start_date,
                    "end_date": role.end_date,
                    "is_current": role.is_current,
                    "notes": role.notes,
                }
                for role in people
            ],
            "pipeline_items": [PipelineItemRead.model_validate(item) for item in pipeline_items],
        }
    )


@router.get("/{organization_id}", response_model=OrganizationDetailRead)
def get_organization(organization_id: UUID, db: DbSession) -> OrganizationDetailRead:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return build_organization_detail(organization, db)


@router.patch("/{organization_id}", response_model=OrganizationDetailRead)
def update_organization(organization_id: UUID, payload: OrganizationUpdate, db: DbSession) -> OrganizationDetailRead:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(organization, field, value)
    db.commit()
    db.refresh(organization)
    return get_organization(organization.id, db)


@router.delete("/{organization_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_organization(organization_id: UUID, db: DbSession) -> None:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    organization.deleted_at = datetime.now(timezone.utc)
    db.commit()


@router.post("/{organization_id}/tags", response_model=OrganizationDetailRead, status_code=status.HTTP_201_CREATED)
def add_organization_tag(organization_id: UUID, payload: TagCreate, db: DbSession) -> OrganizationDetailRead:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    tag = db.scalar(select(Tag).where(Tag.name == payload.name))
    if not tag:
        tag = Tag(**payload.model_dump())
        db.add(tag)
        db.flush()
    existing = db.scalar(
        select(EntityTag).where(
            EntityTag.entity_type == "Organization",
            EntityTag.entity_id == organization_id,
            EntityTag.tag_id == tag.id,
        )
    )
    if not existing:
        db.add(EntityTag(entity_type="Organization", entity_id=organization_id, tag_id=tag.id))
    db.commit()
    return get_organization(organization_id, db)


@router.post("/{organization_id}/locations", response_model=OrganizationDetailRead, status_code=status.HTTP_201_CREATED)
def add_organization_location(organization_id: UUID, payload: EntityLocationCreate, db: DbSession) -> OrganizationDetailRead:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    location = Location(**payload.location.model_dump())
    db.add(location)
    db.flush()
    db.add(
        EntityLocation(
            entity_type="Organization",
            entity_id=organization_id,
            location_id=location.id,
            is_primary=payload.is_primary,
            notes=payload.notes,
        )
    )
    if payload.is_primary:
        parts = [location.city, location.region, location.country]
        organization.location = ", ".join(part for part in parts if part) or location.label
    db.commit()
    return get_organization(organization_id, db)


@router.post("/{organization_id}/contact-methods", response_model=OrganizationDetailRead, status_code=status.HTTP_201_CREATED)
def add_organization_contact_method(
    organization_id: UUID, payload: ContactMethodCreate, db: DbSession
) -> OrganizationDetailRead:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    db.add(ContactMethod(organization_id=organization_id, **payload.model_dump()))
    db.commit()
    return get_organization(organization_id, db)


@router.post("/{organization_id}/external-profiles", response_model=OrganizationDetailRead, status_code=status.HTTP_201_CREATED)
def add_organization_external_profile(
    organization_id: UUID, payload: ExternalProfileCreate, db: DbSession
) -> OrganizationDetailRead:
    organization = db.get(Organization, organization_id)
    if not organization or organization.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    db.add(ExternalProfile(organization_id=organization_id, **payload.model_dump()))
    db.commit()
    return get_organization(organization_id, db)
