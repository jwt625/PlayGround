from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select

from app.api.deps import DbSession
from app.models import Reminder
from app.schemas.reminder import ReminderCreate, ReminderRead, ReminderUpdate
from app.services.reminder_state import sync_person_next_reminder_date

router = APIRouter(prefix="/reminders", tags=["reminders"])


@router.get("", response_model=list[ReminderRead])
def list_reminders(
    db: DbSession,
    response: Response,
    q: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    entity_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Reminder]:
    statement = select(Reminder).where(Reminder.deleted_at.is_(None))
    count_statement = select(func.count()).select_from(Reminder).where(Reminder.deleted_at.is_(None))
    if q:
        pattern = f"%{q}%"
        search_filter = or_(Reminder.title.ilike(pattern), Reminder.notes.ilike(pattern))
        statement = statement.where(search_filter)
        count_statement = count_statement.where(search_filter)
    if status_filter:
        status_clause = Reminder.status == status_filter
        statement = statement.where(status_clause)
        count_statement = count_statement.where(status_clause)
    if entity_type:
        entity_type_clause = Reminder.entity_type == entity_type
        statement = statement.where(entity_type_clause)
        count_statement = count_statement.where(entity_type_clause)
    visibility_clause = or_(Reminder.snoozed_until.is_(None), Reminder.snoozed_until <= datetime.now(timezone.utc))
    statement = statement.where(visibility_clause)
    count_statement = count_statement.where(visibility_clause)
    response.headers["X-Total-Count"] = str(db.scalar(count_statement) or 0)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Offset"] = str(offset)
    statement = statement.order_by(Reminder.due_at, Reminder.created_at).limit(limit).offset(offset)
    return list(db.scalars(statement))


@router.post("", response_model=ReminderRead, status_code=status.HTTP_201_CREATED)
def create_reminder(payload: ReminderCreate, db: DbSession) -> Reminder:
    reminder = Reminder(**payload.model_dump())
    db.add(reminder)
    db.flush()
    if reminder.entity_type == "Person" and reminder.entity_id:
        sync_person_next_reminder_date(db, reminder.entity_id)
    db.commit()
    db.refresh(reminder)
    return reminder


@router.get("/{reminder_id}", response_model=ReminderRead)
def get_reminder(reminder_id: UUID, db: DbSession) -> Reminder:
    reminder = db.get(Reminder, reminder_id)
    if not reminder or reminder.deleted_at:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reminder not found")
    return reminder


@router.patch("/{reminder_id}", response_model=ReminderRead)
def update_reminder(reminder_id: UUID, payload: ReminderUpdate, db: DbSession) -> Reminder:
    reminder = get_reminder(reminder_id, db)
    prior_person_id = reminder.entity_id if reminder.entity_type == "Person" else None
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(reminder, field, value)
    if prior_person_id:
        sync_person_next_reminder_date(db, prior_person_id)
    if reminder.entity_type == "Person" and reminder.entity_id:
        sync_person_next_reminder_date(db, reminder.entity_id)
    db.commit()
    db.refresh(reminder)
    return reminder


@router.delete("/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_reminder(reminder_id: UUID, db: DbSession) -> None:
    reminder = get_reminder(reminder_id, db)
    person_id = reminder.entity_id if reminder.entity_type == "Person" else None
    reminder.deleted_at = datetime.now(timezone.utc)
    if person_id:
        sync_person_next_reminder_date(db, person_id)
    db.commit()


@router.post("/{reminder_id}/snooze", response_model=ReminderRead)
def snooze_reminder(reminder_id: UUID, payload: ReminderUpdate, db: DbSession) -> Reminder:
    reminder = get_reminder(reminder_id, db)
    if not payload.snoozed_until:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="snoozed_until is required")
    reminder.status = "Snoozed"
    reminder.snoozed_until = payload.snoozed_until
    if reminder.entity_type == "Person" and reminder.entity_id:
        sync_person_next_reminder_date(db, reminder.entity_id)
    db.commit()
    db.refresh(reminder)
    return reminder


@router.post("/{reminder_id}/complete", response_model=ReminderRead)
def complete_reminder(reminder_id: UUID, db: DbSession) -> Reminder:
    reminder = get_reminder(reminder_id, db)
    reminder.status = "Done"
    reminder.completed_at = datetime.now(timezone.utc)
    if reminder.entity_type == "Person" and reminder.entity_id:
        sync_person_next_reminder_date(db, reminder.entity_id)
    db.commit()
    db.refresh(reminder)
    return reminder
