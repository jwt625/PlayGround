from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Person, Reminder


def sync_person_next_reminder_date(db: Session, person_id: UUID) -> None:
    person = db.get(Person, person_id)
    if not person:
        return

    reminders = list(
        db.scalars(
            select(Reminder)
            .where(
                Reminder.deleted_at.is_(None),
                Reminder.entity_type == "Person",
                Reminder.entity_id == person_id,
                Reminder.status.in_(("Open", "Snoozed")),
            )
            .order_by(Reminder.due_at)
        )
    )
    if not reminders:
        person.next_reminder_date = None
        return

    person.next_reminder_date = min(reminder.snoozed_until or reminder.due_at for reminder in reminders)
