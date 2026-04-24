from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Pipeline, PipelineStage


DEFAULT_PIPELINES = [
    (
        "Relationship nurture",
        "Default personal relationship maintenance pipeline.",
        "Relationship nurture",
        [
            ("Noted", 0, False),
            ("Need follow-up", 1, False),
            ("Conversation started", 2, False),
            ("Building trust", 3, False),
            ("Active relationship", 4, False),
            ("Dormant", 5, True),
        ],
    ),
    (
        "Investor conversation",
        "Secondary template for investor relationship tracking.",
        "Investor conversation",
        [("Introduced", 0, False), ("Meeting", 1, False), ("Diligence", 2, False), ("Closed", 3, True)],
    ),
    (
        "Partner collaboration",
        "Secondary template for partnership progress.",
        "Partner collaboration",
        [("Identified", 0, False), ("Exploring", 1, False), ("Active", 2, False), ("Paused", 3, True)],
    ),
]


def ensure_seed_pipelines(db: Session) -> None:
    existing = db.scalar(select(Pipeline.id).limit(1))
    if existing:
        return

    for name, description, template_type, stages in DEFAULT_PIPELINES:
        pipeline = Pipeline(name=name, description=description, template_type=template_type)
        db.add(pipeline)
        db.flush()
        for stage_name, sort_order, is_terminal in stages:
            db.add(
                PipelineStage(
                    pipeline_id=pipeline.id,
                    name=stage_name,
                    sort_order=sort_order,
                    color=None,
                    is_terminal=is_terminal,
                )
            )
    db.commit()
