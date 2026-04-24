from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select

from app.api.deps import DbSession
from app.models import Pipeline, PipelineItem, PipelineStage
from app.schemas.pipeline import (
    PipelineCreate,
    PipelineDetailRead,
    PipelineItemCreate,
    PipelineItemMove,
    PipelineItemRead,
    PipelineItemUpdate,
    PipelineRead,
    PipelineStageRead,
)
from app.services.pipeline_seed import ensure_seed_pipelines

router = APIRouter(tags=["pipelines"])


def serialize_pipeline(pipeline: Pipeline, db: DbSession) -> PipelineDetailRead:
    stages = list(db.scalars(select(PipelineStage).where(PipelineStage.pipeline_id == pipeline.id).order_by(PipelineStage.sort_order)))
    items = list(db.scalars(select(PipelineItem).where(PipelineItem.pipeline_id == pipeline.id).order_by(PipelineItem.created_at.desc())))
    return PipelineDetailRead.model_validate(
        {
            **pipeline.__dict__,
            "stages": [PipelineStageRead.model_validate(stage) for stage in stages],
            "items": [PipelineItemRead.model_validate(item) for item in items],
        }
    )


@router.get("/pipelines", response_model=list[PipelineRead])
def list_pipelines(db: DbSession, response: Response) -> list[PipelineRead]:
    ensure_seed_pipelines(db)
    pipelines = list(db.scalars(select(Pipeline).order_by(Pipeline.name)))
    response.headers["X-Total-Count"] = str(len(pipelines))
    payload = []
    for pipeline in pipelines:
        stages = list(db.scalars(select(PipelineStage).where(PipelineStage.pipeline_id == pipeline.id).order_by(PipelineStage.sort_order)))
        payload.append(
            PipelineRead.model_validate(
                {
                    **pipeline.__dict__,
                    "stages": [PipelineStageRead.model_validate(stage) for stage in stages],
                }
            )
        )
    return payload


@router.post("/pipelines", response_model=PipelineDetailRead, status_code=status.HTTP_201_CREATED)
def create_pipeline(payload: PipelineCreate, db: DbSession) -> PipelineDetailRead:
    pipeline = Pipeline(
        name=payload.name,
        description=payload.description,
        template_type=payload.template_type,
    )
    db.add(pipeline)
    db.flush()
    for index, stage in enumerate(payload.stages):
        db.add(
            PipelineStage(
                pipeline_id=pipeline.id,
                name=stage.name,
                sort_order=stage.sort_order if stage.sort_order else index,
                color=stage.color,
                is_terminal=stage.is_terminal,
            )
        )
    db.commit()
    db.refresh(pipeline)
    return serialize_pipeline(pipeline, db)


@router.get("/pipelines/{pipeline_id}", response_model=PipelineDetailRead)
def get_pipeline(pipeline_id: UUID, db: DbSession) -> PipelineDetailRead:
    ensure_seed_pipelines(db)
    pipeline = db.get(Pipeline, pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")
    return serialize_pipeline(pipeline, db)


@router.post("/pipelines/{pipeline_id}/items", response_model=PipelineItemRead, status_code=status.HTTP_201_CREATED)
def create_pipeline_item(pipeline_id: UUID, payload: PipelineItemCreate, db: DbSession) -> PipelineItem:
    pipeline = db.get(Pipeline, pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")
    item = PipelineItem(pipeline_id=pipeline_id, **payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.patch("/pipeline-items/{item_id}", response_model=PipelineItemRead)
def update_pipeline_item(item_id: UUID, payload: PipelineItemUpdate, db: DbSession) -> PipelineItem:
    item = db.get(PipelineItem, item_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline item not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, field, value)
    db.commit()
    db.refresh(item)
    return item


@router.post("/pipeline-items/{item_id}/move", response_model=PipelineItemRead)
def move_pipeline_item(item_id: UUID, payload: PipelineItemMove, db: DbSession) -> PipelineItem:
    item = db.get(PipelineItem, item_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline item not found")
    stage = db.get(PipelineStage, payload.stage_id)
    if not stage or stage.pipeline_id != item.pipeline_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Stage does not belong to the item pipeline")
    item.stage_id = payload.stage_id
    db.commit()
    db.refresh(item)
    return item
