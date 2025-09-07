"""
Conversion router for document conversion endpoints.

This module handles PDF/DOCX file upload, conversion processing,
status tracking, and result retrieval.
"""

import logging
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)

from models.conversion import (
    ConversionJobResponse,
    ConversionMode,
    ConversionResult,
    ConversionStatusResponse,
)
from models.github import (
    CreateRepositoryRequest,
    DeploymentConfig,
    TemplateType,
)
from routers import RouterBase
from services.github_service import GitHubService
from shared_services import conversion_service

# Initialize router
router = APIRouter(prefix="/api/convert", tags=["conversion"])

# Router base for logging and error handling
conversion_router = RouterBase("conversion")

# Setup logging
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=ConversionJobResponse)
async def upload_and_convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    template: str = Form("minimal-academic"),  # Always use minimal-academic
    mode: ConversionMode = Form(ConversionMode.AUTO),
    repository_name: str = Form(None),
    auto_deploy: bool = Form(False),
    paper_title: str = Form(None),
    paper_authors: str = Form(None),
    authorization: str = Header(None),
) -> ConversionJobResponse:
    """
    Upload a file and start conversion process.

    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded PDF/DOCX file
        template: Template to use for the website
        mode: Conversion mode (auto, fast, quality)
        repository_name: Optional custom repository name
        auto_deploy: Whether to automatically deploy to GitHub Pages
        paper_title: Title of the paper
        paper_authors: Comma-separated list of authors
        authorization: GitHub OAuth token (required if auto_deploy=True)

    Returns:
        Job response with job ID and status
        
    Raises:
        HTTPException: If file validation fails or processing error occurs
    """
    conversion_router.log_operation("upload_and_convert", f"file={file.filename}, mode={mode}")

    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".pdf", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {file_ext}. "
                    "Only PDF and DOCX are supported."
                )
            )

        # Create conversion job
        job_id = conversion_service.create_job(mode)

        # Save uploaded file to temporary location
        job_status = conversion_service.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=500, detail="Failed to create conversion job")

        job_dir = job_status["job_dir"]
        input_file_path = job_dir / file.filename

        # Save uploaded file
        async with aiofiles.open(input_file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Prepare deployment config if auto_deploy is enabled
        deployment_config = None
        if auto_deploy:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(
                    status_code=400,
                    detail="GitHub OAuth token required for auto-deployment"
                )

            # Parse authors
            authors_list = []
            if paper_authors:
                authors_list = [author.strip() for author in paper_authors.split(",")]

            # Generate repository name (will be refined after conversion with extracted title)
            if not repository_name:
                import time
                timestamp = int(time.time())
                repository_name = f"paper-{timestamp}-{job_id[:8]}"

            deployment_config = {
                "repository_name": repository_name,
                "template": "minimal-academic",  # Always use minimal-academic
                "paper_title": paper_title,  # Will be updated from conversion result
                "paper_authors": authors_list,
                "access_token": authorization.replace("Bearer ", ""),
                "job_id": job_id  # Add job_id for later metadata extraction
            }

        # Start conversion in background
        background_tasks.add_task(
            _run_conversion_task,
            job_id,
            input_file_path,
            deployment_config,
        )

        conversion_router.log_operation("upload_and_convert", f"job_id={job_id}, success")

        return ConversionJobResponse(
            job_id=job_id,
            status=job_status["status"],
            message=f"File uploaded successfully. Conversion started with {mode} mode.",
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Cleanup on error
        if 'job_id' in locals():
            conversion_service.cleanup_job(job_id)
        raise conversion_router.handle_error("upload_and_convert", e)


@router.get("/status/{job_id}", response_model=ConversionStatusResponse)
async def get_conversion_status(job_id: str) -> ConversionStatusResponse:
    """
    Get the current status of a conversion job.

    Args:
        job_id: Job identifier

    Returns:
        Current job status and progress
        
    Raises:
        HTTPException: If job not found
    """
    conversion_router.log_operation("get_conversion_status", f"job_id={job_id}")

    try:
        job_status = conversion_service.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return ConversionStatusResponse(
            job_id=job_id,
            status=job_status["status"],
            phase=job_status["phase"],
            stage=job_status["stage"],
            message=job_status["message"],
            error=job_status.get("error"),
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise conversion_router.handle_error("get_conversion_status", e)


@router.get("/result/{job_id}", response_model=ConversionResult)
async def get_conversion_result(job_id: str) -> ConversionResult:
    """
    Get the result of a completed conversion job.

    Args:
        job_id: Job identifier

    Returns:
        Conversion result with output files and metrics
        
    Raises:
        HTTPException: If job not found or not completed
    """
    conversion_router.log_operation("get_conversion_result", f"job_id={job_id}")

    try:
        job_status = conversion_service.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Job {job_id} is not completed. "
                    f"Current status: {job_status['status']}"
                )
            )

        result = job_status.get("result")
        if not result:
            raise HTTPException(
                status_code=500,
                detail=f"Job {job_id} completed but no result available"
            )

        conversion_router.log_operation("get_conversion_result", f"job_id={job_id}, success")

        # Type assertion since we know result is ConversionResult from the service
        return result  # type: ignore[no-any-return]
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise conversion_router.handle_error("get_conversion_result", e)


@router.delete("/cancel/{job_id}")
async def cancel_conversion(job_id: str) -> dict[str, str]:
    """
    Cancel a conversion job and clean up files.

    Args:
        job_id: Job identifier

    Returns:
        Success message
        
    Raises:
        HTTPException: If job not found or cleanup fails
    """
    conversion_router.log_operation("cancel_conversion", f"job_id={job_id}")

    try:
        job_status = conversion_service.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Clean up job files
        success = conversion_service.cleanup_job(job_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cleanup job {job_id}"
            )

        conversion_router.log_operation("cancel_conversion", f"job_id={job_id}, success")

        return {"message": f"Job {job_id} cancelled and cleaned up successfully"}
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise conversion_router.handle_error("cancel_conversion", e)


async def _run_conversion_task(
    job_id: str,
    input_file_path: Path,
    deployment_config: dict[str, Any] | None = None
) -> None:
    """
    Background task to run the conversion process and optionally deploy.

    Args:
        job_id: Job identifier
        input_file_path: Path to input file
        deployment_config: Optional deployment configuration
    """
    try:
        # Run conversion
        result = await conversion_service.convert_file(job_id, input_file_path)

        # If deployment config is provided, deploy to GitHub via automated workflow
        if deployment_config and result.success:
            try:
                logger.info(f"Starting automated GitHub deployment for job {job_id}")

                # Extract paper title and authors from conversion result metadata
                paper_title = "Untitled"
                paper_authors = []

                if result.metadata:
                    if result.metadata.title:
                        paper_title = result.metadata.title
                        logger.info(f"✅ Extracted paper title: {paper_title}")
                    if result.metadata.authors:
                        paper_authors = result.metadata.authors
                        logger.info(f"✅ Extracted authors: {paper_authors}")

                # Create GitHub service
                github_service = GitHubService(deployment_config["access_token"])

                # Use existing repository service to generate unique, valid name
                repo_name = await github_service.repository_service.generate_unique_repository_name(paper_title)
                logger.info(f"🏗️ Generated repository name: {repo_name}")

                # Always use minimal-academic template
                template_enum = TemplateType.MINIMAL_ACADEMIC

                repo_request = CreateRepositoryRequest(
                    name=repo_name,
                    description=f"Academic paper website: {paper_title}",
                    template=template_enum,
                    conversion_job_id=job_id
                )

                # Create repository with GitHub Actions workflows pre-configured
                repo_response = await github_service.create_repository_from_template(
                    repo_request
                )

                # Deploy source content to trigger GitHub Actions
                deploy_config = DeploymentConfig(
                    repository_name=repo_name,
                    template=template_enum,
                    paper_title=paper_title,
                    paper_authors=paper_authors,
                )

                # Upload source files to trigger automated conversion and deployment
                await github_service.deploy_converted_content(
                    repo_response.deployment_id,
                    Path(result.output_dir),
                    deploy_config
                )

                logger.info(f"Auto-deployment completed for job {job_id}")

            except Exception as e:
                logger.error(f"Auto-deployment failed for job {job_id}: {e}")
                # Don't fail the conversion if deployment fails

    except Exception as e:
        logger.error(f"Conversion task failed for job {job_id}: {e}")
        # Error handling is done within the conversion service
