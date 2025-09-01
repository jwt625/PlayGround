"""
FastAPI backend for one-click paper page service.
Handles GitHub OAuth authentication and API endpoints.
"""

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import requests
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.conversion import (
    ConversionJobResponse,
    ConversionMode,
    ConversionResult,
    ConversionStatusResponse,
)
from models.github import (
    CommitRequest,
    CreateRepositoryRequest,
    CreateRepositoryResponse,
    DeploymentConfig,
    DeploymentStatusResponse,
    FileContent,
    OAuthRevokeRequest,
    OAuthTokenRequest,
    OAuthTokenResponse,
    TemplateInfo,
    TemplateType,
)
from services.conversion_service import ConversionService
from services.github_service import GitHubService

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="One-Click Paper Page API",
    description="Backend API for converting academic papers to websites",
    version="0.1.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class OAuthTokenRequest(BaseModel):
    code: str
    state: str | None = None
    redirect_uri: str

class OAuthTokenResponse(BaseModel):
    access_token: str
    token_type: str
    scope: str

class OAuthRevokeRequest(BaseModel):
    access_token: str

class ErrorResponse(BaseModel):
    message: str
    details: str | None = None

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

if not GITHUB_CLIENT_SECRET:
    print("WARNING: GITHUB_CLIENT_SECRET not set in environment")

@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "One-Click Paper Page API is running"}

@app.post("/api/github/oauth/token", response_model=OAuthTokenResponse)
async def exchange_oauth_token(request: OAuthTokenRequest) -> OAuthTokenResponse:
    """
    Exchange GitHub OAuth authorization code for access token.

    This endpoint handles the server-side token exchange required by GitHub OAuth.
    The client secret is kept secure on the server side.
    """
    if not GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="GitHub client secret not configured"
        )

    # Prepare token exchange request to GitHub
    token_data = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "code": request.code,
        "redirect_uri": request.redirect_uri,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "one-click-paper-page/0.1.0"
    }

    try:
        # Exchange code for token with GitHub
        response = requests.post(
            "https://github.com/login/oauth/access_token",
            data=token_data,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        token_response = response.json()

        # Check for GitHub API errors
        if "error" in token_response:
            error_desc = token_response.get(
                'error_description', token_response['error']
            )
            raise HTTPException(
                status_code=400,
                detail=f"GitHub OAuth error: {error_desc}"
            )

        return OAuthTokenResponse(
            access_token=token_response["access_token"],
            token_type=token_response.get("token_type", "bearer"),
            scope=token_response.get("scope", "")
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to exchange token with GitHub: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from GitHub: missing {str(e)}"
        )

@app.post("/api/github/oauth/revoke")
async def revoke_oauth_token(request: OAuthRevokeRequest) -> dict[str, str]:
    """
    Revoke GitHub OAuth access token.

    This endpoint revokes the access token with GitHub to ensure
    proper cleanup when users log out.
    """
    if not GITHUB_CLIENT_SECRET or not GITHUB_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GitHub client credentials not configured"
        )

    # Prepare revocation request to GitHub
    revoke_data = {
        "access_token": request.access_token,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "one-click-paper-page/0.1.0"
    }

    # Use basic auth with client credentials
    auth = (GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)

    try:
        # Revoke token with GitHub
        response = requests.delete(
            f"https://api.github.com/applications/{GITHUB_CLIENT_ID}/grant",
            json=revoke_data,
            headers=headers,
            auth=auth,
            timeout=10
        )

        # GitHub returns 204 for successful revocation
        if response.status_code == 204:
            return {"message": "Token revoked successfully"}
        elif response.status_code == 404:
            # Token already revoked or doesn't exist
            return {"message": "Token already revoked"}
        else:
            response.raise_for_status()
            return {"message": "Token revoked successfully"}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to revoke token with GitHub: {str(e)}"
        )


# Initialize conversion service
conversion_service = ConversionService()


@app.post("/api/convert/upload", response_model=ConversionJobResponse)
async def upload_and_convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    template: str = Form(...),
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
    """
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

    try:
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

            deployment_config = {
                "repository_name": repository_name or f"paper-{job_id[:8]}",
                "template": template,
                "paper_title": paper_title,
                "paper_authors": authors_list,
                "access_token": authorization.replace("Bearer ", "")
            }

        # Start conversion in background
        background_tasks.add_task(
            _run_conversion_task,
            job_id,
            input_file_path,
            deployment_config or {},
        )

        return ConversionJobResponse(
            job_id=job_id,
            status=job_status["status"],
            message=f"File uploaded successfully. Conversion started with {mode} mode.",
        )

    except Exception as e:
        # Cleanup on error
        conversion_service.cleanup_job(job_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}"
        )


@app.get("/api/convert/status/{job_id}", response_model=ConversionStatusResponse)
async def get_conversion_status(job_id: str) -> ConversionStatusResponse:
    """
    Get the current status of a conversion job.

    Args:
        job_id: Job identifier

    Returns:
        Current job status and progress
    """
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


@app.get("/api/convert/result/{job_id}", response_model=ConversionResult)
async def get_conversion_result(job_id: str) -> ConversionResult:
    """
    Get the result of a completed conversion job.

    Args:
        job_id: Job identifier

    Returns:
        Conversion result with output files and metrics
    """
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

    # Type assertion since we know result is ConversionResult from the service
    return result  # type: ignore[no-any-return]


@app.delete("/api/convert/cancel/{job_id}")
async def cancel_conversion(job_id: str) -> dict[str, str]:
    """
    Cancel a conversion job and clean up files.

    Args:
        job_id: Job identifier

    Returns:
        Success message
    """
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

    return {"message": f"Job {job_id} cancelled and cleaned up successfully"}


# GitHub Deployment Endpoints

@app.get("/api/templates", response_model=list[TemplateInfo])
async def list_templates() -> list[TemplateInfo]:
    """
    List available templates for GitHub Pages deployment.

    Returns:
        List of available templates with their information
    """
    from services.template_service import template_service
    return template_service.get_all_templates()


@app.post("/api/deployment/{deployment_id}/enable-pages")
async def enable_github_pages_backup(
    deployment_id: str,
    request: Request,
) -> dict[str, Any]:
    """
    Enable GitHub Pages as a backup option when automated deployment fails.
    """
    try:
        # Get access token from request headers
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

        access_token = auth_header.split(" ")[1]
        github_service = GitHubService(access_token)

        # Get deployment info
        deployment = await github_service.get_deployment_status(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        # Enable GitHub Pages as backup
        success = await github_service.enable_github_pages_as_backup(deployment.repository)

        if success:
            return {
                "success": True,
                "message": "GitHub Pages enabled successfully as backup option",
                "pages_url": f"https://{deployment.repository.owner.login}.github.io/{deployment.repository.name}"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to enable GitHub Pages backup"
            )

    except Exception as e:
        logger.error(f"Error enabling GitHub Pages backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/deploy")
async def deploy_to_github(
    request: dict[str, Any],
    authorization: str = Header(None),
) -> dict[str, Any]:
    """
    Full automated deployment: create repository and deploy content.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="GitHub OAuth token required. Please authenticate first."
        )

    access_token = authorization.replace("Bearer ", "")

    try:
        # Get conversion result
        conversion_job_id = request.get("conversion_job_id")
        if not conversion_job_id:
            raise HTTPException(status_code=400, detail="conversion_job_id is required")

        conversion_result = conversion_service.get_job_result(conversion_job_id)
        if not conversion_result:
            raise HTTPException(
                status_code=404,
                detail=f"Conversion job {conversion_job_id} not found or not completed"
            )

        # Create GitHub service
        github_service = GitHubService(access_token)

        # Import required models
        from models.github import (
            CreateRepositoryRequest,
            DeploymentConfig,
            TemplateType,
        )

        # Map template string to enum
        template_map = {
            "academic-pages": TemplateType.ACADEMIC_PAGES,
            "al-folio": TemplateType.AL_FOLIO,
            "minimal-academic": TemplateType.MINIMAL_ACADEMIC,
        }
        template_enum = template_map.get(
            request.get("template", "minimal-academic"), TemplateType.MINIMAL_ACADEMIC
        )

        # Create repository with GitHub Actions workflows
        repo_request = CreateRepositoryRequest(
            name=request["repository_name"],
            description=(
                f"Academic paper website: {request.get('paper_title', 'Untitled')}"
            ),
            template=template_enum,
            conversion_job_id=conversion_job_id
        )

        repo_response = await github_service.create_repository_from_template(repo_request)

        # Deploy content automatically
        deploy_config = DeploymentConfig(
            repository_name=request["repository_name"],
            template=template_enum,
            conversion_job_id=conversion_job_id,
            paper_title=request.get("paper_title"),
            paper_authors=request.get("paper_authors", []),
        )

        # Get conversion result for deployment
        conversion_result = conversion_service.get_job_result(conversion_job_id)
        if not conversion_result or not conversion_result.output_dir:
            raise HTTPException(
                status_code=400,
                detail="Conversion not completed yet"
            )

        await github_service.deploy_converted_content(
            repo_response.deployment_id,
            Path(conversion_result.output_dir),
            deploy_config
        )

        return {
            "success": True,
            "deployment_id": repo_response.deployment_id,
            "repository_url": repo_response.repository.html_url,
            "message": "Repository created and deployment started successfully"
        }

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/repository/create", response_model=CreateRepositoryResponse)
async def create_repository(
    request: CreateRepositoryRequest,
    authorization: str = Header(None),
) -> CreateRepositoryResponse:
    """
    Create a new GitHub repository for the converted paper.

    Args:
        request: Repository creation request
        authorization: GitHub OAuth token (Bearer token)

    Returns:
        Repository creation response with deployment tracking
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="GitHub OAuth token required. Please authenticate first."
        )

    access_token = authorization.replace("Bearer ", "")

    try:
        github_service = GitHubService(access_token)
        response = await github_service.create_repository(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create repository: {str(e)}"
        )


@app.post("/api/github/deploy/{deployment_id}")
async def deploy_converted_content(
    deployment_id: str,
    config: DeploymentConfig,
    authorization: str = Header(None),
) -> dict[str, str]:
    """
    Deploy converted content to GitHub repository.

    Args:
        deployment_id: Deployment job ID
        config: Deployment configuration
        authorization: GitHub OAuth token (Bearer token)

    Returns:
        Deployment status message
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="GitHub OAuth token required. Please authenticate first."
        )

    access_token = authorization.replace("Bearer ", "")

    try:
        # Get conversion result
        if not config.conversion_job_id:
            raise HTTPException(
                status_code=400,
                detail="No conversion job ID provided in deployment config"
            )

        conversion_result = conversion_service.get_job_result(config.conversion_job_id)
        if not conversion_result:
            raise HTTPException(
                status_code=404,
                detail=f"Conversion job {config.conversion_job_id} not found"
            )

        if not conversion_result.output_dir:
            raise HTTPException(
                status_code=400,
                detail="Conversion not completed yet"
            )

        # Start deployment
        github_service = GitHubService(access_token)
        await github_service.deploy_converted_content(
            deployment_id,
            Path(conversion_result.output_dir),
            config
        )

        return {"message": "Deployment started successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deploy content: {str(e)}"
        )


@app.get(
    "/api/github/deployment/{deployment_id}/status",
    response_model=DeploymentStatusResponse,
)
async def get_deployment_status(
    deployment_id: str,
    authorization: str = Header(None),
) -> DeploymentStatusResponse:
    """
    Get deployment status.

    Args:
        deployment_id: Deployment job ID
        authorization: GitHub OAuth token (Bearer token)

    Returns:
        Deployment status information
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="GitHub OAuth token required. Please authenticate first."
        )

    access_token = authorization.replace("Bearer ", "")

    try:
        github_service = GitHubService(access_token)
        status = await github_service.get_deployment_status(deployment_id)
        return status
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get deployment status: {str(e)}"
        )


@app.post("/api/github/test-deploy")
async def test_deployment_workflow(
    authorization: str = Header(None),
) -> dict[str, Any]:
    """
    Full deployment test: fork template repo, commit test content, setup CI/CD.

    This endpoint delegates to GitHubService to test the complete deployment pipeline.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="GitHub OAuth token required. Please authenticate first."
        )

    access_token = authorization.replace("Bearer ", "")

    try:
        github_service = GitHubService(access_token)
        result = await github_service.test_deployment_workflow()
        return result

    except Exception as e:
        logger.error(f"Test deployment failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Test deployment failed: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Test deployment failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Test deployment failed: {str(e)}"
        )


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

                # Create GitHub service
                github_service = GitHubService(deployment_config["access_token"])

                # Create repository with GitHub Actions workflows
                from models.github import (
                    CreateRepositoryRequest,
                    DeploymentConfig,
                    TemplateType,
                )

                # Map template string to enum
                template_map = {
                    "academic-pages": TemplateType.ACADEMIC_PAGES,
                    "al-folio": TemplateType.AL_FOLIO,
                    "minimal-academic": TemplateType.MINIMAL_ACADEMIC,
                }
                template_enum = template_map.get(
                    deployment_config["template"], TemplateType.MINIMAL_ACADEMIC
                )

                repo_request = CreateRepositoryRequest(
                    name=deployment_config["repository_name"],
                    description=(
                        f"Academic paper website: "
                        f"{deployment_config.get('paper_title', 'Untitled')}"
                    ),
                    template=template_enum,
                    conversion_job_id=job_id
                )

                # Create repository with GitHub Actions workflows pre-configured
                repo_response = await github_service.create_repository_from_template(
                    repo_request
                )

                # Deploy source content to trigger GitHub Actions
                deploy_config = DeploymentConfig(
                    repository_name=deployment_config["repository_name"],
                    template=template_enum,
                    paper_title=deployment_config.get("paper_title"),
                    paper_authors=deployment_config.get("paper_authors", []),
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


def main() -> None:
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
