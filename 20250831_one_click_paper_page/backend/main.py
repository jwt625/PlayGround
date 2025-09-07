"""
FastAPI backend for one-click paper page service.
Handles GitHub OAuth authentication and API endpoints.
"""

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
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
    CreateRepositoryRequest,
    CreateRepositoryResponse,
    DeploymentConfig,
    DeploymentStatusResponse,
    GitHubUser,
    TemplateInfo,
    TemplateType,
)
from services.github_service import GitHubService
from routers.auth_router import router as auth_router
from routers.conversion_router import router as conversion_router
from shared_services import conversion_service

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

# Include routers
app.include_router(auth_router)
app.include_router(conversion_router)

# Conversion service is now imported from shared_services

# Additional Pydantic models for request/response

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

# OAuth endpoints moved to routers/auth_router.py


@app.get("/api/github/user", response_model=GitHubUser)
async def get_github_user(
    authorization: str = Header(..., description="Bearer token")
) -> GitHubUser:
    """Get authenticated GitHub user information."""
    try:
        # Extract token from Authorization header
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header format"
            )

        token = authorization.replace("Bearer ", "")
        github_service = GitHubService(token)

        return await github_service.get_authenticated_user()

    except Exception as e:
        logger.error(f"Failed to get GitHub user: {e}")
        raise HTTPException(
            status_code=401,
            detail="Failed to authenticate with GitHub"
        )


@app.get("/api/github/token/scopes")
async def get_token_scopes(
    authorization: str = Header(..., description="Bearer token")
) -> dict[str, list[str]]:
    """Get the scopes for the current GitHub access token."""
    try:
        # Extract token from Authorization header
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header format"
            )

        token = authorization.replace("Bearer ", "")
        github_service = GitHubService(token)

        scopes = await github_service.get_token_scopes()
        return {"scopes": scopes}

    except Exception as e:
        logger.error(f"Failed to get token scopes: {e}")
        raise HTTPException(
            status_code=401,
            detail="Failed to get token scopes"
        )


# Conversion endpoints moved to routers/conversion_router.py
# Original conversion service initialization and endpoints commented out
# All conversion endpoints moved to routers/conversion_router.py




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
            raise HTTPException(
                status_code=401, detail="Missing or invalid authorization header"
            )

        access_token = auth_header.split(" ")[1]
        github_service = GitHubService(access_token)

        # Get deployment info
        deployment = await github_service.get_deployment_status(deployment_id)
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        # Enable GitHub Pages as backup
        success = await github_service.enable_github_pages_as_backup(
            deployment.repository
        )

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

        # Extract paper title and authors from conversion result metadata
        paper_title = "Untitled"
        paper_authors = []

        if conversion_result.metadata:
            if conversion_result.metadata.title:
                paper_title = conversion_result.metadata.title
                logger.info(f"✅ Extracted paper title: {paper_title}")
            if conversion_result.metadata.authors:
                paper_authors = conversion_result.metadata.authors
                logger.info(f"✅ Extracted authors: {paper_authors}")

        # Create GitHub service
        github_service = GitHubService(access_token)

        # Use existing repository service to generate unique, valid name
        repository_name = await github_service.repository_service.generate_unique_repository_name(paper_title)
        logger.info(f"🏗️ Generated repository name: {repository_name}")

        # Import required models
        from models.github import (
            CreateRepositoryRequest,
            DeploymentConfig,
            TemplateType,
        )

        # Always use minimal-academic template
        template_enum = TemplateType.MINIMAL_ACADEMIC

        # Create repository with GitHub Actions workflows
        repo_request = CreateRepositoryRequest(
            name=repository_name,
            description=f"Academic paper website: {paper_title}",
            template=template_enum,
            conversion_job_id=conversion_job_id
        )

        # Use dual deployment if enabled in config
        try:
            if request.get("enable_dual_deployment", True):
                dual_result = await github_service.create_dual_deployment(repo_request)
                repo_response = CreateRepositoryResponse(
                    repository=dual_result.standalone_repo,
                    deployment_id=dual_result.deployment_id,
                    status=dual_result.status,
                    message=dual_result.message
                )
            else:
                repo_response = await github_service.create_repository_from_template(
                    repo_request
                )
        except Exception as e:
            error_msg = str(e)
            if "name already exists" in error_msg.lower():
                # Generate a unique alternative name
                import time
                timestamp = int(time.time())
                alternative_name = f"{repository_name}-{timestamp}"
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Repository name '{repository_name}' already exists. "
                        f"Try using '{alternative_name}' or choose a different name."
                    )
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create repository: {error_msg}"
                )

        # Deploy content automatically
        deploy_config = DeploymentConfig(
            repository_name=repository_name,  # Use generated name
            template=template_enum,
            conversion_job_id=conversion_job_id,
            paper_title=paper_title,
            paper_authors=paper_authors,
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


@app.post("/api/github/test-deploy-optimized")
async def test_optimized_deployment(
    authorization: str = Header(None),
) -> dict[str, Any]:
    """Test the optimized deployment approach - simplified for debugging."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="GitHub OAuth token required. Please authenticate first."
        )

    access_token = authorization.replace("Bearer ", "")

    try:
        github_service = GitHubService(access_token)

        # Simple test - just create the request object first
        import time
        test_repo_name = f"test-optimized-{int(time.time())}"

        # Create request with minimal data
        request = CreateRepositoryRequest(
            name=test_repo_name,
            description="Test optimized deployment",
            template=TemplateType.ACADEMIC_PAGES,
            conversion_job_id="test-optimized"
        )

        # Test the optimized approach
        result = await github_service.create_repository_optimized(request)

        return {
            "success": True,
            "repository": {
                "name": result.repository.name,
                "url": result.repository.html_url,
            },
            "deployment_id": result.deployment_id,
            "message": "Optimized deployment test successful"
        }

    except Exception as e:
        logger.error(f"Optimized deployment test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimized deployment test failed: {str(e)}"
        )


@app.post("/api/github/test-dual-deploy")
async def test_dual_deploy(
    authorization: str = Header(..., alias="Authorization")
) -> dict[str, Any]:
    """Test the dual deployment system."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")

    access_token = authorization.replace("Bearer ", "")

    try:
        github_service = GitHubService(access_token)

        # Create test dual deployment
        import time
        test_repo_name = f"test-dual-{int(time.time())}"

        repo_request = CreateRepositoryRequest(
            name=test_repo_name,
            description="Test dual deployment system",
            template=TemplateType.MINIMAL_ACADEMIC,
            conversion_job_id="test-dual-deployment"
        )

        # Test dual deployment
        result = await github_service.create_dual_deployment(repo_request)

        return {
            "success": True,
            "approach": "dual_deployment",
            "standalone_repo": {
                "name": result.standalone_repo.name,
                "url": result.standalone_repo.html_url,
                "pages_url": result.standalone_url
            },
            "main_repo": {
                "name": result.main_repo.name if result.main_repo else None,
                "url": result.main_repo.html_url if result.main_repo else None,
                "sub_route_url": result.sub_route_url
            },
            "deployment_id": result.deployment_id,
            "status": result.status.value,
            "message": result.message,
            "benefits": [
                "✅ Standalone paper repository created",
                "✅ Main GitHub Pages repo setup/updated",
                "✅ Paper added to main repo papers collection",
                "✅ Sync workflow configured",
                "✅ Both URLs available for sharing"
            ]
        }

    except Exception as e:
        logger.error(f"Dual deployment test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Dual deployment test failed: {str(e)}"
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


# Background conversion task moved to routers/conversion_router.py


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
