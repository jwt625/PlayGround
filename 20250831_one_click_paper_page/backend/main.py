"""
FastAPI backend for one-click paper page service.
Handles GitHub OAuth authentication and API endpoints.
"""

import os
from pathlib import Path

import aiofiles
import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.conversion import (
    ConversionJobResponse,
    ConversionMode,
    ConversionResult,
    ConversionStatusResponse,
)
from services.conversion_service import ConversionService

# Load environment variables
load_dotenv()

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
) -> ConversionJobResponse:
    """
    Upload a file and start conversion process.

    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded PDF/DOCX file
        template: Template to use for the website
        mode: Conversion mode (auto, fast, quality)
        repository_name: Optional custom repository name

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

        # Start conversion in background
        background_tasks.add_task(
            _run_conversion_task,
            job_id,
            input_file_path,
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


async def _run_conversion_task(job_id: str, input_file_path: Path) -> None:
    """
    Background task to run the conversion process.

    Args:
        job_id: Job identifier
        input_file_path: Path to input file
    """
    try:
        await conversion_service.convert_file(job_id, input_file_path)
    except Exception:
        # Error handling is done within the conversion service
        pass


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
