"""
FastAPI backend for one-click paper page service.
Handles GitHub OAuth authentication and API endpoints.
"""

import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
