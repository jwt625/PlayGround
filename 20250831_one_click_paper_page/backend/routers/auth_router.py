"""
Authentication router for GitHub OAuth endpoints.

This module handles GitHub OAuth authentication flow including
token exchange and revocation.
"""

import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException

from models.github import OAuthRevokeRequest, OAuthTokenRequest, OAuthTokenResponse
from routers import RouterBase, handle_router_error

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Initialize router
router = APIRouter(prefix="/api/github/oauth", tags=["authentication"])

# Router base for logging and error handling
auth_router = RouterBase("auth")


@router.post("/token", response_model=OAuthTokenResponse)
async def exchange_oauth_token(request: OAuthTokenRequest) -> OAuthTokenResponse:
    """
    Exchange GitHub OAuth authorization code for access token.

    This endpoint handles the server-side token exchange required by GitHub OAuth.
    The client secret is kept secure on the server side.
    
    Args:
        request: OAuth token request with authorization code
        
    Returns:
        OAuth token response with access token
        
    Raises:
        HTTPException: If token exchange fails or configuration is invalid
    """
    auth_router.log_operation("exchange_oauth_token", f"code={request.code[:8]}...")
    
    try:
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

        auth_router.log_operation("exchange_oauth_token", "success")
        
        return OAuthTokenResponse(
            access_token=token_response["access_token"],
            token_type=token_response.get("token_type", "bearer"),
            scope=token_response.get("scope", "")
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except requests.RequestException as e:
        raise auth_router.handle_error(
            "exchange_oauth_token",
            HTTPException(
                status_code=500,
                detail=f"Failed to exchange token with GitHub: {str(e)}"
            )
        )
    except KeyError as e:
        raise auth_router.handle_error(
            "exchange_oauth_token",
            HTTPException(
                status_code=500,
                detail=f"Invalid response from GitHub: missing {str(e)}"
            )
        )
    except Exception as e:
        raise auth_router.handle_error("exchange_oauth_token", e)


@router.post("/revoke")
async def revoke_oauth_token(request: OAuthRevokeRequest) -> dict[str, str]:
    """
    Revoke GitHub OAuth access token.

    This endpoint revokes the access token with GitHub to ensure
    proper cleanup when users log out.
    
    Args:
        request: OAuth revoke request with access token
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If token revocation fails or configuration is invalid
    """
    auth_router.log_operation("revoke_oauth_token", "starting revocation")
    
    try:
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
            auth_router.log_operation("revoke_oauth_token", "success")
            return {"message": "Token revoked successfully"}
        elif response.status_code == 404:
            # Token already revoked or doesn't exist
            auth_router.log_operation("revoke_oauth_token", "already revoked")
            return {"message": "Token already revoked"}
        else:
            response.raise_for_status()
            auth_router.log_operation("revoke_oauth_token", "success")
            return {"message": "Token revoked successfully"}

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except requests.RequestException as e:
        raise auth_router.handle_error(
            "revoke_oauth_token",
            HTTPException(
                status_code=500,
                detail=f"Failed to revoke token with GitHub: {str(e)}"
            )
        )
    except Exception as e:
        raise auth_router.handle_error("revoke_oauth_token", e)
