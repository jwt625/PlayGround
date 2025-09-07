"""
Tests for authentication router.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.auth_router import router
from models.github import OAuthTokenRequest, OAuthRevokeRequest


# Create test app with auth router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestExchangeOAuthToken:
    """Test OAuth token exchange endpoint."""
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', 'test_secret')
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    @patch('routers.auth_router.requests.post')
    def test_successful_token_exchange(self, mock_post):
        """Test successful OAuth token exchange."""
        # Mock successful GitHub response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "gho_test_token_123",
            "token_type": "bearer",
            "scope": "repo user:email"
        }
        mock_post.return_value = mock_response
        
        # Make request
        response = client.post("/api/github/oauth/token", json={
            "code": "test_auth_code",
            "redirect_uri": "http://localhost:5173/auth/callback"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "gho_test_token_123"
        assert data["token_type"] == "bearer"
        assert data["scope"] == "repo user:email"
        
        # Verify GitHub API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://github.com/login/oauth/access_token"
        assert call_args[1]["data"]["client_id"] == "test_client_id"
        assert call_args[1]["data"]["client_secret"] == "test_secret"
        assert call_args[1]["data"]["code"] == "test_auth_code"
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', None)
    def test_missing_client_secret(self):
        """Test error when client secret is not configured."""
        response = client.post("/api/github/oauth/token", json={
            "code": "test_auth_code",
            "redirect_uri": "http://localhost:5173/auth/callback"
        })
        
        assert response.status_code == 500
        assert "GitHub client secret not configured" in response.json()["detail"]
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', 'test_secret')
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    @patch('routers.auth_router.requests.post')
    def test_github_oauth_error(self, mock_post):
        """Test GitHub OAuth error response."""
        # Mock GitHub error response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "The provided authorization grant is invalid"
        }
        mock_post.return_value = mock_response
        
        response = client.post("/api/github/oauth/token", json={
            "code": "invalid_code",
            "redirect_uri": "http://localhost:5173/auth/callback"
        })
        
        assert response.status_code == 400
        assert "GitHub OAuth error" in response.json()["detail"]
        assert "provided authorization grant is invalid" in response.json()["detail"]
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', 'test_secret')
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    @patch('routers.auth_router.requests.post')
    def test_network_error(self, mock_post):
        """Test network error during token exchange."""
        # Mock network error
        mock_post.side_effect = Exception("Connection timeout")
        
        response = client.post("/api/github/oauth/token", json={
            "code": "test_auth_code",
            "redirect_uri": "http://localhost:5173/auth/callback"
        })
        
        assert response.status_code == 500
        assert "exchange_oauth_token failed" in response.json()["detail"]


class TestRevokeOAuthToken:
    """Test OAuth token revocation endpoint."""
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', 'test_secret')
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    @patch('routers.auth_router.requests.delete')
    def test_successful_token_revocation(self, mock_delete):
        """Test successful OAuth token revocation."""
        # Mock successful GitHub response (204 No Content)
        mock_response = Mock()
        mock_response.status_code = 204
        mock_delete.return_value = mock_response
        
        response = client.post("/api/github/oauth/revoke", json={
            "access_token": "gho_test_token_123"
        })
        
        assert response.status_code == 200
        assert response.json()["message"] == "Token revoked successfully"
        
        # Verify GitHub API call
        mock_delete.assert_called_once()
        call_args = mock_delete.call_args
        assert "applications/test_client_id/grant" in call_args[0][0]
        assert call_args[1]["json"]["access_token"] == "gho_test_token_123"
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', 'test_secret')
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    @patch('routers.auth_router.requests.delete')
    def test_token_already_revoked(self, mock_delete):
        """Test token already revoked (404 response)."""
        # Mock 404 response (token already revoked)
        mock_response = Mock()
        mock_response.status_code = 404
        mock_delete.return_value = mock_response
        
        response = client.post("/api/github/oauth/revoke", json={
            "access_token": "gho_test_token_123"
        })
        
        assert response.status_code == 200
        assert response.json()["message"] == "Token already revoked"
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', None)
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    def test_missing_client_credentials(self):
        """Test error when client credentials are not configured."""
        response = client.post("/api/github/oauth/revoke", json={
            "access_token": "gho_test_token_123"
        })
        
        assert response.status_code == 500
        assert "GitHub client credentials not configured" in response.json()["detail"]
    
    @patch('routers.auth_router.GITHUB_CLIENT_SECRET', 'test_secret')
    @patch('routers.auth_router.GITHUB_CLIENT_ID', 'test_client_id')
    @patch('routers.auth_router.requests.delete')
    def test_network_error_during_revocation(self, mock_delete):
        """Test network error during token revocation."""
        # Mock network error
        mock_delete.side_effect = Exception("Connection timeout")
        
        response = client.post("/api/github/oauth/revoke", json={
            "access_token": "gho_test_token_123"
        })
        
        assert response.status_code == 500
        assert "revoke_oauth_token failed" in response.json()["detail"]


class TestAuthRouterIntegration:
    """Test auth router integration."""
    
    def test_router_prefix_and_tags(self):
        """Test that router has correct prefix and tags."""
        assert router.prefix == "/api/github/oauth"
        assert "authentication" in router.tags
    
    def test_endpoints_registered(self):
        """Test that all endpoints are registered."""
        routes = [route.path for route in router.routes]
        assert "/api/github/oauth/token" in routes
        assert "/api/github/oauth/revoke" in routes
