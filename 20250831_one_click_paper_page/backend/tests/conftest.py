"""
Shared test configuration and fixtures for GitHub services refactoring tests.
"""

from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest


@pytest.fixture
def mock_github_token() -> str:
    """Mock GitHub access token for testing."""
    return "ghp_test_token_1234567890abcdef"


@pytest.fixture
def mock_github_user() -> dict[str, Any]:
    """Mock GitHub user data."""
    return {
        "login": "testuser",
        "id": 12345,
        "name": "Test User",
        "email": "test@example.com",
        "avatar_url": "https://github.com/images/error/testuser_happy.gif"
    }


@pytest.fixture
def mock_github_repository() -> dict[str, Any]:
    """Mock GitHub repository data."""
    return {
        "id": 67890,
        "name": "test-repo",
        "full_name": "testuser/test-repo",
        "owner": {
            "login": "testuser",
            "id": 12345
        },
        "private": False,
        "html_url": "https://github.com/testuser/test-repo",
        "clone_url": "https://github.com/testuser/test-repo.git",
        "default_branch": "main",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_template_content() -> dict[str, Any]:
    """Mock template repository content."""
    return {
        "files": [
            {
                "name": "index.md",
                "path": "index.md",
                "content": "IyBUZXN0IFBhZ2U=",  # Base64 encoded "# Test Page"
                "type": "file"
            },
            {
                "name": "_config.yml",
                "path": "_config.yml",
                "content": "dGl0bGU6IFRlc3QgU2l0ZQ==",  # Base64 encoded "title: Test Site"
                "type": "file"
            }
        ],
        "repository_info": {
            "name": "academic-pages",
            "full_name": "academicpages/academicpages.github.io",
            "default_branch": "master"
        }
    }


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for GitHub API calls."""
    session = AsyncMock(spec=aiohttp.ClientSession)

    # Mock successful responses by default
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"success": True})
    mock_response.text = AsyncMock(return_value='{"success": true}')

    session.get.return_value.__aenter__.return_value = mock_response
    session.post.return_value.__aenter__.return_value = mock_response
    session.patch.return_value.__aenter__.return_value = mock_response
    session.put.return_value.__aenter__.return_value = mock_response

    return session


@pytest.fixture
def mock_github_api_responses():
    """Mock GitHub API response patterns."""
    return {
        "user": {
            "status": 200,
            "data": {
                "login": "testuser",
                "id": 12345,
                "name": "Test User"
            }
        },
        "token_scopes": {
            "status": 200,
            "headers": {"X-OAuth-Scopes": "repo,user,workflow"}
        },
        "create_repository": {
            "status": 201,
            "data": {
                "id": 67890,
                "name": "test-repo",
                "full_name": "testuser/test-repo"
            }
        },
        "fork_repository": {
            "status": 202,
            "data": {
                "id": 67891,
                "name": "forked-repo",
                "full_name": "testuser/forked-repo"
            }
        }
    }


class MockGitHubService:
    """Mock GitHub service for testing routers."""

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "one-click-paper-page/0.1.0",
        }

    async def get_authenticated_user(self):
        return {"login": "testuser", "id": 12345}

    async def get_token_scopes(self):
        return ["repo", "user", "workflow"]

    async def create_repository_optimized(self, request):
        return {
            "repository": {"name": request.name, "full_name": f"testuser/{request.name}"},
            "deployment_id": "test-deployment-123"
        }


@pytest.fixture
def mock_github_service():
    """Mock GitHub service instance."""
    return MockGitHubService("test-token")
