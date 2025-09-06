"""
Test foundation setup for GitHub services refactoring.
"""

import pytest
from tests.conftest import MockGitHubService


class TestFoundationSetup:
    """Test that our foundation setup is working correctly."""

    def test_mock_github_service_creation(self, mock_github_token):
        """Test that we can create a mock GitHub service."""
        service = MockGitHubService(mock_github_token)
        assert service.token == mock_github_token
        assert "Authorization" in service.headers
        assert service.headers["Authorization"] == f"token {mock_github_token}"

    @pytest.mark.asyncio
    async def test_mock_github_service_methods(self, mock_github_service):
        """Test that mock GitHub service methods work."""
        user = await mock_github_service.get_authenticated_user()
        assert user["login"] == "testuser"
        assert user["id"] == 12345

        scopes = await mock_github_service.get_token_scopes()
        assert "repo" in scopes
        assert "user" in scopes
        assert "workflow" in scopes

    def test_mock_fixtures_available(self, mock_github_user, mock_github_repository, mock_template_content):
        """Test that all mock fixtures are available and properly structured."""
        # Test user fixture
        assert mock_github_user["login"] == "testuser"
        assert mock_github_user["id"] == 12345

        # Test repository fixture
        assert mock_github_repository["name"] == "test-repo"
        assert mock_github_repository["owner"]["login"] == "testuser"

        # Test template content fixture
        assert "files" in mock_template_content
        assert len(mock_template_content["files"]) == 2
        assert mock_template_content["files"][0]["name"] == "index.md"

    def test_directory_structure_exists(self):
        """Test that the required directory structure exists."""
        import os
        
        # Check that the services/github directory exists
        assert os.path.exists("services/github")
        
        # Check that the routers directory exists
        assert os.path.exists("routers")
        
        # Check that test directories exist
        assert os.path.exists("tests/services/github")
        assert os.path.exists("tests/routers")
