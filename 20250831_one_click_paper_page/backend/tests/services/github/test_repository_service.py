"""
Unit tests for RepositoryService.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import aiohttp

from models.github import CreateRepositoryRequest, GitHubUser, GitHubRepository
from services.github.repository_service import RepositoryService


class TestRepositoryService:
    """Test cases for RepositoryService."""

    @pytest.fixture
    def repository_service(self, mock_github_token):
        """Create a RepositoryService instance for testing."""
        return RepositoryService(mock_github_token)

    @pytest.fixture
    def create_repo_request(self):
        """Sample repository creation request."""
        return CreateRepositoryRequest(
            name="test-paper-repo",
            description="Test academic paper repository",
            template="academic-pages",
            conversion_job_id="test-job-123"
        )

    def test_service_initialization(self, mock_github_token):
        """Test service initialization with correct headers."""
        service = RepositoryService(mock_github_token)

        assert service.access_token == mock_github_token
        assert service.base_url == "https://api.github.com"
        assert service.headers["Authorization"] == f"token {mock_github_token}"
        assert service.headers["Accept"] == "application/vnd.github.v3+json"
        assert service.headers["User-Agent"] == "one-click-paper-page/0.1.0"

    def test_service_has_required_methods(self, repository_service):
        """Test that service has all required methods."""
        assert hasattr(repository_service, 'get_authenticated_user')
        assert hasattr(repository_service, 'get_token_scopes')
        assert hasattr(repository_service, 'create_empty_repository')
        assert hasattr(repository_service, 'fork_repository')
        assert hasattr(repository_service, 'wait_for_repository_ready')
        assert hasattr(repository_service, 'get_repository_info')

        # Check that methods are callable
        assert callable(repository_service.get_authenticated_user)
        assert callable(repository_service.get_token_scopes)
        assert callable(repository_service.create_empty_repository)
        assert callable(repository_service.fork_repository)
        assert callable(repository_service.wait_for_repository_ready)
        assert callable(repository_service.get_repository_info)

    def test_create_repo_request_validation(self, create_repo_request):
        """Test that CreateRepositoryRequest is properly structured."""
        assert create_repo_request.name == "test-paper-repo"
        assert create_repo_request.description == "Test academic paper repository"
        assert create_repo_request.template == "academic-pages"
        assert create_repo_request.conversion_job_id == "test-job-123"
        assert create_repo_request.private is False

    def test_repository_service_methods_exist_and_are_async(self, repository_service):
        """Test that all required async methods exist."""
        import inspect

        # Check async methods
        async_methods = [
            'get_authenticated_user',
            'get_token_scopes',
            'create_empty_repository',
            'fork_repository',
            'wait_for_repository_ready',
            'get_repository_info'
        ]

        for method_name in async_methods:
            method = getattr(repository_service, method_name)
            assert callable(method)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    def test_github_user_model_validation(self):
        """Test that GitHubUser model requires all necessary fields."""
        from models.github import GitHubUser

        # Test that GitHubUser requires html_url field
        user_data = {
            "login": "testuser",
            "id": 12345,
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://github.com/images/error/testuser_happy.gif",
            "html_url": "https://github.com/testuser"
        }

        # This should work
        user = GitHubUser(**user_data)
        assert user.login == "testuser"
        assert user.html_url == "https://github.com/testuser"

        # Test that missing html_url raises validation error
        incomplete_data = user_data.copy()
        del incomplete_data["html_url"]

        try:
            GitHubUser(**incomplete_data)
            assert False, "Should have raised validation error for missing html_url"
        except Exception as e:
            assert "html_url" in str(e), "Error should mention missing html_url field"

    def test_github_repository_model_validation(self):
        """Test that GitHubRepository model requires all necessary fields."""
        from models.github import GitHubRepository, GitHubUser
        from datetime import datetime

        # Create a valid user first
        user = GitHubUser(
            login="testuser",
            id=12345,
            name="Test User",
            email="test@example.com",
            avatar_url="https://github.com/images/error/testuser_happy.gif",
            html_url="https://github.com/testuser"
        )

        # Test that GitHubRepository requires ssh_url field
        repo_data = {
            "id": 67890,
            "name": "test-repo",
            "full_name": "testuser/test-repo",
            "description": "Test repository",
            "html_url": "https://github.com/testuser/test-repo",
            "clone_url": "https://github.com/testuser/test-repo.git",
            "ssh_url": "git@github.com:testuser/test-repo.git",
            "default_branch": "main",
            "private": False,
            "owner": user,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        # This should work
        repo = GitHubRepository(**repo_data)
        assert repo.name == "test-repo"
        assert repo.ssh_url == "git@github.com:testuser/test-repo.git"

        # Test that missing ssh_url raises validation error
        incomplete_data = repo_data.copy()
        del incomplete_data["ssh_url"]

        try:
            GitHubRepository(**incomplete_data)
            assert False, "Should have raised validation error for missing ssh_url"
        except Exception as e:
            assert "ssh_url" in str(e), "Error should mention missing ssh_url field"
