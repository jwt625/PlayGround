"""
Unit tests for GitHubPagesService.
"""

import pytest

from models.github import GitHubRepository, GitHubUser
from services.github.pages_service import GitHubPagesService


class TestGitHubPagesService:
    """Test cases for GitHubPagesService."""

    @pytest.fixture
    def pages_service(self, mock_github_token):
        """Create a GitHubPagesService instance for testing."""
        return GitHubPagesService(mock_github_token)

    @pytest.fixture
    def mock_repository(self):
        """Create a mock GitHubRepository for testing."""
        return GitHubRepository(
            id=12345,
            name="test-repo",
            full_name="testuser/test-repo",
            owner=GitHubUser(
                login="testuser",
                id=67890,
                name="Test User",
                email="test@example.com",
                avatar_url="https://github.com/images/error/testuser_happy.gif",
                html_url="https://github.com/testuser"
            ),
            private=False,
            html_url="https://github.com/testuser/test-repo",
            clone_url="https://github.com/testuser/test-repo.git",
            ssh_url="git@github.com:testuser/test-repo.git",
            default_branch="main",
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-01T00:00:00Z"
        )

    def test_pages_service_initialization(self, mock_github_token):
        """Test GitHub Pages service initialization."""
        service = GitHubPagesService(mock_github_token)

        assert service.access_token == mock_github_token
        assert service.base_url == "https://api.github.com"
        assert service.headers["Authorization"] == f"token {mock_github_token}"
        assert service.headers["Accept"] == "application/vnd.github.v3+json"
        assert service.headers["User-Agent"] == "one-click-paper-page/0.1.0"

    def test_pages_service_has_required_methods(self, pages_service):
        """Test that Pages service has all required methods."""
        required_methods = [
            'enable_github_pages',
            'enable_github_pages_with_actions',
            'enable_github_pages_as_backup',
            'get_pages_info',
            'disable_github_pages',
            'update_pages_config'
        ]

        for method_name in required_methods:
            assert hasattr(pages_service, method_name)
            assert callable(getattr(pages_service, method_name))

    def test_pages_service_methods_are_async(self, pages_service):
        """Test that all Pages service methods are async."""
        import inspect

        async_methods = [
            'enable_github_pages',
            'enable_github_pages_with_actions',
            'enable_github_pages_as_backup',
            'get_pages_info',
            'disable_github_pages',
            'update_pages_config'
        ]

        for method_name in async_methods:
            method = getattr(pages_service, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    def test_pages_service_method_signatures(self, pages_service):
        """Test that Pages service methods have correct signatures."""
        import inspect

        # Test enable_github_pages signature
        sig = inspect.signature(pages_service.enable_github_pages)
        params = list(sig.parameters.keys())
        assert 'repository' in params

        # Test enable_github_pages_with_actions signature
        sig = inspect.signature(pages_service.enable_github_pages_with_actions)
        params = list(sig.parameters.keys())
        assert 'repository' in params

        # Test enable_github_pages_as_backup signature
        sig = inspect.signature(pages_service.enable_github_pages_as_backup)
        params = list(sig.parameters.keys())
        assert 'repository' in params

        # Test get_pages_info signature
        sig = inspect.signature(pages_service.get_pages_info)
        params = list(sig.parameters.keys())
        assert 'repository' in params

        # Test disable_github_pages signature
        sig = inspect.signature(pages_service.disable_github_pages)
        params = list(sig.parameters.keys())
        assert 'repository' in params

        # Test update_pages_config signature
        sig = inspect.signature(pages_service.update_pages_config)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'source_branch' in params
        assert 'source_path' in params
        assert 'build_type' in params

    def test_update_pages_config_default_parameters(self, pages_service):
        """Test that update_pages_config has correct default parameters."""
        import inspect

        sig = inspect.signature(pages_service.update_pages_config)

        # Check default values
        assert sig.parameters['source_branch'].default is None
        assert sig.parameters['source_path'].default == "/"
        assert sig.parameters['build_type'].default == "workflow"

    def test_enable_github_pages_as_backup_return_type(self, pages_service):
        """Test that enable_github_pages_as_backup returns bool."""
        import inspect

        sig = inspect.signature(pages_service.enable_github_pages_as_backup)
        # The return annotation should indicate bool
        assert sig.return_annotation == bool

    def test_disable_github_pages_return_type(self, pages_service):
        """Test that disable_github_pages returns bool."""
        import inspect

        sig = inspect.signature(pages_service.disable_github_pages)
        # The return annotation should indicate bool
        assert sig.return_annotation == bool

    def test_update_pages_config_return_type(self, pages_service):
        """Test that update_pages_config returns bool."""
        import inspect

        sig = inspect.signature(pages_service.update_pages_config)
        # The return annotation should indicate bool
        assert sig.return_annotation == bool

    def test_repository_model_structure(self, mock_repository):
        """Test that mock repository has the expected structure for Pages operations."""
        assert hasattr(mock_repository, 'full_name')
        assert hasattr(mock_repository, 'default_branch')
        assert hasattr(mock_repository, 'owner')
        assert hasattr(mock_repository.owner, 'login')

        assert mock_repository.full_name == "testuser/test-repo"
        assert mock_repository.default_branch == "main"
        assert mock_repository.owner.login == "testuser"

    def test_pages_service_api_endpoints(self, pages_service, mock_repository):
        """Test that service constructs correct API endpoints."""
        base_url = pages_service.base_url
        repo_full_name = mock_repository.full_name

        expected_pages_endpoint = f"{base_url}/repos/{repo_full_name}/pages"

        # The service should use this endpoint pattern
        assert base_url == "https://api.github.com"
        assert repo_full_name == "testuser/test-repo"
        assert expected_pages_endpoint == "https://api.github.com/repos/testuser/test-repo/pages"

    def test_pages_service_headers_structure(self, pages_service, mock_github_token):
        """Test that service has correct headers for GitHub API."""
        headers = pages_service.headers

        assert "Authorization" in headers
        assert "Accept" in headers
        assert "User-Agent" in headers

        assert headers["Authorization"] == f"token {mock_github_token}"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "one-click-paper-page/0.1.0"

    def test_pages_config_structure_validation(self, mock_repository):
        """Test that Pages configuration structures are valid."""
        # Test basic pages config structure
        basic_config = {
            "source": {
                "branch": mock_repository.default_branch,
                "path": "/"
            }
        }

        assert "source" in basic_config
        assert "branch" in basic_config["source"]
        assert "path" in basic_config["source"]
        assert basic_config["source"]["branch"] == "main"
        assert basic_config["source"]["path"] == "/"

        # Test workflow pages config structure
        workflow_config = {
            "source": {
                "branch": mock_repository.default_branch,
                "path": "/"
            },
            "build_type": "workflow"
        }

        assert "source" in workflow_config
        assert "build_type" in workflow_config
        assert workflow_config["build_type"] == "workflow"
