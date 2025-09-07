"""
Unit tests for GitOperationsService.
"""

import pytest

from models.github import GitHubRepository, GitHubUser
from services.github.git_operations_service import GitOperationsService


class TestGitOperationsService:
    """Test cases for GitOperationsService."""

    @pytest.fixture
    def git_operations_service(self, mock_github_token):
        """Create a GitOperationsService instance for testing."""
        return GitOperationsService(mock_github_token)

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

    @pytest.fixture
    def mock_template_data(self):
        """Create mock template data for testing."""
        return {
            "repo": "academicpages/academicpages.github.io",
            "tree": [
                {
                    "path": "index.md",
                    "type": "blob",
                    "sha": "abc123",
                    "mode": "100644"
                },
                {
                    "path": "_config.yml",
                    "type": "blob",
                    "sha": "def456",
                    "mode": "100644"
                },
                {
                    "path": ".github/workflows/deploy.yml",
                    "type": "blob",
                    "sha": "ghi789",
                    "mode": "100644"
                }
            ]
        }

    def test_git_operations_service_initialization(self, mock_github_token):
        """Test Git operations service initialization."""
        service = GitOperationsService(mock_github_token)

        assert service.access_token == mock_github_token
        assert service.base_url == "https://api.github.com"
        assert service.headers["Authorization"] == f"token {mock_github_token}"
        assert service.headers["Accept"] == "application/vnd.github.v3+json"
        assert service.headers["User-Agent"] == "one-click-paper-page/0.1.0"

    def test_git_operations_service_has_required_methods(self, git_operations_service):
        """Test that Git operations service has all required methods."""
        required_methods = [
            'get_reference',
            'create_blob',
            'create_tree',
            'create_commit',
            'update_reference',
            'get_blob_content',
            'copy_template_content_bulk'
        ]

        for method_name in required_methods:
            assert hasattr(git_operations_service, method_name)
            assert callable(getattr(git_operations_service, method_name))

    def test_git_operations_service_methods_are_async(self, git_operations_service):
        """Test that all Git operations methods are async."""
        import inspect

        async_methods = [
            'get_reference',
            'create_blob',
            'create_tree',
            'create_commit',
            'update_reference',
            'get_blob_content',
            'copy_template_content_bulk'
        ]

        for method_name in async_methods:
            method = getattr(git_operations_service, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    def test_git_operations_service_method_signatures(self, git_operations_service, mock_repository):
        """Test that Git operations methods have correct signatures."""
        import inspect

        # Test get_reference signature
        sig = inspect.signature(git_operations_service.get_reference)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'ref' in params

        # Test create_blob signature
        sig = inspect.signature(git_operations_service.create_blob)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'content' in params
        assert 'encoding' in params

        # Test create_tree signature
        sig = inspect.signature(git_operations_service.create_tree)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'tree_items' in params

        # Test create_commit signature
        sig = inspect.signature(git_operations_service.create_commit)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'message' in params
        assert 'tree_sha' in params
        assert 'parent_shas' in params

        # Test update_reference signature
        sig = inspect.signature(git_operations_service.update_reference)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'ref' in params
        assert 'sha' in params

        # Test get_blob_content signature
        sig = inspect.signature(git_operations_service.get_blob_content)
        params = list(sig.parameters.keys())
        assert 'template_repo' in params
        assert 'blob_sha' in params

        # Test copy_template_content_bulk signature
        sig = inspect.signature(git_operations_service.copy_template_content_bulk)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'template_data' in params

    def test_git_operations_service_default_parameters(self, git_operations_service, mock_repository):
        """Test that methods have correct default parameters."""
        import inspect

        # Test get_reference default ref parameter
        sig = inspect.signature(git_operations_service.get_reference)
        assert sig.parameters['ref'].default is None

        # Test create_blob default encoding parameter
        sig = inspect.signature(git_operations_service.create_blob)
        assert sig.parameters['encoding'].default == "base64"

    def test_template_data_structure_validation(self, mock_template_data):
        """Test that mock template data has the expected structure."""
        assert "repo" in mock_template_data
        assert "tree" in mock_template_data
        assert isinstance(mock_template_data["tree"], list)

        # Check that tree items have required fields
        for item in mock_template_data["tree"]:
            assert "path" in item
            assert "type" in item
            assert "sha" in item
            assert "mode" in item
            assert item["type"] == "blob"

    def test_repository_model_structure(self, mock_repository):
        """Test that mock repository has the expected structure."""
        assert hasattr(mock_repository, 'full_name')
        assert hasattr(mock_repository, 'default_branch')
        assert mock_repository.full_name == "testuser/test-repo"
        assert mock_repository.default_branch == "main"
