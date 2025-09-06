"""
Unit tests for WorkflowService.
"""

import pytest

from models.github import GitHubRepository, GitHubUser, WorkflowRun
from services.github.workflow_service import WorkflowService


class TestWorkflowService:
    """Test cases for WorkflowService."""

    @pytest.fixture
    def workflow_service(self, mock_github_token):
        """Create a WorkflowService instance for testing."""
        return WorkflowService(mock_github_token)

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
    def mock_template_data_with_workflow(self):
        """Create mock template data that includes deployment workflow."""
        return {
            "tree": [
                {
                    "path": "index.md",
                    "type": "blob",
                    "sha": "abc123",
                    "mode": "100644"
                },
                {
                    "path": ".github/workflows/jekyll.yml",
                    "type": "blob",
                    "sha": "def456",
                    "mode": "100644"
                }
            ]
        }

    @pytest.fixture
    def mock_template_data_without_workflow(self):
        """Create mock template data that does not include deployment workflow."""
        return {
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
                }
            ]
        }

    def test_workflow_service_initialization(self, mock_github_token):
        """Test workflow service initialization."""
        service = WorkflowService(mock_github_token)
        
        assert service.access_token == mock_github_token
        assert service.base_url == "https://api.github.com"
        assert service.headers["Authorization"] == f"token {mock_github_token}"
        assert service.headers["Accept"] == "application/vnd.github.v3+json"
        assert service.headers["User-Agent"] == "one-click-paper-page/0.1.0"

    def test_workflow_service_has_required_methods(self, workflow_service):
        """Test that workflow service has all required methods."""
        required_methods = [
            'has_deployment_workflow',
            'add_deployment_workflow_if_needed',
            'add_deployment_workflow',
            'get_latest_workflow_run',
            'get_jekyll_deployment_workflow'
        ]
        
        for method_name in required_methods:
            assert hasattr(workflow_service, method_name)
            assert callable(getattr(workflow_service, method_name))

    def test_workflow_service_methods_are_async_or_sync(self, workflow_service):
        """Test that workflow service methods have correct async/sync signatures."""
        import inspect
        
        # Async methods
        async_methods = [
            'has_deployment_workflow',
            'add_deployment_workflow_if_needed',
            'add_deployment_workflow',
            'get_latest_workflow_run'
        ]
        
        for method_name in async_methods:
            method = getattr(workflow_service, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"
        
        # Sync methods
        sync_methods = ['get_jekyll_deployment_workflow']
        
        for method_name in sync_methods:
            method = getattr(workflow_service, method_name)
            assert not inspect.iscoroutinefunction(method), f"{method_name} should be sync"

    @pytest.mark.asyncio
    async def test_has_deployment_workflow_with_workflow(self, workflow_service, mock_template_data_with_workflow):
        """Test detecting deployment workflow when it exists."""
        has_workflow = await workflow_service.has_deployment_workflow(mock_template_data_with_workflow)
        assert has_workflow is True

    @pytest.mark.asyncio
    async def test_has_deployment_workflow_without_workflow(self, workflow_service, mock_template_data_without_workflow):
        """Test detecting deployment workflow when it doesn't exist."""
        has_workflow = await workflow_service.has_deployment_workflow(mock_template_data_without_workflow)
        assert has_workflow is False

    def test_get_jekyll_deployment_workflow(self, workflow_service):
        """Test getting Jekyll deployment workflow content."""
        workflow_content = workflow_service.get_jekyll_deployment_workflow()
        
        assert isinstance(workflow_content, str)
        assert "name: Deploy Academic Site" in workflow_content
        assert "jekyll build" in workflow_content
        assert "actions/checkout@v4" in workflow_content
        assert "ruby/setup-ruby@v1" in workflow_content
        assert "actions/deploy-pages@v4" in workflow_content

    def test_workflow_content_structure(self, workflow_service):
        """Test that workflow content has proper YAML structure."""
        workflow_content = workflow_service.get_jekyll_deployment_workflow()
        
        # Check for key YAML sections
        assert "on:" in workflow_content
        assert "permissions:" in workflow_content
        assert "jobs:" in workflow_content
        assert "build:" in workflow_content
        assert "deploy:" in workflow_content
        
        # Check for GitHub Actions specific content
        assert "github-pages" in workflow_content
        assert "pages: write" in workflow_content
        assert "id-token: write" in workflow_content

    def test_workflow_service_method_signatures(self, workflow_service):
        """Test that workflow service methods have correct signatures."""
        import inspect
        
        # Test has_deployment_workflow signature
        sig = inspect.signature(workflow_service.has_deployment_workflow)
        params = list(sig.parameters.keys())
        assert 'template_data' in params
        
        # Test add_deployment_workflow_if_needed signature
        sig = inspect.signature(workflow_service.add_deployment_workflow_if_needed)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'template_data' in params
        
        # Test add_deployment_workflow signature
        sig = inspect.signature(workflow_service.add_deployment_workflow)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        
        # Test get_latest_workflow_run signature
        sig = inspect.signature(workflow_service.get_latest_workflow_run)
        params = list(sig.parameters.keys())
        assert 'repository' in params

    def test_template_data_structure_validation(self, mock_template_data_with_workflow, mock_template_data_without_workflow):
        """Test that mock template data has the expected structure."""
        # Test template with workflow
        assert "tree" in mock_template_data_with_workflow
        assert isinstance(mock_template_data_with_workflow["tree"], list)
        
        workflow_files = [
            f for f in mock_template_data_with_workflow["tree"] 
            if f["path"].startswith(".github/workflows/")
        ]
        assert len(workflow_files) == 1
        assert "jekyll" in workflow_files[0]["path"]
        
        # Test template without workflow
        assert "tree" in mock_template_data_without_workflow
        assert isinstance(mock_template_data_without_workflow["tree"], list)
        
        workflow_files = [
            f for f in mock_template_data_without_workflow["tree"] 
            if f["path"].startswith(".github/workflows/")
        ]
        assert len(workflow_files) == 0
