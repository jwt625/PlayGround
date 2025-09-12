"""
Integration tests for GitHubServiceOrchestrator.
"""

import pytest

from models.github import CreateRepositoryRequest, TemplateType
from services.github_service import GitHubService
from services.github_service_orchestrator import GitHubServiceOrchestrator


class TestGitHubServiceOrchestrator:
    """Test cases for GitHubServiceOrchestrator."""

    @pytest.fixture
    def orchestrator(self, mock_github_token):
        """Create a GitHubServiceOrchestrator instance for testing."""
        return GitHubServiceOrchestrator(mock_github_token)

    @pytest.fixture
    def original_service(self, mock_github_token):
        """Create original GitHubService for comparison."""
        return GitHubService(mock_github_token)

    @pytest.fixture
    def create_repo_request(self):
        """Sample repository creation request."""
        return CreateRepositoryRequest(
            name="test-paper-repo",
            description="Test academic paper repository",
            template=TemplateType.ACADEMIC_PAGES,
            conversion_job_id="test-job-123"
        )

    def test_orchestrator_initialization(self, mock_github_token):
        """Test orchestrator initialization."""
        orchestrator = GitHubServiceOrchestrator(mock_github_token)

        assert orchestrator.access_token == mock_github_token

        # Check that all modular services are initialized
        assert orchestrator.repository_service is not None
        assert orchestrator.template_manager is not None
        assert orchestrator.git_operations_service is not None
        assert orchestrator.workflow_service is not None
        assert orchestrator.pages_service is not None
        assert orchestrator.deployment_tracker is not None

        # Check compatibility properties
        assert orchestrator.headers is not None
        assert orchestrator.base_url == "https://api.github.com"
        assert orchestrator.template_service is not None

    def test_orchestrator_has_same_public_api_as_original(self, orchestrator, original_service):
        """Test that orchestrator has the same public API as original service."""
        # Get public methods from both services
        orchestrator_methods = [
            method for method in dir(orchestrator)
            if not method.startswith('_') and callable(getattr(orchestrator, method))
        ]

        original_methods = [
            method for method in dir(original_service)
            if not method.startswith('_') and callable(getattr(original_service, method))
        ]

        # Key public methods that must exist in both
        required_methods = [
            'get_authenticated_user',
            'get_token_scopes',
            'list_templates',
            'create_repository',
            'create_repository_optimized',
            'create_repository_from_template',
            'get_deployment_status',
            'enable_github_pages_as_backup'
        ]

        for method in required_methods:
            assert method in orchestrator_methods, f"Orchestrator missing method: {method}"
            assert method in original_methods, f"Original service missing method: {method}"

    def test_orchestrator_method_signatures_match_original(self, orchestrator, original_service):
        """Test that orchestrator method signatures match original service."""
        import inspect

        methods_to_check = [
            'get_authenticated_user',
            'get_token_scopes',
            'list_templates',
            'create_repository',
            'create_repository_optimized',
            'get_deployment_status'
        ]

        for method_name in methods_to_check:
            if hasattr(orchestrator, method_name) and hasattr(original_service, method_name):
                orchestrator_sig = inspect.signature(getattr(orchestrator, method_name))
                original_sig = inspect.signature(getattr(original_service, method_name))

                # Parameters should match
                orchestrator_params = list(orchestrator_sig.parameters.keys())
                original_params = list(original_sig.parameters.keys())

                assert orchestrator_params == original_params, f"Parameter mismatch in {method_name}"

    def test_orchestrator_async_methods_are_async(self, orchestrator):
        """Test that async methods in orchestrator are properly async."""
        import inspect

        async_methods = [
            'get_authenticated_user',
            'get_token_scopes',
            'create_repository',
            'create_repository_optimized',
            'create_repository_from_template',
            'deploy_converted_content',
            'get_deployment_status',
            'enable_github_pages_as_backup',
            'create_dual_deployment'
        ]

        for method_name in async_methods:
            if hasattr(orchestrator, method_name):
                method = getattr(orchestrator, method_name)
                assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    def test_orchestrator_sync_methods_are_sync(self, orchestrator):
        """Test that sync methods in orchestrator are properly sync."""
        import inspect

        sync_methods = []  # list_templates is now async in original service

        for method_name in sync_methods:
            if hasattr(orchestrator, method_name):
                method = getattr(orchestrator, method_name)
                assert not inspect.iscoroutinefunction(method), f"{method_name} should be sync"

    def test_orchestrator_delegates_to_correct_services(self, orchestrator):
        """Test that orchestrator properly delegates to modular services."""
        # Check that services have the expected methods
        assert hasattr(orchestrator.repository_service, 'get_authenticated_user')
        assert hasattr(orchestrator.repository_service, 'create_empty_repository')

        assert hasattr(orchestrator.template_manager, 'get_template_content_cached')
        assert hasattr(orchestrator.template_manager, 'filter_essential_template_files')

        assert hasattr(orchestrator.git_operations_service, 'copy_template_content_bulk')
        assert hasattr(orchestrator.git_operations_service, 'create_blob')

        assert hasattr(orchestrator.workflow_service, 'add_deployment_workflow_if_needed')
        assert hasattr(orchestrator.workflow_service, 'get_jekyll_deployment_workflow')

        assert hasattr(orchestrator.pages_service, 'enable_github_pages_with_actions')
        assert hasattr(orchestrator.pages_service, 'enable_github_pages_as_backup')

        assert hasattr(orchestrator.deployment_tracker, 'create_deployment_job')
        assert hasattr(orchestrator.deployment_tracker, 'get_deployment_status')

    def test_orchestrator_maintains_compatibility_properties(self, orchestrator, original_service):
        """Test that orchestrator maintains compatibility properties."""
        # Both should have same headers structure
        assert "Authorization" in orchestrator.headers
        assert "Accept" in orchestrator.headers
        assert "User-Agent" in orchestrator.headers

        assert "Authorization" in original_service.headers
        assert "Accept" in original_service.headers
        assert "User-Agent" in original_service.headers

        # Headers should be identical
        assert orchestrator.headers["Accept"] == original_service.headers["Accept"]
        assert orchestrator.headers["User-Agent"] == original_service.headers["User-Agent"]

        # Base URL should match
        assert orchestrator.base_url == original_service.base_url

        # Template service should be available
        assert orchestrator.template_service is not None
        assert original_service.template_service is not None

    def test_orchestrator_service_initialization_with_same_token(self, mock_github_token):
        """Test that all services are initialized with the same token."""
        orchestrator = GitHubServiceOrchestrator(mock_github_token)

        # All services should have the same access token
        assert orchestrator.repository_service.access_token == mock_github_token
        assert orchestrator.template_manager.access_token == mock_github_token
        assert orchestrator.git_operations_service.access_token == mock_github_token
        assert orchestrator.workflow_service.access_token == mock_github_token
        assert orchestrator.pages_service.access_token == mock_github_token
        assert orchestrator.deployment_tracker.access_token == mock_github_token

    def test_orchestrator_can_be_used_as_drop_in_replacement(self, orchestrator, original_service):
        """Test that orchestrator can be used as a drop-in replacement."""
        # Both should have the same interface for key methods
        key_methods = ['list_templates']

        for method_name in key_methods:
            orchestrator_method = getattr(orchestrator, method_name)
            original_method = getattr(original_service, method_name)

            # Both should be callable
            assert callable(orchestrator_method)
            assert callable(original_method)

            # Both should be async methods now
            import inspect
            assert inspect.iscoroutinefunction(orchestrator_method)
            assert inspect.iscoroutinefunction(original_method)

    def test_create_repo_request_validation(self, create_repo_request):
        """Test that CreateRepositoryRequest is properly structured for orchestrator."""
        assert create_repo_request.name == "test-paper-repo"
        assert create_repo_request.description == "Test academic paper repository"
        assert create_repo_request.template == TemplateType.ACADEMIC_PAGES
        assert create_repo_request.conversion_job_id == "test-job-123"

    def test_dual_deployment_result_structure(self, orchestrator):
        """Test that create_dual_deployment method has correct signature and return type."""
        import inspect

        from models.github import DualDeploymentResult

        # Check method signature
        method = getattr(orchestrator, 'create_dual_deployment')
        assert callable(method)
        assert inspect.iscoroutinefunction(method), "create_dual_deployment should be async"

        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert 'request' in params

        # Check return type annotation
        assert sig.return_annotation == DualDeploymentResult

    def test_dual_deployment_result_model_fields(self):
        """Test that DualDeploymentResult has the expected fields."""
        from datetime import datetime

        from models.github import (
            DeploymentStatus,
            DualDeploymentResult,
            GitHubRepository,
            GitHubUser,
        )

        # Create a mock repository
        user = GitHubUser(
            login="testuser",
            id=12345,
            name="Test User",
            email="test@example.com",
            avatar_url="https://github.com/images/error/testuser_happy.gif",
            html_url="https://github.com/testuser"
        )

        repo = GitHubRepository(
            id=67890,
            name="test-repo",
            full_name="testuser/test-repo",
            description="Test repository",
            html_url="https://github.com/testuser/test-repo",
            clone_url="https://github.com/testuser/test-repo.git",
            ssh_url="git@github.com:testuser/test-repo.git",
            default_branch="main",
            private=False,
            owner=user,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Test DualDeploymentResult creation
        result = DualDeploymentResult(
            standalone_repo=repo,
            standalone_url="https://github.com/testuser/test-repo",
            deployment_id="test-deployment-123",
            status=DeploymentStatus.PENDING,
            message="Test dual deployment"
        )

        assert result.standalone_repo == repo
        assert result.standalone_url == "https://github.com/testuser/test-repo"
        assert result.deployment_id == "test-deployment-123"
        assert result.status == DeploymentStatus.PENDING
        assert result.message == "Test dual deployment"

    def test_customize_html_content_with_images(self, orchestrator):
        """Test HTML content customization with image conversion."""
        from models.github import DeploymentConfig, TemplateType

        config = DeploymentConfig(
            paper_title="Test Paper",
            paper_authors=["John Doe"],
            template=TemplateType.MINIMAL_ACADEMIC,
            repository_name="test-repo"
        )

        html_content = """
        <html>
        <head>
            <title>Document</title>
        </head>
        <body>
            <p>Some text before image.</p>
            <p>![](_page_1_Figure_2.jpeg)</p>
            <p>Fig. 1. Caption text.</p>
            <p>More content.</p>
            <p>![](_page_7_Figure_1.jpeg)</p>
            <p>Another figure.</p>
        </body>
        </html>
        """

        result = orchestrator._customize_html_content(html_content, config)

        # Check that title and author are updated
        assert "Test Paper" in result
        assert "John Doe" in result

        # Check that Markdown image syntax is converted to HTML img tags
        assert "![](_page_1_Figure_2.jpeg)" not in result
        assert "![](_page_7_Figure_1.jpeg)" not in result
        assert '<img src="images/_page_1_Figure_2.jpeg" alt="Figure 2"' in result
        assert '<img src="images/_page_7_Figure_1.jpeg" alt="Figure 1"' in result
        assert 'style="max-width: 100%; height: auto;"' in result
