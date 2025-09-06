"""
Unit tests for DeploymentTracker.
"""

import pytest
from datetime import datetime, timedelta

from models.github import (
    CreateRepositoryRequest, 
    DeploymentJob, 
    DeploymentStatus, 
    GitHubRepository, 
    GitHubUser,
    TemplateType,
    WorkflowRun
)
from services.github.deployment_tracker import DeploymentTracker


class TestDeploymentTracker:
    """Test cases for DeploymentTracker."""

    @pytest.fixture
    def deployment_tracker(self, mock_github_token):
        """Create a DeploymentTracker instance for testing."""
        return DeploymentTracker(mock_github_token)

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
    def mock_create_request(self):
        """Create a mock CreateRepositoryRequest for testing."""
        return CreateRepositoryRequest(
            name="test-paper-repo",
            description="Test academic paper repository",
            template=TemplateType.ACADEMIC_PAGES,
            conversion_job_id="test-job-123"
        )

    @pytest.fixture
    def mock_workflow_run(self):
        """Create a mock WorkflowRun for testing."""
        return WorkflowRun(
            id=123456,
            name="Deploy Academic Site",
            status="completed",
            conclusion="success",
            html_url="https://github.com/testuser/test-repo/actions/runs/123456",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    def test_deployment_tracker_initialization(self, mock_github_token):
        """Test deployment tracker initialization."""
        tracker = DeploymentTracker(mock_github_token)
        
        assert tracker.access_token == mock_github_token
        assert tracker.base_url == "https://api.github.com"
        assert tracker.headers["Authorization"] == f"token {mock_github_token}"
        assert tracker.headers["Accept"] == "application/vnd.github.v3+json"
        assert tracker.headers["User-Agent"] == "one-click-paper-page/0.1.0"
        assert isinstance(tracker._deployments, dict)
        assert len(tracker._deployments) == 0

    def test_deployment_tracker_has_required_methods(self, deployment_tracker):
        """Test that deployment tracker has all required methods."""
        required_methods = [
            'create_deployment_job',
            'get_deployment_job',
            'update_deployment_status',
            'get_deployment_status',
            'update_deployment_from_workflow',
            'set_workflow_run',
            'get_all_deployments',
            'cleanup_completed_deployments'
        ]
        
        for method_name in required_methods:
            assert hasattr(deployment_tracker, method_name)
            assert callable(getattr(deployment_tracker, method_name))

    def test_deployment_tracker_methods_async_sync(self, deployment_tracker):
        """Test that deployment tracker methods have correct async/sync signatures."""
        import inspect
        
        # Async methods
        async_methods = [
            'get_deployment_status',
            'update_deployment_from_workflow'
        ]
        
        for method_name in async_methods:
            method = getattr(deployment_tracker, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"
        
        # Sync methods
        sync_methods = [
            'create_deployment_job',
            'get_deployment_job',
            'update_deployment_status',
            'set_workflow_run',
            'get_all_deployments',
            'cleanup_completed_deployments'
        ]
        
        for method_name in sync_methods:
            method = getattr(deployment_tracker, method_name)
            assert not inspect.iscoroutinefunction(method), f"{method_name} should be sync"

    def test_create_deployment_job(self, deployment_tracker, mock_repository, mock_create_request):
        """Test creating a deployment job."""
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        
        assert isinstance(deployment_id, str)
        assert len(deployment_id) > 0
        
        # Check that deployment was stored
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        assert deployment is not None
        assert deployment.id == deployment_id
        assert deployment.repository == mock_repository
        assert deployment.conversion_job_id == mock_create_request.conversion_job_id
        assert deployment.status == DeploymentStatus.PENDING
        assert deployment.template == mock_create_request.template
        assert isinstance(deployment.created_at, datetime)

    def test_get_deployment_job_existing(self, deployment_tracker, mock_repository, mock_create_request):
        """Test getting an existing deployment job."""
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        assert deployment is not None
        assert deployment.id == deployment_id

    def test_get_deployment_job_nonexistent(self, deployment_tracker):
        """Test getting a non-existent deployment job."""
        deployment = deployment_tracker.get_deployment_job("nonexistent-id")
        assert deployment is None

    def test_update_deployment_status_success(self, deployment_tracker, mock_repository, mock_create_request):
        """Test updating deployment status successfully."""
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        
        success = deployment_tracker.update_deployment_status(
            deployment_id, 
            DeploymentStatus.IN_PROGRESS,
            message="Deployment started"
        )
        
        assert success is True
        
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        assert deployment.status == DeploymentStatus.IN_PROGRESS
        assert "Deployment started" in deployment.build_logs

    def test_update_deployment_status_with_error(self, deployment_tracker, mock_repository, mock_create_request):
        """Test updating deployment status with error."""
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        
        success = deployment_tracker.update_deployment_status(
            deployment_id,
            DeploymentStatus.FAILURE,
            error_message="Deployment failed"
        )
        
        assert success is True
        
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        assert deployment.status == DeploymentStatus.FAILURE
        assert deployment.error_message == "Deployment failed"
        assert "Error: Deployment failed" in deployment.build_logs
        assert deployment.completed_at is not None

    def test_update_deployment_status_nonexistent(self, deployment_tracker):
        """Test updating status of non-existent deployment."""
        success = deployment_tracker.update_deployment_status(
            "nonexistent-id",
            DeploymentStatus.SUCCESS
        )
        
        assert success is False

    def test_set_workflow_run(self, deployment_tracker, mock_repository, mock_create_request, mock_workflow_run):
        """Test setting workflow run for deployment."""
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        
        success = deployment_tracker.set_workflow_run(deployment_id, mock_workflow_run)
        assert success is True
        
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        assert deployment.workflow_run == mock_workflow_run

    def test_set_workflow_run_nonexistent(self, deployment_tracker, mock_workflow_run):
        """Test setting workflow run for non-existent deployment."""
        success = deployment_tracker.set_workflow_run("nonexistent-id", mock_workflow_run)
        assert success is False

    def test_get_all_deployments(self, deployment_tracker, mock_repository, mock_create_request):
        """Test getting all deployments."""
        # Initially empty
        deployments = deployment_tracker.get_all_deployments()
        assert len(deployments) == 0
        
        # Create some deployments
        id1 = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        id2 = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        
        deployments = deployment_tracker.get_all_deployments()
        assert len(deployments) == 2
        assert id1 in deployments
        assert id2 in deployments

    def test_cleanup_completed_deployments(self, deployment_tracker, mock_repository, mock_create_request):
        """Test cleaning up completed deployments."""
        # Create deployment and mark as completed
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        
        # Manually set completed_at to old date
        deployment.completed_at = datetime.now() - timedelta(hours=25)
        deployment.status = DeploymentStatus.SUCCESS
        
        # Cleanup should remove it
        cleanup_count = deployment_tracker.cleanup_completed_deployments(max_age_hours=24)
        assert cleanup_count == 1
        
        # Deployment should be gone
        assert deployment_tracker.get_deployment_job(deployment_id) is None

    def test_cleanup_completed_deployments_recent(self, deployment_tracker, mock_repository, mock_create_request):
        """Test that recent completed deployments are not cleaned up."""
        # Create deployment and mark as recently completed
        deployment_id = deployment_tracker.create_deployment_job(mock_repository, mock_create_request)
        deployment = deployment_tracker.get_deployment_job(deployment_id)
        
        # Set completed_at to recent date
        deployment.completed_at = datetime.now() - timedelta(hours=1)
        deployment.status = DeploymentStatus.SUCCESS
        
        # Cleanup should not remove it
        cleanup_count = deployment_tracker.cleanup_completed_deployments(max_age_hours=24)
        assert cleanup_count == 0
        
        # Deployment should still exist
        assert deployment_tracker.get_deployment_job(deployment_id) is not None

    def test_deployment_tracker_method_signatures(self, deployment_tracker):
        """Test that deployment tracker methods have correct signatures."""
        import inspect
        
        # Test create_deployment_job signature
        sig = inspect.signature(deployment_tracker.create_deployment_job)
        params = list(sig.parameters.keys())
        assert 'repository' in params
        assert 'request' in params
        
        # Test update_deployment_status signature
        sig = inspect.signature(deployment_tracker.update_deployment_status)
        params = list(sig.parameters.keys())
        assert 'deployment_id' in params
        assert 'status' in params
        assert 'message' in params
        assert 'error_message' in params
