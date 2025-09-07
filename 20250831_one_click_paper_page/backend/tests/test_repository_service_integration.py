"""
Integration test to verify RepositoryService can be used alongside existing GitHubService.
"""

from services.github.repository_service import RepositoryService
from services.github_service import GitHubService


class TestRepositoryServiceIntegration:
    """Test integration between RepositoryService and existing code."""

    def test_repository_service_can_be_imported(self):
        """Test that RepositoryService can be imported without issues."""
        assert RepositoryService is not None

    def test_repository_service_instantiation(self):
        """Test that RepositoryService can be instantiated."""
        service = RepositoryService("test-token")
        assert service.access_token == "test-token"

    def test_github_service_still_works(self):
        """Test that existing GitHubService still works."""
        service = GitHubService("test-token")
        assert service.access_token == "test-token"

        # Check that GitHubService still has its methods
        assert hasattr(service, 'get_authenticated_user')
        assert hasattr(service, 'create_repository')

    def test_both_services_have_same_interface_methods(self):
        """Test that both services have the same interface for repository methods."""
        repo_service = RepositoryService("test-token")
        github_service = GitHubService("test-token")

        # Methods that should exist in both
        common_methods = [
            'get_authenticated_user',
            'get_token_scopes'
        ]

        for method in common_methods:
            assert hasattr(repo_service, method), f"RepositoryService missing {method}"
            assert hasattr(github_service, method), f"GitHubService missing {method}"

    def test_repository_service_headers_match_github_service(self):
        """Test that both services use the same header format."""
        token = "test-token-123"
        repo_service = RepositoryService(token)
        github_service = GitHubService(token)

        # Both should have the same headers
        assert repo_service.headers["Authorization"] == github_service.headers["Authorization"]
        assert repo_service.headers["Accept"] == github_service.headers["Accept"]
        assert repo_service.headers["User-Agent"] == github_service.headers["User-Agent"]
