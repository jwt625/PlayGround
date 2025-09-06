"""
GitHub service for automated repository creation and deployment via GitHub Actions.

This module now uses the modular orchestrator approach while maintaining
backward compatibility with the original API.
"""

from services.github_service_orchestrator import GitHubServiceOrchestrator

# For backward compatibility, alias the orchestrator as GitHubService
GitHubService = GitHubServiceOrchestrator
