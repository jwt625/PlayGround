"""
GitHub Pages service for Pages enablement and configuration.

This service handles:
- GitHub Pages enablement with different source configurations
- GitHub Actions integration for Pages deployment
- Pages configuration management
- Backup Pages enablement
"""

import logging
from typing import Any

import aiohttp

from models.github import GitHubRepository

logger = logging.getLogger(__name__)


class GitHubPagesService:
    """Service for GitHub Pages configuration and management."""

    def __init__(self, access_token: str):
        """
        Initialize GitHub Pages service.

        Args:
            access_token: GitHub OAuth access token
        """
        self.access_token = access_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "one-click-paper-page/0.1.0",
        }

    async def enable_github_pages(self, repository: GitHubRepository) -> None:
        """Enable GitHub Pages for the repository with branch source."""
        try:
            pages_config = {
                "source": {
                    "branch": repository.default_branch,
                    "path": "/"
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/repos/{repository.full_name}/pages",
                    headers=self.headers,
                    json=pages_config
                ) as response:
                    if response.status == 201:
                        logger.info(f"GitHub Pages enabled for {repository.full_name}")
                    elif response.status == 409:
                        logger.info(
                            f"GitHub Pages already enabled for {repository.full_name}"
                        )
                    else:
                        error_data = await response.text()
                        logger.warning(
                            f"Failed to enable GitHub Pages: {response.status} - "
                            f"{error_data}"
                        )

        except Exception as e:
            logger.warning(
                f"Failed to enable GitHub Pages for {repository.full_name}: {e}"
            )
            # Don't fail the deployment if Pages setup fails

    async def enable_github_pages_with_actions(
        self, repository: GitHubRepository
    ) -> None:
        """Enable GitHub Pages with GitHub Actions as the source."""
        try:
            pages_config = {
                "source": {
                    "branch": repository.default_branch,
                    "path": "/"
                },
                "build_type": "workflow"  # Use GitHub Actions for deployment
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/repos/{repository.full_name}/pages",
                    headers=self.headers,
                    json=pages_config
                ) as response:
                    if response.status == 201:
                        logger.info(
                            f"Enabled GitHub Pages with Actions for "
                            f"{repository.full_name}"
                        )
                    elif response.status == 409:
                        # Pages already enabled, update to use Actions
                        async with session.put(
                            f"{self.base_url}/repos/{repository.full_name}/pages",
                            headers=self.headers,
                            json=pages_config
                        ) as update_response:
                            if update_response.status == 200:
                                logger.info(
                                    f"Updated GitHub Pages to use Actions for "
                                    f"{repository.full_name}"
                                )
                            else:
                                error_data = await update_response.text()
                                logger.warning(
                                    f"Failed to update Pages to Actions: "
                                    f"{update_response.status} - {error_data}"
                                )
                    else:
                        error_data = await response.text()
                        logger.warning(
                            f"Failed to enable GitHub Pages with Actions: "
                            f"{response.status} - {error_data}"
                        )

        except Exception as e:
            logger.warning(
                f"Failed to enable GitHub Pages with Actions for "
                f"{repository.full_name}: {e}"
            )
            # Don't fail the deployment if Pages setup fails

    async def enable_github_pages_as_backup(
        self, repository: GitHubRepository
    ) -> bool:
        """
        Enable GitHub Pages as a backup option when automated deployment fails.

        Args:
            repository: GitHub repository to enable Pages for

        Returns:
            True if Pages was enabled successfully, False otherwise
        """
        try:
            logger.info(f"Enabling GitHub Pages as backup for {repository.full_name}")
            await self.enable_github_pages_with_actions(repository)
            return True
        except Exception as e:
            logger.error(
                f"Failed to enable GitHub Pages backup for {repository.full_name}: {e}"
            )
            return False

    async def get_pages_info(self, repository: GitHubRepository) -> dict[str, Any] | None:
        """Get GitHub Pages information for a repository."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/repos/{repository.full_name}/pages",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        logger.info(f"GitHub Pages not enabled for {repository.full_name}")
                        return None
                    else:
                        error_data = await response.text()
                        logger.warning(
                            f"Failed to get Pages info: {response.status} - {error_data}"
                        )
                        return None
        except Exception as e:
            logger.error(f"Failed to get Pages info for {repository.full_name}: {e}")
            return None

    async def disable_github_pages(self, repository: GitHubRepository) -> bool:
        """
        Disable GitHub Pages for a repository.

        Args:
            repository: GitHub repository to disable Pages for

        Returns:
            True if Pages was disabled successfully, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/repos/{repository.full_name}/pages",
                    headers=self.headers
                ) as response:
                    if response.status == 204:
                        logger.info(f"GitHub Pages disabled for {repository.full_name}")
                        return True
                    elif response.status == 404:
                        logger.info(f"GitHub Pages was not enabled for {repository.full_name}")
                        return True
                    else:
                        error_data = await response.text()
                        logger.warning(
                            f"Failed to disable Pages: {response.status} - {error_data}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Failed to disable Pages for {repository.full_name}: {e}")
            return False

    async def update_pages_config(
        self, 
        repository: GitHubRepository, 
        source_branch: str = None,
        source_path: str = "/",
        build_type: str = "workflow"
    ) -> bool:
        """
        Update GitHub Pages configuration.

        Args:
            repository: GitHub repository
            source_branch: Source branch for Pages (defaults to repository default branch)
            source_path: Source path for Pages
            build_type: Build type ("legacy" or "workflow")

        Returns:
            True if configuration was updated successfully, False otherwise
        """
        try:
            if source_branch is None:
                source_branch = repository.default_branch

            pages_config = {
                "source": {
                    "branch": source_branch,
                    "path": source_path
                },
                "build_type": build_type
            }

            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.base_url}/repos/{repository.full_name}/pages",
                    headers=self.headers,
                    json=pages_config
                ) as response:
                    if response.status == 200:
                        logger.info(f"Updated Pages config for {repository.full_name}")
                        return True
                    else:
                        error_data = await response.text()
                        logger.warning(
                            f"Failed to update Pages config: {response.status} - {error_data}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Failed to update Pages config for {repository.full_name}: {e}")
            return False
