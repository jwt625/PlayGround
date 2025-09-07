"""
Repository service for GitHub repository CRUD operations and user management.

This service handles:
- User authentication and information retrieval
- Token scope validation
- Repository creation and management
- Repository information retrieval
"""

import logging
from datetime import datetime
from typing import Any

import aiohttp

from models.github import (
    CreateRepositoryRequest,
    GitHubRepository,
    GitHubUser,
)

logger = logging.getLogger(__name__)


class RepositoryService:
    """Service for GitHub repository CRUD operations and user management."""

    def __init__(self, access_token: str):
        """
        Initialize repository service.

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

    async def get_authenticated_user(self) -> GitHubUser:
        """Get the authenticated user information."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/user",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get user info: {response.status}")

                data = await response.json()
                return GitHubUser(**data)

    async def get_token_scopes(self) -> list[str]:
        """Get the scopes for the current access token."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/user",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get user info: {response.status}")

                # GitHub returns scopes in the X-OAuth-Scopes header
                scopes_header = response.headers.get("X-OAuth-Scopes", "")
                scopes = [
                    scope.strip()
                    for scope in scopes_header.split(",")
                    if scope.strip()
                ]
                return scopes

    def validate_repository_name(self, name: str) -> str:
        """Convert paper title to valid GitHub repository name."""
        import re
        import time

        if not name:
            timestamp = int(time.time())
            return f"paper-{timestamp}"

        # Additional cleaning for paper titles
        clean_name = name

        # Remove markdown links: [text](url) -> text
        clean_name = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_name)

        # Remove arXiv references and dates
        clean_name = re.sub(r'arXiv:\d+\.\d+v?\d*\s*\[[^\]]+\]\s*\d+\s+\w+\s+\d+', '', clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'\[arXiv:[^\]]+\]', '', clean_name, flags=re.IGNORECASE)

        # Convert to repository name format
        clean_name = re.sub(r'[^a-zA-Z0-9\s-]', '', clean_name)
        clean_name = re.sub(r'\s+', '-', clean_name.strip()).lower()

        # Ensure starts with letter (GitHub requirement)
        if not clean_name or not clean_name[0].isalpha():
            clean_name = f"paper-{clean_name}" if clean_name else f"paper-{int(time.time())}"

        # Remove trailing hyphens and limit length
        clean_name = clean_name.rstrip('-')[:50]  # Leave room for timestamp

        # Add timestamp for uniqueness
        timestamp = int(time.time())
        final_name = f"{clean_name}-{timestamp}" if clean_name else f"paper-{timestamp}"

        return final_name[:100]  # GitHub limit

    async def check_repository_exists(self, repo_name: str) -> bool:
        """Check if a repository with the given name already exists."""
        try:
            # Get current user first
            user = await self.get_current_user()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/repos/{user.login}/{repo_name}",
                    headers=self.headers
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def generate_unique_repository_name(self, base_name: str) -> str:
        """Generate a unique repository name by checking for conflicts."""
        import time

        # First validate and clean the base name
        clean_base = self.validate_repository_name(base_name)
        logger.info(f"Cleaned repository name: '{base_name}' -> '{clean_base}'")

        # Try the cleaned base name
        if not await self.check_repository_exists(clean_base):
            return clean_base

        # If it exists, try with timestamp
        timestamp = int(time.time())
        timestamped_name = f"{clean_base}-{timestamp}"
        timestamped_name = self.validate_repository_name(timestamped_name)
        if not await self.check_repository_exists(timestamped_name):
            return timestamped_name

        # If that also exists, add a counter
        for i in range(1, 100):
            candidate_name = f"{clean_base}-{timestamp}-{i}"
            candidate_name = self.validate_repository_name(candidate_name)
            if not await self.check_repository_exists(candidate_name):
                return candidate_name

        # Fallback to UUID if all else fails
        import uuid
        fallback_name = f"paper-{timestamp}-{str(uuid.uuid4())[:8]}"
        return self.validate_repository_name(fallback_name)

    async def create_empty_repository(
        self, request: CreateRepositoryRequest
    ) -> GitHubRepository:
        """Create an empty repository with unique name checking."""
        # Ensure unique repository name
        unique_name = await self.generate_unique_repository_name(request.name)
        if unique_name != request.name:
            logger.info(f"Repository name '{request.name}' exists, using '{unique_name}'")

        repo_data = {
            "name": unique_name,
            "description": request.description or
                f"Academic paper website - {unique_name}",
            "private": False,
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False,
            "auto_init": True,  # Initialize with README
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/user/repos",
                headers=self.headers,
                json=repo_data
            ) as response:
                if response.status != 201:
                    error_data = await response.json()
                    raise Exception(f"Failed to create repository: {error_data}")

                repo_json = await response.json()
                logger.info(f"✅ Created repository: {repo_json['html_url']}")

                return GitHubRepository(
                    id=repo_json["id"],
                    name=repo_json["name"],
                    full_name=repo_json["full_name"],
                    owner=GitHubUser(
                        login=repo_json["owner"]["login"],
                        id=repo_json["owner"]["id"],
                        name=repo_json["owner"].get("name"),
                        email=repo_json["owner"].get("email"),
                        avatar_url=repo_json["owner"]["avatar_url"],
                        html_url=repo_json["owner"]["html_url"]
                    ),
                    private=repo_json["private"],
                    html_url=repo_json["html_url"],
                    clone_url=repo_json["clone_url"],
                    ssh_url=repo_json["ssh_url"],
                    default_branch=repo_json["default_branch"],
                    created_at=datetime.fromisoformat(
                        repo_json["created_at"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        repo_json["updated_at"].replace("Z", "+00:00")
                    )
                )

    async def fork_repository(
        self, template_owner: str, template_repo: str, new_name: str
    ) -> GitHubRepository:
        """Fork a repository with a new name."""
        fork_data = {"name": new_name}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.github.com/repos/{template_owner}/{template_repo}/forks",
                headers=self.headers,
                json=fork_data
            ) as response:
                if response.status != 202:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to fork repository: {response.status} - {error_text}"
                    )

                repo_info = await response.json()
                logger.info(f"Fork initiated: {repo_info['html_url']}")

                return GitHubRepository(
                    id=repo_info["id"],
                    name=repo_info["name"],
                    full_name=repo_info["full_name"],
                    owner=GitHubUser(
                        login=repo_info["owner"]["login"],
                        id=repo_info["owner"]["id"],
                        name=repo_info["owner"].get("name"),
                        email=repo_info["owner"].get("email"),
                        avatar_url=repo_info["owner"]["avatar_url"],
                        html_url=repo_info["owner"]["html_url"]
                    ),
                    private=repo_info["private"],
                    html_url=repo_info["html_url"],
                    clone_url=repo_info["clone_url"],
                    ssh_url=repo_info["ssh_url"],
                    default_branch=repo_info["default_branch"],
                    created_at=datetime.fromisoformat(
                        repo_info["created_at"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        repo_info["updated_at"].replace("Z", "+00:00")
                    )
                )

    async def wait_for_repository_ready(
        self, repo_full_name: str, max_wait_seconds: int = 60
    ) -> str:
        """Wait for repository to be ready and return default branch."""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait_seconds:
            try:
                async with aiohttp.ClientSession() as session:
                    # Check if repository is accessible
                    async with session.get(
                        f"{self.base_url}/repos/{repo_full_name}",
                        headers=self.headers
                    ) as response:
                        if response.status == 200:
                            repo_data = await response.json()
                            default_branch = str(
                                repo_data.get("default_branch", "main")
                            )

                            # Check if the default branch exists
                            async with session.get(
                                f"{self.base_url}/repos/{repo_full_name}/git/refs/heads/{default_branch}",
                                headers=self.headers
                            ) as branch_response:
                                if branch_response.status == 200:
                                    logger.info(
                                        f"Repository {repo_full_name} is ready with branch "
                                        f"'{default_branch}'"
                                    )
                                    return default_branch

            except Exception as e:
                logger.debug(f"Repository not ready yet: {e}")

            await asyncio.sleep(2)

        raise Exception(
            f"Repository {repo_full_name} not ready after {max_wait_seconds} seconds"
        )

    async def get_repository_info(self, repo_full_name: str) -> dict[str, Any]:
        """Get repository information."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/repos/{repo_full_name}",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get repository info: {response.status}")

                return await response.json()
