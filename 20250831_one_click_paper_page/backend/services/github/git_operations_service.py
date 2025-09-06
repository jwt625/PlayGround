"""
Git operations service for low-level GitHub Git API operations.

This service handles:
- Git blob creation and retrieval
- Git tree creation and management
- Git commit creation
- Git reference management
- Bulk template content copying using Git API
"""

import logging
from typing import Any

import aiohttp

from models.github import GitHubRepository

logger = logging.getLogger(__name__)


class GitOperationsService:
    """Service for low-level GitHub Git API operations."""

    def __init__(self, access_token: str):
        """
        Initialize Git operations service.

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

    async def get_reference(
        self, repository: GitHubRepository, ref: str = None
    ) -> dict[str, Any]:
        """Get a Git reference (branch/tag)."""
        if ref is None:
            ref = f"heads/{repository.default_branch}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/repos/{repository.full_name}/git/refs/{ref}",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get reference {ref}: {response.status}")
                
                return await response.json()

    async def create_blob(
        self, repository: GitHubRepository, content: str, encoding: str = "base64"
    ) -> str:
        """Create a Git blob and return its SHA."""
        blob_data = {
            "content": content,
            "encoding": encoding
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/repos/{repository.full_name}/git/blobs",
                headers=self.headers,
                json=blob_data
            ) as response:
                if response.status != 201:
                    error_data = await response.json()
                    raise Exception(f"Failed to create blob: {error_data}")
                
                result = await response.json()
                return result["sha"]

    async def create_tree(
        self, repository: GitHubRepository, tree_items: list[dict[str, Any]]
    ) -> str:
        """Create a Git tree and return its SHA."""
        tree_data = {"tree": tree_items}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/repos/{repository.full_name}/git/trees",
                headers=self.headers,
                json=tree_data
            ) as response:
                if response.status != 201:
                    error_data = await response.json()
                    raise Exception(f"Failed to create tree: {error_data}")
                
                result = await response.json()
                return result["sha"]

    async def create_commit(
        self, 
        repository: GitHubRepository, 
        message: str, 
        tree_sha: str, 
        parent_shas: list[str]
    ) -> str:
        """Create a Git commit and return its SHA."""
        commit_data = {
            "message": message,
            "tree": tree_sha,
            "parents": parent_shas
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/repos/{repository.full_name}/git/commits",
                headers=self.headers,
                json=commit_data
            ) as response:
                if response.status != 201:
                    error_data = await response.json()
                    raise Exception(f"Failed to create commit: {error_data}")
                
                result = await response.json()
                return result["sha"]

    async def update_reference(
        self, repository: GitHubRepository, ref: str, sha: str
    ) -> None:
        """Update a Git reference to point to a new commit."""
        ref_data = {"sha": sha}
        
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                f"{self.base_url}/repos/{repository.full_name}/git/refs/{ref}",
                headers=self.headers,
                json=ref_data
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"Failed to update reference {ref}: {error_data}")

    async def get_blob_content(
        self, template_repo: str, blob_sha: str
    ) -> dict[str, Any]:
        """Get blob content from a repository."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/repos/{template_repo}/git/blobs/{blob_sha}",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get blob {blob_sha}: {response.status}")
                
                return await response.json()

    async def copy_template_content_bulk(
        self, repository: GitHubRepository, template_data: dict[str, Any]
    ) -> None:
        """Copy template content using Git API bulk operations."""
        logger.info(f"📁 Copying template content to {repository.full_name}")

        try:
            # Step 1: Get current repository state
            ref_data = await self.get_reference(repository)
            current_commit_sha = ref_data["object"]["sha"]

            # Step 2: Create blobs for template files in the new repository
            template_repo = template_data["repo"]
            blob_shas = {}

            # Include ALL blob files from template
            all_template_files = template_data['tree']
            blob_files = [f for f in all_template_files if f['type'] == 'blob']

            logger.info(f"Creating blobs for {len(blob_files)} files")

            # Debug: Log which .github files are being processed
            github_blob_files = [
                f['path'] for f in blob_files if f['path'].startswith('.github/')
            ]

            if github_blob_files:
                logger.info(f"📁 GitHub files found: {github_blob_files}")
            else:
                logger.warning("⚠️ No .github files found in template files!")

            for file_item in template_data["tree"]:
                if file_item["type"] == "blob":  # Only files, not directories
                    try:
                        # Fetch file content from template repository
                        blob_data = await self.get_blob_content(template_repo, file_item['sha'])

                        # Create new blob in target repository
                        new_blob_sha = await self.create_blob(
                            repository, 
                            blob_data["content"], 
                            blob_data["encoding"]
                        )

                        blob_shas[file_item["path"]] = {
                            "sha": new_blob_sha,
                            "mode": file_item["mode"]
                        }

                    except Exception as e:
                        logger.warning(f"Failed to process blob for {file_item['path']}: {e}")
                        continue

            logger.info(
                f"✅ Successfully created {len(blob_shas)} blobs out of "
                f"{len(blob_files)} files"
            )

            # Step 3: Create new tree with ONLY blob files
            tree_items = []
            for file_path, blob_info in blob_shas.items():
                tree_items.append({
                    "path": file_path,
                    "mode": blob_info["mode"],
                    "type": "blob",
                    "sha": blob_info["sha"]
                })

            logger.info(f"🌳 Creating tree with {len(tree_items)} items")

            # Create tree
            new_tree_sha = await self.create_tree(repository, tree_items)

            # Step 4: Create commit
            commit_message = f"Initialize repository with {template_data['repo']} template"
            new_commit_sha = await self.create_commit(
                repository, 
                commit_message, 
                new_tree_sha, 
                [current_commit_sha]
            )

            # Step 5: Update branch reference
            await self.update_reference(
                repository, 
                f"heads/{repository.default_branch}", 
                new_commit_sha
            )

            logger.info(
                f"✅ Template content copied successfully ({len(tree_items)} files)"
            )

        except Exception as e:
            logger.error(f"❌ Failed to copy template content: {e}")
            raise
