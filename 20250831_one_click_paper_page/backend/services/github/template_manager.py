"""
Template manager for GitHub template repository caching and content management.

This service handles:
- Template repository content caching with TTL
- Template file filtering for essential files
- Template content fetching from GitHub API
"""

import logging
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class TemplateCache:
    """Cache for GitHub template repository content to reduce API calls."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._ttl = ttl_seconds

    def is_cached(self, template_repo: str) -> bool:
        """Check if template is cached and not expired."""
        if template_repo not in self._cache:
            return False

        age = time.time() - self._cache_timestamps[template_repo]
        return age < self._ttl

    def get(self, template_repo: str) -> dict[str, Any] | None:
        """Get cached template data if available and not expired."""
        if self.is_cached(template_repo):
            return self._cache[template_repo]
        return None

    def set(self, template_repo: str, data: dict[str, Any]) -> None:
        """Cache template data with timestamp."""
        self._cache[template_repo] = data
        self._cache_timestamps[template_repo] = time.time()

    def clear_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp >= self._ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)


class TemplateManager:
    """Service for GitHub template repository management and caching."""

    def __init__(self, access_token: str):
        """
        Initialize template manager.

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
        
        # Class-level template cache shared across instances
        self._template_cache = TemplateCache()

    async def get_template_content_cached(
        self, template_repo_url: str
    ) -> dict[str, Any]:
        """Get template repository content with caching."""
        # Extract repo identifier from URL
        template_repo = template_repo_url.replace("https://github.com/", "")

        # Check cache first
        cached_data = self._template_cache.get(template_repo)
        if cached_data:
            logger.info(f"📦 Using cached template data for {template_repo}")
            return cached_data

        logger.info(f"🔍 Fetching template content from {template_repo}")

        try:
            async with aiohttp.ClientSession() as session:
                # Get repository tree (all files)
                async with session.get(
                    f"{self.base_url}/repos/{template_repo}/git/trees/master?recursive=1",
                    headers=self.headers
                ) as response:
                    if response.status == 404:
                        # Try 'main' branch if 'master' doesn't exist
                        async with session.get(
                            f"{self.base_url}/repos/{template_repo}/git/trees/main?recursive=1",
                            headers=self.headers
                        ) as main_response:
                            if main_response.status != 200:
                                raise Exception(f"Failed to get template tree: {main_response.status}")
                            tree_data = await main_response.json()
                            default_branch = "main"
                    elif response.status != 200:
                        raise Exception(f"Failed to get template tree: {response.status}")
                    else:
                        tree_data = await response.json()
                        default_branch = "master"

                # Log all files before filtering
                all_files_raw = tree_data["tree"]
                logger.info(f"📋 Raw template files ({len(all_files_raw)} total):")
                for i, f in enumerate(all_files_raw):  # Show ALL files
                    logger.info(f"  {i+1}. {f['path']} ({f['type']})")

                # Use ALL files from template, only skip README.md to avoid conflict
                all_files = [f for f in tree_data["tree"] if f["path"] != "README.md"]

                # NO FILTERING - Use all files including .github
                logger.info(
                    f"📋 Using ALL template files ({len(all_files)} total):"
                )
                for i, f in enumerate(all_files):  # Show ALL files
                    logger.info(f"  {i+1}. {f['path']} ({f['type']})")

                template_data = {
                    "tree": all_files,
                    "default_branch": default_branch,
                    "repo": template_repo
                }

                # Cache the result
                self._template_cache.set(template_repo, template_data)
                logger.info(f"✅ Cached template data for {template_repo}")

                return template_data

        except Exception as e:
            logger.error(f"❌ Failed to fetch template content: {e}")
            raise

    def filter_essential_template_files(
        self, tree_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter template files to essential ones only."""
        # Essential directories and files for academic templates
        essential_patterns = [
            "_config.yml",
            "_layouts/",
            "_includes/",
            "_sass/",
            "_data/",
            "assets/",
            "index.html",
            "index.md",
            "_pages/",
            "_publications/",
            "_posts/",
            "_portfolio/",
            "_talks/",
            "_teaching/",
            "Gemfile",
            "package.json",
            ".github/",  # Include .github directory (now with workflow scope)
        ]

        # Files to skip
        skip_patterns = [
            ".git",
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "talkmap.ipynb",
            "talkmap.py",
            "scripts/",
            "markdown_generator/",
            # Note: Now including .github files since we have workflow scope
        ]

        essential_files = []
        skipped_files = []

        for item in tree_items:
            path = item["path"]

            # Explicitly include ALL .github files first (now that we have workflow scope)
            if path.startswith(".github"):
                essential_files.append(item)
                continue

            # Skip files we don't need
            if any(skip in path for skip in skip_patterns):
                skipped_files.append(path)
                continue

            # Include essential files/directories
            if any(essential in path for essential in essential_patterns):
                essential_files.append(item)
            else:
                # Log files that don't match any pattern
                logger.debug(f"File doesn't match any pattern: {path}")

        logger.info(
            f"📋 Filtered {len(tree_items)} files to {len(essential_files)} "
            f"essential files, skipped {len(skipped_files)} files"
        )

        # Log ALL skipped files for debugging
        if skipped_files:
            logger.info(f"🚫 Skipped files ({len(skipped_files)} total):")
            for i, path in enumerate(skipped_files):
                logger.info(f"  {i+1}. {path}")

        return essential_files

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear_expired()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "cached_templates": list(self._template_cache._cache.keys()),
            "cache_size": len(self._template_cache._cache),
            "ttl_seconds": self._template_cache._ttl
        }
