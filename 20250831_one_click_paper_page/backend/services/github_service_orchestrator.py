"""
GitHub service orchestrator - lightweight coordinator for modular GitHub services.

This orchestrator delegates to specialized services while maintaining the same public API
as the original monolithic GitHubService for zero breaking changes.
"""

import logging
from pathlib import Path
from typing import Any

import aiohttp

from models.github import (
    CreateRepositoryRequest,
    CreateRepositoryResponse,
    DeploymentConfig,
    DeploymentStatusResponse,
    DualDeploymentResult,
    GitHubRepository,
    GitHubUser,
    TemplateInfo,
)
from services.github import (
    DeploymentTracker,
    GitHubPagesService,
    GitOperationsService,
    RepositoryService,
    TemplateManager,
    WorkflowService,
)

logger = logging.getLogger(__name__)


class GitHubServiceOrchestrator:
    """
    Orchestrator for GitHub operations using modular services.

    This class maintains the same public API as the original GitHubService
    but delegates to specialized services for implementation.
    """

    def __init__(self, access_token: str):
        """
        Initialize GitHub service orchestrator.

        Args:
            access_token: GitHub OAuth access token
        """
        self.access_token = access_token

        # Initialize all modular services
        self.repository_service = RepositoryService(access_token)
        self.template_manager = TemplateManager(access_token)
        self.git_operations_service = GitOperationsService(access_token)
        self.workflow_service = WorkflowService(access_token)
        self.pages_service = GitHubPagesService(access_token)
        self.deployment_tracker = DeploymentTracker(access_token)

        # Maintain compatibility with existing headers property
        self.headers = self.repository_service.headers
        self.base_url = self.repository_service.base_url

        # Import template service for compatibility
        from services.template_service import template_service
        self.template_service = template_service

    # ============================================================================
    # PUBLIC API METHODS - Maintain exact same signatures as original GitHubService
    # ============================================================================

    async def get_authenticated_user(self) -> GitHubUser:
        """Get the authenticated user information."""
        return await self.repository_service.get_authenticated_user()

    async def get_token_scopes(self) -> list[str]:
        """Get the scopes for the current access token."""
        return await self.repository_service.get_token_scopes()

    async def list_templates(self) -> list[TemplateInfo]:
        """List available templates."""
        return self.template_service.get_all_templates()

    async def create_repository(
        self, request: CreateRepositoryRequest
    ) -> CreateRepositoryResponse:
        """Create repository using optimized Git API approach."""
        return await self.create_repository_optimized(request)

    async def create_repository_optimized(
        self, request: CreateRepositoryRequest
    ) -> CreateRepositoryResponse:
        """
        Create repository using optimized Git API approach (no forking).

        This method:
        1. Creates a fresh repository (no fork security issues)
        2. Copies template content using Git API (bulk operations)
        3. Adds custom deployment workflow
        4. Enables GitHub Pages automatically

        Args:
            request: Repository creation request

        Returns:
            Repository creation response
        """
        logger.info(f"🚀 Creating optimized repository: {request.name}")

        try:
            # Step 1: Get template repository info and content (cached)
            template_repo = self.template_service.get_template_repository(
                request.template
            )
            template_url = f"https://github.com/{template_repo.full_name}"
            template_data = await self.template_manager.get_template_content_cached(template_url)

            # Step 2: Create empty repository
            repository = await self.repository_service.create_empty_repository(request)

            # Step 3: Copy template content using Git API
            await self.git_operations_service.copy_template_content_bulk(repository, template_data)

            # Step 4: Add deployment workflow (if not already present in template)
            await self.workflow_service.add_deployment_workflow_if_needed(repository, template_data)

            # Step 5: Enable GitHub Pages
            await self.pages_service.enable_github_pages_with_actions(repository)

            # Step 6: Create deployment job for tracking
            deployment_id = self.deployment_tracker.create_deployment_job(repository, request)

            logger.info(f"✅ Optimized repository created: {repository.full_name}")

            return CreateRepositoryResponse(
                repository=repository,
                deployment_id=deployment_id,
                status=self.deployment_tracker.get_deployment_job(deployment_id).status,
                message="Repository created with optimized Git API approach.",
            )

        except Exception as e:
            logger.error(f"❌ Failed to create optimized repository: {e}")
            raise

    async def create_repository_from_template(
        self, request: CreateRepositoryRequest
    ) -> CreateRepositoryResponse:
        """
        Create a new repository by forking from the actual template repository.

        Args:
            request: Repository creation request

        Returns:
            Repository creation response with deployment tracking
        """
        try:
            # Get template repository configuration
            template_repo = self.template_service.get_template_repository(
                request.template
            )

            # Step 1: Fork the template repository
            logger.info(f"Forking template repository {template_repo.full_name}")
            repository = await self.repository_service.fork_repository(
                template_repo.owner, template_repo.name, request.name
            )

            # Step 2: Wait for repository to be ready
            await self.repository_service.wait_for_repository_ready(repository.full_name)

            # Step 3: Enable GitHub Pages
            await self.pages_service.enable_github_pages_with_actions(repository)

            # Step 4: Create deployment job for tracking
            deployment_id = self.deployment_tracker.create_deployment_job(repository, request)

            logger.info(
                f"Forked repository {repository.full_name} from template "
                f"{template_repo.full_name} with deployment {deployment_id}"
            )

            return CreateRepositoryResponse(
                repository=repository,
                deployment_id=deployment_id,
                status=self.deployment_tracker.get_deployment_job(deployment_id).status,
                message=(
                    "Repository created with GitHub Actions workflows. "
                    "Ready for content upload."
                ),
            )

        except Exception as e:
            logger.error(f"Failed to create repository from template: {e}")
            raise

    async def deploy_converted_content(
        self, deployment_id: str, converted_content_dir: Path, config: DeploymentConfig
    ) -> None:
        """
        Deploy converted content to GitHub repository via GitHub Actions.

        Args:
            deployment_id: Deployment job ID
            converted_content_dir: Directory containing converted content
            config: Deployment configuration
        """
        try:
            deployment = self.deployment_tracker.get_deployment_job(deployment_id)
            if not deployment:
                raise Exception(f"Deployment {deployment_id} not found")

            # Update status
            from models.github import DeploymentStatus
            self.deployment_tracker.update_deployment_status(
                deployment_id,
                DeploymentStatus.IN_PROGRESS,
                message="Starting automated deployment via GitHub Actions..."
            )

            # Prepare converted content files for upload
            files_to_commit = await self._prepare_converted_content_files(
                converted_content_dir, config
            )

            if files_to_commit:
                # Commit the converted content to the repository
                await self._commit_converted_content(
                    deployment.repository, files_to_commit, config
                )

                self.deployment_tracker.update_deployment_status(
                    deployment_id,
                    DeploymentStatus.SUCCESS,
                    message="Converted content uploaded successfully! GitHub Actions will build and deploy the site."
                )
            else:
                self.deployment_tracker.update_deployment_status(
                    deployment_id,
                    DeploymentStatus.SUCCESS,
                    message="No content files found to deploy."
                )

            logger.info(f"Deployment {deployment_id} content uploaded successfully")

        except Exception as e:
            from models.github import DeploymentStatus
            self.deployment_tracker.update_deployment_status(
                deployment_id,
                DeploymentStatus.FAILURE,
                error_message=str(e)
            )
            logger.error(f"Deployment {deployment_id} failed: {e}")
            raise

    async def get_deployment_status(
        self, deployment_id: str
    ) -> DeploymentStatusResponse:
        """
        Get deployment status with GitHub Actions workflow monitoring.

        Args:
            deployment_id: Deployment job ID

        Returns:
            Deployment status response
        """
        return await self.deployment_tracker.get_deployment_status(deployment_id)

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
        return await self.pages_service.enable_github_pages_as_backup(repository)

    async def create_dual_deployment(
        self, request: CreateRepositoryRequest
    ) -> DualDeploymentResult:
        """
        Create dual deployment (optimized + backup).

        Args:
            request: Repository creation request

        Returns:
            Dual deployment result
        """
        try:
            # Create optimized repository
            primary_result = await self.create_repository_optimized(request)

            # Enable backup Pages
            await self.enable_github_pages_as_backup(primary_result.repository)

            return DualDeploymentResult(
                standalone_repo=primary_result.repository,
                standalone_url=primary_result.repository.html_url,
                deployment_id=primary_result.deployment_id,
                status=primary_result.status,
                message="Dual deployment created successfully"
            )

        except Exception as e:
            logger.error(f"❌ Failed to create dual deployment: {e}")
            raise

    async def _prepare_converted_content_files(
        self, converted_content_dir: Path, config: DeploymentConfig
    ) -> list[dict[str, Any]]:
        """Prepare converted content files for upload to repository."""
        import base64
        import json

        files_to_commit = []

        if not converted_content_dir.exists():
            logger.warning(f"Converted content directory does not exist: {converted_content_dir}")
            return files_to_commit

        logger.info(f"📁 Preparing files from {converted_content_dir}")

        # Process all files in the converted content directory
        for file_path in converted_content_dir.rglob("*"):
            if file_path.is_file():
                try:
                    # Determine relative path from content directory
                    rel_path = file_path.relative_to(converted_content_dir)

                    # Skip hidden files and system files
                    if any(part.startswith('.') for part in rel_path.parts):
                        continue

                    logger.info(f"📄 Processing file: {rel_path}")

                    # Read file content based on type
                    if file_path.suffix.lower() in [
                        ".md", ".html", ".txt", ".yml", ".yaml", ".json", ".css", ".js"
                    ]:
                        # Text files - read as UTF-8
                        with open(file_path, encoding='utf-8') as f:
                            content = f.read()

                        # Customize content with paper metadata if it's an HTML file
                        if file_path.suffix.lower() == ".html":
                            content = self._customize_html_content(content, config)

                        files_to_commit.append({
                            "path": str(rel_path),
                            "content": content,
                            "encoding": "utf-8"
                        })
                    else:
                        # Binary files (images, PDFs, etc.) - encode as base64
                        with open(file_path, 'rb') as f:
                            binary_content = f.read()

                        encoded_content = base64.b64encode(binary_content).decode("utf-8")
                        files_to_commit.append({
                            "path": str(rel_path),
                            "content": encoded_content,
                            "encoding": "base64"
                        })

                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue

        # Add paper configuration file
        config_content = {
            "paper_title": config.paper_title,
            "paper_authors": config.paper_authors,
            "paper_date": config.paper_date,
            "template": config.template.value,
            "repository_name": config.repository_name,
        }

        files_to_commit.append({
            "path": "paper-config.json",
            "content": json.dumps(config_content, indent=2),
            "encoding": "utf-8"
        })

        logger.info(f"✅ Prepared {len(files_to_commit)} files for upload")
        return files_to_commit

    async def _commit_converted_content(
        self, repository: GitHubRepository, files_to_commit: list[dict[str, Any]], config: DeploymentConfig
    ) -> None:
        """Commit converted content files to the repository using Git API."""
        if not files_to_commit:
            logger.info("No files to commit")
            return

        logger.info(f"🚀 Committing {len(files_to_commit)} files to {repository.full_name}")

        try:
            # Step 1: Get current repository state
            ref_data = await self.git_operations_service.get_reference(repository)
            current_commit_sha = ref_data["object"]["sha"]

            # Step 2: Get existing repository tree to preserve existing files (same as workflow service)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.git_operations_service.base_url}/repos/{repository.full_name}/git/trees/{current_commit_sha}?recursive=1",
                    headers=self.git_operations_service.headers
                ) as tree_response:
                    if tree_response.status != 200:
                        logger.warning(f"Failed to get current tree: {tree_response.status}")
                        existing_tree_items = []
                    else:
                        current_tree_data = await tree_response.json()
                        existing_tree_items = current_tree_data["tree"]

            logger.info(f"📋 Found {len(existing_tree_items)} existing files in repository")

            # Log existing .github files specifically
            github_files = [item["path"] for item in existing_tree_items if item["path"].startswith(".github/")]
            if github_files:
                logger.info(f"🔧 Found existing .github files: {github_files}")
            else:
                logger.warning("⚠️ No .github files found in existing repository")

            # Step 3: Create blobs for new files
            blob_shas = {}
            for file_info in files_to_commit:
                file_path = file_info["path"]
                content = file_info["content"]
                encoding = file_info["encoding"]

                # Convert encoding format for GitHub API
                api_encoding = "base64" if encoding == "base64" else "utf-8"

                # For UTF-8 files, we need to encode as base64 for the API
                if encoding == "utf-8":
                    import base64
                    content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                    api_encoding = "base64"

                blob_sha = await self.git_operations_service.create_blob(
                    repository, content, api_encoding
                )
                blob_shas[file_path] = blob_sha
                logger.info(f"📄 Created blob for {file_path}")

            # Step 4: Merge existing files with new files (same as workflow service)
            new_file_paths = set(blob_shas.keys())

            # Add existing files that are not being overwritten
            tree_items = []
            for existing_item in existing_tree_items:
                if existing_item["path"] not in new_file_paths and existing_item["type"] == "blob":
                    tree_items.append({
                        "path": existing_item["path"],
                        "mode": existing_item["mode"],
                        "type": existing_item["type"],
                        "sha": existing_item["sha"]
                    })

            # Add new files
            for file_path, blob_sha in blob_shas.items():
                tree_items.append({
                    "path": file_path,
                    "mode": "100644",  # Regular file mode
                    "type": "blob",
                    "sha": blob_sha
                })

            new_tree_sha = await self.git_operations_service.create_tree(repository, tree_items)
            logger.info(f"🌳 Created tree with {len(tree_items)} files ({len(existing_tree_items)} existing + {len(blob_shas)} new)")

            # Step 5: Create commit
            commit_message = f"Add converted paper content: {config.paper_title or 'Untitled'}"
            new_commit_sha = await self.git_operations_service.create_commit(
                repository, commit_message, new_tree_sha, [current_commit_sha]
            )
            logger.info(f"📝 Created commit: {commit_message}")

            # Step 6: Update branch reference
            await self.git_operations_service.update_reference(
                repository, f"heads/{repository.default_branch}", new_commit_sha
            )
            logger.info(f"✅ Updated {repository.default_branch} branch")

            logger.info(f"🎉 Successfully committed converted content to {repository.full_name}")

        except Exception as e:
            logger.error(f"❌ Failed to commit converted content: {e}")
            raise

    def _customize_html_content(self, content: str, config: DeploymentConfig) -> str:
        """Customize HTML content with paper metadata."""
        # Replace title
        if config.paper_title:
            content = content.replace(
                "<title>Document</title>",
                f"<title>{config.paper_title}</title>"
            )
            content = content.replace(
                "<title>Document Conversion</title>",
                f"<title>{config.paper_title}</title>"
            )

        # Replace author meta tag
        if config.paper_authors:
            authors_str = ", ".join(config.paper_authors)
            content = content.replace(
                '<meta name="author" content="">',
                f'<meta name="author" content="{authors_str}">'
            )
            # Also add author meta tag if it doesn't exist
            if '<meta name="author"' not in content and '<head>' in content:
                content = content.replace(
                    '<head>',
                    f'<head>\n    <meta name="author" content="{authors_str}">'
                )

        return content
