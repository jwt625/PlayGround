"""
GitHub service for automated repository creation and deployment via GitHub Actions.
"""

import asyncio
import base64
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp

from models.github import (
    CommitRequest,
    CommitResponse,
    CreateRepositoryRequest,
    CreateRepositoryResponse,
    DeploymentConfig,
    DeploymentJob,
    DeploymentStatus,
    DeploymentStatusResponse,
    FileContent,
    GitHubRepository,
    GitHubUser,
    TemplateInfo,
    WorkflowRun,
)

logger = logging.getLogger(__name__)


class GitHubService:
    """Service for automated GitHub repository creation and deployment via GitHub
    Actions."""

    def __init__(self, access_token: str):
        """
        Initialize GitHub service.

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

        # In-memory deployment tracking (use Redis in production)
        self._deployments: dict[str, DeploymentJob] = {}

        # Import template service
        from services.template_service import template_service
        self.template_service = template_service

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

    async def create_repository_from_template(
        self,
        request: CreateRepositoryRequest
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
            template_repo = self.template_service.get_template_repository(request.template)

            # Step 1: Fork the template repository
            logger.info(f"Forking template repository {template_repo.full_name}")

            fork_data = {
                "name": request.name,
                "default_branch_only": True,  # Only fork the default branch
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/repos/{template_repo.repository_owner}/"
                    f"{template_repo.repository_name}/forks",
                    headers=self.headers,
                    json=fork_data
                ) as response:
                    if response.status != 202:  # GitHub returns 202 for fork creation
                        error_data = await response.text()
                        logger.error(f"Fork creation failed: {response.status} - {error_data}")
                        # Fallback to regular repository creation
                        return await self._create_regular_repository_with_workflows(
                            request
                        )

                    repo_json = await response.json()

            # Convert to our model
            repository = GitHubRepository(
                id=repo_json["id"],
                name=repo_json["name"],
                full_name=repo_json["full_name"],
                description=repo_json.get("description"),
                html_url=repo_json["html_url"],
                clone_url=repo_json["clone_url"],
                ssh_url=repo_json["ssh_url"],
                default_branch=repo_json["default_branch"],
                private=repo_json["private"],
                owner=GitHubUser(
                    id=repo_json["owner"]["id"],
                    login=repo_json["owner"]["login"],
                    name=repo_json["owner"].get("name"),
                    email=repo_json["owner"].get("email"),
                    avatar_url=repo_json["owner"]["avatar_url"],
                    html_url=repo_json["owner"]["html_url"],
                ),
                created_at=datetime.fromisoformat(
                    repo_json["created_at"].replace("Z", "+00:00")
                ),
                updated_at=datetime.fromisoformat(
                    repo_json["updated_at"].replace("Z", "+00:00")
                ),
            )

            # Step 2: Wait for fork to be ready, then enable GitHub Pages
            await asyncio.sleep(3)  # Give GitHub time to set up the fork
            await self._enable_github_pages(repository)

            # Create deployment job for tracking
            deployment_id = str(uuid.uuid4())
            deployment_job = DeploymentJob(
                id=deployment_id,
                repository=repository,
                conversion_job_id=request.conversion_job_id,
                status=DeploymentStatus.PENDING,
                template=request.template,
                created_at=datetime.now(),
            )

            self._deployments[deployment_id] = deployment_job

            logger.info(
                f"Forked repository {repository.full_name} from template {template_repo.full_name} "
                f"with deployment {deployment_id}"
            )

            return CreateRepositoryResponse(
                repository=repository,
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                message=(
                    "Repository created with GitHub Actions workflows. "
                    "Ready for content upload."
                ),
            )

        except Exception as e:
            logger.error(f"Failed to create repository from template: {e}")
            raise

    # Keep the old method name for backward compatibility
    async def create_repository(
        self, request: CreateRepositoryRequest
    ) -> CreateRepositoryResponse:
        """Create repository (delegates to template-based creation)."""
        return await self.create_repository_from_template(request)

    async def deploy_converted_content(
        self,
        deployment_id: str,
        converted_content_dir: Path,
        config: DeploymentConfig,
    ) -> None:
        """
        Deploy converted content by uploading source files and triggering GitHub
        Actions.

        Args:
            deployment_id: Deployment job ID
            converted_content_dir: Directory containing converted content
            config: Deployment configuration
        """
        try:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                raise Exception(f"Deployment {deployment_id} not found")

            # Update status
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.build_logs.append(
                "Starting automated deployment via GitHub Actions..."
            )

            # Prepare source files for GitHub Actions to process
            files = await self._prepare_source_files(
                converted_content_dir, config
            )

            # Commit source files to trigger GitHub Actions workflow
            commit_request = CommitRequest(
                repository_name=deployment.repository.name,
                message="Add converted paper content - trigger automated deployment",
                files=files,
            )

            commit_response = await self._commit_files(
                deployment.repository, commit_request
            )

            # Wait for GitHub Actions workflow to start
            await asyncio.sleep(2)  # Give GitHub a moment to detect the push

            # Find and track the workflow run
            workflow_run = await self._get_latest_workflow_run(deployment.repository)
            if workflow_run:
                deployment.workflow_run = workflow_run
                deployment.build_logs.append(
                    f"GitHub Actions workflow started: {workflow_run.html_url}"
                )

            # Set status to in progress - GitHub Actions will handle the actual
            # deployment
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.pages_url = (
                f"https://{deployment.repository.owner.login}.github.io/"
                f"{deployment.repository.name}"
            )
            deployment.build_logs.append(
                "GitHub Actions workflow triggered. Deployment in progress..."
            )

            logger.info(f"Deployment {deployment_id} triggered via GitHub Actions")

        except Exception as e:
            deployment = self._deployments.get(deployment_id)
            if deployment:
                deployment.status = DeploymentStatus.FAILURE
                deployment.error_message = str(e)
                deployment.build_logs.append(f"Deployment failed: {e}")

            logger.error(f"Deployment {deployment_id} failed: {e}")
            raise

    async def _create_regular_repository_with_workflows(
        self, request: CreateRepositoryRequest
    ) -> CreateRepositoryResponse:
        """
        Fallback method to create repository and add GitHub Actions workflows manually.
        """
        # Create regular repository
        repo_data = {
            "name": request.name,
            "description": (
                request.description or f"Academic paper website - {request.name}"
            ),
            "private": False,
            "auto_init": True,
            "gitignore_template": "Jekyll",
            "license_template": "mit",
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

        repository = GitHubRepository(
            id=repo_json["id"],
            name=repo_json["name"],
            full_name=repo_json["full_name"],
            description=repo_json.get("description"),
            html_url=repo_json["html_url"],
            clone_url=repo_json["clone_url"],
            ssh_url=repo_json["ssh_url"],
            default_branch=repo_json["default_branch"],
            private=repo_json["private"],
            owner=GitHubUser(
                id=repo_json["owner"]["id"],
                login=repo_json["owner"]["login"],
                name=repo_json["owner"].get("name"),
                email=repo_json["owner"].get("email"),
                avatar_url=repo_json["owner"]["avatar_url"],
                html_url=repo_json["owner"]["html_url"],
            ),
            created_at=datetime.fromisoformat(
                repo_json["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                repo_json["updated_at"].replace("Z", "+00:00")
            ),
        )

        # Template repositories come with their own workflows - no need to add them manually

        # Note: GitHub Pages will be enabled by GitHub Actions workflow
        # or manually as a backup option if deployment fails

        # Create deployment job
        deployment_id = str(uuid.uuid4())
        deployment_job = DeploymentJob(
            id=deployment_id,
            repository=repository,
            conversion_job_id=request.conversion_job_id,
            status=DeploymentStatus.PENDING,
            template=request.template,
            created_at=datetime.now(),
        )

        self._deployments[deployment_id] = deployment_job

        return CreateRepositoryResponse(
            repository=repository,
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            message="Repository created with GitHub Actions workflows.",
        )

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
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            raise Exception(f"Deployment {deployment_id} not found")

        # Update deployment status from GitHub Actions workflow
        await self._update_deployment_from_workflow(deployment)

        # Calculate progress percentage based on workflow status
        progress = 0
        if deployment.status == DeploymentStatus.PENDING:
            progress = 10
        elif deployment.status == DeploymentStatus.QUEUED:
            progress = 20
        elif deployment.status == DeploymentStatus.IN_PROGRESS:
            if deployment.workflow_run:
                if deployment.workflow_run.status == "queued":
                    progress = 30
                elif deployment.workflow_run.status == "in_progress":
                    progress = 70
            else:
                progress = 40
        elif deployment.status == DeploymentStatus.SUCCESS:
            progress = 100
        elif deployment.status == DeploymentStatus.FAILURE:
            progress = 0

        # Generate status message based on GitHub Actions workflow
        message = "Deployment pending"
        if deployment.status == DeploymentStatus.IN_PROGRESS:
            if deployment.workflow_run:
                if deployment.workflow_run.status == "queued":
                    message = "GitHub Actions workflow queued..."
                elif deployment.workflow_run.status == "in_progress":
                    message = "GitHub Actions building and deploying..."
            else:
                message = "Waiting for GitHub Actions workflow to start..."
        elif deployment.status == DeploymentStatus.SUCCESS:
            message = "GitHub Actions deployment completed successfully!"
        elif deployment.status == DeploymentStatus.FAILURE:
            message = f"GitHub Actions deployment failed: {deployment.error_message}"

        return DeploymentStatusResponse(
            deployment_id=deployment_id,
            status=deployment.status,
            repository=deployment.repository,
            pages_url=deployment.pages_url,
            workflow_run=deployment.workflow_run,
            progress_percentage=progress,
            message=message,
            error_message=deployment.error_message,
        )

    async def _enable_github_pages(self, repository: GitHubRepository) -> None:
        """Enable GitHub Pages for the repository."""
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
                        logger.info(f"GitHub Pages already enabled for {repository.full_name}")
                    else:
                        error_data = await response.text()
                        logger.warning(f"Failed to enable GitHub Pages: {response.status} - {error_data}")

        except Exception as e:
            logger.warning(f"Failed to enable GitHub Pages for {repository.full_name}: {e}")
            # Don't fail the deployment if Pages setup fails

    async def _enable_github_pages_with_actions(
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
                        error_data = await response.json()
                        logger.warning(f"Failed to enable GitHub Pages: {error_data}")

        except Exception as e:
            logger.error(
                f"Failed to enable GitHub Pages for {repository.full_name}: {e}"
            )
            # Don't raise - Pages can be enabled manually

    async def list_templates(self) -> list[TemplateInfo]:
        """List available templates."""
        return self.template_service.get_all_templates()

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
            await self._enable_github_pages_with_actions(repository)
            return True
        except Exception as e:
            logger.error(f"Failed to enable GitHub Pages backup for {repository.full_name}: {e}")
            return False

    # Removed _get_workflow_template_files - templates now come with their own workflows

    async def _prepare_source_files(
        self, converted_content_dir: Path, config: DeploymentConfig
    ) -> list[FileContent]:
        """Prepare source files for GitHub Actions to process."""
        files = []

        # Add the converted content files
        if converted_content_dir.exists():
            for file_path in converted_content_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        # Determine relative path from content directory
                        rel_path = file_path.relative_to(converted_content_dir)

                        # Read file content
                        if file_path.suffix.lower() in [
                            ".md",
                            ".html",
                            ".txt",
                            ".yml",
                            ".yaml",
                            ".json",
                        ]:
                            # Text files
                            with open(file_path, encoding='utf-8') as f:
                                content = f.read()

                            files.append(FileContent(
                                path=str(rel_path),
                                content=content
                            ))
                        else:
                            # Binary files (images, etc.)
                            with open(file_path, 'rb') as f:
                                binary_content = f.read()

                            # Encode as base64 for GitHub API
                            encoded_content = base64.b64encode(binary_content).decode(
                                "utf-8"
                            )

                            files.append(FileContent(
                                path=str(rel_path),
                                content=encoded_content,
                                encoding="base64"
                            ))
                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}")

        # Add configuration file for the deployment
        config_content = {
            "paper_title": config.paper_title,
            "paper_authors": config.paper_authors,
            "paper_date": config.paper_date,
            "template": config.template.value,
            "repository_name": config.repository_name,
        }

        files.append(FileContent(
            path="paper-config.json",
            content=json.dumps(config_content, indent=2)
        ))

        return files

    async def _prepare_deployment_files(
        self,
        content_dir: Path,
        template: TemplateInfo,
        config: DeploymentConfig,
    ) -> list[FileContent]:
        """Prepare files for deployment."""
        files = []

        # Add converted content
        if (content_dir / "index.html").exists():
            with open(content_dir / "index.html", encoding="utf-8") as f:
                content = f.read()
                # Customize content with paper metadata
                content = self._customize_content(content, config)
                files.append(FileContent(path="index.html", content=content))

        # Add images
        images_dir = content_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*"):
                if img_file.is_file():
                    with open(img_file, "rb") as f:
                        img_content = base64.b64encode(f.read()).decode("utf-8")
                        files.append(FileContent(
                            path=f"images/{img_file.name}",
                            content=img_content,
                            encoding="base64"
                        ))

        # Template files are already in the cloned repository - no need to add them manually

        return files

    def _customize_content(self, content: str, config: DeploymentConfig) -> str:
        """Customize content with paper metadata."""
        # Simple template replacement
        if config.paper_title:
            content = content.replace(
                "<title>Document</title>", f"<title>{config.paper_title}</title>"
            )

        if config.paper_authors:
            authors_str = ", ".join(config.paper_authors)
            content = content.replace(
                '<meta name="author" content="">',
                f'<meta name="author" content="{authors_str}">'
            )

        return content

    # Removed _get_template_files - templates now come with their own configuration

    def _customize_config(self, content: str, config: DeploymentConfig) -> str:
        """Customize _config.yml with paper metadata."""
        if config.paper_title:
            content = content.replace(
                'title: "Academic Paper"', f'title: "{config.paper_title}"'
            )

        if config.paper_authors:
            authors_str = ', '.join(config.paper_authors)
            content = content.replace(
                'author: "Author Name"', f'author: "{authors_str}"'
            )

        content = content.replace(
            'description: "Academic paper website generated by one-click-paper-page"',
            f'description: "{config.repository_name}"',
        )

        return content

    def _customize_readme(self, content: str, config: DeploymentConfig) -> str:
        """Customize README.md with paper metadata."""
        if config.paper_title:
            content = content.replace(
                "# Academic Paper Website", f"# {config.paper_title}"
            )

        return content

    async def _commit_files(
        self, repository: GitHubRepository, request: CommitRequest
    ) -> CommitResponse:
        """Commit files to repository."""
        try:
            owner = repository.owner.login
            repo = repository.name

            # Get current commit SHA
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/repos/{owner}/{repo}/git/refs/heads/{request.branch}",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to get branch ref: {response.status}")

                    ref_data = await response.json()
                    current_sha = ref_data["object"]["sha"]

                # Get current tree
                async with session.get(
                    f"{self.base_url}/repos/{owner}/{repo}/git/commits/{current_sha}",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to get commit: {response.status}")

                    commit_data = await response.json()
                    tree_sha = commit_data["tree"]["sha"]

                # Create blobs for each file
                blob_shas = {}
                for file in request.files:
                    blob_data = {
                        "content": file.content,
                        "encoding": file.encoding,
                    }

                    async with session.post(
                        f"{self.base_url}/repos/{owner}/{repo}/git/blobs",
                        headers=self.headers,
                        json=blob_data
                    ) as response:
                        if response.status != 201:
                            raise Exception(f"Failed to create blob for {file.path}")

                        blob_response = await response.json()
                        blob_shas[file.path] = blob_response["sha"]

                # Create new tree
                tree_items = []
                for file in request.files:
                    tree_items.append({
                        "path": file.path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_shas[file.path],
                    })

                tree_data = {
                    "base_tree": tree_sha,
                    "tree": tree_items,
                }

                async with session.post(
                    f"{self.base_url}/repos/{owner}/{repo}/git/trees",
                    headers=self.headers,
                    json=tree_data
                ) as response:
                    if response.status != 201:
                        raise Exception("Failed to create tree")

                    tree_response = await response.json()
                    new_tree_sha = tree_response["sha"]

                # Create commit
                commit_data = {
                    "message": request.message,
                    "tree": new_tree_sha,
                    "parents": [current_sha],
                }

                async with session.post(
                    f"{self.base_url}/repos/{owner}/{repo}/git/commits",
                    headers=self.headers,
                    json=commit_data
                ) as response:
                    if response.status != 201:
                        raise Exception("Failed to create commit")

                    commit_response = await response.json()
                    new_commit_sha = commit_response["sha"]

                # Update reference
                ref_data = {"sha": new_commit_sha}
                async with session.patch(
                    f"{self.base_url}/repos/{owner}/{repo}/git/refs/heads/{request.branch}",
                    headers=self.headers,
                    json=ref_data
                ) as response:
                    if response.status != 200:
                        raise Exception("Failed to update reference")

                return CommitResponse(
                    sha=new_commit_sha,
                    url=commit_response["url"],
                    html_url=commit_response["html_url"],
                    message=request.message,
                )

        except Exception as e:
            logger.error(f"Failed to commit files: {e}")
            raise

    async def _enable_github_pages(self, repository: GitHubRepository) -> None:
        """Enable GitHub Pages for the repository."""
        try:
            owner = repository.owner.login
            repo = repository.name

            pages_data = {
                "source": {
                    "branch": "main",
                    "path": "/",
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/repos/{owner}/{repo}/pages",
                    headers=self.headers,
                    json=pages_data
                ) as response:
                    if response.status == 201:
                        logger.info(f"GitHub Pages enabled for {repository.full_name}")
                    elif response.status == 409:
                        # Pages already enabled
                        logger.info(
                            f"GitHub Pages already enabled for {repository.full_name}"
                        )
                    else:
                        error_data = await response.json()
                        logger.warning(f"Failed to enable GitHub Pages: {error_data}")

        except Exception as e:
            logger.warning(f"Failed to enable GitHub Pages: {e}")
            # Don't fail the deployment if Pages setup fails

    async def _get_latest_workflow_run(
        self, repository: GitHubRepository
    ) -> WorkflowRun | None:
        """Get the latest workflow run for the repository."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/repos/{repository.full_name}/actions/runs",
                    headers=self.headers,
                    params={
                        "per_page": 1,
                        "status": "queued,in_progress,completed",
                    },
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        runs = data.get("workflow_runs", [])
                        if runs:
                            run_data = runs[0]
                            return WorkflowRun(
                                id=run_data["id"],
                                name=run_data["name"],
                                status=run_data["status"],
                                conclusion=run_data.get("conclusion"),
                                html_url=run_data["html_url"],
                                created_at=datetime.fromisoformat(
                                    run_data["created_at"].replace("Z", "+00:00")
                                ),
                                updated_at=datetime.fromisoformat(
                                    run_data["updated_at"].replace("Z", "+00:00")
                                ),
                            )
            return None
        except Exception as e:
            logger.error(f"Failed to get workflow run for {repository.full_name}: {e}")
            return None

    async def _update_deployment_from_workflow(self, deployment: DeploymentJob) -> None:
        """Update deployment status based on GitHub Actions workflow."""
        if not deployment.workflow_run:
            return

        try:
            # Get updated workflow run status
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/repos/{deployment.repository.full_name}/actions/runs/{deployment.workflow_run.id}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        run_data = await response.json()

                        # Update workflow run
                        deployment.workflow_run.status = run_data["status"]
                        deployment.workflow_run.conclusion = run_data.get("conclusion")
                        deployment.workflow_run.updated_at = datetime.fromisoformat(
                            run_data["updated_at"].replace("Z", "+00:00")
                        )

                        # Update deployment status based on workflow
                        if run_data["status"] == "completed":
                            if run_data.get("conclusion") == "success":
                                deployment.status = DeploymentStatus.SUCCESS
                                deployment.completed_at = datetime.now()
                                deployment.build_logs.append(
                                    "GitHub Actions deployment completed successfully!"
                                )
                            else:
                                deployment.status = DeploymentStatus.FAILURE
                                deployment.completed_at = datetime.now()
                                deployment.error_message = (
                                    f"GitHub Actions workflow failed: "
                                    f"{run_data.get('conclusion')}"
                                )
                                deployment.build_logs.append(
                                    f"GitHub Actions workflow failed: "
                                    f"{run_data.get('conclusion')}"
                                )
                        elif run_data["status"] in ["queued", "in_progress"]:
                            deployment.status = DeploymentStatus.IN_PROGRESS

        except Exception as e:
            logger.error(f"Failed to update deployment from workflow: {e}")
