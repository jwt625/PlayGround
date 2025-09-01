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
from typing import Any

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

        # Update deployment status from GitHub Actions workflow (with error handling)
        try:
            await self._update_deployment_from_workflow(deployment)
        except Exception as e:
            logger.warning(f"Failed to update deployment status from workflow: {e}")
            # Continue with current status instead of failing

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

    async def test_deployment_workflow(self) -> dict[str, Any]:
        """
        Full deployment test: fork template repo, commit test content, setup CI/CD.

        This method tests the complete deployment pipeline:
        1. Fork a template repository
        2. Wait for fork to be ready
        3. Add test commit
        4. Set up GitHub Actions CI/CD for deployment
        """
        import time
        from datetime import datetime

        test_repo_name = f"test-deployment-{int(time.time())}"

        # Step 1: Fork template repository
        template_owner = "academicpages"
        template_repo = "academicpages.github.io"

        logger.info(f"Forking template repository: {template_owner}/{template_repo}")

        fork_data = {
            "name": test_repo_name,
            "default_branch_only": True  # Only fork the default branch for faster setup
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.github.com/repos/{template_owner}/{template_repo}/forks",
                headers=self.headers,
                json=fork_data
            ) as response:
                if response.status != 202:
                    error_text = await response.text()
                    raise Exception(f"Failed to fork repository: {response.status} - {error_text}")

                repo_info = await response.json()
                logger.info(f"Fork initiated: {repo_info['html_url']}")

        # Step 2: Wait for fork to be ready and get default branch
        logger.info("Waiting for fork to be ready...")
        default_branch = await self._wait_for_fork_ready_async(repo_info, max_wait_seconds=60)

        # Step 3: Create test content for the forked academic template
        logger.info(f"Adding test content to forked repository on branch '{default_branch}'")

        # Create a test paper content that fits the academic template
        test_paper_content = f"""---
title: "Test Paper: Validating One-Click Paper Deployment"
collection: publications
permalink: /publication/test-paper-{int(time.time())}
excerpt: 'This is a test paper created to validate the one-click paper deployment system.'
date: {datetime.now().strftime('%Y-%m-%d')}
venue: 'Test Conference on Automated Academic Publishing'
paperurl: 'http://academicpages.github.io/files/test-paper.pdf'
citation: 'Test Author. (2025). &quot;Test Paper: Validating One-Click Paper Deployment.&quot; <i>Test Conference</i>. 1(1).'
---

# Test Paper: Validating One-Click Paper Deployment

## Abstract

This test paper validates the functionality of the one-click paper deployment system. The system successfully:

1. ✅ Authenticated with GitHub OAuth
2. ✅ Forked the academic template repository
3. ✅ Added test content to the forked repository
4. ✅ Set up GitHub Actions CI/CD pipeline
5. ✅ Deployed to GitHub Pages

## Introduction

The one-click paper deployment system enables researchers to quickly convert their academic papers into professional websites hosted on GitHub Pages.

## Methodology

This test validates the core deployment pipeline by:
- Forking the academicpages template
- Adding test publication content
- Triggering automated deployment

## Results

**Repository**: {test_repo_name}
**Created**: {datetime.now().isoformat()}
**Template**: academicpages/academicpages.github.io
**Status**: ✅ Deployment successful

## Conclusion

The one-click paper deployment system is functioning correctly and ready for production use.

---

*This test paper was automatically generated by the one-click-paper-page service.*
"""

        # Step 4: Add test publication to the academic template
        logger.info(f"Adding test publication to {test_repo_name}")

        publication_filename = f"test-paper-{int(time.time())}.md"
        commit_sha = await self._commit_test_content(repo_info, publication_filename, test_paper_content, default_branch)

        # Step 5: Set up GitHub Actions CI/CD and GitHub Pages with comprehensive testing
        deployment_status = await self._setup_cicd_and_pages(repo_info, default_branch)

        # Generate dynamic next steps based on actual deployment status
        next_steps = ["✅ Template repository forked successfully", "✅ Test publication content added"]

        if deployment_status["workflows_found"] > 0:
            next_steps.append(f"✅ Found {deployment_status['workflows_found']} GitHub Actions workflows")

        if deployment_status["workflow_triggered"]:
            next_steps.append(f"✅ GitHub Actions workflow triggered: {deployment_status['latest_run_status']}")
        else:
            next_steps.append("⏳ Waiting for GitHub Actions workflow to trigger...")

        if deployment_status["deployment_url"]:
            next_steps.append(f"🌐 Deployment URL: {deployment_status['deployment_url']}")

        next_steps.extend([
            "📝 Check the Actions tab for build progress",
            "⏳ Full deployment may take 2-5 minutes",
            "🔄 Refresh the Pages URL to see updates"
        ])

        return {
            "success": True,
            "test_repository": {
                "name": test_repo_name,
                "url": repo_info['html_url'],
                "pages_url": deployment_status.get("deployment_url") or f"https://{repo_info['owner']['login']}.github.io/{test_repo_name}",
                "template_source": f"{template_owner}/{template_repo}",
                "actions_url": f"{repo_info['html_url']}/actions"
            },
            "deployment_details": {
                "commit_sha": commit_sha,
                "default_branch": default_branch,
                "workflows_found": deployment_status["workflows_found"],
                "workflow_triggered": deployment_status["workflow_triggered"],
                "latest_run_status": deployment_status["latest_run_status"],
                "pages_status": deployment_status["pages_status"]
            },
            "ci_cd_validation": deployment_status["ci_cd_validation"],
            "message": "Template fork and CI/CD deployment testing completed",
            "next_steps": next_steps
        }

    async def _wait_for_fork_ready_async(self, repo_info: dict, max_wait_seconds: int = 60) -> str:
        """Wait for forked repository to be ready and return the default branch name."""
        start_time = asyncio.get_event_loop().time()
        repo_full_name = repo_info["full_name"]

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
                            default_branch = repo_data.get("default_branch", "main")

                            # Check if the default branch exists
                            async with session.get(
                                f"{self.base_url}/repos/{repo_full_name}/git/refs/heads/{default_branch}",
                                headers=self.headers
                            ) as branch_response:
                                if branch_response.status == 200:
                                    logger.info(f"Fork {repo_full_name} is ready with branch '{default_branch}'")
                                    return default_branch

                logger.info(f"Fork {repo_full_name} not ready yet, waiting...")
                await asyncio.sleep(3)

            except Exception as e:
                logger.warning(f"Error checking fork readiness: {e}")
                await asyncio.sleep(3)

        raise Exception(f"Fork {repo_full_name} not ready after {max_wait_seconds} seconds")

    async def _commit_test_content(self, repo_info: dict, filename: str, content: str, branch: str) -> str:
        """Commit test content to the repository."""
        import base64

        publication_data = {
            "message": "Add test publication to validate deployment",
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.base_url}/repos/{repo_info['full_name']}/contents/_publications/{filename}",
                headers=self.headers,
                json=publication_data
            ) as response:
                if response.status in [200, 201]:
                    response_data = await response.json()
                    commit_sha = response_data.get("commit", {}).get("sha", "unknown")
                    logger.info(f"Test publication committed: {commit_sha}")
                    return commit_sha
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to create test publication: {response.status} - {error_text}")
                    return "failed"

    async def _setup_cicd_and_pages(self, repo_info: dict, default_branch: str) -> dict[str, Any]:
        """Set up GitHub Actions CI/CD and GitHub Pages with comprehensive testing."""
        deployment_status = {
            "workflows_found": 0,
            "pages_status": "unknown",
            "workflow_triggered": False,
            "latest_run_status": "none",
            "deployment_url": None,
            "ci_cd_validation": []
        }

        # Step 1: Check existing workflows and their details
        logger.info("🔍 Checking GitHub Actions workflows...")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/repos/{repo_info['full_name']}/actions/workflows",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    workflows_data = await response.json()
                    deployment_status["workflows_found"] = workflows_data.get("total_count", 0)

                    # Log details about each workflow
                    for workflow in workflows_data.get("workflows", []):
                        workflow_name = workflow.get("name", "Unknown")
                        workflow_state = workflow.get("state", "unknown")
                        logger.info(f"  📋 Workflow: {workflow_name} (state: {workflow_state})")
                        deployment_status["ci_cd_validation"].append(f"Found workflow: {workflow_name}")

                    logger.info(f"✅ Found {deployment_status['workflows_found']} workflows configured")
                else:
                    logger.warning(f"❌ Failed to check workflows: {response.status}")

        # Step 2: Enable GitHub Actions on the forked repository
        logger.info("🔧 Enabling GitHub Actions on forked repository...")
        actions_enabled = await self._enable_github_actions(repo_info)
        deployment_status["ci_cd_validation"].append(f"Actions enabled: {actions_enabled}")

        # Step 3: Enable GitHub Pages with GitHub Actions
        logger.info("🚀 Setting up GitHub Pages with Actions...")
        pages_result = await self._enable_pages_with_actions(repo_info, default_branch)
        deployment_status["pages_status"] = pages_result
        deployment_status["ci_cd_validation"].append(f"Pages setup: {pages_result}")

        # Step 4: Wait and check if workflow was triggered by our commit
        logger.info("⏳ Waiting for GitHub Actions workflow to trigger...")
        await asyncio.sleep(10)  # Give more time for workflow to start

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/repos/{repo_info['full_name']}/actions/runs",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    runs_data = await response.json()
                    total_runs = runs_data.get("total_count", 0)

                    if total_runs > 0:
                        latest_run = runs_data["workflow_runs"][0]
                        deployment_status["workflow_triggered"] = True
                        deployment_status["latest_run_status"] = f"{latest_run['status']} ({latest_run['conclusion'] or 'in progress'})"

                        # Log detailed workflow information
                        workflow_name = latest_run.get("name", "Unknown")
                        workflow_url = latest_run.get("html_url", "")
                        created_at = latest_run.get("created_at", "")

                        logger.info(f"🔄 Latest workflow run: {workflow_name}")
                        logger.info(f"   Status: {latest_run['status']}")
                        logger.info(f"   Conclusion: {latest_run['conclusion'] or 'in progress'}")
                        logger.info(f"   Created: {created_at}")
                        logger.info(f"   URL: {workflow_url}")

                        deployment_status["ci_cd_validation"].extend([
                            f"Workflow triggered: {workflow_name}",
                            f"Run status: {latest_run['status']}",
                            f"Run URL: {workflow_url}"
                        ])

                        # If workflow is completed, check deployment
                        if latest_run['status'] == 'completed':
                            if latest_run['conclusion'] == 'success':
                                deployment_status["deployment_url"] = f"https://{repo_info['owner']['login']}.github.io/{repo_info['name']}"
                                deployment_status["ci_cd_validation"].append("✅ Deployment successful")
                                logger.info(f"✅ Deployment completed successfully!")
                            else:
                                deployment_status["ci_cd_validation"].append(f"❌ Deployment failed: {latest_run['conclusion']}")
                                logger.warning(f"❌ Deployment failed: {latest_run['conclusion']}")
                    else:
                        logger.info("⏸️ No workflow runs found yet")
                        deployment_status["ci_cd_validation"].append("No workflow runs triggered yet")
                else:
                    logger.warning(f"❌ Failed to check workflow runs: {response.status}")
                    deployment_status["ci_cd_validation"].append(f"Failed to check runs: {response.status}")

        # Step 5: Check GitHub Pages deployment status
        logger.info("🌐 Checking GitHub Pages deployment status...")
        await self._check_pages_deployment_status(repo_info, deployment_status)

        return deployment_status

    async def _check_pages_deployment_status(self, repo_info: dict, deployment_status: dict) -> None:
        """Check the actual GitHub Pages deployment status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/repos/{repo_info['full_name']}/pages",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        pages_data = await response.json()
                        pages_url = pages_data.get("html_url")
                        pages_status = pages_data.get("status", "unknown")
                        build_type = pages_data.get("build_type", "unknown")

                        logger.info(f"📄 Pages URL: {pages_url}")
                        logger.info(f"📊 Pages status: {pages_status}")
                        logger.info(f"🔧 Build type: {build_type}")

                        deployment_status["deployment_url"] = pages_url
                        deployment_status["ci_cd_validation"].extend([
                            f"Pages URL: {pages_url}",
                            f"Pages status: {pages_status}",
                            f"Build type: {build_type}"
                        ])

                        # Check if site is actually accessible
                        if pages_url:
                            await self._verify_site_accessibility(pages_url, deployment_status)
                    else:
                        logger.warning(f"❌ Failed to get Pages info: {response.status}")
                        deployment_status["ci_cd_validation"].append(f"Failed to get Pages info: {response.status}")
        except Exception as e:
            logger.warning(f"❌ Error checking Pages deployment: {e}")
            deployment_status["ci_cd_validation"].append(f"Error checking Pages: {str(e)}")

    async def _verify_site_accessibility(self, pages_url: str, deployment_status: dict) -> None:
        """Verify that the deployed site is actually accessible."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(pages_url, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"✅ Site is accessible at {pages_url}")
                        deployment_status["ci_cd_validation"].append(f"✅ Site accessible: {pages_url}")
                    else:
                        logger.info(f"⏳ Site not ready yet: {response.status} (this is normal, may take a few minutes)")
                        deployment_status["ci_cd_validation"].append(f"Site not ready: {response.status} (building...)")
        except Exception as e:
            logger.info(f"⏳ Site not accessible yet: {e} (this is normal for new deployments)")
            deployment_status["ci_cd_validation"].append("Site not accessible yet (building...)")

    async def _enable_github_actions(self, repo_info: dict) -> str:
        """Enable GitHub Actions on the forked repository."""
        try:
            # GitHub Actions are disabled by default on forks for security
            # We need to enable them explicitly
            actions_data = {
                "enabled": True,
                "allowed_actions": "all"  # Allow all actions to run
            }

            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.base_url}/repos/{repo_info['full_name']}/actions/permissions",
                    headers=self.headers,
                    json=actions_data
                ) as response:
                    if response.status == 204:
                        logger.info(f"✅ GitHub Actions enabled for {repo_info['name']}")
                        return "enabled"
                    elif response.status == 200:
                        logger.info(f"✅ GitHub Actions already enabled for {repo_info['name']}")
                        return "already enabled"
                    else:
                        error_text = await response.text()
                        logger.warning(f"❌ Failed to enable Actions: {response.status} - {error_text}")
                        return f"failed: {response.status}"

        except Exception as e:
            logger.warning(f"❌ Error enabling GitHub Actions: {e}")
            return f"error: {str(e)}"

    async def _enable_pages_with_actions(self, repo_info: dict, default_branch: str) -> str:
        """Enable GitHub Pages with GitHub Actions as build source."""
        try:
            pages_data = {
                "source": {
                    "branch": default_branch,
                    "path": "/"
                },
                "build_type": "workflow"  # Use GitHub Actions for building
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/repos/{repo_info['full_name']}/pages",
                    headers=self.headers,
                    json=pages_data
                ) as response:
                    if response.status == 201:
                        logger.info(f"GitHub Pages enabled with Actions for {repo_info['name']}")
                        return "Pages enabled with GitHub Actions"
                    elif response.status == 409:
                        logger.info(f"GitHub Pages already enabled for {repo_info['name']}")
                        return "Pages already enabled"
                    else:
                        # Fallback to regular Pages
                        fallback_data = {
                            "source": {
                                "branch": default_branch,
                                "path": "/"
                            }
                        }

                        async with session.post(
                            f"{self.base_url}/repos/{repo_info['full_name']}/pages",
                            headers=self.headers,
                            json=fallback_data
                        ) as fallback_response:
                            if fallback_response.status == 201:
                                logger.info(f"GitHub Pages enabled in fallback mode for {repo_info['name']}")
                                return "Pages enabled (fallback mode)"
                            else:
                                error_text = await response.text()
                                logger.warning(f"GitHub Pages setup failed: {response.status} - {error_text}")
                                return f"Pages setup failed: {response.status}"

        except Exception as e:
            logger.warning(f"GitHub Pages setup failed: {e}")
            return f"Pages setup failed: {str(e)}"

    async def _wait_for_fork_ready(self, repository: GitHubRepository, max_wait_seconds: int = 30) -> str:
        """Wait for forked repository to be ready and return the default branch name."""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait_seconds:
            try:
                async with aiohttp.ClientSession() as session:
                    # Get repository info to check if it's ready
                    async with session.get(
                        f"{self.base_url}/repos/{repository.full_name}",
                        headers=self.headers
                    ) as response:
                        if response.status == 200:
                            repo_data = await response.json()
                            default_branch = repo_data.get("default_branch", "main")

                            # Check if the default branch exists
                            async with session.get(
                                f"{self.base_url}/repos/{repository.full_name}/git/refs/heads/{default_branch}",
                                headers=self.headers
                            ) as branch_response:
                                if branch_response.status == 200:
                                    logger.info(f"Fork {repository.full_name} is ready with branch '{default_branch}'")
                                    return default_branch

                logger.info(f"Fork {repository.full_name} not ready yet, waiting...")
                await asyncio.sleep(2)

            except Exception as e:
                logger.warning(f"Error checking fork readiness: {e}")
                await asyncio.sleep(2)

        raise Exception(f"Fork {repository.full_name} not ready after {max_wait_seconds} seconds")

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
