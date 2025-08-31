"""
GitHub API service for repository creation and deployment.
"""

import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from models.github import (
    AVAILABLE_TEMPLATES,
    CommitRequest,
    CommitResponse,
    CreateRepositoryRequest,
    CreateRepositoryResponse,
    DeploymentConfig,
    DeploymentJob,
    DeploymentStatus,
    DeploymentStatusResponse,
    FileContent,
    GitHubPagesConfig,
    GitHubRepository,
    GitHubUser,
    TemplateInfo,
    TemplateType,
    WorkflowRun,
)

logger = logging.getLogger(__name__)


class GitHubService:
    """Service for GitHub API operations."""

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
        self._deployments: Dict[str, DeploymentJob] = {}

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

    async def create_repository(
        self, 
        request: CreateRepositoryRequest
    ) -> CreateRepositoryResponse:
        """
        Create a new repository for the converted paper.

        Args:
            request: Repository creation request

        Returns:
            Repository creation response with deployment tracking
        """
        try:
            # Create the repository
            repo_data = {
                "name": request.name,
                "description": request.description or f"Academic paper website - {request.name}",
                "private": False,  # Always public for open science
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
                created_at=datetime.fromisoformat(repo_json["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(repo_json["updated_at"].replace("Z", "+00:00")),
            )

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

            logger.info(f"Created repository {repository.full_name} with deployment {deployment_id}")

            return CreateRepositoryResponse(
                repository=repository,
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                message="Repository created successfully. Deployment will begin shortly.",
            )

        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise

    async def deploy_converted_content(
        self,
        deployment_id: str,
        converted_content_dir: Path,
        config: DeploymentConfig,
    ) -> None:
        """
        Deploy converted content to the repository.

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
            deployment.build_logs.append("Starting deployment process...")

            # Get template
            template = next(
                (t for t in AVAILABLE_TEMPLATES if t.id == config.template),
                None
            )
            if not template:
                raise Exception(f"Template {config.template} not found")

            # Prepare files for commit
            files = await self._prepare_deployment_files(
                converted_content_dir, template, config
            )

            # Commit files to repository
            commit_request = CommitRequest(
                repository_name=deployment.repository.name,
                message="Deploy converted academic paper",
                files=files,
            )

            await self._commit_files(deployment.repository, commit_request)

            # Enable GitHub Pages
            await self._enable_github_pages(deployment.repository)

            # Update deployment status
            deployment.status = DeploymentStatus.SUCCESS
            deployment.completed_at = datetime.now()
            deployment.pages_url = f"https://{deployment.repository.owner.login}.github.io/{deployment.repository.name}"
            deployment.build_logs.append("Deployment completed successfully!")

            logger.info(f"Deployment {deployment_id} completed successfully")

        except Exception as e:
            deployment = self._deployments.get(deployment_id)
            if deployment:
                deployment.status = DeploymentStatus.FAILURE
                deployment.error_message = str(e)
                deployment.build_logs.append(f"Deployment failed: {e}")
            
            logger.error(f"Deployment {deployment_id} failed: {e}")
            raise

    async def get_deployment_status(self, deployment_id: str) -> DeploymentStatusResponse:
        """
        Get deployment status.

        Args:
            deployment_id: Deployment job ID

        Returns:
            Deployment status response
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            raise Exception(f"Deployment {deployment_id} not found")

        # Calculate progress percentage
        progress = 0
        if deployment.status == DeploymentStatus.PENDING:
            progress = 10
        elif deployment.status == DeploymentStatus.QUEUED:
            progress = 20
        elif deployment.status == DeploymentStatus.IN_PROGRESS:
            progress = 60
        elif deployment.status == DeploymentStatus.SUCCESS:
            progress = 100
        elif deployment.status == DeploymentStatus.FAILURE:
            progress = 0

        # Generate status message
        message = "Deployment pending"
        if deployment.status == DeploymentStatus.IN_PROGRESS:
            message = "Deploying to GitHub Pages..."
        elif deployment.status == DeploymentStatus.SUCCESS:
            message = "Deployment completed successfully!"
        elif deployment.status == DeploymentStatus.FAILURE:
            message = f"Deployment failed: {deployment.error_message}"

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

    async def list_templates(self) -> List[TemplateInfo]:
        """List available templates."""
        return AVAILABLE_TEMPLATES

    async def _prepare_deployment_files(
        self,
        content_dir: Path,
        template: TemplateInfo,
        config: DeploymentConfig,
    ) -> List[FileContent]:
        """Prepare files for deployment."""
        files = []
        
        # Add converted content
        if (content_dir / "index.html").exists():
            with open(content_dir / "index.html", "r", encoding="utf-8") as f:
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

        # Add template-specific files
        files.extend(await self._get_template_files(template, config))

        return files

    def _customize_content(self, content: str, config: DeploymentConfig) -> str:
        """Customize content with paper metadata."""
        # Simple template replacement
        if config.paper_title:
            content = content.replace("<title>Document</title>", f"<title>{config.paper_title}</title>")
        
        if config.paper_authors:
            authors_str = ", ".join(config.paper_authors)
            content = content.replace(
                '<meta name="author" content="">',
                f'<meta name="author" content="{authors_str}">'
            )
        
        return content

    async def _get_template_files(
        self, template: TemplateInfo, config: DeploymentConfig
    ) -> List[FileContent]:
        """Get template-specific files."""
        files = []

        # Get template directory
        template_dir = Path(__file__).parent.parent.parent / "template" / "minimal-academic"

        if config.template == TemplateType.MINIMAL_ACADEMIC and template_dir.exists():
            # Load template files
            template_files = [
                "_config.yml",
                "Gemfile",
                "README.md",
                ".github/workflows/pages.yml"
            ]

            for file_path in template_files:
                full_path = template_dir / file_path
                if full_path.exists():
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Customize content
                    if file_path == "_config.yml":
                        content = self._customize_config(content, config)
                    elif file_path == "README.md":
                        content = self._customize_readme(content, config)

                    files.append(FileContent(path=file_path, content=content))
        else:
            # Fallback to basic configuration
            files.append(FileContent(
                path="_config.yml",
                content=f"""title: {config.paper_title or 'Academic Paper'}
description: {config.repository_name}
author: {', '.join(config.paper_authors) if config.paper_authors else 'Author Name'}
theme: minima
plugins:
  - jekyll-feed
  - jekyll-sitemap
  - jekyll-seo-tag
"""
            ))

        return files

    def _customize_config(self, content: str, config: DeploymentConfig) -> str:
        """Customize _config.yml with paper metadata."""
        if config.paper_title:
            content = content.replace('title: "Academic Paper"', f'title: "{config.paper_title}"')

        if config.paper_authors:
            authors_str = ', '.join(config.paper_authors)
            content = content.replace('author: "Author Name"', f'author: "{authors_str}"')

        content = content.replace('description: "Academic paper website generated by one-click-paper-page"',
                                f'description: "{config.repository_name}"')

        return content

    def _customize_readme(self, content: str, config: DeploymentConfig) -> str:
        """Customize README.md with paper metadata."""
        if config.paper_title:
            content = content.replace('# Academic Paper Website', f'# {config.paper_title}')

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
                        logger.info(f"GitHub Pages already enabled for {repository.full_name}")
                    else:
                        error_data = await response.json()
                        logger.warning(f"Failed to enable GitHub Pages: {error_data}")

        except Exception as e:
            logger.warning(f"Failed to enable GitHub Pages: {e}")
            # Don't fail the deployment if Pages setup fails
