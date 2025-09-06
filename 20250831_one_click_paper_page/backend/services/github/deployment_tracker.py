"""
Deployment tracker service for deployment job tracking and status management.

This service handles:
- In-memory deployment job tracking (use Redis in production)
- Deployment status updates from GitHub Actions workflows
- Deployment progress monitoring
- Deployment job lifecycle management
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import aiohttp

from models.github import (
    CreateRepositoryRequest,
    DeploymentJob,
    DeploymentStatus,
    DeploymentStatusResponse,
    GitHubRepository,
    WorkflowRun,
)

logger = logging.getLogger(__name__)


class DeploymentTracker:
    """Service for deployment job tracking and status management."""

    def __init__(self, access_token: str):
        """
        Initialize deployment tracker.

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

    def create_deployment_job(
        self, 
        repository: GitHubRepository, 
        request: CreateRepositoryRequest
    ) -> str:
        """
        Create a new deployment job for tracking.

        Args:
            repository: GitHub repository
            request: Repository creation request

        Returns:
            Deployment job ID
        """
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
        logger.info(f"Created deployment job {deployment_id} for {repository.full_name}")
        
        return deployment_id

    def get_deployment_job(self, deployment_id: str) -> DeploymentJob | None:
        """
        Get deployment job by ID.

        Args:
            deployment_id: Deployment job ID

        Returns:
            Deployment job or None if not found
        """
        return self._deployments.get(deployment_id)

    def update_deployment_status(
        self, 
        deployment_id: str, 
        status: DeploymentStatus,
        message: str = None,
        error_message: str = None
    ) -> bool:
        """
        Update deployment status.

        Args:
            deployment_id: Deployment job ID
            status: New deployment status
            message: Optional status message
            error_message: Optional error message

        Returns:
            True if updated successfully, False if deployment not found
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            logger.warning(f"Deployment {deployment_id} not found for status update")
            return False

        deployment.status = status
        
        if message:
            deployment.build_logs.append(message)
            
        if error_message:
            deployment.error_message = error_message
            deployment.build_logs.append(f"Error: {error_message}")

        if status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILURE]:
            deployment.completed_at = datetime.now()

        logger.info(f"Updated deployment {deployment_id} status to {status}")
        return True

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
            await self.update_deployment_from_workflow(deployment)
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
                    progress = 60
                else:
                    progress = 50
            else:
                progress = 40
        elif deployment.status == DeploymentStatus.SUCCESS:
            progress = 100
        elif deployment.status == DeploymentStatus.FAILURE:
            progress = 100

        return DeploymentStatusResponse(
            deployment_id=deployment_id,
            status=deployment.status,
            progress=progress,
            repository_url=deployment.repository.html_url,
            pages_url=f"https://{deployment.repository.owner.login}.github.io/{deployment.repository.name}",
            build_logs=deployment.build_logs,
            workflow_run=deployment.workflow_run,
            created_at=deployment.created_at,
            completed_at=deployment.completed_at,
            error_message=deployment.error_message,
        )

    async def update_deployment_from_workflow(self, deployment: DeploymentJob) -> None:
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

    def set_workflow_run(self, deployment_id: str, workflow_run: WorkflowRun) -> bool:
        """
        Set workflow run for a deployment.

        Args:
            deployment_id: Deployment job ID
            workflow_run: Workflow run information

        Returns:
            True if set successfully, False if deployment not found
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            logger.warning(f"Deployment {deployment_id} not found for workflow run update")
            return False

        deployment.workflow_run = workflow_run
        logger.info(f"Set workflow run {workflow_run.id} for deployment {deployment_id}")
        return True

    def get_all_deployments(self) -> dict[str, DeploymentJob]:
        """
        Get all deployment jobs.

        Returns:
            Dictionary of deployment ID to deployment job
        """
        return self._deployments.copy()

    def cleanup_completed_deployments(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed deployments older than specified age.

        Args:
            max_age_hours: Maximum age in hours for completed deployments

        Returns:
            Number of deployments cleaned up
        """
        current_time = datetime.now()
        cleanup_count = 0
        
        deployment_ids_to_remove = []
        
        for deployment_id, deployment in self._deployments.items():
            if deployment.completed_at:
                age_hours = (current_time - deployment.completed_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    deployment_ids_to_remove.append(deployment_id)
        
        for deployment_id in deployment_ids_to_remove:
            del self._deployments[deployment_id]
            cleanup_count += 1
            
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} completed deployments")
            
        return cleanup_count
