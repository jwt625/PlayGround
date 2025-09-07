"""
Workflow service for GitHub Actions workflow management.

This service handles:
- GitHub Actions workflow detection and management
- Custom deployment workflow creation
- Workflow run status monitoring
- Jekyll deployment workflow templates
"""

import base64
import logging
from typing import Any

import aiohttp

from models.github import GitHubRepository, WorkflowRun

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for GitHub Actions workflow management."""

    def __init__(self, access_token: str):
        """
        Initialize workflow service.

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

    async def has_deployment_workflow(
        self, template_data: dict[str, Any]
    ) -> bool:
        """Check if template already has deployment workflows."""
        template_files = template_data.get("tree", [])
        has_deployment_workflow = any(
            f["path"].startswith(".github/workflows/") and
            ("jekyll" in f["path"].lower() or "deploy" in f["path"].lower() or "pages" in f["path"].lower())
            for f in template_files
        )

        if has_deployment_workflow:
            logger.info("✅ Template already has deployment workflow")
        else:
            logger.info("⚙️ Template does not have deployment workflow")

        return has_deployment_workflow

    async def add_deployment_workflow_if_needed(
        self, repository: GitHubRepository, template_data: dict[str, Any]
    ) -> None:
        """Add deployment workflow if not already present in template."""
        if await self.has_deployment_workflow(template_data):
            logger.info("✅ Template already has deployment workflow, skipping custom workflow")
            return

        logger.info(f"⚙️ Adding custom deployment workflow to {repository.full_name}")
        await self.add_deployment_workflow(repository)

    async def add_deployment_workflow(self, repository: GitHubRepository) -> None:
        """Add a custom GitHub Actions deployment workflow."""
        logger.info(f"⚙️ Adding deployment workflow to {repository.full_name}")

        # Get the Jekyll deployment workflow content
        workflow_content = self.get_jekyll_deployment_workflow()
        workflow_path = ".github/workflows/deploy.yml"
        workflow_content_b64 = base64.b64encode(workflow_content.encode()).decode()

        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: Get current branch reference
                async with session.get(
                    f"{self.base_url}/repos/{repository.full_name}/git/refs/heads/{repository.default_branch}",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to get branch ref: {response.status}")

                    ref_data = await response.json()
                    current_commit_sha = ref_data["object"]["sha"]

                # Step 2: Create blob for workflow file
                workflow_blob_data = {
                    "content": workflow_content_b64,
                    "encoding": "base64"
                }

                async with session.post(
                    f"{self.base_url}/repos/{repository.full_name}/git/blobs",
                    headers=self.headers,
                    json=workflow_blob_data
                ) as blob_response:
                    if blob_response.status != 201:
                        raise Exception(
                            f"Failed to create workflow blob: {blob_response.status}"
                        )

                    blob_result = await blob_response.json()
                    workflow_blob_sha = blob_result["sha"]

                # Step 3: Get current tree and add workflow file to it
                async with session.get(
                    f"{self.base_url}/repos/{repository.full_name}/git/trees/{current_commit_sha}?recursive=1",
                    headers=self.headers
                ) as tree_response:
                    if tree_response.status != 200:
                        raise Exception(f"Failed to get current tree: {tree_response.status}")

                    current_tree_data = await tree_response.json()
                    existing_tree_items = current_tree_data["tree"]

                # Add workflow file to existing tree items
                tree_items = existing_tree_items + [{
                    "path": workflow_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": workflow_blob_sha
                }]

                tree_data = {"tree": tree_items}
                async with session.post(
                    f"{self.base_url}/repos/{repository.full_name}/git/trees",
                    headers=self.headers,
                    json=tree_data
                ) as tree_response:
                    if tree_response.status != 201:
                        tree_error = await tree_response.json()
                        raise Exception(f"Failed to create tree: {tree_error}")

                    tree_result = await tree_response.json()
                    new_tree_sha = tree_result["sha"]

                # Step 4: Create commit
                commit_data = {
                    "message": "Add GitHub Actions deployment workflow",
                    "tree": new_tree_sha,
                    "parents": [current_commit_sha]
                }

                async with session.post(
                    f"{self.base_url}/repos/{repository.full_name}/git/commits",
                    headers=self.headers,
                    json=commit_data
                ) as commit_response:
                    if commit_response.status != 201:
                        commit_error = await commit_response.json()
                        raise Exception(f"Failed to create commit: {commit_error}")

                    commit_result = await commit_response.json()
                    new_commit_sha = commit_result["sha"]

                # Step 5: Update branch reference
                ref_update = {"sha": new_commit_sha}
                async with session.patch(
                    f"{self.base_url}/repos/{repository.full_name}/git/refs/heads/{repository.default_branch}",
                    headers=self.headers,
                    json=ref_update
                ) as ref_response:
                    if ref_response.status != 200:
                        ref_error = await ref_response.json()
                        raise Exception(f"Failed to update branch: {ref_error}")

                logger.info(f"✅ Deployment workflow added to {repository.full_name}")

        except Exception as e:
            logger.error(f"❌ Failed to add deployment workflow: {e}")
            # Don't fail the entire process if workflow addition fails

    async def get_latest_workflow_run(
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
                                created_at=run_data["created_at"],
                                updated_at=run_data["updated_at"]
                            )
                    return None
        except Exception as e:
            logger.error(f"Failed to get latest workflow run: {e}")
            return None

    def get_jekyll_deployment_workflow(self) -> str:
        """Get the Jekyll deployment workflow YAML content."""
        return """name: Deploy Academic Site

on:
  push:
    branches: [ main, master ]
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.1'
          bundler-cache: true

      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4

      - name: Build with Jekyll
        run: bundle exec jekyll build --baseurl "${{ steps.pages.outputs.base_path }}"
        env:
          JEKYLL_ENV: production

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
"""
