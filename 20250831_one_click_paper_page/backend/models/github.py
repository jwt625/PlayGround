"""
Pydantic models for GitHub API integration.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DeploymentStatus(str, Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class TemplateType(str, Enum):
    """Available template types."""
    ACADEMIC_PAGES = "academic-pages"
    AL_FOLIO = "al-folio"
    MINIMAL_ACADEMIC = "minimal-academic"


class TemplateRepository(BaseModel):
    """Template repository configuration."""
    id: TemplateType
    name: str
    description: str
    repository_owner: str
    repository_name: str
    branch: str = Field(default="main")
    features: list[str] = Field(default_factory=list)
    preview_url: str | None = None

    @property
    def full_name(self) -> str:
        """Get the full repository name (owner/repo)."""
        return f"{self.repository_owner}/{self.repository_name}"

    @property
    def clone_url(self) -> str:
        """Get the HTTPS clone URL."""
        return f"https://github.com/{self.full_name}.git"


class GitHubUser(BaseModel):
    """GitHub user information."""
    id: int
    login: str
    name: str | None = None
    email: str | None = None
    avatar_url: str
    html_url: str


class GitHubRepository(BaseModel):
    """GitHub repository information."""
    id: int
    name: str
    full_name: str
    description: str | None = None
    html_url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    private: bool
    owner: GitHubUser
    created_at: datetime
    updated_at: datetime
    pages_url: str | None = None


class CreateRepositoryRequest(BaseModel):
    """Request to create a new repository."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=350)
    private: bool = Field(default=False, description="Always false for open science")
    template: TemplateType = Field(default=TemplateType.ACADEMIC_PAGES)
    conversion_job_id: str = Field(..., description="ID of the conversion job")


class CreateRepositoryResponse(BaseModel):
    """Response from repository creation."""
    repository: GitHubRepository
    deployment_id: str
    status: DeploymentStatus
    message: str


class FileContent(BaseModel):
    """File content for repository commits."""
    path: str
    content: str
    encoding: str = Field(default="utf-8")


class CommitRequest(BaseModel):
    """Request to commit files to repository."""
    repository_name: str
    message: str
    files: list[FileContent]
    branch: str = Field(default="main")


class CommitResponse(BaseModel):
    """Response from file commit."""
    sha: str
    url: str
    html_url: str
    message: str


class WorkflowRun(BaseModel):
    """GitHub Actions workflow run information."""
    id: int
    name: str
    status: str  # queued, in_progress, completed
    conclusion: str | None = None  # success, failure, neutral, cancelled, etc.
    html_url: str
    created_at: datetime
    updated_at: datetime


class DeploymentJob(BaseModel):
    """Deployment job tracking."""
    id: str
    repository: GitHubRepository
    conversion_job_id: str
    status: DeploymentStatus
    template: TemplateType
    workflow_run: WorkflowRun | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    pages_url: str | None = None
    build_logs: list[str] = Field(default_factory=list)


class DeploymentStatusResponse(BaseModel):
    """Response for deployment status queries."""
    deployment_id: str
    status: DeploymentStatus
    repository: GitHubRepository
    pages_url: str | None = None
    workflow_run: WorkflowRun | None = None
    progress_percentage: int = Field(ge=0, le=100)
    message: str
    error_message: str | None = None
    estimated_completion: datetime | None = None


class TemplateInfo(BaseModel):
    """Template information for frontend."""
    id: str  # Changed from TemplateType to string for frontend compatibility
    name: str
    description: str
    preview_url: str | None = None
    features: list[str]
    repository_url: str
    repository_owner: str
    repository_name: str


class GitHubPagesConfig(BaseModel):
    """GitHub Pages configuration."""
    source_branch: str = Field(default="main")
    source_path: str = Field(default="/")
    custom_domain: str | None = None
    enforce_https: bool = Field(default=True)


class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""
    repository_name: str
    template: TemplateType
    conversion_job_id: str | None = None
    github_pages: GitHubPagesConfig = Field(default_factory=GitHubPagesConfig)
    paper_title: str | None = None
    paper_authors: list[str] = Field(default_factory=list)
    paper_date: str | None = None


# Template definitions moved to TemplateService


class OAuthTokenRequest(BaseModel):
    """Request for OAuth token exchange."""
    code: str
    state: str | None = None
    redirect_uri: str


class OAuthTokenResponse(BaseModel):
    """Response from OAuth token exchange."""
    access_token: str
    token_type: str = "bearer"
    scope: str = ""


class OAuthRevokeRequest(BaseModel):
    """Request to revoke OAuth token."""
    access_token: str
