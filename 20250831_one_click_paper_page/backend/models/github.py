"""
Pydantic models for GitHub API integration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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


class GitHubUser(BaseModel):
    """GitHub user information."""
    id: int
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: str
    html_url: str


class GitHubRepository(BaseModel):
    """GitHub repository information."""
    id: int
    name: str
    full_name: str
    description: Optional[str] = None
    html_url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    private: bool
    owner: GitHubUser
    created_at: datetime
    updated_at: datetime
    pages_url: Optional[str] = None


class CreateRepositoryRequest(BaseModel):
    """Request to create a new repository."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=350)
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
    files: List[FileContent]
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
    conclusion: Optional[str] = None  # success, failure, neutral, cancelled, etc.
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
    workflow_run: Optional[WorkflowRun] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    pages_url: Optional[str] = None
    build_logs: List[str] = Field(default_factory=list)


class DeploymentStatusResponse(BaseModel):
    """Response for deployment status queries."""
    deployment_id: str
    status: DeploymentStatus
    repository: GitHubRepository
    pages_url: Optional[str] = None
    workflow_run: Optional[WorkflowRun] = None
    progress_percentage: int = Field(ge=0, le=100)
    message: str
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None


class TemplateInfo(BaseModel):
    """Template information."""
    id: TemplateType
    name: str
    description: str
    preview_url: Optional[str] = None
    features: List[str]
    repository_url: str


class GitHubPagesConfig(BaseModel):
    """GitHub Pages configuration."""
    source_branch: str = Field(default="main")
    source_path: str = Field(default="/")
    custom_domain: Optional[str] = None
    enforce_https: bool = Field(default=True)


class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""
    repository_name: str
    template: TemplateType
    github_pages: GitHubPagesConfig = Field(default_factory=GitHubPagesConfig)
    paper_title: Optional[str] = None
    paper_authors: List[str] = Field(default_factory=list)
    paper_date: Optional[str] = None


# Template definitions
AVAILABLE_TEMPLATES: List[TemplateInfo] = [
    TemplateInfo(
        id=TemplateType.ACADEMIC_PAGES,
        name="Academic Pages (Jekyll)",
        description="Full academic personal site with publications, talks, CV, portfolio",
        repository_url="https://github.com/academicpages/academicpages.github.io",
        features=[
            "Publications page",
            "Talks and presentations",
            "CV/Resume section",
            "Portfolio showcase",
            "Blog functionality",
            "Google Analytics integration"
        ]
    ),
    TemplateInfo(
        id=TemplateType.AL_FOLIO,
        name="al-folio (Jekyll)",
        description="Clean, responsive minimal academic landing page",
        repository_url="https://github.com/alshedivat/al-folio",
        features=[
            "Responsive design",
            "Publication list",
            "Project showcase",
            "Blog support",
            "Dark/light theme",
            "Math formula support"
        ]
    ),
    TemplateInfo(
        id=TemplateType.MINIMAL_ACADEMIC,
        name="Minimal Academic (Static)",
        description="Simple, fast-loading academic paper presentation",
        features=[
            "Minimal design",
            "Fast loading",
            "Mobile responsive",
            "PDF download",
            "Citation export",
            "Social sharing"
        ],
        repository_url="https://github.com/jwt625/minimal-academic-template"
    )
]
