"""
Pydantic models for conversion API endpoints.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConversionMode(str, Enum):
    """Conversion mode options."""
    AUTO = "auto"      # Smart mode: try fast, fallback to quality if needed
    FAST = "fast"      # Fast mode: disable OCR (~40 seconds)
    QUALITY = "quality"  # Quality mode: full OCR (~6 minutes)


class ConversionStatus(str, Enum):
    """Conversion job status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileUploadRequest(BaseModel):
    """Request model for file upload."""
    template: str = Field(..., description="Template to use for the website")
    mode: ConversionMode = Field(ConversionMode.AUTO, description="Conversion mode")
    repository_name: str | None = Field(None, description="Custom repository name")


class ConversionJobResponse(BaseModel):
    """Response model for conversion job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: ConversionStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")


class ConversionPhase(str, Enum):
    """Conversion phases."""
    QUEUED = "queued"
    PREPARING = "preparing"
    ANALYZING = "analyzing"
    CONVERTING = "converting"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


class ConversionStatusResponse(BaseModel):
    """Response model for conversion status check."""
    job_id: str = Field(..., description="Job identifier")
    status: ConversionStatus = Field(..., description="Current job status")
    phase: ConversionPhase = Field(..., description="Current conversion phase")
    stage: str = Field(..., description="Current processing stage")
    message: str = Field(..., description="Human-readable status message")
    progress: int = Field(..., description="Conversion progress percentage (0-100)")
    error: str | None = Field(None, description="Error message if failed")


class QualityAssessment(BaseModel):
    """PDF quality assessment results."""
    has_good_text: bool = Field(..., description="Whether PDF has good text extraction")
    recommended_mode: ConversionMode = Field(
        ..., description="Recommended conversion mode"
    )
    confidence: str = Field(..., description="Confidence level: low, medium, high")
    avg_chars_per_page: float = Field(..., description="Average characters per page")
    text_coverage: float = Field(..., description="Percentage of pages with text")


class ConversionMetrics(BaseModel):
    """Conversion performance metrics."""
    total_conversion_time: float = Field(
        ..., description="Total conversion time in seconds"
    )
    mode_used: ConversionMode = Field(..., description="Actual conversion mode used")
    quality_assessment: QualityAssessment = Field(
        ..., description="PDF quality assessment"
    )
    model_load_time: float | None = Field(
        None, description="Model loading time in seconds"
    )
    processing_time: float | None = Field(
        None, description="Processing time in seconds"
    )


class PaperMetadata(BaseModel):
    """Extracted paper metadata."""
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None


class ConversionResult(BaseModel):
    """Complete conversion result."""
    job_id: str = Field(..., description="Job identifier")
    status: ConversionStatus = Field(..., description="Final job status")
    success: bool = Field(..., description="Whether conversion was successful")
    output_dir: str = Field(..., description="Output directory path")
    output_files: list[str] = Field(..., description="List of generated output files")
    metrics: ConversionMetrics = Field(..., description="Performance metrics")
    markdown_length: int = Field(..., description="Length of extracted markdown")
    image_count: int = Field(..., description="Number of extracted images")
    html_file: str = Field(..., description="Path to generated HTML file")
    markdown_file: str = Field(..., description="Path to generated markdown file")
    metadata: PaperMetadata | None = Field(None, description="Extracted paper metadata")


class ConversionError(BaseModel):
    """Error response model."""
    job_id: str = Field(..., description="Job identifier")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")


class WebSocketMessage(BaseModel):
    """WebSocket message for real-time updates."""
    job_id: str = Field(..., description="Job identifier")
    type: str = Field(
        ..., description="Message type: progress, status, error, complete"
    )
    data: dict[str, Any] = Field(..., description="Message data")
    timestamp: float = Field(..., description="Unix timestamp")
