"""
Conversion service for integrating marker converter into FastAPI backend.
"""

import asyncio
import logging
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from models.conversion import (
    ConversionMetrics,
    ConversionMode,
    ConversionResult,
    ConversionStatus,
    QualityAssessment,
)

# Setup logging
logger = logging.getLogger(__name__)

# Import the marker converter from scripts
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

try:
    from marker_converter import (
        ConversionMode as ScriptConversionMode,
    )
    from marker_converter import (  # type: ignore[import-not-found]
        MarkerConverter as ScriptMarkerConverter,
    )
    MARKER_AVAILABLE = True
except ImportError:
    logger.warning("Marker converter not available, using placeholder mode")
    MARKER_AVAILABLE = False


def _convert_mode_to_script_mode(mode: ConversionMode) -> Any:
    """Convert API ConversionMode to script ConversionMode."""
    if not MARKER_AVAILABLE:
        return mode  # Return as-is if marker not available

    # Map API enum values to script enum values
    mode_mapping = {
        ConversionMode.AUTO: ScriptConversionMode.AUTO,
        ConversionMode.FAST: ScriptConversionMode.FAST,
        ConversionMode.QUALITY: ScriptConversionMode.QUALITY,
    }
    return mode_mapping.get(mode, ScriptConversionMode.AUTO)


class ConversionService:
    """
    Service for handling PDF/DOCX to HTML conversion using Marker.

    Provides async wrapper around the marker converter with job tracking
    and progress reporting.
    """

    def __init__(self, temp_dir: Path | None = None):
        """
        Initialize the conversion service.

        Args:
            temp_dir: Directory for temporary files. If None, uses system temp.
        """
        self.temp_dir = temp_dir or Path("/tmp/conversions")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Job tracking
        self._jobs: dict[str, dict[str, Any]] = {}

        # Initialize marker converter if available
        if MARKER_AVAILABLE:
            script_mode = _convert_mode_to_script_mode(ConversionMode.AUTO)
            self._converter = ScriptMarkerConverter(mode=script_mode)
        else:
            self._converter = None
            logger.warning("Marker converter not available")

    def create_job(self, mode: ConversionMode = ConversionMode.AUTO) -> str:
        """
        Create a new conversion job.

        Args:
            mode: Conversion mode to use

        Returns:
            Unique job ID
        """
        job_id = str(uuid.uuid4())

        # Create job directory
        job_dir = self.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Initialize job tracking
        self._jobs[job_id] = {
            "status": ConversionStatus.QUEUED,
            "progress": 0,
            "stage": "initialized",
            "message": "Job created",
            "mode": mode,
            "job_dir": job_dir,
            "created_at": time.time(),
            "error": None,
            "result": None,
        }

        logger.info(f"Created conversion job {job_id} with mode {mode}")
        return job_id

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """
        Get the current status of a conversion job.

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if job not found
        """
        return self._jobs.get(job_id)

    async def convert_file(
        self,
        job_id: str,
        input_file: Path,
        progress_callback: Callable[..., Any] | None = None
    ) -> ConversionResult:
        """
        Convert a file asynchronously.

        Args:
            job_id: Job identifier
            input_file: Path to input PDF/DOCX file
            progress_callback: Optional callback for progress updates

        Returns:
            Conversion result

        Raises:
            ValueError: If job not found or invalid
            RuntimeError: If conversion fails
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]
        job_dir = job["job_dir"]
        mode = job["mode"]

        try:
            # Update job status
            self._update_job_status(
                job_id, ConversionStatus.PROCESSING, 10, "starting_conversion",
                "Starting conversion process"
            )
            if progress_callback:
                await progress_callback(job_id, 10, "starting_conversion")

            # Use input file directly (it's already in the job directory)
            input_copy = input_file

            # Update progress
            self._update_job_status(
                job_id, ConversionStatus.PROCESSING, 20, "file_prepared",
                "Input file prepared"
            )
            if progress_callback:
                await progress_callback(job_id, 20, "file_prepared")

            # Run conversion in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_conversion,
                job_id,
                input_copy,
                job_dir,
                mode,
                progress_callback,
            )

            # Update job with result
            job["result"] = result
            self._update_job_status(
                job_id, ConversionStatus.COMPLETED, 100, "completed",
                "Conversion completed successfully"
            )
            if progress_callback:
                await progress_callback(job_id, 100, "completed")

            return result

        except Exception as e:
            logger.error(f"Conversion failed for job {job_id}: {e}")
            self._update_job_status(
                job_id, ConversionStatus.FAILED, 0, "failed",
                f"Conversion failed: {str(e)}", error=str(e)
            )
            if progress_callback:
                await progress_callback(job_id, 0, "failed")
            raise RuntimeError(f"Conversion failed: {e}")

    def _run_conversion(
        self,
        job_id: str,
        input_file: Path,
        output_dir: Path,
        mode: ConversionMode,
        progress_callback: Callable[..., Any] | None = None
    ) -> ConversionResult:
        """
        Run the actual conversion (blocking operation).

        This method runs in a thread pool to avoid blocking the async event loop.
        """
        start_time = time.time()

        try:
            if not MARKER_AVAILABLE or not self._converter:
                # Fallback to placeholder implementation
                return self._placeholder_conversion(job_id, input_file, output_dir)

            # Update converter mode (convert API enum to script enum)
            self._converter.mode = _convert_mode_to_script_mode(mode)

            # Update progress
            self._update_job_status(
                job_id, ConversionStatus.PROCESSING, 30, "loading_models",
                "Loading conversion models"
            )

            # Run conversion
            success = self._converter.convert_to_html(input_file, output_dir)

            if not success:
                raise RuntimeError("Conversion failed")

            # Update progress
            self._update_job_status(
                job_id, ConversionStatus.PROCESSING, 90, "finalizing",
                "Finalizing output files"
            )

            # Get conversion metrics
            metrics = self._converter.get_performance_metrics()

            # Build result
            html_file = output_dir / "index.html"
            markdown_file = output_dir / "document.md"
            images_dir = output_dir / "images"

            output_files = [str(html_file)]
            if markdown_file.exists():
                output_files.append(str(markdown_file))

            # Count images
            image_count = 0
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*")))
                output_files.extend([str(f) for f in images_dir.glob("*")])

            # Get markdown length
            markdown_length = 0
            if markdown_file.exists():
                markdown_length = len(markdown_file.read_text(encoding='utf-8'))

            # Create conversion result
            result = ConversionResult(
                job_id=job_id,
                status=ConversionStatus.COMPLETED,
                output_files=output_files,
                metrics=ConversionMetrics(
                    total_conversion_time=time.time() - start_time,
                    mode_used=metrics.get("mode_used", mode),
                    quality_assessment=QualityAssessment(
                        has_good_text=metrics.get("quality_assessment", {}).get(
                            "has_good_text", True
                        ),
                        recommended_mode=metrics.get("quality_assessment", {}).get(
                            "recommended_mode", mode
                        ),
                        confidence=metrics.get("quality_assessment", {}).get(
                            "confidence", "medium"
                        ),
                        avg_chars_per_page=metrics.get("quality_assessment", {}).get(
                            "avg_chars_per_page", 0
                        ),
                        text_coverage=metrics.get("quality_assessment", {}).get(
                            "text_coverage", 1.0
                        ),
                    ),
                    model_load_time=metrics.get("model_load_time"),
                    processing_time=metrics.get("processing_time"),
                ),
                markdown_length=markdown_length,
                image_count=image_count,
                html_file=str(html_file),
                markdown_file=str(markdown_file),
            )

            return result

        except Exception as e:
            logger.error(f"Conversion execution failed: {e}")
            raise

    def _placeholder_conversion(
        self, job_id: str, input_file: Path, output_dir: Path
    ) -> ConversionResult:
        """Placeholder conversion when Marker is not available."""
        logger.info("Using placeholder conversion")

        # Create placeholder files
        html_file = output_dir / "index.html"
        markdown_file = output_dir / "document.md"

        placeholder_content = f"""# Document Conversion (Placeholder)

This is a placeholder conversion for: {input_file.name}

The actual Marker converter is not available in this environment.
"""

        markdown_file.write_text(placeholder_content, encoding='utf-8')

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Conversion</title>
</head>
<body>
    <h1>Document Conversion (Placeholder)</h1>
    <p>This is a placeholder conversion for: {input_file.name}</p>
    <p>The actual Marker converter is not available in this environment.</p>
</body>
</html>"""

        html_file.write_text(html_content, encoding='utf-8')

        return ConversionResult(
            job_id=job_id,
            status=ConversionStatus.COMPLETED,
            output_files=[str(html_file), str(markdown_file)],
            metrics=ConversionMetrics(
                total_conversion_time=1.0,
                mode_used=ConversionMode.FAST,
                quality_assessment=QualityAssessment(
                    has_good_text=True,
                    recommended_mode=ConversionMode.FAST,
                    confidence="low",
                    avg_chars_per_page=100,
                    text_coverage=1.0,
                ),
                model_load_time=None,
                processing_time=None,
            ),
            markdown_length=len(placeholder_content),
            image_count=0,
            html_file=str(html_file),
            markdown_file=str(markdown_file),
        )

    def _update_job_status(
        self,
        job_id: str,
        status: ConversionStatus,
        progress: int,
        stage: str,
        message: str,
        error: str | None = None
    ) -> None:
        """Update job status in tracking dictionary."""
        if job_id in self._jobs:
            self._jobs[job_id].update({
                "status": status,
                "progress": progress,
                "stage": stage,
                "message": message,
                "error": error,
                "updated_at": time.time(),
            })

    def cleanup_job(self, job_id: str) -> bool:
        """
        Clean up job files and remove from tracking.

        Args:
            job_id: Job identifier

        Returns:
            True if cleanup successful
        """
        if job_id not in self._jobs:
            return False

        try:
            job_dir = self._jobs[job_id]["job_dir"]
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)

            del self._jobs[job_id]
            logger.info(f"Cleaned up job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup job {job_id}: {e}")
            return False
