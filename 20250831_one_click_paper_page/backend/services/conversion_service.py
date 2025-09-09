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
    ConversionPhase,
    ConversionResult,
    ConversionStatus,
    QualityAssessment,
)

# Setup logging
logger = logging.getLogger(__name__)

# Import the marker converter from scripts
scripts_path = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

try:
    from marker_converter import (  # type: ignore[import-untyped]
        ConversionMode as ScriptConversionMode,
    )
    from marker_converter import (  # type: ignore[import-untyped]
        MarkerConverter as ScriptMarkerConverter,
    )
    MARKER_AVAILABLE = True
    logger.info(f"✅ Marker converter imported successfully from {scripts_path}")
except ImportError as e:
    logger.warning(f"❌ Marker converter not available: {e}")
    MARKER_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Error importing marker converter: {e}")
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
            "phase": ConversionPhase.QUEUED,
            "stage": "initialized",
            "message": "Job created",
            "mode": mode,
            "job_dir": job_dir,
            "created_at": time.time(),
            "progress": 0,
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

    def get_job_result(self, job_id: str) -> ConversionResult | None:
        """
        Get the conversion result for a completed job.

        Args:
            job_id: Job identifier

        Returns:
            ConversionResult if job is completed, None otherwise
        """
        job = self._jobs.get(job_id)
        if not job or job["status"] != ConversionStatus.COMPLETED:
            return None

        return job.get("result")

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
                job_id,
                ConversionStatus.PROCESSING,
                10,
                "starting_conversion",
                "Starting conversion process",
                phase=ConversionPhase.PREPARING
            )
            if progress_callback:
                await progress_callback(job_id, 10, "starting_conversion")

            # Use input file directly (it's already in the job directory)
            input_copy = input_file

            # Update progress
            self._update_job_status(
                job_id,
                ConversionStatus.PROCESSING,
                20,
                "file_prepared",
                "Input file prepared",
                phase=ConversionPhase.PREPARING
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
                job_id,
                ConversionStatus.COMPLETED,
                100,
                "completed",
                "Conversion completed successfully",
                phase=ConversionPhase.COMPLETED
            )
            if progress_callback:
                await progress_callback(job_id, 100, "completed")

            return result

        except Exception as e:
            logger.error(f"Conversion failed for job {job_id}: {e}")
            self._update_job_status(
                job_id, ConversionStatus.FAILED, 0, "failed",
                f"Conversion failed: {str(e)}",
                phase=ConversionPhase.QUEUED,
                error=str(e)
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
                # Marker library is required for conversion
                raise RuntimeError(f"Marker library is not available. MARKER_AVAILABLE={MARKER_AVAILABLE}, converter={self._converter}")

            # Update converter mode (convert API enum to script enum)
            self._converter.mode = _convert_mode_to_script_mode(mode)

            # Update progress
            self._update_job_status(
                job_id,
                ConversionStatus.PROCESSING,
                30,
                "loading_models",
                "Loading conversion models",
                phase=ConversionPhase.ANALYZING
            )

            # Run conversion with progress tracking
            success = self._run_conversion_with_progress(
                job_id, input_file, output_dir
            )

            if not success:
                raise RuntimeError("Conversion failed")

            # Update progress
            self._update_job_status(
                job_id,
                ConversionStatus.PROCESSING,
                90,
                "finalizing",
                "Finalizing output files",
                phase=ConversionPhase.FINALIZING
            )

            # Get conversion metrics
            metrics = self._converter.get_performance_metrics()
            if metrics is None:
                metrics = {}  # Provide default empty dict if metrics are None

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

            # Load paper metadata if available
            paper_metadata = None
            metadata_file = output_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    import json

                    from models.conversion import PaperMetadata
                    with open(metadata_file, encoding='utf-8') as f:
                        metadata_dict = json.load(f)
                    paper_metadata = PaperMetadata(**metadata_dict)
                    logger.info(f"Loaded paper metadata: {paper_metadata}")
                except Exception as e:
                    logger.warning(f"Failed to load paper metadata: {e}")

            # Create conversion result
            result = ConversionResult(
                job_id=job_id,
                status=ConversionStatus.COMPLETED,
                success=True,
                output_dir=str(output_dir),
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
                metadata=paper_metadata,
            )

            return result

        except Exception as e:
            logger.error(f"Conversion execution failed: {e}")
            raise

    def _run_conversion_with_progress(
        self, job_id: str, input_file: Path, output_dir: Path
    ) -> bool:
        """
        Run conversion with detailed progress updates.

        Args:
            job_id: Job identifier
            input_file: Path to input file
            output_dir: Output directory

        Returns:
            True if conversion successful
        """
        try:
            # Phase 1: Analyzing document
            self._update_job_status(
                job_id,
                ConversionStatus.PROCESSING,
                25,
                "assessing_quality",
                "Analyzing document structure and quality...",
                phase=ConversionPhase.ANALYZING
            )

            # Phase 2: Converting document
            self._update_job_status(
                job_id,
                ConversionStatus.PROCESSING,
                40,
                "loading_models",
                "Loading AI models and starting conversion...",
                phase=ConversionPhase.CONVERTING
            )

            # Run the actual conversion with timer-based progress
            success = self._run_conversion_with_timer(
                job_id, input_file, output_dir
            )

            if success:
                # Phase 3: Processing results
                self._update_job_status(
                    job_id,
                    ConversionStatus.PROCESSING,
                    75,
                    "extracting_images",
                    "Processing images and extracting content...",
                    phase=ConversionPhase.PROCESSING
                )

                # Phase 4: Finalizing
                self._update_job_status(
                    job_id,
                    ConversionStatus.PROCESSING,
                    95,
                    "finalizing",
                    "Finalizing output files...",
                    phase=ConversionPhase.FINALIZING
                )

            return success

        except Exception as e:
            logger.error(f"Conversion with progress failed: {e}")
            return False

    def _run_conversion_with_timer(
        self, job_id: str, input_file: Path, output_dir: Path
    ) -> bool:
        """
        Run conversion with phase-based progress updates and real Marker progress
        capture.

        This provides meaningful progress updates during the actual conversion.
        """
        import threading
        import time

        conversion_result: dict[str, bool | str | None] = {
            "success": False,
            "error": None
        }
        progress_thread_active = True

        def progress_updater() -> None:
            """Background thread to update progress during conversion."""
            # More realistic timing based on actual Marker conversion stages
            conversion_updates = [
                # Phase, stage, message, wait_seconds, progress
                (
                    ConversionPhase.CONVERTING,
                    "layout_analysis",
                    "Analyzing document layout...",
                    8,
                    45,
                ),
                (
                    ConversionPhase.CONVERTING,
                    "text_recognition",
                    "Recognizing text and formulas...",
                    12,
                    55,
                ),
                (
                    ConversionPhase.CONVERTING,
                    "processing_text",
                    "Processing text blocks...",
                    10,
                    65,
                ),
                (
                    ConversionPhase.PROCESSING,
                    "extracting_images",
                    "Extracting images and diagrams...",
                    8,
                    80,
                ),
                (
                    ConversionPhase.PROCESSING,
                    "processing_equations",
                    "Processing mathematical equations...",
                    6,
                    90,
                ),
            ]

            for phase, stage, message, wait_seconds, progress in conversion_updates:
                if not progress_thread_active:
                    break

                self._update_job_status(
                    job_id,
                    ConversionStatus.PROCESSING,
                    progress,
                    stage,
                    message,
                    phase=phase
                )

                # Wait for the specified time, checking for cancellation
                for _ in range(wait_seconds * 4):  # Check every 0.25s
                    if not progress_thread_active:
                        break
                    time.sleep(0.25)

        def run_conversion() -> None:
            """Run the actual conversion."""
            try:
                success = self._converter.convert_to_html(input_file, output_dir)
                conversion_result["success"] = success
            except Exception as e:
                conversion_result["error"] = str(e)
                conversion_result["success"] = False

        # Start progress updater thread
        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()

        # Start conversion thread
        conversion_thread = threading.Thread(target=run_conversion, daemon=True)
        conversion_thread.start()

        # Wait for conversion to complete
        conversion_thread.join()

        # Stop progress updater
        progress_thread_active = False

        if conversion_result["error"]:
            raise RuntimeError(conversion_result["error"])

        return bool(conversion_result["success"])



    def _update_job_status(
        self,
        job_id: str,
        status: ConversionStatus,
        progress: int,
        stage: str,
        message: str,
        phase: ConversionPhase | None = None,
        error: str | None = None
    ) -> None:
        """Update job status in tracking dictionary."""
        if job_id in self._jobs:
            update_data = {
                "status": status,
                "progress": progress,
                "stage": stage,
                "message": message,
                "error": error,
                "updated_at": time.time(),
            }
            if phase is not None:
                update_data["phase"] = phase
            self._jobs[job_id].update(update_data)

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
