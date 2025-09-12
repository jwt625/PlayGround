"""
Unit tests for the conversion service.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from models.conversion import ConversionMode, ConversionStatus
from services.conversion_service import ConversionService


class TestConversionService:
    """Test cases for ConversionService."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def conversion_service(self, temp_dir: Path) -> ConversionService:
        """Create a ConversionService instance for testing."""
        return ConversionService(temp_dir=temp_dir)

    def test_create_job(self, conversion_service: ConversionService) -> None:
        """Test job creation."""
        job_id = conversion_service.create_job(ConversionMode.AUTO)

        assert job_id is not None
        assert len(job_id) > 0

        # Check job status
        job_status = conversion_service.get_job_status(job_id)
        assert job_status is not None
        assert job_status["status"] == ConversionStatus.QUEUED
        assert job_status["mode"] == ConversionMode.AUTO
        assert job_status["progress"] == 0
        assert job_status["stage"] == "initialized"

    def test_get_job_status_not_found(self, conversion_service):
        """Test getting status for non-existent job."""
        job_status = conversion_service.get_job_status("non-existent-job")
        assert job_status is None

    def test_cleanup_job(self, conversion_service):
        """Test job cleanup."""
        job_id = conversion_service.create_job(ConversionMode.FAST)

        # Verify job exists
        assert conversion_service.get_job_status(job_id) is not None

        # Cleanup job
        success = conversion_service.cleanup_job(job_id)
        assert success is True

        # Verify job is removed
        assert conversion_service.get_job_status(job_id) is None

    def test_cleanup_job_not_found(self, conversion_service):
        """Test cleanup for non-existent job."""
        success = conversion_service.cleanup_job("non-existent-job")
        assert success is False

    @pytest.mark.asyncio
    async def test_convert_file_job_not_found(self, conversion_service, temp_dir):
        """Test conversion with non-existent job."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("fake pdf content")

        with pytest.raises(ValueError, match="Job .* not found"):
            await conversion_service.convert_file("non-existent-job", test_file)

    @pytest.mark.asyncio
    async def test_convert_file_with_progress_callback(
        self, conversion_service, temp_dir
    ):
        """Test conversion with progress callback."""
        # Create a test file
        test_file = temp_dir / "test.pdf"
        test_file.write_text("fake pdf content")

        # Create job
        job_id = conversion_service.create_job(ConversionMode.FAST)

        # Mock progress callback
        progress_callback = AsyncMock()

        # Mock the marker converter to not be available (use placeholder)
        with patch.object(conversion_service, '_converter', None):
            result = await conversion_service.convert_file(
                job_id, test_file, progress_callback
            )

        # Verify progress callback was called
        assert progress_callback.call_count >= 3  # At least start, middle, end

        # Verify result
        assert result.status == ConversionStatus.COMPLETED

    def test_update_job_status(self, conversion_service):
        """Test job status updates."""
        job_id = conversion_service.create_job(ConversionMode.QUALITY)

        # Update job status
        conversion_service._update_job_status(
            job_id,
            ConversionStatus.PROCESSING,
            50,
            "converting",
            "Converting document",
        )

        # Verify update
        job_status = conversion_service.get_job_status(job_id)
        assert job_status["status"] == ConversionStatus.PROCESSING
        assert job_status["progress"] == 50
        assert job_status["stage"] == "converting"
        assert job_status["message"] == "Converting document"

    @pytest.mark.asyncio
    async def test_convert_file_error_handling(self, conversion_service, temp_dir):
        """Test error handling during conversion."""
        # Create a test file
        test_file = temp_dir / "test.pdf"
        test_file.write_text("fake pdf content")

        # Create job
        job_id = conversion_service.create_job(ConversionMode.AUTO)

        # Mock the conversion to raise an exception
        with patch.object(
            conversion_service,
            '_run_conversion',
            side_effect=RuntimeError("Test error")
        ):
            with pytest.raises(RuntimeError, match="Conversion failed: Test error"):
                await conversion_service.convert_file(job_id, test_file)

        # Verify job status was updated to failed
        job_status = conversion_service.get_job_status(job_id)
        assert job_status["status"] == ConversionStatus.FAILED
        assert job_status["error"] == "Test error"

    def test_multiple_jobs(self, conversion_service):
        """Test handling multiple concurrent jobs."""
        # Create multiple jobs
        job_ids = []
        for i in range(3):
            job_id = conversion_service.create_job(ConversionMode.AUTO)
            job_ids.append(job_id)

        # Verify all jobs exist and are independent
        for i, job_id in enumerate(job_ids):
            job_status = conversion_service.get_job_status(job_id)
            assert job_status is not None
            assert job_status["status"] == ConversionStatus.QUEUED

            # Update one job status
            conversion_service._update_job_status(
                job_id,
                ConversionStatus.PROCESSING,
                i * 10,
                f"stage_{i}",
                f"message_{i}"
            )

        # Verify updates are independent
        for i, job_id in enumerate(job_ids):
            job_status = conversion_service.get_job_status(job_id)
            assert job_status["progress"] == i * 10
            assert job_status["stage"] == f"stage_{i}"
            assert job_status["message"] == f"message_{i}"


if __name__ == "__main__":
    pytest.main([__file__])
