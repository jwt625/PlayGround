"""
Tests for conversion router.
"""

from io import BytesIO
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.conversion import ConversionMode
from routers.conversion_router import router

# Create test app with conversion router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestUploadAndConvert:
    """Test file upload and conversion endpoint."""

    @patch('routers.conversion_router.conversion_service')
    @patch('routers.conversion_router.aiofiles.open')
    def test_successful_upload_pdf(self, mock_aiofiles, mock_service):
        """Test successful PDF upload and conversion."""
        # Mock conversion service
        mock_service.create_job.return_value = "test_job_123"
        from pathlib import Path
        mock_service.get_job_status.return_value = {
            "status": "queued",
            "job_dir": Path("/tmp/test_job_123"),
            "phase": "queued",
            "stage": "initializing",
            "message": "Job created"
        }

        # Mock file operations
        mock_file = AsyncMock()
        mock_aiofiles.return_value.__aenter__.return_value = mock_file

        # Create test file
        test_file = BytesIO(b"fake pdf content")

        response = client.post(
            "/api/convert/upload",
            files={"file": ("test.pdf", test_file, "application/pdf")},
            data={
                "template": "minimal-academic",
                "mode": "auto",
                "auto_deploy": "false"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_job_123"
        assert data["status"] == "queued"
        assert "Conversion started with" in data["message"]
        assert "mode" in data["message"]

        # Verify service calls
        mock_service.create_job.assert_called_once_with(ConversionMode.AUTO)
        mock_service.get_job_status.assert_called_once_with("test_job_123")

    def test_invalid_file_type(self):
        """Test upload with invalid file type."""
        test_file = BytesIO(b"fake content")

        response = client.post(
            "/api/convert/upload",
            files={"file": ("test.txt", test_file, "text/plain")},
            data={
                "template": "minimal-academic",
                "mode": "auto"
            }
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
        assert "Only PDF and DOCX are supported" in response.json()["detail"]

    def test_no_file_provided(self):
        """Test upload without file."""
        response = client.post(
            "/api/convert/upload",
            data={
                "template": "minimal-academic",
                "mode": "auto"
            }
        )

        assert response.status_code == 422  # FastAPI validation error
        assert "field required" in response.json()["detail"][0]["msg"].lower()

    @patch('routers.conversion_router.conversion_service')
    @patch('routers.conversion_router.aiofiles.open')
    def test_auto_deploy_without_token(self, mock_aiofiles, mock_service):
        """Test auto deploy without authorization token."""
        # Mock conversion service
        mock_service.create_job.return_value = "test_job_123"
        from pathlib import Path
        mock_service.get_job_status.return_value = {
            "status": "pending",
            "job_dir": Path("/tmp/test_job_123"),
            "phase": "queued",
            "stage": "initializing",
            "message": "Job created"
        }

        test_file = BytesIO(b"fake pdf content")

        response = client.post(
            "/api/convert/upload",
            files={"file": ("test.pdf", test_file, "application/pdf")},
            data={
                "template": "minimal-academic",
                "mode": "auto",
                "auto_deploy": "true"
            }
        )

        assert response.status_code == 400
        assert "GitHub OAuth token required for auto-deployment" in response.json()["detail"]


class TestGetConversionStatus:
    """Test conversion status endpoint."""

    @patch('routers.conversion_router.conversion_service')
    def test_get_status_success(self, mock_service):
        """Test successful status retrieval."""
        mock_service.get_job_status.return_value = {
            "status": "processing",
            "phase": "analyzing",
            "stage": "extracting_text",
            "message": "Extracting text from PDF",
            "error": None
        }

        response = client.get("/api/convert/status/test_job_123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_job_123"
        assert data["status"] == "processing"
        assert data["phase"] == "analyzing"
        assert data["stage"] == "extracting_text"
        assert data["message"] == "Extracting text from PDF"
        assert data["error"] is None

    @patch('routers.conversion_router.conversion_service')
    def test_get_status_job_not_found(self, mock_service):
        """Test status retrieval for non-existent job."""
        mock_service.get_job_status.return_value = None

        response = client.get("/api/convert/status/nonexistent_job")

        assert response.status_code == 404
        assert "Job nonexistent_job not found" in response.json()["detail"]


class TestGetConversionResult:
    """Test conversion result endpoint."""

    @patch('routers.conversion_router.conversion_service')
    def test_get_result_success(self, mock_service):
        """Test successful result retrieval."""
        from models.conversion import ConversionMode, ConversionStatus
        mock_result = {
            "job_id": "test_job_123",
            "status": ConversionStatus.COMPLETED,
            "success": True,
            "output_dir": "/tmp/test_job_123/output",
            "output_files": ["index.html", "style.css"],
            "metrics": {
                "total_conversion_time": 45.2,
                "mode_used": ConversionMode.AUTO,
                "quality_assessment": {
                    "has_good_text": True,
                    "recommended_mode": ConversionMode.AUTO,
                    "confidence": "high",
                    "avg_chars_per_page": 2500.0,
                    "text_coverage": 95.5
                }
            },
            "markdown_length": 5000,
            "image_count": 3,
            "html_file": "/tmp/test_job_123/output/index.html",
            "markdown_file": "/tmp/test_job_123/output/paper.md"
        }

        mock_service.get_job_status.return_value = {
            "status": "completed",
            "result": mock_result
        }

        response = client.get("/api/convert/result/test_job_123")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["output_dir"] == "/tmp/test_job_123/output"
        assert "index.html" in data["output_files"]

    @patch('routers.conversion_router.conversion_service')
    def test_get_result_job_not_completed(self, mock_service):
        """Test result retrieval for incomplete job."""
        mock_service.get_job_status.return_value = {
            "status": "processing"
        }

        response = client.get("/api/convert/result/test_job_123")

        assert response.status_code == 400
        assert "is not completed" in response.json()["detail"]
        assert "Current status: processing" in response.json()["detail"]

    @patch('routers.conversion_router.conversion_service')
    def test_get_result_no_result_available(self, mock_service):
        """Test result retrieval when no result is available."""
        mock_service.get_job_status.return_value = {
            "status": "completed",
            "result": None
        }

        response = client.get("/api/convert/result/test_job_123")

        assert response.status_code == 500
        assert "completed but no result available" in response.json()["detail"]


class TestCancelConversion:
    """Test conversion cancellation endpoint."""

    @patch('routers.conversion_router.conversion_service')
    def test_cancel_success(self, mock_service):
        """Test successful job cancellation."""
        mock_service.get_job_status.return_value = {
            "status": "processing"
        }
        mock_service.cleanup_job.return_value = True

        response = client.delete("/api/convert/cancel/test_job_123")

        assert response.status_code == 200
        assert "cancelled and cleaned up successfully" in response.json()["message"]

        mock_service.cleanup_job.assert_called_once_with("test_job_123")

    @patch('routers.conversion_router.conversion_service')
    def test_cancel_job_not_found(self, mock_service):
        """Test cancellation of non-existent job."""
        mock_service.get_job_status.return_value = None

        response = client.delete("/api/convert/cancel/nonexistent_job")

        assert response.status_code == 404
        assert "Job nonexistent_job not found" in response.json()["detail"]

    @patch('routers.conversion_router.conversion_service')
    def test_cancel_cleanup_failure(self, mock_service):
        """Test cancellation when cleanup fails."""
        mock_service.get_job_status.return_value = {
            "status": "processing"
        }
        mock_service.cleanup_job.return_value = False

        response = client.delete("/api/convert/cancel/test_job_123")

        assert response.status_code == 500
        assert "Failed to cleanup job" in response.json()["detail"]


class TestConversionRouterIntegration:
    """Test conversion router integration."""

    def test_router_prefix_and_tags(self):
        """Test that router has correct prefix and tags."""
        assert router.prefix == "/api/convert"
        assert "conversion" in router.tags

    def test_endpoints_registered(self):
        """Test that all endpoints are registered."""
        routes = [route.path for route in router.routes]
        assert "/api/convert/upload" in routes
        assert "/api/convert/status/{job_id}" in routes
        assert "/api/convert/result/{job_id}" in routes
        assert "/api/convert/cancel/{job_id}" in routes
