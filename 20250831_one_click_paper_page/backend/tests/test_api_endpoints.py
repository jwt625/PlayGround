"""
Integration tests for API endpoints.
"""

import io

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "One-Click Paper Page API is running"}

    def test_upload_and_convert_invalid_file_type(self):
        """Test upload with invalid file type."""
        # Create a fake text file
        fake_file = io.BytesIO(b"fake content")

        response = client.post(
            "/api/convert/upload",
            files={"file": ("test.txt", fake_file, "text/plain")},
            data={"template": "academic-pages", "mode": "auto"}
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_and_convert_no_file(self):
        """Test upload without file."""
        response = client.post(
            "/api/convert/upload",
            data={"template": "academic-pages", "mode": "auto"}
        )

        assert response.status_code == 422  # Validation error

    def test_upload_and_convert_pdf_success(self):
        """Test successful PDF upload and conversion."""
        # Create a fake PDF file
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content")

        response = client.post(
            "/api/convert/upload",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            data={"template": "academic-pages", "mode": "fast"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "FAST mode" in data["message"]

        return data["job_id"]

    def test_get_conversion_status_not_found(self):
        """Test getting status for non-existent job."""
        response = client.get("/api/convert/status/non-existent-job")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_conversion_status_success(self):
        """Test getting status for existing job."""
        # First create a job
        job_id = self.test_upload_and_convert_pdf_success()

        # Get status
        response = client.get(f"/api/convert/status/{job_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress" in data
        assert "stage" in data
        assert "message" in data

    def test_get_conversion_result_not_found(self):
        """Test getting result for non-existent job."""
        response = client.get("/api/convert/result/non-existent-job")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_conversion_result_not_completed(self):
        """Test getting result for non-completed job."""
        # Create a job
        job_id = self.test_upload_and_convert_pdf_success()

        # Try to get result immediately (should not be completed)
        response = client.get(f"/api/convert/result/{job_id}")
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]

    def test_cancel_conversion_not_found(self):
        """Test cancelling non-existent job."""
        response = client.delete("/api/convert/cancel/non-existent-job")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_cancel_conversion_success(self):
        """Test successful job cancellation."""
        # Create a job
        job_id = self.test_upload_and_convert_pdf_success()

        # Cancel the job
        response = client.delete(f"/api/convert/cancel/{job_id}")
        assert response.status_code == 200
        assert "cancelled and cleaned up successfully" in response.json()["message"]

        # Verify job is gone
        response = client.get(f"/api/convert/status/{job_id}")
        assert response.status_code == 404

    def test_upload_docx_file(self):
        """Test uploading DOCX file."""
        # Create a fake DOCX file
        fake_docx = io.BytesIO(b"PK fake docx content")

        response = client.post(
            "/api/convert/upload",
            files={
                "file": (
                    "test.docx",
                    fake_docx,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            data={"template": "academic-pages", "mode": "quality"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "QUALITY mode" in data["message"]

    def test_conversion_modes(self):
        """Test different conversion modes."""
        modes = ["auto", "fast", "quality"]

        for mode in modes:
            fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content")

            response = client.post(
                "/api/convert/upload",
                files={"file": (f"test_{mode}.pdf", fake_pdf, "application/pdf")},
                data={"template": "academic-pages", "mode": mode}
            )

            assert response.status_code == 200
            data = response.json()
            assert f"{mode.upper()} mode" in data["message"]

            # Clean up
            job_id = data["job_id"]
            client.delete(f"/api/convert/cancel/{job_id}")

    def test_oauth_endpoints_still_work(self):
        """Test that existing OAuth endpoints still work."""
        # Test OAuth token endpoint (should fail without proper data)
        response = client.post(
            "/api/github/oauth/token",
            json={"code": "test", "redirect_uri": "http://localhost:5173/auth/callback"}
        )

        # Should fail due to validation or missing GitHub client secret,
        # but endpoint should exist
        assert response.status_code in [400, 500]
        if response.status_code == 500:
            assert "client secret not configured" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_full_conversion_workflow_real_pdf(self):
        """Test full conversion workflow using real PDF file."""
        import asyncio
        from pathlib import Path

        # Use actual PDF from test folder
        test_pdf_path = Path(__file__).parent.parent.parent / "tests" / "pdf" / "attention_is_all_you_need.pdf"
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"

        # Read the real PDF file
        with open(test_pdf_path, "rb") as f:
            pdf_content = f.read()

        response = client.post(
            "/api/convert/upload",
            files={"file": ("attention_is_all_you_need.pdf", io.BytesIO(pdf_content), "application/pdf")},
            data={"template": "minimal-academic", "mode": "fast"}
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for background processing (real PDF takes longer)
        await asyncio.sleep(5)

        # Check status
        response = client.get(f"/api/convert/status/{job_id}")
        assert response.status_code == 200

        status_data = response.json()
        print(f"Job status: {status_data}")

        # If completed, get result
        if status_data["status"] == "completed":
            response = client.get(f"/api/convert/result/{job_id}")
            assert response.status_code == 200

            result_data = response.json()
            assert result_data["job_id"] == job_id
            assert result_data["status"] == "completed"
            assert len(result_data["output_files"]) >= 2
            assert result_data["markdown_length"] > 0
            assert result_data["success"] is True

        # Clean up
        client.delete(f"/api/convert/cancel/{job_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
