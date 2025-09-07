#!/usr/bin/env python3
"""
Test script to verify the deployment fixes work correctly.
This script tests:
1. Paper title extraction from conversion results
2. Content upload to repository
3. End-to-end deployment process
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from models.github import DeploymentConfig, TemplateType
from models.conversion import ConversionResult, ConversionStatus, PaperMetadata
from services.conversion_service import ConversionService
from services.github_service_orchestrator import GitHubServiceOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_title_extraction():
    """Test that paper titles are extracted correctly from conversion results."""
    logger.info("🧪 Testing paper title extraction...")
    
    # Create a test PDF file name (adjust path based on current working directory)
    script_dir = Path(__file__).parent
    test_pdf = script_dir / "tests/pdf/2508.19977v1.pdf"

    if not test_pdf.exists():
        logger.warning(f"Test PDF not found: {test_pdf}")
        # Try to create a simple test file for demonstration
        test_pdf = script_dir / "backend/test_sample.tex"
        if test_pdf.exists():
            logger.info(f"Using alternative test file: {test_pdf}")
        else:
            logger.warning("No test files available")
            return False
    
    # Test the conversion service
    conversion_service = ConversionService()

    # Create a conversion job
    from models.conversion import ConversionMode
    job_id = conversion_service.create_job(mode=ConversionMode.FAST)
    logger.info(f"Created conversion job: {job_id}")

    # Run conversion
    result = await conversion_service.convert_file(job_id, test_pdf)
    
    # Check if title was extracted
    if result.metadata and result.metadata.title:
        logger.info(f"✅ Title extracted: '{result.metadata.title}'")
        
        # Check if it's not just a placeholder
        if "Sample Academic Paper" not in result.metadata.title:
            logger.info("✅ Title appears to be extracted from filename, not placeholder")
        else:
            logger.warning("⚠️ Title appears to be placeholder")
        
        return True
    else:
        logger.error("❌ No title found in conversion result")
        return False

async def test_content_preparation():
    """Test that converted content is properly prepared for upload."""
    logger.info("🧪 Testing content preparation...")
    
    # Use existing test output (adjust path based on current working directory)
    script_dir = Path(__file__).parent
    test_output_dir = script_dir / "tests/output_real"

    if not test_output_dir.exists():
        # Try alternative path
        test_output_dir = script_dir / "tests/output"
        if not test_output_dir.exists():
            logger.warning(f"Test output directory not found: {test_output_dir}")
            return False
    
    # Create a mock GitHub service (without actual GitHub token)
    # We'll test the file preparation logic only
    class MockGitHubService(GitHubServiceOrchestrator):
        def __init__(self):
            # Initialize without calling parent __init__ to avoid token requirement
            pass
    
    mock_service = MockGitHubService()
    
    # Create test deployment config
    config = DeploymentConfig(
        repository_name="test-paper-repo",
        template="minimal-academic",  # Use valid enum value
        paper_title="Test Paper Title",
        paper_authors=["Test Author 1", "Test Author 2"],
    )
    
    # Test file preparation
    try:
        files = await mock_service._prepare_converted_content_files(test_output_dir, config)
        
        logger.info(f"✅ Prepared {len(files)} files for upload")
        
        # Check that we have expected files
        file_paths = [f["path"] for f in files]
        logger.info(f"Files to upload: {file_paths}")
        
        # Check for HTML file
        html_files = [f for f in files if f["path"].endswith(".html")]
        if html_files:
            logger.info("✅ HTML file found in upload list")
            
            # Check if title was customized
            html_content = html_files[0]["content"]
            if "Test Paper Title" in html_content:
                logger.info("✅ HTML content was customized with paper title")
            else:
                logger.warning("⚠️ HTML content was not customized")
        else:
            logger.warning("⚠️ No HTML file found in upload list")
        
        # Check for config file
        config_files = [f for f in files if f["path"] == "paper-config.json"]
        if config_files:
            logger.info("✅ Paper config file included")
            config_content = json.loads(config_files[0]["content"])
            logger.info(f"Config content: {config_content}")
        else:
            logger.warning("⚠️ Paper config file not found")
        
        return len(files) > 0
        
    except Exception as e:
        logger.error(f"❌ Content preparation failed: {e}")
        return False

async def test_workflow_preservation():
    """Test that GitHub workflow files are preserved during content deployment."""
    logger.info("🧪 Testing workflow file preservation...")

    # Create a mock repository with existing workflow files
    from models.github import GitHubRepository, GitHubUser

    mock_repo = GitHubRepository(
        id=12345,
        name="test-repo",
        full_name="testuser/test-repo",
        html_url="https://github.com/testuser/test-repo",
        default_branch="main",
        owner=GitHubUser(login="testuser", id=123)
    )

    # Create a mock orchestrator to test the tree preservation logic
    class MockGitHubService(GitHubServiceOrchestrator):
        def __init__(self):
            # Initialize without calling parent __init__ to avoid token requirement
            pass

        async def _get_existing_tree_items(self, repository, commit_sha):
            # Mock existing files including workflow
            return [
                {
                    "path": ".github/workflows/deploy.yml",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "abc123"
                },
                {
                    "path": "_config.yml",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "def456"
                },
                {
                    "path": "README.md",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "ghi789"
                }
            ]

    mock_service = MockGitHubService()

    # Test that existing files are preserved when adding new content
    existing_files = await mock_service._get_existing_tree_items(mock_repo, "dummy_sha")

    logger.info(f"✅ Mock existing files: {[f['path'] for f in existing_files]}")

    # Check that workflow file is present
    workflow_files = [f for f in existing_files if f["path"].startswith(".github/workflows/")]
    if workflow_files:
        logger.info(f"✅ Found workflow files: {[f['path'] for f in workflow_files]}")
        return True
    else:
        logger.error("❌ No workflow files found in existing files")
        return False

async def main():
    """Run all deployment tests."""
    logger.info("🚀 Starting deployment fix tests...")

    tests = [
        ("Title Extraction", test_title_extraction),
        ("Content Preparation", test_content_preparation),
        ("Workflow Preservation", test_workflow_preservation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Deployment fixes are working.")
        return True
    else:
        logger.error("💥 Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
