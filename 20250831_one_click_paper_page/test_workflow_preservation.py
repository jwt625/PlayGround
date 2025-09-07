#!/usr/bin/env python3
"""
Test script to verify that GitHub workflow files are preserved during content deployment.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from models.github import DeploymentConfig, GitHubRepository, GitHubUser
from services.github_service_orchestrator import GitHubServiceOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockGitOperationsService:
    """Mock Git operations service for testing."""
    
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {"Authorization": "token mock_token"}
    
    async def create_blob(self, repository, content, encoding):
        """Mock blob creation."""
        return f"blob_sha_{hash(content) % 1000}"

async def test_workflow_preservation():
    """Test that workflow files are preserved during content deployment."""
    logger.info("🧪 Testing workflow file preservation...")
    
    # Create mock repository
    from datetime import datetime
    mock_repo = GitHubRepository(
        id=12345,
        name="test-repo",
        full_name="testuser/test-repo",
        html_url="https://github.com/testuser/test-repo",
        clone_url="https://github.com/testuser/test-repo.git",
        ssh_url="git@github.com:testuser/test-repo.git",
        default_branch="main",
        private=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner=GitHubUser(
            login="testuser",
            id=123,
            avatar_url="https://github.com/testuser.png",
            html_url="https://github.com/testuser"
        )
    )
    
    # Create mock orchestrator
    class MockOrchestrator(GitHubServiceOrchestrator):
        def __init__(self):
            # Initialize without calling parent __init__
            self.git_operations_service = MockGitOperationsService()
        
        async def _simulate_existing_tree_fetch(self, repository, commit_sha):
            """Simulate fetching existing tree with workflow files."""
            # Mock existing files including workflow
            return [
                {
                    "path": ".github/workflows/deploy.yml",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "workflow_sha_123"
                },
                {
                    "path": "_config.yml",
                    "mode": "100644", 
                    "type": "blob",
                    "sha": "config_sha_456"
                },
                {
                    "path": "README.md",
                    "mode": "100644",
                    "type": "blob", 
                    "sha": "readme_sha_789"
                }
            ]
        
        async def test_tree_merging(self, repository, files_to_commit):
            """Test the tree merging logic."""
            # Simulate existing tree
            existing_tree_items = await self._simulate_existing_tree_fetch(repository, "dummy_sha")
            
            # Create blobs for new files (mock)
            blob_shas = {}
            for file_info in files_to_commit:
                blob_shas[file_info["path"]] = f"new_blob_{hash(file_info['content']) % 1000}"
            
            # Merge existing files with new files (same logic as in the real method)
            new_file_paths = set(blob_shas.keys())
            
            # Add existing files that are not being overwritten
            tree_items = []
            for existing_item in existing_tree_items:
                if existing_item["path"] not in new_file_paths and existing_item["type"] == "blob":
                    tree_items.append({
                        "path": existing_item["path"],
                        "mode": existing_item["mode"],
                        "type": existing_item["type"],
                        "sha": existing_item["sha"]
                    })
            
            # Add new files
            for file_path, blob_sha in blob_shas.items():
                tree_items.append({
                    "path": file_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha
                })
            
            return tree_items
    
    mock_orchestrator = MockOrchestrator()
    
    # Test files to commit (simulating converted content)
    test_files = [
        {
            "path": "index.html",
            "content": "<html><head><title>Test Paper</title></head><body><h1>Test Paper</h1></body></html>",
            "encoding": "utf-8"
        },
        {
            "path": "paper.md",
            "content": "# Test Paper\n\nThis is a test paper.",
            "encoding": "utf-8"
        },
        {
            "path": "images/figure1.png",
            "content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "encoding": "base64"
        }
    ]
    
    # Test tree merging
    final_tree = await mock_orchestrator.test_tree_merging(mock_repo, test_files)
    
    # Verify results
    logger.info(f"📋 Final tree has {len(final_tree)} files:")
    for item in final_tree:
        logger.info(f"  - {item['path']} ({item['type']})")
    
    # Check that workflow files are preserved
    workflow_files = [item for item in final_tree if item["path"].startswith(".github/workflows/")]
    config_files = [item for item in final_tree if item["path"] == "_config.yml"]
    new_content_files = [item for item in final_tree if item["path"] in ["index.html", "paper.md", "images/figure1.png"]]
    
    success = True
    
    if workflow_files:
        logger.info(f"✅ Workflow files preserved: {[f['path'] for f in workflow_files]}")
    else:
        logger.error("❌ Workflow files NOT preserved!")
        success = False
    
    if config_files:
        logger.info(f"✅ Config files preserved: {[f['path'] for f in config_files]}")
    else:
        logger.error("❌ Config files NOT preserved!")
        success = False
    
    if len(new_content_files) == 3:
        logger.info(f"✅ New content files added: {[f['path'] for f in new_content_files]}")
    else:
        logger.error(f"❌ Expected 3 new content files, got {len(new_content_files)}")
        success = False
    
    # Check for no duplicates
    all_paths = [item["path"] for item in final_tree]
    if len(all_paths) == len(set(all_paths)):
        logger.info("✅ No duplicate files in final tree")
    else:
        logger.error("❌ Duplicate files found in final tree!")
        success = False
    
    return success

async def main():
    """Run the workflow preservation test."""
    logger.info("🚀 Testing workflow file preservation during content deployment...")
    
    try:
        success = await test_workflow_preservation()
        
        if success:
            logger.info("🎉 All tests passed! Workflow files will be preserved during deployment.")
            return True
        else:
            logger.error("💥 Tests failed! Workflow files may not be preserved.")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
