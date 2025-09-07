"""
Unit tests for TemplateManager and TemplateCache.
"""

import time

import pytest

from services.github.template_manager import TemplateCache, TemplateManager


class TestTemplateCache:
    """Test cases for TemplateCache."""

    @pytest.fixture
    def template_cache(self):
        """Create a TemplateCache instance for testing."""
        return TemplateCache(ttl_seconds=10)  # Short TTL for testing

    def test_cache_initialization(self, template_cache):
        """Test cache initialization."""
        assert template_cache._ttl == 10
        assert len(template_cache._cache) == 0
        assert len(template_cache._cache_timestamps) == 0

    def test_cache_set_and_get(self, template_cache):
        """Test setting and getting cache data."""
        test_data = {"files": ["test.md"], "repo": "test/repo"}
        template_cache.set("test/repo", test_data)

        assert template_cache.is_cached("test/repo")
        cached_data = template_cache.get("test/repo")
        assert cached_data == test_data

    def test_cache_miss(self, template_cache):
        """Test cache miss for non-existent key."""
        assert not template_cache.is_cached("nonexistent/repo")
        assert template_cache.get("nonexistent/repo") is None

    def test_cache_expiration(self, template_cache):
        """Test cache expiration."""
        test_data = {"files": ["test.md"], "repo": "test/repo"}
        template_cache.set("test/repo", test_data)

        # Initially cached
        assert template_cache.is_cached("test/repo")

        # Simulate time passing beyond TTL by modifying the timestamp directly
        old_timestamp = time.time() - 15  # 15 seconds ago
        template_cache._cache_timestamps["test/repo"] = old_timestamp

        assert not template_cache.is_cached("test/repo")
        assert template_cache.get("test/repo") is None

    def test_clear_expired(self, template_cache):
        """Test clearing expired cache entries."""
        # Add some data
        template_cache.set("repo1", {"data": "test1"})
        template_cache.set("repo2", {"data": "test2"})

        assert len(template_cache._cache) == 2

        # Simulate time passing by modifying timestamps directly
        old_timestamp = time.time() - 15  # 15 seconds ago
        template_cache._cache_timestamps["repo1"] = old_timestamp
        template_cache._cache_timestamps["repo2"] = old_timestamp

        template_cache.clear_expired()

        # All entries should be cleared since they're expired
        assert len(template_cache._cache) == 0
        assert len(template_cache._cache_timestamps) == 0


class TestTemplateManager:
    """Test cases for TemplateManager."""

    @pytest.fixture
    def template_manager(self, mock_github_token):
        """Create a TemplateManager instance for testing."""
        return TemplateManager(mock_github_token)

    def test_template_manager_initialization(self, mock_github_token):
        """Test template manager initialization."""
        manager = TemplateManager(mock_github_token)

        assert manager.access_token == mock_github_token
        assert manager.base_url == "https://api.github.com"
        assert manager.headers["Authorization"] == f"token {mock_github_token}"
        assert manager.headers["Accept"] == "application/vnd.github.v3+json"
        assert manager.headers["User-Agent"] == "one-click-paper-page/0.1.0"
        assert isinstance(manager._template_cache, TemplateCache)

    def test_template_manager_has_required_methods(self, template_manager):
        """Test that template manager has all required methods."""
        assert hasattr(template_manager, 'get_template_content_cached')
        assert hasattr(template_manager, 'filter_essential_template_files')
        assert hasattr(template_manager, 'clear_cache')
        assert hasattr(template_manager, 'get_cache_info')

        # Check that methods are callable
        assert callable(template_manager.get_template_content_cached)
        assert callable(template_manager.filter_essential_template_files)
        assert callable(template_manager.clear_cache)
        assert callable(template_manager.get_cache_info)

    def test_filter_essential_template_files(self, template_manager):
        """Test filtering of template files."""
        mock_tree_items = [
            {"path": "_config.yml", "type": "blob"},
            {"path": "_layouts/default.html", "type": "blob"},
            {"path": "README.md", "type": "blob"},  # Should be skipped
            {"path": ".github/workflows/deploy.yml", "type": "blob"},
            {"path": "index.md", "type": "blob"},
            {"path": "scripts/build.py", "type": "blob"},  # Should be skipped
            {"path": "_posts/2023-01-01-test.md", "type": "blob"},
            {"path": "LICENSE", "type": "blob"},  # Should be skipped
        ]

        filtered_files = template_manager.filter_essential_template_files(mock_tree_items)

        # Check that essential files are included
        filtered_paths = [f["path"] for f in filtered_files]
        assert "_config.yml" in filtered_paths
        assert "_layouts/default.html" in filtered_paths
        assert ".github/workflows/deploy.yml" in filtered_paths
        assert "index.md" in filtered_paths
        assert "_posts/2023-01-01-test.md" in filtered_paths

        # Check that skipped files are not included
        assert "README.md" not in filtered_paths
        assert "scripts/build.py" not in filtered_paths
        assert "LICENSE" not in filtered_paths

    def test_get_cache_info(self, template_manager):
        """Test getting cache information."""
        cache_info = template_manager.get_cache_info()

        assert "cached_templates" in cache_info
        assert "cache_size" in cache_info
        assert "ttl_seconds" in cache_info
        assert isinstance(cache_info["cached_templates"], list)
        assert isinstance(cache_info["cache_size"], int)
        assert isinstance(cache_info["ttl_seconds"], int)

    def test_clear_cache(self, template_manager):
        """Test clearing the cache."""
        # Add some test data to cache
        template_manager._template_cache.set("test/repo", {"data": "test"})
        assert template_manager._template_cache.is_cached("test/repo")

        # Clear cache
        template_manager.clear_cache()

        # Cache should still have the data since it's not expired
        assert template_manager._template_cache.is_cached("test/repo")

    def test_template_manager_methods_exist_and_are_async(self, template_manager):
        """Test that required async methods exist."""
        import inspect

        # Check async methods
        async_methods = ['get_template_content_cached']

        for method_name in async_methods:
            method = getattr(template_manager, method_name)
            assert callable(method)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

        # Check sync methods
        sync_methods = ['filter_essential_template_files', 'clear_cache', 'get_cache_info']

        for method_name in sync_methods:
            method = getattr(template_manager, method_name)
            assert callable(method)
            assert not inspect.iscoroutinefunction(method), f"{method_name} should be sync"
