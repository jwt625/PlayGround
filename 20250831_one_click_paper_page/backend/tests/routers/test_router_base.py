"""
Tests for router base utilities and shared dependencies.
"""

import pytest
from fastapi import HTTPException
from unittest.mock import Mock, patch

from routers import get_github_service, handle_router_error, RouterBase


class TestGetGitHubService:
    """Test the shared GitHub service dependency."""
    
    def test_valid_bearer_token(self):
        """Test with valid Bearer token."""
        with patch('routers.GitHubService') as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance
            
            result = get_github_service("Bearer test_token_123")
            
            mock_service.assert_called_once_with("test_token_123")
            assert result == mock_instance
    
    def test_invalid_authorization_format(self):
        """Test with invalid authorization header format."""
        with pytest.raises(HTTPException) as exc_info:
            get_github_service("Invalid token_123")
        
        assert exc_info.value.status_code == 401
        assert "Invalid authorization header format" in exc_info.value.detail
    
    def test_missing_bearer_prefix(self):
        """Test with missing Bearer prefix."""
        with pytest.raises(HTTPException) as exc_info:
            get_github_service("token_123")
        
        assert exc_info.value.status_code == 401
        assert "Expected 'Bearer <token>'" in exc_info.value.detail


class TestHandleRouterError:
    """Test the standardized error handling function."""
    
    def test_http_exception_passthrough(self):
        """Test that HTTPException is returned as-is."""
        original_error = HTTPException(status_code=404, detail="Not found")
        
        result = handle_router_error("test_operation", original_error)
        
        assert result == original_error
        assert result.status_code == 404
        assert result.detail == "Not found"
    
    def test_generic_exception_handling(self):
        """Test handling of generic exceptions."""
        original_error = ValueError("Something went wrong")
        
        with patch('routers.logger') as mock_logger:
            result = handle_router_error("test_operation", original_error)
        
        # Check logging
        mock_logger.error.assert_called_once_with("test_operation failed: Something went wrong")
        
        # Check returned exception
        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert "test_operation failed: Something went wrong" in result.detail


class TestRouterBase:
    """Test the RouterBase class."""
    
    def test_initialization(self):
        """Test router base initialization."""
        router = RouterBase("test_router")
        
        assert router.name == "test_router"
        assert router.logger.name == "routers.test_router"
    
    def test_log_operation_without_details(self):
        """Test logging operation without details."""
        router = RouterBase("test_router")
        
        with patch.object(router.logger, 'info') as mock_info:
            router.log_operation("create_user")
        
        mock_info.assert_called_once_with("[test_router] create_user")
    
    def test_log_operation_with_details(self):
        """Test logging operation with details."""
        router = RouterBase("test_router")
        
        with patch.object(router.logger, 'info') as mock_info:
            router.log_operation("create_user", "user_id=123")
        
        mock_info.assert_called_once_with("[test_router] create_user: user_id=123")
    
    def test_handle_error_with_http_exception(self):
        """Test error handling with HTTPException."""
        router = RouterBase("test_router")
        original_error = HTTPException(status_code=400, detail="Bad request")
        
        result = router.handle_error("create_user", original_error)
        
        assert result == original_error
    
    def test_handle_error_with_generic_exception(self):
        """Test error handling with generic exception."""
        router = RouterBase("test_router")
        original_error = RuntimeError("Database connection failed")
        
        with patch('routers.logger') as mock_logger:
            result = router.handle_error("create_user", original_error)
        
        # Check logging
        mock_logger.error.assert_called_once_with(
            "[test_router] create_user failed: Database connection failed"
        )
        
        # Check returned exception
        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert "[test_router] create_user failed: Database connection failed" in result.detail
