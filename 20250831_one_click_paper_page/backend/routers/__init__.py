"""
Router utilities and shared dependencies for API endpoints.

This module provides common patterns and utilities used across all router modules
to ensure consistency and reduce code duplication.
"""

import logging
from typing import Any

from fastapi import Header, HTTPException

from services.github_service import GitHubService

# Setup logging
logger = logging.getLogger(__name__)


def get_github_service(authorization: str = Header(..., description="Bearer token")) -> GitHubService:
    """
    Shared dependency for GitHub authentication.
    
    Args:
        authorization: Authorization header with Bearer token
        
    Returns:
        Authenticated GitHubService instance
        
    Raises:
        HTTPException: If authorization header is invalid
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Expected 'Bearer <token>'"
        )

    token = authorization.replace("Bearer ", "")
    return GitHubService(token)


def handle_router_error(operation_name: str, error: Exception) -> HTTPException:
    """
    Standardized error handling for router operations.
    
    Args:
        operation_name: Name of the operation that failed
        error: The exception that occurred
        
    Returns:
        HTTPException with appropriate status code and message
    """
    # Re-raise HTTP exceptions as-is
    if isinstance(error, HTTPException):
        return error

    # Log the error for debugging
    logger.error(f"{operation_name} failed: {error}")

    # Return generic 500 error for unexpected exceptions
    return HTTPException(
        status_code=500,
        detail=f"{operation_name} failed: {str(error)}"
    )


class RouterBase:
    """
    Base class for router modules providing common functionality.
    
    This class can be extended by router modules to inherit shared
    error handling and logging patterns.
    """

    def __init__(self, name: str):
        """
        Initialize router base.
        
        Args:
            name: Name of the router for logging purposes
        """
        self.name = name
        self.logger = logging.getLogger(f"routers.{name}")

    def log_operation(self, operation: str, details: str = "") -> None:
        """
        Log router operation with consistent formatting.
        
        Args:
            operation: Name of the operation
            details: Additional details to log
        """
        message = f"[{self.name}] {operation}"
        if details:
            message += f": {details}"
        self.logger.info(message)

    def handle_error(self, operation: str, error: Exception) -> HTTPException:
        """
        Handle router operation error with logging.
        
        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
            
        Returns:
            HTTPException with appropriate status code and message
        """
        return handle_router_error(f"[{self.name}] {operation}", error)
