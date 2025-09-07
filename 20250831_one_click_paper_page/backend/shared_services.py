"""
Shared service instances for the application.

This module provides singleton instances of services that need to be shared
across different parts of the application (routers, main app, etc.).
"""

from services.conversion_service import ConversionService

# Shared conversion service instance
# This ensures that all parts of the application use the same service instance
# and can access the same job storage
conversion_service = ConversionService()
