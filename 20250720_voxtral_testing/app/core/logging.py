"""Logging configuration for the FastAPI application."""

import logging
import sys
from typing import Any, Dict

from .config import config


def setup_logging() -> None:
    """Configure application logging."""
    log_level = logging.DEBUG if config.debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("fastapi").setLevel(log_level)
    logging.getLogger("voxtral").setLevel(log_level)
    
    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LoggingMiddleware:
    """Middleware for request/response logging."""
    
    def __init__(self, app) -> None:
        self.app = app
        self.logger = get_logger("app.middleware")
    
    async def __call__(self, scope: Dict[str, Any], receive, send) -> None:
        if scope["type"] == "http":
            method = scope["method"]
            path = scope["path"]
            
            self.logger.info(f"Request: {method} {path}")
            
            async def send_wrapper(message: Dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    self.logger.info(f"Response: {method} {path} - {status_code}")
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
