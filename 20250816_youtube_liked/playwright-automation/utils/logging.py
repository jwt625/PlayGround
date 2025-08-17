"""
Logging utilities for YouTube video removal automation.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from config import LOG_LEVEL, LOG_FORMAT


def setup_logger(name: str = "youtube_remover") -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"youtube_removal_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_removal_progress(logger: logging.Logger, removed_count: int, total_count: int, title: str = None):
    """
    Log removal progress with consistent formatting.
    
    Args:
        logger: Logger instance
        removed_count: Number of videos removed so far
        total_count: Total number of videos to remove
        title: Title of the video being removed (optional)
    """
    progress_pct = (removed_count / total_count) * 100
    
    if title:
        logger.info(f"Removing video: {title}")
    
    logger.info(f"Progress: {removed_count}/{total_count} ({progress_pct:.1f}%)")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log error with additional context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about when/where the error occurred
    """
    error_msg = f"Error {context}: {str(error)}"
    logger.error(error_msg)
    logger.debug(f"Exception details: {type(error).__name__}: {error}", exc_info=True)
