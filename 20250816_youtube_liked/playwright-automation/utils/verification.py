"""
Verification utilities for backup integration.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any


def verify_recent_backup(max_age_hours: int = 24) -> bool:
    """
    Verify that a recent backup exists.

    Args:
        max_age_hours: Maximum age of backup in hours to consider "recent"

    Returns:
        True if recent backup exists, False otherwise
    """
    logger = logging.getLogger("youtube_remover")

    # Look for backup files in the parent directory
    backup_dir = Path("../backup")
    if not backup_dir.exists():
        logger.error("No backup directory found")
        return False

    # Find the most recent backup file
    backup_files = list(backup_dir.glob("youtube_liked_*.json"))
    if not backup_files:
        logger.error("No backup files found")
        return False
    
    # Get the most recent backup
    latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
    backup_age = datetime.now() - datetime.fromtimestamp(latest_backup.stat().st_mtime)
    
    if backup_age > timedelta(hours=max_age_hours):
        logger.error(f"Backup is {backup_age} old (exceeds {max_age_hours}h limit)")
        return False

    logger.info(f"Recent backup found: {latest_backup.name} (age: {backup_age})")
    return True


def get_backup_video_count() -> Optional[int]:
    """
    Get the number of videos in the most recent backup.
    
    Returns:
        Number of videos in backup, or None if backup not found/readable
    """
    logger = logging.getLogger("youtube_remover")
    
    backup_dir = Path("../backup")
    if not backup_dir.exists():
        return None
    
    backup_files = list(backup_dir.glob("youtube_liked_*.json"))
    if not backup_files:
        return None
    
    latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_backup, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
            
        if isinstance(backup_data, list):
            return len(backup_data)
        elif isinstance(backup_data, dict) and 'videos' in backup_data:
            return len(backup_data['videos'])
        else:
            logger.warning(f"Unexpected backup format in {latest_backup.name}")
            return None
            
    except Exception as e:
        logger.error(f"Error reading backup file {latest_backup.name}: {e}")
        return None


def log_backup_status(logger: logging.Logger) -> Dict[str, Any]:
    """
    Log comprehensive backup status information.
    
    Args:
        logger: Logger instance
        
    Returns:
        Dictionary with backup status information
    """
    status = {
        "has_recent_backup": False,
        "backup_count": None,
        "backup_age": None,
        "backup_file": None
    }
    
    backup_dir = Path("../backup")
    if not backup_dir.exists():
        logger.warning("No backup directory found")
        return status
    
    backup_files = list(backup_dir.glob("youtube_liked_*.json"))
    if not backup_files:
        logger.warning("No backup files found")
        return status
    
    latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
    backup_age = datetime.now() - datetime.fromtimestamp(latest_backup.stat().st_mtime)
    
    status["backup_file"] = latest_backup.name
    status["backup_age"] = str(backup_age)
    status["has_recent_backup"] = backup_age < timedelta(hours=24)
    status["backup_count"] = get_backup_video_count()
    
    logger.info(f"Backup status: {status}")
    return status
