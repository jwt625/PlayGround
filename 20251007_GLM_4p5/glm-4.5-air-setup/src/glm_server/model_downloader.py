"""Model download utility for GLM-4.5-Air."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import time

from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
import torch


logger = logging.getLogger(__name__)


class ModelDownloadError(Exception):
    """Custom exception for model download errors."""
    pass


class ModelDownloader:
    """Type-safe model downloader for GLM-4.5-Air."""
    
    def __init__(
        self,
        model_id: str = "zai-org/GLM-4.5-Air",
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None
    ) -> None:
        """Initialize model downloader.
        
        Args:
            model_id: HuggingFace model identifier
            cache_dir: Local cache directory for models
            token: HuggingFace authentication token
        """
        self.model_id = model_id
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
        self.token = token
        self.api = HfApi(token=token)
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from HuggingFace Hub."""
        try:
            model_info = self.api.model_info(self.model_id)
            return {
                "model_id": self.model_id,
                "sha": model_info.sha,
                "last_modified": model_info.last_modified,
                "tags": model_info.tags,
                "pipeline_tag": model_info.pipeline_tag,
                "library_name": model_info.library_name,
                "downloads": getattr(model_info, 'downloads', 0),
                "likes": getattr(model_info, 'likes', 0),
            }
        except HfHubHTTPError as e:
            raise ModelDownloadError(f"Failed to get model info: {e}") from e
    
    def list_model_files(self) -> List[Dict[str, Any]]:
        """List all files in the model repository."""
        try:
            files = self.api.list_repo_files(self.model_id)
            file_info = []
            
            for file_path in files:
                try:
                    file_info_obj = self.api.get_paths_info(
                        self.model_id, 
                        paths=[file_path],
                        repo_type="model"
                    )[0]
                    file_info.append({
                        "path": file_path,
                        "size": getattr(file_info_obj, 'size', 0),
                        "blob_id": getattr(file_info_obj, 'blob_id', ''),
                        "lfs": getattr(file_info_obj, 'lfs', None) is not None,
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for file {file_path}: {e}")
                    file_info.append({
                        "path": file_path,
                        "size": 0,
                        "blob_id": "",
                        "lfs": False,
                    })
            
            return file_info
        except HfHubHTTPError as e:
            raise ModelDownloadError(f"Failed to list model files: {e}") from e
    
    def estimate_download_size(self) -> Dict[str, Any]:
        """Estimate total download size and provide breakdown."""
        files = self.list_model_files()
        
        total_size = 0
        large_files = []
        file_types: Dict[str, int] = {}
        
        for file_info in files:
            size = file_info["size"]
            total_size += size
            
            # Track large files (>100MB)
            if size > 100 * 1024 * 1024:
                large_files.append({
                    "path": file_info["path"],
                    "size_gb": round(size / (1024**3), 2),
                    "lfs": file_info["lfs"]
                })
            
            # Track file types
            ext = Path(file_info["path"]).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + size
        
        return {
            "total_size_gb": round(total_size / (1024**3), 2),
            "total_files": len(files),
            "large_files": large_files,
            "file_types": {
                ext: round(size / (1024**3), 2) 
                for ext, size in file_types.items()
            }
        }
    
    def check_disk_space(self, required_gb: float) -> bool:
        """Check if sufficient disk space is available."""
        try:
            free_space = shutil.disk_usage(self.cache_dir).free
            free_gb = free_space / (1024**3)
            
            logger.info(f"Available disk space: {free_gb:.2f} GB")
            logger.info(f"Required disk space: {required_gb:.2f} GB")
            
            # Add 20% buffer for safety
            required_with_buffer = required_gb * 1.2
            
            if free_gb < required_with_buffer:
                logger.error(
                    f"Insufficient disk space. Need {required_with_buffer:.2f} GB, "
                    f"have {free_gb:.2f} GB"
                )
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume sufficient space if check fails
    
    def download_model(
        self,
        local_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Path:
        """Download the complete model.
        
        Args:
            local_dir: Local directory to download to
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to downloaded model directory
        """
        # Get model info and size estimate
        model_info = self.get_model_info()
        size_info = self.estimate_download_size()
        
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Total size: {size_info['total_size_gb']:.2f} GB")
        logger.info(f"Total files: {size_info['total_files']}")
        
        # Check disk space
        if not self.check_disk_space(size_info['total_size_gb']):
            raise ModelDownloadError("Insufficient disk space for download")
        
        # Set up local directory
        if local_dir is None:
            local_dir = Path.cwd() / "models" / self.model_id.replace("/", "_")
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading to: {local_dir}")
        
        try:
            start_time = time.time()
            
            # Download using snapshot_download for complete model
            downloaded_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(local_dir),
                token=self.token,
                resume_download=True,
                local_files_only=False,
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            
            logger.info(f"Download completed in {download_time:.2f} seconds")
            logger.info(f"Model downloaded to: {downloaded_path}")
            
            return Path(downloaded_path)
            
        except Exception as e:
            raise ModelDownloadError(f"Failed to download model: {e}") from e
    
    def verify_model_files(self, model_path: Path) -> Dict[str, Any]:
        """Verify downloaded model files."""
        if not model_path.exists():
            raise ModelDownloadError(f"Model path does not exist: {model_path}")
        
        # Check for essential files
        essential_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        
        missing_files = []
        found_files = []
        
        for file_name in essential_files:
            file_path = model_path / file_name
            if file_path.exists():
                found_files.append(file_name)
            else:
                missing_files.append(file_name)
        
        # Check for model weight files
        weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        
        verification_result = {
            "model_path": str(model_path),
            "essential_files_found": found_files,
            "essential_files_missing": missing_files,
            "weight_files_count": len(weight_files),
            "weight_files": [f.name for f in weight_files],
            "total_size_gb": round(total_size / (1024**3), 2),
            "verification_passed": len(missing_files) == 0 and len(weight_files) > 0
        }
        
        if verification_result["verification_passed"]:
            logger.info("Model verification passed")
        else:
            logger.error(f"Model verification failed: {verification_result}")
        
        return verification_result


async def download_glm_model(
    model_id: str = "zai-org/GLM-4.5-Air",
    local_dir: Optional[Path] = None,
    token: Optional[str] = None
) -> Path:
    """Async wrapper for model download."""
    downloader = ModelDownloader(model_id=model_id, token=token)
    
    # Run download in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    model_path = await loop.run_in_executor(
        None, 
        downloader.download_model,
        local_dir
    )
    
    # Verify the download
    verification = downloader.verify_model_files(model_path)
    if not verification["verification_passed"]:
        raise ModelDownloadError("Model verification failed after download")
    
    return model_path
