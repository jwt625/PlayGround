#!/usr/bin/env python3
"""
Minimal inference server for Qwen-Image-Edit-2509 model.
Provides a simple API endpoint for multi-image editing.
"""

import os
import io
import base64
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from diffusers import QwenImageEditPlusPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global pipeline
    
    logger.info("Loading Qwen-Image-Edit-2509 model...")
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16
        )
        
        # Use both GPUs if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            pipeline.to('cuda:0')  # Primary GPU
        else:
            pipeline.to('cuda')
        
        pipeline.set_progress_bar_config(disable=True)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()


app = FastAPI(
    title="Qwen Image Edit API",
    description="Multi-image editing API using Qwen-Image-Edit-2509",
    version="1.0.0",
    lifespan=lifespan
)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": "Qwen-Image-Edit-2509",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
            })
    
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "gpus": gpu_info
    }


@app.post("/edit")
async def edit_images(
    images: List[UploadFile] = File(..., description="1-3 input images"),
    prompt: str = Form(..., description="Text prompt describing the desired edit"),
    negative_prompt: Optional[str] = Form(" ", description="Negative prompt"),
    num_inference_steps: int = Form(40, description="Number of inference steps"),
    guidance_scale: float = Form(1.0, description="Guidance scale"),
    true_cfg_scale: float = Form(4.0, description="True CFG scale"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
):
    """
    Edit images based on text prompt.
    
    Supports 1-3 input images for multi-image editing.
    Returns the edited image as base64-encoded PNG.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate number of images
    if not (1 <= len(images) <= 3):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 1-3 images, got {len(images)}"
        )
    
    try:
        # Load images
        logger.info(f"Processing {len(images)} image(s) with prompt: {prompt}")
        input_images = []
        for img_file in images:
            img_bytes = await img_file.read()
            img = load_image_from_bytes(img_bytes)
            input_images.append(img)
            logger.info(f"Loaded image: {img.size}")
        
        # Prepare inputs
        generator = torch.manual_seed(seed) if seed is not None else None
        
        inputs = {
            "image": input_images if len(input_images) > 1 else input_images[0],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
            "num_images_per_prompt": 1,
        }
        
        if generator is not None:
            inputs["generator"] = generator
        
        # Run inference
        logger.info("Running inference...")
        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
        
        logger.info(f"Generated image: {output_image.size}")
        
        # Convert to base64
        image_b64 = image_to_base64(output_image)
        
        return JSONResponse({
            "success": True,
            "image": image_b64,
            "size": output_image.size,
            "num_input_images": len(input_images)
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

